
import os
import logging
from typing import Annotated
from typing_extensions import TypedDict

from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.messages import AIMessage, AIMessageChunk, HumanMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import InMemorySaver
from langchain_tavily import TavilySearch

from .crm_init import search_database, search_documents, update_database, db

logger = logging.getLogger(__name__)

# ─── 对话 LLM（deepseek-v4-pro）──────────────────────────
# 与 crm_init.llm（qwen-plus，用于 SQL 生成）是不同的实例
chat_llm = ChatOpenAI(
    model="deepseek-v4-pro",
    base_url="https://api.deepseek.com",
    api_key=os.environ.get("DEEPSEEK_API_KEY"),
    request_timeout=30,  # 防止 LLM 调用无限等待占满 ThreadPoolExecutor 线程
)

# ─── Tavily 网络搜索工具 ─────────────────────────────────
# TAVILY_API_KEY 由 crm_init 的 dotenv.load_dotenv() 加载到 os.environ
web_search = TavilySearch(max_results=1)


# ─── State 定义 ──────────────────────────────────────────
class State(TypedDict):
    messages: Annotated[list, add_messages]
    user_info: str   # 用户画像信息
    user_id: str     # 用户 ID


# ─── RAG 搜索工具 ────────────────────────────────────────
@tool
def search(query: str, k: int) -> str:
    """
    在旅游攻略向量数据库中进行语义搜索，获取最相关的攻略内容片段。
    当前数据库仅包含中国部分城市（丽江、成都、西安、厦门）的旅游攻略，涵盖景点、美食、行程规划等。

    Args:
      query（str）: 对旅游信息的自然语言描述或关键词，必须使用中文（英文查询可能返回空结果）。
                     建议包含具体城市和需求，例如："丽江古城必去景点" 或 "成都周边美食推荐"。
      k: 返回结果数量，取值1-10，一般建议取值为4
    Returns:
      str: 检索到的与查询最相关的几个旅游攻略文本片段
    """
    docs = search_documents(1, query, k)
    formatted = []
    for doc in docs:
        page = doc.metadata.get("page_label", "未知页码")
        formatted.append(f"[第{page}页] {doc.page_content}")
    return "\n\n".join(formatted)


tools = [search, web_search]
chat_llm_with_tools = chat_llm.bind_tools(tools)


# ─── 节点函数 ─────────────────────────────────────────────
def verify_identity(state: State):
    """前置节点：提取名字 -> 查画像 -> 注入上下文。"""
    last_msg = state["messages"][-1].content
    info = search_database(last_msg)
    if not info or info == "无用户信息":
        return {
            "messages": [AIMessage(content="未识别到有效身份。请以「我是XX」的格式重新输入。")],
            "user_info": "",
            "user_id": ""
        }
    # search_database 返回的字符串形如 [('user_001', '小明', ...)]，用 eval 解析后取第一行第一列
    user_id = ""
    try:
        parsed = eval(info)
        if isinstance(parsed, list) and len(parsed) > 0:
            row = parsed[0]
            if isinstance(row, (tuple, list)) and len(row) > 0:
                user_id = str(row[0])
    except Exception:
        pass
    return {
        "user_info": str(info),
        "user_id": user_id,
        "messages": [AIMessage(content=f"已识别您的画像：{info}\n请问您想了解什么旅游信息？")]
    }


def chatbot(state: State):
    """带用户画像上下文的对话节点。"""
    system = state.get("user_info", "")
    messages = state["messages"]
    if system:
        messages = [("system", f"你是一位旅游助理。请基于以下用户画像提供个性化推荐：\n{system}")] + messages
    return {"messages": [chat_llm_with_tools.invoke(messages)]}


def update_profile(state: State):
    """根据本轮对话，用 LLM 提取新偏好并生成 UPDATE SQL 写入数据库。"""
    user_id = state.get("user_id", "")
    if not user_id:
        return {}  # 没有 user_id 就跳过
    current_profile = state.get("user_info", "")
    msgs = state["messages"]
    last_user = ""
    last_ai = ""
    for m in reversed(msgs):
        if isinstance(m, HumanMessage) and not last_user:
            last_user = m.content
        if isinstance(m, AIMessage) and m.content and not m.tool_calls and not last_ai:
            last_ai = m.content
        if last_user and last_ai:
            break
    conversation_text = f"[user]: {last_user}\n[assistant]: {last_ai}"
    prompt = f"""你是 CRM 画像更新专家。请根据用户本轮对话，判断用户是否表达了新的旅游偏好/信息。
用户ID: {user_id}
当前画像: {current_profile}
本轮对话:
{conversation_text}
可用表结构（实际）：
{db.get_table_info()}
规则：
1. 无新偏好 -> 仅返回 NO_UPDATE
2. 有新偏好 -> 返回一条或多条 SQL（每条一行以分号结尾），例如：
   UPDATE users SET food_pref='清淡,少辣,偏爱菌菇' WHERE user_id='{user_id}';
   INSERT INTO travel_preferences(user_id,category,tag,weight) VALUES('{user_id}','scenery','雪山',2.5);
3. SQL 必须用正确的 user_id，字符串值用单引号
4. 只输出 SQL，不要解释
5. SQL 值里不要包含分号，用中文逗号代替"""
    response = chat_llm.invoke(prompt).content.strip().removeprefix("```sql").removeprefix("```").removesuffix("```").strip()
    if response == "NO_UPDATE" or not response:
        logger.info("[update_profile] 无需更新画像")
        return {}
    for sql in response.split(";"):
        sql = sql.strip()
        if sql and sql.upper().startswith(("UPDATE", "INSERT", "DELETE")):
            try:
                update_database(sql)
            except Exception as e:
                logger.error(f"[画像更新] SQL 执行失败: {e}")
    logger.info("[update_profile] 画像更新完毕")
    return {}


def route_from_start(state: State):
    """START 条件分流：有画像 -> 直接进 chatbot；无画像 -> 先验证身份。"""
    if state.get("user_info"):
        return "chatbot"
    return "verify_identity"


def route_tools(state: State):
    """chatbot 后的路由：有 tool_calls -> tools 节点；否则 -> update_profile。"""
    if isinstance(state, list):
        ai_message = state[-1]
    elif messages := state.get("messages", []):
        ai_message = messages[-1]
    else:
        raise ValueError(f"No messages found in input state to tool_edge: {state}")
    if hasattr(ai_message, "tool_calls") and len(ai_message.tool_calls) > 0:
        return "tools"
    return "update_profile"


# ─── 构建图 ───────────────────────────────────────────────
graph_builder = StateGraph(State)
graph_builder.add_node("verify_identity", verify_identity)
graph_builder.add_node("chatbot", chatbot)
graph_builder.add_node("update_profile", update_profile)
graph_builder.add_conditional_edges(
    START,
    route_from_start,
    {"verify_identity": "verify_identity", "chatbot": "chatbot"},
)
graph_builder.add_edge("verify_identity", END)
tool_node = ToolNode(tools)
graph_builder.add_node("tools", tool_node)
graph_builder.add_conditional_edges(
    "chatbot",
    route_tools,
    {"tools": "tools", "update_profile": "update_profile"},
)
graph_builder.add_edge("tools", "chatbot")
graph_builder.add_edge("update_profile", END)

checkpointer = InMemorySaver()
graph = graph_builder.compile(checkpointer=checkpointer)


# ─── 核心接口：run_agent_stream ───────────────────────────
def run_agent_stream(session_id: str, user_input: str, user_info: str = ""):
    """同步 generator，可在 ThreadPoolExecutor 线程里直接调用。

    使用 stream_mode=["messages", "updates"] 双模式：
    - "messages" 模式拦截节点内的 LLM 调用，yield token 级 AIMessageChunk
    - "updates" 模式在每个节点完成后 yield 完整状态更新

    过滤策略：
    - "messages" 只取 langgraph_node=="chatbot" 的 chunk（过滤 verify_identity/update_profile 内部的 SQL LLM 调用）
    - "updates" 从 verify_identity 取身份验证回复消息 + user_info/user_id
    - "updates" 从 tools 取工具执行结果（截断显示）

    Args:
        session_id: 会话 ID，用作 LangGraph thread_id（跨轮对话记忆）
        user_input: 用户本轮输入文本
        user_info: 用户画像信息（外部传入，非空时跳过 verify_identity 直接进 chatbot）

    Yields:
        dict 事件，type 取值：
        - "text": {"type": "text", "content": str}          chatbot 文本 token（逐字/逐句）
        - "tool_call": {"type": "tool_call", "name": str}   LLM 决定调用工具
        - "tool_result": {"type": "tool_result", "content": str}  工具执行结果（截断 200 字）
        - "identity": {"type": "identity", "user_info": str, "user_id": str}  身份验证结果
        - "error": {"type": "error", "content": str}        执行异常
        - "done": {"type": "done"}                          图执行完毕
    """
    config = {"configurable": {"thread_id": session_id}}
    initial_state = {"messages": [("user", user_input)]}
    if user_info:
        initial_state["user_info"] = user_info

    try:
        for mode, data in graph.stream(
            initial_state,
            config=config,
            stream_mode=["messages", "updates"],
        ):
            if mode == "messages":
                chunk, metadata = data
                node = metadata.get("langgraph_node", "")
                # 只从 chatbot 节点取文本 token
                # 过滤 verify_identity/update_profile 内部的 SQL 生成 LLM 调用
                if node == "chatbot" and isinstance(chunk, AIMessageChunk):
                    if chunk.content:
                        yield {"type": "text", "content": chunk.content}
                    if chunk.tool_call_chunks:
                        for tc in chunk.tool_call_chunks:
                            if tc.get("name"):
                                yield {"type": "tool_call", "name": tc["name"]}

            elif mode == "updates":
                for node_name, state_update in data.items():
                    # update_profile 返回 {} 时，LangGraph updates 模式可能 yield None
                    if state_update is None:
                        continue
                    msgs = state_update.get("messages", [])

                    if node_name == "verify_identity":
                        # verify_identity 的回复消息是手动构造的 AIMessage，
                        # messages 模式抓不到，只能从 updates 取
                        for m in msgs:
                            if hasattr(m, "content") and m.content:
                                yield {"type": "text", "content": m.content}
                        if state_update.get("user_info"):
                            yield {
                                "type": "identity",
                                "user_info": state_update["user_info"],
                                "user_id": state_update.get("user_id", ""),
                            }

                    elif node_name == "tools":
                        # 工具执行结果（ToolMessage），截断后通知前端
                        for m in msgs:
                            if hasattr(m, "content") and m.content:
                                yield {
                                    "type": "tool_result",
                                    "content": m.content[:200],
                                }

                    # chatbot 的 updates 不需要处理——messages 模式已覆盖文本 token

    except Exception as e:
        logger.error(f"[Agent] 执行异常: {e}", exc_info=True)
        yield {"type": "error", "content": str(e)}

    yield {"type": "done"}
