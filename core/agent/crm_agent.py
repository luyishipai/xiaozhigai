import os
import ast
import logging
import threading
from typing import Annotated
from typing_extensions import TypedDict

from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.messages import AIMessage, AIMessageChunk, HumanMessage, trim_messages
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import InMemorySaver
from langchain_tavily import TavilySearch

from .crm_init import search_database, search_documents, update_database, db

logger = logging.getLogger(__name__)

# ─── 对话 LLM（deepseek-v4-pro）──────────────────────────
chat_llm = ChatOpenAI(
    model="deepseek-v4-pro",
    base_url="https://api.deepseek.com",
    api_key=os.environ.get("DEEPSEEK_API_KEY"),
    request_timeout=30,
)

web_search = TavilySearch(max_results=1)

# ─── 线程局部上下文（2.5: search 工具用真实 user_id）─────────
_ctx = threading.local()


def _set_ctx(user_id="", user_info=""):
    """在 run_agent_stream 入口设置，供 @tool 函数读取。"""
    _ctx.user_id = str(user_id) if user_id else "1"
    _ctx.user_info = user_info or ""


# ─── State 定义 ──────────────────────────────────────────
class State(TypedDict):
    messages: Annotated[list, add_messages]
    user_info: str
    user_id: str


# ─── RAG 搜索工具（2.5: 用 _ctx.user_id 替代硬编码 1）────────
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
    user_id = getattr(_ctx, "user_id", "1")
    docs = search_documents(user_id, query, k)
    formatted = []
    for doc in docs:
        page = doc.metadata.get("page_label", "未知页码")
        formatted.append(f"[第{page}页] {doc.page_content}")
    return "\n\n".join(formatted)


tools = [search, web_search]
chat_llm_with_tools = chat_llm.bind_tools(tools)


# ─── 节点函数 ─────────────────────────────────────────────
def verify_identity(state: State):
    """前置节点：提取名字 -> 查画像 -> 注入上下文。

    2.6: 用 ast.literal_eval 替代 eval()
    2.10: 格式化为人类可读文本，不直接输出原始 DB 字符串
    """
    last_msg = state["messages"][-1].content
    info = search_database(last_msg)
    if not info or info == "无用户信息":
        return {
            "messages": [AIMessage(content="未识别到有效身份。请以「我是XX」的格式重新输入。")],
            "user_info": "",
            "user_id": ""
        }

    # 2.6: 安全解析 DB 返回字符串
    user_id = ""
    user_display = str(info)
    try:
        parsed = ast.literal_eval(info)
        if isinstance(parsed, list) and len(parsed) > 0:
            row = parsed[0]
            if isinstance(row, (tuple, list)) and len(row) > 0:
                user_id = str(row[0])
                # 2.10: 提取可读字段
                name = str(row[1]) if len(row) > 1 else user_id
                # 收集非空字段
                fields = []
                for i, val in enumerate(row):
                    if val is not None and str(val).strip():
                        fields.append(str(val))
                user_display = f"用户{name}（ID: {user_id}）" if name != user_id else f"用户ID: {user_id}"
                # 更新线程上下文
                _set_ctx(user_id=user_id, user_info=str(info))
    except (ValueError, SyntaxError):
        pass

    return {
        "user_info": str(info),
        "user_id": user_id,
        "messages": [AIMessage(
            content=f"已识别您的身份：{user_display}\n请问您想了解什么旅游信息？"
        )]
    }


def _estimate_token_count(messages):
    """粗略估算消息列表的 token 数（中英文混合，约 3 字符/token）。

    替代 chat_llm 作为 token_counter，避免 deepseek 等模型不支持
    get_num_tokens_from_messages() 的问题。
    """
    total = 0
    for msg in messages:
        content = msg.content if hasattr(msg, "content") else str(msg)
        if isinstance(content, list):
            content = " ".join(str(c) for c in content)
        total += len(str(content))
    return total // 3


def chatbot(state: State):
    """带用户画像上下文的对话节点。

    3.4: 约束 LLM 输出为自然口语化文本，禁止 Markdown 格式
    4.3: 使用 trim_messages 限制历史消息窗口（防止 token 爆炸）
    """
    system = state.get("user_info", "")
    messages = state["messages"]

    # 4.3: 修剪历史消息到约 4000 token
    messages = trim_messages(
        messages,
        max_tokens=4000,
        strategy="last",
        token_counter=_estimate_token_count,
        start_on="human",
        allow_partial=False,
    )

    if system:
        messages = [("system",
            f"你是一位旅游助理。请基于以下用户画像提供个性化推荐：\n{system}\n\n"
            "重要：请用自然口语化的方式回复，就像和朋友聊天一样。"
            "不要使用Markdown格式（如表格、代码块、标题#等），直接用纯文本对话方式回答。"
        )] + messages
    else:
        messages = [("system",
            "你是一位旅游助理。请用自然口语化的方式回复，"
            "不要使用Markdown格式，直接用纯文本对话方式回答。"
        )] + messages

    return {"messages": [chat_llm_with_tools.invoke(messages)]}


def update_profile(state: State):
    """3.1: 画像更新移至后台线程，不阻塞 done 事件。

    节点本身立即返回空 dict，实际 LLM 调用在后台线程执行（fire-and-forget）。
    """
    user_id = state.get("user_id", "")
    if not user_id:
        return {}

    # 捕获快照，避免在后台线程中访问可变 state
    current_profile = state.get("user_info", "")
    msgs = list(state["messages"])

    def _bg_update():
        try:
            _do_update_profile(user_id, current_profile, msgs)
        except Exception as e:
            logger.error(f"[update_profile] 后台更新失败: {e}")

    threading.Thread(target=_bg_update, daemon=True).start()
    return {}


def _do_update_profile(user_id, current_profile, msgs):
    """实际的画像更新逻辑，在后台线程中执行。"""
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
        return
    for sql in response.split(";"):
        sql = sql.strip()
        if sql and sql.upper().startswith(("UPDATE", "INSERT", "DELETE")):
            try:
                update_database(sql)
            except Exception as e:
                logger.error(f"[画像更新] SQL 执行失败: {e}")
    logger.info("[update_profile] 画像更新完毕")


# 4.11: 简化 route_tools（移除永不可达的 isinstance(state, list) 分支）
def route_tools(state: State):
    """chatbot 后的路由：有 tool_calls -> tools；否则 -> update_profile。"""
    ai_message = state["messages"][-1]
    if hasattr(ai_message, "tool_calls") and len(ai_message.tool_calls) > 0:
        return "tools"
    return "update_profile"


def route_from_start(state: State):
    """START 条件分流：有画像 -> 直接进 chatbot；无画像 -> 先验证身份。"""
    if state.get("user_info"):
        return "chatbot"
    return "verify_identity"


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


# 3.2/4.9: 清理指定会话的 checkpoint 数据
def cleanup_session_checkpoint(session_id: str):
    """关闭会话时清理 LangGraph InMemorySaver 中对应的 checkpoint。"""
    try:
        if hasattr(checkpointer, "storage"):
            checkpointer.storage.pop(session_id, None)
    except Exception as e:
        logger.debug(f"清理 checkpoint 失败: {e}")


# ─── 核心接口：run_agent_stream ───────────────────────────
def run_agent_stream(session_id: str, user_input: str, user_info: str = "",
                     user_id: str = "", logger=None, tag=None):
    """同步 generator，可在 ThreadPoolExecutor 线程里直接调用。

    2.4: 接受 logger 参数，使 Agent 日志带 session_id
    2.5: 通过 user_id 参数设置线程上下文
    3.1: update_profile 不阻塞 done 事件（后台线程执行）

    Yields:
        dict 事件，type 取值：
        - "text": {"type": "text", "content": str}
        - "tool_call": {"type": "tool_call", "name": str}
        - "tool_result": {"type": "tool_result", "content": str}
        - "identity": {"type": "identity", "user_info": str, "user_id": str}
        - "error": {"type": "error", "content": str}
        - "done": {"type": "done"}
    """
    _log = logger or logging.getLogger(__name__)
    _set_ctx(user_id=user_id, user_info=user_info)

    config = {"configurable": {"thread_id": session_id}}
    initial_state = {"messages": [("user", user_input)]}
    if user_info:
        initial_state["user_info"] = user_info
    if user_id:
        initial_state["user_id"] = user_id

    try:
        for mode, data in graph.stream(
            initial_state,
            config=config,
            stream_mode=["messages", "updates"],
        ):
            if mode == "messages":
                chunk, metadata = data
                node = metadata.get("langgraph_node", "")
                if node == "chatbot" and isinstance(chunk, AIMessageChunk):
                    if chunk.content:
                        yield {"type": "text", "content": chunk.content}
                    if chunk.tool_call_chunks:
                        for tc in chunk.tool_call_chunks:
                            if tc.get("name"):
                                yield {"type": "tool_call", "name": tc["name"]}

            elif mode == "updates":
                for node_name, state_update in data.items():
                    if state_update is None:
                        continue
                    msgs = state_update.get("messages", [])

                    if node_name == "verify_identity":
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
                        for m in msgs:
                            if hasattr(m, "content") and m.content:
                                yield {
                                    "type": "tool_result",
                                    "content": m.content[:200],
                                }

    except Exception as e:
        _log.error(f"[Agent] 执行异常: {e}", exc_info=True)
        # 2.9: 错误事件内容用友好描述，不暴露原始英文异常
        yield {"type": "error", "content": "服务处理异常"}

    yield {"type": "done"}
