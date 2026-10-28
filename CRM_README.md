# CRM 智能旅游助理 — 系统说明文档

> 基于小智服务端（xiaozhi-server）改造的 AI CRM 问答服务，以 LangGraph Agent 为核心，
> 结合 RAG 检索、联网搜索、用户画像自动维护，提供个性化旅游推荐对话。

---

## 一、系统定位

本项目将原小智 ESP32 语音助手的问答模块解耦为独立的 HTTP 服务，核心改造为：

1. **去 WebSocket 化** — 删除 12 个 WS provider、12 个文本消息处理器、VAD/AEC/MQTT/设备绑定等模块，仅保留 ASR + TTS 两个 provider
2. **LangGraph Agent 接管对话** — 废弃小智原有的 `func_handler` / `intent` / `DIRECT_ANSWER_TOOL` / `tool_call_timeout`，由 Agent 全权负责工具调用与对话流转
3. **FakeWebSocket + SSE 流式返回** — 通过适配器模式零改动复用 handle 模块，`/api/chat` 以 Server-Sent Events 实时推送文字与音频
4. **用户画像自动维护** — 对话结束后后台线程自动提取新偏好，生成 SQL 更新数据库

---

## 二、系统架构

```
                         ┌──────────────────────────────────────────────────┐
                         │              前端 (Gradio / HTTP Client)            │
                         │  jiemian.py — SSE 流式接收，文字逐句显示              │
                         └───────────┬───────────────┬──────────────┬─────────┘
                                     │               │              │
                              POST /api/asr   POST /api/tts   POST /api/chat
                                     │               │              │
                         ┌───────────▼───────────────▼──────────────▼─────────┐
                         │              FastAPI (http_server_fastapi.py)        │
                         │  端口 8003 · lifespan 初始化 ASR + SessionManager    │
                         └───────────┬───────────────────────┬────────────────┘
                                     │                       │
                    ┌────────────────▼───┐          ┌────────▼────────┐
                    │  SessionManager     │          │  ConnectionHandler │
                    │  (session.py)       │          │  (connection.py)    │
                    │  - 按 session_id    │◀────────▶│  - 会话生命周期     │
                    │    管理 conn 实例    │  注入     │  - TTS/ASR 组件初始化│
                    │  - 并发锁 (Lock)    │  FakeWS  │  - chat() 委托 Agent │
                    │  - 空闲会话清理     │          │  - close() 清理     │
                    └────────────────────┘          └────────┬──────────┘
                                                             │ chat()
                    ┌────────────────────────────────────────┐│
                    │              AgentBridge                 ││
                    │  (agent_bridge.py)                       ││◀─
                    │  - run_agent_stream() 事件流消费          │
                    │  - text 事件 → TTS 文本队列              │
                    │  - tool_call/result → WebSocket JSON     │
                    │  - identity 事件 → 回写 conn.user_info   │
                    │  - 首 token 超时 → "正在思考中..."       │
                    └────────────────────┬───────────────────┘
                                         │
                    ┌────────────────────▼───────────────────┐
                    │         LangGraph CRM Agent             │
                    │         (crm_agent.py)                   │
                    │                                          │
                    │   ┌─START──route_from_start             │
                    │   │                                      │
                    │   ├─► verify_identity (无画像时)          │
                    │   │      └─► END                         │
                    │   │                                      │
                    │   └─► chatbot (有画像时)                 │
                    │          │                              │
                    │          ├─► route_tools                │
                    │          │   ├─► tools (ToolNode)       │
                    │          │   │     └─► chatbot (循环)    │
                    │          │   └─► update_profile         │
                    │          │         └─► END              │
                    └──────────────────────────────────────────┘
                                         │
                    ┌────────────────────▼───────────────────┐
                    │         CRM 初始化层                    │
                    │         (crm_init.py)                    │
                    │  - SQLDatabase (travel.db) 延迟初始化   │
                    │  - search_database() 身份查询            │
                    │  - update_database() 画像写入            │
                    │  - DashScope 向量库 (Chroma)            │
                    │  - search_documents() RAG 检索           │
                    │  - TavilySearch 联网搜索                  │
                    └─────────────────────────────────────────┘
```

### 四套并行执行体

| 执行体 | 所在层 | 职责 |
|--------|--------|------|
| `conn.loop` | FastAPI 主事件循环 | 所有 async 代码的宿主，SSE 流式消费在此 |
| `conn.executor` | ThreadPoolExecutor(max_workers=5) | 跑同步 `conn.chat()` → AgentBridge → `graph.stream()` |
| `tts_text_priority_thread` | TTS daemon 线程 | 消费 `tts_text_queue`，调 `_get_segment_text()` 分段后送入 `tts_audio_queue` |
| `_audio_play_priority_thread` | TTS daemon 线程 | 消费 `tts_audio_queue`，调 `sendAudioMessage()` 通过 FakeWebSocket 发送 |

> **跨线程桥梁**：`asyncio.run_coroutine_threadsafe(coro, self.conn.loop)` — 在非主线程中安全调度 async 任务到主循环。

---

## 三、核心模块职责

| 文件 | 行数 | 职责 |
|------|------|------|
| `core/http_server_fastapi.py` | ~509 | FastAPI 入口：6 个端点 + lifespan 初始化 + SSE 流式返回 + opus→WAV 转换 |
| `core/session.py` | ~138 | SessionManager：会话创建/复用/关闭，FakeWebSocket 适配器，空闲清理 |
| `core/connection.py` | ~227 | ConnectionHandler：会话生命周期，TTS/ASR 组件初始化，`chat()` 委托 Agent |
| `core/agent/agent_bridge.py` | ~184 | AgentBridge：Agent 事件流 → TTS 队列 / WebSocket JSON 的桥接层 |
| `core/agent/crm_agent.py` | ~362 | LangGraph 图定义 + 节点函数 + `run_agent_stream()` 同步生成器 |
| `core/agent/crm_init.py` | ~201 | 数据访问层：SQLDatabase 延迟初始化 + 向量库 + 环境变量加载 |
| `core/handle/receiveAudioHandle.py` | ~40 | `startToChat()`：解析输入 → `send_stt_message` → 提交 `conn.chat` |
| `core/handle/sendAudioHandle.py` | ~250 | `sendAudioMessage()`：发送 TTS 音频 + stt 消息到 FakeWebSocket |
| `core/providers/tts/base.py` | ~640 | TTS 基类：文本分段（防 TextTooLong）+ 两个 daemon 线程 |
| `core/providers/asr/base.py` | ~330 | ASR 基类：`receive_audio` / `handle_voice_stop` / `batch_recognize` |

---

## 四、Agent 工作流

### 4.1 LangGraph 图结构

```
              ┌─────────┐
              │  START   │
              └────┬─────┘
                   │ route_from_start()
          ┌────────┴────────┐
          │                 │
    user_info 为空      user_info 非空
          │                 │
          ▼                 ▼
  ┌───────────────┐  ┌───────────┐
  │verify_identity│  │  chatbot  │◀────────────┐
  └───────┬───────┘  └─────┬─────┘             │
          │                 │ route_tools()      │
          │           ┌─────┴─────┐              │
          │           │           │              │
          │      有tool_calls  无tool_calls       │
          │           │           │              │
          │           ▼           ▼              │
          │     ┌─────────┐ ┌──────────────┐     │
          │     │  tools  │ │update_profile│     │
          │     └────┬────┘ └──────┬───────┘     │
          │          │            │              │
          │          └────────────┘ END         │
          │                        │            │
          │                        │            │
          ▼                        │            │
         END                                    │
                                               │
                            tools → chatbot 循环 ┘
```

### 4.2 节点说明

| 节点 | 入口条件 | 职责 | 出口 |
|------|----------|------|------|
| `verify_identity` | `user_info` 为空 | 提取用户名字 → `search_database()` 查画像 → `ast.literal_eval` 安全解析 → 注入 `user_info` + `user_id` | END |
| `chatbot` | `user_info` 非空或从 tools 返回 | `trim_messages` 修剪历史 → 注入 system prompt（含画像 + 口语化约束）→ `chat_llm_with_tools.invoke()` | route_tools |
| `tools` | chatbot 返回有 `tool_calls` | ToolNode 执行 `search` 或 `web_search` | chatbot（循环） |
| `update_profile` | chatbot 返回无 `tool_calls` | 后台线程 fire-and-forget：提取本轮对话 → LLM 生成 SQL → 执行写入 | END |

### 4.3 路由函数

- **`route_from_start(state)`**：`state.user_info` 非空 → `chatbot`；为空 → `verify_identity`
- **`route_tools(state)`**：`messages[-1].tool_calls` 非空 → `tools`；为空 → `update_profile`

### 4.4 工具定义

| 工具名 | 来源 | 功能 |
|--------|------|------|
| `search` | `crm_init.search_documents` | 在用户专属向量库（Chroma）中语义检索旅游攻略片段，返回带页码标注的文本 |
| `web_search` | `TavilySearch(max_results=1)` | 联网搜索最新信息（天气、时事、实时价格等） |

### 4.5 Agent 事件流

`run_agent_stream()` 使用 `graph.stream(stream_mode=["messages", "updates"])` 双模式：

| 流模式 | 拦截内容 | yield 事件 |
|--------|----------|-----------|
| `messages` | `chatbot` 节点的 `AIMessageChunk` | `text`（内容片段）/ `tool_call`（工具名） |
| `updates` | 节点完成后的 state 差量 | `verify_identity` → `text` + `identity`；`tools` → `tool_result` |

事件类型汇总：

```
{"type": "text", "content": "..."}              — LLM 流式文字片段
{"type": "tool_call", "name": "search"}         — Agent 发起工具调用
{"type": "tool_result", "content": "..."}       — 工具执行结果（截断 200 字符）
{"type": "identity", "user_info": "...", "user_id": "..."}  — 身份验证完成
{"type": "error", "content": "服务处理异常"}     — 异常（友好中文，不暴露英文）
{"type": "done"}                                 — 全部结束
```

### 4.6 对话记忆

- **InMemorySaver checkpointer**：`thread_id = session_id`
- 同一 `session_id` 的多轮对话共享 LangGraph state（messages 列表持续追加）
- 会话关闭时 `cleanup_session_checkpoint(session_id)` 清理对应 checkpoint

### 4.7 消息窗口控制

`chatbot` 节点使用 `trim_messages` 防止 token 爆炸：

```python
messages = trim_messages(
    messages,
    max_tokens=4000,
    strategy="last",          # 保留最近的消息
    token_counter=_estimate_token_count,  # 粗略估算（字符数 ÷ 3）
    start_on="human",         # 确保从 human 消息开始
    allow_partial=False,
)
```

> **注**：`token_counter` 不能直接传 `ChatOpenAI` 实例 — deepseek 等模型不支持 `get_num_tokens_from_messages()`，会抛出运行时异常。改用自定义 `_estimate_token_count` 函数按字符数粗略估算。

---

## 五、数据模型

### 5.1 数据库

- **类型**：SQLite，文件路径 `data/travel.db`
- **访问层**：LangChain `SQLDatabase`，延迟初始化 + 自动重连（`crm_init.get_db()` / `db_run_safe()`）

### 5.2 表结构

#### users — 用户画像主表

| 列名 | 类型 | 说明 |
|------|------|------|
| user_id | TEXT PK | 用户唯一标识（如 `user_001`） |
| name | TEXT | 用户名（如 `小明`） |
| budget_range | TEXT | 预算范围（如 `1000-3000`） |
| travel_style | TEXT | 旅行风格（如 `自然风光,人文历史`） |
| pace | TEXT | 节奏偏好（如 `慢节奏`） |
| transport_pref | TEXT | 交通偏好（如 `高铁`） |
| food_pref | TEXT | 饮食偏好（如 `清淡`） |
| accommodation_pref | TEXT | 住宿偏好（如 `民宿,经济型`） |
| avoid_crowds | INTEGER | 避拥挤人群（1=是, 0=否） |

#### visited_places — 历史访问记录

| 列名 | 类型 | 说明 |
|------|------|------|
| id | INTEGER PK | 自增 |
| user_id | TEXT FK | 关联 users.user_id |
| place_name | TEXT | 地点名（如 `大理`） |
| province | TEXT | 省份（如 `云南`） |
| visit_date | TEXT | 访问日期 |
| trip_id | TEXT FK | 关联 trips.trip_id |
| rating | INTEGER | 评分（1-5） |
| notes | TEXT | 备注 |

#### travel_preferences — 偏好标签表

| 列名 | 类型 | 说明 |
|------|------|------|
| id | INTEGER PK | 自增 |
| user_id | TEXT FK | 关联 users.user_id |
| category | TEXT | 分类（如 `scenery`） |
| tag | TEXT | 标签（如 `自然风光`） |
| weight | REAL | 权重（如 2.5） |

#### trips — 旅行计划

| 列名 | 类型 | 说明 |
|------|------|------|
| trip_id | TEXT PK | 旅行计划 ID |
| user_id | TEXT FK | 关联 users.user_id |
| destination | TEXT | 目的地 |
| start_date | TEXT | 开始日期 |
| end_date | TEXT | 结束日期 |
| status | TEXT | 状态（`planned` / `completed`） |
| budget | REAL | 预算 |
| actual_spending | REAL | 实际花费 |

#### conversations — 对话记录

| 列名 | 类型 | 说明 |
|------|------|------|
| id | INTEGER PK | 自增 |
| user_id | TEXT | 用户 ID |
| timestamp | TEXT | 时间戳 |
| user_input | TEXT | 用户输入 |
| intent | TEXT | 意图 |
| response | TEXT | 回复 |

#### guides — 旅游攻略

| 列名 | 类型 | 说明 |
|------|------|------|
| id | INTEGER PK | 自增 |
| user_id | TEXT | 关联用户 |
| title | TEXT | 标题 |
| place_name | TEXT | 地点 |
| content_path | TEXT | 内容文件路径 |
| tags | TEXT | 标签 |

### 5.3 向量库

- **存储**：`data/vectorstore/{user_id}/` 目录，Chroma 持久化
- **Embedding**：`DashScopeEmbeddings(model="text-embedding-v2")`
- **内容来源**：丽江、成都、西安、厦门等城市的旅游攻略 PDF/DOCX/TXT
- **文档分割**：`RecursiveCharacterTextSplitter(chunk_size=250, chunk_overlap=25)`
- **检索接口**：`search_documents(user_id, query, k)` — 语义搜索返回最相关片段

---

## 六、API 接口

### 6.1 端点一览

| 方法 | 路径 | 功能 | 说明 |
|------|------|------|------|
| POST | `/api/asr` | 语音识别 | 上传 WAV → 返回文本 |
| POST | `/api/tts` | 文字转语音 | 传入文本 → 返回 opus 音频 |
| POST | `/api/chat` | 完整问答 | SSE 流式返回，支持文字/音频输入 |
| GET | `/api/session/{id}` | 查询会话 | 返回会话状态 |
| DELETE | `/api/session/{id}` | 删除会话 | 清理资源 |
| GET | `/health` | 健康检查 | 返回 `{"status": "ok"}` |
| GET | `/api/welcome` | 欢迎消息 | 返回 `"请先输入您的身份（如：我是小明）"` |

### 6.2 /api/chat — SSE 事件格式

**请求**：

```
POST /api/chat
Content-Type: multipart/form-data

file: (可选) WAV 音频文件
text: (可选) 文字输入
session_id: (可选) 多轮对话传同一个
output_audio: (可选) 是否返回 TTS 音频，默认 true
```

**响应**：`text/event-stream`，每行 `data: {JSON}\n\n`

```
data: {"type":"stt","text":"去厦门旅游怎么样"}                    — ASR 识别结果

data: {"type":"tts","state":"start"}                              — TTS 开始

data: {"type":"agent_tool","event":"tool_call","name":"search"}   — Agent 调用工具

data: {"type":"agent_tool","event":"tool_result","content":"..."}  — 工具结果

data: {"type":"tts","state":"sentence_start","text":"厦门是个"}    — 逐句回复文字

data: {"type":"tts","state":"sentence_start","text":"很好的选择"}  — 逐句回复文字

data: {"type":"done","session_id":"xxx","asr_text":"...","reply_text":"...","audio_base64":"...","audio_type":"wav"}  — 全部完成
```

### 6.3 错误处理

| 场景 | HTTP 状态 | 响应 |
|------|-----------|------|
| 无输入（text 和 file 都为空） | 400 | `{"detail": "请提供 text 或 file 之一"}` |
| TTS 初始化超时 | 503 | `{"detail": "TTS 初始化超时，请重试"}` |
| 同一会话并发请求 | 429 | `{"detail": "该会话正在处理另一请求，请稍后重试"}` |
| SSE 等待超时(60s) | 200 | `data: {"type":"timeout","message":"TTS处理超时"}` |
| 内部异常 | 500 | `{"detail": "..."}` |

---

## 七、配置说明

### 7.1 服务配置 `data/.config.yaml`

```yaml
selected_module:
  LLM: ChatGLMLLM        # 声明值（Agent 实际使用 deepseek-v4-pro，在 crm_agent.py 硬编码）
  ASR: FunASR             # 声明值（实际配置为 BaiduASR）
  TTS: TencentTTS         # 声明值（实际配置为 TencentTTS）

ASR:
  BaiduASR:               # 百度语音识别（HTTP 模式）
    app_id: ...
    api_key: ...
    secret_key: ...

TTS:
  TencentTTS:             # 腾讯云语音合成
    app_id: "101001"
    region: "ap-guangzhou"
    token: ...
    ...
    delete_audio: true    # TTS 返回 opus 包列表而非文件路径
```

### 7.2 Agent 环境变量 `config/agent.env`

```env
DEEPSEEK_API_KEY='sk-...'           # 对话 LLM (deepseek-v4-pro) 的 API key
qwen_api_key='sk-...'               # 通义千问 (qwen-plus SQL 生成 + DashScope Embedding)
TAVILY_API_KEY='tvly-dev-...'       # Tavily 联网搜索
```

### 7.3 LLM 配置

| 用途 | 模型 | API 端点 | 配置位置 |
|------|------|----------|----------|
| 对话 + 工具调用 | `deepseek-v4-pro` | `https://api.deepseek.com` | `crm_agent.py` |
| SQL 生成（身份查询 + 画像更新） | `qwen-plus` | `https://dashscope.aliyuncs.com/compatible-mode/v1` | `crm_init.py` |
| 向量 Embedding | `text-embedding-v2` | DashScope | `crm_init.py` |

---

## 八、启动方式

### 8.1 环境要求

- **Python**：3.10.x（conda 环境 `D:/conda/envs/xiaozhi-esp32-server`）
- **关键依赖**：fastapi, uvicorn, langchain-openai, langgraph, langchain-chroma, langchain-tavily, opuslib-next, numpy 1.26.x
- **系统依赖**：ffmpeg（音频格式转换）、opus.dll（Windows 下需在 PATH 中）

### 8.2 启动命令

```bash
# 激活 conda 环境
conda activate xiaozhi-esp32-server

# 进入项目目录
cd E:/cyy/xiangmu2/xiaozhi-server

# 启动服务（默认端口 8003，host 0.0.0.0）
python app.py
```

### 8.3 前端启动

```bash
# 进入前端目录
cd E:/cyy/xiangmu1_5

# 激活 venv
venv/Scripts/activate

# 启动 Gradio 前端（默认端口 7860）
python jiemian.py
```

### 8.4 Windows opus.dll 路径修复

`app.py` 顶部包含引导代码，在 `import opuslib_next` 前自动将 conda `Library/bin/opus.dll` 加入 `os.environ['PATH']`。
如果直接运行独立脚本（非通过 `app.py` 启动），需手动设置：

```python
import os, sys
os.environ['PATH'] = os.path.join(os.path.dirname(sys.executable), 'Library', 'bin') + os.pathsep + os.environ.get('PATH', '')
```

---

## 九、目录结构

```
xiaozhi-server/
├── app.py                          # 入口（opus.dll PATH 修复 + uvicorn 启动）
├── config/
│   ├── agent.env                   # Agent 环境变量（3 个 API key）
│   ├── settings.py                 # 配置加载
│   └── logger.py                   # 日志配置
├── core/
│   ├── http_server_fastapi.py      # FastAPI 服务（6 端点 + SSE）
│   ├── session.py                  # SessionManager + FakeWebSocket
│   ├── connection.py               # ConnectionHandler（会话生命周期）
│   ├── agent/
│   │   ├── __init__.py             # 导出 run_agent_stream
│   │   ├── crm_agent.py            # LangGraph 图 + 节点 + 事件流
│   │   ├── crm_init.py             # 数据访问层（SQL + 向量库）
│   │   └── agent_bridge.py         # Agent 事件 → TTS/WebSocket 桥接
│   ├── handle/
│   │   ├── receiveAudioHandle.py   # startToChat() 入口
│   │   └── sendAudioHandle.py      # sendAudioMessage() 音频发送
│   ├── providers/
│   │   ├── asr/                    # ASR provider（base + BaiduASR 等）
│   │   └── tts/                    # TTS provider（base + TencentTTS 等）
│   └── utils/
│       ├── dialogue.py             # Dialogue 对象（兼容性保留）
│       ├── modules_initialize.py   # 模块初始化（ASR + TTS）
│       └── ...                     # 其他工具函数
├── data/
│   ├── .config.yaml                # 服务配置（selected_module + ASR/TTS 参数）
│   ├── travel.db                   # SQLite 数据库（7 张表）
│   └── vectorstore/                # 向量库目录（按 user_id 子目录）
│       └── 1/                      # user_id=1 的攻略向量库
├── CRM_README.md                   # 本文档
└── TEST_RECORD.md                  # 测试记录
```

---

## 十、关键设计决策

| 编号 | 决策 | 理由 |
|------|------|------|
| 方案1 | 废弃 `func_handler`/`intent`/`DIRECT_ANSWER_TOOL` | Agent 全权负责工具调用，避免意图分析与 Agent 工具调用的职责重叠 |
| 3.1-A | `update_profile` 移至后台线程 | 画像更新不阻塞 `done` 事件，用户无需等待 SQL 执行完成 |
| 3.4 | LLM 层约束纯文本输出 | 避免 Markdown 表格导致 TTS 超长切分失败（腾讯 TTS 单次限 150 字） |
| 4.1 | FakeWebSocket 适配器模式 | handle 模块零改动复用，`send()` 按类型路由到 asyncio.Queue |
| 4.3 | `trim_messages` 限制历史窗口 | 防止多轮对话 token 累积爆炸，max_tokens=4000 |
| 4.7 | SQLDatabase 延迟初始化 + 重连 | 避免启动时 DB 文件不存在导致崩溃，查询失败自动重连 |
| 4.9 | `cleanup_session_checkpoint` | 会话关闭时清理 InMemorySaver 对应 thread 的 checkpoint |
| SSE | StreamingResponse 流式返回 | 用户 ~2s 看到首句回复（vs 原 completion_event.wait() ~8-12s） |
