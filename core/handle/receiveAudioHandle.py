import json
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from core.connection import ConnectionHandler
from core.handle.sendAudioHandle import send_stt_message

TAG = __name__


async def startToChat(conn: "ConnectionHandler", text):
    """处理用户文本输入，启动 Agent 对话。

    精简版（1.1/1.8/3.3）：
    - 移除 handle_user_intent 调用（Agent 全权负责意图分析）
    - 移除 VAD/AEC/设备绑定/输出限制逻辑（HTTP 模式不需要）
    - 直接 send_stt_message → conn.chat()
    """
    # 解析 JSON 格式输入（说话人信息）
    actual_text = text
    try:
        if text.strip().startswith("{") and text.strip().endswith("}"):
            data = json.loads(text)
            if "content" in data:
                actual_text = data["content"]
                if "speaker" in data:
                    conn.current_speaker = data["speaker"]
    except (json.JSONDecodeError, KeyError):
        pass

    # 发送 STT 消息（通知前端识别到的文本 + TTS start 信号）
    await send_stt_message(conn, actual_text)

    # 准备新会话
    conn.client_abort = False

    # 在线程池中执行 chat（Agent + TTS 流程）
    conn.executor.submit(conn.chat, actual_text)
