"""
AgentBridge：LangGraph Agent 事件 → TTS 队列的适配层。

从 ConnectionHandler.chat() 中提取，职责单一：
1. 调用 run_agent_stream() 获取 Agent 事件流
2. 将 text 事件 → TTS 文本队列
3. 将 tool_call/tool_result 事件 → WebSocket (FakeWebSocket)
4. 将 identity 事件 → 回写 conn.user_info
5. 将 error 事件 → 友好中文提示（不播报英文异常）
6. 首 token 超时 → 向前端发送"正在思考中..."
7. tool_call 去重（防止前端闪烁）
"""

import json
import uuid
import asyncio
import threading

from core.providers.tts.dto.dto import ContentType, TTSMessageDTO, SentenceType

TAG = __name__


class AgentBridge:
    """LangGraph Agent 与 TTS 管线之间的桥接器。"""

    def __init__(self, conn):
        self.conn = conn
        self.session_id = conn.session_id
        self.logger = conn.logger
        self.loop = conn.loop

    def process_agent_stream(self, query, user_info=""):
        """处理 Agent 事件流，输出到 TTS 队列。

        Returns: True（始终，供 conn.chat() 返回）
        """
        from core.agent import run_agent_stream

        current_sentence_id = str(uuid.uuid4().hex)
        self.conn.sentence_id = current_sentence_id
        self.logger.bind(tag=TAG).info(f"大模型收到用户消息: {query}")

        # TTS FIRST 标记
        self.conn.tts.tts_text_queue.put(
            TTSMessageDTO(
                sentence_id=current_sentence_id,
                sentence_type=SentenceType.FIRST,
                content_type=ContentType.ACTION,
            )
        )

        response_message = []
        seen_tool_calls = set()  # 4.2: 去重 tool_call 事件
        first_token_received = threading.Event()

        # 3.5: 首 token 超时 → 发送"正在思考中..."
        def _send_thinking_hint():
            if not first_token_received.is_set():
                asyncio.run_coroutine_threadsafe(
                    self.conn.websocket.send(
                        json.dumps(
                            {"type": "status", "state": "thinking", "text": "正在思考中..."},
                            ensure_ascii=False,
                        )
                    ),
                    self.loop,
                )

        thinking_timer = threading.Timer(8.0, _send_thinking_hint)
        thinking_timer.daemon = True
        thinking_timer.start()

        try:
            # 4.8: 传递 logger 使 Agent 日志带上 session_id
            for event in run_agent_stream(
                self.session_id, query, user_info,
                logger=self.logger, tag=TAG,
            ):
                if self.conn.client_abort:
                    break

                evt_type = event.get("type")

                if evt_type == "text":
                    content = event["content"]
                    if content:
                        if not first_token_received.is_set():
                            first_token_received.set()
                            thinking_timer.cancel()
                        response_message.append(content)
                        self.conn.tts.tts_text_queue.put(
                            TTSMessageDTO(
                                sentence_id=current_sentence_id,
                                sentence_type=SentenceType.MIDDLE,
                                content_type=ContentType.TEXT,
                                content_detail=content,
                            )
                        )

                elif evt_type == "tool_call":
                    tool_name = event["name"]
                    # 4.2: 同名工具只通知一次
                    if tool_name not in seen_tool_calls:
                        seen_tool_calls.add(tool_name)
                        self.logger.bind(tag=TAG).info(f"Agent 调用工具: {tool_name}")
                        if self.conn.websocket:
                            asyncio.run_coroutine_threadsafe(
                                self.conn.websocket.send(
                                    json.dumps(
                                        {"type": "agent_tool", "event": "tool_call", "name": tool_name},
                                        ensure_ascii=False,
                                    )
                                ),
                                self.loop,
                            )

                elif evt_type == "tool_result":
                    result_content = event["content"]
                    self.logger.bind(tag=TAG).debug(f"工具结果: {result_content[:100]}")
                    if self.conn.websocket:
                        asyncio.run_coroutine_threadsafe(
                            self.conn.websocket.send(
                                json.dumps(
                                    {"type": "agent_tool", "event": "tool_result", "content": result_content},
                                    ensure_ascii=False,
                                )
                            ),
                            self.loop,
                        )

                elif evt_type == "identity":
                    self.conn.user_info = event["user_info"]
                    self.logger.bind(tag=TAG).info(
                        f"身份验证完成: user_id={event.get('user_id', '')}"
                    )

                elif evt_type == "error":
                    error_msg = event["content"]
                    self.logger.bind(tag=TAG).error(f"Agent 错误: {error_msg}")
                    # 2.9: 不播报英文异常，用友好中文提示
                    self.conn.tts.tts_text_queue.put(
                        TTSMessageDTO(
                            sentence_id=current_sentence_id,
                            sentence_type=SentenceType.MIDDLE,
                            content_type=ContentType.TEXT,
                            content_detail="抱歉，我遇到了一些问题，请稍后再试。",
                        )
                    )

                elif evt_type == "done":
                    break

        except Exception as e:
            self.logger.bind(tag=TAG).error(f"Agent 执行出错: {e}", exc_info=True)
            self.conn.tts.tts_text_queue.put(
                TTSMessageDTO(
                    sentence_id=current_sentence_id,
                    sentence_type=SentenceType.MIDDLE,
                    content_type=ContentType.TEXT,
                    content_detail="抱歉，我遇到了一些问题，请稍后再试。",
                )
            )
        finally:
            thinking_timer.cancel()
            if not first_token_received.is_set():
                first_token_received.set()

        # 存储完整回复文本（供 TTS store_tts_text 用）
        if response_message:
            text_buff = "".join(response_message)
            self.conn.tts.store_tts_text(current_sentence_id, text_buff)

        # TTS LAST 标记
        self.conn.tts.tts_text_queue.put(
            TTSMessageDTO(
                sentence_id=current_sentence_id,
                sentence_type=SentenceType.LAST,
                content_type=ContentType.ACTION,
            )
        )

        return True
