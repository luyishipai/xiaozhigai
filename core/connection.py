import os
import json
import uuid
import time
import queue
import asyncio
import threading

from typing import Dict, Any
from core.utils.modules_initialize import initialize_tts, initialize_asr
from core.providers.tts.default import DefaultTTS
from concurrent.futures import ThreadPoolExecutor
from core.utils.dialogue import Dialogue
from core.providers.asr.dto.dto import InterfaceType
from core.providers.tts.dto.dto import ContentType, TTSMessageDTO, SentenceType
from config.logger import setup_logging, build_module_string, create_connection_logger

TAG = __name__


class ConnectionHandler:
    """HTTP 模式下的会话处理器（精简版）。

    职责：会话生命周期管理 + TTS/ASR 组件初始化 + Agent 桥接。
    Agent 事件处理委托给 AgentBridge，TTS 分段逻辑在 base.py。
    """

    def __init__(self, config: Dict[str, Any], _asr=None, server=None):
        self.common_config = config
        self.config = config
        self.session_id = str(uuid.uuid4())
        self.logger = setup_logging()
        self.server = server

        # HTTP 模式适配器（FakeWebSocket）
        self.websocket = None
        self.headers = None
        self.device_id = None
        self.client_ip = None
        self.welcome_msg = None
        self.audio_format = "opus"
        self.sample_rate = 16000

        # 客户端状态
        self.client_abort = False
        self.client_is_speaking = False
        self.client_listen_mode = "auto"
        self.asr_audio = []  # ASR 音频缓存（流式 ASR 子类使用）
        self.current_speaker = ""  # 当前说话人（ASR 识别结果）

        # 线程任务
        self.loop = None
        self.stop_event = threading.Event()
        self.executor = ThreadPoolExecutor(max_workers=5)

        # 依赖的组件
        self.vad = None
        self.asr = None
        self.tts = None
        self._asr = _asr

        # 对话/会话相关
        self.dialogue = Dialogue()  # 保留对象兼容性，但不再写入（2.2）
        self.sentence_id = None
        self.tts_MessageText = ""
        self.user_info = ""  # CRM Agent 用户画像（由 verify_identity 节点填充）
        self.cmd_exit = self.config["exit_commands"]
        self.close_after_chat = False

        # 特性标记
        self.features = None

        # 超时
        self.timeout_seconds = int(self.config.get("close_connection_no_voice_time", 120)) + 60
        self.timeout_task = None

        # 活动时间
        self.last_activity_time = 0.0

    def initialize_session(self):
        self.last_activity_time = time.time() * 1000
        self.welcome_msg = self.config["xiaozhi"]
        self.welcome_msg["session_id"] = self.session_id
        self.sample_rate = self.welcome_msg["audio_params"]["sample_rate"]

    def _initialize_components(self):
        """初始化 TTS + ASR 组件（精简版，无 VAD/voiceprint/intent/memory/report）。"""
        try:
            if self.tts is None:
                self.tts = self._initialize_tts()
            asyncio.run_coroutine_threadsafe(
                self.tts.open_audio_channels(self), self.loop
            )

            if self.asr is None:
                self.asr = self._initialize_asr()
            asyncio.run_coroutine_threadsafe(
                self.asr.open_audio_channels(self), self.loop
            )

            self.selected_module_str = build_module_string(
                self.config.get("selected_module", {})
            )
            self.logger = create_connection_logger(self.selected_module_str)
            self.logger.bind(tag=TAG).info("组件初始化完成（TTS + ASR）")

        except Exception as e:
            self.logger.bind(tag=TAG).error(f"实例化组件失败: {e}")

    def _initialize_tts(self):
        """初始化 TTS"""
        tts = initialize_tts(self.config)
        if tts is None:
            tts = DefaultTTS(self.config, delete_audio_file=True)
        return tts

    def _initialize_asr(self):
        """初始化 ASR"""
        if (
            self._asr is not None
            and hasattr(self._asr, "interface_type")
            and self._asr.interface_type == InterfaceType.LOCAL
        ):
            return self._asr
        return initialize_asr(self.config)

    async def _background_initialize(self):
        """后台初始化组件（不阻塞主循环）"""
        try:
            self.executor.submit(self._initialize_components)
        except Exception as e:
            self.logger.bind(tag=TAG).error(f"后台初始化失败: {e}")

    def chat(self, query):
        """调用 CRM Agent 处理用户消息，流式输出到 TTS。

        2.1: Agent 事件处理委托给 AgentBridge
        2.8: 移除 depth 参数（Agent 自己管工具调用循环）
        3.6: 移除 emotion_flag / get_emotion
        """
        from core.agent.agent_bridge import AgentBridge
        bridge = AgentBridge(self)
        return bridge.process_agent_stream(query, getattr(self, "user_info", ""))

    def clearSpeakStatus(self):
        self.client_is_speaking = False
        self.logger.bind(tag=TAG).debug("清除服务端讲话状态")

    async def close(self, ws=None):
        """资源清理方法（2.3: 移除 asyncio.run/new_event_loop 反模式，4.9: 补充清理）"""
        try:
            # 触发停止事件
            if self.stop_event:
                self.stop_event.set()

            # 清空任务队列
            self.clear_queues()

            # 关闭 WebSocket（FakeWebSocket）
            _ws = ws or self.websocket
            if _ws:
                try:
                    if hasattr(_ws, "closed") and not _ws.closed:
                        await _ws.close()
                    elif hasattr(_ws, "state") and _ws.state.name != "CLOSED":
                        await _ws.close()
                    else:
                        await _ws.close()
                except Exception:
                    pass

            # 关闭 TTS / ASR
            if self.tts:
                await self.tts.close()
            if self.asr:
                await self.asr.close()

            # 关闭线程池
            if self.executor:
                try:
                    self.executor.shutdown(wait=False)
                except Exception as executor_error:
                    self.logger.bind(tag=TAG).error(f"关闭线程池时出错: {executor_error}")
                self.executor = None

            # 4.9: 清理 LangGraph InMemorySaver checkpoint
            try:
                from core.agent.crm_agent import cleanup_session_checkpoint
                cleanup_session_checkpoint(self.session_id)
            except Exception as e:
                self.logger.bind(tag=TAG).debug(f"清理 Agent checkpoint: {e}")

            # 4.4: 重置 user_info（防止会话复用时残留）
            self.user_info = ""

            self.logger.bind(tag=TAG).info("连接资源已释放")
        except Exception as e:
            self.logger.bind(tag=TAG).error(f"关闭连接时出错: {e}")
        finally:
            if self.stop_event:
                self.stop_event.set()

    def clear_queues(self):
        """清空任务队列"""
        if self.tts:
            self.logger.bind(tag=TAG).debug(
                f"开始清理: TTS队列大小={self.tts.tts_text_queue.qsize()}, "
                f"音频队列大小={self.tts.tts_audio_queue.qsize()}"
            )
            for q in [self.tts.tts_text_queue, self.tts.tts_audio_queue]:
                if not q:
                    continue
                while True:
                    try:
                        q.get_nowait()
                    except queue.Empty:
                        break

            if hasattr(self, "audio_rate_controller") and self.audio_rate_controller:
                self.audio_rate_controller.reset()
                self.logger.bind(tag=TAG).debug("已重置音频流控器")

            self.logger.bind(tag=TAG).debug(
                f"清理结束: TTS队列大小={self.tts.tts_text_queue.qsize()}, "
                f"音频队列大小={self.tts.tts_audio_queue.qsize()}"
            )
