"""
会话管理模块：把 ConnectionHandler 从 WebSocket 连接处理器
改造成 HTTP API 的会话管理器。

核心思路：给 ConnectionHandler 注入一个"伪 WebSocket 对象"，
它的 send() 方法把数据收集到队列里，FastAPI 路由层从队列取出返回。
"""
import asyncio
import json
import time
import uuid
import threading
from typing import Optional, Dict, Any, List, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from core.connection import ConnectionHandler

class FakeWebSocket:
    """
    伪装的 websockets.ServerConnection 对象。
    只实现 handle 模块用到的 send() 方法，收集数据到列表。

    当 sendAudioHandle 发出 {"type":"tts","state":"stop"} 时，
    表示 chat() + TTS 全链路完成，设置 completion_event 通知路由层。
    """

    def __init__(self):
        self.sent: List[Tuple[str, Any]] = []
        self._lock = asyncio.Lock()
        self.closed = False
        self.completion_event = asyncio.Event()
        self.state = type("State", (), {"name": "OPEN"})()

    async def send(self, data):
        """handle 模块调 conn.websocket.send() 时走这里"""
        async with self._lock:
            if isinstance(data, bytes):
                self.sent.append(("audio", data))
            else:
                self.sent.append(("json", data))
                # 检测 TTS 完成信号
                # sendAudioHandle.sendAudioMessage 在 SentenceType.LAST 时
                # 调 send_tts_message(conn, "stop", None) → 发出这条 JSON
                try:
                    msg = json.loads(data) if isinstance(data, str) else data
                    if (isinstance(msg, dict)
                            and msg.get("type") == "tts"
                            and msg.get("state") == "stop"):
                        self.completion_event.set()
                except Exception:
                    pass

    async def close(self):
        self.closed = True
        self.state.name = "CLOSED"

    def get_and_clear(self):
        """FastAPI 路由取走所有输出"""
        items = self.sent[:]
        self.sent.clear()
        return items

class SessionManager:
    """
    全局会话管理器。
    按 session_id 持有 ConnectionHandler 实例，支持多轮对话。
    """

    def __init__(self, config, vad, asr, llm, memory, intent):
        self.config = config
        self._vad = vad
        self._asr = asr
        self._llm = llm
        self._memory = memory
        self._intent = intent
        self._sessions: Dict[str, "ConnectionHandler"] = {}
        self._session_locks: Dict[str, asyncio.Lock] = {}
        self._lock = asyncio.Lock()

    async def create_session(self) -> Tuple[str, "ConnectionHandler"]:
        """创建新会话"""
        from core.connection import ConnectionHandler

        conn = ConnectionHandler(
            self.config,
            self._vad,
            self._asr,
            self._llm,
            self._memory,
            self._intent,
        )
        # 注入伪 WebSocket（关键！handle 模块调 conn.websocket.send() 走这里）
        conn.websocket = FakeWebSocket()
        conn.loop = asyncio.get_running_loop()
        conn.headers = {}  # HTTP 场景无 WS headers，给空 dict

        # 初始化基础状态（原来 handle_connection 里的非 WS 部分）
        conn.initialize_session()
        # 后台初始化 TTS / Intent / 插件 等组件
        await conn._background_initialize()

        async with self._lock:
            self._sessions[conn.session_id] = conn
            self._session_locks[conn.session_id] = asyncio.Lock()

        return conn.session_id, conn

    async def get_session(self, session_id: str) -> Optional["ConnectionHandler"]:
        """获取已有会话"""
        async with self._lock:
            return self._sessions.get(session_id)

    def get_session_lock(self, session_id: str) -> Optional[asyncio.Lock]:
        """获取会话的并发锁（防止同一 session 的并发请求互相干扰）"""
        return self._session_locks.get(session_id)

    async def close_session(self, session_id: str):
        """关闭会话并清理资源（TTS/ASR 线程、executor 等）"""
        async with self._lock:
            conn = self._sessions.pop(session_id, None)
            self._session_locks.pop(session_id, None)
        if conn:
            conn.stop_event.set()
            try:
                await conn.close()
            except Exception:
                pass

    async def cleanup_expired(self, idle_seconds: int = 1800):
        """清理空闲会话（默认30分钟无活动）"""
        now = time.time() * 1000
        expired_ids = []
        async with self._lock:
            for sid, conn in self._sessions.items():
                if now - conn.last_activity_time > idle_seconds * 1000:
                    expired_ids.append(sid)
        for sid in expired_ids:
            await self.close_session(sid)

    async def close_all(self):
        """关闭所有会话（服务关闭时调用）"""
        async with self._lock:
            all_ids = list(self._sessions.keys())
        for sid in all_ids:
            await self.close_session(sid)