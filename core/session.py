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
    HTTP 模式适配器：把 handle 模块的 conn.websocket.send() 调用
    转化为 asyncio.Queue 的 put() 操作，供 /api/chat SSE 流式消费。

    - bytes  → put(("audio", data))   — opus 音频包
    - JSON 含 {"type":"tts","state":"stop"} → put(None) — 哨兵，TTS 全链路完成
    - 其他 JSON → put(("json", data))  — 状态消息（stt / sentence_start / start）
    """

    def __init__(self):
        self.audio_queue: asyncio.Queue = asyncio.Queue()
        self.closed = False
        self.state = type("State", (), {"name": "OPEN"})()

    async def send(self, data):
        """handle 模块调 conn.websocket.send() 时走这里"""
        if self.closed:
            return
        if isinstance(data, bytes):
            # opus 音频包
            await self.audio_queue.put(("audio", data))
        else:
            # JSON 文本：检测 TTS stop 信号
            try:
                msg = json.loads(data) if isinstance(data, str) else data
                if (isinstance(msg, dict)
                        and msg.get("type") == "tts"
                        and msg.get("state") == "stop"):
                    # 哨兵值 None：通知 SSE 消费者 TTS 全链路完成
                    await self.audio_queue.put(None)
                    return
            except Exception:
                pass
            await self.audio_queue.put(("json", data))

    async def close(self):
        self.closed = True
        self.state.name = "CLOSED"
        # 确保消费者不会永远阻塞在 queue.get()
        await self.audio_queue.put(None)

    def reset_queue(self):
        """为新一轮对话重置队列（多轮对话复用同一 session）"""
        self.audio_queue = asyncio.Queue()

class SessionManager:
    """
    全局会话管理器。
    按 session_id 持有 ConnectionHandler 实例，支持多轮对话。
    """

    def __init__(self, config, asr):
        self.config = config
        self._asr = asr
        self._sessions: Dict[str, "ConnectionHandler"] = {}
        self._session_locks: Dict[str, asyncio.Lock] = {}
        self._lock = asyncio.Lock()

    async def create_session(self) -> Tuple[str, "ConnectionHandler"]:
        """创建新会话"""
        from core.connection import ConnectionHandler

        conn = ConnectionHandler(
            self.config,
            self._asr,
        )
        # 注入适配器（关键！handle 模块调 conn.websocket.send() → asyncio.Queue.put()）
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