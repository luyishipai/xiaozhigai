"""
FastAPI 入口：替换原 websocket_server + http_server。

提供三个核心接口：
  POST /api/asr    音频→文字
  POST /api/tts    文字→音频
  POST /api/chat   完整对话（文字输入 或 音频输入）

设计要点：
  - 每个请求自动创建/复用 ConnectionHandler 会话
  - handle 模块完全不动（通过 FakeWebSocket 收集输出）
  - TTS 流控、function_call、intent、memory 全部复用
"""
import asyncio
import io
import json
import os
import sys
import time
import wave
import struct
import numpy as np
from typing import Optional, List
from contextlib import asynccontextmanager

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse, Response
from pydantic import BaseModel

from config.logger import setup_logging
from config.settings import load_config
from core.utils.util import check_ffmpeg_installed
from core.utils.modules_initialize import initialize_modules
from core.session import SessionManager

TAG = __name__
logger = setup_logging()

# 全局单例
session_manager: Optional[SessionManager] = None
_app_config: dict = {}

# ─── 启动/关闭生命周期 ────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    global session_manager, _app_config
    check_ffmpeg_installed()
    config = await load_config()
    _app_config = config

    # 工厂层初始化（和原 websocket_server 一样！）
    modules = initialize_modules(
        logger, config,
        init_vad=True, init_asr=True, init_llm=True, init_memory=True, init_intent=True, init_tts=False
    )

    session_manager = SessionManager(
        config=config,
        vad=modules["vad"],
        asr=modules["asr"],
        llm=modules["llm"],
        memory=modules["memory"],
        intent=modules["intent"],
    )

    # 定时清理空闲会话（每 5 分钟跑一次，清理超过 30 分钟无活动的会话）
    async def _cleanup_loop():
        while True:
            await asyncio.sleep(300)
            try:
                await session_manager.cleanup_expired(idle_seconds=1800)
                logger.bind(tag=TAG).debug("空闲会话清理完成")
            except Exception as e:
                logger.bind(tag=TAG).warning(f"空闲会话清理异常: {e}")

    cleanup_task = asyncio.create_task(_cleanup_loop())

    logger.bind(tag=TAG).info("FastAPI 问答服务启动完成")
    yield

    # 关闭：取消清理任务，释放所有会话资源
    cleanup_task.cancel()
    try:
        await cleanup_task
    except asyncio.CancelledError:
        pass
    if session_manager:
        await session_manager.close_all()
    session_manager = None

app = FastAPI(
    title="小智问答服务",
    description="从 xiaozhi-server 提取的独立语音问答核心服务",
    version="2.0.0",
    lifespan=lifespan,
)

# ─── 工具函数 ────────────────────────────────────────────────────────────

def wav_to_pcm16k(audio_bytes: bytes) -> bytes:
    """wav → 16kHz / 单声道 / 16bit PCM"""
    try:
        with wave.open(io.BytesIO(audio_bytes), "rb") as wf:
            n_channels = wf.getnchannels()
            framerate = wf.getframerate()
            raw = wf.readframes(wf.getnframes())
        audio_np = np.frombuffer(raw, dtype=np.int16)
        if n_channels > 1:
            audio_np = audio_np.reshape(-1, n_channels).mean(axis=1).astype(np.int16)
        if framerate != 16000:
            ratio = 16000 / framerate
            indices = np.linspace(0, len(audio_np) - 1, int(len(audio_np) * ratio))
            audio_np = np.interp(indices, np.arange(len(audio_np)), audio_np).astype(np.int16)
        return audio_np.tobytes()
    except Exception as e:
        logger.bind(tag=TAG).warning(f"wav 解析失败: {e}")
        return audio_bytes

def opus_chunks_to_wav_bytes(audio_chunks: list, sample_rate: int = 16000) -> bytes:
    """将 opus 包列表逐包解码为 PCM，再拼成 WAV 字节流。

    audio_chunks 中每个元素是一个独立的 opus 编码包（60ms/帧），
    直接 b"".join 会丢失包边界导致无法解码，必须逐包 decode。
    """
    try:
        # Windows 下 ctypes.util.find_library("opus") 只搜索 os.environ['PATH'] 中的目录，
        # conda 环境的 Library/bin 可能不在其中，需要手动添加
        import ctypes.util
        if ctypes.util.find_library("opus") is None:
            _candidate_dirs = [
                os.path.join(os.path.dirname(sys.executable), "Library", "bin"),  # conda env
                os.path.dirname(sys.executable),  # python.exe 同级目录
            ]
            for _d in _candidate_dirs:
                _opus_dll = os.path.join(_d, "opus.dll")
                if os.path.isfile(_opus_dll):
                    os.environ["PATH"] = _d + os.pathsep + os.environ.get("PATH", "")
                    break

        import opuslib_next
        frame_size = int(sample_rate * 60 / 1000)  # 960 samples
        decoder = opuslib_next.Decoder(sample_rate, 1)
        pcm_data = bytearray()
        for chunk in audio_chunks:
            if not chunk:
                continue
            try:
                pcm_frame = decoder.decode(chunk, frame_size)
                pcm_data.extend(pcm_frame)
            except Exception:
                continue
        if not pcm_data:
            return b""
        wav_buf = io.BytesIO()
        with wave.open(wav_buf, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sample_rate)
            wf.writeframes(bytes(pcm_data))
        return wav_buf.getvalue()
    except Exception as e:
        logger.bind(tag=TAG).warning(f"opus → wav 转换失败: {e}")
        return b""

async def _wait_for_tts_ready(conn, timeout: float = 10.0):
    """等待 TTS 初始化完成（conn.tts 不为 None 且 open_audio_channels 已调用）。

    _initialize_components 在线程池里异步执行，会先创建 TTS 实例，
    再通过 run_coroutine_threadsafe 调 open_audio_channels 设置 tts.conn。
    这里轮询等待两个条件都满足。
    """
    deadline = time.time() + timeout
    while time.time() < deadline:
        if conn.tts is not None and hasattr(conn.tts, "conn") and conn.tts.conn is not None:
            return True
        await asyncio.sleep(0.1)
    return False

# ─── 请求/响应模型 ────────────────────────────────────────────────────────

class ChatTextRequest(BaseModel):
    text: str
    session_id: Optional[str] = None
    voiceprint: Optional[str] = None  # 说话人名称

class ChatResponse(BaseModel):
    code: int = 0
    session_id: str
    asr_text: Optional[str] = None    # 用户输入的文字
    reply_text: str
    audio_base64: Optional[str] = None  # TTS 合成的音频（base64）
    audio_type: Optional[str] = None

# ─── 1. ASR 接口 ─────────────────────────────────────────────────────────

@app.post("/api/asr", summary="语音识别")
async def api_asr(file: UploadFile = File(...)):
    """上传 wav，返回识别文本"""
    conn = None
    session_id = None
    try:
        session_id, conn = await session_manager.create_session()

        audio_bytes = await file.read()
        pcm_bytes = wav_to_pcm16k(audio_bytes)

        # 优先使用批量识别（流式 ASR 如 DoubaoStream 在 HTTP 场景下需要）
        asr = conn._asr
        if hasattr(asr, "batch_recognize"):
            content = await asr.batch_recognize(pcm_bytes, asr.rate)
        else:
            text, _ = await asr.speech_to_text_wrapper([pcm_bytes], session_id)
            if isinstance(text, dict):
                content = text.get("content", "").strip()
            else:
                content = str(text).strip()

        return JSONResponse({"code": 0, "text": content, "session_id": session_id})

    except Exception as e:
        logger.bind(tag=TAG).error(f"ASR 异常: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if conn:
            await session_manager.close_session(session_id)

# ─── 2. TTS 接口 ─────────────────────────────────────────────────────────

@app.post("/api/tts", summary="文字转语音")
async def api_tts(req: ChatTextRequest):
    """传入 text，返回合成音频"""
    conn = None
    session_id = None
    try:
        session_id, conn = await session_manager.create_session()

        # 等待 TTS 初始化完成（_initialize_components 在线程池异步执行）
        if not await _wait_for_tts_ready(conn, timeout=10.0):
            raise HTTPException(status_code=503, detail="TTS 初始化超时，请重试")

        # to_tts() 是同步方法，放到线程池执行
        result = await asyncio.to_thread(conn.tts.to_tts, req.text)

        if result is None:
            raise HTTPException(status_code=500, detail="TTS 合成失败，请检查网络或服务配置")

        # to_tts() 返回两种类型：
        #   1. list[bytes] — opus 编码的音频数据块（delete_audio_file=True 时）
        #   2. str — 音频文件路径（delete_audio_file=False 时）
        if isinstance(result, list):
            audio_bytes = b"".join(result)
            return Response(
                content=audio_bytes,
                media_type="audio/opus",
                headers={
                    "Content-Disposition": 'attachment; filename="tts.opus"',
                    "X-Session-Id": session_id,
                },
            )
        elif isinstance(result, str):
            # 文件路径：读取文件内容并返回
            import os
            if not os.path.exists(result):
                raise HTTPException(status_code=500, detail="TTS 音频文件不存在")
            with open(result, "rb") as f:
                audio_bytes = f.read()
            ext = os.path.splitext(result)[1].lower()
            media_types = {
                ".wav": "audio/wav",
                ".mp3": "audio/mpeg",
                ".opus": "audio/opus",
                ".pcm": "audio/pcm",
            }
            media_type = media_types.get(ext, "application/octet-stream")
            filename = os.path.basename(result)
            return Response(
                content=audio_bytes,
                media_type=media_type,
                headers={
                    "Content-Disposition": f'attachment; filename="{filename}"',
                    "X-Session-Id": session_id,
                },
            )
        else:
            raise HTTPException(status_code=500, detail=f"TTS 返回未知类型: {type(result)}")

    except HTTPException:
        raise
    except Exception as e:
        logger.bind(tag=TAG).error(f"TTS 异常: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if conn:
            await session_manager.close_session(session_id)

# ─── 3. Chat 接口（核心！复用完整链路） ──────────────────────────────────

@app.post("/api/chat", summary="完整问答")
async def api_chat(
    file: Optional[UploadFile] = File(None, description="音频输入（可选，与 text 二选一）"),
    text: Optional[str] = Form(None, description="文字输入（可选，与 file 二选一）"),
    session_id: Optional[str] = Form(None, description="会话ID（多轮对话传同一个）"),
    output_audio: bool = Form(True, description="是否返回TTS音频"),
):
    """
    完整问答流程：
    - 音频模式：wav → ASR → LLM → TTS → 返回
    - 文字模式：text → LLM → TTS → 返回
    - 多轮对话：传同一个 session_id
    """
    from core.handle.receiveAudioHandle import startToChat

    conn = None
    _session_lock = None
    _lock_acquired = False
    try:
        # ① 获取或创建会话
        if session_id:
            conn = await session_manager.get_session(session_id)

        if conn is None:
            session_id, conn = await session_manager.create_session()
            # 等待 TTS 初始化完成（startToChat 链路依赖 conn.tts）
            if not await _wait_for_tts_ready(conn, timeout=10.0):
                logger.bind(tag=TAG).warning("TTS 初始化超时，chat 链路可能异常")
        else:
            # 复用已有会话：获取并发锁，防止同一 session 的并发请求互相干扰
            _session_lock = session_manager.get_session_lock(session_id)
            if _session_lock:
                try:
                    await asyncio.wait_for(_session_lock.acquire(), timeout=30.0)
                    _lock_acquired = True
                except asyncio.TimeoutError:
                    raise HTTPException(
                        status_code=429,
                        detail="该会话正在处理另一请求，请稍后重试"
                    )
            # 刷新活动时间，防止被 cleanup_expired 误清理
            conn.last_activity_time = time.time() * 1000

        # ② 如果有音频，先 ASR
        asr_text = None
        if file:
            audio_bytes = await file.read()
            pcm_bytes = wav_to_pcm16k(audio_bytes)

            # 优先使用批量识别（流式 ASR 如 DoubaoStream 在 HTTP 场景下需要）
            asr = conn._asr
            if hasattr(asr, "batch_recognize"):
                asr_text = await asr.batch_recognize(pcm_bytes, asr.rate)
            else:
                result = await asr.speech_to_text_wrapper([pcm_bytes], session_id)
                raw_text, _ = result
                asr_text = raw_text.get("content", "") if isinstance(raw_text, dict) else str(raw_text)
            user_input = asr_text
        elif text:
            user_input = text
        else:
            raise HTTPException(status_code=400, detail="请提供 text 或 file 之一")

        if not user_input.strip():
            return JSONResponse({"code": -1, "error": "输入为空", "session_id": session_id})

        logger.bind(tag=TAG).info(f"Chat 输入: {user_input[:60]}")

        # ③ 调 startToChat —— 复用整个意图分析 → function_call → chat → TTS 链路！
        # 清除上一次的完成信号（多轮对话复用同一 session）
        conn.websocket.completion_event.clear()
        await startToChat(conn, user_input)

        # 等待 TTS 全链路完成：chat() → tts_text_queue → TTS线程 → sendAudioMessage
        # → send_tts_message(conn, "stop") → FakeWebSocket 收到 stop → completion_event.set()
        try:
            await asyncio.wait_for(
                conn.websocket.completion_event.wait(), timeout=30
            )
        except asyncio.TimeoutError:
            logger.bind(tag=TAG).warning("TTS 完成超时(30s)，返回已有数据")

        # ④ 从 FakeWebSocket 取输出
        sent_items = conn.websocket.get_and_clear()
        reply_text = ""
        audio_chunks = []

        for item_type, item_data in sent_items:
            if item_type == "json":
                try:
                    msg = json.loads(item_data) if isinstance(item_data, str) else item_data
                    if isinstance(msg, dict):
                        if msg.get("type") == "stt":
                            asr_text = msg.get("text", asr_text)
                        # 从 TTS sentence_start 消息提取回复文本
                        if msg.get("type") == "tts" and msg.get("state") == "sentence_start":
                            t = msg.get("text")
                            if t:
                                reply_text += t
                except Exception:
                    pass
            elif item_type == "audio":
                audio_chunks.append(item_data)

        # 如果没从 TTS 消息提取到文本，从 dialogue 拿最后一条 assistant 消息
        if not reply_text and conn.dialogue:
            reply_text = conn.dialogue.get_last_assistant()

        # ⑤ 组装响应
        import base64
        response_body = {
            "code": 0,
            "session_id": session_id,
            "asr_text": asr_text,
            "reply_text": reply_text,
        }

        if output_audio and audio_chunks:
            wav_bytes = opus_chunks_to_wav_bytes(audio_chunks, conn.sample_rate)
            if wav_bytes:
                response_body["audio_base64"] = base64.b64encode(wav_bytes).decode("utf-8")
                response_body["audio_type"] = "wav"

        response = JSONResponse(response_body)
        response.headers["X-Session-Id"] = session_id
        return response

    except Exception as e:
        logger.bind(tag=TAG).error(f"Chat 异常: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        # 释放会话并发锁（仅对复用的会话有效）
        if _lock_acquired:
            _session_lock.release()

# ─── 4. 会话管理接口 ─────────────────────────────────────────────────────

@app.get("/api/session/{session_id}", summary="查询会话状态")
async def api_get_session(session_id: str):
    conn = await session_manager.get_session(session_id)
    if not conn:
        raise HTTPException(status_code=404, detail="会话不存在")
    dialogue_count = len(conn.dialogue.dialogue) if conn.dialogue else 0
    return JSONResponse({
        "code": 0,
        "session_id": session_id,
        "dialogue_count": dialogue_count,
    })

@app.delete("/api/session/{session_id}", summary="删除会话")
async def api_delete_session(session_id: str):
    await session_manager.close_session(session_id)
    return JSONResponse({"code": 0})

# ─── 健康检查 ────────────────────────────────────────────────────────────

@app.get("/health", summary="健康检查")
async def health():
    return JSONResponse({"status": "ok"})
