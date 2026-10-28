
# ── Windows opus.dll 路径修复（必须在所有其他 import 之前执行）──────────────
# opuslib_next 内部用 ctypes.util.find_library("opus") 查找 DLL，
# 该函数在 Windows 下只搜索 os.environ['PATH'] 中的目录。
# conda 环境的 Library/bin 不一定在 PATH 中，需要手动添加。
import os as _os
import sys as _sys
import ctypes.util as _ctutil
if _ctutil.find_library("opus") is None:
    for _d in [
        _os.path.join(_os.path.dirname(_sys.executable), "Library", "bin"),  # conda env
        _os.path.dirname(_sys.executable),                                   # python.exe 同级
    ]:
        if _os.path.isfile(_os.path.join(_d, "opus.dll")):
            _os.environ["PATH"] = _d + _os.pathsep + _os.environ.get("PATH", "")
            break
# ── 修复结束 ──────────────────────────────────────────────────────────────────

import asyncio
import uuid

import uvicorn

from config.settings import load_config
from config.logger import setup_logging
from core.utils.util import check_ffmpeg_installed, get_local_ip

TAG = __name__
logger = setup_logging()


def main():
    """初始化检查 + 读取端口 + 启动 uvicorn"""
    check_ffmpeg_installed()
    config = asyncio.run(load_config())

    # auth_key（HTTP API 不需要 WS 认证，但保留用于其他接口）
    auth_key = config["server"].get("auth_key", "")
    if not auth_key or len(auth_key) == 0 or "你" in auth_key:
        auth_key = config.get("manager-api", {}).get("secret", "")
        if not auth_key or len(auth_key) == 0 or "你" in auth_key:
            auth_key = str(uuid.uuid4().hex)
    config["server"]["auth_key"] = auth_key

    port = int(config["server"].get("http_port", 8003))
    host = config["server"].get("http_host", "0.0.0.0")

    logger.bind(tag=TAG).info(
        f"ASR 接口:  http://{get_local_ip()}:{port}/api/asr"
    )
    logger.bind(tag=TAG).info(
        f"TTS 接口:  http://{get_local_ip()}:{port}/api/tts"
    )
    logger.bind(tag=TAG).info(
        f"Chat 接口: http://{get_local_ip()}:{port}/api/chat"
    )

    # 启动 FastAPI 服务（uvicorn.run 是同步阻塞调用，内部管理事件循环）
    uvicorn.run(
        "core.http_server_fastapi:app",
        host=host,
        port=port,
        reload=False,
    )


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("手动中断，程序终止。")