from typing import Dict, Any
from config.logger import setup_logging
from core.utils import tts, asr

TAG = __name__
logger = setup_logging()


def initialize_modules(
    logger,
    config: Dict[str, Any],
    init_asr=False,
    init_tts=False,
) -> Dict[str, Any]:
    """
    初始化模块组件（精简版：仅 ASR + TTS，移除 LLM/Intent/Memory/VAD）。

    Args:
        config: 配置字典
        init_asr: 是否初始化 ASR
        init_tts: 是否初始化 TTS

    Returns:
        Dict[str, Any]: 包含初始化后的模块的字典
    """
    modules = {}

    # 初始化TTS模块
    if init_tts:
        select_tts_module = config["selected_module"]["TTS"]
        modules["tts"] = initialize_tts(config)
        logger.bind(tag=TAG).info(f"初始化组件: tts成功 {select_tts_module}")

    # 初始化ASR模块
    if init_asr:
        select_asr_module = config["selected_module"]["ASR"]
        modules["asr"] = initialize_asr(config)
        logger.bind(tag=TAG).info(f"初始化组件: asr成功 {select_asr_module}")

    return modules


def initialize_tts(config):
    select_tts_module = config["selected_module"]["TTS"]
    tts_config = config["TTS"][select_tts_module].copy()
    tts_config.setdefault("tts_timeout", config.get("tts_timeout", 15))
    tts_type = (
        select_tts_module
        if "type" not in tts_config
        else tts_config["type"]
    )
    new_tts = tts.create_instance(
        tts_type,
        tts_config,
        str(config.get("delete_audio", True)).lower() in ("true", "1", "yes"),
    )
    return new_tts


def initialize_asr(config):
    select_asr_module = config["selected_module"]["ASR"]
    asr_type = (
        select_asr_module
        if "type" not in config["ASR"][select_asr_module]
        else config["ASR"][select_asr_module]["type"]
    )
    new_asr = asr.create_instance(
        asr_type,
        config["ASR"][select_asr_module],
        str(config.get("delete_audio", True)).lower() in ("true", "1", "yes"),
    )
    logger.bind(tag=TAG).info("ASR模块初始化完成")
    return new_asr
