import json
import time
import asyncio
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from core.connection import ConnectionHandler
from core.utils import textUtils
from core.utils.util import audio_to_data
from core.providers.tts.dto.dto import SentenceType
from core.utils.audioRateController import AudioRateController

TAG = __name__
# 音频帧时长（毫秒）
AUDIO_FRAME_DURATION = 60
# 预缓冲包数量，直接发送以减少延迟
PRE_BUFFER_COUNT = 5


async def sendAudioMessage(conn: "ConnectionHandler", sentenceType, audios, text, sentence_id=None):
    # 跳过旧句子残留音频
    if sentence_id is not None and sentence_id != conn.sentence_id:
        return

    if conn.tts.tts_audio_first_sentence:
        conn.logger.bind(tag=TAG).info(f"发送第一段语音: {text}")
        conn.tts.tts_audio_first_sentence = False

    if sentenceType == SentenceType.FIRST:
        if (
            hasattr(conn, "audio_rate_controller")
            and conn.audio_rate_controller
            and getattr(conn, "audio_flow_control", {}).get("sentence_id")
            == conn.sentence_id
        ):
            conn.audio_rate_controller.add_message(
                lambda: send_tts_message(conn, "sentence_start", text)
            )
        else:
            await send_tts_message(conn, "sentence_start", text)

    await sendAudio(conn, audios)
    if sentenceType is not SentenceType.MIDDLE:
        conn.logger.bind(tag=TAG).info(f"发送音频消息: {sentenceType}, {text}")

    if sentenceType == SentenceType.LAST:
        await send_tts_message(conn, "stop", None)
        if conn.close_after_chat:
            await conn.close()


async def _wait_for_audio_completion(conn: "ConnectionHandler"):
    """等待音频队列清空并等待预缓冲包播放完成"""
    if hasattr(conn, "audio_rate_controller") and conn.audio_rate_controller:
        rate_controller = conn.audio_rate_controller
        send_delay_ms = conn.config.get("tts_audio_send_delay", 0)
        conn.logger.bind(tag=TAG).debug(
            f"等待音频发送完成，队列中还有 {len(rate_controller.queue)} 个包"
        )
        await rate_controller.queue_empty_event.wait()

        if send_delay_ms > 0:
            playback_time = 2 * rate_controller.interval_ms / 1000.0
        else:
            playback_time = (PRE_BUFFER_COUNT + 2) * rate_controller.interval_ms / 1000.0
        await asyncio.sleep(playback_time)

        conn.logger.bind(tag=TAG).debug("音频发送完成")


async def sendAudio(
    conn: "ConnectionHandler", audios, frame_duration=AUDIO_FRAME_DURATION
):
    """发送音频包，使用 AudioRateController 进行精确的流量控制"""
    if audios is None or len(audios) == 0:
        return

    send_delay_ms = conn.config.get("tts_audio_send_delay", 0)
    is_single_packet = isinstance(audios, bytes)

    rate_controller, flow_control = _get_or_create_rate_controller(
        conn, frame_duration, is_single_packet, send_delay_ms
    )

    audio_list = [audios] if is_single_packet else audios

    await _send_audio_with_rate_control(
        conn, audio_list, rate_controller, flow_control, send_delay_ms
    )


def _get_or_create_rate_controller(
    conn: "ConnectionHandler", frame_duration, is_single_packet, send_delay_ms=0
):
    need_reset = False

    if not hasattr(conn, "audio_rate_controller"):
        need_reset = True
    else:
        rate_controller = conn.audio_rate_controller
        if (
            not rate_controller.pending_send_task
            or rate_controller.pending_send_task.done()
        ):
            need_reset = True
        elif (
            getattr(conn, "audio_flow_control", {}).get("sentence_id")
            != conn.sentence_id
        ):
            need_reset = True

    if need_reset:
        if not hasattr(conn, "audio_rate_controller"):
            conn.audio_rate_controller = AudioRateController(
                frame_duration, send_delay=send_delay_ms
            )
        else:
            conn.audio_rate_controller.reset()

        conn.audio_flow_control = {
            "packet_count": 0,
            "sequence": 0,
            "sentence_id": conn.sentence_id,
        }

        _start_background_sender(
            conn, conn.audio_rate_controller, conn.audio_flow_control
        )

    return conn.audio_rate_controller, conn.audio_flow_control


def _start_background_sender(conn: "ConnectionHandler", rate_controller, flow_control):
    async def send_callback(packet):
        if conn.client_abort:
            raise asyncio.CancelledError("客户端已中止")

        conn.last_activity_time = time.time() * 1000
        await _do_send_audio(conn, packet, flow_control)

    rate_controller.start_sending(send_callback)


async def _send_audio_with_rate_control(
    conn: "ConnectionHandler", audio_list, rate_controller, flow_control, send_delay_ms
):
    for packet in audio_list:
        if conn.client_abort:
            return

        conn.last_activity_time = time.time() * 1000

        if send_delay_ms > 0:
            rate_controller.add_audio(packet)
        else:
            if flow_control["packet_count"] < PRE_BUFFER_COUNT:
                await _do_send_audio(conn, packet, flow_control)
            else:
                rate_controller.add_audio(packet)


async def _do_send_audio(conn: "ConnectionHandler", opus_packet, flow_control):
    """执行实际的音频发送（精简版：移除 MQTT gateway 分支）"""
    packet_index = flow_control.get("packet_count", 0)
    sequence = flow_control.get("sequence", 0)

    # 直接发送 opus 数据包（HTTP 模式通过 FakeWebSocket）
    await conn.websocket.send(opus_packet)

    flow_control["packet_count"] = packet_index + 1
    flow_control["sequence"] = sequence + 1


async def send_tts_message(conn: "ConnectionHandler", state, text=None):
    """发送 TTS 状态消息"""
    if text is None and state == "sentence_start":
        return
    message = {"type": "tts", "state": state, "session_id": conn.session_id}
    if text is not None:
        message["text"] = textUtils.check_emoji(text)

    if state == "stop":
        current_sentence_id = conn.sentence_id
        tts_notify = conn.config.get("enable_stop_tts_notify", False)
        if tts_notify:
            stop_tts_notify_voice = conn.config.get(
                "stop_tts_notify_voice", "config/assets/tts_notify.mp3"
            )
            audios = await audio_to_data(stop_tts_notify_voice, is_opus=True)
            await sendAudio(conn, audios)
        await _wait_for_audio_completion(conn)

        if current_sentence_id != conn.sentence_id:
            return

        if hasattr(conn, "audio_rate_controller") and conn.audio_rate_controller:
            conn.audio_rate_controller.stop_sending()
        conn.clearSpeakStatus()

    await conn.websocket.send(json.dumps(message))


async def send_stt_message(conn: "ConnectionHandler", text):
    """发送 STT 状态消息"""
    end_prompt_str = conn.config.get("end_prompt", {}).get("prompt")
    if end_prompt_str and end_prompt_str == text:
        await send_tts_message(conn, "start")
        return

    display_text = text
    try:
        if text.strip().startswith("{") and text.strip().endswith("}"):
            parsed_data = json.loads(text)
            if isinstance(parsed_data, dict) and "content" in parsed_data:
                display_text = parsed_data["content"]
                if "speaker" in parsed_data:
                    conn.current_speaker = parsed_data["speaker"]
    except (json.JSONDecodeError, TypeError):
        pass
    stt_text = textUtils.get_string_no_punctuation_or_emoji(display_text)
    await conn.websocket.send(
        json.dumps({"type": "stt", "text": stt_text, "session_id": conn.session_id})
    )
    await send_tts_message(conn, "start")
    conn.client_is_speaking = True


async def send_display_message(conn: "ConnectionHandler", text):
    """发送纯显示消息"""
    message = {
        "type": "stt",
        "text": text,
        "session_id": conn.session_id
    }
    await conn.websocket.send(json.dumps(message))
