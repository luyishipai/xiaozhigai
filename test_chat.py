import requests
import sounddevice as sd
import numpy as np
import wave
import os

BASE = "http://127.0.0.1:8003"

# 1. 录音
print("🎙️ 即将录音，说 5 秒钟话（比如 '今天天气怎么样'）...")
fs = 16000
duration = 5
recording = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='int16')
sd.wait()
print("✅ 录音结束")

# 保存 wav
wav_path = "chat_input.wav"
with wave.open(wav_path, "w") as wf:
    wf.setnchannels(1)
    wf.setsampwidth(2)
    wf.setframerate(fs)
    wf.writeframes(recording.tobytes())
print(f"📁 已保存: {wav_path}")

# 2. 调 /api/chat
print("\n💬 调用 /api/chat ...")
try:
    with open(wav_path, "rb") as f:
        resp = requests.post(f"{BASE}/api/chat", files={"audio": f}, timeout=60)
    
    print(f"  HTTP 状态码: {resp.status_code}")
    if resp.status_code == 200:
        out = "chat_reply.mp3"
        with open(out, "wb") as f:
            f.write(resp.content)
        print(f"  ✅ 成功！回复已保存到: {os.path.abspath(out)}")
        print(f"  📦 大小: {len(resp.content)} 字节")
        print("\n👉 双击 chat_reply.mp3 就能听到小智的语音回复！")
    else:
        print(f"  ❌ 失败: {resp.text}")
except Exception as e:
    print(f"  ❌ 连接错误: {e}")