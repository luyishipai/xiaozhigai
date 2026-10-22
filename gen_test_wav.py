import wave
import struct

# 生成一段 1 秒的 16k 采样率空 wav（静音）
# 注意：空的识别不出东西，但能测接口通不通
sample_rate = 16000
duration = 3  # 3 秒
n_samples = sample_rate * duration

with wave.open("test.wav", "w") as f:
    f.setnchannels(1)       # 单声道
    f.setsampwidth(2)       # 16bit
    f.setframerate(16000)   # 16k 采样率
    # 生成静音数据
    data = struct.pack("<" + "h" * n_samples, *([0] * n_samples))
    f.writeframes(data)

print("✅ 生成 test.wav（3秒静音）")