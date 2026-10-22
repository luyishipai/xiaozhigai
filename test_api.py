import requests
import glob
import os

BASE = "http://127.0.0.1:8003"

print("=" * 50)
print("  小智语音接口测试")
print("=" * 50)

# ===== 测试 TTS =====
print("\n🔊 测试 TTS 接口 (文字→语音)...")
try:
    resp = requests.post(
        f"{BASE}/api/tts",
        json={"text": "你好小智，这是测试语音"},
        timeout=15
    )
    print(f"  HTTP 状态码: {resp.status_code}")
    if resp.status_code == 200:
        with open("test_output.mp3", "wb") as f:
            f.write(resp.content)
        print(f"  ✅ TTS 成功！")
        print(f"  📁 生成文件: {os.path.abspath('test_output.mp3')}")
        print(f"  📦 文件大小: {len(resp.content)} 字节")
    else:
        print(f"  ❌ TTS 失败: {resp.text}")
except Exception as e:
    print(f"  ❌ 连接失败: {e}")

# ===== 测试 ASR =====
print("\n🎤 测试 ASR 接口 (语音→文字)...")
wav_files = glob.glob("*.wav") + glob.glob("tmp/*.wav")
if wav_files:
    wav_path = wav_files[0]
    print(f"  用音频文件: {wav_path}")
    try:
        with open(wav_path, "rb") as f:
            resp = requests.post(
                f"{BASE}/api/asr",
                files={"audio": f},
                timeout=30
            )
        print(f"  HTTP 状态码: {resp.status_code}")
        if resp.status_code == 200:
            data = resp.json()
            print(f"  ✅ ASR 成功！识别结果: \"{data.get('text', '未知')}\"")
        else:
            print(f"  ❌ ASR 失败: {resp.text}")
    except Exception as e:
        print(f"  ❌ 连接失败: {e}")
else:
    print("  ⚠️ 没找到 wav 文件，跳过 ASR 测试")
    print("     用 Windows 录音机录一段，保存为 test.wav 放在当前目录")

print("\n" + "=" * 50)
print("  测试结束")
print("=" * 50)