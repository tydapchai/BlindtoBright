import urllib.request
import wave
import time
import os
import sys

# Địa chỉ IP của ESP32 (mặc định là IP SoftAP hoặc bạn truyền qua dòng lệnh)
DEFAULT_IP = "192.168.4.1"

def record_and_play(esp_ip=DEFAULT_IP, duration=5, output_file="record_5s.wav"):
    url = f"http://{esp_ip}/mic"
    sample_rate = 16000
    bytes_per_sample = 2  # 16-bit Mono PCM
    total_bytes = sample_rate * bytes_per_sample * duration

    print("\n==================================================")
    print("      THU ÂM VÀ NGHE LẠI 5 GIÂY TỪ MICRO ESP32    ")
    print("==================================================")
    print(f"[-] Đang kết nối tới Micro ESP32 tại: {url}")

    try:
        req = urllib.request.urlopen(url, timeout=6.0)
    except Exception as e:
        print(f"\n[X] Không thể kết nối tới {url}: {e}")
        print("    -> Đảm bảo ESP32 đã nạp code có hỗ trợ mic và laptop đã cùng mạng.")
        return

    print(f"\n[+] BẮT ĐẦU THU ÂM {duration} GIÂY... HÃY NÓI VÀO MIC!")
    raw_data = bytearray()
    start_time = time.time()

    while len(raw_data) < total_bytes:
        chunk = req.read(512)
        if not chunk:
            break
        raw_data.extend(chunk)
        elapsed = time.time() - start_time
        percent = min(100, len(raw_data) * 100 // total_bytes)
        print(f"    [Đang thu]: {percent}% | {elapsed:.1f}s / {duration}s", end="\r")

    req.close()
    print(f"\n[+] Đã thu xong {duration} giây ({len(raw_data)} bytes)!")

    # Lưu thành file chuẩn WAV 16kHz 16-bit Mono
    with wave.open(output_file, "wb") as wf:
        wf.setnchannels(1)      # Mono
        wf.setsampwidth(2)      # 16-bit
        wf.setframerate(16000)  # 16kHz
        wf.writeframes(raw_data[:total_bytes])

    abs_path = os.path.abspath(output_file)
    print(f"[SUCCESS] Đã lưu file tại: {abs_path}")
    print("[-] Đang tự động mở file để loa Laptop phát lại âm thanh...")

    # Mở file bằng trình phát nhạc mặc định của Windows
    try:
        os.system(f'start "" "{abs_path}"')
    except Exception as e:
        print(f"Vui lòng mở file {abs_path} để nghe.")

if __name__ == "__main__":
    target_ip = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_IP
    record_and_play(target_ip)
