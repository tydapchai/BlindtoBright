import os
import io
import wave
import time
import subprocess
from gemini_api import GeminiClient

def record_audio(duration: int = 4, sample_rate: int = 16000) -> bytes:
    """Thu âm từ micro của laptop/PC bằng lệnh hệ thống hoặc thư viện sounddevice"""
    try:
        import sounddevice as sd
        import numpy as np
        print(f"\n[STT TEST] Bắt đầu thu âm trong {duration} giây... HÃY NÓI ĐI!")
        # Thu âm mảng float32
        recording = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype='float32')
        sd.wait() # Chờ thu xong
        print("[STT TEST] Đã thu âm xong. Đang xử lý...")
        
        # Chuyển đổi float32 sang int16 PCM
        audio_int16 = (recording * 32767).astype(np.int16)
        return audio_int16.tobytes()
    except ImportError:
        print("[Lỗi] Bạn cần cài thư viện để thu âm: pip install sounddevice numpy")
        return b""

def play_audio(pcm_data: bytes, sample_rate: int = 16000):
    """Phát file PCM bằng sounddevice"""
    try:
        import sounddevice as sd
        import numpy as np
        
        # gemini_api.py của bạn đang trả về mảng Stereo (nhân đôi kênh mono). 
        # Chúng ta chia nó ra để lấy số mẫu chuẩn.
        audio_int16 = np.frombuffer(pcm_data, dtype=np.int16)
        
        # Chuyển mảng 1D (interleaved L-R L-R) thành ma trận 2D để sounddevice hiểu là 2 kênh
        stereo_audio = audio_int16.reshape(-1, 2)
        
        print("\n[TTS TEST] Đang phát âm thanh ra loa laptop...")
        sd.play(stereo_audio, samplerate=sample_rate)
        sd.wait()
        print("[TTS TEST] Phát xong!")
    except ImportError:
        print("[Lỗi] Cần cài thư viện sounddevice: pip install sounddevice numpy")

def main():
    print("=======================================")
    print("   TEST GEMINI STT & TTS (STANDALONE)  ")
    print("=======================================")
    
    # Khởi tạo Client (Sẽ tự động đọc API Key từ biến môi trường hoặc script gemini_api.py)
    gemini = GeminiClient()
    
    while True:
        print("\nChọn chức năng muốn test:")
        print("1. Test STT (Nói vào mic -> In ra chữ)")
        print("2. Test TTS (Gõ chữ -> Loa đọc lên)")
        print("3. Thoát")
        
        choice = input("Nhập số (1/2/3): ").strip()
        
        if choice == '1':
            # 1. Thu âm
            pcm_bytes = record_audio(duration=5) 
            if not pcm_bytes: continue
            
            # 2. Gửi cho Gemini dịch
            start_time = time.time()
            text = gemini.transcribe_audio(pcm_bytes)
            elapsed = time.time() - start_time
            
            print("---------------------------------------")
            print(f"🎯 KẾT QUẢ STT (mất {elapsed:.2f}s):")
            print(f"👉 {text}")
            print("---------------------------------------")
            
        elif choice == '2':
            text = input("\n📝 Nhập câu tiếng Việt muốn đọc: ").strip()
            if not text: continue
            
            # 1. Gửi cho Gemini lấy file âm thanh
            start_time = time.time()
            pcm_audio = gemini.generate_speech(text)
            elapsed = time.time() - start_time
            
            if pcm_audio:
                print(f"Đã nhận phản hồi TTS từ Gemini (mất {elapsed:.2f}s, dung lượng: {len(pcm_audio)} bytes)")
                # 2. Phát loa
                play_audio(pcm_audio)
            else:
                print("❌ Lỗi: Không nhận được âm thanh từ Gemini TTS.")
                
        elif choice == '3':
            print("Thoát.")
            break
        else:
            print("Lựa chọn không hợp lệ.")

if __name__ == "__main__":
    main()