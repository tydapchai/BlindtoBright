import os
import io
import base64
import requests
import wave
import numpy as np
import itertools

# Điền 3 API key của bạn vào đây (ngăn cách bằng dấu phẩy)
# Hoặc truyền qua biến môi trường: export GEMINI_API_KEYS="key1,key2,key3"
KEYS_ENV = os.getenv("GEMINI_API_KEYS")

# Tách chuỗi thành danh sách các key hợp lệ
API_KEYS = [k.strip() for k in KEYS_ENV.split(",") if k.strip()]

class GeminiClient:
    def __init__(self):
        if not API_KEYS:
            raise ValueError("Vui lòng cung cấp ít nhất 1 API Key!")
        
        self.api_keys = API_KEYS
        # Khởi tạo bộ xoay vòng vô tận qua danh sách key
        self.key_pool = itertools.cycle(self.api_keys)
        self.base_url = "https://generativelanguage.googleapis.com/v1beta/models"

    def _get_next_key(self):
        """Lấy key tiếp theo trong vòng xoay và in log để dễ debug"""
        key = next(self.key_pool)
        # Ẩn bớt key khi in ra console để bảo mật
        print(f"[Gemini API] Đang sử dụng Key: {key[:6]}...{key[-4:]}") 
        return key

    def transcribe_audio(self, pcm_data: bytes, sample_rate=16000) -> str:
        """Sử dụng gemini để chuyển PCM 16-bit thành Text với cơ chế xoay vòng Key"""
        try:
            # Đóng gói PCM thành file WAV trong bộ nhớ
            wav_io = io.BytesIO()
            with wave.open(wav_io, 'wb') as wav_file:
                wav_file.setnchannels(1)
                wav_file.setsampwidth(2) # 16-bit
                wav_file.setframerate(sample_rate)
                wav_file.writeframes(pcm_data)
            
            b64_audio = base64.b64encode(wav_io.getvalue()).decode("utf-8")
            
            # Thử tối đa số lần bằng đúng số lượng key đang có
            for _ in range(len(self.api_keys)):
                current_key = self._get_next_key()
                url = f"{self.base_url}/gemini-1.5-pro:generateContent?key={current_key}"
                payload = {
                    "contents": [{
                        "parts": [
                            {"text": "Transcribe this audio precisely in Vietnamese. Only return the transcription, no other text."},
                            {"inline_data": {"mime_type": "audio/wav", "data": b64_audio}}
                        ]
                    }]
                }
                
                resp = requests.post(url, json=payload, timeout=15.0)
                
                # Nếu bị Rate Limit (hết quota phút), bỏ qua và thử key tiếp theo ngay lập tức
                if resp.status_code == 429:
                    print(f"[Gemini STT Warning] Key {current_key[:6]}... bị Rate Limit (429). Chuyển key khác...")
                    continue 
                    
                resp.raise_for_status()
                
                # Parse kết quả thành công
                data = resp.json()
                text = data["candidates"][0]["content"]["parts"][0]["text"].strip()
                return text
                
            print("[Gemini STT Error] TẤT CẢ các Key đều đang bị Rate Limit!")
            return ""
            
        except Exception as e:
            print(f"[Gemini STT Error] {e}")
            return ""

    def generate_speech(self, text: str) -> bytes:
        """Sử dụng gemini-tts để chuyển Text thành PCM audio bytes với cơ chế xoay vòng Key"""
        try:
            # Thử tối đa số lần bằng đúng số lượng key đang có
            for _ in range(len(self.api_keys)):
                current_key = self._get_next_key()
                url = f"{self.base_url}/gemini-2.5-flash-preview-tts:predict?key={current_key}"
                payload = {
                    "instances": [{"text": text, "language": "vi-VN"}],
                    "parameters": {"voice_name": "vi-VN-Standard-A", "audio_encoding": "LINEAR16"}
                }
                
                resp = requests.post(url, json=payload, timeout=10.0)
                
                # Tự động nhảy sang key khác nếu gặp lỗi giới hạn 429
                if resp.status_code == 429:
                    print(f"[Gemini TTS Warning] Key {current_key[:6]}... bị Rate Limit (429). Chuyển key khác...")
                    continue
                    
                resp.raise_for_status()
                
                data = resp.json()
                audio_b64 = data["predictions"][0]["audio_content"]
                pcm_bytes = base64.b64decode(audio_b64)
                
                # Cắt header WAV (RIFF) nếu model trả về format WAV thay vì RAW PCM
                if pcm_bytes.startswith(b'RIFF'):
                    pcm_bytes = pcm_bytes[44:]
                    
                mono_array = np.frombuffer(pcm_bytes, dtype=np.int16)
                stereo_array = np.column_stack((mono_array, mono_array)).flatten()
                return stereo_array.tobytes()
                
            print("[Gemini TTS Error] TẤT CẢ các Key đều đang bị Rate Limit!")
            return b""
            
        except Exception as e:
            print(f"[Gemini TTS Error] {e}")
            return b""