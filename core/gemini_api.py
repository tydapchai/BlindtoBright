import os
import io
import base64
import requests
import wave
import numpy as np
import itertools
from dotenv import load_dotenv

# Tự động nạp file .env ngay khi import
load_dotenv()

# Đọc key từ biến môi trường
KEYS_ENV = os.getenv("GEMINI_API_KEYS") or ""
API_KEYS = [k.strip() for k in KEYS_ENV.split(",") if k.strip()]

class GeminiClient:
    def __init__(self):
        if not API_KEYS:
            raise ValueError("Vui lòng cung cấp ít nhất 1 API Key trong file .env!")
        
        self.api_keys = API_KEYS
        self.key_pool = itertools.cycle(self.api_keys)
        self.base_url = "https://generativelanguage.googleapis.com/v1beta/models"
        
        self.stt_model = "gemini-3.1-flash-lite" 
        self.tts_model = "gemini-2.5-flash-preview-tts"

    def _get_next_key(self):
        key = next(self.key_pool)
        print(f"[Gemini API] Đang dùng Key: {key[:6]}...{key[-4:]}") 
        return key

    def transcribe_audio(self, pcm_data: bytes, sample_rate=16000) -> str:
        """Sử dụng Gemini để chuyển Audio thành Text (STT)"""
        try:
            # Đóng gói PCM thành WAV trong RAM
            wav_io = io.BytesIO()
            with wave.open(wav_io, 'wb') as wav_file:
                wav_file.setnchannels(1)
                wav_file.setsampwidth(2)
                wav_file.setframerate(sample_rate)
                wav_file.writeframes(pcm_data)
            
            b64_audio = base64.b64encode(wav_io.getvalue()).decode("utf-8")
            
            for _ in range(len(self.api_keys)):
                current_key = self._get_next_key()
                # Endpoint CHUẨN theo tài liệu: generateContent
                url = f"{self.base_url}/{self.stt_model}:generateContent?key={current_key}"
                
                payload = {
                    "contents": [{
                        "parts": [
                            {"text": "Trích xuất văn bản từ âm thanh này bằng tiếng Việt. Chỉ trả về văn bản, không thêm bất kỳ chữ nào khác."},
                            {"inline_data": {"mime_type": "audio/wav", "data": b64_audio}}
                        ]
                    }]
                }
                
                resp = requests.post(url, json=payload, timeout=15.0)
                
                if resp.status_code == 429:
                    print(f"[Gemini STT Warning] Key bị quá tải (429). Đang chuyển key...")
                    continue 
                    
                if resp.status_code != 200:
                    print(f"[Gemini STT Error] {resp.status_code}: {resp.text}")
                    return ""
                
                data = resp.json()
                
                # Trích xuất text an toàn từ JSON response
                try:
                    text = data["candidates"][0]["content"]["parts"][0]["text"].strip()
                    return text
                except (KeyError, IndexError):
                     print("[Gemini STT Error] Không tìm thấy text trong phản hồi.")
                     return ""
                
            print("[Gemini STT Error] TẤT CẢ các Key đều đang bị Rate Limit!")
            return ""
            
        except Exception as e:
            print(f"[Gemini STT Error] Lỗi hệ thống: {e}")
            return ""

    def generate_speech(self, text: str) -> bytes:
        """Sử dụng tính năng Audio Modality của Gemini để sinh giọng đọc (TTS)"""
        try:
            for _ in range(len(self.api_keys)):
                current_key = self._get_next_key()
                
                # Vẫn sử dụng generateContent cho việc sinh âm thanh
                url = f"{self.base_url}/{self.tts_model}:generateContent?key={current_key}"
                
                payload = {
                    "contents": [{
                        "parts": [{"text": f"Đọc to câu sau bằng tiếng Việt với giọng điệu tự nhiên: {text}"}]
                    }],
                    "generationConfig": {
                        "responseModalities": ["AUDIO"],
                        "speechConfig": {
                            "voiceConfig": {
                                "prebuiltVoiceConfig": {
                                    "voiceName": "Aoede"
                                }
                            }
                        }
                    }
                }
                
                resp = requests.post(url, json=payload, timeout=15.0)
                
                if resp.status_code == 429:
                    print(f"[Gemini TTS Warning] Key bị quá tải (429). Đang chuyển key...")
                    continue
                    
                if resp.status_code != 200:
                    print(f"[Gemini TTS Error] {resp.status_code}: {resp.text}")
                    return b""
                
                data = resp.json()
                
                # Bóc tách dữ liệu Audio PCM base64 từ JSON
                try:
                    parts = data["candidates"][0]["content"]["parts"]
                    audio_b64 = None
                    for part in parts:
                        if "inlineData" in part and part["inlineData"].get("mimeType", "").startswith("audio/"):
                            audio_b64 = part["inlineData"]["data"]
                            break
                            
                    if not audio_b64:
                        print("[Gemini TTS Error] Gemini không trả về dữ liệu âm thanh.")
                        return b""
                        
                    pcm_bytes = base64.b64decode(audio_b64)
                    
                    # Chuyển đổi tần số lấy mẫu từ 24kHz của Gemini xuống 16kHz của ESP32
                    audio_int16 = np.frombuffer(pcm_bytes, dtype=np.int16)
                    orig_indices = np.arange(len(audio_int16))
                    target_length = int(len(audio_int16) * (16000 / 24000))
                    new_indices = np.linspace(0, len(audio_int16) - 1, target_length)
                    
                    resampled = np.interp(new_indices, orig_indices, audio_int16).astype(np.int16)
                    
                    # Chuyển thành Stereo (L+R)
                    stereo_array = np.column_stack((resampled, resampled)).flatten()
                    return stereo_array.tobytes()

                except (KeyError, IndexError):
                     print("[Gemini TTS Error] Cấu trúc phản hồi JSON không đúng.")
                     return b""
                
            print("[Gemini TTS Error] TẤT CẢ các Key đều đang bị Rate Limit!")
            return b""
            
        except Exception as e:
            print(f"[Gemini TTS Error] Lỗi hệ thống: {e}")
            return b""