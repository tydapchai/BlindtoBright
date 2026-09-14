import os
import io
import base64
import requests
import wave
import numpy as np
import itertools
import hashlib
from pathlib import Path

def load_env_file():
    """Tự động đọc file .env ở thư mục configs/, core/ hoặc thư mục gốc dự án."""
    current_dir = Path(__file__).resolve().parent
    root_dir = current_dir.parent
    for env_path in [root_dir / "configs" / ".env", current_dir / ".env", root_dir / ".env"]:
        if env_path.is_file():
            try:
                with open(env_path, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if line and not line.startswith("#") and "=" in line:
                            k, v = line.split("=", 1)
                            k = k.strip()
                            v = v.strip().strip("'\"")
                            if k not in os.environ:
                                os.environ[k] = v
            except Exception:
                pass

load_env_file()

# Nhận cả GEMINI_API_KEYS (nhiều key ngăn cách bởi dấu phẩy) hoặc GEMINI_API_KEY (1 key)
KEYS_ENV = os.getenv("GEMINI_API_KEYS") or os.getenv("GEMINI_API_KEY") or ""

# Đọc key từ biến môi trường
KEYS_ENV = os.getenv("GEMINI_API_KEYS") or ""
API_KEYS = [k.strip() for k in KEYS_ENV.split(",") if k.strip()]

class GeminiClient:
    def __init__(self, api_keys=None):
        if api_keys:
            if isinstance(api_keys, str):
                self.api_keys = [k.strip() for k in api_keys.split(",") if k.strip()]
            else:
                self.api_keys = list(api_keys)
        else:
            self.api_keys = API_KEYS

        if not self.api_keys:
            print("[Gemini API Warning] Chưa có GEMINI_API_KEYS. Hãy set biến môi trường GEMINI_API_KEYS hoặc truyền --gemini-key!")
            self.key_pool = None
        else:
            self.key_pool = itertools.cycle(self.api_keys)

        self.base_url = "https://generativelanguage.googleapis.com/v1beta/models"
        
        self.stt_model = "gemini-3.1-flash-lite" 
        self.tts_model = "gemini-3.1-flash-tts-preview"

        # Khởi tạo thư mục Cache âm thanh và bộ nhớ RAM cache
        self.cache_dir = Path(__file__).resolve().parent.parent / "cache" / "audio"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._mem_cache = {}

    def _get_next_key(self):
        """Lấy key tiếp theo trong vòng xoay và in log để dễ debug"""
        if not self.key_pool:
            return None
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
        """Sử dụng tính năng Audio Modality của Gemini để sinh giọng đọc (TTS) với Cache siêu tốc (< 1ms)."""
        clean_text = text.strip() if text else ""
        if not clean_text:
            return b""

        # 1. Kiểm tra RAM cache (< 0.0001s)
        if clean_text in self._mem_cache:
            return self._mem_cache[clean_text]

        # 2. Kiểm tra Disk cache (< 0.002s)
        cache_key = hashlib.md5(clean_text.encode("utf-8")).hexdigest()
        cache_file = self.cache_dir / f"{cache_key}.pcm"
        if cache_file.is_file():
            try:
                pcm_data = cache_file.read_bytes()
                if len(pcm_data) > 0:
                    self._mem_cache[clean_text] = pcm_data
                    return pcm_data
            except Exception:
                pass

        # 3. Chưa có trong Cache -> Gọi Cloud API của Gemini
        try:
            for _ in range(len(self.api_keys)):
                current_key = self._get_next_key()
                
                # Vẫn sử dụng generateContent cho việc sinh âm thanh
                url = f"{self.base_url}/{self.tts_model}:generateContent?key={current_key}"
                
                payload = {
                    "contents": [{
                        "parts": [{"text": f"Đọc to câu sau bằng tiếng Việt với giọng điệu tự nhiên: {clean_text}"}]
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
                    pcm_result = stereo_array.tobytes()

                    # Lưu vào RAM và Disk cache cho mọi lần sau phát tức thì
                    self._mem_cache[clean_text] = pcm_result
                    try:
                        cache_file.write_bytes(pcm_result)
                        meta_file = self.cache_dir / f"{cache_key}.txt"
                        meta_file.write_text(clean_text, encoding="utf-8")
                    except Exception:
                        pass

                    return pcm_result

                except (KeyError, IndexError):
                     print("[Gemini TTS Error] Cấu trúc phản hồi JSON không đúng.")
                     return b""
                
            print("[Gemini TTS Error] TẤT CẢ các Key đều đang bị Rate Limit!")
            return b""
            
        except Exception as e:
            print(f"[Gemini TTS Error] Lỗi hệ thống: {e}")
            return b""

    def preload_cache(self, texts: list):
        """Khởi động và nạp sẵn cache cho danh sách câu để phát ra loa tức thì (0ms latency)."""
        loaded = 0
        missing = []
        for text in texts:
            clean = text.strip() if text else ""
            if not clean:
                continue
            cache_key = hashlib.md5(clean.encode("utf-8")).hexdigest()
            cache_file = self.cache_dir / f"{cache_key}.pcm"
            if cache_file.is_file():
                try:
                    self._mem_cache[clean] = cache_file.read_bytes()
                    loaded += 1
                except Exception:
                    missing.append(clean)
            else:
                missing.append(clean)

        if loaded > 0:
            print(f"[Audio Cache] Đã nạp sẵn {loaded} câu âm thanh từ cache ổ đĩa vào RAM.")

        if missing:
            print(f"[Audio Cache] Còn {len(missing)} câu chưa cache, đang sinh trước qua Gemini...")
            for t in missing:
                self.generate_speech(t)