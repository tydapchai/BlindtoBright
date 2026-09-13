from __future__ import annotations

import argparse
import logging
import os
import signal
import json
import queue
import threading
import time
from collections import deque
from pathlib import Path
import tempfile

# Bypass any system / environment proxies for direct local LAN communication
os.environ["NO_PROXY"] = "*"
os.environ["no_proxy"] = "*"
for _k in ["HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy", "ALL_PROXY", "all_proxy"]:
    os.environ.pop(_k, None)

import requests
http_session = requests.Session()
http_session.trust_env = False
http_session.proxies = {"http": None, "https": None}

os.environ.setdefault(
    "OPENCV_FFMPEG_CAPTURE_OPTIONS",
    "fflags;nobuffer|flags;low_delay|analyzeduration;0|probesize;32"
)

import cv2
import numpy as np
import torch
import torch.nn as nn
from flask import Flask, jsonify, request, render_template_string
from faster_whisper import WhisperModel
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from PIL import Image, ImageDraw, ImageFont

try:
    vn_font_large = ImageFont.truetype("arial.ttf", 20)
    vn_font_title = ImageFont.truetype("arial.ttf", 14)
except Exception:
    try:
        vn_font_large = ImageFont.truetype("segoeui.ttf", 20)
        vn_font_title = ImageFont.truetype("segoeui.ttf", 14)
    except Exception:
        vn_font_large = ImageFont.load_default()
        vn_font_title = vn_font_large

def draw_vn_banner(frame, title: str, text: str, font_title=vn_font_title, font_body=vn_font_large):
    h, w = frame.shape[:2]
    box_h = 76
    box_y = max(0, h - box_h - 10)
    box_x1 = 15
    box_x2 = w - 15

    overlay = frame.copy()
    cv2.rectangle(overlay, (box_x1, box_y), (box_x2, h - 10), (15, 15, 15), -1)
    cv2.rectangle(overlay, (box_x1, box_y), (box_x2, h - 10), (0, 215, 255), 2)
    cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)

    box_crop = frame[box_y:h-10, box_x1:box_x2]
    crop_rgb = cv2.cvtColor(box_crop, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(crop_rgb)
    draw = ImageDraw.Draw(pil_img)

    if len(text) > 38:
        words = text.split()
        line1, line2 = [], []
        curr = line1
        for w_word in words:
            if curr is line1 and len(" ".join(line1 + [w_word])) > 38:
                curr = line2
            curr.append(w_word)
        l1_str = " ".join(line1)
        l2_str = " ".join(line2)
        if len(l2_str) > 42:
            l2_str = l2_str[:39] + "..."
        draw.text((12, 4), title, font=font_title, fill=(0, 240, 255))
        draw.text((12, 26), l1_str, font=font_body, fill=(255, 255, 255))
        draw.text((12, 48), l2_str, font=font_body, fill=(255, 255, 255))
    else:
        draw.text((12, 6), title, font=font_title, fill=(0, 240, 255))
        draw.text((12, 32), text, font=font_body, fill=(255, 255, 255))

    res_bgr = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    frame[box_y:h-10, box_x1:box_x2] = res_bgr
    return frame

HAND_FEATURES = 63
RAW_FEATURES = 126

HAND_CONNECTIONS = [
    # Thumb
    (0, 1), (1, 2), (2, 3), (3, 4),
    # Index finger
    (0, 5), (5, 6), (6, 7), (7, 8),
    # Middle finger
    (9, 10), (10, 11), (11, 12),
    # Ring finger
    (13, 14), (14, 15), (15, 16),
    # Pinky
    (0, 17), (17, 18), (18, 19), (19, 20),
    # Knuckle and palm connections
    (5, 9), (9, 13), (13, 17), (2, 5)
]

HTML = """
<!doctype html>
<html lang="vi">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Blind to Bright - Trợ lý Giao tiếp</title>
<style>
body{font-family:'Segoe UI',Roboto,sans-serif;max-width:800px;margin:auto;padding:18px;background:#121212;color:#f0f0f0}
h1{color:#4da6ff;text-align:center;margin-bottom:20px}
.card{background:#1e1e1e;padding:18px;border-radius:12px;margin-bottom:16px;box-shadow:0 4px 10px rgba(0,0,0,0.3)}
button{background:#007bff;color:#fff;padding:10px 18px;border:0;border-radius:8px;margin:4px;font-size:15px;cursor:pointer;font-weight:600}
button:hover{background:#0056b3}
textarea{width:100%;min-height:80px;border-radius:8px;padding:10px;background:#2a2a2a;color:#fff;border:1px solid #444;box-sizing:border-box}
.item{padding:12px;border-bottom:1px solid #333;display:flex;justify-content:space-between;align-items:center}
.item-text{font-size:16px}
.tag-gesture{background:#9c27b0;color:#fff;padding:4px 8px;border-radius:6px;font-size:13px;font-weight:bold;margin-right:8px}
.tag-speech{background:#00897b;color:#fff;padding:4px 8px;border-radius:6px;font-size:13px;font-weight:bold;margin-right:8px}
small{color:#888;font-size:12px}
</style>
</head>
<body>
<h1>🌟 Blind to Bright - Trợ lý Giao tiếp</h1>
<div class="card">
<h2>🎙️ Ghi âm & Nhận dạng giọng nói</h2>
<input id="audio" type="file" accept="audio/*" capture>
<button onclick="uploadAudio()">Chuyển thành chữ</button>
<p id="status" style="color:#00e676;font-weight:bold"></p>
</div>
<div class="card">
<h2>📝 Tóm tắt hội thoại</h2>
<button onclick="loadSummary()">Tạo tóm tắt</button>
<textarea id="summary" readonly></textarea>
</div>
<div class="card">
<h2>💬 Nhật ký hội thoại (Thời gian thực)</h2>
<div id="history"></div>
</div>
<script>
async function refresh(){
  try{
    const r=await fetch('/api/history');
    const data=await r.json();
    document.getElementById('history').innerHTML=data.slice().reverse().map(x=>{
      const isGesture = x.source === 'gesture';
      const badge = isGesture 
        ? '<span class="tag-gesture">🤟 KÝ HIỆU</span>' 
        : '<span class="tag-speech">🎙️ GIỌNG NÓI</span>';
      return `<div class="item">
        <div class="item-text">${badge} <b>${x.text}</b></div>
        <small>${x.time || ''}</small>
      </div>`;
    }).join('');
  }catch(e){}
}
async function uploadAudio(){
  const f=document.getElementById('audio').files[0];
  if(!f){alert('Chọn hoặc ghi âm trước');return;}
  const fd=new FormData(); fd.append('audio',f);
  document.getElementById('status').innerText='Đang xử lý...';
  const r=await fetch('/api/transcribe',{method:'POST',body:fd});
  const data=await r.json();
  document.getElementById('status').innerText=data.text || data.error;
  refresh();
}
async function loadSummary(){
  const r=await fetch('/api/summary');
  const data=await r.json();
  document.getElementById('summary').value=data.summary;
}
refresh(); setInterval(refresh,2000);
</script>
</body>
</html>
"""

class GestureBiGRU(nn.Module):
    def __init__(self,input_size,num_classes,hidden_size=128,num_layers=2,dropout=0.4,bidirectional=True):
        super().__init__()
        self.input_projection=nn.Sequential(
            nn.Linear(input_size,hidden_size),
            nn.LayerNorm(hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.gru=nn.GRU(
            hidden_size,hidden_size,num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers>1 else 0,
            bidirectional=bidirectional,
        )
        output_size=hidden_size*2 if bidirectional else hidden_size
        self.attention=nn.Sequential(
            nn.Linear(output_size,hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size,1),
        )
        self.classifier=nn.Sequential(
            nn.LayerNorm(output_size),
            nn.Dropout(dropout),
            nn.Linear(output_size,num_classes),
        )

    def forward(self,x):
        x=self.input_projection(x)
        seq,_=self.gru(x)
        weights=torch.softmax(self.attention(seq).squeeze(-1),dim=1).unsqueeze(-1)
        return self.classifier(torch.sum(seq*weights,dim=1))

class State:
    def __init__(self):
        self.history=[]
        self.lock=threading.Lock()

    def add(self,source,text,extra=None):
        item={
            "source":source,
            "text":text,
            "time":time.strftime("%H:%M:%S"),
        }
        if extra:
            item.update(extra)
        with self.lock:
            self.history.append(item)
            self.history=self.history[-100:]

    def get(self):
        with self.lock:
            return list(self.history)

state=State()
web=Flask(__name__)
whisper_model=None
whisper_lock=threading.Lock()
shutdown_event=threading.Event()

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s | %(levelname)s | %(threadName)s | %(message)s",
)
logger=logging.getLogger("communication-assistant")

@web.get("/")
def home():
    return render_template_string(HTML)

@web.get("/api/history")
def history():
    return jsonify(state.get())

@web.post("/api/transcribe")
def transcribe():
    global whisper_model
    if "audio" not in request.files:
        return jsonify({"error":"Missing audio"}),400

    upload=request.files["audio"]
    temp=Path("temp_audio")
    temp.mkdir(exist_ok=True)
    path=temp/f"{time.time_ns()}_{upload.filename or 'audio.webm'}"
    upload.save(path)

    try:
        with whisper_lock:
            if whisper_model is None:
                whisper_model=WhisperModel(
                    os.getenv("WHISPER_MODEL", "small"),
                    device=os.getenv("WHISPER_DEVICE", "cpu"),
                    compute_type=os.getenv("WHISPER_COMPUTE_TYPE", "int8"),
                )

            segments,info=whisper_model.transcribe(
                str(path),
                vad_filter=True,
                beam_size=int(os.getenv("WHISPER_BEAM_SIZE", "3")),
            )
            text=" ".join(s.text.strip() for s in segments).strip()
    except Exception:
        logger.exception("Audio transcription failed")
        return jsonify({"error":"Không thể xử lý âm thanh."}),500
    finally:
        path.unlink(missing_ok=True)

    if not text:
        text="Không nhận được nội dung rõ ràng."

    state.add("speech",text,{"language":info.language})
    return jsonify({"text":text,"language":info.language})

@web.get("/api/summary")
def summary():
    items=state.get()
    texts=[x["text"] for x in items if x["text"]]

    if not texts:
        return jsonify({"summary":"Chưa có nội dung hội thoại."})

    # Lightweight extractive summary for hackathon:
    # keeps the latest distinct messages instead of hallucinating.
    distinct=[]
    for text in texts:
        if not distinct or text.lower()!=distinct[-1].lower():
            distinct.append(text)

    selected=distinct[-6:]
    result=" • ".join(selected)
    return jsonify({"summary":result})

def run_web(host,port):
    try:
        from waitress import serve
        logger.info("Web server listening on http://%s:%s", host, port)
        serve(web, host=host, port=port, threads=4)
    except ImportError:
        logger.warning("waitress is not installed; using Flask development server")
        web.run(
            host=host,
            port=port,
            debug=False,
            use_reloader=False,
            threaded=True,
        )

def normalize_hand(points):
    points=points.astype(np.float32).copy()
    points-=points[0]
    scale=np.max(np.linalg.norm(points,axis=1))
    if scale>1e-6:
        points/=scale
    return points

def frame_features(result, force_right_hand=True):
    output=np.zeros(RAW_FEATURES,dtype=np.float32)
    if not result.hand_landmarks:
        return output

    # If force_right_hand is enabled and only one hand is detected,
    # we treat it as the right hand (the dominant hand in WLASL dataset).
    if force_right_hand and len(result.hand_landmarks) == 1:
        points=np.array([[p.x,p.y,p.z] for p in result.hand_landmarks[0]],dtype=np.float32)
        values=normalize_hand(points).reshape(-1)
        output[HAND_FEATURES:2*HAND_FEATURES]=values
        return output

    used=set()
    for i,landmarks in enumerate(result.hand_landmarks):
        points=np.array([[p.x,p.y,p.z] for p in landmarks],dtype=np.float32)
        values=normalize_hand(points).reshape(-1)

        hand=None
        if result.handedness and i<len(result.handedness) and result.handedness[i]:
            hand=result.handedness[i][0].category_name.lower()

        start=0 if hand=="left" else HAND_FEATURES
        if hand not in {"left","right"}:
            start=0 if i==0 else HAND_FEATURES
        if start in used:
            start=HAND_FEATURES if start==0 else 0

        output[start:start+HAND_FEATURES]=values
        used.add(start)

    return output

def resample(sequence,target):
    sequence=np.asarray(sequence,dtype=np.float32)
    if len(sequence)==1:
        return np.repeat(sequence,target,axis=0)

    old=np.linspace(0,1,len(sequence))
    new=np.linspace(0,1,target)
    result=np.empty((target,sequence.shape[1]),dtype=np.float32)

    for j in range(sequence.shape[1]):
        result[:,j]=np.interp(new,old,sequence[:,j])

    return result

def fill_missing_frames(sequence):
    if not sequence:
        return sequence
    filled = [arr.copy() for arr in sequence]
    n = len(filled)
    
    # Forward fill
    last_valid = None
    for i in range(n):
        if np.any(filled[i] != 0):
            last_valid = filled[i]
        elif last_valid is not None:
            filled[i] = last_valid.copy()
            
    # Backward fill
    last_valid = None
    for i in range(n - 1, -1, -1):
        if np.any(filled[i] != 0):
            last_valid = filled[i]
        elif last_valid is not None:
            filled[i] = last_valid.copy()
            
    return filled

def prepare(sequence,length,mean,std):
    filled_sequence = fill_missing_frames(sequence)
    sequence=resample(filled_sequence,length)
    velocity=np.diff(sequence,axis=0,prepend=sequence[:1])
    sequence=np.concatenate([sequence,velocity],axis=1)
    return ((sequence-mean)/std).astype(np.float32)

def convert_wav_to_16bit_stereo(wav_path):
    import wave
    import os

    if not os.path.exists(wav_path):
        return None

    try:
        with wave.open(wav_path, "rb") as w:
            nchannels = w.getnchannels()
            sampwidth = w.getsampwidth()
            framerate = w.getframerate()
            nframes = w.getnframes()
            frames = w.readframes(nframes)

        if not frames:
            return None

        # Chuyển đổi dữ liệu mẫu sang mảng numpy float32
        if sampwidth == 2:
            data = np.frombuffer(frames, dtype=np.int16).astype(np.float32)
        elif sampwidth == 1:
            data = (np.frombuffer(frames, dtype=np.uint8).astype(np.float32) - 128.0) * 256.0
        else:
            return None

        # Gộp thành mono nếu đầu vào là stereo
        if nchannels == 2:
            data = data.reshape(-1, 2).mean(axis=1)

        # Resample sang 16000Hz nếu cần
        target_rate = 16000
        if framerate != target_rate and len(data) > 1:
            duration = len(data) / framerate
            target_samples = int(duration * target_rate)
            orig_indices = np.linspace(0, len(data) - 1, len(data))
            new_indices = np.linspace(0, len(data) - 1, target_samples)
            data = np.interp(new_indices, orig_indices, data)

        # Cân bằng âm lượng 85% để tránh méo tiếng khi khuếch đại qua MAX98357A
        data = np.clip(data * 0.85, -32768, 32767).astype(np.int16)
        # Nhân đôi kênh thành Stereo (L + R) cho MAX98357A I2S
        stereo_data = np.column_stack((data, data)).flatten()
        return stereo_data.tobytes()
    except Exception as e:
        logger.warning("Error converting WAV to 16-bit stereo PCM: %s", e)
        return None


def start_tts_worker(camera_source, language="vi", esp_ip=None):
    items = queue.Queue()

    # Parse ESP32 IP from camera_source if not provided directly
    if not esp_ip and isinstance(camera_source, str) and camera_source.startswith("http"):
        from urllib.parse import urlparse
        try:
            parsed = urlparse(camera_source)
            esp_ip = parsed.hostname
        except Exception:
            pass

    def worker():
        logger.info("TTS worker starting (target: ESP32 External Speaker ONLY, esp_ip: %s)", esp_ip)

        try:
            import pyttsx3
            import requests
            import os

            engine = pyttsx3.init(driverName="sapi5")
            engine.setProperty("rate", 160)
            engine.setProperty("volume", 1.0)

            # Configure Vietnamese voice if language is 'vi'
            if language == "vi":
                vn_voice_id = r"HKEY_LOCAL_MACHINE\SOFTWARE\Microsoft\Speech_OneCore\Voices\Tokens\MSTTS_V110_viVN_An"
                try:
                    engine.setProperty("voice", vn_voice_id)
                    logger.info("Configured Vietnamese TTS voice: Microsoft An (OneCore)")
                except Exception as e:
                    logger.warning("Could not set OneCore Vietnamese voice: %s", e)

            voices = engine.getProperty("voices")
            logger.info("TTS initialized with %d voices", len(voices))

        except Exception:
            logger.exception("TTS initialization failed")
            return

        while True:
            text = items.get()

            try:
                if text is None:
                    logger.info("TTS worker stopping")
                    return

                logger.info("TTS generating speech for ESP32 speaker: %s", text)

                # 1. Synthesize to temp WAV file
                temp_wav = "temp_tts.wav"
                try:
                    if os.path.exists(temp_wav):
                        os.remove(temp_wav)
                except Exception:
                    pass

                engine.save_to_file(str(text), temp_wav)
                engine.runAndWait()

                # 2. Convert to 16-bit stereo PCM
                pcm_data = convert_wav_to_16bit_stereo(temp_wav)

                try:
                    if os.path.exists(temp_wav):
                        os.remove(temp_wav)
                except Exception:
                    pass

                # 3. Stream to ESP32 MAX98357A Speaker via HTTP POST /play
                if esp_ip and pcm_data:
                    play_url = f"http://{esp_ip}/play"
                    logger.info("Sending audio (%d bytes) to ESP32 speaker: %s", len(pcm_data), play_url)
                    try:
                        resp = http_session.post(
                            play_url,
                            data=pcm_data,
                            headers={"Content-Type": "application/octet-stream"},
                            timeout=8.0
                        )
                        if resp.status_code == 200:
                            logger.info("ESP32 external speaker playback successful!")
                        else:
                            logger.warning("ESP32 speaker returned status %d", resp.status_code)
                    except Exception as e:
                        logger.warning("Failed to send audio to ESP32 speaker: %s", e)
                else:
                    logger.warning("Cannot play on ESP32 speaker (esp_ip=%s, data_len=%d)", esp_ip, len(pcm_data) if pcm_data else 0)

            except Exception:
                logger.exception("TTS playback failed: %s", text)

            finally:
                items.task_done()

    thread = threading.Thread(
        target=worker,
        name="tts-worker",
        daemon=True,
    )
    thread.start()

    return items


class MicrophoneRecorder:
    """Record mono audio from either the system microphone or the ESP32 network microphone."""

    def __init__(self, sample_rate: int = 16000, device=None, esp_ip=None):
        self._sample_rate = sample_rate
        self._device = device
        self._esp_ip = esp_ip
        self._frames: list[np.ndarray] = []
        self._stream = None
        self._recording = False
        self._thread = None

    @property
    def recording(self) -> bool:
        return self._recording

    def start(self):
        self._frames = []
        self._recording = True

        if self._esp_ip:
            # Record from ESP32 I2S microphone via HTTP stream
            self._thread = threading.Thread(
                target=self._record_esp32,
                name="esp32-mic-recorder",
                daemon=True
            )
            self._thread.start()
        else:
            # Record from local system microphone using sounddevice
            import sounddevice as sd
            self._stream = sd.InputStream(
                samplerate=self._sample_rate,
                channels=1,
                dtype="float32",
                device=self._device,
                callback=self._callback,
            )
            self._stream.start()

    def _callback(self, indata, frame_count, time_info, status):
        if status:
            logger.warning("Microphone status: %s", status)
        self._frames.append(indata.copy())

    def _record_esp32(self):
        url = f"http://{self._esp_ip}:82/mic"
        logger.info("Connecting to ESP32 microphone stream: %s", url)
        try:
            # Use stream=True to read chunks as they arrive
            resp = http_session.get(url, stream=True, timeout=5.0)
            if resp.status_code != 200:
                logger.error("ESP32 microphone stream returned status %d", resp.status_code)
                return

            for chunk in resp.iter_content(chunk_size=1024):
                if not self._recording:
                    break
                if chunk:
                    # Convert 16-bit PCM bytes to float32 array
                    arr = np.frombuffer(chunk, dtype=np.int16).astype(np.float32) / 32768.0
                    self._frames.append(arr.reshape(-1, 1))

        except Exception as e:
            logger.warning("Error reading ESP32 microphone stream: %s", e)

    def stop(self) -> Path | None:
        import soundfile as sf

        self._recording = False

        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None

        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None

        if not self._frames:
            return None

        audio = np.concatenate(self._frames, axis=0)
        self._frames = []

        if np.max(np.abs(audio)) < 1e-4:
            logger.warning("Microphone recorded silence")
            return None

        fd, temp_path = tempfile.mkstemp(suffix=".wav")
        os.close(fd)
        path = Path(temp_path)

        sf.write(str(path), audio, self._sample_rate)
        logger.info(
            "Saved recording: %s (%.1f seconds)",
            path.name,
            len(audio) / self._sample_rate,
        )
        return path


def transcribe_audio_file(
    audio_path: Path,
    language: str = "en",
) -> str:
    """Transcribe a WAV file using the already-loaded faster-whisper model."""
    global whisper_model, whisper_lock

    with whisper_lock:
        if whisper_model is None:
            whisper_model = WhisperModel(
                os.getenv("WHISPER_MODEL", "small"),
                device=os.getenv("WHISPER_DEVICE", "cpu"),
                compute_type=os.getenv("WHISPER_COMPUTE_TYPE", "int8"),
            )

    segments, info = whisper_model.transcribe(
        str(audio_path),
        language=language,
        vad_filter=True,
        beam_size=3,
    )
    text = " ".join(s.text.strip() for s in segments).strip()
    return text


class LatestFrameCamera:
    """Continuously drains the source and exposes only the newest decoded frame."""

    def __init__(
        self,
        source,
        reconnect_initial: float = 0.25,
        reconnect_max: float = 4.0,
    ):
        self.source = source
        self.reconnect_initial = reconnect_initial
        self.reconnect_max = reconnect_max

        self._capture = None
        self._frame = None
        self._frame_id = 0
        self._last_frame_at = 0.0
        self._lock = threading.Lock()
        self._connected = threading.Event()
        self._local_stop = threading.Event()

        self._thread = threading.Thread(
            target=self._capture_loop,
            name="camera-capture",
            daemon=True,
        )
        self._thread.start()

    def _open(self):
        backend = cv2.CAP_FFMPEG if isinstance(self.source, str) else cv2.CAP_ANY
        capture = cv2.VideoCapture(self.source, backend)
        capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        if not capture.isOpened():
            capture.release()
            return None

        return capture

    def _capture_loop(self):
        delay = self.reconnect_initial
        is_http = isinstance(self.source, str) and self.source.startswith("http")

        if is_http:
            while not self._local_stop.is_set() and not shutdown_event.is_set():
                logger.info("Connecting to HTTP camera stream: %s", self.source)
                try:
                    resp = http_session.get(self.source, stream=True, timeout=5.0)
                    if resp.status_code == 200:
                        self._connected.set()
                        logger.info("HTTP camera stream connected successfully!")
                        bytes_data = b""
                        for chunk in resp.iter_content(chunk_size=4096):
                            if self._local_stop.is_set() or shutdown_event.is_set():
                                break
                            if not chunk:
                                continue
                            bytes_data += chunk
                            a = bytes_data.find(b"\xff\xd8")
                            b = bytes_data.find(b"\xff\xd9", a + 2) if a != -1 else -1
                            if a != -1 and b != -1:
                                jpg = bytes_data[a:b+2]
                                bytes_data = bytes_data[b+2:]
                                if len(jpg) > 100:
                                    frame = cv2.imdecode(np.frombuffer(jpg, dtype=np.uint8), cv2.IMREAD_COLOR)
                                    if frame is not None:
                                        with self._lock:
                                            self._frame = frame
                                            self._frame_id += 1
                                            self._last_frame_at = time.monotonic()
                        resp.close()
                    else:
                        logger.warning("HTTP camera stream returned status %d", resp.status_code)
                except Exception as e:
                    logger.warning("HTTP camera stream error: %s", e)

                self._connected.clear()
                self._local_stop.wait(delay)
                delay = min(delay * 2, self.reconnect_max)

            self._connected.clear()
            return

        while not self._local_stop.is_set() and not shutdown_event.is_set():
            if self._capture is None:
                logger.info("Connecting to camera: %s", self.source)
                self._capture = self._open()

                if self._capture is None:
                    self._connected.clear()
                    self._local_stop.wait(delay)
                    delay = min(delay * 2, self.reconnect_max)
                    continue

                delay = self.reconnect_initial
                self._connected.set()
                logger.info("Camera connected")

            ok, frame = self._capture.read()

            if not ok or frame is None:
                logger.warning("Camera read failed; reconnecting")
                self._connected.clear()
                self._capture.release()
                self._capture = None
                self._local_stop.wait(delay)
                delay = min(delay * 2, self.reconnect_max)
                continue

            with self._lock:
                self._frame = frame
                self._frame_id += 1
                self._last_frame_at = time.monotonic()

        if self._capture is not None:
            self._capture.release()
            self._capture = None

        self._connected.clear()

    def read_latest(self, last_frame_id: int = -1):
        with self._lock:
            if self._frame is None or self._frame_id == last_frame_id:
                return False, last_frame_id, None

            return True, self._frame_id, self._frame.copy()

    def wait_until_connected(self, timeout: float) -> bool:
        return self._connected.wait(timeout)

    @property
    def connected(self) -> bool:
        return self._connected.is_set()

    @property
    def frame_age_seconds(self) -> float:
        with self._lock:
            if self._last_frame_at == 0:
                return float("inf")
            return time.monotonic() - self._last_frame_at

    def release(self):
        self._local_stop.set()
        self._thread.join(timeout=3.0)


class TranscriptEvent:
    def __init__(self, partial_text: str, final_text: str, timestamp: float, language: str):
        self.partial_text = partial_text
        self.final_text = final_text
        self.timestamp = timestamp
        self.language = language

def run_camera(args):
    torch.set_num_threads(max(1, args.torch_threads))

    checkpoint=torch.load(
        args.checkpoint,
        map_location="cpu",
        weights_only=False,
    )
    config=checkpoint["config"]

    model=GestureBiGRU(
        checkpoint["input_size"],
        checkpoint["num_classes"],
        config["hidden_size"],
        config["gru_layers"],
        config["dropout"],
        config["bidirectional"],
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    classes=checkpoint["classes"]
    mean=np.asarray(checkpoint["feature_mean"],dtype=np.float32).reshape(-1)
    std=np.asarray(checkpoint["feature_std"],dtype=np.float32).reshape(-1)
    std=np.where(std<1e-6,1,std)
    threshold=float(checkpoint.get("confidence_threshold",0.65))
    length=int(config["sequence_length"])
    min_frames=int(config["min_detected_frames"])

    conversation=json.loads(
        Path(args.conversation).read_text(encoding="utf-8")
    )
    sentence_map=conversation["intent_sentences"]

    options=vision.HandLandmarkerOptions(
        base_options=python.BaseOptions(
            model_asset_path=str(Path(args.hand_model))
        ),
        running_mode=vision.RunningMode.VIDEO,
        num_hands=2,
        min_hand_detection_confidence=args.hand_detection_confidence,
        min_hand_presence_confidence=args.hand_presence_confidence,
        min_tracking_confidence=args.hand_tracking_confidence,
    )
    detector=vision.HandLandmarker.create_from_options(options)

    # Parse ESP32 IP for OLED, Speaker, Mic and Camera stream
    esp_ip = getattr(args, "esp_ip", None)
    if not esp_ip and isinstance(args.camera, str) and args.camera.startswith("http"):
        from urllib.parse import urlparse as _urlparse
        try:
            esp_ip = _urlparse(args.camera).hostname
        except Exception:
            pass

    # Auto-discover ESP32 if current IP is unreachable or changed by DHCP
    def resolve_esp_ip(target_ip):
        if target_ip:
            try:
                r = http_session.get(f"http://{target_ip}/status", timeout=0.6)
                if r.status_code == 200 and "Blind to Bright" in r.text:
                    return target_ip
            except Exception:
                logger.warning("ESP32 IP '%s' is not responding, scanning local network...", target_ip)

        import subprocess
        try:
            out = subprocess.check_output("arp -a", shell=True, text=True)
            for line in out.splitlines():
                parts = line.split()
                if len(parts) >= 2 and parts[0].count(".") == 3:
                    cand = parts[0]
                    if cand == target_ip:
                        continue
                    try:
                        r = http_session.get(f"http://{cand}/status", timeout=0.3)
                        if r.status_code == 200 and "Blind to Bright" in r.text:
                            logger.info("Found ESP32 online at new IP: %s", cand)
                            return cand
                    except Exception:
                        pass
        except Exception as e:
            logger.warning("ARP scan error: %s", e)

        return target_ip

    resolved_ip = resolve_esp_ip(esp_ip)
    if resolved_ip and resolved_ip != esp_ip:
        logger.info("Auto-updated ESP32 IP from %s to %s", esp_ip, resolved_ip)
        esp_ip = resolved_ip
        if isinstance(args.camera, str) and args.camera.startswith("http"):
            args.camera = f"http://{esp_ip}:81/stream"

    source = int(args.camera) if str(args.camera).isdigit() else args.camera
    camera = LatestFrameCamera(source)
    camera.wait_until_connected(args.camera_connect_timeout)

    tts = start_tts_worker(args.camera, language=args.language, esp_ip=esp_ip)
    time.sleep(0.3)
    ready_msg = "Hệ thống Blind to Bright đã sẵn sàng" if args.language == "vi" else "Blind to Bright is ready"
    tts.put(ready_msg)

    # Microphone recorder for speech input
    mic_device = None
    if hasattr(args, "audio_device") and args.audio_device is not None:
        try:
            mic_device = int(args.audio_device)
        except ValueError:
            mic_device = args.audio_device

    def remove_accents(s: str) -> str:
        import unicodedata
        s = unicodedata.normalize("NFD", str(s))
        s = "".join(c for c in s if unicodedata.category(c) != "Mn")
        return s.replace("đ", "d").replace("Đ", "D")

    # Non-blocking background worker for OLED messages
    oled_queue = queue.Queue(maxsize=10)

    def oled_worker():
        while not shutdown_event.is_set():
            try:
                msg = oled_queue.get(timeout=0.2)
            except queue.Empty:
                continue
            if esp_ip:
                try:
                    clean_text = remove_accents(msg).upper()
                    http_session.post(
                        f"http://{esp_ip}/oled",
                        data=clean_text.encode("utf-8"),
                        timeout=1.5,
                    )
                    logger.info("Sent to OLED: %s", clean_text)
                except Exception as e:
                    logger.warning("Failed to send to OLED: %s", e)
            oled_queue.task_done()

    oled_thread = threading.Thread(target=oled_worker, name="oled-worker", daemon=True)
    oled_thread.start()

    def send_to_oled(text: str):
        if esp_ip is None:
            return
        try:
            oled_queue.put_nowait(text)
        except queue.Full:
            pass

    # Gửi thông báo sẵn sàng lên OLED ngay khi khởi động
    send_to_oled("BLIND TO BRIGHT: SAN SANG")

    # Queues for pipeline
    audio_raw_queue = queue.Queue()
    whisper_task_queue = queue.Queue()
    transcript_queue = queue.Queue()

    # Shared structures
    landmarks_lock = threading.Lock()
    latest_landmarks = []
    
    speech_transcripts: deque[str] = deque(maxlen=3)
    current_partial_text = ""
    transcript_display_until = 0.0

    mic_volume_level = 0.0
    mic_source_str = "Connecting..."
    speech_active = False
    session_active = False
    segment_active = False
    
    label = "READY"
    confidence = 0.0
    sequence = []
    detected = 0
    segment_started_at = None
    last_hand_at = None

    def reset_segment():
        nonlocal segment_active, sequence, detected, segment_started_at, last_hand_at
        segment_active = False
        sequence = []
        detected = 0
        segment_started_at = None
        last_hand_at = None

    def predict_segment():
        nonlocal label, confidence
        if len(sequence) < min_frames or detected < min_frames:
            logger.info(
                "Segment ignored: %d frames, %d detected (< %d required)",
                len(sequence),
                detected,
                min_frames,
            )
            return

        try:
            features = prepare(sequence, length, mean, std)
            tensor = torch.from_numpy(features).unsqueeze(0)

            with torch.no_grad():
                logits = model(tensor)
                probs = torch.softmax(logits, dim=1).squeeze(0).numpy()

            best_idx = int(np.argmax(probs))
            best_conf = float(probs[best_idx])
            predicted_class = str(classes[best_idx])

            logger.info(
                "Inference result: %s (%.1f%%, threshold: %.1f%%)",
                predicted_class,
                best_conf * 100,
                threshold * 100,
            )

            if best_conf >= threshold:
                sentence = sentence_map.get(predicted_class, {}).get(
                    args.language, predicted_class
                )
                label = f"{predicted_class.upper()}"
                confidence = best_conf

                logger.info(
                    "Gesture matched: %s -> '%s' (%.1f%%)",
                    predicted_class,
                    sentence,
                    best_conf * 100,
                )

                # 1. Add to conversation history (displays on Web UI)
                state.add(
                    "gesture",
                    sentence,
                    {"intent": predicted_class, "confidence": f"{best_conf:.1%}"}
                )

                # 2. Speak out loud through laptop speaker
                tts.put(sentence)

                # 3. Send to ESP32 OLED if available
                send_to_oled(f"SIGN: {sentence}")
            else:
                logger.info(
                    "Confidence %.1f%% below threshold %.1f%%, ignoring",
                    best_conf * 100,
                    threshold * 100,
                )
                label = f"UNCERTAIN ({best_conf:.0%})"
                confidence = best_conf

        except Exception as e:
            logger.exception("Error predicting gesture segment: %s", e)

    # Audio Thread
    def audio_thread_func():
        nonlocal mic_volume_level, mic_source_str
        esp_failed_last = False
        last_esp_retry = 0.0

        while not shutdown_event.is_set():
            if esp_ip and not esp_failed_last:
                url = f"http://{esp_ip}:82/mic"
                logger.info("Audio Thread: Connecting to ESP32 mic stream: %s", url)
                try:
                    resp = http_session.get(url, stream=True, timeout=5.0)
                    if resp.status_code == 200:
                        logger.info("Audio Thread: Connected to ESP32 mic stream (Port 82)")
                        mic_source_str = "ESP32 (INMP441)"
                        raw_buf = bytearray()
                        for chunk in resp.iter_content(chunk_size=1024):
                            if shutdown_event.is_set():
                                break
                            if chunk:
                                raw_buf.extend(chunk)
                                while len(raw_buf) >= 1024:
                                    chunk_bytes = bytes(raw_buf[:1024])
                                    del raw_buf[:1024]
                                    raw_int16 = np.frombuffer(chunk_bytes, dtype=np.int16)
                                    samples = raw_int16.astype(np.float32) / 32768.0
                                    # Digital gain 4.0x for INMP441 sensitivity
                                    samples = np.clip(samples * 4.0, -1.0, 1.0)
                                    peak = float(np.max(np.abs(samples)))
                                    mic_volume_level = 0.7 * mic_volume_level + 0.3 * peak
                                    audio_raw_queue.put(samples)
                        resp.close()
                        continue
                    else:
                        logger.error("Audio Thread: ESP32 HTTP %d, using local mic fallback", resp.status_code)
                        esp_failed_last = True
                        last_esp_retry = time.monotonic()
                except Exception as e:
                    logger.warning("Audio Thread: ESP32 mic error: %s, using local mic fallback", e)
                    esp_failed_last = True
                    last_esp_retry = time.monotonic()

            logger.info("Audio Thread: Starting local mic capture (sounddevice)")
            mic_source_str = "Laptop Mic"
            import sounddevice as sd
            
            def sd_callback(indata, frames, time_info, status):
                nonlocal mic_volume_level
                if status:
                    logger.warning("Local mic status: %s", status)
                data = indata.flatten().copy()
                peak = float(np.max(np.abs(data)))
                mic_volume_level = 0.7 * mic_volume_level + 0.3 * peak
                audio_raw_queue.put(data)

            try:
                stream = sd.InputStream(
                    samplerate=16000,
                    channels=1,
                    dtype="float32",
                    blocksize=512,
                    device=mic_device,
                    callback=sd_callback,
                )
                with stream:
                    while not shutdown_event.is_set():
                        if esp_ip and esp_failed_last:
                            if time.monotonic() - last_esp_retry > 8.0:
                                logger.info("Audio Thread: Retrying ESP32 mic...")
                                try:
                                    test_resp = http_session.get(f"http://{esp_ip}:82/mic", stream=True, timeout=2.0)
                                    if test_resp.status_code == 200:
                                        test_resp.close()
                                        logger.info("Audio Thread: ESP32 mic back online, switching...")
                                        esp_failed_last = False
                                        break
                                except Exception:
                                    last_esp_retry = time.monotonic()
                        time.sleep(0.1)
            except Exception as e:
                logger.error("Audio Thread: Failed to start local microphone: %s", e)
                time.sleep(2.0)

    # VAD Thread
    def vad_thread_func():
        nonlocal speech_active
        from silero_vad import load_silero_vad
        logger.info("VAD Thread: Loading Silero VAD...")
        vad_model = load_silero_vad()
        logger.info("VAD Thread: Silero VAD loaded.")
        
        speech_buffer = []
        silence_frames = 0
        max_silence_frames = 20  # 20 * 32ms = 640ms silence to conclude speech
        speech_threshold = 0.32  # Optimal sensitivity with 4x gain
        
        frames_since_partial = 0
        partial_interval_frames = 25  # ~800ms
        
        while not shutdown_event.is_set():
            try:
                chunk = audio_raw_queue.get(timeout=0.1)
            except queue.Empty:
                continue
                
            chunk_tensor = torch.from_numpy(chunk).unsqueeze(0)
            with torch.no_grad():
                prob = vad_model(chunk_tensor, 16000).item()
                
            is_speech = prob > speech_threshold
            
            if is_speech:
                if not speech_active:
                    speech_active = True
                    speech_buffer = []
                    silence_frames = 0
                    frames_since_partial = 0
                    logger.info("VAD: Speech started (prob=%.2f)", prob)
                    
                speech_buffer.append(chunk)
                silence_frames = 0
                frames_since_partial += 1
                
                # Throttle partials: only enqueue when Whisper has finished previous task
                if frames_since_partial >= partial_interval_frames and whisper_task_queue.empty():
                    frames_since_partial = 0
                    accumulated = np.concatenate(speech_buffer, axis=0)
                    whisper_task_queue.put((accumulated, False))
                    
            else:
                if speech_active:
                    speech_buffer.append(chunk)
                    silence_frames += 1
                    
                    if silence_frames >= max_silence_frames:
                        speech_active = False
                        logger.info("VAD: Speech ended (silence detected, %d chunks)", len(speech_buffer))
                        accumulated = np.concatenate(speech_buffer, axis=0)
                        # Clear old partials from queue so final is processed immediately
                        while not whisper_task_queue.empty():
                            try:
                                whisper_task_queue.get_nowait()
                            except queue.Empty:
                                break
                        if len(speech_buffer) >= 12:  # at least ~380ms of audio
                            whisper_task_queue.put((accumulated, True))
                        speech_buffer = []
                        silence_frames = 0
                        frames_since_partial = 0

    # Whisper Thread
    def whisper_thread_func():
        global whisper_model, whisper_lock
        
        speech_lang = getattr(args, "speech_language", None) or getattr(args, "language", "vi")
        model_name = os.getenv("WHISPER_MODEL", "base")

        with whisper_lock:
            if whisper_model is None:
                cuda = torch.cuda.is_available()
                logger.info(
                    "Whisper Thread: Initializing Whisper '%s' (cuda=%s, lang=%s)...",
                    model_name,
                    cuda,
                    speech_lang,
                )
                whisper_model = WhisperModel(
                    model_name,
                    device="cuda" if cuda else "cpu",
                    compute_type="float16" if cuda else "int8"
                )
                logger.info("Whisper Thread: Whisper initialized.")
                
        while not shutdown_event.is_set():
            try:
                audio_samples, is_final = whisper_task_queue.get(timeout=0.1)
            except queue.Empty:
                continue
                
            try:
                with whisper_lock:
                    segments, info = whisper_model.transcribe(
                        audio_samples,
                        beam_size=2,
                        language=speech_lang,
                        vad_filter=False
                    )
                    text = " ".join(s.text.strip() for s in segments).strip()
                    
                if is_final:
                    if text:
                        logger.info("Whisper Final: '%s' (lang=%s)", text, speech_lang)
                        event = TranscriptEvent(
                            partial_text="",
                            final_text=text,
                            timestamp=time.time(),
                            language=speech_lang
                        )
                        transcript_queue.put(event)
                else:
                    if text:
                        logger.info("Whisper Partial: '%s'", text)
                        event = TranscriptEvent(
                            partial_text=text,
                            final_text="",
                            timestamp=time.time(),
                            language=speech_lang
                        )
                        transcript_queue.put(event)
                    
            except Exception as e:
                logger.error("Whisper Thread: Transcription error: %s", e)

    # Sign Thread
    def sign_thread_func():
        nonlocal label, confidence, segment_active, sequence, detected, last_hand_at, segment_started_at, latest_landmarks
        last_frame_id = -1
        last_mp_at = 0.0
        last_mp_timestamp_ms = -1
        mp_interval = 1.0 / max(1.0, args.mediapipe_fps)
        
        while not shutdown_event.is_set():
            if not session_active:
                time.sleep(0.01)
                continue
                
            is_new, last_frame_id, frame = camera.read_latest(last_frame_id)
            if not is_new:
                time.sleep(0.005)
                continue
                
            now = time.monotonic()
            if now - last_mp_at < mp_interval:
                time.sleep(0.002)
                continue
                
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            timestamp_ms = int(now * 1000)
            if timestamp_ms <= last_mp_timestamp_ms:
                timestamp_ms = last_mp_timestamp_ms + 1
            last_mp_timestamp_ms = timestamp_ms
            
            try:
                result = detector.detect_for_video(
                    mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb),
                    timestamp_ms
                )
            except Exception as e:
                logger.warning("MediaPipe detection error: %s", e)
                result = None
            last_mp_at = now
            
            features = frame_features(result, force_right_hand=not args.no_force_right)
            hand_present = bool(result is not None and result.hand_landmarks and np.any(features != 0))
            
            with landmarks_lock:
                latest_landmarks = []
                if result and result.hand_landmarks:
                    for landmarks in result.hand_landmarks:
                        latest_landmarks.append([(p.x, p.y) for p in landmarks])
                        
            if hand_present:
                last_hand_at = now
                if not segment_active:
                    segment_active = True
                    segment_started_at = now
                    sequence = []
                    detected = 0
                    label = "CAPTURING"
                    confidence = 0.0
                    logger.info("Gesture segment started")
                    
                sequence.append(features.copy())
                detected += 1
            elif segment_active:
                sequence.append(features.copy())
                gap = now - last_hand_at if last_hand_at is not None else 0.0
                duration = now - segment_started_at if segment_started_at is not None else 0.0
                
                if gap >= args.segment_end_gap or duration >= args.max_segment_seconds:
                    logger.info("Gesture segment ended: %d frames, %.2fs", len(sequence), duration)
                    predict_segment()
                    reset_segment()

    # Start helper threads
    threads = [
        threading.Thread(target=audio_thread_func, name="audio-thread", daemon=True),
        threading.Thread(target=vad_thread_func, name="vad-thread", daemon=True),
        threading.Thread(target=whisper_thread_func, name="whisper-thread", daemon=True),
        threading.Thread(target=sign_thread_func, name="sign-thread", daemon=True),
    ]
    for t in threads:
        t.start()

    # Main UI loop
    fps_started = time.monotonic()
    fps_frames = 0
    display_fps = 0.0
    last_frame_id = -1
    
    try:
        while not shutdown_event.is_set():
            is_new, last_frame_id, frame = camera.read_latest(last_frame_id)
            if not is_new:
                time.sleep(0.002)
                continue
                
            now = time.monotonic()
            h, w = frame.shape[:2]
            
            # Non-blocking get of transcript events
            try:
                while True:
                    event = transcript_queue.get_nowait()
                    if event.final_text:
                        final_text = event.final_text
                        logger.info("Final Speech: %s", final_text)
                        state.add("speech", final_text, {"language": event.language})
                        speech_transcripts.append(final_text)
                        transcript_display_until = time.monotonic() + 10.0
                        send_to_oled(f"MIC: {final_text}")
                        current_partial_text = ""
                    else:
                        current_partial_text = event.partial_text
                        transcript_display_until = time.monotonic() + 10.0
                        if len(current_partial_text) > 3:
                            send_to_oled(f"MIC: {current_partial_text}")
            except queue.Empty:
                pass
                
            # Draw hand landmarks
            with landmarks_lock:
                landmarks_to_draw = list(latest_landmarks)
                
            for landmarks in landmarks_to_draw:
                points = [(int(x * w), int(y * h)) for x, y in landmarks]
                for start_idx, end_idx in HAND_CONNECTIONS:
                    if start_idx < len(points) and end_idx < len(points):
                        cv2.line(frame, points[start_idx], points[end_idx], (0, 255, 0), 1)
                for point in points:
                    cv2.circle(frame, point, 2, (255, 0, 255), -1)
                    
            # Compute FPS
            fps_frames += 1
            elapsed = now - fps_started
            if elapsed >= 1.0:
                display_fps = fps_frames / elapsed
                fps_frames = 0
                fps_started = now
                
            # Status overlays (Top Left)
            if speech_active:
                status_text = "LISTENING"
                status_color = (0, 165, 255)
            elif segment_active:
                status_text = "CAPTURING"
                status_color = (0, 0, 255)
            elif session_active:
                status_text = "SESSION ON"
                status_color = (0, 255, 0)
            else:
                status_text = "READY"
                status_color = (0, 255, 0)
                
            cv2.putText(
                frame,
                status_text,
                (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                status_color,
                2,
            )
            cv2.putText(
                frame,
                f"FPS {display_fps:.1f} | AI {args.mediapipe_fps:.1f}",
                (20, 65),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                1,
            )
            cv2.putText(
                frame,
                f"SIGN: {label} ({confidence:.0%})",
                (20, 95),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 255, 255),
                2,
            )

            # VU Level Meter & Microphone Status (Top Right)
            meter_w = 120
            meter_h = 12
            meter_x = w - meter_w - 20
            meter_y = 22

            # Background bar
            cv2.rectangle(frame, (meter_x, meter_y), (meter_x + meter_w, meter_y + meter_h), (40, 40, 40), -1)
            cv2.rectangle(frame, (meter_x, meter_y), (meter_x + meter_w, meter_y + meter_h), (130, 130, 130), 1)

            # Filled audio meter bar
            fill_w = int(np.clip(mic_volume_level * 2.5, 0.0, 1.0) * meter_w)
            if fill_w > 0:
                meter_color = (0, 255, 0) if fill_w < meter_w * 0.6 else ((0, 220, 255) if fill_w < meter_w * 0.85 else (0, 0, 255))
                cv2.rectangle(frame, (meter_x + 1, meter_y + 1), (meter_x + fill_w, meter_y + meter_h - 1), meter_color, -1)

            # Source label
            cv2.putText(frame, f"MIC: {mic_source_str}", (meter_x, meter_y - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (220, 220, 220), 1)

            # Blinking REC dot when VAD detects speech
            if speech_active and (int(now * 3) % 2 == 0):
                cv2.circle(frame, (meter_x - 12, meter_y + 6), 5, (0, 0, 255), -1)
                cv2.putText(frame, "REC", (meter_x - 45, meter_y + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 0, 255), 1)

            # UI control hints (Above banner)
            hints = [
                "SPACE: Sign session",
                "Q: Quit",
            ]
            for i, hint in enumerate(hints):
                cv2.putText(
                    frame,
                    hint,
                    (w - 200, h - 95 - (len(hints) - 1 - i) * 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    (180, 180, 180),
                    1,
                )

            # High-contrast Speech-to-Text Subtitle Banner (Bottom)
            display_title = ""
            display_text = ""
            if current_partial_text:
                display_title = "🎙️ ĐANG NÓI (ESP32 MIC)..."
                display_text = current_partial_text
            elif speech_transcripts and now < transcript_display_until:
                display_title = "🎙️ LỜI NÓI (ESP32 MICROPHONE):"
                display_text = speech_transcripts[-1]
            elif speech_active:
                display_title = "🎙️ MICROPHONE:"
                display_text = "Đang lắng nghe giọng nói..."

            if display_title and display_text:
                frame = draw_vn_banner(frame, display_title, display_text)
                    
            cv2.imshow("Communication Assistant", frame)
            key = cv2.waitKey(1) & 0xFF
            
            if key in (ord("q"), 27):
                shutdown_event.set()
                break
                
            if key == 32:
                session_active = not session_active
                if session_active:
                    reset_segment()
                    label = "SESSION ON"
                    confidence = 0.0
                    logger.info("Continuous recognition session started")
                else:
                    reset_segment()
                    label = "READY"
                    confidence = 0.0
                    logger.info("Continuous recognition session stopped")

    finally:
        shutdown_event.set()
        camera.release()
        detector.close()
        tts.put(None)
        cv2.destroyAllWindows()

def install_signal_handlers():
    def stop_handler(signum, frame):
        logger.info("Received signal %s; shutting down", signum)
        shutdown_event.set()

    signal.signal(signal.SIGINT, stop_handler)

    if hasattr(signal, "SIGTERM"):
        signal.signal(signal.SIGTERM, stop_handler)


def main():
    root_dir = Path(__file__).resolve().parent.parent
    def _res(preferred, fallback):
        p = root_dir / preferred
        return str(p if p.exists() else (root_dir / fallback))

    parser = argparse.ArgumentParser(
        description="Low-latency sign-language communication assistant"
    )
    parser.add_argument("--checkpoint", default=_res("models/best_bigru_v2.pt", "best_bigru_v2.pt"))
    parser.add_argument("--conversation", default=_res("configs/conversation.json", "conversation_config.json"))
    parser.add_argument("--hand-model", default=_res("models/hand_landmarker.task", "hand_landmarker.task"))
    parser.add_argument("--camera", default="0")
    parser.add_argument("--language", choices=["vi", "en"], default="vi")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--web-only", action="store_true")
    parser.add_argument("--no-force-right", action="store_true")

    parser.add_argument("--mediapipe-fps", type=float, default=10.0)
    parser.add_argument("--torch-threads", type=int, default=2)
    parser.add_argument("--camera-connect-timeout", type=float, default=8.0)
    parser.add_argument("--camera-stale-timeout", type=float, default=3.0)
    parser.add_argument("--segment-end-gap", type=float, default=0.45)
    parser.add_argument("--max-segment-seconds", type=float, default=4.0)

    parser.add_argument("--audio-device", default=None,
                        help="Microphone device index or name")
    parser.add_argument("--sample-rate", type=int, default=16000,
                        help="Microphone sample rate in Hz")
    parser.add_argument("--esp-ip", default=None,
                        help="IP address of ESP32 (if different from camera hostname)")
    parser.add_argument("--speech-language", default="vi",
                        help="Language for speech transcription (vi or en)")

    parser.add_argument(
        "--hand-detection-confidence",
        type=float,
        default=0.30,
    )
    parser.add_argument(
        "--hand-presence-confidence",
        type=float,
        default=0.30,
    )
    parser.add_argument(
        "--hand-tracking-confidence",
        type=float,
        default=0.30,
    )

    args = parser.parse_args()
    install_signal_handlers()

    web_thread = threading.Thread(
        target=run_web,
        args=(args.host, args.port),
        name="web-server",
        daemon=True,
    )
    web_thread.start()

    logger.info(
        "Phone notes URL: http://YOUR_LAPTOP_IP:%d",
        args.port,
    )

    try:
        if args.web_only:
            while not shutdown_event.wait(1.0):
                pass
        else:
            run_camera(args)
    except KeyboardInterrupt:
        shutdown_event.set()
    except Exception:
        logger.exception("Fatal application error")
        shutdown_event.set()
        raise


if __name__ == "__main__":
    main()