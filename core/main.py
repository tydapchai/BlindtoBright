import os
import sys
from pathlib import Path

# Bypass proxy cho mạng nội bộ kết nối trực tiếp đến ESP32
os.environ["NO_PROXY"] = "*"
os.environ["no_proxy"] = "*"
for _k in ["HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy", "ALL_PROXY", "all_proxy"]:
    os.environ.pop(_k, None)

ROOT_DIR = Path(__file__).resolve().parent.parent
CORE_DIR = Path(__file__).resolve().parent
if str(CORE_DIR) not in sys.path:
    sys.path.insert(0, str(CORE_DIR))
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import cv2
import time
import json
import torch
import numpy as np
import threading
import argparse
import requests
import subprocess
import mediapipe as mp

from config import state, shutdown_event
from models import GestureBiGRU, frame_features, prepare, HAND_CONNECTIONS
from mediapipe.tasks.python import vision
from gemini_api import GeminiClient
from web_app import run_web

http_session = requests.Session()
http_session.trust_env = False

class LatestFrameCamera:
    """Đọc luồng camera mượt mà không bị treo socket/proxy trên Windows."""
    def __init__(self, source, reconnect_initial=0.25, reconnect_max=4.0):
        self.source = source
        self.reconnect_initial = reconnect_initial
        self.reconnect_max = reconnect_max
        self._frame = None
        self._frame_id = 0
        self._lock = threading.Lock()
        self._connected = threading.Event()
        self._local_stop = threading.Event()

        self._thread = threading.Thread(target=self._capture_loop, name="camera-capture", daemon=True)
        self._thread.start()

    def _capture_loop(self):
        delay = self.reconnect_initial
        is_http = isinstance(self.source, str) and self.source.startswith("http")

        if is_http:
            while not self._local_stop.is_set() and not shutdown_event.is_set():
                try:
                    resp = http_session.get(self.source, stream=True, timeout=5.0)
                    if resp.status_code == 200:
                        self._connected.set()
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
                        resp.close()
                except Exception:
                    pass
                self._connected.clear()
                self._local_stop.wait(delay)
                delay = min(delay * 2, self.reconnect_max)
            return

        cap = cv2.VideoCapture(self.source)
        while not self._local_stop.is_set() and not shutdown_event.is_set():
            ret, frame = cap.read()
            if ret and frame is not None:
                self._connected.set()
                with self._lock:
                    self._frame = frame
                    self._frame_id += 1
            else:
                self._connected.clear()
                time.sleep(0.01)
        cap.release()

    def read_latest(self, last_seen_id=-1):
        with self._lock:
            if self._frame is None:
                return False, last_seen_id, None
            is_new = (self._frame_id != last_seen_id)
            return is_new, self._frame_id, self._frame.copy()

    def read(self):
        with self._lock:
            if self._frame is None:
                return False, None
            return True, self._frame.copy()

    def wait_until_connected(self, timeout=5.0):
        return self._connected.wait(timeout)

    def release(self):
        self._local_stop.set()

def draw_vn_banner(frame, title, text):
    h, w = frame.shape[:2]
    banner_h = 75
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, h - banner_h), (w, h), (15, 15, 15), -1)
    cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)
    cv2.line(frame, (0, h - banner_h), (w, h - banner_h), (0, 215, 255), 2)
    
    cv2.putText(frame, title.upper(), (25, h - banner_h + 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 215, 255), 2)
    cv2.putText(frame, text, (25, h - 16), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (255, 255, 255), 2)
    return frame

def _resolve_default(preferred: Path, fallback: Path) -> str:
    return str(preferred if preferred.exists() else fallback)

def resolve_esp_ip(target_ip):
    session = requests.Session()
    session.trust_env = False

    # 1. Thử kết nối IP mục tiêu trước (IP truyền vào hoặc IP đã lưu trong configs/esp_ip.txt)
    if target_ip:
        try:
            r = session.get(f"http://{target_ip}/status", timeout=1.2)
            if r.status_code == 200 and "Blind to Bright" in r.text:
                return target_ip
        except Exception:
            print(f"[ESP32] IP cấu hình '{target_ip}' không phản hồi qua Wi-Fi...")

    # 2. TỰ ĐỘNG FALLBACK VỀ MẠNG CỦA ESP (SoftAP 192.168.4.1) KHI ESP KHÔNG VÀO ĐƯỢC WI-FI
    if target_ip != "192.168.4.1":
        try:
            r = session.get("http://192.168.4.1/status", timeout=0.8)
            if r.status_code == 200 and "Blind to Bright" in r.text:
                print("\n" + "*" * 65)
                print("[ESP32 Fallback] Đã tự động FALLBACK sang mạng của ESP (SoftAP): 192.168.4.1")
                print("*" * 65 + "\n")
                return "192.168.4.1"
        except Exception:
            pass

    # 3. Quét thông minh bảng ARP trong mạng LAN nếu ESP được router cấp IP DHCP mới
    ESP_MACS = (
        "14-c1-9f", "24-0a-c4", "24-6f-28", "24-dc-c3", "30-ae-a4", "3c-61-05", "3c-71-bf",
        "40-22-d8", "40-91-51", "48-27-e2", "48-31-b7", "48-55-19", "54-32-04", "54-43-b2",
        "70-04-1d", "7c-df-a1", "84-0d-8e", "84-cc-a8", "84-f7-03", "94-3c-c6", "a4-cf-12",
        "a4-e5-7c", "b4-e6-2d", "bc-dd-c2", "c4-4f-33", "c4-dd-57", "cc-50-e3", "dc-54-75",
        "e8-31-cd", "e8-db-84", "ec-94-cb", "f0-08-d1"
    )
    try:
        import subprocess
        from concurrent.futures import ThreadPoolExecutor

        out = subprocess.check_output("arp -a", shell=True, text=True)
        esp_cands = []
        other_cands = []
        for line in out.splitlines():
            parts = line.split()
            if len(parts) >= 2 and parts[0].count(".") == 3:
                ip = parts[0]
                mac = parts[1].lower() if len(parts) > 1 else ""
                if ip not in [target_ip, "192.168.4.1", "255.255.255.255"] and not ip.startswith("224.") and not ip.startswith("239.") and not ip.endswith(".255"):
                    if any(mac.startswith(pfx) for pfx in ESP_MACS):
                        esp_cands.append(ip)
                    else:
                        other_cands.append(ip)

        ordered_cands = esp_cands + other_cands
        def _check(ip):
            try:
                res = session.get(f"http://{ip}/status", timeout=0.5)
                if res.status_code == 200 and "Blind to Bright" in res.text:
                    return ip
            except Exception:
                pass
            return None

        with ThreadPoolExecutor(max_workers=20) as pool:
            for cand_ip in pool.map(_check, ordered_cands):
                if cand_ip:
                    print(f"[ESP32 Auto-Discover] Đã tìm thấy ESP32 tại IP mạng LAN: {cand_ip}")
                    return cand_ip
    except Exception:
        pass

    # 4. Kiểm tra xem có đang phát Wi-Fi SoftAP gần không
    try:
        import subprocess
        wlan_out = subprocess.check_output("netsh wlan show networks", shell=True, text=True, stderr=subprocess.DEVNULL)
        if "ESP32-S3-CAMERA" in wlan_out:
            print("\n" + "=" * 65)
            print("[ESP32 Fallback] Phát hiện Wi-Fi SoftAP của ESP: 'ESP32-S3-CAMERA'!")
            print("[ESP32 Fallback] Đang thử kết nối máy tính vào mạng của ESP...")
            print("=" * 65)
            subprocess.run(["netsh", "wlan", "connect", "name=ESP32-S3-CAMERA"], capture_output=True, timeout=5)
            time.sleep(2.5)
            try:
                r = session.get("http://192.168.4.1/status", timeout=1.0)
                if r.status_code == 200 and "Blind to Bright" in r.text:
                    print("[ESP32 Fallback] Đã kết nối thành công tới ESP32 tại 192.168.4.1!\n")
                    return "192.168.4.1"
            except Exception:
                pass
    except Exception:
        pass

    fallback = target_ip or "192.168.4.1"
    print("\n" + "!" * 65)
    print(f"[ESP32 Cảnh báo] Không thể kết nối tới {target_ip} hoặc mạng của ESP (192.168.4.1). Sử dụng tạm: {fallback}")
    print(f"       -> Mẹo: Nếu ESP không vào được Wi-Fi, hãy chuyển Wi-Fi trên máy tính sang:")
    print(f"               SSID: 'ESP32-S3-CAMERA' (mật khẩu: 12345678)")
    print("!" * 65 + "\n")
    return fallback


def main():
    default_ckpt = _resolve_default(ROOT_DIR / "models" / "best_bigru_v2.pt", ROOT_DIR / "best_bigru_v2.pt")
    default_conv = _resolve_default(ROOT_DIR / "configs" / "conversation.json", ROOT_DIR / "conversation_config.json")
    default_hand = _resolve_default(ROOT_DIR / "models" / "hand_landmarker.task", ROOT_DIR / "hand_landmarker.task")

    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default=default_ckpt)
    parser.add_argument("--conversation", default=default_conv)
    parser.add_argument("--hand-model", default=default_hand)
    parser.add_argument("--camera", default="0")
    parser.add_argument("--esp-ip", default=None, help="IP của ESP32")
    parser.add_argument("--camera-connect-timeout", type=float, default=6.0)
    parser.add_argument("--gemini-key", default=None, help="Google Gemini API Key")
    args = parser.parse_args()

    threading.Thread(target=run_web, args=("0.0.0.0", 8000), daemon=True).start()

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config = checkpoint["config"]
    model = GestureBiGRU(checkpoint["input_size"], checkpoint["num_classes"], config["hidden_size"])
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    classes = checkpoint["classes"]
    mean = np.asarray(checkpoint["feature_mean"], dtype=np.float32).reshape(-1)
    std = np.asarray(checkpoint["feature_std"], dtype=np.float32).reshape(-1)
    
    with open(args.conversation, 'r', encoding='utf-8') as f:
        sentence_map = json.load(f)["intent_sentences"]

    options = vision.HandLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=args.hand_model),
        running_mode=vision.RunningMode.VIDEO, num_hands=2
    )
    detector = vision.HandLandmarker.create_from_options(options)

    # =========================================================================
    # CẤU HÌNH IP ESP32 (ƯU TIÊN CLI -> FILE CONFIG configs/esp_ip.txt)
    # =========================================================================
    config_ip_file = ROOT_DIR / "configs" / "esp_ip.txt"
    esp_ip = getattr(args, "esp_ip", None)

    # Nếu người dùng truyền URL trực tiếp trong --camera
    if not esp_ip and isinstance(args.camera, str) and args.camera.startswith("http"):
        from urllib.parse import urlparse as _urlparse
        try:
            esp_ip = _urlparse(args.camera).hostname
        except Exception:
            pass

    # Nếu chưa có, đọc từ file cấu hình configs/esp_ip.txt
    if not esp_ip and config_ip_file.is_file():
        try:
            cached = config_ip_file.read_text(encoding="utf-8").strip()
            if cached:
                esp_ip = cached
        except Exception:
            pass

    # Mặc định dự phòng nếu chưa có cấu hình
    if not esp_ip:
        esp_ip = "10.3.79.128"

    # Kiểm tra IP và TỰ ĐỘNG FALLBACK về mạng của ESP (SoftAP 192.168.4.1) khi không vào được Wi-Fi
    if str(args.camera).lower() in ["esp", "esp32", "cam"] or (esp_ip and not str(args.camera).isdigit()):
        esp_ip = resolve_esp_ip(esp_ip)

    # Lưu lại IP để các lần sau không cần nhập lại
    try:
        config_ip_file.parent.mkdir(parents=True, exist_ok=True)
        config_ip_file.write_text(esp_ip, encoding="utf-8")
    except Exception:
        pass

    args.esp_ip = esp_ip

    # Thiết lập nguồn camera
    if str(args.camera).lower() in ["esp", "esp32", "cam"]:
        args.camera = f"http://{esp_ip}:81/stream"
        print(f"[Camera] Đang sử dụng Camera ESP32: {args.camera}")
        print(f"[ESP32] IP điều khiển (Loa / Mic / OLED): {esp_ip}")
    elif isinstance(args.camera, str) and args.camera.startswith("http"):
        print(f"[Camera] Đang sử dụng luồng: {args.camera}")
        print(f"[ESP32] IP điều khiển: {esp_ip}")
    else:
        print(f"[Camera] Đang sử dụng Webcam máy tính: {args.camera}")
        if esp_ip:
            print(f"[ESP32] IP kết nối phụ trợ (Loa / Mic / OLED): {esp_ip}")
    # =========================================================================

    # Khởi tạo Camera và Gemini 
    source = int(args.camera) if str(args.camera).isdigit() else args.camera
    cap = LatestFrameCamera(source)
    cap.wait_until_connected(args.camera_connect_timeout)
    gemini = GeminiClient(api_keys=args.gemini_key)
    gemini.preload_cache([v.get("vi", k) for k, v in sentence_map.items()])

    # Phát lời chào sẵn sàng qua TTS Gemini
    ready_msg = "Hệ thống Blind to Bright đã sẵn sàng"
    def welcome_speaker():
        pcm = gemini.generate_speech(ready_msg)
        if args.esp_ip and pcm:
            try:
                http_session.post(f"http://{args.esp_ip}/play", data=pcm, timeout=5)
            except: pass
    threading.Thread(target=welcome_speaker).start()

    # Các biến trạng thái của UI
    is_recording_audio = False
    is_capturing_sign = False
    audio_buffer = bytearray()
    sign_sequence = []
    display_text, display_title = "", ""

    # Luồng hút âm thanh từ Mic I2S của ESP32
    def audio_fetch_worker():
        nonlocal audio_buffer, is_recording_audio
        while not shutdown_event.is_set():
            if is_recording_audio and args.esp_ip:
                try:
                    resp = http_session.get(f"http://{args.esp_ip}:82/mic", stream=True, timeout=2.0)
                    if resp.status_code == 200:
                        for chunk in resp.iter_content(chunk_size=1024):
                            if not is_recording_audio or shutdown_event.is_set():
                                break
                            if chunk:
                                audio_buffer.extend(chunk)
                except Exception:
                    time.sleep(0.5)
            else:
                time.sleep(0.1)
                
    threading.Thread(target=audio_fetch_worker, daemon=True).start()

    print("Hệ thống đã sẵn sàng! Bấm 'SPACE' thu ký hiệu, bấm 'M' thu âm.")

    last_frame_id = -1
    last_mp_timestamp_ms = -1
    result = None

    WINDOW_NAME = "Communication Assistant"
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, 1024, 768)

    while not shutdown_event.is_set():
        is_new, last_frame_id, frame = cap.read_latest(last_frame_id)
        if frame is None:
            time.sleep(0.01)
            continue

        key = cv2.waitKey(1) & 0xFF

        # 1. BẤM NÚT KÝ HIỆU (SPACE)
        if key == 32: 
            is_capturing_sign = not is_capturing_sign
            if is_capturing_sign:
                sign_sequence = []
                display_title = "DANG THU KY HIEU..."
                display_text = "Thuc hien dong tac..."
            else:
                if len(sign_sequence) > 5:
                    features = prepare(sign_sequence, config["sequence_length"], mean, std)
                    tensor = torch.from_numpy(features).unsqueeze(0)
                    with torch.no_grad():
                        logits = model(tensor)
                        probs = torch.softmax(logits, dim=1).squeeze(0).numpy()
                    
                    best_idx = int(np.argmax(probs))
                    best_class = classes[best_idx]
                    sentence = sentence_map.get(best_class, {}).get("vi", best_class)
                    
                    display_title = "DICH KY HIEU"
                    display_text = sentence
                    state.add("gesture", sentence)
                    
                    def speak_out():
                        t0 = time.time()
                        pcm = gemini.generate_speech(sentence)
                        if args.esp_ip and pcm:
                            try:
                                http_session.post(f"http://{args.esp_ip}/play", data=pcm, timeout=5)
                                print(f"[Speaker] Đã phát ra loa ({time.time() - t0:.3f}s): {sentence}")
                            except Exception as e:
                                print(f"[Speaker Error] Gửi âm thanh thất bại: {e}")
                    threading.Thread(target=speak_out).start()
                else:
                    display_title = "LOI"
                    display_text = "Dong tac qua ngan"

        # 2. BẤM NÚT GHI ÂM (M)
        elif key == ord('m') or key == ord('M'):
            is_recording_audio = not is_recording_audio
            if is_recording_audio:
                audio_buffer = bytearray()
                display_title = "DANG THU AM..."
                display_text = "Nguoi doi dien hay noi..."
            else:
                display_title = "DANG DICH..."
                display_text = "Cho Gemini xu ly..."
                
                def transcribe_job(pcm_bytes):
                    nonlocal display_title, display_text
                    if len(pcm_bytes) < 4000:
                        display_title, display_text = "STT", "Khong nghe gi"
                        return
                    text = gemini.transcribe_audio(pcm_bytes)
                    display_title = "STT (Nguoi noi)"
                    display_text = text if text else "Khong nghe ro."
                    state.add("speech", display_text)
                    if args.esp_ip:
                        try:
                            import unicodedata
                            clean = unicodedata.normalize('NFKD', display_text).encode('ASCII', 'ignore')
                            http_session.post(f"http://{args.esp_ip}/oled", data=clean, timeout=2.0)
                        except: pass

                threading.Thread(target=transcribe_job, args=(bytes(audio_buffer),)).start()

        elif key == 27 or key == ord('q'): 
            shutdown_event.set()
            break

        # MediaPipe Vision - Chỉ xử lý khi có frame mới và đảm bảo timestamp strictly monotonic
        if is_new:
            now_ms = int(time.monotonic() * 1000)
            if now_ms <= last_mp_timestamp_ms:
                now_ms = last_mp_timestamp_ms + 1
            last_mp_timestamp_ms = now_ms

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            try:
                result = detector.detect_for_video(mp_image, now_ms)
            except Exception:
                result = None

            if is_capturing_sign and result and result.hand_landmarks:
                feats = frame_features(result)
                sign_sequence.append(feats)
        else:
            time.sleep(0.005)
        
        # Phóng to khung hình lên kích thước lớn (chuẩn HD/XGA) để cửa sổ to rõ ràng
        orig_h, orig_w = frame.shape[:2]
        target_w = max(orig_w, 960)
        target_h = int(orig_h * (target_w / orig_w))
        display_frame = cv2.resize(frame, (target_w, target_h), interpolation=cv2.INTER_LINEAR)

        # Vẽ khung xương tay trên hình ảnh đã phóng to
        if result and result.hand_landmarks:
            for landmarks in result.hand_landmarks:
                for idx_a, idx_b in HAND_CONNECTIONS:
                    pt_a = (int(landmarks[idx_a].x * target_w), int(landmarks[idx_a].y * target_h))
                    pt_b = (int(landmarks[idx_b].x * target_w), int(landmarks[idx_b].y * target_h))
                    cv2.line(display_frame, pt_a, pt_b, (0, 255, 0), 2)
                    cv2.circle(display_frame, pt_a, 4, (0, 0, 255), -1)
                    cv2.circle(display_frame, pt_b, 4, (0, 0, 255), -1)

        # Thanh hướng dẫn phía trên
        cv2.putText(display_frame, "[SPACE]: Thu Ky Hieu | [M]: Ghi Am | [Q]: Thoat", (25, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)
        
        if is_capturing_sign:
            cv2.circle(display_frame, (35, 78), 12, (0, 0, 255), -1)
            cv2.putText(display_frame, "DANG THU KY HIEU...", (58, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        if is_recording_audio:
            cv2.circle(display_frame, (35, 120), 12, (0, 165, 255), -1)
            cv2.putText(display_frame, "DANG THU MIC (INMP441)...", (58, 127), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)

        if display_title:
            display_frame = draw_vn_banner(display_frame, display_title, display_text)

        cv2.imshow(WINDOW_NAME, display_frame)

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()