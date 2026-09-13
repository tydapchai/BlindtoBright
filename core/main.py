import cv2
import time
import json
import torch
import numpy as np
import threading
import argparse
import requests
import subprocess # Thêm thư viện để chạy lệnh arp
from pathlib import Path
import mediapipe as mp

from config import state, shutdown_event
from models import GestureBiGRU, frame_features, prepare, HAND_CONNECTIONS
from mediapipe.tasks.python import vision
from gemini_api import GeminiClient
from web_app import run_web

http_session = requests.Session()

def draw_vn_banner(frame, title, text):
    cv2.putText(frame, f"{title}: {text}", (20, frame.shape[0] - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    return frame

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="best_bigru_v2.pt")
    parser.add_argument("--conversation", default="conversation_config.json")
    parser.add_argument("--hand-model", default="hand_landmarker.task")
    parser.add_argument("--camera", default="0")
    parser.add_argument("--esp-ip", default=None, help="IP của ESP32")
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
    # LOGIC AUTO-DISCOVER ESP32 IP
    # =========================================================================
    esp_ip = getattr(args, "esp_ip", None)
    if not esp_ip and isinstance(args.camera, str) and args.camera.startswith("http"):
        from urllib.parse import urlparse as _urlparse
        try: # Đã thêm khối try bị thiếu
            esp_ip = _urlparse(args.camera).hostname
        except Exception:
            pass

    def resolve_esp_ip(target_ip):
        if target_ip:
            try:
                r = http_session.get(f"http://{target_ip}/status", timeout=0.6)
                if r.status_code == 200 and "Blind to Bright" in r.text:
                    return target_ip
            except Exception:
                print(f"[Auto-Discover] IP '{target_ip}' không phản hồi, đang quét mạng LAN...")

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
                            print(f"[Auto-Discover] Đã tìm thấy ESP32 tại IP mới: {cand}")
                            return cand
                    except Exception:
                        pass
        except Exception as e:
            print(f"[Auto-Discover] Lỗi quét ARP: {e}")

        return target_ip

    resolved_ip = resolve_esp_ip(esp_ip)
    if resolved_ip and resolved_ip != esp_ip:
        print(f"[Auto-Discover] Cập nhật IP ESP32 từ {esp_ip} thành {resolved_ip}")
        esp_ip = resolved_ip
        args.esp_ip = resolved_ip # Lưu lại vào args để luồng Audio_fetch dùng chung
        if isinstance(args.camera, str) and args.camera.startswith("http"):
            args.camera = f"http://{esp_ip}:81/stream"
    # =========================================================================

    # Khởi tạo Camera và Gemini 
    source = int(args.camera) if str(args.camera).isdigit() else args.camera
    cap = cv2.VideoCapture(source)
    gemini = GeminiClient()

    # Phát lời chào sẵn sàng qua TTS Gemini
    ready_msg = "Hệ thống Blind to Bright đã sẵn sàng"
    def welcome_speaker():
        pcm = gemini.generate_speech(ready_msg)
        if args.esp_ip and pcm:
            try:
                requests.post(f"http://{args.esp_ip}/play", data=pcm, timeout=5)
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

    while not shutdown_event.is_set():
        ret, frame = cap.read()
        if not ret: continue
        key = cv2.waitKey(1) & 0xFF
        now_ms = int(time.monotonic() * 1000)

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
                        pcm = gemini.generate_speech(sentence)
                        if args.esp_ip and pcm:
                            try:
                                requests.post(f"http://{args.esp_ip}/play", data=pcm, timeout=5)
                            except: pass
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
                            requests.post(f"http://{args.esp_ip}/oled", data=clean)
                        except: pass

                threading.Thread(target=transcribe_job, args=(bytes(audio_buffer),)).start()

        elif key == 27 or key == ord('q'): 
            shutdown_event.set()
            break

        # MediaPipe Vision
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = detector.detect_for_video(mp_image, now_ms)
        
        if result and result.hand_landmarks:
            for landmarks in result.hand_landmarks:
                for idx_a, idx_b in HAND_CONNECTIONS:
                    pt_a = (int(landmarks[idx_a].x * frame.shape[1]), int(landmarks[idx_a].y * frame.shape[0]))
                    pt_b = (int(landmarks[idx_b].x * frame.shape[1]), int(landmarks[idx_b].y * frame.shape[0]))
                    cv2.line(frame, pt_a, pt_b, (0, 255, 0), 2)
            
            if is_capturing_sign:
                feats = frame_features(result)
                sign_sequence.append(feats)

        cv2.putText(frame, "[SPACE]: Thu Kieu Hieu | [M]: Ghi Am | [Q]: Thoat", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1)
        if is_capturing_sign:
            cv2.circle(frame, (30, 60), 10, (0, 0, 255), -1)
            cv2.putText(frame, "DANG QUAY", (50, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)
        if is_recording_audio:
            cv2.circle(frame, (30, 90), 10, (0, 165, 255), -1)
            cv2.putText(frame, "DANG THU MIC", (50, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,165,255), 2)

        if display_title:
            frame = draw_vn_banner(frame, display_title, display_text)

        cv2.imshow("Blind to Bright", frame)

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()