import cv2
import time
import json
import torch
import numpy as np
import threading
import argparse
import requests
from pathlib import Path
import mediapipe as mp # Đã bổ sung import

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
        base_options=mp.tasks.BaseOptions(model_asset_path=args.hand_model), # Đã sửa cú pháp
        running_mode=vision.RunningMode.VIDEO, num_hands=2
    )
    detector = vision.HandLandmarker.create_from_options(options)
    cap = cv2.VideoCapture(int(args.camera) if args.camera.isdigit() else args.camera)
    gemini = GeminiClient()

    is_recording_audio = False
    is_capturing_sign = False
    audio_buffer = bytearray()
    sign_sequence = []
    display_text, display_title = "", ""

    # ==============================================================
    # VÁ LỖI 1: THÊM LUỒNG HÚT ÂM THANH TỪ ESP32
    # ==============================================================
    def audio_fetch_worker():
        nonlocal audio_buffer, is_recording_audio
        while not shutdown_event.is_set():
            if is_recording_audio and args.esp_ip:
                try:
                    # Gọi endpoint stream âm thanh của ESP32 (port 82 là ví dụ từ code cũ)
                    resp = http_session.get(f"http://{args.esp_ip}:82/mic", stream=True, timeout=2.0)
                    if resp.status_code == 200:
                        for chunk in resp.iter_content(chunk_size=1024):
                            if not is_recording_audio or shutdown_event.is_set():
                                break
                            if chunk:
                                audio_buffer.extend(chunk)
                except Exception as e:
                    time.sleep(0.5)
            else:
                time.sleep(0.1)
                
    threading.Thread(target=audio_fetch_worker, daemon=True).start()
    # ==============================================================

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
                audio_buffer = bytearray() # Reset buffer khi bắt đầu thu
                display_title = "DANG THU AM..."
                display_text = "Nguoi doi dien hay noi..."
            else:
                display_title = "DANG DICH..."
                display_text = "Cho Gemini xu ly..."
                
                def transcribe_job(pcm_bytes):
                    nonlocal display_title, display_text
                    if len(pcm_bytes) < 4000: # Tránh gửi file âm thanh trống
                        display_title, display_text = "STT", "Khong nghe gi"
                        return
                    text = gemini.transcribe_audio(pcm_bytes)
                    display_title = "STT (Nguoi noi)"
                    display_text = text if text else "Khong nghe ro."
                    state.add("speech", display_text)
                    if args.esp_ip:
                        try:
                            # Encode ascii để ESP32 OLED không bị lỗi font nếu chưa có font TV
                            import unicodedata
                            clean = unicodedata.normalize('NFKD', display_text).encode('ASCII', 'ignore')
                            requests.post(f"http://{args.esp_ip}/oled", data=clean)
                        except: pass

                threading.Thread(target=transcribe_job, args=(bytes(audio_buffer),)).start()

        elif key == 27 or key == ord('q'): 
            shutdown_event.set()
            break

        # Sửa cú pháp truyền ảnh vào MediaPipe
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