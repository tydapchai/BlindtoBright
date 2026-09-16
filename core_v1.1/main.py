import argparse
import json
import queue
import sys
import threading
import time
from urllib.parse import urlparse
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
import requests
import torch

ROOT_DIR = Path(__file__).resolve().parent.parent
CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))
if str(ROOT_DIR / "core") not in sys.path:
    sys.path.insert(0, str(ROOT_DIR / "core"))

from camera import LatestFrameCamera
from decoder import TemporalDecoder
from gemini_api import GeminiClient
from preprocess import build_tensor, extract_landmarks, motion_energy, new_buffer
from stgcn_model import load_stgcn_checkpoint


def read_labels(path):
    with open(path, "r", encoding="utf-8") as file:
        raw = json.load(file)
    if isinstance(raw, list):
        return {index: value for index, value in enumerate(raw)}
    return {int(value): key for key, value in raw.items()}


def resolve_path(value, fallback):
    path = Path(value) if value else fallback
    if not path.exists():
        raise FileNotFoundError(f"Khong tim thay file: {path}")
    return path


def draw_status(frame, title, text, active):
    height, width = frame.shape[:2]
    color = (0, 0, 255) if active else (0, 215, 255)
    cv2.putText(frame, "[SPACE] Bat/tat | [Q] Thoat", (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    if active:
        cv2.circle(frame, (30, 70), 10, color, -1)
        cv2.putText(frame, "DANG NHAN DIEN ST-GCN", (52, 77),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)
    if title:
        banner_top = height - 72
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, banner_top), (width, height), (15, 15, 15), -1)
        cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)
        cv2.putText(frame, title, (20, banner_top + 27),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 215, 255), 2)
        cv2.putText(frame, text, (20, height - 17),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)


def main():
    parser = argparse.ArgumentParser(description="BlindtoBright ST-GCN Transformer v1.1")
    parser.add_argument("--checkpoint", default=str(ROOT_DIR / "models" / "best_vsl_model.pth"))
    parser.add_argument(
        "--labels",
        default=str(ROOT_DIR.parent / "VSL_pipeline" / "modules" / "label_map_472.json"),
    )
    parser.add_argument("--camera", default="0")
    parser.add_argument("--esp-ip", default=None)
    parser.add_argument("--gemini-key", default=None)
    parser.add_argument("--no-tts", action="store_true")
    parser.add_argument("--stride", type=int, default=4)
    parser.add_argument("--device", default=None, choices=["cpu", "cuda"])
    args = parser.parse_args()

    if not args.esp_ip and isinstance(args.camera, str) and args.camera.startswith("http"):
        args.esp_ip = urlparse(args.camera).hostname
    if not args.esp_ip:
        ip_file = ROOT_DIR / "configs" / "esp_ip.txt"
        if ip_file.is_file():
            args.esp_ip = ip_file.read_text(encoding="utf-8").strip() or None

    device_name = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_name)
    checkpoint_path = resolve_path(args.checkpoint, Path(args.checkpoint))
    labels_path = resolve_path(args.labels, Path(args.labels))
    model, checkpoint, num_classes = load_stgcn_checkpoint(checkpoint_path, device)
    idx_to_class = read_labels(labels_path)
    if len(idx_to_class) != num_classes:
        raise ValueError(f"So nhan ({len(idx_to_class)}) khac output model ({num_classes})")

    gemini = GeminiClient(api_keys=args.gemini_key)

    print(f"[Model] ST-GCN Transformer | classes={num_classes} | device={device}")
    if args.esp_ip:
        print(f"[ESP32] TTS/OLED endpoint: {args.esp_ip}")
    if "val_acc" in checkpoint:
        print(f"[Model] checkpoint val_acc={checkpoint['val_acc']:.4f}")

    source = int(args.camera) if str(args.camera).isdigit() else args.camera
    camera = LatestFrameCamera(source)
    detector = mp.solutions.holistic.Holistic(
        static_image_mode=False,
        model_complexity=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    decoder = TemporalDecoder(idx_to_class)
    frame_buffer = new_buffer()
    recognition_enabled = False
    frame_counter = 0
    last_frame_id = -1
    previous_landmarks = None
    title, text = "", ""
    speech_session = requests.Session()
    speech_session.trust_env = False
    sentence_queue = queue.Queue()

    def llm_tts_worker():
        while True:
            glosses = sentence_queue.get()
            if glosses is None:
                sentence_queue.task_done()
                return
            try:
                raw_sentence = " ".join(glosses)
                final_sentence = gemini.rewrite_signs(glosses)
                title_holder[0] = "DICH HOAN CHINH"
                text_holder[0] = final_sentence or raw_sentence
                print(f"[LLM] {raw_sentence} -> {text_holder[0]}")
                if not args.no_tts:
                    pcm = gemini.generate_speech(text_holder[0])
                    if args.esp_ip and pcm:
                        speech_session.post(
                            f"http://{args.esp_ip}/play", data=pcm, timeout=10
                        )
            except requests.RequestException as error:
                print(f"[TTS] Không gửi được âm thanh tới ESP32: {error}")
            except Exception as error:
                print(f"[LLM/TTS] Lỗi xử lý câu: {error}")
            finally:
                sentence_queue.task_done()

    title_holder = [title]
    text_holder = [text]
    threading.Thread(target=llm_tts_worker, name="llm-tts-worker", daemon=True).start()

    try:
        while True:
            is_new, last_frame_id, frame = camera.read_latest(last_frame_id)
            if frame is None:
                time.sleep(0.01)
                continue

            key = cv2.waitKey(1) & 0xFF
            if key == 32:
                recognition_enabled = not recognition_enabled
                frame_buffer.clear()
                decoder.reset()
                previous_landmarks = None
                title = "SAN SANG" if recognition_enabled else "TAM DUNG"
                text = "Bat dau thu ky hieu" if recognition_enabled else "Da xoa bo dem"
            elif key == ord("q") or key == 27:
                break

            if is_new:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = detector.process(rgb)
                landmarks = extract_landmarks(results)
                hand_detected = bool(results.left_hand_landmarks or results.right_hand_landmarks)

                if recognition_enabled:
                    frame_buffer.append(landmarks)
                    frame_counter += 1
                    if previous_landmarks is None:
                        current_motion = 0.0
                    else:
                        current_motion = motion_energy(landmarks, previous_landmarks)
                    previous_landmarks = landmarks

                    if len(frame_buffer) == 48 and frame_counter % max(1, args.stride) == 0:
                        input_tensor = build_tensor(list(frame_buffer), device)
                        with torch.no_grad():
                            logits = model(input_tensor)
                            probabilities = torch.softmax(logits, dim=-1)[0].cpu().numpy()
                        new_word, sentence = decoder.process(
                            probabilities, current_motion, hand_detected
                        )
                        if new_word:
                            title, text = "NHAN DIEN", new_word
                        if sentence:
                            title = "DANG XU LY LLM"
                            text = " ".join(sentence)
                            title_holder[0] = title
                            text_holder[0] = text
                            sentence_queue.put(sentence)
                            if args.esp_ip:
                                try:
                                    speech_session.post(
                                        f"http://{args.esp_ip}/oled",
                                        data=text.encode("ascii", "ignore"),
                                        timeout=2,
                                    )
                                except requests.RequestException:
                                    pass

                title = title_holder[0]
                text = text_holder[0]

                display = frame.copy()
                if results.pose_landmarks:
                    for point in results.pose_landmarks.landmark:
                        x = int(point.x * display.shape[1])
                        y = int(point.y * display.shape[0])
                        if 0 <= x < display.shape[1] and 0 <= y < display.shape[0]:
                            cv2.circle(display, (x, y), 2, (0, 255, 0), -1)
                draw_status(display, title, text, recognition_enabled)
                cv2.imshow("BlindtoBright ST-GCN v1.1", display)
    finally:
        camera.release()
        detector.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()