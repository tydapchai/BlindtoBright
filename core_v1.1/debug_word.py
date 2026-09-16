import argparse
import json
import os
import sys
import time
import warnings
from pathlib import Path
from urllib.parse import urlparse

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
warnings.filterwarnings("ignore")
try:
    sys.stdout.reconfigure(line_buffering=True)
    sys.stderr.reconfigure(line_buffering=True)
except Exception:
    pass

import cv2
import mediapipe as mp
import numpy as np
import requests
import torch
from PIL import Image, ImageDraw, ImageFont

ROOT_DIR = Path(__file__).resolve().parent.parent
CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))
if str(ROOT_DIR / "core") not in sys.path:
    sys.path.insert(0, str(ROOT_DIR / "core"))

from camera import LatestFrameCamera
from preprocess import build_tensor, extract_landmarks, motion_energy
from stgcn_model import load_stgcn_checkpoint

try:
    FONT_TITLE = ImageFont.truetype("arial.ttf", 18)
    FONT_TEXT = ImageFont.truetype("arial.ttf", 22)
    FONT_BOLD = ImageFont.truetype("arialbd.ttf", 19)
    FONT_HINT = ImageFont.truetype("arial.ttf", 15)
    FONT_SMALL = ImageFont.truetype("arial.ttf", 14)
except Exception:
    FONT_TITLE = ImageFont.load_default()
    FONT_TEXT = ImageFont.load_default()
    FONT_BOLD = ImageFont.load_default()
    FONT_HINT = ImageFont.load_default()
    FONT_SMALL = ImageFont.load_default()


def read_labels(path):
    with open(path, "r", encoding="utf-8") as file:
        raw = json.load(file)
    if isinstance(raw, list):
        return {index: value for index, value in enumerate(raw)}
    return {int(value): key for key, value in raw.items()}


def resolve_esp_ip(target_ip):
    session = requests.Session()
    session.trust_env = False

    # 1. Thử IP mục tiêu
    if target_ip:
        try:
            r = session.get(f"http://{target_ip}/status", timeout=1.2)
            if r.status_code == 200 and "Blind to Bright" in r.text:
                return target_ip
        except Exception:
            print(f"[ESP32] IP '{target_ip}' khong phan hoi qua Wi-Fi...")

    # 2. Kiểm tra mạng SoftAP của ESP (192.168.4.1)
    if target_ip != "192.168.4.1":
        try:
            r = session.get("http://192.168.4.1/status", timeout=0.8)
            if r.status_code == 200 and "Blind to Bright" in r.text:
                print("\n[ESP32 Fallback] Da FALLBACK sang mang SoftAP cua ESP: 192.168.4.1\n")
                return "192.168.4.1"
        except Exception:
            pass

    # 3. Quét ARP LAN
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

        ordered = esp_cands + other_cands
        def _check(ip):
            try:
                res = session.get(f"http://{ip}/status", timeout=0.5)
                if res.status_code == 200 and "Blind to Bright" in res.text:
                    return ip
            except Exception:
                pass
            return None

        with ThreadPoolExecutor(max_workers=20) as pool:
            for cand in pool.map(_check, ordered):
                if cand:
                    print(f"[ESP32 Auto-Discover] Tim thay ESP32 tai IP: {cand}")
                    return cand
    except Exception:
        pass

    fallback = target_ip or "192.168.4.1"
    return fallback


def draw_debug_overlay(display, is_recording, mode, frames_count, top5, target_word=None, target_info=None):
    h, w = display.shape[:2]

    # 1. Top Bar
    top_bar_h = 42
    overlay = display.copy()
    cv2.rectangle(overlay, (0, 0), (w, top_bar_h), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.75, display, 0.25, 0, display)
    cv2.line(display, (0, top_bar_h), (w, top_bar_h), (50, 50, 50), 1)

    # 2. Right Info Panel (Bảng Top 5)
    panel_w = 340
    panel_x = w - panel_w - 15
    panel_y = top_bar_h + 15
    panel_h = 320
    panel_overlay = display.copy()
    cv2.rectangle(panel_overlay, (panel_x, panel_y), (panel_x + panel_w, panel_y + panel_h), (15, 15, 15), -1)
    cv2.addWeighted(panel_overlay, 0.85, display, 0.15, 0, display)
    cv2.rectangle(display, (panel_x, panel_y), (panel_x + panel_w, panel_y + panel_h), (0, 215, 255), 2)

    # 3. Bottom Guide Bar
    bot_h = 40
    bot_y = h - bot_h
    bot_overlay = display.copy()
    cv2.rectangle(bot_overlay, (0, bot_y), (w, h), (18, 18, 18), -1)
    cv2.addWeighted(bot_overlay, 0.8, display, 0.2, 0, display)
    cv2.line(display, (0, bot_y), (w, bot_y), (0, 215, 255), 2)

    # Vẽ chữ bằng PIL
    pil_img = Image.fromarray(cv2.cvtColor(display, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_img)

    # Top Bar: Trạng thái & Chế độ
    if is_recording:
        draw.ellipse((18, 13, 30, 25), fill=(255, 40, 40))
        rec_str = f"ĐANG THU CỬ CHỈ ({frames_count} frames) - BẤM SPACE ĐỂ DỪNG"
        rec_color = (255, 80, 80)
    else:
        draw.ellipse((18, 13, 30, 25), fill=(40, 255, 40))
        rec_str = "SẴN SÀNG - BẤM [SPACE] ĐỂ BẮT ĐẦU THU CỬ CHỈ"
        rec_color = (80, 255, 80)
    draw.text((38, 10), rec_str, font=FONT_BOLD, fill=rec_color)

    mode_tag = f"Chế độ: {mode.upper()}"
    draw.text((w - 180, 11), mode_tag, font=FONT_HINT, fill=(200, 200, 200))

    # Right Panel Header
    draw.text((panel_x + 15, panel_y + 12), "TOP 5 DỰ ĐOÁN MÔ HÌNH", font=FONT_BOLD, fill=(0, 215, 255))
    draw.line([(panel_x + 15, panel_y + 36), (panel_x + panel_w - 15, panel_y + 36)], fill=(70, 70, 70), width=1)

    # Top 5 Danh Sách
    if top5:
        cur_y = panel_y + 45
        for rank, (word, conf) in enumerate(top5[:5], 1):
            rank_color = (255, 215, 0) if rank == 1 else ((200, 200, 200) if rank == 2 else (140, 140, 140))
            draw.text((panel_x + 15, cur_y), f"#{rank}", font=FONT_HINT, fill=rank_color)

            # Tên từ
            draw.text((panel_x + 45, cur_y), word[:14], font=FONT_HINT, fill=(245, 245, 245))

            # Phần trăm
            pct_str = f"{conf * 100:5.1f}%"
            draw.text((panel_x + panel_w - 75, cur_y), pct_str, font=FONT_HINT, fill=rank_color)

            # Thanh progress bar
            bar_x = panel_x + 45
            bar_y = cur_y + 20
            bar_w = panel_w - 65
            bar_h = 4
            draw.rectangle([bar_x, bar_y, bar_x + bar_w, bar_y + bar_h], fill=(45, 45, 45))
            fill_w = int(bar_w * min(1.0, max(0.0, conf)))
            fill_color = (0, 255, 120) if rank == 1 else (0, 180, 255)
            if fill_w > 0:
                draw.rectangle([bar_x, bar_y, bar_x + fill_w, bar_y + bar_h], fill=fill_color)

            cur_y += 35

        # Chênh lệch Top 1 - Top 2
        if len(top5) >= 2:
            diff = (top5[0][1] - top5[1][1]) * 100
            diff_color = (100, 255, 100) if diff >= 10.0 else (255, 160, 50)
            draw.text((panel_x + 15, panel_y + panel_h - 60), f"Chênh lệch #1 vượt #2: +{diff:.1f}%", font=FONT_SMALL, fill=diff_color)
    else:
        draw.text((panel_x + 25, panel_y + 90), "Chưa có dữ liệu cử chỉ.", font=FONT_HINT, fill=(150, 150, 150))
        draw.text((panel_x + 25, panel_y + 115), "Bấm [SPACE] để bắt đầu thu.", font=FONT_SMALL, fill=(110, 110, 110))

    # Hiển thị Target Word (nếu có)
    if target_word:
        draw.line([(panel_x + 15, panel_y + panel_h - 35), (panel_x + panel_w - 15, panel_y + panel_h - 35)], fill=(60, 60, 60), width=1)
        if target_info:
            tg_rank, tg_conf = target_info
            tg_str = f"Mục tiêu: '{target_word}' -> Hạng #{tg_rank} ({tg_conf*100:.1f}%)"
            tg_color = (100, 255, 100) if tg_rank == 1 else ((255, 215, 0) if tg_rank <= 3 else (255, 80, 80))
        else:
            tg_str = f"Mục tiêu: '{target_word}' (Chờ làm cử chỉ)"
            tg_color = (200, 200, 200)
        draw.text((panel_x + 15, panel_y + panel_h - 26), tg_str, font=FONT_SMALL, fill=tg_color)

    # Bottom Guide Bar
    guide_str = "[SPACE] Bắt đầu / Dừng thu & Phân tích  |  [M] Đổi chế độ  |  [C] Xóa kết quả  |  [Q] Thoát"
    draw.text((25, bot_y + 10), guide_str, font=FONT_HINT, fill=(220, 220, 220))

    display[:] = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)


def print_terminal_report(frames_count, duration, top5, target_word=None, target_info=None, mode="manual"):
    fps = frames_count / duration if duration > 0 else 0
    print("\n" + "=" * 72)
    print("  [KẾT QUẢ ĐO ĐỘ CHÍNH XÁC CỬ CHỈ ĐỘC LẬP]")
    print(f"  - Số khung hình : {frames_count} frames | Thời lượng: {duration:.2f}s ({fps:.1f} FPS)")
    print(f"  - Chế độ thu    : {mode.upper()}")
    if target_word and target_info:
        rank, conf = target_info
        status_txt = "PASS (ĐỨNG ĐẦU #1!)" if rank == 1 else (f"TOP {rank}" if rank <= 5 else "MISS")
        print(f"  - Từ mục tiêu   : '{target_word}' -> HẠNG #{rank} ({conf*100:.2f}%) [{status_txt}]")
    print("  " + "-" * 68)
    print(f"  {'Hạng':<6} {'Từ vựng':<24} {'Độ tin cậy':<12} {'Biểu đồ xác suất'}")
    print("  " + "-" * 68)

    for i, (word, conf) in enumerate(top5[:5], 1):
        bar_len = int(conf * 25)
        bar = "█" * bar_len + "░" * (25 - bar_len)
        star = " <-- MỤC TIÊU" if (target_word and word.lower() == target_word.lower()) else ""
        print(f"  #{i:<5} {word:<24} {conf*100:6.2f}%      [{bar}]{star}")

    print("  " + "-" * 68)
    if len(top5) >= 2:
        diff = (top5[0][1] - top5[1][1]) * 100
        print(f"  * Khoảng cách Top-1 so với Top-2: +{diff:.2f}%")
        if diff >= 15.0 and top5[0][1] >= 0.35:
            print("  * Đánh giá mô hình: RẤT RÕ RÀNG VÀ TỰ TIN (Clear winner)")
        elif diff >= 8.0:
            print("  * Đánh giá mô hình: ĐẠT NGƯỠNG (Acceptable)")
        else:
            print("  * Đánh giá mô hình: PHÂN VÂN / DỄ NHẦM LẪN (Ambiguous - Margin quá thấp)")
    print("=" * 72 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Công cụ Debug độ chính xác từng từ VSL ST-GCN v1.1")
    parser.add_argument("--checkpoint", default=str(ROOT_DIR / "models" / "best_vsl_model.pth"))
    parser.add_argument("--labels", default=str(ROOT_DIR / "core_v1.1" / "label_map_472.json"))
    parser.add_argument("--camera", default="esp", help="'esp', '0' (webcam), hoặc URL http")
    parser.add_argument("--esp-ip", default=None, help="IP thủ công của ESP32")
    parser.add_argument("--target", default=None, help="Từ mục tiêu bạn đang muốn test (ví dụ: 'Cảm ơn')")
    parser.add_argument("--device", default=None, choices=["cpu", "cuda"])
    parser.add_argument("--mode", default="manual", choices=["manual", "auto"], help="Chế độ thu: manual (bấm SPACE) hoặc auto (tự spotting)")
    parser.add_argument("--list", action="store_true", help="In ra danh sách 472 nhãn từ vựng rồi thoát")
    parser.add_argument("--search", default=None, help="Tìm kiếm nhãn theo từ khóa rồi thoát")
    args = parser.parse_args()

    labels_path = Path(args.labels)
    if not labels_path.exists():
        labels_path = ROOT_DIR / "core_v1.1" / "label_map_472.json"
    idx_to_class = read_labels(labels_path)

    # In danh sách hoặc tìm kiếm nhãn
    if args.list:
        print(f"\nTổng số nhãn: {len(idx_to_class)}")
        for idx in sorted(idx_to_class.keys()):
            print(f"[{idx:3d}] {idx_to_class[idx]}")
        return

    if args.search:
        kw = args.search.strip().lower()
        matches = [(idx, name) for idx, name in idx_to_class.items() if kw in name.lower()]
        print(f"\nKết quả tìm kiếm từ khóa '{args.search}' ({len(matches)} kết quả):")
        for idx, name in matches:
            print(f"  - [{idx:3d}] {name}")
        return

    # Khởi tạo mô hình
    device_name = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_name)
    checkpoint_path = Path(args.checkpoint)
    model, checkpoint, num_classes = load_stgcn_checkpoint(checkpoint_path, device)
    print(f"\n[Model] Đã tải ST-GCN Transformer | classes={num_classes} | device={device}")
    if "val_acc" in checkpoint:
        print(f"[Model] Checkpoint val_acc={checkpoint['val_acc']:.4f}")

    # Xử lý IP Camera ESP32
    if str(args.camera).lower() in ["esp", "esp32", "cam"]:
        ip_file = ROOT_DIR / "configs" / "esp_ip.txt"
        target_ip = args.esp_ip
        if not target_ip and ip_file.is_file():
            target_ip = ip_file.read_text(encoding="utf-8").strip() or None
        target_ip = target_ip or "10.3.79.128"
        actual_ip = resolve_esp_ip(target_ip)
        try:
            ip_file.parent.mkdir(parents=True, exist_ok=True)
            ip_file.write_text(actual_ip, encoding="utf-8")
        except Exception:
            pass
        source = f"http://{actual_ip}:81/stream"
        print(f"[Camera] Kết nối Camera ESP32: {source}")
    elif str(args.camera).isdigit():
        source = int(args.camera)
        print(f"[Camera] Sử dụng Webcam: {source}")
    else:
        source = args.camera
        print(f"[Camera] Sử dụng luồng: {source}")

    camera = LatestFrameCamera(source)

    print("[System] Đang khởi tạo MediaPipe Holistic...")
    detector = mp.solutions.holistic.Holistic(
        static_image_mode=False,
        model_complexity=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    mp_drawing = mp.solutions.drawing_utils
    mp_holistic = mp.solutions.holistic

    CANVAS_WIDTH = 960
    CANVAS_HEIGHT = 720
    WINDOW_NAME = "BlindtoBright - Word Accuracy Debugger"
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, CANVAS_WIDTH, CANVAS_HEIGHT)

    mode = args.mode  # 'manual' hoặc 'auto'
    target_word = args.target

    is_recording = False
    gesture_frames = []
    record_start_time = 0.0

    last_top5 = []
    last_target_info = None

    auto_idle_count = 0
    previous_landmarks = None
    last_frame_id = -1

    print("\n" + "=" * 65)
    print("  [DEBUGGER SẴN SÀNG]")
    print("  - Chế độ: THỦ CÔNG (Manual).")
    print("    -> Bấm [SPACE] lần 1: Bắt đầu thu cử chỉ.")
    print("    -> Làm cử chỉ.")
    print("    -> Bấm [SPACE] lần 2: Dừng thu & Phân tích Top 5 ngay lập tức!")
    print("  - Phím [M]: Đổi giữa THỦ CÔNG và TỰ ĐỘNG.")
    print("  - Phím [C]: Xóa kết quả phân tích cũ.")
    print("  - Phím [Q]: Thoát.")
    print("=" * 65 + "\n")

    try:
        while True:
            is_new, last_frame_id, frame = camera.read_latest(last_frame_id)
            if frame is None:
                splash = np.zeros((CANVAS_HEIGHT, CANVAS_WIDTH, 3), dtype=np.uint8)
                cv2.putText(splash, "Dang ket noi Camera...", (150, 360), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 215, 255), 2)
                cv2.imshow(WINDOW_NAME, splash)
                key = cv2.waitKey(30) & 0xFF
                if key == ord("q") or key == 27:
                    break
                continue

            key = cv2.waitKey(1) & 0xFF

            # [SPACE]: Bật/Tắt thu nhận cử chỉ trong Manual Mode
            if key == 32:
                if mode == "manual":
                    if not is_recording:
                        # Bắt đầu thu
                        is_recording = True
                        gesture_frames = []
                        record_start_time = time.time()
                        print("\n[REC] >>> BẮT ĐẦU THU CỬ CHỈ... Hãy làm động tác!")
                    else:
                        # Dừng thu và phân tích ngay lập tức
                        is_recording = False
                        duration = time.time() - record_start_time
                        print(f"[REC] >>> ĐÃ DỪNG THU! Tổng: {len(gesture_frames)} frames ({duration:.2f}s). Đang phân tích...")

                        if len(gesture_frames) >= 8:
                            input_tensor = build_tensor(gesture_frames, device)
                            with torch.no_grad():
                                logits = model(input_tensor)
                                probs = torch.softmax(logits, dim=-1)[0].cpu().numpy()

                            top_indices = np.argsort(probs)[::-1]
                            last_top5 = [(idx_to_class.get(int(idx), f"Class_{idx}"), float(probs[idx])) for idx in top_indices[:5]]

                            # Kiểm tra Target Word nếu có
                            last_target_info = None
                            if target_word:
                                for r, idx in enumerate(top_indices, 1):
                                    w_name = idx_to_class.get(int(idx), "")
                                    if w_name.lower() == target_word.lower():
                                        last_target_info = (r, float(probs[idx]))
                                        break

                            print_terminal_report(len(gesture_frames), duration, last_top5, target_word, last_target_info, mode="manual")
                        else:
                            print(f"[Cảnh báo] Số frames quá ít ({len(gesture_frames)} frames < 8). Cử chỉ bị hủy.")
                else:
                    # Auto mode: Reset
                    gesture_frames = []
                    is_recording = False

            # [M]: Đổi chế độ Manual <-> Auto
            elif key == ord("m") or key == ord("M"):
                mode = "auto" if mode == "manual" else "manual"
                is_recording = False
                gesture_frames = []
                print(f"\n[Mode] Đã chuyển sang chế độ: {mode.upper()}")

            # [C]: Xóa kết quả
            elif key == ord("c") or key == ord("C"):
                last_top5 = []
                last_target_info = None
                print("\n[Action] Đã xóa kết quả phân tích cũ.")

            # [Q] hoặc [ESC]: Thoát
            elif key == ord("q") or key == ord("Q") or key == 27:
                break

            if is_new:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = detector.process(rgb)
                landmarks = extract_landmarks(results)

                # Chuyển động
                if previous_landmarks is None:
                    motion = 0.0
                else:
                    motion = motion_energy(landmarks, previous_landmarks)
                previous_landmarks = landmarks

                hand_detected = bool(results.left_hand_landmarks or results.right_hand_landmarks)

                # =========================================================
                # XỬ LÝ THEO CHẾ ĐỘ
                # =========================================================
                if mode == "manual":
                    if is_recording:
                        gesture_frames.append(landmarks)
                else:
                    # Chế độ AUTO SPOTTING
                    if not is_recording:
                        if hand_detected and motion >= 0.005:
                            is_recording = True
                            gesture_frames = [landmarks]
                            record_start_time = time.time()
                            auto_idle_count = 0
                            print("\n[Auto-REC] >>> Phát hiện cử chỉ! Đang thu...")
                    else:
                        gesture_frames.append(landmarks)
                        if motion < 0.0035 or not hand_detected:
                            auto_idle_count += 1
                        else:
                            auto_idle_count = 0

                        if auto_idle_count >= 4 or len(gesture_frames) >= 75:
                            is_recording = False
                            duration = time.time() - record_start_time
                            valid_frames = gesture_frames[:-auto_idle_count] if auto_idle_count > 0 else gesture_frames
                            if len(valid_frames) >= 10:
                                input_tensor = build_tensor(valid_frames, device)
                                with torch.no_grad():
                                    logits = model(input_tensor)
                                    probs = torch.softmax(logits, dim=-1)[0].cpu().numpy()

                                top_indices = np.argsort(probs)[::-1]
                                last_top5 = [(idx_to_class.get(int(idx), f"Class_{idx}"), float(probs[idx])) for idx in top_indices[:5]]

                                last_target_info = None
                                if target_word:
                                    for r, idx in enumerate(top_indices, 1):
                                        w_name = idx_to_class.get(int(idx), "")
                                        if w_name.lower() == target_word.lower():
                                            last_target_info = (r, float(probs[idx]))
                                            break

                                print_terminal_report(len(valid_frames), duration, last_top5, target_word, last_target_info, mode="auto")
                            gesture_frames = []

                # Resize lên Canvas 960x720
                display = cv2.resize(frame, (CANVAS_WIDTH, CANVAS_HEIGHT), interpolation=cv2.INTER_LINEAR)

                # Vẽ xương khớp
                if results.left_hand_landmarks:
                    mp_drawing.draw_landmarks(
                        display, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                        mp_drawing.DrawingSpec(color=(121, 22, 76), thickness=2, circle_radius=2),
                        mp_drawing.DrawingSpec(color=(121, 44, 250), thickness=2, circle_radius=1),
                    )
                if results.right_hand_landmarks:
                    mp_drawing.draw_landmarks(
                        display, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                        mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
                        mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=1),
                    )
                if results.pose_landmarks:
                    mp_drawing.draw_landmarks(
                        display, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                        mp_drawing.DrawingSpec(color=(80, 22, 10), thickness=2, circle_radius=2),
                        mp_drawing.DrawingSpec(color=(80, 44, 121), thickness=2, circle_radius=1),
                    )

                draw_debug_overlay(
                    display,
                    is_recording,
                    mode,
                    len(gesture_frames),
                    last_top5,
                    target_word=target_word,
                    target_info=last_target_info,
                )

                cv2.imshow(WINDOW_NAME, display)
            else:
                time.sleep(0.005)

    finally:
        camera.release()
        detector.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
