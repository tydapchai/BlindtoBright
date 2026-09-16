import argparse
import json
import os
import queue
import sys
import threading
import time
import warnings
from urllib.parse import urlparse
from pathlib import Path

# Cấu hình log và buffer stdout ngay lập tức
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
from decoder import TemporalDecoder
from gemini_api import GeminiClient
from preprocess import build_tensor, extract_landmarks, motion_energy, new_buffer
from stgcn_model import load_stgcn_checkpoint

# Tải font Arial với kích cỡ chuẩn cho canvas 960x720
try:
    FONT_TITLE = ImageFont.truetype("arial.ttf", 18)
    FONT_TEXT = ImageFont.truetype("arial.ttf", 23)
    FONT_HINT = ImageFont.truetype("arial.ttf", 15)
except Exception:
    FONT_TITLE = ImageFont.load_default()
    FONT_TEXT = ImageFont.load_default()
    FONT_HINT = ImageFont.load_default()


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


def draw_status(
    display,
    title,
    text,
    active,
    sentence_buffer=None,
    top_pred="",
    top_conf=0.0,
    hand_ok=False,
    top3_list=None,
    signing_len=0,
):
    height, width = display.shape[:2]  # Canvas 720 x 960
    top_bar_height = 42
    banner_height = 112

    overlay = display.copy()
    cv2.rectangle(overlay, (0, 0), (width, top_bar_height), (18, 18, 18), -1)
    banner_top = height - banner_height
    cv2.rectangle(overlay, (0, banner_top), (width, height), (15, 15, 15), -1)
    cv2.addWeighted(overlay, 0.82, display, 0.18, 0, display)

    if signing_len > 0:
        status_color = (0, 215, 255)  # Vàng cam khi đang thu nhận cử chỉ
    elif active:
        status_color = (0, 255, 0)    # Xanh lá khi sẵn sàng
    else:
        status_color = (0, 0, 255)    # Đỏ khi tạm dừng

    cv2.line(display, (0, top_bar_height), (width, top_bar_height), (45, 45, 45), 1)
    cv2.line(display, (0, banner_top), (width, banner_top), (0, 215, 255), 2)
    cv2.circle(display, (20, 21), 7, status_color, -1)

    try:
        pil_img = Image.fromarray(cv2.cvtColor(display, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(pil_img)

        # Cột 1: Chế độ nhận diện
        if signing_len > 0:
            mode_str = f"ĐANG THU NHẬN ({signing_len}f)"
            mode_color = (255, 215, 0)
        elif active:
            mode_str = "SẴN SÀNG"
            mode_color = (100, 255, 100)
        else:
            mode_str = "TẠM DỪNG"
            mode_color = (255, 160, 50)
        draw.text((36, 11), mode_str, font=FONT_HINT, fill=mode_color)

        # Cột 2: HUD Dự đoán thời gian thực & Cảm biến tay
        if signing_len > 0:
            hud_str = "Đang làm cử chỉ... Dừng/hạ tay để chốt từ"
            draw.text((230, 11), hud_str, font=FONT_HINT, fill=(255, 230, 100))
        elif top3_list and len(top3_list) >= 2 and active:
            w1, c1 = top3_list[0]
            w2, c2 = top3_list[1]
            hud_str = f"Top: 1. {w1} ({c1*100:.0f}%) | 2. {w2} ({c2*100:.0f}%) | Tay: {'OK' if hand_ok else 'Chưa'}"
            draw.text((215, 11), hud_str, font=FONT_HINT, fill=(240, 240, 240))
        elif active and top_pred:
            hud_str = f"Dự đoán: {top_pred} ({top_conf*100:.0f}%) | Tay: {'OK' if hand_ok else 'Chưa'}"
            draw.text((215, 11), hud_str, font=FONT_HINT, fill=(240, 240, 240))
        elif active:
            draw.text((215, 11), f"Tay: {'OK' if hand_ok else 'Chưa thấy'}", font=FONT_HINT, fill=(180, 180, 180))

        # Cột 3: Phím tắt điều khiển ở top bar
        draw.text((width - 240, 11), "[SPACE] Bật/Tắt | [Q] Thoát", font=FONT_HINT, fill=(190, 190, 190))

        # Banner dưới Dòng 1: Tiêu đề trạng thái & Badge số từ
        if title:
            draw.text((25, banner_top + 8), title.upper(), font=FONT_TITLE, fill=(0, 215, 255))

        if sentence_buffer:
            badge_text = f"Đang có {len(sentence_buffer)} từ: {' -> '.join(sentence_buffer[-3:])}"
            draw.text((width - 400, banner_top + 10), badge_text, font=FONT_HINT, fill=(255, 215, 0))

        # Banner dưới Dòng 2: Nội dung chính
        if text:
            display_text = text if len(text) <= 62 else text[:59] + "..."
            draw.text((25, banner_top + 38), display_text, font=FONT_TEXT, fill=(255, 255, 255))

        # Banner dưới Dòng 3: Hướng dẫn ngắt câu thủ công & thao tác
        hint_str = "[ENTER] Chốt câu & Đọc loa   •   [BACKSPACE] Xóa từ vừa nhận   •   [C] Xóa cả câu   •   [SPACE] Bật/Tắt"
        draw.text((25, banner_top + 80), hint_str, font=FONT_HINT, fill=(180, 180, 180))

        display[:] = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    except Exception:
        cv2.putText(display, mode_str, (36, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 1)
        if title:
            cv2.putText(display, title, (25, banner_top + 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 215, 255), 2)
        if text:
            cv2.putText(display, text[:50], (25, banner_top + 65), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(display, "[ENTER] Chot cau | [BACKSPACE] Xoa tu | [C] Xoa cau", (25, height - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)


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
    parser = argparse.ArgumentParser(description="BlindtoBright ST-GCN Transformer v1.1")
    parser.add_argument("--checkpoint", default=str(ROOT_DIR / "models" / "best_vsl_model.pth"))
    parser.add_argument(
        "--labels",
        default=str(ROOT_DIR / "core_v1.1" / "label_map_472.json"),
    )
    parser.add_argument("--camera", default="0")
    parser.add_argument("--esp-ip", default=None)
    parser.add_argument("--gemini-key", default=None)
    parser.add_argument("--no-tts", action="store_true")
    parser.add_argument("--stride", type=int, default=4)
    parser.add_argument("--device", default=None, choices=["cpu", "cuda"])
    parser.add_argument("--conf", "--confidence", type=float, default=0.30, help="Ngưỡng độ tin cậy nhận diện từ (0.1 - 1.0, mặc định 0.30)")
    parser.add_argument("--margin", type=float, default=0.08, help="Ngưỡng chênh lệch giữa Top-1 và Top-2 để chấp nhận từ (mặc định 0.08)")
    parser.add_argument("--min-frames", type=int, default=10, help="Số frame tối thiểu cho 1 cử chỉ hợp lệ (mặc định 10)")
    parser.add_argument("--max-frames", type=int, default=50, help="Số frame tối đa cho 1 cử chỉ (mặc định 50)")
    parser.add_argument("--idle-stop", type=int, default=3, help="Số frame dừng/hạ tay để chốt cử chỉ (mặc định 3)")
    parser.add_argument("--motion-start", type=float, default=0.008, help="Ngưỡng vận động bắt đầu cử chỉ (mặc định 0.008)")
    parser.add_argument("--motion-stop", type=float, default=0.007, help="Ngưỡng vận động dừng cử chỉ (mặc định 0.007)")
    parser.add_argument("--pause-on-start", action="store_true", help="Bắt đầu ở trạng thái tạm dừng thay vì tự động nhận diện")
    args = parser.parse_args()

    # Cấu hình IP ESP32
    if not args.esp_ip and isinstance(args.camera, str) and args.camera.startswith("http"):
        args.esp_ip = urlparse(args.camera).hostname
    if not args.esp_ip:
        ip_file = ROOT_DIR / "configs" / "esp_ip.txt"
        if ip_file.is_file():
            args.esp_ip = ip_file.read_text(encoding="utf-8").strip() or None
    if not args.esp_ip:
        args.esp_ip = "10.3.79.128"

    # Kiểm tra IP và TỰ ĐỘNG FALLBACK về mạng của ESP (SoftAP 192.168.4.1) khi không vào được Wi-Fi
    should_resolve = (
        str(args.camera).lower() in ["esp", "esp32", "cam"]
        or (args.esp_ip and not str(args.camera).isdigit())
        or (not args.no_tts)
    )
    if should_resolve and args.esp_ip:
        args.esp_ip = resolve_esp_ip(args.esp_ip)

    # Lưu lại IP để các lần sau tiện sử dụng
    try:
        ip_file = ROOT_DIR / "configs" / "esp_ip.txt"
        ip_file.parent.mkdir(parents=True, exist_ok=True)
        ip_file.write_text(args.esp_ip, encoding="utf-8")
    except Exception:
        pass

    if str(args.camera).lower() in ["esp", "esp32", "cam"]:
        args.camera = f"http://{args.esp_ip}:81/stream"
        print(f"[Camera] Đang sử dụng Camera ESP32: {args.camera}")

    device_name = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_name)
    checkpoint_path = resolve_path(args.checkpoint, Path(args.checkpoint))
    labels_path = resolve_path(args.labels, Path(args.labels))
    model, checkpoint, num_classes = load_stgcn_checkpoint(checkpoint_path, device)
    idx_to_class = read_labels(labels_path)
    if len(idx_to_class) != num_classes:
        raise ValueError(f"Số nhãn ({len(idx_to_class)}) khác output model ({num_classes})")

    gemini = GeminiClient(api_keys=args.gemini_key)

    print(f"[Model] ST-GCN Transformer | classes={num_classes} | device={device} | Ngưỡng tin cậy={args.conf}")
    if args.esp_ip:
        print(f"[ESP32] TTS/OLED endpoint: {args.esp_ip}")
    if "val_acc" in checkpoint:
        print(f"[Model] Checkpoint val_acc={checkpoint['val_acc']:.4f}")

    source = int(args.camera) if str(args.camera).isdigit() else args.camera
    camera = LatestFrameCamera(source)

    # Kích thước Canvas chuẩn 960x720 để đảm bảo giao diện luôn rộng rãi, sắc nét và không bao giờ bị tràn chữ
    CANVAS_WIDTH = 960
    CANVAS_HEIGHT = 720
    WINDOW_NAME = "BlindtoBright ST-GCN v1.1"
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, CANVAS_WIDTH, CANVAS_HEIGHT)

    splash = np.zeros((CANVAS_HEIGHT, CANVAS_WIDTH, 3), dtype=np.uint8)
    cv2.putText(splash, "BlindtoBright ST-GCN v1.1", (100, 300),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 215, 255), 2)
    cv2.putText(splash, "Dang ket noi camera...", (100, 360),
                cv2.FONT_HERSHEY_SIMPLEX, 0.85, (255, 255, 255), 2)
    cv2.putText(splash, f"Nguon: {source}", (100, 410),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (180, 180, 180), 1)
    cv2.imshow(WINDOW_NAME, splash)
    cv2.waitKey(50)

    print("[System] Đang khởi tạo MediaPipe Holistic...")
    detector = mp.solutions.holistic.Holistic(
        static_image_mode=False,
        model_complexity=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    mp_drawing = mp.solutions.drawing_utils
    mp_holistic = mp.solutions.holistic

    decoder = TemporalDecoder(
        idx_to_class,
        confidence_threshold=args.conf,
        margin_threshold=args.margin,
        motion_threshold=0.004,
        max_idle_steps=8,
    )
    frame_buffer = new_buffer()
    recognition_enabled = not args.pause_on_start
    frame_counter = 0
    last_frame_id = -1
    previous_landmarks = None

    # Biến trạng thái cho Gesture Spotting (Phát hiện biên cử chỉ tự động)
    is_signing = False
    gesture_frames = []
    signing_idle_count = 0
    recent_top3 = []

    active_title = ""
    active_text = ""
    active_text_time = 0.0

    top_prediction = ""
    top_confidence = 0.0

    speech_session = requests.Session()
    speech_session.trust_env = False
    sentence_queue = queue.Queue()

    llm_lock = threading.Lock()
    llm_state = {
        "title": "",
        "text": "",
        "updated_at": 0.0
    }

    def llm_tts_worker():
        while True:
            glosses = sentence_queue.get()
            if glosses is None:
                sentence_queue.task_done()
                return
            try:
                raw_sentence = " ".join(glosses)
                print(f"\n[LLM] Đang xử lý chuỗi ký hiệu: {raw_sentence} ...")
                final_sentence = gemini.rewrite_signs(glosses)
                result_text = final_sentence or raw_sentence
                with llm_lock:
                    llm_state["title"] = "DỊCH HOÀN CHỈNH"
                    llm_state["text"] = result_text
                    llm_state["updated_at"] = time.time()
                print(f"[LLM] Kết quả dịch: '{raw_sentence}' -> '{result_text}'")

                if args.esp_ip:
                    try:
                        speech_session.post(
                            f"http://{args.esp_ip}/oled",
                            data=result_text.encode("utf-8"),
                            timeout=2,
                        )
                    except requests.RequestException:
                        pass

                if not args.no_tts:
                    print(f"[TTS] Đang tạo giọng nói cho câu dịch...")
                    pcm = gemini.generate_speech(result_text)
                    if args.esp_ip and pcm:
                        print(f"[TTS] Đang phát âm thanh qua Loa ESP32: http://{args.esp_ip}/play ...")
                        speech_session.post(
                            f"http://{args.esp_ip}/play", data=pcm, timeout=10
                        )
                        print(f"[TTS] Đã phát âm thanh xong!")
            except requests.RequestException as error:
                print(f"[TTS] Không gửi được âm thanh tới ESP32: {error}")
            except Exception as error:
                print(f"[LLM/TTS] Lỗi xử lý câu: {error}")
            finally:
                sentence_queue.task_done()

    threading.Thread(target=llm_tts_worker, name="llm-tts-worker", daemon=True).start()

    system_ready_printed = False

    try:
        while True:
            is_new, last_frame_id, frame = camera.read_latest(last_frame_id)
            if frame is None:
                splash = np.zeros((CANVAS_HEIGHT, CANVAS_WIDTH, 3), dtype=np.uint8)
                cv2.putText(splash, "BlindtoBright ST-GCN v1.1", (100, 300),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 215, 255), 2)
                cv2.putText(splash, "Dang ket noi camera...", (100, 360),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.85, (255, 255, 255), 2)
                cv2.putText(splash, f"Nguon: {source}", (100, 410),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.65, (180, 180, 180), 1)
                cv2.imshow(WINDOW_NAME, splash)
                key = cv2.waitKey(30) & 0xFF
                if key == ord("q") or key == 27:
                    break
                time.sleep(0.02)
                continue

            if not system_ready_printed:
                system_ready_printed = True
                print("\n" + "=" * 70)
                print("  [BlindtoBright v1.1] HỆ THỐNG ĐÃ KHỞI ĐỘNG XONG & SẴN SÀNG!")
                print(f"  - Nguồn Camera: {source}")
                print(f"  - Cửa sổ hiển thị: '{WINDOW_NAME}' (960x720)")
                print(f"  - Nhận diện ban đầu: {'BẬT (đang nhận)' if recognition_enabled else 'TẮT (SPACE để bật)'}")
                print(f"  - Ngưỡng tin cậy từ: {args.conf * 100:.0f}% (Margin: {args.margin * 100:.0f}%)")
                print("  " + "-" * 66)
                print("  HƯỚNG DẪN ĐIỀU KHIỂN GHÉP CÂU:")
                print("  • Nhận diện từ:     Giơ tay làm cử chỉ -> Dừng/hạ tay để tự động chốt từ.")
                print("  • Phím [ENTER]:     CHỐT CÂU THỦ CÔNG -> Gửi Gemini LLM dịch & Phát loa ESP32.")
                print("  • Phím [BACKSPACE]: XÓA TỪ CUỐI CÙNG (nếu nhận diện nhầm).")
                print("  • Phím [C]:         XÓA TOÀN BỘ CÂU đang ghép để làm lại câu mới.")
                print("  • Phím [SPACE]:     Bật / Tạm dừng nhận diện.")
                print("  • Phím [Q] / [ESC]: Thoát chương trình.")
                print("=" * 70 + "\n")

            raw_key = cv2.waitKey(1)
            if raw_key != -1:
                key = raw_key & 0xFF

                # [ENTER]: Chốt câu thủ công -> Gửi Gemini LLM hoàn thiện ngữ pháp -> Phát loa ESP32
                if key in (13, 10):
                    sentence = decoder.commit_sentence()
                    if sentence:
                        raw_text = " ".join(sentence)
                        active_title = f"CHỐT CÂU ({len(sentence)} TỪ) -> ĐANG DỊCH..."
                        active_text = raw_text
                        active_text_time = time.time()
                        print("\n" + "=" * 65)
                        print(f"[User Action] >>> [ENTER] ĐÃ CHỐT CÂU: '{raw_text}' ({len(sentence)} từ)")
                        print(f"              -> Gửi Gemini LLM hoàn thiện ngữ pháp & Phát loa ESP32...")
                        print("=" * 65)
                        sentence_queue.put(sentence)
                        if args.esp_ip:
                            try:
                                speech_session.post(
                                    f"http://{args.esp_ip}/oled",
                                    data=raw_text.encode("utf-8", errors="ignore"),
                                    timeout=1,
                                )
                            except requests.RequestException:
                                pass
                    else:
                        active_title = "CHƯA CÓ TỪ NÀO"
                        active_text = "Câu đang trống! Hãy làm cử chỉ trước rồi mới bấm [ENTER] để chốt."
                        active_text_time = time.time()
                        print("\n[System] >>> Chưa có từ nào trong câu. Hãy thực hiện cử chỉ trước!")

                # [BACKSPACE]: Xóa từ cuối cùng vừa nhận diện nếu bị sai (phím 8 hoặc 127)
                elif key == 8 or key == 127:
                    removed = decoder.remove_last_word()
                    if removed:
                        current_buf = " ".join(decoder.sentence_buffer)
                        active_title = "ĐÃ XÓA TỪ VỪA NHẬN"
                        active_text = f"Đã xóa: '{removed}'. Câu còn lại: {current_buf or '(trống)'}"
                        active_text_time = time.time()
                        print(f"\n[User Action] >>> [BACKSPACE] Đã xóa từ: '{removed}'. Câu hiện tại: {current_buf or '(trống)'}")
                        if args.esp_ip:
                            try:
                                speech_session.post(
                                    f"http://{args.esp_ip}/oled",
                                    data=(current_buf or " ").encode("utf-8", errors="ignore"),
                                    timeout=1,
                                )
                            except requests.RequestException:
                                pass
                    else:
                        print("\n[System] >>> Câu đang trống, không có từ nào để xóa.")

                # [C]: Xóa toàn bộ câu đang ghép để làm lại câu mới
                elif key in (ord("c"), ord("C")):
                    if decoder.sentence_buffer:
                        old_sentence = " ".join(decoder.sentence_buffer)
                        decoder.clear_sentence()
                        active_title = "ĐÃ HỦY CÂU"
                        active_text = f"Đã xóa câu: '{old_sentence}'. Sẵn sàng ghép câu mới."
                        active_text_time = time.time()
                        print(f"\n[User Action] >>> [C] ĐÃ XÓA TOÀN BỘ CÂU: '{old_sentence}'")
                        if args.esp_ip:
                            try:
                                speech_session.post(
                                    f"http://{args.esp_ip}/oled",
                                    data=" ".encode("utf-8"),
                                    timeout=1,
                                )
                            except requests.RequestException:
                                pass
                    else:
                        print("\n[System] >>> Câu đang trống.")

                # [SPACE]: Bật/Tắt nhận diện cử chỉ
                elif key == 32:
                    recognition_enabled = not recognition_enabled
                    frame_buffer.clear()
                    gesture_frames.clear()
                    is_signing = False
                    signing_idle_count = 0
                    recent_top3.clear()
                    decoder.reset()
                    previous_landmarks = None
                    top_prediction = ""
                    top_confidence = 0.0
                    if recognition_enabled:
                        active_title = "SẴN SÀNG"
                        active_text = "Đưa tay lên thực hiện cử chỉ..."
                        active_text_time = time.time()
                        print("[System] >>> ĐÃ BẬT nhận diện ký hiệu (SPACE để tạm dừng)")
                    else:
                        active_title = "TẠM DỪNG"
                        active_text = "Đã tạm dừng nhận diện. Bấm SPACE để tiếp tục."
                        active_text_time = time.time()
                        print("[System] >>> ĐÃ TẠM DỪNG nhận diện (SPACE để tiếp tục)")

                # [Q] hoặc [ESC]: Thoát chương trình
                elif key == ord("q") or key == 27:
                    break

            if is_new:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = detector.process(rgb)
                landmarks = extract_landmarks(results)

                # Tính toán mức độ vận động
                if previous_landmarks is None:
                    current_motion = 0.0
                else:
                    current_motion = motion_energy(landmarks, previous_landmarks)
                previous_landmarks = landmarks

                # Kiểm tra tay: Bàn tay hoặc cổ tay trong pose được nâng lên vùng ký hiệu (ngực/mặt)
                wrist_visible = False
                wrist_raised = False
                if results.pose_landmarks:
                    lw = results.pose_landmarks.landmark[15]
                    rw = results.pose_landmarks.landmark[16]
                    l_hip = results.pose_landmarks.landmark[23]
                    r_hip = results.pose_landmarks.landmark[24]
                    hip_y = min(l_hip.y, r_hip.y) if (l_hip.visibility > 0.3 and r_hip.visibility > 0.3) else 0.75
                    wrist_raised = ((lw.visibility > 0.3 and lw.y < hip_y) or (rw.visibility > 0.3 and rw.y < hip_y))
                    wrist_visible = (lw.visibility > 0.4 or rw.visibility > 0.4)

                hand_present = bool(results.left_hand_landmarks or results.right_hand_landmarks)
                hand_detected = bool(hand_present or wrist_visible)

                if recognition_enabled:
                    if not is_signing:
                        # 1. TRẠNG THÁI NGHỈ (IDLE): Chờ người dùng giơ tay bắt đầu ký hiệu
                        # Điều kiện bắt đầu: Tay được nâng lên và có vận động rõ rệt (>= motion_start)
                        if (hand_present or wrist_raised) and current_motion >= args.motion_start:
                            is_signing = True
                            gesture_frames = [landmarks]
                            signing_idle_count = 0
                    else:
                        # 2. TRẠNG THÁI THU NHẬN CỬ CHỈ (SIGNING): Thu nhận chuỗi frame trọn vẹn theo cơ chế Dynamic
                        gesture_frames.append(landmarks)

                        # Nhận biết người dùng đã dừng hoặc hạ tay:
                        # - Chuyển động dừng hẳn: current_motion < args.motion_stop (mặc định 0.007)
                        # - Hoặc tay đã hạ xuống dưới tầm ngực/hông: not wrist_raised
                        # - Hoặc không còn thấy bàn tay: not hand_present
                        hand_stopped = (current_motion < args.motion_stop) or (not wrist_raised) or (not hand_present)
                        if hand_stopped:
                            signing_idle_count += 1
                        else:
                            signing_idle_count = 0

                        # Điều kiện kết thúc cử chỉ:
                        # - Dừng/hạ tay đủ số frame (idle_stop, mặc định 3 frame) sau khi đã làm >= min_frames (mặc định 10)
                        # - Hoặc đạt giới hạn an toàn max_frames (mặc định 50 frames ~ 2s)
                        gesture_finished = (
                            (signing_idle_count >= args.idle_stop and len(gesture_frames) >= args.min_frames)
                            or (len(gesture_frames) >= args.max_frames)
                        )

                        if gesture_finished:
                            is_signing = False
                            valid_len = len(gesture_frames) - signing_idle_count if signing_idle_count > 0 else len(gesture_frames)
                            valid_frames = gesture_frames[:valid_len] if valid_len >= args.min_frames else gesture_frames

                            if len(valid_frames) >= args.min_frames:
                                # Co giãn nội suy về đúng chuẩn 48 frame và đưa vào model
                                input_tensor = build_tensor(valid_frames, device)
                                with torch.no_grad():
                                    logits = model(input_tensor)
                                    probabilities = torch.softmax(logits, dim=-1)[0].cpu().numpy()

                                new_word, is_valid, top3 = decoder.decode_segment(probabilities, hand_detected=True)
                                recent_top3 = top3
                                top1_w, top1_c = top3[0]
                                top2_w, top2_c = top3[1] if len(top3) > 1 else ("", 0.0)
                                margin = top1_c - top2_c
                                top_prediction = top1_w
                                top_confidence = top1_c

                                if is_valid and new_word:
                                    current_sentence = " ".join(decoder.sentence_buffer)
                                    active_title = f"ĐÃ THÊM TỪ: {new_word.upper()}"
                                    active_text = f"Đã nhận '{new_word}' ({top1_c*100:.0f}%) -> Câu: {current_sentence}"
                                    active_text_time = time.time()
                                    print(f"[Model] >>> ĐÃ NHẬN DIỆN TỪ: '{new_word}' ({top1_c*100:.1f}% | +{margin*100:.1f}%) [{len(valid_frames)} frames] -> Câu ({len(decoder.sentence_buffer)} từ): {current_sentence}")
                                    if args.esp_ip:
                                        try:
                                            speech_session.post(
                                                f"http://{args.esp_ip}/oled",
                                                data=current_sentence.encode("utf-8", errors="ignore"),
                                                timeout=1,
                                            )
                                        except requests.RequestException:
                                            pass
                                else:
                                    active_title = "CHƯA RÕ CỬ CHỈ"
                                    active_text = f"Cân nhắc: 1.{top1_w} ({top1_c*100:.0f}%) | 2.{top2_w} ({top2_c*100:.0f}%) [Margin: {margin*100:.0f}%]"
                                    active_text_time = time.time()
                                    print(f"[Model] Cử chỉ chưa chắc chắn (1. '{top1_w}' {top1_c*100:.1f}% vs 2. '{top2_w}' {top2_c*100:.1f}% | Margin {margin*100:.1f}%) [{len(valid_frames)} frames]. Bỏ qua.")

                            gesture_frames = []
                            signing_idle_count = 0

                # Xác định nội dung hiển thị trên banner dưới
                with llm_lock:
                    llm_recent = (time.time() - llm_state["updated_at"] < 8.0) and bool(llm_state["text"])
                    llm_t = llm_state["title"]
                    llm_txt = llm_state["text"]

                if llm_recent and (time.time() - active_text_time > 2.0 or not recognition_enabled):
                    banner_title = llm_t
                    banner_text = llm_txt
                elif is_signing:
                    banner_title = f"ĐANG THU NHẬN CỬ CHỈ ({len(gesture_frames)} frames)"
                    banner_text = "Đang làm cử chỉ... Dừng/hạ tay sau khi làm xong để hệ thống chốt từ."
                elif active_text and (time.time() - active_text_time < 5.0):
                    banner_title = active_title
                    banner_text = active_text
                elif decoder.sentence_buffer:
                    banner_title = f"CÂU ĐANG GHÉP ({len(decoder.sentence_buffer)} TỪ)"
                    banner_text = f"{' '.join(decoder.sentence_buffer)}   (Bấm [ENTER] để dịch & phát loa)"
                elif recognition_enabled:
                    banner_title = "SẴN SÀNG"
                    banner_text = "Hãy giơ tay thực hiện cử chỉ... Dừng tay để chốt từ."
                else:
                    banner_title = "TẠM DỪNG"
                    banner_text = "Bấm SPACE trên cửa sổ video để BẬT nhận diện"

                # Phóng to khung hình camera 320x240 lên canvas 960x720 để vẽ sắc nét và không bao giờ bị tràn mép
                display = cv2.resize(frame, (CANVAS_WIDTH, CANVAS_HEIGHT), interpolation=cv2.INTER_LINEAR)

                # Vẽ xương tay trái
                if results.left_hand_landmarks:
                    mp_drawing.draw_landmarks(
                        display,
                        results.left_hand_landmarks,
                        mp_holistic.HAND_CONNECTIONS,
                        mp_drawing.DrawingSpec(color=(121, 22, 76), thickness=2, circle_radius=2),
                        mp_drawing.DrawingSpec(color=(121, 44, 250), thickness=2, circle_radius=1),
                    )

                # Vẽ xương tay phải
                if results.right_hand_landmarks:
                    mp_drawing.draw_landmarks(
                        display,
                        results.right_hand_landmarks,
                        mp_holistic.HAND_CONNECTIONS,
                        mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
                        mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=1),
                    )

                # Vẽ các điểm cơ thể (vai, khuỷu, cổ tay)
                if results.pose_landmarks:
                    mp_drawing.draw_landmarks(
                        display,
                        results.pose_landmarks,
                        mp_holistic.POSE_CONNECTIONS,
                        mp_drawing.DrawingSpec(color=(80, 22, 10), thickness=2, circle_radius=2),
                        mp_drawing.DrawingSpec(color=(80, 44, 121), thickness=2, circle_radius=1),
                    )

                draw_status(
                    display,
                    banner_title,
                    banner_text,
                    recognition_enabled,
                    sentence_buffer=decoder.sentence_buffer,
                    top_pred=top_prediction,
                    top_conf=top_confidence,
                    hand_ok=hand_detected,
                    top3_list=recent_top3,
                    signing_len=len(gesture_frames) if is_signing else 0,
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