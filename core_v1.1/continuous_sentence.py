import argparse
from collections import deque
import json
import os
import queue
import sys
import threading
import time
import warnings
from pathlib import Path
from urllib.parse import urlparse

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
warnings.filterwarnings("ignore")
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)
    sys.stderr.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)
except Exception:
    pass

import cv2
import mediapipe as mp
import numpy as np
import requests
import torch
from PIL import Image, ImageDraw, ImageFont

# Thư viện âm thanh sounddevice (nếu có sẵn trên máy)
try:
    import sounddevice as sd
    AUDIO_AVAILABLE = True
except Exception:
    AUDIO_AVAILABLE = False

ROOT_DIR = Path(__file__).resolve().parent.parent
CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))
if str(ROOT_DIR / "core") not in sys.path:
    sys.path.insert(0, str(ROOT_DIR / "core"))

from camera import LatestFrameCamera
from decoder import ContinuousWordSpotter, is_idle_word
from gemini_api import GeminiClient
from preprocess import build_tensor, extract_landmarks, motion_energy
from ctr_gcn_model import load_ctr_gcn_checkpoint, load_model_checkpoint
from stgcn_model import load_stgcn_checkpoint

try:
    FONT_TITLE = ImageFont.truetype("arial.ttf", 18)
    FONT_TEXT = ImageFont.truetype("arial.ttf", 22)
    FONT_BOLD = ImageFont.truetype("arialbd.ttf", 19)
    FONT_BIG = ImageFont.truetype("arialbd.ttf", 23)
    FONT_HINT = ImageFont.truetype("arial.ttf", 15)
    FONT_SMALL = ImageFont.truetype("arial.ttf", 14)
except Exception:
    FONT_TITLE = ImageFont.load_default()
    FONT_TEXT = ImageFont.load_default()
    FONT_BOLD = ImageFont.load_default()
    FONT_BIG = ImageFont.load_default()
    FONT_HINT = ImageFont.load_default()
    FONT_SMALL = ImageFont.load_default()


def read_labels(path):
    with open(path, "r", encoding="utf-8") as file:
        raw = json.load(file)
    if isinstance(raw, list):
        return {index: value for index, value in enumerate(raw)}
    return {int(value): key for key, value in raw.items()}


def resolve_esp_ip(target_ip=None, timeout=0.15):
    """Tự động quét và tìm IP ESP32 trong mạng nội bộ."""
    def _check(ip):
        try:
            s = requests.Session()
            s.trust_env = False
            r = s.get(f"http://{ip}/", timeout=timeout)
            if r.status_code == 200 and ("esp" in r.text.lower() or "cam" in r.text.lower() or "stream" in r.text.lower()):
                return ip
        except Exception:
            pass
        return None

    if target_ip and _check(target_ip):
        return target_ip

    candidates = [
        "10.245.192.219", "192.168.4.1", "192.168.1.100", "192.168.1.101",
        "192.168.1.102", "192.168.1.105", "192.168.1.110", "192.168.1.150"
    ]
    ordered = [target_ip] + [c for c in candidates if c != target_ip] if target_ip else candidates
    from concurrent.futures import ThreadPoolExecutor
    try:
        with ThreadPoolExecutor(max_workers=8) as pool:
            for cand in pool.map(_check, ordered):
                if cand:
                    print(f"[ESP32 Auto-Discover] Tìm thấy ESP32 tại IP: {cand}")
                    return cand
    except Exception:
        pass

    fallback = target_ip or "192.168.4.1"
    return fallback


def translate_glosses_with_gemini(gemini_client, words: list[str]) -> str:
    """
    Dịch chuỗi từ khóa ngôn ngữ ký hiệu (VSL glosses) thành câu tiếng Việt hoàn chỉnh và tự nhiên.
    Được phép tự động suy luận và thêm các từ nối, trợ từ, liên từ (ví dụ: 'chơi', 'bị', 'ở', 'đến'...).
    """
    clean_words = [str(w).strip() for w in words if str(w).strip() and not is_idle_word(str(w))]
    if not clean_words:
        return ""

    fallback = " ".join(clean_words).capitalize() + "."
    if not gemini_client or not gemini_client.api_keys:
        return fallback

    prompt = (
        "Bạn là chuyên gia thông dịch ngôn ngữ ký hiệu tiếng Việt (VSL) sang tiếng Việt tự nhiên.\n"
        "Người khiếm thính giao tiếp bằng cách ghép các từ khóa cốt lõi (gloss), thường thiếu liên từ, "
        "giới từ, trợ từ, động từ phụ hoặc sai thứ tự ngữ pháp.\n"
        "Nhiệm vụ: Dựa vào chuỗi từ khóa đã nhận diện, hãy suy luận ngữ cảnh và khôi phục thành một câu tiếng Việt "
        "tự nhiên, chuẩn ngữ pháp, lưu loát và sát nghĩa nhất. Được phép thêm các từ nối, trợ từ, liên từ, động từ phụ cần thiết "
        "(ví dụ thêm: 'chơi', 'bị', 'đi', 'đã', 'ở', 'đang', 'muốn', 'rất', 'vào'...). Không thêm thông tin bịa đặt.\n"
        "Chỉ trả về đúng MỘT câu tiếng Việt hoàn chỉnh, không có dấu ngoặc kép thừa hay giải thích.\n\n"
        "Ví dụ:\n"
        "- ['tôi', 'bóng chuyền', 'đau'] -> Tôi chơi bóng chuyền bị đau.\n"
        "- ['mẹ', 'chợ', 'mua', 'cá'] -> Mẹ đi chợ mua cá.\n"
        "- ['tôi', 'bệnh viện', 'đi'] -> Tôi đi đến bệnh viện.\n"
        "- ['bạn', 'ăn', 'cơm', 'chưa'] -> Bạn đã ăn cơm chưa?\n"
        "- ['hôm nay', 'trời', 'mưa', 'lớn'] -> Hôm nay trời mưa rất lớn.\n"
        "- ['tôi', 'uống', 'nước'] -> Tôi muốn uống nước.\n\n"
        f"Chuỗi từ khóa: {clean_words}\n"
        "Câu tiếng Việt hoàn chỉnh:"
    )

    try:
        for _ in range(len(gemini_client.api_keys)):
            current_key = gemini_client._get_next_key()
            url = f"{gemini_client.base_url}/{gemini_client.llm_model}:generateContent?key={current_key}"
            payload = {
                "contents": [{"parts": [{"text": prompt}]}],
                "generationConfig": {"temperature": 0.2},
            }
            resp = requests.post(url, json=payload, timeout=8.0)
            if resp.status_code == 429:
                continue
            if resp.status_code != 200:
                continue
            data = resp.json()
            cand_text = data["candidates"][0]["content"]["parts"][0]["text"].strip()
            cand_text = cand_text.strip('"\n\r')
            if cand_text:
                return cand_text
    except Exception as error:
        print(f"[Gemini Translator] Lỗi gọi API: {error}")

    return fallback


def draw_continuous_sentence_overlay(
    display,
    words_sequence,
    candidate_word,
    candidate_conf,
    top5,
    translated_sentence,
    is_translating,
    translation_duration,
    tts_enabled,
    is_speaking,
    auto_delay,
    time_since_last_word,
    mode="continuous",
):
    """Vẽ giao diện hiển thị chuỗi từ ký hiệu và câu tiếng Việt hoàn chỉnh."""
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

    # 3. Left Container (Hộp chuỗi từ & Hộp câu hoàn chỉnh)
    left_x = 15
    left_w = panel_x - 30

    # Hộp 3A: Chuỗi từ khóa nhận diện liên tục (Gloss stream)
    box_gloss_y = top_bar_h + 15
    box_gloss_h = 92
    box_gloss_overlay = display.copy()
    cv2.rectangle(box_gloss_overlay, (left_x, box_gloss_y), (left_x + left_w, box_gloss_y + box_gloss_h), (16, 22, 30), -1)
    cv2.addWeighted(box_gloss_overlay, 0.88, display, 0.12, 0, display)
    cv2.rectangle(display, (left_x, box_gloss_y), (left_x + left_w, box_gloss_y + box_gloss_h), (0, 200, 240), 1)

    # Hộp 3B: Câu tiếng Việt hoàn chỉnh (Gemini AI Translation)
    box_trans_y = box_gloss_y + box_gloss_h + 12
    box_trans_h = 135
    box_trans_overlay = display.copy()
    cv2.rectangle(box_trans_overlay, (left_x, box_trans_y), (left_x + left_w, box_trans_y + box_trans_h), (20, 28, 22), -1)
    cv2.addWeighted(box_trans_overlay, 0.90, display, 0.10, 0, display)
    border_color = (0, 255, 150) if translated_sentence else (80, 120, 90)
    cv2.rectangle(display, (left_x, box_trans_y), (left_x + left_w, box_trans_y + box_trans_h), border_color, 2)

    # Hộp 3C: Trạng thái Âm thanh & Gợi ý nhanh
    box_sub_y = box_trans_y + box_trans_h + 12
    box_sub_h = 58
    box_sub_overlay = display.copy()
    cv2.rectangle(box_sub_overlay, (left_x, box_sub_y), (left_x + left_w, box_sub_y + box_sub_h), (18, 18, 18), -1)
    cv2.addWeighted(box_sub_overlay, 0.85, display, 0.15, 0, display)
    cv2.rectangle(display, (left_x, box_sub_y), (left_x + left_w, box_sub_y + box_sub_h), (70, 70, 70), 1)

    # 4. Bottom Guide Bar
    bot_h = 40
    bot_y = h - bot_h
    bot_overlay = display.copy()
    cv2.rectangle(bot_overlay, (0, bot_y), (w, h), (18, 18, 18), -1)
    cv2.addWeighted(bot_overlay, 0.8, display, 0.2, 0, display)
    cv2.line(display, (0, bot_y), (w, bot_y), (0, 215, 255), 2)

    # Vẽ chữ tiếng Việt bằng PIL
    pil_img = Image.fromarray(cv2.cvtColor(display, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_img)

    # --- TOP BAR ---
    if is_translating:
        draw.ellipse((18, 13, 30, 25), fill=(255, 180, 0))
        rec_str = "GEMINI AI ĐANG SUY LUẬN NGỮ NGHĨA & GHÉP CÂU..."
        rec_color = (255, 210, 80)
    elif candidate_word:
        draw.ellipse((18, 13, 30, 25), fill=(0, 230, 115))
        rec_str = f"LIÊN TỤC: Đang nhận cử chỉ [{candidate_word.upper()}] ({candidate_conf*100:.0f}%)"
        rec_color = (0, 255, 140)
    elif is_speaking:
        draw.ellipse((18, 13, 30, 25), fill=(180, 100, 255))
        rec_str = "LOA ĐANG PHÁT ÂM THANH GIỌNG ĐỌC..."
        rec_color = (210, 150, 255)
    else:
        draw.ellipse((18, 13, 30, 25), fill=(80, 220, 120))
        rec_str = "SẴN SÀNG: Hãy thực hiện các cử chỉ liên tục trước camera..."
        rec_color = (180, 240, 200)

    draw.text((38, 10), rec_str, font=FONT_BOLD, fill=rec_color)
    draw.text((w - 240, 11), "Chế độ: TỰ GHÉP CÂU AI", font=FONT_HINT, fill=(0, 215, 255))

    # --- HỘP 3A: CHUỖI TỪ KHÓA (GLOSS STREAM) ---
    valid_words = [w for w in words_sequence if not is_idle_word(w)]
    header_gloss = f"1. TỪ KHÓA KÝ HIỆU ({len(valid_words)} từ)"
    draw.text((left_x + 12, box_gloss_y + 8), header_gloss, font=FONT_SMALL, fill=(0, 215, 255))

    # Đếm ngược tự động dịch
    if len(valid_words) > 0 and auto_delay > 0:
        remain = max(0.0, auto_delay - time_since_last_word)
        if remain > 0:
            timer_txt = f"Tự động dịch sau {remain:.1f}s dừng tay (hoặc bấm [ENTER])"
            draw.text((left_x + left_w - 330, box_gloss_y + 8), timer_txt, font=FONT_SMALL, fill=(180, 180, 180))

    if valid_words and len(valid_words) > 0:
        seq_display = "  ➔  ".join(f"[{w.upper()}]" for w in valid_words[-6:])
        draw.text((left_x + 12, box_gloss_y + 36), seq_display, font=FONT_BOLD, fill=(255, 225, 70))
    else:
        draw.text((left_x + 12, box_gloss_y + 40), "(Chưa có từ nào. Hãy làm các cử chỉ trước camera...)", font=FONT_HINT, fill=(150, 165, 180))

    # --- HỘP 3B: CÂU TIẾNG VIỆT HOÀN CHỈNH ---
    draw.text((left_x + 12, box_trans_y + 8), "2. CÂU TIẾNG VIỆT HOÀN CHỈNH (GEMINI AI SUY LUẬN):", font=FONT_SMALL, fill=(0, 255, 170))

    if is_translating:
        draw.text((left_x + 12, box_trans_y + 45), "Đang phân tích ngữ pháp & ghép câu tự nhiên...", font=FONT_BIG, fill=(255, 215, 0))
    elif translated_sentence:
        # Wrap văn bản để hiển thị vừa vặn
        words_in_trans = translated_sentence.split()
        line1, line2 = "", ""
        for w_tok in words_in_trans:
            if len(line1 + " " + w_tok) < 38:
                line1 = (line1 + " " + w_tok).strip()
            else:
                line2 = (line2 + " " + w_tok).strip()

        quote_line1 = f'"{line1}'
        if not line2:
            quote_line1 += '"'
        draw.text((left_x + 12, box_trans_y + 38), quote_line1, font=FONT_BIG, fill=(255, 255, 255))
        if line2:
            draw.text((left_x + 20, box_trans_y + 68), f'{line2}"', font=FONT_BIG, fill=(255, 255, 255))

        badge = f"[ĐÃ DỊCH TRONG {translation_duration:.2f}s]"
        draw.text((left_x + 12, box_trans_y + 105), badge, font=FONT_SMALL, fill=(0, 255, 150))
        draw.text((left_x + 220, box_trans_y + 105), "Bấm [C] để xóa câu cũ và bắt đầu câu mới", font=FONT_SMALL, fill=(180, 180, 180))
    else:
        draw.text((left_x + 12, box_trans_y + 48), 'Ví dụ: ["tôi", "bóng chuyền", "đau"] ➔ "Tôi chơi bóng chuyền bị đau."', font=FONT_HINT, fill=(130, 160, 140))
        draw.text((left_x + 12, box_trans_y + 78), '(Khi hoàn thành chuỗi ký hiệu, dừng tay 2s hoặc bấm [ENTER] để dịch)', font=FONT_SMALL, fill=(100, 130, 110))

    # --- HỘP 3C: ÂM THANH & TRẠNG THÁI ---
    tts_status = "BẬT (Laptop Speaker)" if tts_enabled else "TẮT"
    tts_color = (0, 255, 150) if tts_enabled else (160, 160, 160)
    draw.text((left_x + 12, box_sub_y + 8), f"Giọng đọc TTS: {tts_status}", font=FONT_HINT, fill=tts_color)
    draw.text((left_x + 12, box_sub_y + 32), "Phím tắt: [ENTER] Dịch ngay  |  [C] Xóa câu mới  |  [BACKSPACE] Xóa từ cuối", font=FONT_SMALL, fill=(200, 200, 200))

    # --- BẢNG BÊN PHẢI (TOP 5) ---
    draw.text((panel_x + 15, panel_y + 12), "TOP 5 DỰ ĐOÁN MÔ HÌNH", font=FONT_BOLD, fill=(0, 215, 255))
    draw.line([(panel_x + 15, panel_y + 36), (panel_x + panel_w - 15, panel_y + 36)], fill=(70, 70, 70), width=1)

    if top5:
        cur_y = panel_y + 45
        for rank, (word, conf) in enumerate(top5[:5], 1):
            rank_color = (255, 215, 0) if rank == 1 else ((200, 200, 200) if rank == 2 else (140, 140, 140))
            draw.text((panel_x + 15, cur_y), f"#{rank}", font=FONT_HINT, fill=rank_color)
            draw.text((panel_x + 45, cur_y), word[:14], font=FONT_HINT, fill=(245, 245, 245))

            pct_str = f"{conf * 100:5.1f}%"
            draw.text((panel_x + panel_w - 75, cur_y), pct_str, font=FONT_HINT, fill=rank_color)

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

        if len(top5) >= 2:
            diff = (top5[0][1] - top5[1][1]) * 100
            diff_color = (100, 255, 100) if diff >= 10.0 else (255, 160, 50)
            draw.text((panel_x + 15, panel_y + panel_h - 40), f"Chênh lệch #1 vượt #2: +{diff:.1f}%", font=FONT_SMALL, fill=diff_color)
    else:
        draw.text((panel_x + 25, panel_y + 90), "Chưa có dữ liệu cử chỉ.", font=FONT_HINT, fill=(150, 150, 150))
        draw.text((panel_x + 25, panel_y + 115), "Hãy làm cử chỉ trước camera.", font=FONT_SMALL, fill=(110, 110, 110))

    # --- BOTTOM GUIDE BAR ---
    guide_str = "[ENTER] Dịch câu ngay  |  [C] Xóa câu mới  |  [BACKSPACE] Xóa từ cuối  |  [T] Bật/Tắt TTS  |  [Q] Thoát"
    draw.text((25, bot_y + 10), guide_str, font=FONT_HINT, fill=(220, 220, 220))

    display[:] = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)


def main():
    parser = argparse.ArgumentParser(description="Nhận diện cử chỉ VSL liên tục (CTR-GCN / ST-GCN) & Ghép thành câu tiếng Việt tự nhiên (Gemini AI)")
    parser.add_argument("--checkpoint", default=str(ROOT_DIR / "models" / "best_vsl_model_ctr_gcn.pth"))
    parser.add_argument("--labels", default=str(ROOT_DIR / "core_v1.1" / "label_map_472_10w.json"))
    parser.add_argument("--camera", default="0", help="'0' (webcam laptop), 'esp' (camera ESP32), hoặc URL http")
    parser.add_argument("--esp-ip", default=None, help="IP thủ công của ESP32 (ví dụ: 10.3.83.97)")
    parser.add_argument("--device", default=None, choices=["cpu", "cuda"])
    parser.add_argument("--frames", type=int, default=None, help="Độ dài chuỗi frame (tự nhận diện: 15 nếu checkpoint 15-frame, 48 cho checkpoint chuẩn)")
    parser.add_argument("--stride", type=int, default=3, help="Bước trượt sliding window (mặc định 3 frames)")
    parser.add_argument("--conf", type=float, default=0.35, help="Ngưỡng tin cậy tối thiểu cho continuous spotter (mặc định 0.35)")
    parser.add_argument("--margin", type=float, default=0.08, help="Ngưỡng chênh lệch Top1 - Top2 tối thiểu (mặc định 0.08)")
    parser.add_argument("--min-peak", type=float, default=0.42, help="Ngưỡng đỉnh xác suất tối thiểu để công nhận cử chỉ (mặc định 0.42)")
    parser.add_argument("--min-hold", type=int, default=2, help="Số bước duy trì nhận diện tối thiểu để công nhận từ (mặc định 2)")
    parser.add_argument("--cooldown", type=int, default=5, help="Số bước giãn cách để tránh lặp từ vừa nhận diện (mặc định 5)")
    parser.add_argument("--auto-delay", type=float, default=2.0, help="Thời gian dừng tay (giây) để tự động dịch câu (mặc định 2.0s, 0 để tắt)")
    parser.add_argument("--no-tts", action="store_true", help="Tắt tính năng phát âm thanh giọng đọc")
    args = parser.parse_args()

    # Khởi tạo mô hình
    device_name = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_name)
    checkpoint_path = Path(args.checkpoint)
    model, checkpoint, num_classes = load_model_checkpoint(checkpoint_path, device)
    model_name = "CTR-GCN" if "CTR" in type(model).__name__ else "ST-GCN Transformer"
    print(f"\n[Model] Đã tải {model_name} | classes={num_classes} | device={device}")
    if "val_acc" in checkpoint:
        print(f"[Model] Checkpoint val_acc={checkpoint['val_acc']:.4f}")

    labels_path = Path(args.labels)
    if "label_map" in checkpoint and isinstance(checkpoint["label_map"], dict) and len(checkpoint["label_map"]) == num_classes:
        idx_to_class = {int(v): k for k, v in checkpoint["label_map"].items()}
    elif labels_path.exists():
        idx_to_class = read_labels(labels_path)
    else:
        labels_path = ROOT_DIR / "core_v1.1" / "label_map_472.json"
        idx_to_class = read_labels(labels_path)

    # Tự động khớp nhãn nếu file nhãn truyền vào không khớp với số classes mô hình
    if len(idx_to_class) != num_classes:
        if num_classes == 10 and (CURRENT_DIR / "label_map_472_10w.json").exists():
            idx_to_class = read_labels(CURRENT_DIR / "label_map_472_10w.json")
        elif (CURRENT_DIR / "label_map_472.json").exists() and len(read_labels(CURRENT_DIR / "label_map_472.json")) == num_classes:
            idx_to_class = read_labels(CURRENT_DIR / "label_map_472.json")
        else:
            raise ValueError(f"Số nhãn ({len(idx_to_class)}) khác output model ({num_classes})")

    if args.frames is not None:
        window_size = int(args.frames)
    elif "15_frame" in str(checkpoint_path).lower():
        window_size = 15
    else:
        window_size = 48
    print(f"[Model] Cấu hình cửa sổ thời gian (window_size): {window_size} frames")

    # Khởi tạo Gemini Client
    gemini_client = GeminiClient()
    if gemini_client.api_keys:
        print(f"[Gemini] Đã sẵn sàng {len(gemini_client.api_keys)} API Keys cho dịch câu và TTS.")
    else:
        print("[Gemini Warning] Không tìm thấy API Keys. Sẽ dùng phương thức ghép từ dự phòng.")

    # Xử lý Camera
    if str(args.camera).lower() in ["esp", "esp32", "cam"]:
        ip_file = ROOT_DIR / "configs" / "esp_ip.txt"
        target_ip = args.esp_ip
        if not target_ip and ip_file.is_file():
            target_ip = ip_file.read_text(encoding="utf-8").strip() or None
        target_ip = target_ip or "10.245.192.219"
        actual_ip = resolve_esp_ip(target_ip)
        source = f"http://{actual_ip}:81/stream"
        print(f"[Camera] Kết nối Camera ESP32: {source}")

        # Kiểm tra fallback nếu ESP32 offline
        esp_cam_ok = False
        try:
            test_session = requests.Session()
            test_session.trust_env = False
            r = test_session.get(source, stream=True, timeout=1.8)
            if r.status_code == 200:
                esp_cam_ok = True
            r.close()
        except Exception:
            esp_cam_ok = False

        if not esp_cam_ok:
            print("\n" + "=" * 70)
            print("  [Camera Fallback] KHÔNG TÌM THẤY CAMERA ESP32!")
            print("  -> TỰ ĐỘNG CHUYỂN SANG DÙNG WEBCAM LAPTOP (Camera 0)")
            print("=" * 70 + "\n")
            source = 0
    elif str(args.camera).isdigit():
        source = int(args.camera)
        print(f"[Camera] Sử dụng Webcam Laptop: {source}")
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
    WINDOW_NAME = "BlindtoBright - Continuous Sentence Translator"
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, CANVAS_WIDTH, CANVAS_HEIGHT)

    # Quản lý thuật toán Continuous Spotter
    spotter = ContinuousWordSpotter(
        idx_to_class,
        confidence_threshold=args.conf,
        margin_threshold=args.margin,
        min_peak_conf=args.min_peak,
        min_hold_steps=args.min_hold,
        cooldown_steps=args.cooldown,
    )
    sliding_window = deque(maxlen=window_size)
    frame_counter = 0
    candidate_word = None
    candidate_conf = 0.0

    last_top5 = []
    previous_landmarks = None
    last_frame_id = -1

    # Quản lý dịch câu & TTS
    tts_enabled = not args.no_tts
    is_speaking = False
    is_translating = False
    translated_sentence = ""
    translation_duration = 0.0
    last_word_commit_time = 0.0
    sentence_already_translated = False

    # Hàng đợi tác vụ dịch nền (để không làm giật khung hình camera)
    translation_queue = queue.Queue()

    def translation_worker():
        nonlocal is_translating, translated_sentence, translation_duration, is_speaking
        while True:
            item = translation_queue.get()
            if item is None:
                translation_queue.task_done()
                break

            words_to_translate = item
            is_translating = True
            t0 = time.time()
            print(f"\n[AI Translator] >>> Đang ghép câu từ từ khóa: {words_to_translate} ...")

            result_sentence = translate_glosses_with_gemini(gemini_client, words_to_translate)
            dt = time.time() - t0

            translated_sentence = result_sentence
            translation_duration = dt
            is_translating = False
            print(f"[AI Translator] >>> KẾT QUẢ: \"{result_sentence}\" (Thời gian: {dt:.2f}s)\n")

            # Phát âm thanh giọng đọc nếu bật TTS
            if tts_enabled and gemini_client.api_keys and result_sentence:
                try:
                    is_speaking = True
                    pcm = gemini_client.generate_speech(result_sentence)
                    if pcm and AUDIO_AVAILABLE:
                        audio_data = np.frombuffer(pcm, dtype=np.int16)
                        sd.play(audio_data, samplerate=24000)
                        sd.wait()
                    elif args.esp_ip and pcm:
                        try:
                            requests.post(f"http://{args.esp_ip}/play", data=pcm, timeout=10)
                        except Exception:
                            pass
                except Exception as tts_err:
                    print(f"[TTS Warning] Không phát được giọng đọc: {tts_err}")
                finally:
                    is_speaking = False

            translation_queue.task_done()

    worker_thread = threading.Thread(target=translation_worker, daemon=True)
    worker_thread.start()

    print("\n" + "=" * 72)
    print("  [BLINDTOBRIGHT - HỆ THỐNG NHẬN DIỆN LIÊN TỤC & TỰ ĐỘNG GHÉP CÂU AI]")
    print(f"  - Checkpoint       : {checkpoint_path.name} ({window_size} frames)")
    print(f"  - Cửa sổ trượt     : Stride={args.stride} frames | Dynamic Peak Detection")
    print(f"  - Tự động dịch sau : {args.auto_delay}s dừng tay (hoặc bấm [ENTER] để dịch ngay)")
    print(f"  - Giọng đọc (TTS)  : {'BẬT' if tts_enabled else 'TẮT'}")
    print("  - Hướng dẫn:")
    print("    * Làm các cử chỉ nối tiếp nhau tự nhiên (ví dụ: 'tôi' -> 'bóng chuyền' -> 'đau')")
    print("    * Dừng tay 2s: Gemini AI sẽ tự động đoán nghĩa thành câu hoàn chỉnh!")
    print("  - Phím tắt:")
    print("    * [ENTER]    : Dịch câu ngay lập tức")
    print("    * [C]        : Xóa sạch chuỗi từ & câu cũ để bắt đầu câu mới")
    print("    * [BACKSPACE]: Xóa từ cuối cùng vừa nhận (nếu nhận nhầm)")
    print("    * [T]        : Bật / Tắt giọng đọc (TTS)")
    print("    * [Q]        : Thoát")
    print("=" * 72 + "\n")

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

            raw_key = cv2.waitKey(1)
            if raw_key != -1:
                key = raw_key & 0xFF

                # [ENTER] (key 13): Kích hoạt dịch câu ngay lập tức
                if key in (13, 10):
                    if len(spotter.words_sequence) > 0 and not is_translating:
                        sentence_already_translated = True
                        translation_queue.put(list(spotter.words_sequence))

                # [C]: Xóa câu và chuỗi từ để làm câu mới
                elif key in (ord("c"), ord("C")):
                    spotter.clear()
                    sliding_window.clear()
                    candidate_word = None
                    candidate_conf = 0.0
                    last_top5 = []
                    translated_sentence = ""
                    sentence_already_translated = False
                    print("\n[Action] Đã xóa toàn bộ câu và chuỗi từ. Sẵn sàng cho câu mới!")

                # [BACKSPACE] hoặc [DEL]: Xóa từ cuối cùng
                elif key in (8, 127):
                    removed = spotter.remove_last_word()
                    if removed:
                        sentence_already_translated = False
                        last_word_commit_time = time.time()
                        seq_str = " ➔ ".join(f"[{w}]" for w in spotter.words_sequence) if spotter.words_sequence else "(Trống)"
                        print(f"\n[Action] Đã xóa từ cuối: [{removed}] | Chuỗi còn lại: {seq_str}")
                    else:
                        print("\n[Action] Chuỗi từ đang trống, không có gì để xóa.")

                # [T]: Bật/Tắt giọng đọc TTS
                elif key in (ord("t"), ord("T")):
                    tts_enabled = not tts_enabled
                    print(f"\n[TTS] Đã {'BẬT' if tts_enabled else 'TẮT'} giọng đọc âm thanh.")

                # [Q] hoặc [ESC]: Thoát
                elif key in (ord("q"), ord("Q"), 27):
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

                # Cập nhật sliding window
                sliding_window.append(landmarks)
                frame_counter += 1

                # Thực hiện suy luận theo nhịp stride khi có đủ khung hình
                min_req = min(12, window_size)
                if (frame_counter % args.stride == 0) and (len(sliding_window) >= min_req):
                    input_tensor = build_tensor(list(sliding_window), device, target_len=window_size)
                    with torch.no_grad():
                        logits = model(input_tensor)
                        probs = torch.softmax(logits, dim=-1)[0].cpu().numpy()

                    committed_word, candidate_word, candidate_conf, last_top5 = spotter.update(
                        probs, hand_detected=hand_detected, motion=motion
                    )

                    if committed_word:
                        last_word_commit_time = time.time()
                        sentence_already_translated = False
                        seq_str = " ➔ ".join(f"[{w.upper()}]" for w in spotter.words_sequence)
                        print(f"\n[CONTINUOUS SPOT] >>> ĐÃ CHỐT TỪ: [{committed_word.upper()}]  |  Chuỗi: {seq_str}")

                # Tự động kích hoạt dịch câu khi người dùng dừng tay sau auto_delay giây
                now = time.time()
                time_since_last_word = now - last_word_commit_time if last_word_commit_time > 0 else 0.0
                if (
                    len(spotter.words_sequence) > 0
                    and not sentence_already_translated
                    and not is_translating
                    and args.auto_delay > 0
                    and time_since_last_word >= args.auto_delay
                ):
                    sentence_already_translated = True
                    translation_queue.put(list(spotter.words_sequence))

                # Resize lên Canvas 960x720
                display = cv2.resize(frame, (CANVAS_WIDTH, CANVAS_HEIGHT), interpolation=cv2.INTER_LINEAR)

                # Vẽ xương khớp MediaPipe
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

                # Vẽ toàn bộ giao diện thông tin & dịch câu
                draw_continuous_sentence_overlay(
                    display=display,
                    words_sequence=spotter.words_sequence,
                    candidate_word=candidate_word,
                    candidate_conf=candidate_conf,
                    top5=last_top5,
                    translated_sentence=translated_sentence,
                    is_translating=is_translating,
                    translation_duration=translation_duration,
                    tts_enabled=tts_enabled,
                    is_speaking=is_speaking,
                    auto_delay=args.auto_delay,
                    time_since_last_word=time_since_last_word,
                )

                cv2.imshow(WINDOW_NAME, display)
            else:
                time.sleep(0.005)

    finally:
        translation_queue.put(None)
        camera.release()
        detector.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
