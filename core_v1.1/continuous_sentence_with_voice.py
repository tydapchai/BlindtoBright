import argparse
from collections import deque
import json
import os
import queue
import sys
import threading
import time
import unicodedata
import warnings
from pathlib import Path

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
from decoder import is_idle_word
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


def remove_vietnamese_accents(text: str) -> str:
    """
    Chuyển đổi tiếng Việt có dấu thành không dấu chuẩn để hiển thị rõ nét trên
    font ASCII của màn hình OLED SSD1306 ESP32.
    """
    text = text.replace("Đ", "D").replace("đ", "d")
    nfkd = unicodedata.normalize("NFKD", text)
    clean = "".join([c for c in nfkd if not unicodedata.combining(c)])
    return clean


def resolve_esp_ip(target_ip=None, timeout=0.15):
    """Tự động quét và tìm IP ESP32 trong mạng nội bộ nếu IP mặc định không phản hồi."""
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
        "192.168.100.176", "10.3.71.144", "10.245.192.219", "192.168.4.1",
        "192.168.1.100", "192.168.1.101", "192.168.1.102", "192.168.1.105"
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

    fallback = target_ip or "192.168.100.176"
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
    translated_sentence,
    is_translating,
    translation_duration,
    tts_enabled,
    is_speaking,
    is_recording,
    recorded_frames,
    max_frames,
    sentence_idle,
    time_since_last_word,
    is_guard_phase,
    guard_remain,
    is_recording_audio=False,
    stt_status="",
    stt_text="",
    stt_working=False,
):
    """
    Giao diện VSL trực quan:
    - Giữ lại: Thanh trạng thái trên cùng, Bảng chuỗi từ khóa đã nhận (Hộp 1), Hộp câu tiếng Việt hoàn chỉnh (Hộp 2), Hộp trạng thái âm thanh & phím tắt (Hộp 3).
    - Đã bỏ: Khung bảng Top 5 dự đoán bên phải (giúp màn hình bên phải hoàn toàn thông thoáng cho camera).
    """
    h, w = display.shape[:2]

    # 1. Top Bar (Thanh trạng thái trên cùng)
    top_bar_h = 42
    overlay = display.copy()
    cv2.rectangle(overlay, (0, 0), (w, top_bar_h), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.75, display, 0.25, 0, display)
    cv2.line(display, (0, top_bar_h), (w, top_bar_h), (50, 50, 50), 1)

    # 2. Left Container (Hộp chuỗi từ & Hộp câu hoàn chỉnh)
    # Vì đã bỏ bảng Top 5 bên phải, vùng hiển thị bên trái có kích thước vừa vặn (580px),
    # để trống toàn bộ bên phải (từ x=600 đến 960) cho người dùng nhìn rõ tay & cử chỉ camera.
    left_x = 15
    left_w = 580

    # Hộp 2A: Chuỗi từ khóa nhận diện (Gloss stream & Auto-REC)
    box_gloss_y = top_bar_h + 15
    box_gloss_h = 100
    box_gloss_overlay = display.copy()
    cv2.rectangle(box_gloss_overlay, (left_x, box_gloss_y), (left_x + left_w, box_gloss_y + box_gloss_h), (16, 22, 30), -1)
    cv2.addWeighted(box_gloss_overlay, 0.88, display, 0.12, 0, display)
    border_gloss_color = (0, 100, 255) if is_recording else (0, 200, 240)
    cv2.rectangle(display, (left_x, box_gloss_y), (left_x + left_w, box_gloss_y + box_gloss_h), border_gloss_color, 2 if is_recording else 1)

    # Hộp 2B: Câu tiếng Việt hoàn chỉnh (Gemini AI Translation)
    box_trans_y = box_gloss_y + box_gloss_h + 12
    box_trans_h = 135
    box_trans_overlay = display.copy()
    cv2.rectangle(box_trans_overlay, (left_x, box_trans_y), (left_x + left_w, box_trans_y + box_trans_h), (20, 28, 22), -1)
    cv2.addWeighted(box_trans_overlay, 0.90, display, 0.10, 0, display)
    border_trans_color = (0, 255, 150) if translated_sentence else (80, 120, 90)
    cv2.rectangle(display, (left_x, box_trans_y), (left_x + left_w, box_trans_y + box_trans_h), border_trans_color, 2)

    # Hộp 2C: Trạng thái Âm thanh & Gợi ý nhanh
    box_sub_y = box_trans_y + box_trans_h + 12
    box_sub_h = 58
    box_sub_overlay = display.copy()
    cv2.rectangle(box_sub_overlay, (left_x, box_sub_y), (left_x + left_w, box_sub_y + box_sub_h), (18, 18, 18), -1)
    cv2.addWeighted(box_sub_overlay, 0.85, display, 0.15, 0, display)
    cv2.rectangle(display, (left_x, box_sub_y), (left_x + left_w, box_sub_y + box_sub_h), (70, 70, 70), 1)

    # 3. Bottom Guide Bar
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
    elif is_speaking:
        draw.ellipse((18, 13, 30, 25), fill=(180, 100, 255))
        rec_str = "LOA ĐANG PHÁT ÂM THANH GIỌNG ĐỌC..."
        rec_color = (210, 150, 255)
    elif is_recording:
        draw.ellipse((18, 13, 30, 25), fill=(255, 50, 50))
        rec_str = f"🔴 ĐANG THU CỬ CHỈ... ({recorded_frames}/{max_frames} frames)"
        rec_color = (255, 90, 90)
    elif is_guard_phase:
        draw.ellipse((18, 13, 30, 25), fill=(255, 160, 0))
        rec_str = f"⏳ CHỜ TĨNH TAY ({guard_remain:.1f}s) ĐỂ BẮT ĐẦU CÂU MỚI..."
        rec_color = (255, 190, 80)
    elif len(words_sequence) > 0:
        draw.ellipse((18, 13, 30, 25), fill=(0, 230, 115))
        rec_str = f"ĐANG TRONG CÂU ({len(words_sequence)} từ): Làm từ tiếp theo hoặc dừng tay để ngắt câu"
        rec_color = (120, 255, 180)
    else:
        draw.ellipse((18, 13, 30, 25), fill=(80, 220, 120))
        rec_str = "SẴN SÀNG: Hãy thực hiện cử chỉ trước camera..."
        rec_color = (180, 240, 200)

    draw.text((38, 10), rec_str, font=FONT_BOLD, fill=rec_color)
    draw.text((w - 240, 11), f"Chế độ: AUTO-REC ({max_frames}f)", font=FONT_HINT, fill=(0, 215, 255))

    # --- HỘP 2A: CHUỖI TỪ KHÓA (GLOSS STREAM & TIẾN TRÌNH NGẮT CÂU) ---
    valid_words = [w for w in words_sequence if not is_idle_word(w)]
    header_gloss = f"1. TỪ KHÓA KÝ HIỆU ({len(valid_words)} từ)"
    draw.text((left_x + 12, box_gloss_y + 8), header_gloss, font=FONT_SMALL, fill=(0, 215, 255))

    if is_recording:
        draw.text((left_x + left_w - 280, box_gloss_y + 8), f"Đang thu từ #{len(valid_words) + 1} ({recorded_frames}/{max_frames}f)...", font=FONT_SMALL, fill=(255, 120, 120))
    elif len(valid_words) > 0 and sentence_idle > 0:
        remain = max(0.0, sentence_idle - time_since_last_word)
        timer_txt = f"Ngắt câu sau {remain:.1f}s dừng tay (hoặc [ENTER])"
        draw.text((left_x + left_w - 340, box_gloss_y + 8), timer_txt, font=FONT_SMALL, fill=(0, 255, 200))

        bar_x = left_x + 12
        bar_y = box_gloss_y + box_gloss_h - 13
        bar_w = left_w - 24
        bar_h = 5
        draw.rectangle([bar_x, bar_y, bar_x + bar_w, bar_y + bar_h], fill=(30, 40, 50))
        pct = min(1.0, max(0.0, remain / sentence_idle))
        fill_w = int(bar_w * pct)
        if fill_w > 0:
            bar_color = (0, 255, 120) if pct > 0.5 else ((255, 200, 0) if pct > 0.25 else (255, 80, 80))
            draw.rectangle([bar_x, bar_y, bar_x + fill_w, bar_y + bar_h], fill=bar_color)

    if valid_words and len(valid_words) > 0:
        seq_display = "  ➔  ".join(f"[{w.upper()}]" for w in valid_words[-6:])
        draw.text((left_x + 12, box_gloss_y + 36), seq_display, font=FONT_BOLD, fill=(255, 225, 70))
    elif is_recording:
        draw.text((left_x + 12, box_gloss_y + 40), "(Đang thực hiện cử chỉ từ thứ 1... Đang ghi nhận)", font=FONT_HINT, fill=(255, 180, 100))
    else:
        draw.text((left_x + 12, box_gloss_y + 40), "(Chưa có từ nào. Hãy làm cử chỉ trước camera để bắt đầu câu...)", font=FONT_HINT, fill=(150, 165, 180))

    # --- HỘP 2B: CÂU TIẾNG VIỆT HOÀN CHỈNH ---
    draw.text((left_x + 12, box_trans_y + 8), "2. CÂU TIẾNG VIỆT HOÀN CHỈNH (GEMINI AI SUY LUẬN):", font=FONT_SMALL, fill=(0, 255, 170))

    if is_translating:
        draw.text((left_x + 12, box_trans_y + 45), "Đang phân tích ngữ pháp & ghép câu tự nhiên...", font=FONT_BIG, fill=(255, 215, 0))
    elif translated_sentence:
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
        draw.text((left_x + 12, box_trans_y + 78), f'(Làm các từ liên tiếp. Dừng tay {sentence_idle:.1f}s để hệ thống tự động ngắt câu)', font=FONT_SMALL, fill=(100, 130, 110))

    # --- HỘP 2C: ÂM THANH & TRẠNG THÁI (TTS + MIC STT) ---
    tts_status_str = "BẬT (Laptop Speaker)" if tts_enabled else "TẮT"
    tts_color = (0, 255, 150) if tts_enabled else (160, 160, 160)
    draw.text((left_x + 12, box_sub_y + 5), f"TTS: {tts_status_str}", font=FONT_HINT, fill=tts_color)

    # Hiển thị trạng thái Mic/STT
    if is_recording_audio:
        mic_icon_color = (255, 60, 60)
        draw.ellipse((left_x + 200, box_sub_y + 7, left_x + 212, box_sub_y + 19), fill=mic_icon_color)
        draw.text((left_x + 218, box_sub_y + 5), "MIC ĐANG THU ÂM...", font=FONT_HINT, fill=(255, 100, 100))
    elif stt_working:
        draw.text((left_x + 200, box_sub_y + 5), "⏳ ĐANG NHẬN DẠNG GIỌNG NÓI...", font=FONT_HINT, fill=(255, 215, 0))
    elif stt_status:
        draw.text((left_x + 200, box_sub_y + 5), stt_status[:50], font=FONT_HINT, fill=(100, 220, 255))

    # Hiển thị câu nghe được gần nhất (nếu có)
    if stt_text and not is_recording_audio:
        stt_display = f'🗣️ "{stt_text}"'
        draw.text((left_x + 12, box_sub_y + 28), stt_display[:60], font=FONT_SMALL, fill=(180, 230, 255))
    else:
        draw.text((left_x + 12, box_sub_y + 28), "[M] Thu âm người nói | [ENTER] Ngắt & Dịch | [C] Xóa câu | [BACKSPACE] Xóa từ", font=FONT_SMALL, fill=(200, 200, 200))

    # --- BOTTOM GUIDE BAR ---
    guide_str = "[M] Thu âm  |  [ENTER] Dịch  |  [R] Xoay  |  [C] Xóa câu  |  [BACKSPACE] Xóa từ  |  [T] TTS  |  [Q] Thoát"
    draw.text((25, bot_y + 10), guide_str, font=FONT_HINT, fill=(220, 220, 220))

    display[:] = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)


def main():
    parser = argparse.ArgumentParser(description="BlindtoBright 2-Way: VSL Auto-REC 25f + Voice Recording (Mic ESP32/Laptop → Gemini STT → OLED)")
    parser.add_argument("--checkpoint", default=str(ROOT_DIR / "models" / "best_vsl_model_ctr_gcn.pth"))
    parser.add_argument("--labels", default=str(ROOT_DIR / "core_v1.1" / "label_map_472_10w.json"))
    parser.add_argument("--camera", default="esp", help="'0' (webcam laptop), 'esp' (camera ESP32), hoặc URL http")
    parser.add_argument("--esp-ip", default="192.168.100.176", help="Địa chỉ IP của ESP32 (mặc định: 192.168.100.176)")
    parser.add_argument("--device", default=None, choices=["cpu", "cuda"])

    # Cấu hình Auto-REC chuẩn debug với 25 frames
    parser.add_argument("--frames", type=int, default=25, help="Độ dài phân đoạn cử chỉ tối đa (mặc định 25 frames)")
    parser.add_argument("--motion-start", type=float, default=0.005, help="Ngưỡng vận tốc bắt đầu thu cử chỉ (mặc định 0.005)")
    parser.add_argument("--motion-stop", type=float, default=0.0035, help="Ngưỡng vận tốc dừng tĩnh tay (mặc định 0.0035)")
    parser.add_argument("--idle-frames", type=int, default=4, help="Số frames tĩnh tay để dừng thu từ (mặc định 4 frames)")

    parser.add_argument("--conf", type=float, default=0.35, help="Ngưỡng tin cậy tối thiểu để chốt từ (mặc định 0.35)")
    parser.add_argument("--margin", type=float, default=0.08, help="Ngưỡng chênh lệch Top1 - Top2 tối thiểu (mặc định 0.08)")
    parser.add_argument("--sentence-idle", type=float, default=1.8, help="Thời gian dừng tay (giây) để tự động ngắt câu và dịch (mặc định 1.8s, 0 để tắt)")
    parser.add_argument("--start-idle", type=float, default=0.8, help="Thời gian tĩnh tay (giây) tối thiểu trước khi nhận câu mới (mặc định 0.8s)")
    parser.add_argument("--no-tts", action="store_true", help="Tắt tính năng phát âm thanh giọng đọc")
    parser.add_argument("--rotate-180", action="store_true", help="Xoay ngược khung hình camera 180 độ")
    parser.add_argument("--mic-source", default="esp", choices=["esp", "laptop"], help="Nguồn micro: 'esp' (micro INMP441 ESP32 port 82) hoặc 'laptop' (micro laptop qua sounddevice)")
    parser.add_argument("--mic-duration", type=float, default=5.0, help="Thời lượng tối đa mỗi lần ghi âm (giây, mặc định 5s)")
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

    window_size = int(args.frames)
    print(f"[Model] Phân đoạn cử chỉ cấu hình chuẩn Auto-REC: window_size={window_size} frames")

    # Khởi tạo Gemini Client
    gemini_client = GeminiClient()
    if gemini_client.api_keys:
        print(f"[Gemini] Đã sẵn sàng {len(gemini_client.api_keys)} API Keys cho dịch câu và TTS.")
    else:
        print("[Gemini Warning] Không tìm thấy API Keys. Sẽ dùng phương thức ghép từ dự phòng.")

    # Xử lý Camera & ESP32 IP
    ip_file = ROOT_DIR / "configs" / "esp_ip.txt"
    target_ip = args.esp_ip
    if not target_ip and ip_file.is_file():
        target_ip = ip_file.read_text(encoding="utf-8").strip() or None
    target_ip = target_ip or "192.168.100.176"
    actual_ip = target_ip

    if str(args.camera).lower() in ["esp", "esp32", "cam"]:
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

    camera = LatestFrameCamera(source, rotate_180=args.rotate_180)

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

    # Quản lý Hàng đợi gửi OLED ESP32 bất đồng bộ
    oled_queue = queue.Queue(maxsize=10)

    def oled_worker():
        session = requests.Session()
        session.trust_env = False
        while True:
            text = oled_queue.get()
            if text is None:
                oled_queue.task_done()
                break
            if actual_ip:
                clean_payload = remove_vietnamese_accents(text)
                try:
                    session.post(f"http://{actual_ip}/oled", data=clean_payload.encode("utf-8"), timeout=1.5)
                except Exception:
                    pass
            oled_queue.task_done()

    oled_thread = threading.Thread(target=oled_worker, daemon=True)
    oled_thread.start()

    def send_oled_async(msg: str):
        try:
            if oled_queue.full():
                try:
                    oled_queue.get_nowait()
                    oled_queue.task_done()
                except Exception:
                    pass
            oled_queue.put_nowait(msg)
        except Exception:
            pass

    send_oled_async("SIGN: SAN SANG...")

    # ===========================================================================
    # CHIỀU 2: LUỒNG GHI ÂM & NHẬN DẠNG GIỌNG NÓI (SPEECH-TO-TEXT)
    # ===========================================================================
    is_recording_audio = False
    audio_buffer = bytearray()
    stt_text = ""           # Câu nghe được gần nhất từ người nói
    stt_status = ""         # Trạng thái hiển thị trên HUD ("ĐANG THU ÂM...", "ĐANG DỊCH...", v.v.)
    stt_working = False     # True khi đang gọi Gemini STT
    mic_start_time = 0.0    # Thời điểm bắt đầu ghi âm (cho auto-stop)

    # HTTP session riêng cho mic streaming (tránh xung đột với session OLED/TTS)
    mic_http_session = requests.Session()
    mic_http_session.trust_env = False

    # Luồng hút âm thanh liên tục từ Mic I2S ESP32 (Port 82)
    def audio_fetch_worker_esp():
        nonlocal audio_buffer, is_recording_audio
        while True:
            if is_recording_audio and actual_ip:
                try:
                    resp = mic_http_session.get(
                        f"http://{actual_ip}:82/mic", stream=True, timeout=2.0
                    )
                    if resp.status_code == 200:
                        for chunk in resp.iter_content(chunk_size=1024):
                            if not is_recording_audio:
                                break
                            if chunk:
                                audio_buffer.extend(chunk)
                except Exception:
                    time.sleep(0.5)
            else:
                time.sleep(0.1)

    # Luồng ghi âm từ Mic Laptop (qua sounddevice) - dùng khi --mic-source=laptop
    def audio_fetch_worker_laptop():
        nonlocal audio_buffer, is_recording_audio
        SAMPLE_RATE = 16000
        CHUNK_DURATION = 0.25  # 250ms mỗi lần đọc
        CHUNK_SAMPLES = int(SAMPLE_RATE * CHUNK_DURATION)
        while True:
            if is_recording_audio and AUDIO_AVAILABLE:
                try:
                    recording = sd.rec(
                        CHUNK_SAMPLES, samplerate=SAMPLE_RATE, channels=1, dtype="int16"
                    )
                    sd.wait()
                    if is_recording_audio:
                        audio_buffer.extend(recording.tobytes())
                except Exception:
                    time.sleep(0.3)
            else:
                time.sleep(0.1)

    # Khởi chạy worker phù hợp với nguồn micro
    if args.mic_source == "laptop" and AUDIO_AVAILABLE:
        mic_worker_thread = threading.Thread(target=audio_fetch_worker_laptop, daemon=True)
        mic_source_label = "Laptop Microphone (sounddevice)"
    else:
        mic_worker_thread = threading.Thread(target=audio_fetch_worker_esp, daemon=True)
        mic_source_label = f"ESP32 INMP441 (http://{actual_ip}:82/mic)"
    mic_worker_thread.start()
    print(f"[Mic] Luồng thu âm sẵn sàng: {mic_source_label}")

    # Hàm xử lý transcribe âm thanh chạy nền (gọi Gemini STT)
    def transcribe_audio_job(pcm_bytes):
        nonlocal stt_text, stt_status, stt_working
        if len(pcm_bytes) < 4000:
            stt_status = "Không nghe gì (âm thanh quá ngắn)"
            stt_working = False
            return
        stt_working = True
        stt_status = "ĐANG NHẬN DẠNG GIỌNG NÓI..."
        t0 = time.time()
        try:
            text = gemini_client.transcribe_audio(pcm_bytes)
            dt = time.time() - t0
            if text:
                stt_text = text
                stt_status = f'NGƯỜI NÓI ({dt:.1f}s): "{text}"'
                print(f'\n[STT] ✅ Nhận dạng thành công ({dt:.2f}s): "{text}"')

                # Gửi lên OLED ESP32 cho người khiếm thính đọc
                clean_ascii = remove_vietnamese_accents(text)
                send_oled_async(f"MIC: {clean_ascii}")
            else:
                stt_status = "Không nghe rõ (Gemini không trả text)"
                print(f"[STT] ⚠️ Gemini không trả về text.")
        except Exception as e:
            stt_status = f"Lỗi STT: {e}"
            print(f"[STT] ❌ Lỗi: {e}")
        finally:
            stt_working = False

    # Quản lý thuật toán thu từ Auto-REC (Khớp 100% logic bản debug_word.py)
    is_recording = False
    gesture_frames = []
    auto_idle_count = 0
    record_start_time = 0.0

    words_sequence = []
    candidate_word = None
    candidate_conf = 0.0
    last_top5 = []
    previous_landmarks = None
    last_frame_id = -1

    # Quản lý ngắt câu thông minh & thời gian tĩnh tay (Idle Boundary & Guard)
    last_word_commit_time = 0.0
    sentence_already_translated = False
    guard_until_time = time.time() + args.start_idle

    # Quản lý dịch câu & TTS
    tts_enabled = not args.no_tts
    is_speaking = False
    is_translating = False
    translated_sentence = ""
    translation_duration = 0.0

    # Hàng đợi dịch câu nền
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
            print(f"[AI Translator] >>> KẾT QUẢ DỊCH: \"{result_sentence}\" (Thời gian: {dt:.2f}s)\n")

            send_oled_async(f"SIGN: {result_sentence}")

            if tts_enabled and gemini_client.api_keys and result_sentence:
                try:
                    is_speaking = True
                    pcm = gemini_client.generate_speech(result_sentence)
                    if pcm and AUDIO_AVAILABLE:
                        audio_data = np.frombuffer(pcm, dtype=np.int16)
                        sd.play(audio_data, samplerate=24000)
                        sd.wait()
                    elif actual_ip and pcm:
                        try:
                            requests.post(f"http://{actual_ip}/play", data=pcm, timeout=10)
                        except Exception:
                            pass
                except Exception as tts_err:
                    print(f"[TTS Warning] Không phát được giọng đọc: {tts_err}")
                finally:
                    is_speaking = False

            translation_queue.task_done()

    worker_thread = threading.Thread(target=translation_worker, daemon=True)
    worker_thread.start()

    print("\n" + "=" * 75)
    print("  [BLINDTOBRIGHT 2-WAY: SIGN→SPEECH + VOICE→TEXT (FULL DUPLEX)]")
    print(f"  - Checkpoint           : {checkpoint_path.name}")
    print(f"  - Cửa sổ phân đoạn     : window_size = {window_size} frames (Chuẩn 25 frames)")
    print(f"  - Thuật toán thu từ    : Auto-REC giống 100% debug_word.py")
    print(f"  - Ngưỡng bắt đầu       : motion >= {args.motion_start}")
    print(f"  - Ngưỡng dừng tay      : motion < {args.motion_stop} trong {args.idle_frames} frames hoặc đủ {window_size} frames")
    print(f"  - Ngưỡng chốt từ       : conf >= {args.conf:.2f} | margin >= {args.margin:.2f}")
    print(f"  - Tĩnh tay bắt đầu câu : {args.start_idle:.1f}s")
    print(f"  - Tự động ngắt câu sau : {args.sentence_idle:.1f}s dừng tay (hoặc bấm [ENTER])")
    print(f"  - Giọng đọc (TTS)      : {'BẬT' if tts_enabled else 'TẮT'}")
    print(f"  - Màn hình ESP32 OLED  : http://{actual_ip}/oled")
    print(f"  - Nguồn Micro (STT)   : {mic_source_label}")
    print(f"  - Ghi âm tối đa       : {args.mic_duration:.0f}s mỗi lần")
    print("  - Phím tắt:")
    print("    * [M]        : Bật/Tắt ghi âm người nói (STT → OLED)")
    print("    * [ENTER]    : Ngắt câu và dịch ngay lập tức")
    print("    * [BACKSPACE]: Xóa từ cuối cùng (nếu nhận nhầm)")
    print("    * [C]        : Xóa sạch câu cũ để bắt đầu câu mới")
    print("    * [R]        : Xoay ngược camera 180 độ")
    print("    * [T]        : Bật / Tắt giọng đọc (TTS)")
    print("    * [Q]        : Thoát")
    print("=" * 75 + "\n")

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

                # [ENTER]: Ngắt câu & dịch ngay lập tức
                if key in (13, 10):
                    if len(words_sequence) > 0 and not is_translating:
                        sentence_already_translated = True
                        words_to_translate = list(words_sequence)
                        print("\n[Action] [ENTER] Bấm phím ngắt câu & kích hoạt dịch ngay!")
                        translation_queue.put(words_to_translate)
                        words_sequence.clear()
                        candidate_word = None
                        candidate_conf = 0.0
                        guard_until_time = time.time() + args.start_idle

                # [C]: Xóa câu và chuỗi từ để làm lại câu mới
                elif key in (ord("c"), ord("C")):
                    is_recording = False
                    gesture_frames.clear()
                    auto_idle_count = 0
                    words_sequence.clear()
                    candidate_word = None
                    candidate_conf = 0.0
                    last_top5.clear()
                    translated_sentence = ""
                    sentence_already_translated = False
                    guard_until_time = time.time() + args.start_idle
                    send_oled_async("SIGN: SAN SANG...")
                    print("\n[Action] Đã xóa toàn bộ câu và chuỗi từ. Sẵn sàng cho câu mới!")

                # [BACKSPACE] hoặc [DEL]: Xóa từ cuối cùng
                elif key in (8, 127):
                    if words_sequence:
                        removed = words_sequence.pop()
                        last_word_commit_time = time.time()
                        sentence_already_translated = False
                        seq_str = " ➔ ".join(f"[{w.upper()}]" for w in words_sequence) if words_sequence else "(Trống)"
                        print(f"\n[Action] Đã xóa từ cuối: [{removed}] | Chuỗi còn lại: {seq_str}")
                        if words_sequence:
                            send_oled_async(f"SIGN: {' '.join(words_sequence)}")
                        else:
                            send_oled_async("SIGN: SAN SANG...")
                    else:
                        print("\n[Action] Chuỗi từ đang trống, không có gì để xóa.")

                # [T]: Bật/Tắt giọng đọc TTS
                elif key in (ord("t"), ord("T")):
                    tts_enabled = not tts_enabled
                    print(f"\n[TTS] Đã {'BẬT' if tts_enabled else 'TẮT'} giọng đọc âm thanh.")

                # [M]: Bật/Tắt ghi âm giọng nói người đối diện (Speech-to-Text)
                elif key in (ord("m"), ord("M")):
                    is_recording_audio = not is_recording_audio
                    if is_recording_audio:
                        audio_buffer = bytearray()
                        stt_status = f"🎙️ ĐANG THU ÂM... (bấm [M] để dừng, tối đa {args.mic_duration:.0f}s)"
                        print(f"\n[Mic] 🎙️ BẮT ĐẦU GHI ÂM từ {mic_source_label}")
                        mic_start_time = time.time()
                    else:
                        stt_status = "⏳ ĐANG GỬI ÂM THANH CHO GEMINI AI NHẬN DẠNG..."
                        print(f"[Mic] ⏹️ DỪNG GHI ÂM. Đã thu {len(audio_buffer)} bytes. Đang transcribe...")
                        threading.Thread(
                            target=transcribe_audio_job,
                            args=(bytes(audio_buffer),),
                            daemon=True,
                        ).start()

                # [R]: Xoay ngược khung hình camera 180 độ
                elif key in (ord("r"), ord("R")):
                    camera.toggle_rotate()

                # [Q] hoặc [ESC]: Thoát
                elif key in (ord("q"), ord("Q"), 27):
                    break

            # Auto-stop ghi âm sau mic_duration giây
            if is_recording_audio and (time.time() - mic_start_time) >= args.mic_duration:
                is_recording_audio = False
                stt_status = "⏳ HẾT THỜI GIAN GHI ÂM. ĐANG NHẬN DẠNG..."
                print(f"[Mic] ⏹️ Tự động dừng ghi âm sau {args.mic_duration:.0f}s. Đã thu {len(audio_buffer)} bytes.")
                threading.Thread(
                    target=transcribe_audio_job,
                    args=(bytes(audio_buffer),),
                    daemon=True,
                ).start()

            if is_new:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = detector.process(rgb)
                landmarks = extract_landmarks(results)

                # Đo năng lượng chuyển động
                if previous_landmarks is None:
                    motion = 0.0
                else:
                    motion = motion_energy(landmarks, previous_landmarks)
                previous_landmarks = landmarks

                hand_detected = bool(results.left_hand_landmarks or results.right_hand_landmarks)
                now = time.time()

                # =========================================================
                # 1. KIỂM TRA PRE-SENTENCE IDLE GUARD (Tĩnh tay trước khi nhận câu)
                # =========================================================
                if now < guard_until_time:
                    is_guard_phase = True
                    guard_remain = max(0.0, guard_until_time - now)
                    can_start_gesture = False
                    if hand_detected and motion >= args.motion_start:
                        guard_until_time = now + args.start_idle
                else:
                    is_guard_phase = False
                    guard_remain = 0.0
                    can_start_gesture = True

                # =========================================================
                # 2. THUẬT TOÁN THU CỬ CHỈ AUTO-REC (KHỚP 100% DEBUG_WORD.PY)
                # =========================================================
                if not is_recording:
                    # Bắt đầu thu cử chỉ khi có tay và vận tốc >= motion_start
                    if can_start_gesture and hand_detected and motion >= args.motion_start:
                        is_recording = True
                        gesture_frames = [landmarks]
                        record_start_time = now
                        auto_idle_count = 0
                        print(f"\n[Auto-REC 25f] >>> Bắt đầu thu cử chỉ từ #{len(words_sequence) + 1}! (motion={motion:.4f})")
                else:
                    # Đang thu frames cử chỉ
                    gesture_frames.append(landmarks)
                    if motion < args.motion_stop or not hand_detected:
                        auto_idle_count += 1
                    else:
                        auto_idle_count = 0

                    # Kết thúc cử chỉ: Dừng tay >= 4 frames hoặc đủ window_size (25 frames)
                    if auto_idle_count >= args.idle_frames or len(gesture_frames) >= window_size:
                        is_recording = False
                        duration = time.time() - record_start_time
                        valid_frames = gesture_frames[:window_size]

                        if len(valid_frames) >= 8:
                            # Resample về 48 frames cho mô hình CTR-GCN
                            input_tensor = build_tensor(valid_frames, device, target_len=48)
                            with torch.no_grad():
                                logits = model(input_tensor)
                                probs = torch.softmax(logits, dim=-1)[0].cpu().numpy()

                            top_indices = np.argsort(probs)[::-1]
                            last_top5 = [(idx_to_class.get(int(idx), f"Class_{idx}"), float(probs[idx])) for idx in top_indices[:5]]

                            best_idx = int(top_indices[0])
                            best_word = idx_to_class.get(best_idx, f"Class_{best_idx}")
                            best_conf = float(probs[best_idx])
                            sec_conf = float(probs[top_indices[1]]) if len(top_indices) > 1 else 0.0
                            margin = best_conf - sec_conf

                            print("-" * 68)
                            print(f"[Auto-REC 25f] Thu {len(valid_frames)} frames ({duration:.2f}s) | Dự đoán: [{best_word}] ({best_conf*100:.1f}%) | Margin: +{margin*100:.1f}%")
                            top3_str = ", ".join([f"{w} ({c*100:.1f}%)" for w, c in last_top5[:3]])
                            print(f"               Top 3: {top3_str}")

                            # Tiêu chuẩn chốt từ hợp lệ
                            is_valid = (not is_idle_word(best_word)) and (best_conf >= args.conf) and (margin >= args.margin)
                            eval_now = time.time()

                            if is_valid:
                                # Tránh lặp từ liên tiếp do rung tay (< 0.8s)
                                if words_sequence and words_sequence[-1] == best_word and (eval_now - last_word_commit_time < 0.8):
                                    print(f"               ⚠️ Bỏ qua từ lặp tức thì: [{best_word}]")
                                else:
                                    words_sequence.append(best_word)
                                    last_word_commit_time = eval_now
                                    sentence_already_translated = False
                                    candidate_word = best_word
                                    candidate_conf = best_conf
                                    seq_str = " ➔ ".join(f"[{w.upper()}]" for w in words_sequence)
                                    print(f"               ✅ ĐÃ CHỐT TỪ: [{best_word.upper()}]")
                                    print(f"               📝 Chuỗi câu hiện tại ({len(words_sequence)} từ): {seq_str}")
                                    send_oled_async(f"SIGN: {' '.join(words_sequence)}")
                            else:
                                reason = "Nhãn Idle" if is_idle_word(best_word) else f"Chưa đạt ngưỡng (conf < {args.conf:.2f} hoặc margin < {args.margin:.2f})"
                                print(f"               ❌ Không chốt từ ({reason})")
                            print("-" * 68 + "\n")
                        else:
                            print(f"[Auto-REC] Cử chỉ quá ngắn ({len(valid_frames)} frames < 8 frames), bỏ qua.")

                        gesture_frames = []

                # =========================================================
                # 3. TỰ ĐỘNG NGẮT CÂU KHI IDLE ĐỦ THỜI GIAN (SENTENCE BOUNDARY)
                # =========================================================
                time_since_last_word = now - last_word_commit_time if last_word_commit_time > 0 else 0.0
                if (
                    len(words_sequence) > 0
                    and not is_recording
                    and not sentence_already_translated
                    and not is_translating
                    and args.sentence_idle > 0
                    and time_since_last_word >= args.sentence_idle
                ):
                    sentence_already_translated = True
                    words_to_translate = list(words_sequence)
                    print("\n" + "=" * 68)
                    print(f"⚡ [TỰ ĐỘNG NGẮT CÂU] Người dùng đã dừng tay {args.sentence_idle:.1f}s.")
                    print(f"   Chuỗi từ khóa: {' ➔ '.join(f'[{w.upper()}]' for w in words_to_translate)}")
                    print("=" * 68 + "\n")

                    translation_queue.put(words_to_translate)
                    words_sequence.clear()
                    candidate_word = None
                    candidate_conf = 0.0
                    guard_until_time = now + args.start_idle

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

                # Vẽ giao diện thông tin VSL & câu dịch hoàn chỉnh (đã bỏ bảng Top 5)
                draw_continuous_sentence_overlay(
                    display=display,
                    words_sequence=words_sequence,
                    candidate_word=candidate_word,
                    candidate_conf=candidate_conf,
                    translated_sentence=translated_sentence,
                    is_translating=is_translating,
                    translation_duration=translation_duration,
                    tts_enabled=tts_enabled,
                    is_speaking=is_speaking,
                    is_recording=is_recording,
                    recorded_frames=len(gesture_frames),
                    max_frames=window_size,
                    sentence_idle=args.sentence_idle,
                    time_since_last_word=time_since_last_word,
                    is_guard_phase=is_guard_phase,
                    guard_remain=guard_remain,
                    is_recording_audio=is_recording_audio,
                    stt_status=stt_status,
                    stt_text=stt_text,
                    stt_working=stt_working,
                )

                cv2.imshow(WINDOW_NAME, display)
            else:
                time.sleep(0.005)

    finally:
        translation_queue.put(None)
        oled_queue.put(None)
        camera.release()
        detector.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
