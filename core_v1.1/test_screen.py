"""
=============================================================================
BLIND TO BRIGHT - CÔNG CỤ TEST MÀN HÌNH OLED ESP32 (TEST SCREEN)
=============================================================================
Script gửi văn bản (tiếng Việt có dấu / không dấu) trực tiếp lên màn hình OLED
SSD1306 của ESP32 qua HTTP POST endpoint '/oled' (Port 80).

Hỗ trợ các nhãn tiêu đề hiển thị:
- Tiền tố 'SIGN:' -> Tiêu đề OLED: 'KY HIEU -> LOA'
- Tiền tố 'MIC:'  -> Tiêu đề OLED: 'MICROPHONE -> CHU'
- Mặc định        -> Tiêu đề OLED: 'THONG TIN'
=============================================================================
"""

import argparse
import sys
import time
import unicodedata
from pathlib import Path
import requests

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)
    sys.stderr.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)
except Exception:
    pass

ROOT_DIR = Path(__file__).resolve().parent.parent


def remove_vietnamese_accents(text: str) -> str:
    """
    Chuyển đổi tiếng Việt có dấu thành không dấu chuẩn để hiển thị rõ nét trên
    font ASCII chuẩn của thư viện Adafruit_SSD1306, tránh bị lỗi ký tự lạ (rác UTF-8).
    """
    # Thay thế chữ Đ, đ trước
    text = text.replace("Đ", "D").replace("đ", "d")
    # Tách dấu tổ hợp (NFD) và lọc bỏ các ký tự dấu
    nfkd = unicodedata.normalize("NFKD", text)
    clean = "".join([c for c in nfkd if not unicodedata.combining(c)])
    return clean


def send_to_oled(ip: str, text: str, raw_utf8: bool = False, timeout: float = 3.0) -> bool:
    """
    Gửi chuỗi văn bản lên ESP32 qua HTTP POST /oled.
    """
    url = f"http://{ip}/oled"

    if not raw_utf8:
        # Nếu không ép gửi UTF-8 thô, tự động chuẩn hóa tiếng Việt không dấu để chữ hiển thị đẹp nhất
        payload = remove_vietnamese_accents(text)
    else:
        payload = text

    try:
        session = requests.Session()
        session.trust_env = False
        r = session.post(url, data=payload.encode("utf-8"), timeout=timeout)
        if r.status_code == 200:
            return True
        else:
            print(f"[Lỗi] ESP32 trả về mã lỗi HTTP {r.status_code}")
            return False
    except requests.exceptions.RequestException as e:
        print(f"[Lỗi kết nối] Không thể gửi tới {url}: {e}")
        return False


def run_demo(ip: str):
    """Chạy bài test tự động hiển thị lần lượt các câu mẫu tiếng Việt."""
    samples = [
        ("SIGN: XIN CHÀO", "Chào mừng (Ký hiệu)"),
        ("SIGN: HÔM NAY TÔI KHỎE", "Cử chỉ hội thoại"),
        ("MIC: BẠN CÓ KHỎE KHÔNG?", "Giọng nói nhận diện"),
        ("SIGN: CẢM ƠN RẤT NHIỀU", "Lời cảm ơn"),
        ("BLIND TO BRIGHT v1.1", "Thông tin hệ thống"),
    ]

    print("\n" + "=" * 65)
    print(f"🚀 BẮT ĐẦU CHẠY BÀI TEST TỰ ĐỘNG LÊN MÀN HÌNH OLED ({ip})")
    print("=" * 65)

    for i, (msg, desc) in enumerate(samples, 1):
        clean_msg = remove_vietnamese_accents(msg)
        print(f"\n[{i}/{len(samples)}] Đang gửi: \"{msg}\" ({desc})")
        print(f"      -> Chữ hiển thị trên OLED: \"{clean_msg}\"")
        ok = send_to_oled(ip, msg)
        if ok:
            print("      ✅ Đã hiển thị thành công trên màn hình OLED!")
        else:
            print("      ❌ Gửi thất bại!")
        time.sleep(2.5)

    print("\n" + "=" * 65)
    print("🎉 ĐÃ HOÀN TẤT TOÀN BỘ BÀI TEST DEMO MÀN HÌNH OLED!")
    print("=" * 65 + "\n")


def interactive_mode(ip: str, raw_utf8: bool = False):
    """Chế độ nhập tay từ bàn phím để test chữ trực tiếp lên màn hình."""
    print("\n" + "=" * 65)
    print(f"⌨️  CHẾ ĐỘ NHẬP CHỮ TRỰC TIẾP LÊN MÀN HÌNH OLED ({ip})")
    print("  • Nhập bất kỳ câu gì (tiếng Việt có dấu hoặc không dấu) rồi ấn ENTER.")
    print("  • Thêm tiền tố 'SIGN: <chữ>' để hiện tiêu đề 'KY HIEU -> LOA'.")
    print("  • Thêm tiền tố 'MIC: <chữ>'  để hiện tiêu đề 'MICROPHONE -> CHU'.")
    print("  • Gõ 'demo' để chạy bài test tự động.")
    print("  • Gõ 'exit' hoặc 'quit' để thoát.")
    print("=" * 65 + "\n")

    while True:
        try:
            user_input = input("👉 Nhập chữ muốn hiện lên màn hình: ").strip()
            if not user_input:
                continue
            if user_input.lower() in ["exit", "quit", "q"]:
                print("Đã thoát chế độ test.")
                break
            if user_input.lower() == "demo":
                run_demo(ip)
                continue

            clean = remove_vietnamese_accents(user_input) if not raw_utf8 else user_input
            print(f"   Đang gửi: \"{clean}\" ...")
            ok = send_to_oled(ip, user_input, raw_utf8=raw_utf8)
            if ok:
                print("   ✅ Đã hiển thị lên màn hình OLED thành công!\n")
            else:
                print("   ❌ Không gửi được tới ESP32!\n")
        except (KeyboardInterrupt, EOFError):
            print("\nĐã thoát.")
            break


def main():
    parser = argparse.ArgumentParser(description="Test hiển thị chữ tiếng Việt lên màn hình OLED ESP32")
    parser.add_argument("text", nargs="?", default=None, help="Đoạn văn bản muốn gửi lên màn hình (nếu để trống sẽ vào menu/demo)")
    parser.add_argument("--ip", default=None, help="Địa chỉ IP của ESP32 (mặc định đọc từ configs/esp_ip.txt)")
    parser.add_argument("--demo", action="store_true", help="Chạy bài test tự động các câu mẫu")
    parser.add_argument("--raw", action="store_true", help="Gửi UTF-8 thô (không loại bỏ dấu tiếng Việt)")
    args = parser.parse_args()

    # 1. Lấy địa chỉ IP
    ip = args.ip
    if not ip:
        ip_file = ROOT_DIR / "configs" / "esp_ip.txt"
        if ip_file.is_file():
            ip = ip_file.read_text(encoding="utf-8").strip() or None
    ip = ip or "192.168.100.176"

    print(f"\n[ESP32 Screen Tester] Đang kết nối tới ESP32 tại: http://{ip}")

    # 2. Kiểm tra kết nối nhanh tới ESP32
    session = requests.Session()
    session.trust_env = False
    try:
        r = session.get(f"http://{ip}/status", timeout=2.0)
        if r.status_code == 200:
            print(f"[Trạng thái ESP32] Kết nối OK! Phản hồi từ thiết bị:\n{r.text.strip()}\n")
        else:
            print(f"[Cảnh báo] Cổng /status trả về HTTP {r.status_code}")
    except Exception as e:
        print(f"[Cảnh báo kết nối] Không ping được /status ({e}). Vẫn tiếp tục thử gửi...")

    # 3. Điều hướng xử lý
    if args.text:
        # Gửi trực tiếp văn bản từ tham số dòng lệnh
        print(f"Đang gửi nội dung: \"{args.text}\" ...")
        ok = send_to_oled(ip, args.text, raw_utf8=args.raw)
        if ok:
            print(f"✅ Đã hiển thị \"{remove_vietnamese_accents(args.text)}\" lên màn hình OLED thành công!")
        else:
            print("❌ Gửi thất bại. Hãy kiểm tra lại kết nối mạng và IP của ESP32.")
    elif args.demo:
        run_demo(ip)
    else:
        interactive_mode(ip, raw_utf8=args.raw)


if __name__ == "__main__":
    main()
