"""
=============================================================================
BLIND TO BRIGHT - ESP32 CAMERA FRAME & PERFORMANCE TESTER (v1.1)
=============================================================================
Công cụ chuyên dụng kiểm tra luồng video (stream), đo đạc hiệu năng FPS,
băng thông mạng, độ trễ và độ ổn định của camera ESP32 (hoặc Webcam Laptop).

Giao diện chuẩn hóa cố định trên Canvas 960x720 (đồng bộ hoàn toàn với main.py):
- Tự động căn chỉnh và phóng to mọi độ phân giải camera (320x240, 640x480, ...)
  vào khung hình 960x720, giữ nguyên tỉ lệ gốc (không bị méo hay vỡ giao diện).
- Đo FPS thời gian thực (FPS tức thời, FPS trung bình, FPS trượt 1s).
- Đo độ trễ giữa các frames (frame interval in ms) & phát hiện khựng hình (lag spikes).
- Đo băng thông mạng thực tế (Bitrate KB/s, Mbps) và dung lượng trung bình mỗi frame.
- Đọc thông số phần cứng từ ESP32 (/status): Wi-Fi RSSI, Free Heap, Free PSRAM.
- Biểu đồ thời gian thực (Sparkline latency graph) hiển thị độ ổn định mạng.
- Phím tắt:
    [S] : Chụp ảnh lưu khung hình gốc + ảnh có HUD vào thư mục saved_frames/
    [P] : Tạm dừng / Tiếp tục xem hình
    [H] : Ẩn / Hiện bảng thông số kỹ thuật (HUD)
    [R] : Đặt lại (Reset) các chỉ số thống kê
    [Q] hoặc [ESC] : Thoát và in bảng tổng kết chi tiết ra Terminal
=============================================================================
"""

import argparse
from collections import deque
from datetime import datetime
import os
import subprocess
import sys
import threading
import time

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import requests

# Kích thước Canvas chuẩn thống nhất toàn hệ thống Blind to Bright
CANVAS_WIDTH = 960
CANVAS_HEIGHT = 720

# Cấu hình Font chữ tiếng Việt hiển thị trên màn hình
try:
    FONT_TITLE = ImageFont.truetype("arialbd.ttf", 17)
    FONT_LARGE = ImageFont.truetype("arialbd.ttf", 26)
    FONT_TEXT = ImageFont.truetype("arial.ttf", 15)
    FONT_BOLD = ImageFont.truetype("arialbd.ttf", 15)
    FONT_SMALL = ImageFont.truetype("arial.ttf", 13)
    FONT_HINT = ImageFont.truetype("arial.ttf", 14)
except Exception:
    FONT_TITLE = ImageFont.load_default()
    FONT_LARGE = ImageFont.load_default()
    FONT_TEXT = ImageFont.load_default()
    FONT_BOLD = ImageFont.load_default()
    FONT_SMALL = ImageFont.load_default()
    FONT_HINT = ImageFont.load_default()


def resolve_esp_ip(target_ip: str | None = None) -> str:
    """Tự động tìm kiếm IP của ESP32 trên mạng nội bộ."""
    session = requests.Session()
    session.trust_env = False

    # 1. Thử IP mục tiêu nếu được truyền vào
    if target_ip:
        print(f"[Dò IP] Kiểm tra IP chỉ định: {target_ip} ...")
        try:
            r = session.get(f"http://{target_ip}/status", timeout=1.2)
            if r.status_code == 200 and "Blind to Bright" in r.text:
                print(f"[Dò IP] -> Xác nhận ESP32 hoạt động tại {target_ip}")
                return target_ip
        except Exception:
            print(f"[Dò IP] IP '{target_ip}' không phản hồi.")

    # 2. Kiểm tra mạng SoftAP của ESP (192.168.4.1)
    if target_ip != "192.168.4.1":
        print("[Dò IP] Kiểm tra mạng SoftAP mặc định: 192.168.4.1 ...")
        try:
            r = session.get("http://192.168.4.1/status", timeout=0.8)
            if r.status_code == 200 and "Blind to Bright" in r.text:
                print("[Dò IP] -> Tìm thấy ESP32 tại SoftAP: 192.168.4.1")
                return "192.168.4.1"
        except Exception:
            pass

    # 3. Quét ARP LAN tìm ESP MAC OUI
    ESP_MACS = (
        "14-c1-9f", "24-0a-c4", "24-6f-28", "24-dc-c3", "30-ae-a4", "3c-61-05", "3c-71-bf",
        "40-22-d8", "40-91-51", "48-27-e2", "48-31-b7", "48-55-19", "54-32-04", "54-43-b2",
        "70-04-1d", "7c-df-a1", "84-0d-8e", "84-cc-a8", "84-f7-03", "94-3c-c6", "a4-cf-12",
        "a4-e5-7c", "b4-e6-2d", "bc-dd-c2", "c4-4f-33", "c4-dd-57", "cc-50-e3", "dc-54-75",
        "e8-31-cd", "e8-db-84", "ec-94-cb", "f0-08-d1"
    )
    try:
        from concurrent.futures import ThreadPoolExecutor
        print("[Dò IP] Đang quét bảng ARP để tìm ESP32 trong mạng LAN...")
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
                res = session.get(f"http://{ip}/status", timeout=0.6)
                if res.status_code == 200 and "Blind to Bright" in res.text:
                    return ip
            except Exception:
                pass
            return None

        with ThreadPoolExecutor(max_workers=20) as pool:
            for cand in pool.map(_check, ordered):
                if cand:
                    print(f"[Dò IP] -> Tự động tìm thấy ESP32 tại IP LAN: {cand}")
                    return cand
    except Exception as e:
        print(f"[Dò IP Cảnh báo] Quét ARP gặp lỗi: {e}")

    fallback = target_ip or "192.168.4.1"
    print(f"[Dò IP] Không dò được tự động, sử dụng IP mặc định: {fallback}")
    return fallback


def fetch_esp_hardware_status(ip: str) -> dict:
    """Lấy thông số chẩn đoán từ endpoint /status của ESP32."""
    session = requests.Session()
    session.trust_env = False
    info = {
        "reachable": False,
        "rssi": None,
        "free_heap": None,
        "free_psram": None,
        "oled": None,
        "raw": ""
    }
    try:
        r = session.get(f"http://{ip}/status", timeout=1.5)
        if r.status_code == 200:
            info["reachable"] = True
            info["raw"] = r.text
            for line in r.text.splitlines():
                line = line.strip()
                if "Wi-Fi RSSI:" in line:
                    parts = line.split(":")
                    if len(parts) > 1:
                        val = parts[1].replace("dBm", "").strip()
                        try:
                            info["rssi"] = int(val)
                        except ValueError:
                            pass
                elif "Free Heap:" in line:
                    parts = line.split(":")
                    if len(parts) > 1:
                        info["free_heap"] = parts[1].replace("bytes", "").strip()
                elif "Free PSRAM:" in line:
                    parts = line.split(":")
                    if len(parts) > 1:
                        info["free_psram"] = parts[1].replace("bytes", "").strip()
                elif "OLED:" in line:
                    parts = line.split(":")
                    if len(parts) > 1:
                        info["oled"] = parts[1].strip()
    except Exception:
        pass
    return info


class FrameStreamWorker:
    """Luồng nền độc lập thu nhận và giải mã khung hình từ ESP32 MJPEG stream hoặc Webcam."""

    def __init__(self, source):
        self.source = source
        self.is_http = isinstance(source, str) and source.startswith("http")
        self.lock = threading.Lock()
        self.stop_event = threading.Event()

        # Dữ liệu frame mới nhất
        self.current_frame = None
        self.frame_id = 0
        self.last_frame_time = 0.0
        self.frame_width = 0
        self.frame_height = 0

        # Thống kê hiệu năng
        self.total_frames = 0
        self.total_bytes = 0
        self.start_time = 0.0
        self.last_bytes_check_time = time.time()
        self.last_bytes_count = 0
        self.current_bitrate_kbps = 0.0

        # Lịch sử để đo FPS & Jitter
        self.fps_window = deque(maxlen=60)          # Timestamp 60 frame gần nhất
        self.interval_history = deque(maxlen=60)     # Khoảng thời gian (ms) giữa 2 frame
        self.jpeg_size_history = deque(maxlen=60)    # Kích thước JPEG (KB) 60 frame gần nhất
        self.lag_spikes = 0                          # Số lần frame interval > 150ms

        # Trạng thái kết nối
        self.is_connected = False
        self.reconnect_count = 0
        self.status_msg = "Đang khởi tạo..."

        # Khởi động luồng đọc
        self.thread = threading.Thread(target=self._run, name="FrameStreamWorker", daemon=True)
        self.thread.start()

    def _run(self):
        if self.is_http:
            self._run_http_stream()
        else:
            self._run_cv_capture()

    def _run_http_stream(self):
        session = requests.Session()
        session.trust_env = False

        while not self.stop_event.is_set():
            self.status_msg = f"Đang kết nối luồng {self.source}..."
            try:
                response = session.get(self.source, stream=True, timeout=4.0)
                if response.status_code != 200:
                    self.is_connected = False
                    self.status_msg = f"HTTP {response.status_code}, thử lại sau 1s..."
                    time.sleep(1.0)
                    continue

                self.is_connected = True
                self.status_msg = "Kết nối thành công!"
                if self.start_time == 0.0:
                    self.start_time = time.time()

                data = bytearray()
                raw_stream = response.raw
                prev_time = time.time()

                while not self.stop_event.is_set():
                    chunk = raw_stream.read(4096)
                    if not chunk:
                        break

                    chunk_len = len(chunk)
                    data.extend(chunk)
                    now = time.time()

                    with self.lock:
                        self.total_bytes += chunk_len

                    # Tính bitrate định kỳ 0.5s
                    elapsed_bitrate = now - self.last_bytes_check_time
                    if elapsed_bitrate >= 0.5:
                        delta_bytes = self.total_bytes - self.last_bytes_count
                        self.current_bitrate_kbps = (delta_bytes / 1024.0) / elapsed_bitrate
                        self.last_bytes_count = self.total_bytes
                        self.last_bytes_check_time = now

                    # Tìm khung hình JPEG hoàn chỉnh (từ b'\xff\xd8' tới b'\xff\xd9')
                    b = data.rfind(b"\xff\xd9")
                    if b != -1:
                        a = data.rfind(b"\xff\xd8", 0, b)
                        if a != -1:
                            jpg = data[a:b+2]
                            data = data[b+2:]
                            jpg_size_kb = len(jpg) / 1024.0

                            if len(jpg) > 100:
                                image = cv2.imdecode(np.frombuffer(jpg, dtype=np.uint8), cv2.IMREAD_COLOR)
                                if image is not None:
                                    cur_time = time.time()
                                    interval_ms = (cur_time - prev_time) * 1000.0 if prev_time > 0 else 0.0
                                    prev_time = cur_time

                                    with self.lock:
                                        self.current_frame = image
                                        self.frame_height, self.frame_width = image.shape[:2]
                                        self.frame_id += 1
                                        self.total_frames += 1
                                        self.last_frame_time = cur_time

                                        self.fps_window.append(cur_time)
                                        if interval_ms > 0:
                                            self.interval_history.append(interval_ms)
                                            if interval_ms > 150.0:
                                                self.lag_spikes += 1
                                        self.jpeg_size_history.append(jpg_size_kb)

                    # Giới hạn buffer chống tràn RAM
                    if len(data) > 65536:
                        data = data[-16384:]

                response.close()

            except Exception as e:
                self.is_connected = False
                self.reconnect_count += 1
                self.status_msg = f"Mất kết nối ({e}), đang thử lại..."
                time.sleep(1.0)

    def _run_cv_capture(self):
        cap = cv2.VideoCapture(self.source)
        if not cap.isOpened():
            self.status_msg = f"Không mở được webcam: {self.source}"
            self.is_connected = False
            return

        self.is_connected = True
        self.status_msg = "Webcam hoạt động"
        if self.start_time == 0.0:
            self.start_time = time.time()

        prev_time = time.time()
        while not self.stop_event.is_set():
            ret, frame = cap.read()
            if ret and frame is not None:
                cur_time = time.time()
                interval_ms = (cur_time - prev_time) * 1000.0 if prev_time > 0 else 0.0
                prev_time = cur_time

                frame_bytes = frame.nbytes

                with self.lock:
                    self.current_frame = frame
                    self.frame_height, self.frame_width = frame.shape[:2]
                    self.frame_id += 1
                    self.total_frames += 1
                    self.total_bytes += frame_bytes
                    self.last_frame_time = cur_time

                    self.fps_window.append(cur_time)
                    if interval_ms > 0:
                        self.interval_history.append(interval_ms)
                        if interval_ms > 150.0:
                            self.lag_spikes += 1
            else:
                self.is_connected = False
                time.sleep(0.01)

        cap.release()

    def get_latest(self):
        """Trả về frame mới nhất cùng các số liệu thống kê."""
        with self.lock:
            frame_copy = self.current_frame.copy() if self.current_frame is not None else None
            now = time.time()

            # Tính FPS trượt 1s
            while self.fps_window and now - self.fps_window[0] > 1.0:
                self.fps_window.popleft()
            rolling_fps = float(len(self.fps_window))

            # Tính FPS trung bình toàn thời gian
            uptime = max(0.001, now - self.start_time) if self.start_time > 0 else 0.001
            avg_fps = self.total_frames / uptime

            # Khoảng thời gian frame gần nhất và trung bình
            cur_interval = self.interval_history[-1] if self.interval_history else 0.0
            avg_interval = float(np.mean(self.interval_history)) if self.interval_history else 0.0
            min_interval = float(np.min(self.interval_history)) if self.interval_history else 0.0
            max_interval = float(np.max(self.interval_history)) if self.interval_history else 0.0

            # Kích thước frame trung bình (KB)
            avg_kb = float(np.mean(self.jpeg_size_history)) if self.jpeg_size_history else 0.0

            stats = {
                "connected": self.is_connected,
                "status_msg": self.status_msg,
                "frame_id": self.frame_id,
                "total_frames": self.total_frames,
                "width": self.frame_width,
                "height": self.frame_height,
                "rolling_fps": rolling_fps,
                "avg_fps": avg_fps,
                "cur_interval_ms": cur_interval,
                "avg_interval_ms": avg_interval,
                "min_interval_ms": min_interval,
                "max_interval_ms": max_interval,
                "bitrate_kbps": self.current_bitrate_kbps,
                "avg_jpeg_kb": avg_kb,
                "lag_spikes": self.lag_spikes,
                "total_mb": self.total_bytes / (1024.0 * 1024.0),
                "uptime": uptime,
                "history_intervals": list(self.interval_history),
            }
            return frame_copy, stats

    def reset_stats(self):
        """Đặt lại toàn bộ số liệu thống kê."""
        with self.lock:
            self.total_frames = 0
            self.total_bytes = 0
            self.start_time = time.time()
            self.fps_window.clear()
            self.interval_history.clear()
            self.jpeg_size_history.clear()
            self.lag_spikes = 0
            self.last_bytes_count = 0
            self.last_bytes_check_time = time.time()

    def stop(self):
        self.stop_event.set()


def fit_frame_to_canvas(raw_frame: np.ndarray, target_w: int = CANVAS_WIDTH, target_h: int = CANVAS_HEIGHT) -> np.ndarray:
    """
    Đặt khung hình camera vào canvas chuẩn 960x720, giữ nguyên tỉ lệ khung hình (Aspect Ratio),
    tránh hoàn toàn hiện tượng méo ảnh hoặc lệch HUD do kích thước camera quá nhỏ/lớn.
    """
    fh, fw = raw_frame.shape[:2]
    canvas = np.zeros((target_h, target_w, 3), dtype=np.uint8)

    # Tính tỉ lệ phóng to bảo toàn Aspect Ratio
    scale = min(target_w / fw, target_h / fh)
    nw = int(fw * scale)
    nh = int(fh * scale)

    resized = cv2.resize(raw_frame, (nw, nh), interpolation=cv2.INTER_LINEAR)

    # Căn giữa khung hình vào canvas
    x_offset = (target_w - nw) // 2
    y_offset = (target_h - nh) // 2
    canvas[y_offset:y_offset + nh, x_offset:x_offset + nw] = resized
    return canvas


def draw_sparkline_graph(canvas: np.ndarray, x: int, y: int, w: int, h: int, values: list[float], max_val: float = 120.0):
    """Vẽ một biểu đồ sparkline nhỏ biểu thị độ trễ frame trong thời gian thực."""
    if not values or len(values) < 2:
        return

    # Khung nền đồ thị bán trong suốt
    overlay = canvas.copy()
    cv2.rectangle(overlay, (x, y), (x + w, y + h), (18, 22, 30), -1)
    cv2.addWeighted(overlay, 0.7, canvas, 0.3, 0, canvas)
    cv2.rectangle(canvas, (x, y), (x + w, y + h), (55, 68, 85), 1)

    # Đường tham chiếu 33.3ms (~30 FPS) và 66.6ms (~15 FPS)
    y_30fps = int(y + h - (33.3 / max_val) * h)
    y_15fps = int(y + h - (66.6 / max_val) * h)
    if y <= y_30fps <= y + h:
        cv2.line(canvas, (x, y_30fps), (x + w, y_30fps), (40, 130, 60), 1)
    if y <= y_15fps <= y + h:
        cv2.line(canvas, (x, y_15fps), (x + w, y_15fps), (60, 60, 150), 1)

    # Vẽ các điểm giá trị
    points = []
    num_pts = len(values)
    dx = w / max(1, num_pts - 1)

    for i, val in enumerate(values):
        clipped = min(max_val, max(0.0, val))
        px = int(x + i * dx)
        py = int(y + h - (clipped / max_val) * h)
        points.append((px, py))

    for i in range(len(points) - 1):
        pt1 = points[i]
        pt2 = points[i + 1]
        val = values[i + 1]
        color = (50, 215, 80) if val < 45 else ((60, 190, 245) if val < 90 else (50, 60, 245))
        cv2.line(canvas, pt1, pt2, color, 2)


def render_hud_overlay(
    canvas: np.ndarray,
    stats: dict,
    esp_info: dict,
    is_paused: bool,
    show_hud: bool,
    toast_msg: str,
    stream_url: str
) -> np.ndarray:
    """
    Vẽ toàn bộ giao diện HUD thông số kỹ thuật lên trên Canvas 960x720 chuẩn.
    Mọi tọa độ đều được khóa cố định theo kích thước 960x720 nên không bao giờ bị méo chữ hay tràn viền.
    """
    w = CANVAS_WIDTH
    h = CANVAS_HEIGHT

    # 1. Vẽ Notification Toast nổi ở giữa màn hình (nếu có)
    if toast_msg:
        toast_w, toast_h = 500, 44
        tx = (w - toast_w) // 2
        ty = 25
        overlay = canvas.copy()
        cv2.rectangle(overlay, (tx, ty), (tx + toast_w, ty + toast_h), (20, 20, 20), -1)
        cv2.addWeighted(overlay, 0.85, canvas, 0.15, 0, canvas)
        cv2.rectangle(canvas, (tx, ty), (tx + toast_w, ty + toast_h), (0, 225, 130), 2)

        pil_img = Image.fromarray(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(pil_img)
        draw.text((tx + 22, ty + 11), toast_msg, font=FONT_BOLD, fill=(0, 255, 160))
        canvas = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

    if not show_hud:
        # Nếu ẩn HUD, chỉ hiển thị phím tắt gợi ý nhỏ ở góc dưới
        cv2.putText(canvas, "[H] Hien HUD thong so", (20, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (220, 220, 220), 1)
        return canvas

    # 2. Bảng thông số chính (Top-Left Panel) - Cố định kích thước chuẩn 360 x 310
    panel_w = 360
    panel_h = 310
    px, py = 20, 20

    overlay = canvas.copy()
    cv2.rectangle(overlay, (px, py), (px + panel_w, py + panel_h), (16, 20, 28), -1)
    cv2.addWeighted(overlay, 0.82, canvas, 0.18, 0, canvas)
    cv2.rectangle(canvas, (px, py), (px + panel_w, py + panel_h), (55, 70, 90), 1)

    # 3. Vẽ biểu đồ Sparkline độ trễ (ở đáy panel chính)
    chart_x = px + 15
    chart_y = py + 245
    chart_w = panel_w - 30
    chart_h = 52
    draw_sparkline_graph(canvas, chart_x, chart_y, chart_w, chart_h, stats.get("history_intervals", []), max_val=120.0)

    # 4. Vẽ văn bản thông số bằng Pillow
    pil_img = Image.fromarray(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_img)

    # Header & Trạng thái kết nối
    if is_paused:
        status_color = (245, 158, 11)   # Cam/Vàng
        status_dot = "⏸"
        status_text = "PAUSED (TẠM DỪNG)"
    elif stats["connected"]:
        status_color = (34, 197, 94)    # Xanh lá
        status_dot = "●"
        status_text = "LIVE STREAM (ESP32)" if "http" in stream_url else "LIVE (WEBCAM LAPTOP)"
    else:
        status_color = (239, 68, 68)    # Đỏ
        status_dot = "✖"
        status_text = "MẤT KẾT NỐI CAMERA"

    draw.text((px + 14, py + 12), f"{status_dot} {status_text}", font=FONT_TITLE, fill=status_color)

    # FPS lớn nổi bật
    fps = stats["rolling_fps"]
    if fps >= 25.0:
        fps_color = (34, 197, 94)       # Xanh lá (>25 fps)
        fps_grade = "RẤT MƯỢT"
    elif fps >= 15.0:
        fps_color = (234, 179, 8)       # Vàng (15-25 fps)
        fps_grade = "TẠM ỔN"
    else:
        fps_color = (239, 68, 68)       # Đỏ (<15 fps)
        fps_grade = "CHẬM / LAG"

    draw.text((px + 14, py + 38), f"{fps:4.1f}", font=FONT_LARGE, fill=fps_color)
    draw.text((px + 92, py + 46), f"FPS  ({fps_grade})", font=FONT_BOLD, fill=fps_color)

    # Đường phân cách ngang
    draw.line([(px + 14, py + 78), (px + panel_w - 14, py + 78)], fill=(60, 75, 95), width=1)

    # Danh sách thông số
    cur_y = py + 88
    line_h = 23

    # Độ phân giải phần cứng gốc
    draw.text((px + 14, cur_y), "Độ phân giải gốc:", font=FONT_TEXT, fill=(180, 195, 210))
    draw.text((px + 155, cur_y), f"{stats['width']} x {stats['height']} px", font=FONT_BOLD, fill=(255, 255, 255))
    cur_y += line_h

    # Băng thông / Bitrate
    bitrate = stats["bitrate_kbps"]
    bitrate_str = f"{bitrate:.1f} KB/s ({bitrate * 8 / 1024:.2f} Mbps)" if bitrate > 0 else "0 KB/s"
    draw.text((px + 14, cur_y), "Băng thông mạng:", font=FONT_TEXT, fill=(180, 195, 210))
    draw.text((px + 155, cur_y), bitrate_str, font=FONT_BOLD, fill=(100, 220, 255))
    cur_y += line_h

    # Kích thước frame trung bình
    draw.text((px + 14, cur_y), "Cỡ frame JPEG:", font=FONT_TEXT, fill=(180, 195, 210))
    draw.text((px + 155, cur_y), f"{stats['avg_jpeg_kb']:.1f} KB / frame", font=FONT_BOLD, fill=(255, 255, 255))
    cur_y += line_h

    # Độ trễ frame (Frame Time)
    draw.text((px + 14, cur_y), "Độ trễ frame:", font=FONT_TEXT, fill=(180, 195, 210))
    draw.text((px + 155, cur_y), f"{stats['cur_interval_ms']:.1f} ms (TB: {stats['avg_interval_ms']:.1f}ms)", font=FONT_BOLD, fill=(255, 215, 0))
    cur_y += line_h

    # Số lần khựng hình / Spikes
    spike_color = (239, 68, 68) if stats["lag_spikes"] > 5 else (200, 200, 200)
    draw.text((px + 14, cur_y), "Khựng (>150ms):", font=FONT_TEXT, fill=(180, 195, 210))
    draw.text((px + 155, cur_y), f"{stats['lag_spikes']} lần", font=FONT_BOLD, fill=spike_color)
    cur_y += line_h

    # Tiêu đề biểu đồ độ trễ
    draw.text((chart_x, chart_y - 17), "Biểu đồ độ trễ (30 FPS: 33ms | 15 FPS: 66ms):", font=FONT_SMALL, fill=(140, 160, 180))

    # 5. Bảng thông số phần cứng ESP32 (Top-Right Panel) - Kích thước 280 x 120
    if esp_info and esp_info.get("reachable"):
        right_w = 280
        right_h = 120
        rx = w - right_w - 20
        ry = 20

        draw.rectangle([(rx, ry), (rx + right_w, ry + right_h)], fill=(16, 20, 28), outline=(55, 70, 90))
        draw.text((rx + 14, ry + 10), "ESP32-S3 THÔNG SỐ", font=FONT_BOLD, fill=(0, 200, 255))

        rssi = esp_info.get("rssi")
        rssi_str = f"{rssi} dBm" if rssi is not None else "N/A"
        rssi_grade = " (Mạnh)" if rssi and rssi > -65 else (" (Ổn định)" if rssi and rssi > -80 else " (Yếu)")
        rssi_color = (34, 197, 94) if rssi and rssi > -65 else ((234, 179, 8) if rssi and rssi > -80 else (239, 68, 68))
        draw.text((rx + 14, ry + 36), f"Wi-Fi RSSI:  {rssi_str}{rssi_grade}", font=FONT_TEXT, fill=rssi_color)

        heap = esp_info.get("free_heap")
        draw.text((rx + 14, ry + 62), f"Free Heap:   {heap} bytes" if heap else "Free Heap: N/A", font=FONT_SMALL, fill=(185, 195, 205))

        psram = esp_info.get("free_psram")
        draw.text((rx + 14, ry + 86), f"Free PSRAM:  {psram} bytes" if psram else "Free PSRAM: N/A", font=FONT_SMALL, fill=(185, 195, 205))

    # 6. Thanh hướng dẫn phím tắt ở cạnh dưới cùng Canvas (960 x 36)
    bar_h = 36
    by = h - bar_h
    draw.rectangle([(0, by), (w, h)], fill=(10, 12, 16))
    hint_text = "[S] Chụp frame  |  [P] Tạm dừng  |  [H] Ẩn/Hiện HUD  |  [R] Reset thống kê  |  [Q] Thoát"
    draw.text((18, by + 8), hint_text, font=FONT_HINT, fill=(225, 230, 235))

    total_txt = f"Tổng: {stats['total_frames']} frames | {stats['total_mb']:.2f} MB | {stats['uptime']:.0f}s"
    draw.text((w - 290, by + 8), total_txt, font=FONT_HINT, fill=(140, 160, 185))

    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)


def print_summary_report(stats: dict, esp_info: dict, stream_source: str):
    """In bảng tổng kết toàn diện hiệu năng luồng video ra Terminal."""
    print("\n" + "=" * 68)
    print("        BÁO CÁO HIỆU NĂNG LUỒNG CAMERA ESP32 (BLIND TO BRIGHT)       ")
    print("=" * 68)
    print(f"  - Nguồn Stream       : {stream_source}")
    print(f"  - Thời gian kiểm tra : {stats['uptime']:.1f} giây")
    print(f"  - Độ phân giải gốc   : {stats['width']} x {stats['height']} pixels")
    print(f"  - Canvas hiển thị    : {CANVAS_WIDTH} x {CANVAS_HEIGHT} pixels")
    print(f"  - Tổng khung hình    : {stats['total_frames']} frames")
    print(f"  - FPS trung bình     : {stats['avg_fps']:.2f} FPS")
    print(f"  - Độ trễ frame TB    : {stats['avg_interval_ms']:.2f} ms")
    print(f"  - Độ trễ min / max   : {stats['min_interval_ms']:.1f} ms / {stats['max_interval_ms']:.1f} ms")
    print(f"  - Khựng hình (>150ms): {stats['lag_spikes']} lần")
    print(f"  - Kích thước frame TB: {stats['avg_jpeg_kb']:.1f} KB")
    print(f"  - Tổng dung lượng tải: {stats['total_mb']:.2f} MB")

    if esp_info and esp_info.get("reachable"):
        print("  - Thông số phần cứng ESP32:")
        print(f"      + Wi-Fi RSSI    : {esp_info.get('rssi')} dBm")
        print(f"      + Free Heap     : {esp_info.get('free_heap')} bytes")
        print(f"      + Free PSRAM    : {esp_info.get('free_psram')} bytes")

    print("-" * 68)
    print("  ĐÁNH GIÁ CHẤT LƯỢNG CHO MÔ HÌNH NHẬN DIỆN CỬ CHỈ (AI MODEL):")
    if stats['avg_fps'] >= 22.0 and stats['lag_spikes'] <= 2:
        print("  -> [XUẤT SẮC] Luồng camera rất mượt mà, độ trễ thấp.")
        print("     Rất lý tưởng cho mô hình ST-GCN Transformer nhận diện 48 frame.")
    elif stats['avg_fps'] >= 15.0:
        print("  -> [ĐẠT YÊU CẦU] Luồng camera ổn định ở mức khá.")
        print("     Mô hình nhận diện hoạt động tốt, hạn chế di chuyển tay quá nhanh.")
    else:
        print("  -> [CẢNH BÁO] FPS thấp (<15) hoặc mạng Wi-Fi bị gián đoạn/khựng hình.")
        print("     Khuyến nghị:")
        print("     1. Đặt ESP32 gần router Wi-Fi hơn hoặc kết nối vào SoftAP 'Blind to Bright'.")
        print("     2. Kiểm tra nguồn cấp ESP32 (cần cáp USB cấp dòng ổn định >= 1A).")
        print("     3. Dùng Webcam Laptop (--camera 0) để có FPS ổn định 30 FPS.")
    print("=" * 68 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Kiểm tra luồng frame camera ESP32 - Blind to Bright")
    parser.add_argument("--ip", default=None, help="IP của ESP32 (Ví dụ: 192.168.1.50 hoặc 192.168.4.1). Mặc định tự động dò.")
    parser.add_argument("--port", type=int, default=81, help="Cổng stream MJPEG của ESP32 (Mặc định: 81)")
    parser.add_argument("--url", default=None, help="URL luồng stream đầy đủ (Ghi đè ip/port, VD: http://192.168.4.1:81/stream)")
    parser.add_argument("--camera", default=None, help="Chỉ định nguồn: 'esp', '0' (webcam laptop), hoặc URL http")
    parser.add_argument("--fallback", action="store_true", default=True, help="Tự động chuyển về Webcam Laptop 0 nếu không tìm thấy ESP")
    parser.add_argument("--save-dir", default="saved_frames", help="Thư mục lưu ảnh chụp màn hình (Mặc định: saved_frames)")
    parser.add_argument("--no-hud", action="store_true", help="Bắt đầu với chế độ ẩn HUD thông số")
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    actual_ip = None
    stream_url = None

    if args.camera == "0" or args.camera == 0:
        stream_url = 0
        print("[Nguồn Camera] Sử dụng trực tiếp Webcam Laptop (Device Index 0).")
    elif args.url:
        stream_url = args.url
        print(f"[Nguồn Camera] Sử dụng URL chỉ định: {stream_url}")
    else:
        print("\n=======================================================")
        print("       BẮT ĐẦU KIỂM TRA LUỒNG FRAME CAMERA ESP32      ")
        print("=======================================================")
        actual_ip = resolve_esp_ip(args.ip)

        test_session = requests.Session()
        test_session.trust_env = False
        target_stream = f"http://{actual_ip}:{args.port}/stream"

        print(f"[Kiểm tra Luồng] Thử kết nối tới: {target_stream} (timeout 2.0s)...")
        stream_reachable = False
        try:
            r = test_session.get(target_stream, stream=True, timeout=2.0)
            if r.status_code == 200:
                stream_reachable = True
            r.close()
        except Exception as e:
            print(f"[Kiểm tra Luồng] Không thể kết nối tới {target_stream}: {e}")

        if stream_reachable:
            stream_url = target_stream
            print(f"[Kiểm tra Luồng] -> KẾT NỐI THÀNH CÔNG TỚI ESP32: {stream_url}\n")
        else:
            if args.fallback:
                print("\n" + "!" * 64)
                print("[FALLBACK] ESP32 không phản hồi luồng video!")
                print("-> TỰ ĐỘNG CHUYỂN SANG WEBCAM LAPTOP (Device 0) để test...")
                print("!" * 64 + "\n")
                stream_url = 0
            else:
                print(f"[Lỗi] Không kết nối được ESP32 tại {target_stream}. Hãy bật --fallback hoặc kiểm tra lại Wi-Fi.")
                return

    esp_info = {}
    if actual_ip:
        esp_info = fetch_esp_hardware_status(actual_ip)

    worker = FrameStreamWorker(stream_url)

    print("[Khởi động] Đang đợi khung hình đầu tiên...")
    wait_start = time.time()
    first_frame_ok = False
    while time.time() - wait_start < 5.0:
        frame, stats = worker.get_latest()
        if frame is not None:
            first_frame_ok = True
            break
        time.sleep(0.05)

    if not first_frame_ok:
        print("[Cảnh báo] Chưa nhận được frame sau 5s! Kiểm tra lại kết nối mạng hoặc nguồn camera.")

    # Thiết lập cửa sổ với kích thước canvas chuẩn 960x720
    window_name = "Blind to Bright - ESP32 Camera Frame Tester"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, CANVAS_WIDTH, CANVAS_HEIGHT)

    is_paused = False
    show_hud = not args.no_hud
    toast_msg = ""
    toast_expire_time = 0.0
    last_raw_frame = None

    print("\n[Điều khiển]:")
    print("  - [S] : Chụp và lưu ảnh vào thư mục 'saved_frames'")
    print("  - [P] : Tạm dừng / Tiếp tục (Pause/Resume)")
    print("  - [H] : Ẩn / Hiện bảng thông số HUD")
    print("  - [R] : Reset lại toàn bộ thống kê")
    print("  - [Q] / [ESC] : Thoát chương trình và in bảng tổng kết\n")

    try:
        while True:
            if not is_paused:
                frame, stats = worker.get_latest()
                if frame is not None:
                    last_raw_frame = frame
            else:
                _, stats = worker.get_latest()

            if toast_msg and time.time() > toast_expire_time:
                toast_msg = ""

            if last_raw_frame is not None:
                # 1. Căn chỉnh khung hình vào Canvas chuẩn 960x720 giữ nguyên tỉ lệ
                canvas = fit_frame_to_canvas(last_raw_frame, CANVAS_WIDTH, CANVAS_HEIGHT)

                # 2. Vẽ bảng thông số HUD lên Canvas
                display = render_hud_overlay(
                    canvas=canvas,
                    stats=stats,
                    esp_info=esp_info,
                    is_paused=is_paused,
                    show_hud=show_hud,
                    toast_msg=toast_msg,
                    stream_url=str(stream_url)
                )
                cv2.imshow(window_name, display)
            else:
                blank = np.zeros((CANVAS_HEIGHT, CANVAS_WIDTH, 3), dtype=np.uint8)
                cv2.putText(blank, "DANG KET NOI TOI CAMERA...", (CANVAS_WIDTH // 2 - 200, CANVAS_HEIGHT // 2),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 215, 255), 2)
                cv2.imshow(window_name, blank)

            raw_key = cv2.waitKey(15)
            if raw_key != -1:
                key = raw_key & 0xFF

                # [Q] hoặc [ESC]: Thoát
                if key in (ord("q"), ord("Q"), 27):
                    break

                # [P]: Tạm dừng
                elif key in (ord("p"), ord("P")):
                    is_paused = not is_paused
                    toast_msg = "ĐÃ TẠM DỪNG STREAM" if is_paused else "ĐÃ TIẾP TỤC STREAM"
                    toast_expire_time = time.time() + 2.0

                # [H]: Ẩn / Hiện HUD
                elif key in (ord("h"), ord("H")):
                    show_hud = not show_hud
                    toast_msg = "ĐÃ BẬT BẢNG HUD" if show_hud else "ĐÃ ẨN BẢNG HUD"
                    toast_expire_time = time.time() + 1.5

                # [R]: Reset thống kê
                elif key in (ord("r"), ord("R")):
                    worker.reset_stats()
                    toast_msg = "ĐÃ RESET SỐ LIỆU THỐNG KÊ"
                    toast_expire_time = time.time() + 2.0
                    print("[Thống kê] Đã reset toàn bộ thông số về 0.")

                # [S]: Chụp ảnh lưu khung hình
                elif key in (ord("s"), ord("S")):
                    if last_raw_frame is not None:
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        # 1. Lưu ảnh RAW nguyên bản đúng độ phân giải gốc của camera
                        raw_name = f"esp_raw_{timestamp}_{last_raw_frame.shape[1]}x{last_raw_frame.shape[0]}.jpg"
                        raw_path = os.path.join(args.save_dir, raw_name)
                        cv2.imwrite(raw_path, last_raw_frame)

                        # 2. Lưu ảnh toàn màn hình có kèm HUD
                        hud_name = f"esp_hud_{timestamp}.jpg"
                        hud_path = os.path.join(args.save_dir, hud_name)
                        cv2.imwrite(hud_path, display)

                        toast_msg = f"ĐÃ LƯU: {raw_name}"
                        toast_expire_time = time.time() + 2.5
                        print(f"[Snapshot] Đã lưu ảnh RAW: {raw_path}")
                        print(f"[Snapshot] Đã lưu ảnh HUD: {hud_path}")

    except KeyboardInterrupt:
        pass
    finally:
        worker.stop()
        cv2.destroyAllWindows()

        _, final_stats = worker.get_latest()
        print_summary_report(final_stats, esp_info, str(stream_url))


if __name__ == "__main__":
    main()
