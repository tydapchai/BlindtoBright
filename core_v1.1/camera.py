import threading
import time

import cv2
import numpy as np
import requests


class LatestFrameCamera:
    def __init__(self, source):
        self.source = source
        self.frame = None
        self.frame_id = 0
        self.lock = threading.Lock()
        self.stop_event = threading.Event()
        self.connected_event = threading.Event()
        self.session = requests.Session()
        self.session.trust_env = False
        self.thread = threading.Thread(target=self._capture, name="camera-capture", daemon=True)
        self.thread.start()

    def _capture(self):
        if isinstance(self.source, str) and self.source.startswith("http"):
            print(f"[Camera] Đang kết nối tới luồng: {self.source} ...")
            while not self.stop_event.is_set():
                try:
                    response = self.session.get(self.source, stream=True, timeout=5)
                    if response.status_code != 200:
                        self.connected_event.clear()
                        print(f"[Camera Cảnh báo] HTTP {response.status_code}, thử lại sau 1s...")
                        time.sleep(1.0)
                        continue
                    
                    print(f"[Camera] Kết nối thành công tới {self.source}!")
                    self.connected_event.set()
                    data = bytearray()
                    raw_stream = response.raw
                    while not self.stop_event.is_set():
                        chunk = raw_stream.read(4096)
                        if not chunk:
                            break
                        data.extend(chunk)

                        # Lấy khung hình mới nhất (rfind) để tránh trễ tích lũy
                        b = data.rfind(b"\xff\xd9")
                        if b != -1:
                            a = data.rfind(b"\xff\xd8", 0, b)
                            if a != -1:
                                jpg = data[a:b+2]
                                data = data[b+2:]
                                if len(jpg) > 100:
                                    image = cv2.imdecode(
                                        np.frombuffer(jpg, dtype=np.uint8),
                                        cv2.IMREAD_COLOR,
                                    )
                                    if image is not None:
                                        with self.lock:
                                            self.frame = image
                                            self.frame_id += 1

                        if len(data) > 65536:
                            data = data[-16384:]
                    response.close()
                except requests.RequestException as error:
                    self.connected_event.clear()
                    print(f"[Camera Cảnh báo] Mất kết nối ({error}), đang thử kết nối lại...")
                    time.sleep(1.0)
                except Exception as error:
                    self.connected_event.clear()
                    print(f"[Camera Lỗi] {error}, đang thử lại...")
                    time.sleep(1.0)
            return

        print(f"[Camera] Đang mở nguồn webcam/video: {self.source} ...")
        capture = cv2.VideoCapture(self.source)
        if capture.isOpened():
            self.connected_event.set()
            print(f"[Camera] Đã mở Webcam thành công!")
        else:
            print(f"[Camera Lỗi] Không mở được nguồn webcam: {self.source}")

        while not self.stop_event.is_set():
            ok, image = capture.read()
            if ok and image is not None:
                self.connected_event.set()
                with self.lock:
                    self.frame = image
                    self.frame_id += 1
            else:
                self.connected_event.clear()
                time.sleep(0.01)
        capture.release()

    def wait_until_connected(self, timeout=5.0):
        return self.connected_event.wait(timeout)

    def is_connected(self):
        return self.connected_event.is_set()

    def read_latest(self, last_id):
        with self.lock:
            if self.frame is None:
                return False, last_id, None
            return self.frame_id != last_id, self.frame_id, self.frame.copy()

    def release(self):
        self.stop_event.set()