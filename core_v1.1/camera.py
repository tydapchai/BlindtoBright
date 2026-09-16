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
        self.session = requests.Session()
        self.session.trust_env = False
        self.thread = threading.Thread(target=self._capture, daemon=True)
        self.thread.start()

    def _capture(self):
        if isinstance(self.source, str) and self.source.startswith("http"):
            while not self.stop_event.is_set():
                try:
                    response = self.session.get(self.source, stream=True, timeout=5)
                    data = b""
                    for chunk in response.iter_content(chunk_size=4096):
                        if self.stop_event.is_set():
                            break
                        data += chunk
                        start = data.find(b"\xff\xd8")
                        end = data.find(b"\xff\xd9", start + 2)
                        if start >= 0 and end >= 0:
                            image = cv2.imdecode(
                                np.frombuffer(data[start:end + 2], dtype=np.uint8),
                                cv2.IMREAD_COLOR,
                            )
                            data = data[end + 2:]
                            if image is not None:
                                with self.lock:
                                    self.frame = image
                                    self.frame_id += 1
                    response.close()
                except requests.RequestException:
                    time.sleep(0.5)
            return

        capture = cv2.VideoCapture(self.source)
        while not self.stop_event.is_set():
            ok, image = capture.read()
            if ok and image is not None:
                with self.lock:
                    self.frame = image
                    self.frame_id += 1
            else:
                time.sleep(0.01)
        capture.release()

    def read_latest(self, last_id):
        with self.lock:
            if self.frame is None:
                return False, last_id, None
            return self.frame_id != last_id, self.frame_id, self.frame.copy()

    def release(self):
        self.stop_event.set()