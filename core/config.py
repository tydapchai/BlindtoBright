import threading
import time

class State:
    def __init__(self):
        self.history = []
        self.lock = threading.Lock()

    def add(self, source, text, extra=None):
        item = {
            "source": source,
            "text": text,
            "time": time.strftime("%H:%M:%S"),
        }
        if extra:
            item.update(extra)
        with self.lock:
            self.history.append(item)
            self.history = self.history[-100:]

    def get(self):
        with self.lock:
            return list(self.history)

state = State()
shutdown_event = threading.Event()