from __future__ import annotations

import threading
import time

import cv2
from PyQt6.QtCore import QThread, pyqtSignal


class VideoFileThread(QThread):
    frame_ready = pyqtSignal(object)
    info_ready = pyqtSignal(object)
    finished_playback = pyqtSignal()
    error = pyqtSignal(str)

    def __init__(self, path: str, parent=None):
        super().__init__(parent)
        self.path = path
        self._stop_event = threading.Event()
        self._paused = False
        self._pause_lock = threading.Lock()

    def stop(self):
        self._stop_event.set()

    def set_paused(self, paused: bool):
        with self._pause_lock:
            self._paused = bool(paused)

    def run(self):
        cap = cv2.VideoCapture(self.path)
        if not cap.isOpened():
            self.error.emit(f"Could not open video: {self.path}")
            return
        try:
            fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
            frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            self.info_ready.emit({"fps": fps, "width": width, "height": height, "frames": frames})
            interval = 1.0 / fps if fps > 1e-3 else 1.0 / 30.0
            deadline = time.perf_counter()
            while not self._stop_event.is_set():
                with self._pause_lock:
                    paused = self._paused
                if paused:
                    time.sleep(0.02)
                    deadline = time.perf_counter()
                    continue
                ok, frame = cap.read()
                if not ok or frame is None:
                    self.finished_playback.emit()
                    break
                self.frame_ready.emit(frame)
                deadline += interval
                delay = deadline - time.perf_counter()
                if delay > 0:
                    time.sleep(min(delay, interval))
                elif delay < -0.5:
                    deadline = time.perf_counter()
        except Exception as exc:
            self.error.emit(f"Video error: {exc}")
        finally:
            cap.release()
