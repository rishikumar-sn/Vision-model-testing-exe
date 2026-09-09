from __future__ import annotations

import threading
import time
from typing import Optional

import cv2
from PyQt6.QtCore import QObject, QThread, pyqtSignal

from camera.camera_utils import (
    CAMERA_MODE_CACHE, FAST_RESOLUTION_PROBES, open_capture
)
from camera.anti_flicker import WindowsPowerLineController, AntiFlickerPipeline

class CameraThread(QThread):
    frame_ready = pyqtSignal(object)        # np.ndarray, BGR, full camera resolution
    camera_ready = pyqtSignal(object)       # dict with width/height/modes/backend
    camera_error = pyqtSignal(str)
    probe_status = pyqtSignal(str)
    measured_fps = pyqtSignal(float)
    anti_flicker_status = pyqtSignal(object)

    def __init__(
        self,
        camera_index: int,
        anti_flicker_mode: str = "adaptive",
        parent: Optional[QObject] = None,
    ):
        super().__init__(parent)
        self.camera_index = camera_index
        self._stop_event = threading.Event()
        self._cap: Optional[cv2.VideoCapture] = None
        self._command_lock = threading.Lock()
        self._requested_mode = anti_flicker_mode
        self._applied_mode: Optional[str] = None
        self._recalibration_requested = False
        self._driver_settings_requested = False
        self._powerline_controller: Optional[WindowsPowerLineController] = None

    def stop(self) -> None:
        self._stop_event.set()

    def set_anti_flicker_mode(self, mode: str) -> None:
        with self._command_lock:
            self._requested_mode = mode

    def request_recalibration(self) -> None:
        with self._command_lock:
            self._recalibration_requested = True

    def request_driver_settings(self) -> None:
        with self._command_lock:
            self._driver_settings_requested = True

    def _read_stable_frame(self, cap: cv2.VideoCapture, attempts: int = 4) -> Optional[np.ndarray]:
        frame = None
        for _ in range(attempts):
            if self._stop_event.is_set():
                return None
            ok, candidate = cap.read()
            if ok and candidate is not None and candidate.size > 0:
                frame = candidate
        return frame

    def _probe_max_resolution(self, cap: cv2.VideoCapture) -> tuple[tuple[int, int], list[tuple[int, int]]]:
        self.probe_status.emit("Probing camera resolutions...")

        observed_modes: set[tuple[int, int]] = set()
        previous_actual: Optional[tuple[int, int]] = None
        best_mode: Optional[tuple[int, int]] = None
        best_area = -1

        # Ask for MJPG before probing. Many USB webcams expose 2K/4K only in MJPG.
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        cap.set(cv2.CAP_PROP_FPS, 30)

        cached_mode = CAMERA_MODE_CACHE.get(self.camera_index)
        if cached_mode is not None:
            self.probe_status.emit(
                f"Restoring {cached_mode[0]} x {cached_mode[1]}..."
            )
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, cached_mode[0])
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cached_mode[1])
            cached_frame = self._read_stable_frame(cap, attempts=1)
            if cached_frame is not None:
                cached_h, cached_w = cached_frame.shape[:2]
                actual_cached = (int(cached_w), int(cached_h))
                if actual_cached == cached_mode:
                    cap.set(cv2.CAP_PROP_FPS, 30)
                    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
                    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                    return cached_mode, [cached_mode]
            CAMERA_MODE_CACHE.pop(self.camera_index, None)

        for requested_w, requested_h in FAST_RESOLUTION_PROBES:
            if self._stop_event.is_set():
                break

            self.probe_status.emit(f"Testing {requested_w} × {requested_h}...")
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, requested_w)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, requested_h)
            cap.set(cv2.CAP_PROP_FPS, 30)

            frame = self._read_stable_frame(cap, attempts=1)
            if frame is None:
                continue

            actual_h, actual_w = frame.shape[:2]
            actual = (int(actual_w), int(actual_h))
            observed_modes.add(actual)

            area = actual_w * actual_h
            if area > best_area:
                best_area = area
                best_mode = actual

            # Probes are ordered from largest to smallest. An exact match means
            # the remaining requests cannot improve on this working mode.
            if actual == (requested_w, requested_h):
                break

            # A request above Full HD that resolves to an HD-or-better mode is
            # already the driver's best match. This covers Brio/C950/Dell-class
            # webcams without renegotiating several other oversized modes.
            if actual_w * actual_h >= 1920 * 1080:
                break

            # DirectShow commonly maps every oversized request to the camera's
            # maximum mode. Seeing the same fallback twice is enough; continuing
            # to renegotiate more oversized modes only adds several seconds.
            if actual == previous_actual:
                break
            previous_actual = actual

        if best_mode is None:
            # Fall back to whatever the camera currently provides.
            frame = self._read_stable_frame(cap, attempts=4)
            if frame is None:
                raise RuntimeError("Camera opened but no video frames could be read.")
            h, w = frame.shape[:2]
            best_mode = (int(w), int(h))
            observed_modes.add(best_mode)

        # Restore the selected maximum working mode.
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, best_mode[0])
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, best_mode[1])
        cap.set(cv2.CAP_PROP_FPS, 30)
        # BRIO negotiates a much faster Full-HD stream when MJPG is selected
        # after the final size/FPS request.
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        # Read after restoring, because some drivers choose a nearby mode.
        frame = self._read_stable_frame(cap, attempts=2)
        if frame is not None:
            h, w = frame.shape[:2]
            best_mode = (int(w), int(h))
            observed_modes.add(best_mode)

        CAMERA_MODE_CACHE[self.camera_index] = best_mode
        modes = sorted(observed_modes, key=lambda x: (x[0] * x[1], x[0]), reverse=True)
        return best_mode, modes

    def run(self) -> None:
        try:
            cap = open_capture(self.camera_index)
            self._cap = cap
            if cap is None:
                self.camera_error.emit(f"Could not open camera index {self.camera_index}.")
                return

            best_mode, observed_modes = self._probe_max_resolution(cap)
            if self._stop_event.is_set():
                return

            backend_name = "OpenCV"
            try:
                backend_name = cap.getBackendName()
            except Exception:
                pass

            reported_fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
            self.camera_ready.emit(
                {
                    "index": self.camera_index,
                    "width": best_mode[0],
                    "height": best_mode[1],
                    "reported_fps": reported_fps,
                    "observed_modes": observed_modes,
                    "backend": backend_name,
                }
            )
            self.probe_status.emit("Camera ready")

            self._powerline_controller = WindowsPowerLineController(self.camera_index)
            anti_flicker = AntiFlickerPipeline(
                self._powerline_controller,
                self.anti_flicker_status.emit,
            )
            anti_flicker.configure(self._requested_mode, time.perf_counter(), force=True)
            self._applied_mode = self._requested_mode

            frame_count = 0
            fps_window_start = time.perf_counter()
            last_emit = 0.0
            preview_interval = 1.0 / 30.0

            while not self._stop_event.is_set():
                ok, frame = cap.read()
                if not ok or frame is None or frame.size == 0:
                    time.sleep(0.005)
                    continue

                frame_count += 1
                now = time.perf_counter()

                with self._command_lock:
                    requested_mode = self._requested_mode
                    recalibrate = self._recalibration_requested
                    show_settings = self._driver_settings_requested
                    self._recalibration_requested = False
                    self._driver_settings_requested = False

                if requested_mode != self._applied_mode:
                    anti_flicker.configure(requested_mode, now, force=True)
                    self._applied_mode = requested_mode
                if recalibrate:
                    anti_flicker.configure(anti_flicker.mode, now, force=True)
                if show_settings:
                    self.anti_flicker_status.emit(
                        {
                            "mode": anti_flicker.mode,
                            "method": "Driver settings",
                            "status": "DRIVER SETTINGS OPEN",
                            "score": None,
                            "averaged_frames": 0,
                            "target_frames": 0,
                            "ready": False,
                            "supported": self._powerline_controller.supported,
                            "detail": "",
                        }
                    )
                    cap.set(cv2.CAP_PROP_SETTINGS, 0)
                    anti_flicker.configure(anti_flicker.mode, time.perf_counter(), force=True)

                processed_frame = anti_flicker.process_frame(frame, now)

                # Prevent the Qt event queue from filling with more than ~30 full-res frames/s.
                if now - last_emit >= preview_interval:
                    self.frame_ready.emit(processed_frame)
                    last_emit = now

                elapsed = now - fps_window_start
                if elapsed >= 1.0:
                    self.measured_fps.emit(frame_count / elapsed)
                    frame_count = 0
                    fps_window_start = now

        except Exception as exc:
            self.camera_error.emit(f"Camera error: {exc}")
        finally:
            if self._powerline_controller is not None:
                self._powerline_controller.close()
                self._powerline_controller = None
            if self._cap is not None:
                self._cap.release()
                self._cap = None


