from __future__ import annotations

import ctypes
import platform
from collections import deque
from typing import Optional

import cv2
import numpy as np

def prepare_flicker_sample(frame: np.ndarray) -> np.ndarray:
    """Return a tiny, globally normalized grayscale sample for temporal scoring."""
    sample = cv2.resize(frame, (320, 180), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(sample, cv2.COLOR_BGR2GRAY).astype(np.float32)
    gray /= max(float(gray.mean()), 1.0)
    return gray


def temporal_flicker_score(previous: np.ndarray, current: np.ndarray) -> float:
    """Score changing full-width horizontal bands; static printed lines cancel."""
    row_signal = np.median(current - previous, axis=1)
    return float(np.std(row_signal))


class WindowsPowerLineController:
    """DirectShow access to KSPROPERTY_VIDEOPROCAMP_POWERLINE_FREQUENCY."""

    PROPERTY_SET = "{C6E13360-30AC-11D0-A18C-00A0C9118956}"
    PROPERTY_ID = 13
    SUPPORT_GET = 0x01
    SUPPORT_SET = 0x02
    TYPE_GET = 0x01
    TYPE_SET = 0x02
    FLAGS_MANUAL = 0x02

    def __init__(self, camera_index: int):
        self.supported = False
        self.error = "Power-line control is available only on Windows."
        self._co_uninitialize = None
        self._filter = None
        self._ks = None
        self._property_set = None
        self._original_powerline: Optional[int] = None

        if platform.system().lower() != "windows":
            return

        try:
            from comtypes import COMMETHOD, GUID, HRESULT, IUnknown
            from comtypes import CoInitialize, CoUninitialize
            from pygrabber.dshow_graph import DeviceCategories, SystemDeviceEnum

            class IKsPropertySet(IUnknown):
                _iid_ = GUID("{31EFAC30-515C-11D0-A9AA-00AA0061BE93}")
                _methods_ = [
                    COMMETHOD(
                        [], HRESULT, "Set",
                        (["in"], ctypes.POINTER(GUID), "guidPropSet"),
                        (["in"], ctypes.c_ulong, "propertyId"),
                        (["in"], ctypes.c_void_p, "instanceData"),
                        (["in"], ctypes.c_ulong, "instanceSize"),
                        (["in"], ctypes.c_void_p, "propertyData"),
                        (["in"], ctypes.c_ulong, "propertySize"),
                    ),
                    COMMETHOD(
                        [], HRESULT, "Get",
                        (["in"], ctypes.POINTER(GUID), "guidPropSet"),
                        (["in"], ctypes.c_ulong, "propertyId"),
                        (["in"], ctypes.c_void_p, "instanceData"),
                        (["in"], ctypes.c_ulong, "instanceSize"),
                        (["in"], ctypes.c_void_p, "propertyData"),
                        (["in"], ctypes.c_ulong, "propertySize"),
                        (["out"], ctypes.POINTER(ctypes.c_ulong), "bytesReturned"),
                    ),
                    COMMETHOD(
                        [], HRESULT, "QuerySupported",
                        (["in"], ctypes.POINTER(GUID), "guidPropSet"),
                        (["in"], ctypes.c_ulong, "propertyId"),
                        (["out"], ctypes.POINTER(ctypes.c_ulong), "typeSupport"),
                    ),
                ]

            CoInitialize()
            self._co_uninitialize = CoUninitialize
            device_enum = SystemDeviceEnum()
            self._filter, _name = device_enum.get_filter_by_index(
                DeviceCategories.VideoInputDevice,
                camera_index,
            )
            self._ks = self._filter.QueryInterface(IKsPropertySet)
            self._property_set = GUID(self.PROPERTY_SET)
            support = int(
                self._ks.QuerySupported(
                    ctypes.byref(self._property_set),
                    self.PROPERTY_ID,
                )
            )
            self.supported = bool(support & self.SUPPORT_SET)
            self.error = "" if self.supported else "Camera driver does not support setting Power Line Frequency."
            self._original_powerline = self.get_value()
        except Exception as exc:
            self.error = f"Power-line control unavailable: {exc}"

    def _property_buffer(self, request_type: int, value: int = 0):
        # Some UVC drivers return 40 bytes for the nominally 36-byte structure;
        # a zero-filled 64-byte buffer accommodates both implementations.
        data = ctypes.create_string_buffer(64)
        ctypes.memmove(data, ctypes.byref(self._property_set), 16)
        ctypes.c_uint32.from_buffer(data, 16).value = self.PROPERTY_ID
        ctypes.c_uint32.from_buffer(data, 20).value = request_type
        ctypes.c_int32.from_buffer(data, 24).value = int(value)
        ctypes.c_uint32.from_buffer(data, 28).value = self.FLAGS_MANUAL
        return data

    def get_value(self) -> Optional[int]:
        if not self.supported or self._ks is None:
            return None
        try:
            data = self._property_buffer(self.TYPE_GET)
            self._ks.Get(
                ctypes.byref(self._property_set), self.PROPERTY_ID,
                ctypes.byref(data), len(data), ctypes.byref(data), len(data),
            )
            return int(ctypes.c_int32.from_buffer(data, 24).value)
        except Exception:
            return None

    def set_value(self, value: int) -> bool:
        if not self.supported or self._ks is None:
            return False
        try:
            data = self._property_buffer(self.TYPE_SET, value)
            self._ks.Set(
                ctypes.byref(self._property_set), self.PROPERTY_ID,
                ctypes.byref(data), len(data), ctypes.byref(data), len(data),
            )
            return True
        except Exception as exc:
            self.error = f"Camera rejected anti-flicker value {value}: {exc}"
            return False

    def close(self) -> None:
        if self._original_powerline is not None:
            self.set_value(self._original_powerline)
        self._ks = None
        self._filter = None
        if self._co_uninitialize is not None:
            self._co_uninitialize()
            self._co_uninitialize = None


class TemporalFrameAverage:
    """Full-resolution rolling average used by the proven Windows fallback."""

    MAX_STORAGE_BYTES = 640 * 1024 * 1024

    def __init__(self, requested_frames: int):
        self.requested_frames = max(1, min(90, int(requested_frames)))
        self.target_frames = self.requested_frames
        self._frames: deque[np.ndarray] = deque()
        self._sum: Optional[np.ndarray] = None

    @property
    def count(self) -> int:
        return len(self._frames)

    def reset(self) -> None:
        self._frames.clear()
        self._sum = None

    def apply(self, frame: np.ndarray) -> np.ndarray:
        if self.requested_frames <= 1:
            return frame
        if self._sum is None or self._sum.shape != frame.shape:
            self.reset()
            self._sum = np.zeros(frame.shape, dtype=np.float32)
            memory_limit = max(2, self.MAX_STORAGE_BYTES // max(1, frame.nbytes))
            self.target_frames = min(self.requested_frames, int(memory_limit))
        if len(self._frames) == self.target_frames:
            oldest = self._frames.popleft()
            np.subtract(self._sum, oldest, out=self._sum, casting="unsafe")
        stored = frame.copy()
        self._frames.append(stored)
        np.add(self._sum, stored, out=self._sum, casting="unsafe")
        return cv2.convertScaleAbs(self._sum, alpha=1.0 / len(self._frames))


def normalize_horizontal_bands(frame: np.ndarray, gain_limit: float = 1.35) -> np.ndarray:
    """Optional row correction retained as a selectable, non-default method."""
    height, width = frame.shape[:2]
    if height < 80 or width < 80:
        return frame
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    sample_width = min(480, width)
    if sample_width < width:
        sample_height = max(40, int(round(height * sample_width / width)))
        gray = cv2.resize(gray, (sample_width, sample_height), interpolation=cv2.INTER_AREA)
    row_luma = np.median(gray.astype(np.float32), axis=1)
    kernel = max(9, ((gray.shape[0] // 12) * 2) + 1)
    smooth = cv2.GaussianBlur(row_luma.reshape(-1, 1), (1, kernel), 0).reshape(-1)
    target = float(np.median(smooth))
    if target <= 1.0:
        return frame
    gains = np.clip(target / np.maximum(smooth, 1.0), 1.0 / gain_limit, gain_limit)
    if gains.shape[0] != height:
        gains = cv2.resize(gains.reshape(-1, 1), (1, height), interpolation=cv2.INTER_LINEAR).reshape(height)
    gain_map = np.broadcast_to(gains.reshape(height, 1, 1), frame.shape)
    return cv2.multiply(frame, gain_map, dtype=cv2.CV_8U)


class AntiFlickerPipeline:
    """Driver-auto exposure with raw, averaging, row, and adaptive methods."""

    ADAPTIVE_SAMPLE_COUNT = 20
    ADAPTIVE_ENABLE_SCORE = 0.12
    ADAPTIVE_DISABLE_SCORE = 0.07

    def __init__(self, hardware: WindowsPowerLineController, status_callback):
        self.hardware = hardware
        self.status_callback = status_callback
        self.mode = "adaptive"
        self.previous_sample: Optional[np.ndarray] = None
        self.scores: deque[float] = deque(maxlen=30)
        self.average = TemporalFrameAverage(90)
        self.adaptive_averaging = False
        self.last_ready: Optional[bool] = None
        self.last_emit = -1e9

    def configure(self, mode: str, now: float, force: bool = False) -> None:
        if mode == self.mode and not force:
            return
        self.mode = mode
        self.previous_sample = None
        self.scores.clear()
        self.average.reset()
        self.adaptive_averaging = False
        self.last_ready = None

        # Exposure is deliberately untouched. The user's camera-driver setting
        # (automatic or manual) remains authoritative in every anti-flicker mode.
        if mode in {"adaptive", "driver_50", "average_30", "average_90", "row", "hybrid"}:
            self.hardware.set_value(1)
        elif mode == "driver_60":
            self.hardware.set_value(2)
        elif mode == "normal" and self.hardware._original_powerline is not None:
            self.hardware.set_value(self.hardware._original_powerline)

        requested = 30 if mode == "average_30" else 90
        self.average = TemporalFrameAverage(requested)
        self._emit(now, "ANALYZING" if mode == "adaptive" else "ACTIVE", None, False)

    def _emit(
        self,
        now: float,
        status: str,
        score: Optional[float],
        ready: bool,
        method: Optional[str] = None,
    ) -> None:
        method_names = {
            "adaptive": "Adaptive",
            "normal": "Normal raw",
            "driver_50": "Driver 50 Hz",
            "driver_60": "Driver 60 Hz",
            "average_30": "Average 30",
            "average_90": "Average 90",
            "row": "Row correction",
            "hybrid": "Average 90 + row",
        }
        self.status_callback(
            {
                "mode": self.mode,
                "method": method or method_names.get(self.mode, self.mode),
                "status": status,
                "score": score,
                "averaged_frames": self.average.count,
                "target_frames": self.average.target_frames,
                "ready": ready,
                "supported": self.hardware.supported,
                "detail": self.hardware.error,
            }
        )
        self.last_ready = ready
        self.last_emit = now

    def process_frame(self, frame: np.ndarray, now: float) -> np.ndarray:
        sample = prepare_flicker_sample(frame)
        if self.previous_sample is not None:
            self.scores.append(temporal_flicker_score(self.previous_sample, sample))
        self.previous_sample = sample
        score = float(np.median(self.scores)) if self.scores else None

        use_average = self.mode in {"average_30", "average_90", "hybrid"}
        use_row = self.mode in {"row", "hybrid"}
        status = "ACTIVE"
        ready = True
        method = None

        if self.mode == "adaptive":
            if len(self.scores) < self.ADAPTIVE_SAMPLE_COUNT:
                status = "ANALYZING"
                ready = False
            else:
                if score is not None and score >= self.ADAPTIVE_ENABLE_SCORE:
                    self.adaptive_averaging = True
                elif score is not None and score <= self.ADAPTIVE_DISABLE_SCORE:
                    self.adaptive_averaging = False
                    self.average.reset()
                use_average = self.adaptive_averaging
                method = "Adaptive average 90" if use_average else "Adaptive raw"

        output = self.average.apply(frame) if use_average else frame
        if use_row:
            output = normalize_horizontal_bands(output)

        if use_average:
            ready = self.average.count >= self.average.target_frames
            status = "READY" if ready else "STABILIZING"
        elif self.mode == "adaptive" and ready:
            status = "READY"

        # Emit readiness transitions immediately so capture is enabled on the
        # exact frame that completes an averaging window.
        if now - self.last_emit >= 0.35 or not ready or ready != self.last_ready:
            self._emit(now, status, score, ready, method)
        return output


