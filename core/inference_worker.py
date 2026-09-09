from __future__ import annotations

import queue
import statistics
import threading
import time
from typing import Any, Optional

import numpy as np
from PyQt6.QtCore import QThread, pyqtSignal

from adapters import adapter_classes_for
from adapters.base import AdapterError, BaseAdapter
from core.device import DeviceStatus
from core.results import BenchmarkResult, InferenceResult, ModelInfo
from processing.roi import crop_frame, normalized_roi_to_pixels


class InferenceThread(QThread):
    model_loaded = pyqtSignal(object)      # ModelInfo
    model_load_failed = pyqtSignal(str)
    model_unloaded = pyqtSignal()
    result_ready = pyqtSignal(object)      # InferenceResult
    benchmark_ready = pyqtSignal(object)   # BenchmarkResult
    error = pyqtSignal(str)
    status = pyqtSignal(str)
    busy_changed = pyqtSignal(bool)

    def __init__(self, device_status: DeviceStatus, parent=None):
        super().__init__(parent)
        self.device_status = device_status
        self._commands: queue.Queue = queue.Queue()
        self._stop_event = threading.Event()
        self._pending_lock = threading.Lock()
        self._pending_infer: Optional[tuple[np.ndarray, dict[str, Any], Any, bool]] = None
        self._adapter: Optional[BaseAdapter] = None
        self._busy = False

    def _set_busy(self, value: bool) -> None:
        if self._busy != value:
            self._busy = value
            self.busy_changed.emit(value)

    def load_model(self, source: str, adapter_name: str, settings: dict[str, Any]) -> None:
        self._commands.put(("load", source, adapter_name, dict(settings)))

    def unload_model(self) -> None:
        self._commands.put(("unload",))

    def submit_latest(self, frame: np.ndarray, settings: dict[str, Any], roi_norm, use_roi: bool) -> None:
        if frame is None:
            return
        request = (frame.copy(), dict(settings), roi_norm, bool(use_roi))
        with self._pending_lock:
            self._pending_infer = request

    def benchmark(self, frame: np.ndarray, settings: dict[str, Any], roi_norm, use_roi: bool,
                  runs: int = 20, warmup: int = 3) -> None:
        self._commands.put(("benchmark", frame.copy(), dict(settings), roi_norm, bool(use_roi), runs, warmup))

    def stop(self) -> None:
        self._stop_event.set()
        self._commands.put(("stop",))

    def run(self) -> None:
        while not self._stop_event.is_set():
            did_work = False
            try:
                cmd = self._commands.get(timeout=0.01)
                did_work = True
                if not self._handle_command(cmd):
                    break
            except queue.Empty:
                pass
            except Exception as exc:
                self.error.emit(self._format_error(exc))
                self._set_busy(False)

            pending = None
            with self._pending_lock:
                if self._pending_infer is not None:
                    pending = self._pending_infer
                    self._pending_infer = None
            if pending is not None and self._adapter is not None:
                did_work = True
                self._set_busy(True)
                try:
                    result = self._infer_request(*pending)
                    self.result_ready.emit(result)
                except Exception as exc:
                    self.error.emit(self._format_error(exc))
                finally:
                    self._set_busy(False)
            if not did_work:
                time.sleep(0.002)
        self._close_adapter()

    def _handle_command(self, cmd) -> bool:
        name = cmd[0]
        if name == "stop":
            return False
        if name == "unload":
            self._set_busy(True)
            self._close_adapter()
            self.model_unloaded.emit()
            self._set_busy(False)
            return True
        if name == "load":
            _, source, adapter_name, settings = cmd
            self._set_busy(True)
            try:
                self._load(source, adapter_name, settings)
            except Exception as exc:
                self.model_load_failed.emit(self._format_error(exc))
            finally:
                self._set_busy(False)
            return True
        if name == "benchmark":
            _, frame, settings, roi_norm, use_roi, runs, warmup = cmd
            if self._adapter is None:
                self.error.emit("Load a model before benchmarking.")
                return True
            self._set_busy(True)
            try:
                bench = self._run_benchmark(frame, settings, roi_norm, use_roi, int(runs), int(warmup))
                self.benchmark_ready.emit(bench)
            except Exception as exc:
                self.error.emit(self._format_error(exc))
            self._set_busy(False)
        return True

    def _load(self, source: str, adapter_name: str, settings: dict[str, Any]) -> None:
        self._close_adapter()
        errors = []
        for cls in adapter_classes_for(source, adapter_name):
            adapter = cls(self.device_status)
            try:
                self.status.emit(f"Trying {adapter.display_name}...")
                info = adapter.load(source, settings)
                self._adapter = adapter
                self.model_loaded.emit(info)
                self.status.emit(f"Loaded with {adapter.display_name}")
                return
            except Exception as exc:
                try:
                    adapter.close()
                except Exception:
                    pass
                errors.append(f"{adapter.display_name}: {exc}")
        raise AdapterError("\n\n".join(errors) if errors else "No adapter could load the model.")

    def _close_adapter(self) -> None:
        if self._adapter is not None:
            try:
                self._adapter.close()
            except Exception:
                pass
        self._adapter = None
        with self._pending_lock:
            self._pending_infer = None

    def _prepare_request(self, frame, settings, roi_norm, use_roi):
        work, (dx, dy), roi_pixels = crop_frame(frame, roi_norm, use_roi)
        local = dict(settings)
        h, w = work.shape[:2]
        if local.get("fastsam_prompt") == "roi_box":
            if use_roi and roi_norm:
                local["prompt_box"] = [0, 0, max(1, w - 1), max(1, h - 1)]
            elif roi_norm:
                fh, fw = frame.shape[:2]
                x1, y1, x2, y2 = normalized_roi_to_pixels(roi_norm, fw, fh)
                local["prompt_box"] = [x1, y1, x2, y2]
            else:
                local["prompt_box"] = [0, 0, max(1, w - 1), max(1, h - 1)]
        if local.get("fastsam_prompt") == "center_point":
            if use_roi and roi_norm:
                local["prompt_point"] = [w // 2, h // 2]
            elif roi_norm:
                fh, fw = frame.shape[:2]
                x1, y1, x2, y2 = normalized_roi_to_pixels(roi_norm, fw, fh)
                local["prompt_point"] = [(x1 + x2) // 2, (y1 + y2) // 2]
            else:
                local["prompt_point"] = [w // 2, h // 2]
        return work, local, (dx, dy), roi_pixels

    def _infer_request(self, frame, settings, roi_norm, use_roi) -> InferenceResult:
        work, local, (dx, dy), roi_pixels = self._prepare_request(frame, settings, roi_norm, use_roi)
        result = self._adapter.infer(work, local)
        result.shifted(dx, dy)
        result.metadata["roi_pixels"] = roi_pixels
        result.metadata["inference_shape"] = tuple(work.shape[:2][::-1])
        return result

    def _run_benchmark(self, frame, settings, roi_norm, use_roi, runs, warmup) -> BenchmarkResult:
        runs = max(1, min(500, runs))
        warmup = max(0, min(20, warmup))
        work, local, _offset, _roi_pixels = self._prepare_request(frame, settings, roi_norm, use_roi)
        self.status.emit(f"Benchmark warmup: {warmup}")
        for _ in range(warmup):
            self._adapter.infer(work, local)
        times = []
        for i in range(runs):
            t0 = time.perf_counter()
            self._adapter.infer(work, local)
            times.append((time.perf_counter() - t0) * 1000)
            if (i + 1) % 5 == 0 or i + 1 == runs:
                self.status.emit(f"Benchmark {i+1}/{runs}")
        sorted_times = sorted(times)
        p95_index = min(len(sorted_times) - 1, max(0, int(round(0.95 * len(sorted_times))) - 1))
        avg = statistics.fmean(times)
        return BenchmarkResult(
            runs=runs,
            warmup_runs=warmup,
            average_ms=avg,
            median_ms=statistics.median(times),
            minimum_ms=min(times),
            maximum_ms=max(times),
            p95_ms=sorted_times[p95_index],
            fps=1000.0 / avg if avg > 0 else 0.0,
        )

    @staticmethod
    def _format_error(exc: Exception) -> str:
        return f"{type(exc).__name__}: {exc}"
