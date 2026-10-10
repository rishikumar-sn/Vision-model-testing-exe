from __future__ import annotations

import re
import time
from pathlib import Path
from typing import Any

import numpy as np

from adapters.base import AdapterError, BaseAdapter
from core.device import resolve_device
from core.results import Classification, Detection, InferenceResult, KeypointSet, ModelInfo, PolygonMask


class UltralyticsAdapter(BaseAdapter):
    display_name = "Ultralytics YOLO"

    @staticmethod
    def _normalize_imgsz(value: Any):
        """Return an Ultralytics-compatible image size from checkpoint metadata."""
        if isinstance(value, (list, tuple)):
            try:
                dimensions = [int(item) for item in value]
            except (TypeError, ValueError):
                return None
            if len(dimensions) == 1 and dimensions[0] > 0:
                return dimensions[0]
            if len(dimensions) == 2 and all(item > 0 for item in dimensions):
                return dimensions[0] if dimensions[0] == dimensions[1] else dimensions
            return None
        try:
            size = int(value)
        except (TypeError, ValueError):
            return None
        return size if size > 0 else None

    @classmethod
    def _checkpoint_imgsz(cls, model):
        """Read the training image size embedded by different Ultralytics releases."""
        candidates = []
        overrides = getattr(model, "overrides", None)
        if isinstance(overrides, dict):
            candidates.append(overrides.get("imgsz"))

        inner_model = getattr(model, "model", None)
        model_args = getattr(inner_model, "args", None)
        if isinstance(model_args, dict):
            candidates.append(model_args.get("imgsz"))

        checkpoint = getattr(model, "ckpt", None)
        if isinstance(checkpoint, dict):
            train_args = checkpoint.get("train_args")
            if isinstance(train_args, dict):
                candidates.append(train_args.get("imgsz"))

        for candidate in candidates:
            size = cls._normalize_imgsz(candidate)
            if size is not None:
                return size
        return None

    @staticmethod
    def _onnx_input_size(source: str) -> tuple[int, int] | None:
        """Return a fixed ONNX image input as (height, width), if declared."""
        if Path(source).suffix.lower() != ".onnx":
            return None
        try:
            import onnxruntime as ort

            options = ort.SessionOptions()
            options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
            session = ort.InferenceSession(
                source,
                sess_options=options,
                providers=["CPUExecutionProvider"],
            )
            shape = list(session.get_inputs()[0].shape)
            del session
        except Exception:
            # Ultralytics remains responsible for loading the model. Some custom
            # graphs cannot be opened by a separate inspection session.
            return None

        if len(shape) != 4:
            return None
        nchw = shape[1] in (1, 3, 4) or not isinstance(shape[-1], int)
        height, width = (shape[2], shape[3]) if nchw else (shape[1], shape[2])
        if all(isinstance(value, int) and value > 0 for value in (height, width)):
            return int(height), int(width)
        return None

    @staticmethod
    def _input_size_from_ort_error(exc: Exception) -> tuple[int, int] | None:
        """Recover fixed NCHW dimensions from ONNX Runtime's shape error."""
        expected = {
            int(index): int(size)
            for index, size in re.findall(
                r"index:\s*(\d+).*?Expected:\s*(\d+)", str(exc), flags=re.IGNORECASE
            )
        }
        if 2 in expected and 3 in expected:
            return expected[2], expected[3]
        return None

    @staticmethod
    def _predict_imgsz(size: tuple[int, int] | None, fallback: int):
        if size is None:
            return fallback
        height, width = size
        return height if height == width else [height, width]

    def load(self, source: str, settings: dict[str, Any]) -> ModelInfo:
        try:
            from ultralytics import YOLO
        except Exception as exc:
            raise AdapterError("Ultralytics is not installed. Run: pip install ultralytics") from exc
        try:
            task = None
            if Path(source).suffix.lower() == ".onnx":
                try:
                    import onnxruntime as ort

                    session = ort.InferenceSession(source, providers=["CPUExecutionProvider"])
                    task = (session.get_modelmeta().custom_metadata_map or {}).get("task")
                    del session
                except Exception:
                    pass
            self.model = YOLO(source, task=task) if task in {"detect", "segment", "classify", "pose", "obb"} else YOLO(source)
        except Exception as exc:
            raise AdapterError(f"Ultralytics could not load this model: {exc}") from exc
        self.source = source
        self.device = resolve_device(settings.get("device", "auto"), self.device_status, "torch")
        self.task = str(getattr(self.model, "task", None) or "unknown")
        self.fixed_input_size = self._onnx_input_size(source)
        if self.task == "unknown":
            raise AdapterError("Ultralytics loaded the file but could not identify a supported task; trying another adapter is safer.")
        names = getattr(self.model, "names", None)
        requested_imgsz = int(settings.get("imgsz", 640))
        checkpoint_imgsz = self._checkpoint_imgsz(self.model)
        effective_imgsz = self._predict_imgsz(
            self.fixed_input_size,
            checkpoint_imgsz if checkpoint_imgsz is not None else requested_imgsz,
        )
        input_description = f"imgsz={effective_imgsz}"
        if self.fixed_input_size:
            height, width = self.fixed_input_size
            input_description = f"imgsz={width} x {height} (fixed ONNX input)"
        elif checkpoint_imgsz is not None:
            input_description = f"imgsz={effective_imgsz} (from model metadata)"
        return ModelInfo(
            source=source,
            adapter=self.display_name,
            task=self.task,
            device=self.device,
            input_description=input_description,
            output_description=f"Ultralytics Results • classes={len(names) if names else 'unknown'}",
            details={
                "names": names or {},
                "fixed_input_size": self.fixed_input_size,
                "checkpoint_imgsz": checkpoint_imgsz,
                "effective_imgsz": effective_imgsz,
            },
        )

    def infer(self, frame: np.ndarray, settings: dict[str, Any]) -> InferenceResult:
        conf = float(settings.get("conf", 0.25))
        iou = float(settings.get("iou", 0.45))
        requested_imgsz = int(settings.get("imgsz", 640))
        imgsz = self._predict_imgsz(getattr(self, "fixed_input_size", None), requested_imgsz)
        t0 = time.perf_counter()
        try:
            results = self.model.predict(
                source=frame,
                conf=conf,
                iou=iou,
                imgsz=imgsz,
                device=self.device,
                verbose=False,
            )
        except Exception as exc:
            # If load-time inspection was unavailable, ONNX Runtime still reports
            # exact fixed dimensions. Retry once and retain them for later frames.
            recovered_size = None
            if Path(getattr(self, "source", "")).suffix.lower() == ".onnx":
                recovered_size = self._input_size_from_ort_error(exc)
            if recovered_size and recovered_size != getattr(self, "fixed_input_size", None):
                self.fixed_input_size = recovered_size
                retry_imgsz = self._predict_imgsz(recovered_size, requested_imgsz)
                try:
                    results = self.model.predict(
                        source=frame,
                        conf=conf,
                        iou=iou,
                        imgsz=retry_imgsz,
                        device=self.device,
                        verbose=False,
                    )
                except Exception as retry_exc:
                    raise AdapterError(f"Ultralytics inference failed: {retry_exc}") from retry_exc
            else:
                raise AdapterError(f"Ultralytics inference failed: {exc}") from exc
        t1 = time.perf_counter()
        if not results:
            return InferenceResult(adapter=self.display_name, task=self.task, device=self.device,
                                   inference_ms=(t1 - t0) * 1000, total_ms=(t1 - t0) * 1000)

        r = results[0]
        out = InferenceResult(adapter=self.display_name, task=str(getattr(r, "task", self.task)), device=self.device)
        names = getattr(r, "names", {}) or {}

        boxes = getattr(r, "boxes", None)
        if boxes is not None and getattr(boxes, "xyxy", None) is not None:
            xyxy = boxes.xyxy.detach().cpu().numpy()
            confs = boxes.conf.detach().cpu().numpy() if boxes.conf is not None else np.ones(len(xyxy))
            classes = boxes.cls.detach().cpu().numpy().astype(int) if boxes.cls is not None else np.full(len(xyxy), -1)
            for coords, score, class_id in zip(xyxy, confs, classes):
                label = str(names.get(int(class_id), class_id))
                out.detections.append(Detection(*map(float, coords[:4]), float(score), int(class_id), label))

        masks = getattr(r, "masks", None)
        if masks is not None and getattr(masks, "xy", None) is not None:
            mask_polys = list(masks.xy)
            for idx, poly in enumerate(mask_polys):
                class_id = out.detections[idx].class_id if idx < len(out.detections) else -1
                score = out.detections[idx].confidence if idx < len(out.detections) else 1.0
                label = out.detections[idx].label if idx < len(out.detections) else "mask"
                pts = [(float(x), float(y)) for x, y in np.asarray(poly).reshape(-1, 2)]
                out.masks.append(PolygonMask(pts, score, class_id, label))

        obb = getattr(r, "obb", None)
        if obb is not None and getattr(obb, "xyxyxyxy", None) is not None:
            polys = obb.xyxyxyxy.detach().cpu().numpy()
            confs = obb.conf.detach().cpu().numpy() if obb.conf is not None else np.ones(len(polys))
            classes = obb.cls.detach().cpu().numpy().astype(int) if obb.cls is not None else np.full(len(polys), -1)
            out.detections.clear()
            for poly, score, class_id in zip(polys, confs, classes):
                pts_arr = np.asarray(poly).reshape(-1, 2)
                label = str(names.get(int(class_id), class_id))
                out.masks.append(PolygonMask([(float(x), float(y)) for x, y in pts_arr], float(score), int(class_id), label))
                x1, y1 = pts_arr.min(axis=0)
                x2, y2 = pts_arr.max(axis=0)
                out.detections.append(Detection(float(x1), float(y1), float(x2), float(y2), float(score), int(class_id), label))

        probs = getattr(r, "probs", None)
        if probs is not None:
            topk = max(1, int(settings.get("topk", 5)))
            data = probs.data.detach().cpu().numpy().reshape(-1)
            indices = np.argsort(data)[::-1][:topk]
            for idx in indices:
                out.classifications.append(Classification(str(names.get(int(idx), idx)), float(data[idx]), int(idx)))

        keypoints = getattr(r, "keypoints", None)
        if keypoints is not None and getattr(keypoints, "xy", None) is not None:
            xy = keypoints.xy.detach().cpu().numpy()
            conf_arr = None
            if getattr(keypoints, "conf", None) is not None:
                conf_arr = keypoints.conf.detach().cpu().numpy()
            for i, person in enumerate(xy):
                pts = []
                for j, (x, y) in enumerate(person):
                    c = float(conf_arr[i, j]) if conf_arr is not None else 1.0
                    pts.append((float(x), float(y), c))
                out.keypoints.append(KeypointSet(pts))

        speed = getattr(r, "speed", {}) or {}
        out.preprocess_ms = float(speed.get("preprocess", 0.0) or 0.0)
        out.inference_ms = float(speed.get("inference", (t1 - t0) * 1000) or 0.0)
        out.postprocess_ms = float(speed.get("postprocess", 0.0) or 0.0)
        out.total_ms = out.preprocess_ms + out.inference_ms + out.postprocess_ms
        if out.total_ms <= 0:
            out.total_ms = (t1 - t0) * 1000
        out.summary = f"{len(out.detections)} detections • {len(out.masks)} masks"
        if out.classifications:
            out.summary = " • ".join(f"{c.label} {c.score*100:.1f}%" for c in out.classifications[:3])
        return out
