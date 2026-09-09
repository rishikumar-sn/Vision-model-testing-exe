from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from adapters.base import AdapterError, BaseAdapter
from core.results import Classification, Detection, InferenceResult, ModelInfo


class KerasAdapter(BaseAdapter):
    """TensorFlow adapter for Keras, SavedModel, and TensorFlow Lite models."""

    display_name = "TensorFlow / Keras"

    def load(self, source: str, settings: dict[str, Any]) -> ModelInfo:
        try:
            import tensorflow as tf
        except Exception as exc:
            raise AdapterError("TensorFlow is not installed. Run: pip install tensorflow") from exc

        self.tf = tf
        self.source = source
        self.backend = "keras"
        path = Path(source)
        try:
            if path.suffix.lower() == ".tflite":
                self.backend = "tflite"
                self.model = tf.lite.Interpreter(model_path=str(path))
                self.model.allocate_tensors()
                self.input_detail = self.model.get_input_details()[0]
                self.output_details = self.model.get_output_details()
                input_shape = tuple(int(x) for x in self.input_detail.get("shape_signature", self.input_detail["shape"]))
                output_shape = [tuple(int(x) for x in item["shape"]) for item in self.output_details]
            elif path.is_dir() and (path / "saved_model.pb").is_file():
                self.backend = "saved_model"
                loaded = tf.saved_model.load(str(path))
                signatures = getattr(loaded, "signatures", {})
                self.model = signatures.get("serving_default") or next(iter(signatures.values()), None)
                if self.model is None:
                    raise AdapterError("SavedModel has no callable serving signature.")
                _args, kwargs = self.model.structured_input_signature
                if not kwargs:
                    raise AdapterError("SavedModel serving signature has no named image input.")
                self.input_name, spec = next(iter(kwargs.items()))
                input_shape = tuple(spec.shape.as_list())
                output_shape = str(self.model.structured_outputs)
            else:
                self.model = tf.keras.models.load_model(str(path), compile=False)
                input_shape = getattr(self.model, "input_shape", None)
                output_shape = getattr(self.model, "output_shape", None)
        except AdapterError:
            raise
        except Exception as exc:
            kind = "TensorFlow Lite" if self.backend == "tflite" else "TensorFlow/Keras"
            raise AdapterError(f"{kind} could not load this model: {exc}") from exc

        requested = str(settings.get("device", "auto")).lower()
        gpus = tf.config.list_physical_devices("GPU")
        self.device = "/GPU:0" if requested != "cpu" and gpus and self.backend != "tflite" else "/CPU:0"
        self.task = "generic"
        self.input_shape = input_shape
        return ModelInfo(
            source, self.display_name, self.task, self.device,
            f"{self.backend}: {input_shape}", str(output_shape), {"tensorflow_backend": self.backend},
        )

    @staticmethod
    def _dimension(value, fallback: int) -> int:
        try:
            value = int(value)
            return value if value > 0 else fallback
        except (TypeError, ValueError):
            return fallback

    def _prepare(self, frame: np.ndarray, settings: dict[str, Any]) -> np.ndarray:
        shape = list(self.input_shape or (None, 224, 224, 3))
        if len(shape) != 4:
            raise AdapterError(f"TensorFlow image adapter expects a rank-4 input, got {shape}")
        channels_first = shape[1] in (1, 3, 4) and shape[-1] not in (1, 3, 4)
        fallback = int(settings.get("classifier_size", settings.get("imgsz", 224)))
        h = self._dimension(shape[2] if channels_first else shape[1], fallback)
        w = self._dimension(shape[3] if channels_first else shape[2], fallback)
        channels = self._dimension(shape[1] if channels_first else shape[3], 3)

        resized = cv2.resize(frame, (w, h), interpolation=cv2.INTER_LINEAR)
        if channels == 1:
            arr = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)[..., None]
        else:
            arr = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
            if channels == 4:
                arr = np.concatenate([arr, np.full((h, w, 1), 255, dtype=np.uint8)], axis=2)
        arr = arr.astype(np.float32)
        if bool(settings.get("normalize", True)):
            arr /= 255.0
        if channels_first:
            arr = arr.transpose(2, 0, 1)
        return np.expand_dims(arr, 0)

    def _run(self, batch: np.ndarray):
        tf = self.tf
        if self.backend == "tflite":
            detail = self.input_detail
            if tuple(detail["shape"]) != tuple(batch.shape):
                signature = np.asarray(detail.get("shape_signature", detail["shape"]))
                if np.any(signature <= 0):
                    self.model.resize_tensor_input(detail["index"], batch.shape, strict=False)
                    self.model.allocate_tensors()
                    self.input_detail = self.model.get_input_details()[0]
                    self.output_details = self.model.get_output_details()
                    detail = self.input_detail
            dtype = detail["dtype"]
            value = batch
            if np.issubdtype(dtype, np.integer):
                scale, zero = detail.get("quantization", (0.0, 0))
                if scale >= 0.5 and value.max(initial=0) <= 1.0:
                    # Integer image models with scale=1 conventionally consume
                    # raw 0..255 pixels, even when the shared normalize UI is on.
                    value = value * 255.0
                if scale:
                    value = np.round(value / scale + zero)
                elif value.max(initial=0) <= 1.0:
                    value = value * np.iinfo(dtype).max
                value = np.clip(value, np.iinfo(dtype).min, np.iinfo(dtype).max)
            self.model.set_tensor(detail["index"], value.astype(dtype))
            self.model.invoke()
            outputs = {}
            for item in self.output_details:
                value = self.model.get_tensor(item["index"])
                scale, zero = item.get("quantization", (0.0, 0))
                if scale and np.issubdtype(value.dtype, np.integer):
                    value = (value.astype(np.float32) - zero) * scale
                outputs[item["name"]] = value
            return outputs
        tensor = tf.convert_to_tensor(batch)
        with tf.device(self.device):
            if self.backend == "saved_model":
                return self.model(**{self.input_name: tensor})
            return self.model(tensor, training=False)

    def infer(self, frame: np.ndarray, settings: dict[str, Any]) -> InferenceResult:
        t0 = time.perf_counter()
        batch = self._prepare(frame, settings)
        t1 = time.perf_counter()
        try:
            pred = self._run(batch)
        except Exception as exc:
            raise AdapterError(f"TensorFlow inference failed: {exc}") from exc
        t2 = time.perf_counter()
        out = self._postprocess(pred, frame.shape[1], frame.shape[0], settings)
        t3 = time.perf_counter()
        out.adapter = self.display_name
        out.device = self.device
        out.preprocess_ms = (t1 - t0) * 1000
        out.inference_ms = (t2 - t1) * 1000
        out.postprocess_ms = (t3 - t2) * 1000
        out.total_ms = (t3 - t0) * 1000
        return out

    @staticmethod
    def _numpy(value):
        return value.numpy() if hasattr(value, "numpy") else np.asarray(value)

    def _postprocess(self, pred, width: int, height: int, settings: dict[str, Any]) -> InferenceResult:
        out = InferenceResult(task="generic")
        if isinstance(pred, dict):
            values = {str(k).lower(): self._numpy(v) for k, v in pred.items()}
            boxes = next((v for k, v in values.items() if "box" in k and np.asarray(v).shape[-1:] == (4,)), None)
            scores = next((v for k, v in values.items() if "score" in k or "confidence" in k), None)
            classes = next((v for k, v in values.items() if "class" in k or "label" in k), None)
            count = next((v for k, v in values.items() if "num" in k and "detect" in k), None)
            # TFLite often erases semantic output names. Recover the standard
            # boxes/classes/scores/count quartet from compatible tensor shapes.
            if boxes is None:
                arrays = [np.asarray(v) for v in values.values()]
                boxes = next((v for v in arrays if v.ndim >= 2 and v.shape[-1] == 4), None)
                if boxes is not None:
                    n = np.asarray(boxes).reshape(-1, 4).shape[0]
                    vectors = [v.reshape(-1) for v in arrays if v is not boxes and v.size == n]
                    score_candidates = [v for v in vectors if np.issubdtype(v.dtype, np.floating) and np.all(np.isfinite(v)) and np.all((v >= 0) & (v <= 1))]
                    if score_candidates:
                        scores = max(score_candidates, key=lambda v: float(np.mean(np.abs(v - np.round(v)) > 1e-5)))
                        classes = next((v for v in vectors if v is not scores), None)
            if boxes is not None and scores is not None:
                boxes = np.asarray(boxes).reshape(-1, 4)
                scores = np.asarray(scores).reshape(-1)
                classes = np.asarray(classes).reshape(-1) if classes is not None else np.zeros(len(scores))
                limit = int(np.asarray(count).reshape(-1)[0]) if count is not None else min(len(boxes), len(scores))
                labels = self._labels(settings)
                threshold = float(settings.get("conf", 0.25))
                for box, score, class_id in zip(boxes[:limit], scores[:limit], classes[:limit]):
                    if float(score) < threshold:
                        continue
                    # TensorFlow detection models conventionally return normalized y1,x1,y2,x2.
                    y1, x1, y2, x2 = map(float, box)
                    if np.max(np.abs(box)) <= 2.0:
                        x1, x2, y1, y2 = x1 * width, x2 * width, y1 * height, y2 * height
                    cid = int(class_id)
                    label = labels[cid] if 0 <= cid < len(labels) else str(cid)
                    out.detections.append(Detection(x1, y1, x2, y2, float(score), cid, label))
                out.task = "detect"
                out.summary = f"{len(out.detections)} detections"
                return out
            pred = next(iter(values.values()), np.array([]))
        elif isinstance(pred, (list, tuple)):
            pred = pred[0] if pred else np.array([])

        arr = self._numpy(pred).reshape(-1).astype(np.float32)
        if not arr.size:
            out.summary = "No TensorFlow outputs"
            return out
        if arr.size == 1:
            arr = np.array([1.0 - float(arr[0]), float(arr[0])], dtype=np.float32)
        elif arr.min() < 0 or arr.max() > 1 or abs(float(arr.sum()) - 1.0) > 0.05:
            exp = np.exp(arr - np.max(arr))
            arr = exp / max(float(exp.sum()), 1e-12)
        labels = self._labels(settings)
        idxs = np.argsort(arr)[::-1][:max(1, int(settings.get("topk", 5)))]
        out.task = "classify"
        for idx in idxs:
            label = labels[idx] if idx < len(labels) else str(idx)
            out.classifications.append(Classification(label, float(arr[idx]), int(idx)))
        out.summary = " | ".join(f"{c.label} {c.score*100:.1f}%" for c in out.classifications[:3])
        return out
