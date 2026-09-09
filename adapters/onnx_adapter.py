from __future__ import annotations

import ast
import json
import math
import time
from typing import Any

import cv2
import numpy as np

from adapters.base import AdapterError, BaseAdapter
from core.device import resolve_device
from core.results import Classification, Detection, InferenceResult, ModelInfo, PolygonMask


def _shape_text(shape) -> str:
    return " × ".join(str(x) for x in shape)


def _softmax(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float64)
    x = x - np.max(x)
    exp = np.exp(x)
    return (exp / max(exp.sum(), 1e-12)).astype(np.float32)


def _metadata_labels(meta: dict[str, Any]) -> list[str]:
    value = meta.get("names")
    if not value:
        return []
    if isinstance(value, str):
        for decoder in (json.loads, ast.literal_eval):
            try:
                value = decoder(value)
                break
            except (ValueError, SyntaxError, TypeError, json.JSONDecodeError):
                continue
    if isinstance(value, dict):
        try:
            return [str(value[key]) for key in sorted(value, key=lambda key: int(key))]
        except (ValueError, TypeError):
            return [str(item) for item in value.values()]
    if isinstance(value, (list, tuple)):
        return [str(item) for item in value]
    return []


class ONNXAdapter(BaseAdapter):
    display_name = "ONNX Runtime"

    def load(self, source: str, settings: dict[str, Any]) -> ModelInfo:
        try:
            import onnxruntime as ort
        except Exception as exc:
            raise AdapterError("ONNX Runtime is not installed. CPU: pip install onnxruntime; NVIDIA: install onnxruntime-gpu") from exc
        provider = resolve_device(settings.get("device", "auto"), self.device_status, "onnx")
        available = list(ort.get_available_providers())
        providers = [provider]
        if provider != "CPUExecutionProvider" and "CPUExecutionProvider" in available:
            providers.append("CPUExecutionProvider")
        try:
            self.model = ort.InferenceSession(source, providers=providers)
        except Exception as exc:
            raise AdapterError(f"ONNX Runtime could not load this model: {exc}") from exc
        self.source = source
        self.device = self.model.get_providers()[0] if self.model.get_providers() else provider
        inputs = self.model.get_inputs()
        outputs = self.model.get_outputs()
        if not inputs:
            raise AdapterError("ONNX graph has no inputs.")
        self.output_names = [item.name for item in outputs]
        self.task = "generic"
        try:
            meta = self.model.get_modelmeta().custom_metadata_map or {}
            self.task = str(meta.get("task") or "generic")
        except Exception:
            meta = {}
        self.model_labels = _metadata_labels(meta)
        return ModelInfo(
            source, self.display_name, self.task, self.device,
            "; ".join(f"{x.name}: {_shape_text(x.shape)} {x.type}" for x in inputs),
            "; ".join(f"{x.name}: {_shape_text(x.shape)} {x.type}" for x in outputs),
            {"providers": self.model.get_providers(), "metadata": meta, "names": self.model_labels,
             "output_names": self.output_names},
        )

    def _prepare(self, frame: np.ndarray, settings: dict[str, Any]):
        inp = self.model.get_inputs()[0]
        shape = list(inp.shape)
        imgsz = int(settings.get("imgsz", 640))
        h0, w0 = frame.shape[:2]
        nchw = len(shape) == 4 and (shape[1] in (1, 3, 4) or not isinstance(shape[-1], int))
        if len(shape) != 4:
            raise AdapterError(f"Generic ONNX image adapter expects a 4D image input, got {shape}")
        if nchw:
            target_h = shape[2] if isinstance(shape[2], int) and shape[2] > 0 else imgsz
            target_w = shape[3] if isinstance(shape[3], int) and shape[3] > 0 else imgsz
        else:
            target_h = shape[1] if isinstance(shape[1], int) and shape[1] > 0 else imgsz
            target_w = shape[2] if isinstance(shape[2], int) and shape[2] > 0 else imgsz
        target_w, target_h = int(target_w), int(target_h)
        resized = cv2.resize(frame, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
        channels = shape[1] if nchw else shape[3]
        if channels == 1:
            arr = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)[..., None].astype(np.float32)
        else:
            arr = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32)
            if channels == 4:
                arr = np.concatenate([arr, np.full((target_h, target_w, 1), 255, dtype=np.float32)], axis=2)
        if bool(settings.get("normalize", True)):
            arr /= 255.0
        if nchw:
            arr = arr.transpose(2, 0, 1)
        arr = np.expand_dims(arr, 0)
        input_type = str(getattr(inp, "type", "tensor(float)"))
        if "float16" in input_type:
            arr = arr.astype(np.float16)
        elif "uint8" in input_type:
            if bool(settings.get("normalize", True)):
                arr = np.clip(arr * 255.0, 0, 255)
            arr = arr.astype(np.uint8)
        else:
            arr = arr.astype(np.float32)
        feeds = {inp.name: arr}
        for extra in self.model.get_inputs()[1:]:
            extra_name = extra.name.lower()
            extra_type = str(getattr(extra, "type", ""))
            dtype = np.int64 if "int64" in extra_type else np.int32 if "int32" in extra_type else np.bool_ if "bool" in extra_type else np.float32
            if "mask" in extra_name:
                feeds[extra.name] = np.ones((1, target_h, target_w), dtype=dtype)
            elif any(token in extra_name for token in ("orig_target_size", "image_size", "im_shape", "target_size")):
                feeds[extra.name] = np.asarray([[h0, w0]], dtype=dtype)
            elif "scale_factor" in extra_name or "scale_factors" in extra_name:
                feeds[extra.name] = np.asarray([[target_h / h0, target_w / w0]], dtype=dtype)
            else:
                raise AdapterError(
                    f"ONNX model requires additional input '{extra.name}' ({extra.type}, {extra.shape}). "
                    "This generic adapter cannot infer its value."
                )
        return feeds, (target_w, target_h), (w0, h0)

    def infer(self, frame: np.ndarray, settings: dict[str, Any]) -> InferenceResult:
        t0 = time.perf_counter()
        feeds, input_size, original_size = self._prepare(frame, settings)
        t1 = time.perf_counter()
        try:
            raw = self.model.run(None, feeds)
        except Exception as exc:
            raise AdapterError(f"ONNX inference failed: {exc}") from exc
        t2 = time.perf_counter()
        out = self._postprocess(raw, input_size, original_size, settings)
        t3 = time.perf_counter()
        out.adapter = self.display_name
        out.device = self.device
        out.preprocess_ms = (t1 - t0) * 1000
        out.inference_ms = (t2 - t1) * 1000
        out.postprocess_ms = (t3 - t2) * 1000
        out.total_ms = (t3 - t0) * 1000
        return out

    def _postprocess(self, outputs, input_size, original_size, settings) -> InferenceResult:
        out = InferenceResult(task="generic")
        shapes = [tuple(np.asarray(x).shape) for x in outputs]
        out.metadata["output_shapes"] = shapes
        if not outputs:
            out.summary = "No ONNX outputs"
            return out

        if self._parse_named_outputs(outputs, out, original_size, settings):
            return out

        arr = np.asarray(outputs[0])
        squeezed = np.squeeze(arr)
        labels = self._labels(settings) or getattr(self, "model_labels", [])
        topk = max(1, int(settings.get("topk", 5)))

        # Classification: [C] or [1,C]. Avoid treating large detector vectors as classes.
        if squeezed.ndim == 1 and 1 < squeezed.size <= 10000:
            scores = squeezed.astype(np.float32)
            if scores.min() < 0 or scores.max() > 1.0 or not math.isclose(float(scores.sum()), 1.0, rel_tol=0.05, abs_tol=0.05):
                scores = _softmax(scores)
            idxs = np.argsort(scores)[::-1][:topk]
            for idx in idxs:
                label = labels[idx] if idx < len(labels) else str(idx)
                out.classifications.append(Classification(label, float(scores[idx]), int(idx)))
            out.task = "classify"
            out.summary = " • ".join(f"{c.label} {c.score*100:.1f}%" for c in out.classifications[:3])
            return out

        det = squeezed
        if det.ndim == 2:
            # Typical Ultralytics raw output is [4+C, N], transpose to [N, 4+C].
            if det.shape[0] < det.shape[1] and det.shape[0] <= 512 and det.shape[1] >= 100:
                det = det.T

            # Already postprocessed [N,6]: x1,y1,x2,y2,conf,cls.
            if 6 <= det.shape[1] <= 8 and det.shape[0] <= 10000:
                self._parse_nx6(det, out, input_size, original_size, labels, settings)
                if out.detections:
                    out.task = "detect"
                    out.summary = f"{len(out.detections)} detections"
                    return out

            # A YOLO segmentation export adds mask coefficients after the class
            # scores and returns a prototype tensor such as [1, 32, 160, 160].
            proto = self._find_mask_prototypes(outputs[1:])
            mask_dim = int(proto.shape[0]) if proto is not None else 0

            # YOLOv8/11 style raw [N, 4 + classes (+ masks)] (or YOLOv5:
            # 5 + classes + masks when labels reveal the objectness column).
            if 5 <= det.shape[1] <= 512 and det.shape[0] >= 10:
                if self._parse_yolo_raw(
                    det, out, input_size, original_size, labels, settings,
                    mask_dim=mask_dim, proto=proto,
                ):
                    out.task = "segment" if proto is not None else "detect"
                    kind = "segments" if proto is not None else "detections"
                    out.summary = f"{len(out.detections)} {kind} (generic YOLO parser)"
                    return out

        out.summary = "Raw ONNX output: " + ", ".join(str(s) for s in shapes)
        return out

    def _parse_named_outputs(self, outputs, out, original_size, settings) -> bool:
        """Parse common TensorFlow/SSD, DETR, and end-to-end detector outputs."""
        named = {
            str(name).lower(): np.asarray(value)
            for name, value in zip(getattr(self, "output_names", []), outputs)
        }
        if not named:
            return False
        box_item = next(((k, v) for k, v in named.items() if "box" in k and v.shape[-1:] == (4,)), None)
        logits_item = next(((k, v) for k, v in named.items() if "logit" in k and v.ndim >= 2), None)

        # Hugging Face/DETR exports: normalized cx,cy,w,h boxes plus class logits.
        if box_item and logits_item:
            boxes = np.asarray(box_item[1]).reshape(-1, 4).astype(np.float32)
            logits = np.asarray(logits_item[1]).reshape(-1, logits_item[1].shape[-1]).astype(np.float32)
            probs = np.exp(logits - np.max(logits, axis=1, keepdims=True))
            probs /= np.maximum(probs.sum(axis=1, keepdims=True), 1e-12)
            class_probs = probs[:, :-1] if probs.shape[1] > 1 else probs
            classes = np.argmax(class_probs, axis=1)
            scores = class_probs[np.arange(len(class_probs)), classes]
            self._append_named_boxes(boxes, scores, classes, out, original_size, settings, "cxcywh")
            out.task = "detect"
            out.summary = f"{len(out.detections)} detections (DETR ONNX)"
            return True

        score_item = next(((k, v) for k, v in named.items() if "score" in k or "confidence" in k), None)
        if not box_item or not score_item:
            return False
        boxes = np.asarray(box_item[1]).reshape(-1, 4).astype(np.float32)
        score_values = np.asarray(score_item[1])
        class_item = next(((k, v) for k, v in named.items() if "class" in k or "label" in k), None)
        if score_values.ndim >= 3 or (score_values.ndim == 2 and score_values.shape[0] != 1):
            matrix = score_values.reshape(-1, score_values.shape[-1])
            classes = np.argmax(matrix, axis=1)
            scores = matrix[np.arange(len(matrix)), classes]
        else:
            scores = score_values.reshape(-1)
            classes = np.asarray(class_item[1]).reshape(-1).astype(int) if class_item else np.zeros(len(scores), dtype=int)
        count_item = next((v for k, v in named.items() if "num" in k and ("detect" in k or "box" in k)), None)
        if count_item is not None:
            count = max(0, int(np.asarray(count_item).reshape(-1)[0]))
            boxes, scores, classes = boxes[:count], scores[:count], classes[:count]
        layout = "yxyx" if "detection_box" in box_item[0] else "xyxy"
        self._append_named_boxes(boxes, scores, classes, out, original_size, settings, layout)
        out.task = "detect"
        out.summary = f"{len(out.detections)} detections (named ONNX outputs)"
        return True

    def _append_named_boxes(self, boxes, scores, classes, out, original_size, settings, layout):
        width, height = original_size
        threshold = float(settings.get("conf", 0.25))
        labels = self._labels(settings) or getattr(self, "model_labels", [])
        for box, score, class_id in zip(boxes, scores, classes):
            if float(score) < threshold:
                continue
            a, b, c, d = map(float, box)
            if layout == "cxcywh":
                x1, y1, x2, y2 = a - c / 2, b - d / 2, a + c / 2, b + d / 2
            elif layout == "yxyx":
                y1, x1, y2, x2 = a, b, c, d
            else:
                x1, y1, x2, y2 = a, b, c, d
            if np.max(np.abs(box)) <= 2.0:
                x1, x2, y1, y2 = x1 * width, x2 * width, y1 * height, y2 * height
            cid = int(class_id)
            label = labels[cid] if 0 <= cid < len(labels) else str(cid)
            out.detections.append(Detection(x1, y1, x2, y2, float(score), cid, label))

    def _parse_nx6(self, det, out, input_size, original_size, labels, settings):
        conf_threshold = float(settings.get("conf", 0.25))
        iw, ih = input_size
        ow, oh = original_size
        sx, sy = ow / iw, oh / ih
        for row in det:
            x1, y1, x2, y2, score, cls = row[:6]
            if float(score) < conf_threshold:
                continue
            cid = int(cls)
            label = labels[cid] if cid < len(labels) else str(cid)
            out.detections.append(Detection(float(x1*sx), float(y1*sy), float(x2*sx), float(y2*sy), float(score), cid, label))

    @staticmethod
    def _find_mask_prototypes(outputs) -> np.ndarray | None:
        for value in outputs:
            proto = np.squeeze(np.asarray(value))
            if proto.ndim != 3:
                continue
            # Standard Ultralytics layout is [mask_dim, height, width].
            if proto.shape[0] <= 256 and proto.shape[1] > 4 and proto.shape[2] > 4:
                return proto.astype(np.float32)
            # Also accept an NHWC prototype tensor from custom exports.
            if proto.shape[2] <= 256 and proto.shape[0] > 4 and proto.shape[1] > 4:
                return proto.transpose(2, 0, 1).astype(np.float32)
        return None

    def _parse_yolo_raw(
        self, det, out, input_size, original_size, labels, settings,
        mask_dim: int = 0, proto: np.ndarray | None = None,
    ) -> bool:
        conf_threshold = float(settings.get("conf", 0.25))
        iou_threshold = float(settings.get("iou", 0.45))
        cols = det.shape[1]
        prediction_cols = cols - mask_dim
        if prediction_cols <= 4:
            return False
        if labels and prediction_cols == len(labels) + 5:
            objectness = det[:, 4]
            class_scores = det[:, 5:prediction_cols]
        else:
            objectness = np.ones(det.shape[0], dtype=np.float32)
            class_scores = det[:, 4:prediction_cols]
        if class_scores.size == 0:
            return False
        class_ids = np.argmax(class_scores, axis=1)
        class_conf = class_scores[np.arange(det.shape[0]), class_ids]
        scores = objectness * class_conf
        keep = scores >= conf_threshold
        if not np.any(keep):
            return True
        rows = det[keep]
        scores = scores[keep]
        class_ids = class_ids[keep]
        boxes_xywh = rows[:, :4].astype(np.float32)
        iw, ih = input_size
        ow, oh = original_size
        # Heuristic: normalized coordinates.
        if np.nanmax(np.abs(boxes_xywh)) <= 2.0:
            boxes_xywh[:, [0, 2]] *= iw
            boxes_xywh[:, [1, 3]] *= ih
        x = boxes_xywh[:, 0] - boxes_xywh[:, 2] / 2
        y = boxes_xywh[:, 1] - boxes_xywh[:, 3] / 2
        cv_boxes = [[float(a), float(b), float(c), float(d)] for a, b, c, d in zip(x, y, boxes_xywh[:, 2], boxes_xywh[:, 3])]
        indices = cv2.dnn.NMSBoxes(cv_boxes, scores.astype(float).tolist(), conf_threshold, iou_threshold)
        if len(indices) == 0:
            return True
        sx, sy = ow / iw, oh / ih
        for idx in np.asarray(indices).reshape(-1):
            bx, by, bw, bh = cv_boxes[int(idx)]
            cid = int(class_ids[int(idx)])
            label = labels[cid] if cid < len(labels) else str(cid)
            out.detections.append(Detection(bx*sx, by*sy, (bx+bw)*sx, (by+bh)*sy,
                                            float(scores[int(idx)]), cid, label))
            if proto is not None and mask_dim > 0:
                coeffs = rows[int(idx), prediction_cols:prediction_cols + mask_dim]
                polygon = self._decode_yolo_mask(coeffs, proto, (bx, by, bw, bh), input_size, original_size)
                if polygon:
                    out.masks.append(PolygonMask(polygon, float(scores[int(idx)]), cid, label))
        return True

    @staticmethod
    def _decode_yolo_mask(coeffs, proto, box, input_size, original_size):
        mask_h, mask_w = proto.shape[1:]
        logits = np.asarray(coeffs, dtype=np.float32) @ proto.reshape(proto.shape[0], -1)
        mask = (1.0 / (1.0 + np.exp(-np.clip(logits, -30.0, 30.0)))).reshape(mask_h, mask_w)

        input_w, input_h = input_size
        bx, by, bw, bh = box
        px1 = max(0, min(mask_w, int(math.floor(bx * mask_w / input_w))))
        py1 = max(0, min(mask_h, int(math.floor(by * mask_h / input_h))))
        px2 = max(0, min(mask_w, int(math.ceil((bx + bw) * mask_w / input_w))))
        py2 = max(0, min(mask_h, int(math.ceil((by + bh) * mask_h / input_h))))
        cropped = np.zeros_like(mask, dtype=np.uint8)
        if px2 <= px1 or py2 <= py1:
            return []
        cropped[py1:py2, px1:px2] = (mask[py1:py2, px1:px2] >= 0.5).astype(np.uint8)

        original_w, original_h = original_size
        full_mask = cv2.resize(cropped, (original_w, original_h), interpolation=cv2.INTER_NEAREST)
        contours, _ = cv2.findContours(full_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return []
        contour = max(contours, key=cv2.contourArea)
        if cv2.contourArea(contour) < 1.0:
            return []
        epsilon = max(0.5, 0.001 * cv2.arcLength(contour, True))
        contour = cv2.approxPolyDP(contour, epsilon, True).reshape(-1, 2)
        return [(float(x), float(y)) for x, y in contour]
