from __future__ import annotations

import time
from typing import Any

import numpy as np

from adapters.base import AdapterError, BaseAdapter
from core.device import resolve_device
from core.results import Detection, InferenceResult, ModelInfo, PolygonMask


class FastSAMAdapter(BaseAdapter):
    display_name = "FastSAM"

    def load(self, source: str, settings: dict[str, Any]) -> ModelInfo:
        try:
            from ultralytics import FastSAM
        except Exception as exc:
            raise AdapterError("FastSAM requires Ultralytics. Run: pip install ultralytics") from exc
        try:
            self.model = FastSAM(source)
        except Exception as exc:
            raise AdapterError(f"FastSAM could not load this checkpoint: {exc}") from exc
        self.source = source
        self.device = resolve_device(settings.get("device", "auto"), self.device_status, "torch")
        self.task = "segment"
        return ModelInfo(source, self.display_name, self.task, self.device,
                         f"imgsz={settings.get('imgsz', 1024)}", "Masks / prompt-filtered masks")

    def infer(self, frame: np.ndarray, settings: dict[str, Any]) -> InferenceResult:
        kwargs: dict[str, Any] = {
            "device": self.device,
            "retina_masks": bool(settings.get("retina_masks", True)),
            "imgsz": int(settings.get("imgsz", 1024)),
            "conf": float(settings.get("conf", 0.4)),
            "iou": float(settings.get("iou", 0.9)),
            "verbose": False,
        }
        mode = settings.get("fastsam_prompt", "everything")
        if mode == "roi_box" and settings.get("prompt_box"):
            kwargs["bboxes"] = list(settings["prompt_box"])
        elif mode == "center_point" and settings.get("prompt_point"):
            kwargs["points"] = [list(settings["prompt_point"])]
            kwargs["labels"] = [1]
        elif mode == "text":
            text = str(settings.get("fastsam_text", "")).strip()
            if text:
                kwargs["texts"] = text

        t0 = time.perf_counter()
        try:
            results = self.model(frame, **kwargs)
        except Exception as exc:
            raise AdapterError(f"FastSAM inference failed: {exc}") from exc
        t1 = time.perf_counter()
        out = InferenceResult(adapter=self.display_name, task="segment", device=self.device,
                              inference_ms=(t1 - t0) * 1000, total_ms=(t1 - t0) * 1000)
        if not results:
            return out
        r = results[0]
        names = getattr(r, "names", {}) or {}
        boxes = getattr(r, "boxes", None)
        if boxes is not None and getattr(boxes, "xyxy", None) is not None:
            xyxy = boxes.xyxy.detach().cpu().numpy()
            confs = boxes.conf.detach().cpu().numpy() if boxes.conf is not None else np.ones(len(xyxy))
            classes = boxes.cls.detach().cpu().numpy().astype(int) if boxes.cls is not None else np.full(len(xyxy), -1)
            for coords, score, class_id in zip(xyxy, confs, classes):
                label = str(names.get(int(class_id), "segment"))
                out.detections.append(Detection(*map(float, coords[:4]), float(score), int(class_id), label))
        masks = getattr(r, "masks", None)
        if masks is not None and getattr(masks, "xy", None) is not None:
            for idx, poly in enumerate(masks.xy):
                pts = [(float(x), float(y)) for x, y in np.asarray(poly).reshape(-1, 2)]
                score = out.detections[idx].confidence if idx < len(out.detections) else 1.0
                out.masks.append(PolygonMask(pts, score, -1, "segment"))
        speed = getattr(r, "speed", {}) or {}
        out.preprocess_ms = float(speed.get("preprocess", 0) or 0)
        out.inference_ms = float(speed.get("inference", out.inference_ms) or out.inference_ms)
        out.postprocess_ms = float(speed.get("postprocess", 0) or 0)
        out.total_ms = out.preprocess_ms + out.inference_ms + out.postprocess_ms
        out.summary = f"{len(out.masks)} masks"
        return out
