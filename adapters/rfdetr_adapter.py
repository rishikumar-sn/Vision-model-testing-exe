from __future__ import annotations

import time
from typing import Any

import cv2
import numpy as np

from adapters.base import AdapterError, BaseAdapter
from core.device import resolve_device
from core.results import Detection, InferenceResult, ModelInfo


class RFDETRAdapter(BaseAdapter):
    display_name = "RF-DETR"

    def load(self, source: str, settings: dict[str, Any]) -> ModelInfo:
        try:
            from rfdetr import RFDETR, RFDETRBase, RFDETRLarge, RFDETRMedium, RFDETRNano, RFDETRSmall
        except Exception as exc:
            raise AdapterError("RF-DETR is not installed. Run: pip install rfdetr") from exc
        variant = str(settings.get("rfdetr_variant", "auto")).lower()
        cls_map = {
            "base": RFDETRBase, "nano": RFDETRNano, "small": RFDETRSmall,
            "medium": RFDETRMedium, "large": RFDETRLarge,
        }
        self.device = resolve_device(settings.get("device", "auto"), self.device_status, "torch")
        try:
            if variant == "auto" and hasattr(RFDETR, "from_checkpoint"):
                self.model = RFDETR.from_checkpoint(source, device=self.device)
                variant = type(self.model).__name__.removeprefix("RFDETR").lower() or "auto"
            else:
                cls = cls_map.get(variant, RFDETRSmall)
                self.model = cls(pretrain_weights=source, device=self.device)
        except Exception as exc:
            hint = " Select the exact trained variant instead of Auto." if variant == "auto" else ""
            raise AdapterError(f"RF-DETR {variant} could not load checkpoint: {exc}.{hint}") from exc
        self.source = source
        self.task = "detect"
        try:
            raw_names = self.model.class_names
            class_names = list(raw_names.values()) if isinstance(raw_names, dict) else list(raw_names)
        except Exception:
            class_names = []
        try:
            resolution = int(self.model.model_config.resolution)
        except Exception:
            resolution = 0
        return ModelInfo(source, self.display_name, self.task, self.device,
                         f"RGB {resolution}×{resolution}" if resolution else "RF-DETR native resolution",
                         f"Detections • classes={len(class_names) or 'unknown'}",
                         {"variant": variant, "class_names": class_names})

    def infer(self, frame: np.ndarray, settings: dict[str, Any]) -> InferenceResult:
        threshold = float(settings.get("conf", 0.5))
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        t0 = time.perf_counter()
        try:
            detections = self.model.predict(rgb, threshold=threshold)
        except Exception as exc:
            raise AdapterError(f"RF-DETR inference failed: {exc}") from exc
        t1 = time.perf_counter()
        out = InferenceResult(adapter=self.display_name, task="detect", device=self.device,
                              inference_ms=(t1 - t0) * 1000, total_ms=(t1 - t0) * 1000)
        xyxy = np.asarray(getattr(detections, "xyxy", []))
        confs = np.asarray(getattr(detections, "confidence", np.ones(len(xyxy))))
        classes = np.asarray(getattr(detections, "class_id", np.full(len(xyxy), -1))).astype(int)
        try:
            raw_names = self.model.class_names
            names = list(raw_names.values()) if isinstance(raw_names, dict) else list(raw_names)
        except Exception:
            names = self._labels(settings)
        for coords, score, cid in zip(xyxy, confs, classes):
            label = names[cid] if 0 <= cid < len(names) else str(cid)
            out.detections.append(Detection(*map(float, coords[:4]), float(score), int(cid), label))
        out.summary = f"{len(out.detections)} detections"
        return out
