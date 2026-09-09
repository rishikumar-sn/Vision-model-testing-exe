from __future__ import annotations

import time
from typing import Any

import cv2
import numpy as np

from adapters.base import AdapterError, BaseAdapter
from core.device import resolve_device
from core.results import Classification, InferenceResult, ModelInfo


class TorchClassifierAdapter(BaseAdapter):
    display_name = "PyTorch Classifier"

    def load(self, source: str, settings: dict[str, Any]) -> ModelInfo:
        try:
            import torch
        except Exception as exc:
            raise AdapterError("PyTorch is not installed. Install a CUDA or CPU build of torch first.") from exc
        self.torch = torch
        self.device = resolve_device(settings.get("device", "auto"), self.device_status, "torch")
        self.source = source
        model = None
        errors = []
        try:
            model = torch.jit.load(source, map_location=self.device)
        except Exception as exc:
            errors.append(f"TorchScript: {exc}")
        if model is None:
            try:
                try:
                    payload = torch.load(source, map_location=self.device, weights_only=False)
                except TypeError:
                    payload = torch.load(source, map_location=self.device)
                if isinstance(payload, torch.nn.Module):
                    model = payload
                else:
                    model = self._build_from_state_dict(payload, settings)
            except Exception as exc:
                errors.append(f"Checkpoint: {exc}")
        if model is None:
            raise AdapterError("Could not load PyTorch classifier. " + " | ".join(errors))
        self.model = model.to(self.device).eval()
        self.task = "classify"
        input_size = int(settings.get("classifier_size", 224))
        labels = self._labels(settings)
        return ModelInfo(source, self.display_name, self.task, self.device,
                         f"RGB {input_size}×{input_size}", f"Classification logits • labels={len(labels) or 'unknown'}",
                         {"architecture": settings.get("architecture", "auto")})

    def _build_from_state_dict(self, payload, settings):
        import torch
        try:
            import torchvision.models as models
        except Exception as exc:
            raise AdapterError("Loading state_dict classifiers requires torchvision.") from exc

        state = payload
        if isinstance(payload, dict):
            for key in ("state_dict", "model_state_dict", "model"):
                if key in payload and isinstance(payload[key], dict):
                    state = payload[key]
                    break
        if not isinstance(state, dict):
            raise AdapterError("Checkpoint is not a full nn.Module/TorchScript and no state_dict was found.")
        # Strip common Lightning/DataParallel prefixes.
        cleaned = {}
        for key, value in state.items():
            new_key = key
            for prefix in ("model.", "module."):
                if new_key.startswith(prefix):
                    new_key = new_key[len(prefix):]
            cleaned[new_key] = value

        labels = self._labels(settings)
        num_classes = int(settings.get("num_classes") or len(labels) or 1000)
        arch = str(settings.get("architecture", "resnet50")).lower()
        if arch == "resnet18":
            model = models.resnet18(weights=None)
            model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
        elif arch == "mobilenet_v3_large":
            model = models.mobilenet_v3_large(weights=None)
            model.classifier[-1] = torch.nn.Linear(model.classifier[-1].in_features, num_classes)
        elif arch == "efficientnet_b0":
            model = models.efficientnet_b0(weights=None)
            model.classifier[-1] = torch.nn.Linear(model.classifier[-1].in_features, num_classes)
        else:
            model = models.resnet50(weights=None)
            model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
        missing, unexpected = model.load_state_dict(cleaned, strict=False)
        if len(unexpected) > 30:
            raise AdapterError(f"Selected architecture '{arch}' does not match checkpoint (many unexpected keys).")
        return model

    def infer(self, frame: np.ndarray, settings: dict[str, Any]) -> InferenceResult:
        torch = self.torch
        size = int(settings.get("classifier_size", 224))
        t0 = time.perf_counter()
        rgb = cv2.cvtColor(cv2.resize(frame, (size, size)), cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        if bool(settings.get("imagenet_normalize", True)):
            mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
            std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
            rgb = (rgb - mean) / std
        tensor = torch.from_numpy(rgb.transpose(2, 0, 1)).unsqueeze(0).to(self.device)
        t1 = time.perf_counter()
        with torch.no_grad():
            output = self.model(tensor)
        t2 = time.perf_counter()
        if isinstance(output, (list, tuple)):
            output = output[0]
        if isinstance(output, dict):
            output = next(iter(output.values()))
        logits = output.detach().float().cpu().numpy().reshape(-1)
        exp = np.exp(logits - np.max(logits))
        probs = exp / max(float(exp.sum()), 1e-12)
        labels = self._labels(settings)
        topk = max(1, int(settings.get("topk", 5)))
        idxs = np.argsort(probs)[::-1][:topk]
        out = InferenceResult(adapter=self.display_name, task="classify", device=self.device)
        for idx in idxs:
            label = labels[idx] if idx < len(labels) else str(idx)
            out.classifications.append(Classification(label, float(probs[idx]), int(idx)))
        t3 = time.perf_counter()
        out.preprocess_ms = (t1 - t0) * 1000
        out.inference_ms = (t2 - t1) * 1000
        out.postprocess_ms = (t3 - t2) * 1000
        out.total_ms = (t3 - t0) * 1000
        out.summary = " • ".join(f"{c.label} {c.score*100:.1f}%" for c in out.classifications[:3])
        return out
