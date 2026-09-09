from __future__ import annotations

import time
from typing import Any

import cv2
import numpy as np
from PIL import Image

from adapters.base import AdapterError, BaseAdapter
from core.device import resolve_device
from core.results import Classification, InferenceResult, ModelInfo


class SigLIPAdapter(BaseAdapter):
    display_name = "SigLIP / Transformers"

    def load(self, source: str, settings: dict[str, Any]) -> ModelInfo:
        try:
            import torch
            from transformers import AutoModel, AutoProcessor
        except Exception as exc:
            raise AdapterError("SigLIP requires torch + transformers. Run: pip install torch transformers pillow") from exc
        self.torch = torch
        self.device = resolve_device(settings.get("device", "auto"), self.device_status, "torch")
        try:
            self.processor = AutoProcessor.from_pretrained(source)
            self.model = AutoModel.from_pretrained(source).to(self.device).eval()
        except Exception as exc:
            raise AdapterError(
                "Could not load the Hugging Face model/directory. Enter a local model directory or model ID, "
                f"for example google/siglip-base-patch16-224. Error: {exc}"
            ) from exc
        self.source = source
        self.task = "embedding"
        projection = getattr(getattr(self.model, "config", None), "projection_dim", None)
        return ModelInfo(source, self.display_name, self.task, self.device,
                         "Processor-defined RGB image", f"Image/text embeddings{f' • dim={projection}' if projection else ''}")

    def infer(self, frame: np.ndarray, settings: dict[str, Any]) -> InferenceResult:
        prompts = [x.strip() for x in str(settings.get("text_prompts", "")).splitlines() if x.strip()]
        if not prompts:
            prompts = ["object"]
        torch = self.torch
        pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        t0 = time.perf_counter()
        image_inputs = self.processor(images=pil, return_tensors="pt")
        text_inputs = self.processor(text=prompts, padding="max_length", return_tensors="pt")
        image_inputs = {k: v.to(self.device) for k, v in image_inputs.items() if hasattr(v, "to")}
        text_inputs = {k: v.to(self.device) for k, v in text_inputs.items() if hasattr(v, "to")}
        t1 = time.perf_counter()
        with torch.no_grad():
            image_features = self.model.get_image_features(**image_inputs)
            text_features = self.model.get_text_features(**text_inputs)
            if hasattr(image_features, "pooler_output"):
                image_features = image_features.pooler_output
            elif hasattr(image_features, "image_embeds"):
                image_features = image_features.image_embeds
            if hasattr(text_features, "pooler_output"):
                text_features = text_features.pooler_output
            elif hasattr(text_features, "text_embeds"):
                text_features = text_features.text_embeds
            image_features = image_features / image_features.norm(dim=-1, keepdim=True).clamp_min(1e-12)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True).clamp_min(1e-12)
            similarities = (image_features @ text_features.T).squeeze(0)
        t2 = time.perf_counter()
        sims = similarities.detach().float().cpu().numpy().reshape(-1)
        # Softmax is displayed as a convenient relative ranking; raw cosine values are also stored.
        e = np.exp(sims - np.max(sims))
        probs = e / max(float(e.sum()), 1e-12)
        order = np.argsort(probs)[::-1][:max(1, int(settings.get("topk", 5)))]
        out = InferenceResult(adapter=self.display_name, task="embedding", device=self.device)
        for idx in order:
            out.classifications.append(Classification(prompts[idx], float(probs[idx]), int(idx)))
        out.metadata["cosine_similarity"] = {prompts[i]: float(sims[i]) for i in range(len(prompts))}
        t3 = time.perf_counter()
        out.preprocess_ms = (t1 - t0) * 1000
        out.inference_ms = (t2 - t1) * 1000
        out.postprocess_ms = (t3 - t2) * 1000
        out.total_ms = (t3 - t0) * 1000
        out.summary = " • ".join(f"{c.label} {c.score*100:.1f}%" for c in out.classifications[:3])
        return out
