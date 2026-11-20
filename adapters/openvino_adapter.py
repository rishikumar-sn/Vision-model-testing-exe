from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from adapters.base import AdapterError
from adapters.ultralytics_adapter import UltralyticsAdapter
from core.results import ModelInfo


class OpenVINOAdapter(UltralyticsAdapter):
    """Run Ultralytics OpenVINO IR exports, including INT8, on the CPU."""

    display_name = "OpenVINO (INT8 / FP16 / FP32)"

    @staticmethod
    def model_file(source: str) -> Path:
        path = Path(source)
        if path.is_dir():
            models = sorted(path.glob("*.xml"))
            if len(models) != 1:
                raise AdapterError("Choose a directory containing exactly one OpenVINO .xml model, or select the .xml file directly.")
            path = models[0]
        if path.suffix.lower() != ".xml" or not path.is_file():
            raise AdapterError("Select an OpenVINO .xml model or its export directory.")
        if not path.with_suffix(".bin").is_file():
            raise AdapterError(f"OpenVINO weights are missing: {path.with_suffix('.bin')}. Keep the .xml and .bin together.")
        if not (path.parent / "metadata.yaml").is_file():
            raise AdapterError("This adapter requires an Ultralytics export with metadata.yaml beside the .xml and .bin files.")
        if "openvino_model" not in path.parent.name:
            raise AdapterError("Keep the Ultralytics export in a folder ending in _openvino_model so its runtime can identify the format.")
        if len(list(path.parent.glob("*.xml"))) != 1:
            raise AdapterError("Keep exactly one .xml model in the OpenVINO export directory.")
        return path

    def load(self, source: str, settings: dict[str, Any]) -> ModelInfo:
        path = self.model_file(source)
        try:
            import openvino as ov
        except Exception as exc:
            raise AdapterError("OpenVINO Runtime is unavailable. Install it before building: python -m pip install \"openvino>=2026.4.1\"") from exc
        try:
            from ultralytics import YOLO
            from ultralytics.utils import YAML

            metadata = YAML.load(path.parent / "metadata.yaml")
            task = metadata.get("task")
            if task not in {"detect", "segment", "classify", "pose", "obb"}:
                raise AdapterError(f"Unsupported or missing Ultralytics task in metadata.yaml: {task}")
            graph = ov.Core().read_model(str(path))
            if len(graph.inputs) != 1:
                raise AdapterError("OpenVINO YOLO testing requires a single image input.")
            shape = graph.input().partial_shape
            if not shape.is_static:
                raise AdapterError("Export OpenVINO with dynamic=False for this adapter.")
            dimensions = list(graph.input().shape)
            if len(dimensions) != 4 or dimensions[0] != 1 or dimensions[1] != 3:
                raise AdapterError(f"OpenVINO YOLO testing requires a [1, 3, height, width] image input, got {dimensions}.")
            self.fixed_input_size = (int(dimensions[2]), int(dimensions[3]))
            quantized = any(op.get_type_name() == "FakeQuantize" for op in graph.get_ops())
            input_description = str(graph.input().partial_shape)
            output_description = "; ".join(str(port.partial_shape) for port in graph.outputs)
            del graph

            self.source = source
            # Explicit Intel CPU selection also prevents a saved CUDA setting
            # from selecting an incompatible backend or an Intel GPU via AUTO.
            self.device = "intel:cpu"
            self.task = task
            self.model = YOLO(str(path.parent), task=task)
            effective_imgsz = self._predict_imgsz(self.fixed_input_size, 640)
            # Ultralytics initializes exported runtimes lazily. Warm up now so
            # missing DLLs or incompatible IRs surface as model-load errors.
            self.model.predict(
                source=np.zeros((*self.fixed_input_size, 3), dtype=np.uint8),
                imgsz=effective_imgsz, device=self.device, verbose=False,
            )
        except AdapterError:
            raise
        except Exception as exc:
            raise AdapterError(f"OpenVINO could not load this model: {exc}") from exc

        precision = "INT8 (quantized graph)" if quantized else "floating point"
        return ModelInfo(
            source=source, adapter=self.display_name, task=self.task, device="OpenVINO CPU",
            input_description=f"{input_description} (fixed OpenVINO input)",
            output_description=f"{output_description} | {precision}",
            details={
                "names": metadata.get("names", {}),
                "fixed_input_size": self.fixed_input_size,
                "effective_imgsz": effective_imgsz,
                "precision": precision,
                "runtime_version": ov.__version__,
            },
        )

    def infer(self, frame: np.ndarray, settings: dict[str, Any]):
        result = super().infer(frame, settings)
        result.device = "OpenVINO CPU"
        return result
