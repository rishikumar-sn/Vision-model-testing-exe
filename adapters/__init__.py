from __future__ import annotations

from pathlib import Path

from adapters.fastsam_adapter import FastSAMAdapter
from adapters.keras_adapter import KerasAdapter
from adapters.onnx_adapter import ONNXAdapter
from adapters.rfdetr_adapter import RFDETRAdapter
from adapters.siglip_adapter import SigLIPAdapter
from adapters.torch_classifier_adapter import TorchClassifierAdapter
from adapters.ultralytics_adapter import UltralyticsAdapter


ADAPTER_NAMES = [
    "Auto",
    UltralyticsAdapter.display_name,
    ONNXAdapter.display_name,
    RFDETRAdapter.display_name,
    TorchClassifierAdapter.display_name,
    KerasAdapter.display_name,
    FastSAMAdapter.display_name,
    SigLIPAdapter.display_name,
]


def adapter_classes_for(source: str, requested: str):
    mapping = {
        UltralyticsAdapter.display_name: [UltralyticsAdapter],
        ONNXAdapter.display_name: [ONNXAdapter],
        RFDETRAdapter.display_name: [RFDETRAdapter],
        TorchClassifierAdapter.display_name: [TorchClassifierAdapter],
        KerasAdapter.display_name: [KerasAdapter],
        FastSAMAdapter.display_name: [FastSAMAdapter],
        SigLIPAdapter.display_name: [SigLIPAdapter],
    }
    if requested and requested != "Auto":
        return mapping.get(requested, [UltralyticsAdapter])

    lower = source.lower()
    suffix = Path(source).suffix.lower()
    name = Path(source).name.lower()
    if "fastsam" in name:
        return [FastSAMAdapter, UltralyticsAdapter, TorchClassifierAdapter]
    if suffix == ".onnx":
        # Ultralytics exported ONNX models are common; fall back to the generic runtime.
        return [UltralyticsAdapter, ONNXAdapter]
    if suffix in {".keras", ".h5", ".tflite"}:
        return [KerasAdapter]
    if Path(source).is_dir() and (Path(source) / "saved_model.pb").is_file():
        return [KerasAdapter]
    if suffix == ".pth" and ("rfdetr" in name or "rf-detr" in name or "detr" in name):
        return [RFDETRAdapter, TorchClassifierAdapter]
    if suffix in {".pt", ".pth", ".torchscript", ".jit"}:
        return [UltralyticsAdapter, TorchClassifierAdapter]
    if Path(source).is_dir() or "/" in source and not Path(source).exists():
        # Local transformer directory or Hugging Face model ID.
        return [SigLIPAdapter]
    return [UltralyticsAdapter, ONNXAdapter, TorchClassifierAdapter]
