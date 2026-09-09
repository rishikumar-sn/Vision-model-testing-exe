from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class DeviceStatus:
    torch_available: bool = False
    cuda_available: bool = False
    cuda_devices: list[str] = field(default_factory=list)
    onnx_available: bool = False
    onnx_providers: list[str] = field(default_factory=list)

    @property
    def preferred_torch_device(self) -> str:
        return "cuda:0" if self.cuda_available else "cpu"

    @property
    def summary(self) -> str:
        if self.cuda_available and self.cuda_devices:
            return f"CUDA • {self.cuda_devices[0]}"
        if "CUDAExecutionProvider" in self.onnx_providers:
            return "ONNX CUDA"
        return "CPU"


def detect_devices() -> DeviceStatus:
    status = DeviceStatus()
    try:
        import torch
        status.torch_available = True
        status.cuda_available = bool(torch.cuda.is_available())
        if status.cuda_available:
            status.cuda_devices = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
    except Exception:
        pass

    try:
        import onnxruntime as ort
        status.onnx_available = True
        status.onnx_providers = list(ort.get_available_providers())
    except Exception:
        pass
    return status


def resolve_device(requested: str, status: DeviceStatus, backend: str = "torch") -> str:
    requested = (requested or "auto").lower()
    if backend == "onnx":
        if requested.startswith("cpu"):
            return "CPUExecutionProvider"
        if "CUDAExecutionProvider" in status.onnx_providers:
            return "CUDAExecutionProvider"
        return "CPUExecutionProvider"
    if requested.startswith("cpu"):
        return "cpu"
    if requested.startswith("cuda") and status.cuda_available:
        return requested
    return status.preferred_torch_device
