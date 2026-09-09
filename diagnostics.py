#!/usr/bin/env python3
from __future__ import annotations

import importlib
import importlib.metadata
import platform
import sys


def check(name: str, import_name: str | None = None):
    module_name = import_name or name
    try:
        mod = importlib.import_module(module_name)
        version = getattr(mod, "__version__", "installed")
        print(f"[OK] {name}: {version}")
        return mod
    except Exception as exc:
        print(f"[--] {name}: not available ({exc})")
        return None


print("Universal Vision Model Tester - Environment Diagnostics")
print("=" * 62)
print("Python:", sys.version.replace("\n", " "))
print("OS:", platform.platform())
print()

check("NumPy", "numpy")
check("OpenCV", "cv2")
pyqt = check("PyQt6", "PyQt6.QtCore")
check("Pillow", "PIL")
print()

torch = check("PyTorch", "torch")
if torch is not None:
    try:
        print("     CUDA available:", torch.cuda.is_available())
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                print(f"     CUDA:{i}: {torch.cuda.get_device_name(i)}")
    except Exception as exc:
        print("     CUDA check failed:", exc)

ort = check("ONNX Runtime", "onnxruntime")
if ort is not None:
    try:
        print("     Providers:", ort.get_available_providers())
    except Exception as exc:
        print("     Provider check failed:", exc)
ort_distributions = []
for distribution in ("onnxruntime", "onnxruntime-gpu"):
    try:
        importlib.metadata.version(distribution)
        ort_distributions.append(distribution)
    except importlib.metadata.PackageNotFoundError:
        pass
if len(ort_distributions) > 1:
    print("     [WARN] Both onnxruntime and onnxruntime-gpu are installed; keep only one.")

check("Ultralytics", "ultralytics")
check("RF-DETR", "rfdetr")
check("Transformers", "transformers")
check("TensorFlow", "tensorflow")
print()
print("Core GUI requirement: PyQt6 + OpenCV + NumPy + Pillow")
print("Optional model backends are loaded only when their adapter is used.")
