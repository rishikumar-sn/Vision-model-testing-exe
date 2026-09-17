# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller build for the full, optional-backend desktop application.

Build as onedir because PyTorch/TensorFlow/RF-DETR contain many native libraries.
One-file extraction is slow, harder to diagnose, and can cause DLL-loading failures.
"""

import importlib.util
import sys
from pathlib import Path

from PyInstaller.utils.hooks import collect_all, copy_metadata

# TensorFlow + PyTorch + Transformers create a dependency graph deeper than
# Python's default recursion limit during PyInstaller analysis.
sys.setrecursionlimit(max(5000, sys.getrecursionlimit() * 5))

datas = [("applogo.png", "."), ("VERSION", ".")]
binaries = []
# optree's extension depends on a vendored C++ runtime DLL stored beside the
# package, outside the package directory. PyInstaller misses it on Windows.
optree_spec = importlib.util.find_spec("optree")
if optree_spec is not None and optree_spec.origin:
    optree_libs = Path(optree_spec.origin).resolve().parent.parent / "optree.libs"
    if optree_libs.is_dir():
        binaries += [(str(dll), "optree.libs") for dll in optree_libs.glob("*.dll")]
hiddenimports = [
    "torch",
    "torchvision",
    "torchvision.ops",
    "ultralytics",
    "onnxruntime",
    "rfdetr",
    "tensorflow",
    "keras",
    "transformers",
    "transformers.models.siglip.configuration_siglip",
    "transformers.models.siglip.image_processing_siglip",
    "transformers.models.siglip.image_processing_pil_siglip",
    "transformers.models.siglip.modeling_siglip",
    "transformers.models.siglip.processing_siglip",
    "transformers.models.siglip.tokenization_siglip",
    "pygrabber.dshow_graph",
]


def installed(module):
    try:
        return importlib.util.find_spec(module) is not None
    except (ImportError, ModuleNotFoundError, ValueError):
        return False


# RF-DETR dynamically selects architectures and ships non-Python resources that
# static analysis cannot discover. TensorFlow and ONNX Runtime have maintained
# PyInstaller hooks, so merely including their imports activates those hooks.
if installed("rfdetr"):
    package_datas, package_binaries, package_hidden = collect_all(
        "rfdetr",
        filter_submodules=lambda name: not (
            name.startswith("rfdetr.training")
            or name.startswith("rfdetr.callbacks")
        ),
    )
    datas += package_datas
    binaries += package_binaries
    hiddenimports += package_hidden

# Several ML libraries query their wheel metadata at runtime. Missing metadata
# is a common source-tree-works / packaged-exe-fails difference.
for distribution in (
    "torch", "torchvision", "ultralytics", "onnx", "onnxruntime", "onnxruntime-gpu", "rfdetr",
    "tensorflow", "keras", "transformers", "timm", "supervision",
    "huggingface-hub", "safetensors",
):
    try:
        datas += copy_metadata(distribution)
    except Exception:
        pass


a = Analysis(
    ["main.py"],
    pathex=[],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    # Optional dataframe/remote-filesystem stacks are not used by inference.
    # In this environment PyArrow's collect-all hook causes Python's Windows
    # native stack to overflow while traversing the combined TF + Torch graph.
    excludes=[
        "pyarrow",
        "fsspec",
        "dask",
        "distributed",
        "pytorch_lightning",
        "tensorboard",
        "openvino",
    ],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="UniversalVisionModelTester",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,
    disable_windowed_traceback=False,
    icon="applogo.ico",
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    name="UniversalVisionModelTester",
)
