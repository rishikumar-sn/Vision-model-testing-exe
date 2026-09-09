# Universal Vision Model Tester (PyQt6)

A modular desktop application for quickly testing vision models against a live USB/built-in camera, still images, videos, or image folders.

This V1 was built around the workflow of frequently training and converting different vision models and needing one place to inspect, load, run, compare, and benchmark them.

## Main features

- PyQt6 dark desktop UI.
- Camera discovery and maximum-resolution probing.
- Smooth camera preview in a separate capture thread.
- Windows DirectShow camera names and driver settings support.
- Anti-flicker modes reused from the reference dataset collector:
  - Adaptive
  - Normal/raw
  - Driver 50 Hz
  - Driver 60 Hz
  - Average 30
  - Average 90
  - Horizontal row correction
  - Average + row correction
- Persistent mouse-drawn ROI using normalized coordinates.
- Optional **infer only inside ROI** mode to reduce the number of pixels sent into the model.
- Detection/mask/keypoint coordinates are shifted back into full-frame coordinates after ROI inference.
- Sources:
  - Camera
  - Image
  - Video
  - Image folder with Previous/Next navigation
- Single-frame **Capture + Infer**.
- Live inference with a latest-frame policy: stale pending frames are replaced rather than queued indefinitely.
- Automatic CUDA -> CPU fallback for PyTorch backends.
- ONNX Runtime provider selection with CUDA provider -> CPU provider fallback when available.
- Per-inference timing:
  - preprocessing
  - inference
  - postprocessing
  - total latency
  - model FPS
- 20-run benchmark with warmup, average, median, min, max, P95, and FPS.
- Save the current annotated frame.
- Optional labels file (`.txt` or `.json`).
- Per-model `.infer.json` profiles saved beside local model files.
- Informative adapter errors instead of crashing the whole application.

## Supported adapters in V1

### 1. Ultralytics YOLO

Use for Ultralytics-compatible `.pt` and exported models.

Supported result types include:
- detection boxes
- segmentation polygons
- OBB polygons + boxes
- classification probabilities
- pose keypoints

### 2. ONNX Runtime

Generic ONNX support includes:
- model input/output inspection
- NCHW/NHWC image preprocessing
- CPU/CUDA Execution Provider selection
- generic classification output
- common `N x 6` postprocessed detections
- a generic YOLO-style raw detection parser
- raw output-shape reporting when an unknown output cannot safely be interpreted

For unusual custom ONNX models, the app will still run the graph and show raw output shapes even when it cannot understand the model-specific postprocessing.

### 3. RF-DETR

Supports the current RF-DETR size variants:
- nano
- small
- medium
- large

For a local fine-tuned `.pth`, choose the same RF-DETR architecture variant that was used to train the checkpoint.

### 4. PyTorch Classifier

Supports:
- TorchScript models
- fully serialized `torch.nn.Module` checkpoints
- state dictionaries for these selectable torchvision architectures:
  - ResNet-18
  - ResNet-50
  - MobileNetV3-Large
  - EfficientNet-B0

For a state dictionary, choose the correct architecture and provide class labels / number of classes.

### 5. TensorFlow / Keras

Supports `.keras`, compatible `.h5`, TensorFlow SavedModel directories, and `.tflite` files. Classification outputs and common TensorFlow Object Detection API outputs are decoded automatically.

### 6. FastSAM

Supports:
- segment everything
- ROI as a box prompt
- ROI center as a point prompt
- text prompt
- retina masks

### 7. SigLIP / Transformers

The model source box can contain either:
- a local Hugging Face model directory, or
- a Hugging Face model ID such as `google/siglip-base-patch16-224`

Enter one text prompt per line. The app computes normalized image/text features, cosine similarities, and a convenient relative softmax ranking.

## Important definition of "universal"

No application can safely infer the architecture and postprocessing rules for every arbitrary `.pt`, `.pth`, or `.onnx` file.

Examples:
- A YOLO checkpoint often contains enough metadata for its framework to rebuild the model.
- A file made only with `torch.save(model.state_dict(), ...)` contains weights but not the original Python architecture.
- A custom ONNX graph exposes tensor shapes but may not describe what those tensors semantically mean.

This application therefore uses:

1. automatic adapter detection where it is safe,
2. adapter fallback where practical,
3. model-specific controls for ambiguous cases,
4. saved `.infer.json` model profiles.

## Project structure

```text
universal_vision_model_tester/
├── main.py
├── README.md
├── requirements-core.txt
├── requirements-all.txt
├── run_windows.bat
├── run_linux.sh
│
├── adapters/
│   ├── base.py
│   ├── ultralytics_adapter.py
│   ├── onnx_adapter.py
│   ├── rfdetr_adapter.py
│   ├── torch_classifier_adapter.py
│   ├── keras_adapter.py
│   ├── fastsam_adapter.py
│   └── siglip_adapter.py
│
├── camera/
│   ├── camera_utils.py
│   ├── anti_flicker.py
│   ├── worker.py
│   └── video_worker.py
│
├── core/
│   ├── device.py
│   ├── inference_worker.py
│   ├── profile.py
│   └── results.py
│
├── processing/
│   ├── renderer.py
│   └── roi.py
│
├── ui/
│   ├── main_window.py
│   └── preview_widget.py
│
└── profiles/
```

## Installation

Python 3.10 or 3.11 is the safest general-purpose environment if you want to combine PyTorch, TensorFlow, Ultralytics, Transformers, RF-DETR and ONNX Runtime in one environment. Some individual packages may support newer Python versions, but the intersection of all optional stacks can lag.

### Recommended: virtual environment

Windows:

```bat
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements-core.txt
```

Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements-core.txt
```

Then install only the model stacks you need.

### YOLO / FastSAM

```bash
pip install ultralytics
```

Install the appropriate PyTorch build for your CUDA/CPU environment first if you need a specific CUDA version.

### ONNX Runtime CPU

```bash
pip install onnxruntime
```

### ONNX Runtime NVIDIA

Do not keep both CPU and GPU ONNX Runtime packages installed unnecessarily. For an NVIDIA-compatible environment, use the appropriate `onnxruntime-gpu` package/build:

```bash
pip uninstall -y onnxruntime
pip install onnxruntime-gpu
```

### RF-DETR

```bash
pip install rfdetr
```

### SigLIP / Hugging Face

```bash
pip install transformers torch torchvision
```

### Keras

```bash
pip install tensorflow
```

### Everything

`requirements-all.txt` is provided as a convenience, but for an NVIDIA development laptop it is usually better to install PyTorch and ONNX Runtime using versions matched to your CUDA environment rather than blindly installing every default wheel.

## Run

### Ready-to-use Windows executable

The repository includes a prebuilt Windows executable stored with Git LFS. To
download it when cloning the repository, install [Git LFS](https://git-lfs.com/)
and run:

```bat
git lfs install
git clone https://github.com/rishikumar-sn/Vision-model-testing-exe.git
cd Vision-model-testing-exe
git lfs pull
dist\UniversalVisionModelTester.exe
```

You can also download the executable from `dist/UniversalVisionModelTester.exe`
on the repository page. The file is large because the packaged application
contains Python and its machine-learning dependencies. Windows may show a
SmartScreen warning because the executable is not code-signed.

### Run from Python source

```bash
python main.py
```

Windows can also double-click:

```text
run_windows.bat
```

## Basic workflow

1. Start the application.
2. Confirm the top device badge shows CUDA when your PyTorch CUDA installation is working.
3. Select/enter a model.
4. Keep `Adapter = Auto`, or select the adapter explicitly for ambiguous checkpoints.
5. Select `Device = Auto`.
6. Press **Load Model**.
7. Open a camera, image, video, or folder.
8. Optionally draw an ROI.
9. Choose whether inference should be restricted to that ROI.
10. Press **Capture + Infer** for a single test, or enable **Live Inference**.
11. Adjust confidence / IoU / image size / Top-K while testing.
12. Use **Benchmark ×20** for latency statistics.
13. Save annotated output when needed.

## CUDA behavior

PyTorch adapters:

```text
Auto
  ├─ torch.cuda.is_available() == True -> cuda:0
  └─ otherwise                         -> cpu
```

ONNX Runtime adapter:

```text
Auto
  ├─ CUDAExecutionProvider available -> CUDAExecutionProvider, then CPU fallback
  └─ otherwise                       -> CPUExecutionProvider
```

If an NVIDIA GPU exists but a CPU-only PyTorch/ONNX package was installed, the application correctly reports/falls back to CPU. The app cannot add CUDA support to a CPU-only framework installation.

## ROI behavior

The ROI is stored as normalized coordinates `(0..1)` so it survives preview resizing. When **Infer only inside drawn ROI** is enabled:

```text
full frame -> ROI crop -> model preprocessing -> inference
                              |
                              v
                       result coordinates
                              |
                              v
                    add ROI x/y offset
                              |
                              v
                     full-frame overlay
```

## Anti-flicker notes

The anti-flicker implementation comes from the supplied Universal Dataset Collector and keeps the same design principles:

- Prefer camera-driver power-line frequency control where supported.
- Leave exposure control under the camera driver's authority.
- Provide temporal frame averaging as a fallback.
- Provide row-luminance normalization for horizontal banding.
- Adaptive mode measures temporal horizontal-band variation and enables/disables averaging.

For stationary objects, long averaging windows can be very effective. For moving live inference, long averaging can create ghosting; Driver 50/60 Hz, row correction, or Adaptive are normally better starting points.

## Known V1 limitations

- Generic ONNX postprocessing cannot understand every custom graph automatically.
- Arbitrary PyTorch `state_dict` files require the correct architecture.
- RF-DETR custom checkpoints require selecting the correct model-size variant.
- SigLIP public model IDs require internet access the first time the model is downloaded.
- The FastSAM ROI prompt and ROI-crop functions are separate concepts; if both are enabled, the prompt coordinates are adjusted to the inference crop.
- Live overlays represent the newest completed inference and may lag the smooth camera preview when the model is substantially slower than the camera.
- Full TensorRT/OpenVINO/Hailo HEF/QNN adapters are not included in this V1, but the adapter interface is designed so they can be added without rewriting the GUI.

## Adding another adapter

Create a new class under `adapters/` inheriting `BaseAdapter` and implement:

```python
load(source, settings) -> ModelInfo
infer(frame, settings) -> InferenceResult
```

Then register the class in `adapters/__init__.py`.

That is the intended extension point for TensorRT, OpenVINO, Hailo HEF, QNN, custom segmentation networks, or other public models.

## Build the Windows executable

The supplied spec intentionally creates an **onedir** build. Heavy ML stacks contain many DLLs; keeping them beside the executable avoids one-file extraction delays and native-library loading failures. The `.exe` must remain with its `_internal` directory.

From an activated environment containing every backend you want in the build:

```bat
python -m pip install --upgrade -r requirements-build.txt
python diagnostics.py
python -m PyInstaller --clean --noconfirm UniversalVisionModelTester.spec
```

Run the result with:

```bat
dist\UniversalVisionModelTester\UniversalVisionModelTester.exe
```

To rebuild from a completely clean PyInstaller analysis, remove only the generated `build\UniversalVisionModelTester` and `dist\UniversalVisionModelTester` directories, then run the same command again. Do not distribute just the executable; zip and distribute the entire `dist\UniversalVisionModelTester` directory.
