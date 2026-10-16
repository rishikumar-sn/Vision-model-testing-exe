# Linux installation

This is a PyQt6 desktop application. Run it in a graphical Linux session with
X11 or Wayland; a terminal-only server cannot display the window. Start with
Python 3.10 or 3.11 for compatibility with the optional model frameworks.
Install only the backends you need. The Windows executable is not used on Linux.

## 1. Install system packages

On Ubuntu 22.04/24.04 or a compatible Debian-based desktop:

```bash
sudo apt update
sudo apt install python3 python3-venv python3-pip \
  libgl1 libegl1 libglib2.0-0 libdbus-1-3 libfontconfig1 \
  libx11-xcb1 libxkbcommon-x11-0 libxcb-cursor0 libxcb-icccm4 \
  libxcb-image0 libxcb-keysyms1 libxcb-randr0 libxcb-render-util0 \
  libxcb-shape0 libxcb-xfixes0 libxcb-xinerama0 v4l-utils
python3 --version
```

Other distributions need equivalent Python, OpenGL/EGL, GLib and Qt XCB runtime
libraries. See [Qt's Linux requirements](https://doc.qt.io/qt-6/linux-requirements.html).
Optional backend support also depends on CPU architecture and available wheels.

## 2. Copy the source and create a Linux environment

Copy this entire clean project directory to a writable Linux location, such as
`~/projects/universal_vision_model_tester`. Do not reuse a Windows virtual
environment; create a new one on Linux.

```bash
cd ~/projects/universal_vision_model_tester
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-core.txt
```

Windows-only `comtypes` and `pygrabber` are automatically skipped on Linux by
the requirement markers. Core packages open the application; model inference
requires an appropriate optional backend.

## 3. Install model backends

Run the commands for the backends you use inside the activated environment:

| Models | Install command |
| --- | --- |
| YOLO / FastSAM | `python -m pip install ultralytics` |
| Generic ONNX on CPU | `python -m pip install onnxruntime` |
| PyTorch classifiers | `python -m pip install torch torchvision` |
| RF-DETR | `python -m pip install rfdetr` |
| SigLIP | `python -m pip install transformers torch torchvision` |
| Keras / TensorFlow | `python -m pip install tensorflow` |

For convenience, `python -m pip install -r requirements-all.txt` requests all
backends, but their combined dependency constraints may require separate virtual
environments. Dependencies are not locked, so installation is not a guarantee
that every framework combination will work.

For NVIDIA acceleration, install a working Linux NVIDIA driver, check
`nvidia-smi`, then choose a compatible PyTorch wheel from the
[official PyTorch installer](https://pytorch.org/get-started/locally/).
For ONNX CUDA, follow the
[ONNX Runtime CUDA requirements](https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html).
Keep only one ONNX Runtime distribution in the environment:

```bash
python -m pip uninstall -y onnxruntime onnxruntime-gpu
python -m pip install onnxruntime-gpu
```

Use the CPU package instead when CUDA is not needed. The app supports CPU
fallback; hardware acceleration depends on the installed framework and driver.

## 4. Check and run

```bash
python -m pip check
python diagnostics.py
bash run_linux.sh
```

The launcher works from any working directory. It prefers an activated virtual
environment, then this project's `.venv`, then `python3`. You may also run
`python main.py` from the project directory in the activated environment.
To launch the script directly, first run `chmod +x run_linux.sh`.

Diagnostics reports missing optional packages; these matter only if you need
their adapters. First test an image and then a camera. Copy your model files
separately and select their Linux paths; Windows absolute paths do not transfer.
Model profiles and saved images need writable destination directories.

## 5. Camera access and anti-flicker

Linux capture uses OpenCV's V4L2 backend with an automatic fallback. Inspect
available devices and permissions with:

```bash
v4l2-ctl --list-devices
ls -l /dev/video*
```

If device permissions deny access and the devices belong to the `video` group:

```bash
sudo usermod -aG video "$USER"
```

Log out and back in after changing groups. Close other applications using the
camera. The app scans camera indices 0 through 7 by default.

DirectShow camera names, the Driver Settings dialog and the app's Driver 50/60 Hz
control are Windows-only. Adaptive, frame averaging and row correction work in
software on Linux. To configure hardware power-line frequency outside the app:

```bash
v4l2-ctl -d /dev/video0 --list-ctrls
# Only if this control and the corresponding menu values are supported:
v4l2-ctl -d /dev/video0 --set-ctrl=power_line_frequency=1  # 50 Hz
v4l2-ctl -d /dev/video0 --set-ctrl=power_line_frequency=2  # 60 Hz
```

## Troubleshooting

- **Qt xcb plugin / missing library:** install the system packages above and use
  `QT_DEBUG_PLUGINS=1 bash run_linux.sh` to identify missing libraries. The app
  restores Qt environment settings after OpenCV imports to avoid bundled Qt
  plugin conflicts. See the [OpenCV packaging documentation](https://github.com/opencv/opencv-python).
- **Wayland display issue:** try `QT_QPA_PLATFORM=xcb bash run_linux.sh` in a
  desktop session with XWayland available.
- **No display:** use a local graphical desktop or configure remote graphical
  access. `QT_QPA_PLATFORM=offscreen` can check imports but cannot show the UI.
- **Missing backend:** install the adapter's dependencies using the same
  environment's `python -m pip`, then rerun `python diagnostics.py`.
- **Transfer introduced CRLF:** run `sed -i 's/\r$//' run_linux.sh`. The supplied
  script uses LF and `.gitattributes` preserves LF in future Git checkouts.

## Validation scope

This transfer was prepared on Windows. Source syntax and local startup checks
are performed there; a native Linux desktop, camera and GPU run is still needed
to validate your particular Linux machine and installed model backends.
