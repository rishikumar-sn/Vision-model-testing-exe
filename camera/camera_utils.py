from __future__ import annotations

import platform
from typing import Optional

import cv2

MAX_CAMERA_INDEX_TO_SCAN = 8
FAST_RESOLUTION_PROBES = [
    (3840, 2160), (3264, 2448), (2560, 1440),
    (1920, 1080), (1280, 720), (640, 480),
]
CAMERA_MODE_CACHE: dict[int, tuple[int, int]] = {}

def preferred_backend() -> int:
    """Return a practical OpenCV backend for the current OS."""
    system = platform.system().lower()
    if system == "windows":
        return cv2.CAP_DSHOW
    if system == "linux":
        return cv2.CAP_V4L2
    if system == "darwin":
        return cv2.CAP_AVFOUNDATION
    return cv2.CAP_ANY


def open_capture(index: int) -> Optional[cv2.VideoCapture]:
    """Open a camera, trying the preferred backend and then CAP_ANY."""
    backends = [preferred_backend()]
    if backends[0] != cv2.CAP_ANY:
        backends.append(cv2.CAP_ANY)

    for backend in backends:
        cap = cv2.VideoCapture(index, backend)
        if cap.isOpened():
            # MJPG is commonly needed for high-resolution USB webcam modes.
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            return cap
        cap.release()
    return None


def discover_camera_indices(max_index: int = MAX_CAMERA_INDEX_TO_SCAN) -> list[int]:
    """Find camera indices that can be opened and return at least one frame."""
    found: list[int] = []
    for index in range(max_index):
        cap = open_capture(index)
        if cap is None:
            continue
        try:
            # Keep discovery light; full resolution probing happens after selection.
            ok, frame = cap.read()
            if ok and frame is not None and frame.size > 0:
                found.append(index)
        finally:
            cap.release()
    return found


def directshow_camera_names() -> list[str]:
    """Return Windows camera names in the same DirectShow enumeration order."""
    if platform.system().lower() != "windows":
        return []
    try:
        from pygrabber.dshow_graph import DeviceCategories, SystemDeviceEnum

        return list(
            SystemDeviceEnum().get_available_filters(
                DeviceCategories.VideoInputDevice
            )
        )
    except Exception:
        return []

