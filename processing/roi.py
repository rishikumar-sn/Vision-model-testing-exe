from __future__ import annotations

import numpy as np


def normalized_roi_to_pixels(roi_norm, width: int, height: int):
    if not roi_norm:
        return 0, 0, width, height
    x, y, w, h = roi_norm
    x1 = max(0, min(width - 1, int(round(x * width))))
    y1 = max(0, min(height - 1, int(round(y * height))))
    x2 = max(x1 + 1, min(width, int(round((x + w) * width))))
    y2 = max(y1 + 1, min(height, int(round((y + h) * height))))
    return x1, y1, x2, y2


def crop_frame(frame: np.ndarray, roi_norm, enabled: bool = True):
    h, w = frame.shape[:2]
    if not enabled or not roi_norm:
        return frame, (0, 0), (0, 0, w, h)
    x1, y1, x2, y2 = normalized_roi_to_pixels(roi_norm, w, h)
    return frame[y1:y2, x1:x2].copy(), (x1, y1), (x1, y1, x2, y2)
