from __future__ import annotations

import cv2
import numpy as np

from core.results import InferenceResult


def render_result(frame: np.ndarray, result: InferenceResult | None, mask_alpha: float = 0.28) -> np.ndarray:
    if result is None:
        return frame
    out = frame.copy()

    if result.masks:
        overlay = out.copy()
        for mask in result.masks:
            if len(mask.points) < 3:
                continue
            pts = np.asarray(mask.points, dtype=np.int32).reshape((-1, 1, 2))
            cv2.fillPoly(overlay, [pts], (0, 220, 170))
            cv2.polylines(out, [pts], True, (0, 255, 190), 2, cv2.LINE_AA)
        out = cv2.addWeighted(overlay, mask_alpha, out, 1.0 - mask_alpha, 0)

    for det in result.detections:
        p1 = (int(det.x1), int(det.y1))
        p2 = (int(det.x2), int(det.y2))
        cv2.rectangle(out, p1, p2, (0, 220, 170), 2, cv2.LINE_AA)
        text = f"{det.label} {det.confidence:.2f}" if det.confidence >= 0 else det.label
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        ty = max(th + 6, p1[1])
        cv2.rectangle(out, (p1[0], ty - th - 7), (p1[0] + tw + 8, ty + 2), (16, 19, 24), -1)
        cv2.putText(out, text, (p1[0] + 4, ty - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    (240, 245, 248), 1, cv2.LINE_AA)

    for kp_set in result.keypoints:
        for x, y, confidence in kp_set.points:
            if confidence <= 0:
                continue
            cv2.circle(out, (int(x), int(y)), 3, (0, 220, 170), -1, cv2.LINE_AA)

    if result.classifications:
        y = 28
        for item in result.classifications[:5]:
            text = f"{item.label}: {item.score * 100:.1f}%"
            cv2.putText(out, text, (14, y), cv2.FONT_HERSHEY_SIMPLEX, 0.65,
                        (245, 245, 245), 2, cv2.LINE_AA)
            y += 26
    return out
