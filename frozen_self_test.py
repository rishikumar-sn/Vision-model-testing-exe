"""Write model-load and inference diagnostics from either Python or the frozen app."""

import json
import traceback
from pathlib import Path


def run(model_path: str, image_path: str, report_path: str) -> int:
    report = {"model": model_path, "image": image_path, "attempts": []}
    try:
        import cv2

        from adapters import adapter_classes_for
        from core.device import detect_devices

        frame = cv2.imread(image_path)
        if frame is None:
            raise RuntimeError(f"Could not read image: {image_path}")
        settings = {"device": "cpu", "imgsz": 640, "conf": 0.25, "iou": 0.45}
        status = detect_devices()
        report["device"] = status.summary
        for adapter_class in adapter_classes_for(model_path, "Auto"):
            attempt = {"adapter": adapter_class.display_name}
            report["attempts"].append(attempt)
            adapter = adapter_class(status)
            try:
                info = adapter.load(model_path, settings)
                result = adapter.infer(frame, settings)
                attempt.update({
                    "ok": True,
                    "task": info.task,
                    "detections": len(result.detections),
                    "masks": len(result.masks),
                    "summary": result.summary,
                })
                break
            except Exception:
                attempt["error"] = traceback.format_exc()
            finally:
                adapter.close()
    except Exception:
        report["error"] = traceback.format_exc()
    Path(report_path).write_text(json.dumps(report, indent=2), encoding="utf-8")
    return 0 if any(item.get("ok") for item in report["attempts"]) else 1
