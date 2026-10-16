#!/usr/bin/env python3
from __future__ import annotations

import os
import multiprocessing
import sys
from pathlib import Path

# Use automatic display scaling where supported.
os.environ.setdefault("QT_AUTO_SCREEN_SCALE_FACTOR", "1")

def main() -> int:
    multiprocessing.freeze_support()
    if getattr(sys, "frozen", False):
        # A packaged executable cannot install into itself. Keep missing
        # optional backend errors visible instead of spawning a pip child.
        os.environ["YOLO_AUTOINSTALL"] = "False"
    if len(sys.argv) == 5 and sys.argv[1] == "--self-test":
        from frozen_self_test import run

        return run(sys.argv[2], sys.argv[3], sys.argv[4])

    # Linux OpenCV wheels can replace these with their bundled Qt 5 paths.
    # Preserve the user's Qt configuration before importing modules using cv2.
    qt_environment = {
        key: os.environ.get(key)
        for key in ("QT_QPA_PLATFORM_PLUGIN_PATH", "QT_QPA_FONTDIR")
    }
    from PyQt6.QtGui import QIcon
    from PyQt6.QtWidgets import QApplication
    from ui.main_window import MainWindow

    if sys.platform.startswith("linux"):
        for key, value in qt_environment.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value

    app = QApplication(sys.argv)
    app.setApplicationName("Universal Vision Model Tester")
    bundle_dir = Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parent))
    app.setWindowIcon(QIcon(str(bundle_dir / "applogo.png")))
    window = MainWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
