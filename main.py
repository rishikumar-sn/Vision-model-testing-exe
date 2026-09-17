#!/usr/bin/env python3
from __future__ import annotations

import os
import multiprocessing
import sys
from pathlib import Path

# Keep OpenCV/Qt plugin conflicts predictable on Linux.
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

    from PyQt6.QtGui import QIcon
    from PyQt6.QtWidgets import QApplication
    from ui.main_window import MainWindow

    app = QApplication(sys.argv)
    app.setApplicationName("Universal Vision Model Tester")
    bundle_dir = Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parent))
    app.setWindowIcon(QIcon(str(bundle_dir / "applogo.png")))
    window = MainWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
