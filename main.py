#!/usr/bin/env python3
from __future__ import annotations

import os
import multiprocessing
import sys
from pathlib import Path

# Keep OpenCV/Qt plugin conflicts predictable on Linux.
os.environ.setdefault("QT_AUTO_SCREEN_SCALE_FACTOR", "1")

from PyQt6.QtGui import QIcon
from PyQt6.QtWidgets import QApplication

from ui.main_window import MainWindow


def main() -> int:
    multiprocessing.freeze_support()
    app = QApplication(sys.argv)
    app.setApplicationName("Universal Vision Model Tester")
    bundle_dir = Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parent))
    app.setWindowIcon(QIcon(str(bundle_dir / "applogo.png")))
    window = MainWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
