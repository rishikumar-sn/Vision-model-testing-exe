"""Run PyInstaller analysis on a thread with room for deep ML import graphs."""

import sys
import threading
import traceback
from datetime import datetime
from pathlib import Path


def main() -> int:
    from PyInstaller.__main__ import run

    errors = []
    output_root = Path("dist_builds") / datetime.now().strftime("%Y%m%d_%H%M%S")
    executable = output_root / "UniversalVisionModelTester" / "UniversalVisionModelTester.exe"

    def build() -> None:
        try:
            run(["--noconfirm", "--distpath", str(output_root), "UniversalVisionModelTester.spec"])
        except BaseException as exc:
            errors.append(exc)
            traceback.print_exc()

    threading.stack_size(64 * 1024 * 1024)
    worker = threading.Thread(target=build, name="pyinstaller-build")
    worker.start()
    worker.join()
    if errors or not executable.is_file():
        return 1
    print(f"Build ready: {executable.resolve()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
