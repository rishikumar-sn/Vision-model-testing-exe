from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def profile_path_for(source: str) -> Path | None:
    p = Path(source)
    if not p.exists() or p.is_dir():
        return None
    return p.with_suffix(p.suffix + ".infer.json")


def load_profile(source: str) -> dict[str, Any]:
    path = profile_path_for(source)
    if path is None or not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def save_profile(source: str, data: dict[str, Any]) -> Path | None:
    path = profile_path_for(source)
    if path is None:
        return None
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return path
