from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np

from core.device import DeviceStatus
from core.results import InferenceResult, ModelInfo


class AdapterError(RuntimeError):
    pass


class BaseAdapter(ABC):
    display_name = "Base"

    def __init__(self, device_status: DeviceStatus):
        self.device_status = device_status
        self.source = ""
        self.device = "cpu"
        self.task = "unknown"
        self.model: Any = None

    @abstractmethod
    def load(self, source: str, settings: dict[str, Any]) -> ModelInfo:
        raise NotImplementedError

    @abstractmethod
    def infer(self, frame: np.ndarray, settings: dict[str, Any]) -> InferenceResult:
        raise NotImplementedError

    def close(self) -> None:
        self.model = None

    def _labels(self, settings: dict[str, Any]) -> list[str]:
        labels = settings.get("labels") or []
        return [str(x) for x in labels]
