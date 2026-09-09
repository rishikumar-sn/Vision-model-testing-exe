from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Detection:
    x1: float
    y1: float
    x2: float
    y2: float
    confidence: float = 1.0
    class_id: int = -1
    label: str = "object"

    def shifted(self, dx: float, dy: float) -> "Detection":
        return Detection(self.x1 + dx, self.y1 + dy, self.x2 + dx, self.y2 + dy,
                         self.confidence, self.class_id, self.label)


@dataclass
class PolygonMask:
    points: list[tuple[float, float]]
    confidence: float = 1.0
    class_id: int = -1
    label: str = "mask"

    def shifted(self, dx: float, dy: float) -> "PolygonMask":
        return PolygonMask([(x + dx, y + dy) for x, y in self.points],
                           self.confidence, self.class_id, self.label)


@dataclass
class Classification:
    label: str
    score: float
    class_id: int = -1


@dataclass
class KeypointSet:
    points: list[tuple[float, float, float]]

    def shifted(self, dx: float, dy: float) -> "KeypointSet":
        return KeypointSet([(x + dx, y + dy, c) for x, y, c in self.points])


@dataclass
class InferenceResult:
    detections: list[Detection] = field(default_factory=list)
    masks: list[PolygonMask] = field(default_factory=list)
    classifications: list[Classification] = field(default_factory=list)
    keypoints: list[KeypointSet] = field(default_factory=list)
    summary: str = ""
    adapter: str = ""
    task: str = "unknown"
    device: str = ""
    preprocess_ms: float = 0.0
    inference_ms: float = 0.0
    postprocess_ms: float = 0.0
    total_ms: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    def shifted(self, dx: float, dy: float) -> "InferenceResult":
        if not dx and not dy:
            return self
        self.detections = [d.shifted(dx, dy) for d in self.detections]
        self.masks = [m.shifted(dx, dy) for m in self.masks]
        self.keypoints = [k.shifted(dx, dy) for k in self.keypoints]
        return self


@dataclass
class ModelInfo:
    source: str
    adapter: str
    task: str = "unknown"
    device: str = ""
    input_description: str = "—"
    output_description: str = "—"
    details: dict[str, Any] = field(default_factory=dict)


@dataclass
class BenchmarkResult:
    runs: int
    warmup_runs: int
    average_ms: float
    median_ms: float
    minimum_ms: float
    maximum_ms: float
    p95_ms: float
    fps: float
