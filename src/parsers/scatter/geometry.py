from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class MarkerGeometry:
    bbox: tuple[int, int, int, int]
    centroid: tuple[float, float]
    shape: str
    color: tuple[int, int, int] | None = None
    confidence: float = 0.0
    series_id: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class DetectedPoint:
    x: float
    y: float
    confidence: float = 0.0
    marker_shape: str = "unknown"
    series_id: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ScatterSeries:
    series_id: int
    markers: list[MarkerGeometry]
    points: list[DetectedPoint]
    confidence: float = 0.0
    label: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

