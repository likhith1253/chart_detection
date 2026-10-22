from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class DetectedPoint:
    x: float
    y: float
    confidence: float = 0.0
    marker: str = "implicit"
    series_id: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class DetectedSeries:
    series_id: int
    color: tuple[int, int, int] | None
    line_type: str
    points: list[DetectedPoint]
    confidence: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class CurveGeometry:
    points: list[tuple[int, int]]
    bbox: tuple[int, int, int, int]
    curve_type: str
    confidence: float = 0.0
    series_id: int | None = None

