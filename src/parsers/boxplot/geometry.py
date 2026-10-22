from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class BoxGeometry:
    x1: int
    y1: int
    x2: int
    y2: int
    orientation: str
    confidence: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def width(self) -> int:
        return max(0, self.x2 - self.x1)

    @property
    def height(self) -> int:
        return max(0, self.y2 - self.y1)

    @property
    def centroid(self) -> tuple[float, float]:
        return ((self.x1 + self.x2) / 2.0, (self.y1 + self.y2) / 2.0)


@dataclass(frozen=True)
class QuartileGeometry:
    lower_quartile: float
    median: float
    upper_quartile: float


@dataclass(frozen=True)
class WhiskerGeometry:
    lower: float
    upper: float
    lower_cap: tuple[int, int] | None = None
    upper_cap: tuple[int, int] | None = None


@dataclass(frozen=True)
class OutlierPoint:
    x: float
    y: float
    value: float | None = None
    extreme: bool = False
    box_id: int | None = None
    confidence: float = 0.0


@dataclass(frozen=True)
class DetectedBox:
    box_id: int
    geometry: BoxGeometry
    quartiles: QuartileGeometry
    whiskers: WhiskerGeometry
    outliers: list[OutlierPoint] = field(default_factory=list)
    confidence: float = 0.0


@dataclass(frozen=True)
class BoxStatistics:
    box_id: int
    lower_whisker: float
    lower_quartile: float
    median: float
    upper_quartile: float
    upper_whisker: float
    outlier_count: int
    extreme_outlier_count: int
