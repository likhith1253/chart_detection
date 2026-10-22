from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class BarGeometry:
    x1: int
    y1: int
    x2: int
    y2: int
    orientation: str
    centroid_x: float | None = None
    centroid_y: float | None = None
    series_id: int | None = None
    stack_id: int | None = None
    confidence: float = 0.0
    metadata: dict[str, Any] | None = None

    @property
    def width(self) -> int:
        return max(0, self.x2 - self.x1)

    @property
    def height(self) -> int:
        return max(0, self.y2 - self.y1)

    @property
    def area(self) -> int:
        return self.width * self.height

    @property
    def centroid(self) -> tuple[float, float]:
        if self.centroid_x is not None and self.centroid_y is not None:
            return float(self.centroid_x), float(self.centroid_y)
        return ((self.x1 + self.x2) / 2.0, (self.y1 + self.y2) / 2.0)
