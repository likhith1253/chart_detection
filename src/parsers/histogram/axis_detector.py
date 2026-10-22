from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from PIL import Image


@dataclass(frozen=True)
class AxisInfo:
    x_axis_present: bool
    y_axis_present: bool
    origin: tuple[int, int]
    plot_area: tuple[int, int, int, int]
    orientation: str
    confidence: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class AxisDetector:
    metadata: dict[str, Any] = field(default_factory=dict)

    def detect(self, image: Image.Image, bins: list[Any] | None = None) -> AxisInfo:
        width, height = image.size
        bins = bins or []
        if bins:
            x1 = min(getattr(b.geometry, "x1", 0) for b in bins)
            y1 = min(getattr(b.geometry, "y1", 0) for b in bins)
            x2 = max(getattr(b.geometry, "x2", width) for b in bins)
            y2 = max(getattr(b.geometry, "y2", height) for b in bins)
            return AxisInfo(True, True, (x1, y2), (max(0, x1 - 10), max(0, y1 - 10), min(width, x2 + 10), min(height, y2 + 10)), "continuous", confidence=min(0.95, 0.4 + 0.1 * len(bins)))
        return AxisInfo(True, True, (int(width * 0.1), int(height * 0.85)), (int(width * 0.08), int(height * 0.1), int(width * 0.92), int(height * 0.85)), "continuous", confidence=0.3)

