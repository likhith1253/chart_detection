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

    def detect(self, image: Image.Image, points: list[Any] | None = None) -> AxisInfo:
        width, height = image.size
        points = points or []
        if points:
            xs = [getattr(p, "centroid", (getattr(p, "x", 0), getattr(p, "y", 0)))[0] if hasattr(p, "centroid") else getattr(p, "x", 0) for p in points]
            ys = [getattr(p, "centroid", (getattr(p, "x", 0), getattr(p, "y", 0)))[1] if hasattr(p, "centroid") else getattr(p, "y", 0) for p in points]
            plot = (max(0, int(min(xs)) - 12), max(0, int(min(ys)) - 12), min(width, int(max(xs)) + 12), min(height, int(max(ys)) + 12))
            return AxisInfo(True, True, (plot[0], plot[3]), plot, "mixed", confidence=min(0.95, 0.4 + 0.1 * len(points)), metadata={"points": len(points)})
        return AxisInfo(True, True, (int(width * 0.1), int(height * 0.85)), (int(width * 0.08), int(height * 0.1), int(width * 0.92), int(height * 0.85)), "mixed", confidence=0.3)

