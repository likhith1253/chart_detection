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

    def detect(self, image: Image.Image, curves: list[Any] | None = None, points: list[Any] | None = None) -> AxisInfo:
        width, height = image.size
        curves = curves or []
        points = points or []
        if curves:
            xs1 = [getattr(c, "bbox", (0, 0, width, height))[0] for c in curves]
            ys1 = [getattr(c, "bbox", (0, 0, width, height))[1] for c in curves]
            xs2 = [getattr(c, "bbox", (0, 0, width, height))[2] for c in curves]
            ys2 = [getattr(c, "bbox", (0, 0, width, height))[3] for c in curves]
            plot = (max(0, min(xs1) - 10), max(0, min(ys1) - 10), min(width, max(xs2) + 10), min(height, max(ys2) + 10))
            orientation = "vertical" if height >= width else "horizontal"
            return AxisInfo(True, True, (plot[0], plot[3]), plot, orientation, confidence=min(0.95, 0.4 + 0.1 * len(curves)), metadata={"curves": len(curves), "points": len(points)})
        return AxisInfo(True, True, (int(width * 0.1), int(height * 0.85)), (int(width * 0.08), int(height * 0.1), int(width * 0.92), int(height * 0.85)), "vertical", confidence=0.3)

