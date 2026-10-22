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


class AxisDetector:
    def detect(self, image: Image.Image, bars: list[Any] | None = None) -> AxisInfo:
        width, height = image.size
        bars = bars or []
        if bars:
            min_x = min(getattr(bar.geometry, "x1", 0) for bar in bars)
            min_y = min(getattr(bar.geometry, "y1", 0) for bar in bars)
            max_x = max(getattr(bar.geometry, "x2", width) for bar in bars)
            max_y = max(getattr(bar.geometry, "y2", height) for bar in bars)
            vertical_votes = sum(1 for bar in bars if getattr(bar.geometry, "height", 0) >= getattr(bar.geometry, "width", 0))
            orientation = "vertical" if vertical_votes >= len(bars) / 2 else "horizontal"
            return AxisInfo(
                x_axis_present=True,
                y_axis_present=True,
                origin=(max(0, min_x), min(height - 1, max_y)),
                plot_area=(max(0, min_x - 10), max(0, min_y - 10), min(width, max_x + 10), min(height, max_y + 10)),
                orientation=orientation,
                confidence=min(1.0, 0.5 + 0.05 * len(bars)),
                metadata={"bars_used": len(bars)},
            )
        return AxisInfo(
            x_axis_present=True,
            y_axis_present=True,
            origin=(int(width * 0.12), int(height * 0.85)),
            plot_area=(int(width * 0.08), int(height * 0.1), int(width * 0.9), int(height * 0.82)),
            orientation="mixed",
            confidence=0.3,
        )

