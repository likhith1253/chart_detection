from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .ocr.axis_text import TickParseResult


@dataclass(frozen=True)
class AxisCalibration:
    origin_x: float
    origin_y: float
    value_per_pixel_x: float
    value_per_pixel_y: float
    axis_direction_x: int
    axis_direction_y: int
    scale_type: str = "linear"
    metadata: dict[str, Any] | None = None

    def pixel_to_value(self, pixel_value: float, axis: str = "y") -> float:
        if axis == "x":
            return (pixel_value - self.origin_x) * self.value_per_pixel_x * self.axis_direction_x
        return (self.origin_y - pixel_value) * self.value_per_pixel_y * self.axis_direction_y

    @classmethod
    def from_ticks(
        cls,
        origin_x: float,
        origin_y: float,
        ticks: TickParseResult | None,
        axis: str = "y",
        fallback_scale: float = 1.0,
        scale_type: str = "linear",
    ) -> "AxisCalibration":
        tick_list = []
        if ticks is not None:
            tick_list = list(getattr(ticks, "numeric_ticks", [])) or list(getattr(ticks, "ticks", []))
        if ticks is None or not tick_list:
            return cls(origin_x, origin_y, fallback_scale, fallback_scale, 1, 1, scale_type=scale_type, metadata={"source": "fallback"})
        numeric_ticks = [tick for tick in tick_list if isinstance(tick.value, (int, float))]
        if len(numeric_ticks) < 2:
            return cls(origin_x, origin_y, fallback_scale, fallback_scale, 1, 1, scale_type=scale_type, metadata={"source": "insufficient_ticks"})
        first = numeric_ticks[0]
        last = numeric_ticks[-1]
        first_px = first.bounding_box[1] if first.bounding_box and axis == "y" else first.bounding_box[0] if first.bounding_box else origin_y
        last_px = last.bounding_box[1] if last.bounding_box and axis == "y" else last.bounding_box[0] if last.bounding_box else origin_y
        delta_px = (last_px - first_px) or 1.0
        delta_val = float(last.value) - float(first.value)
        per_pixel = abs(delta_val / delta_px)
        return cls(
            origin_x,
            origin_y,
            per_pixel if axis == "x" else fallback_scale,
            per_pixel if axis == "y" else fallback_scale,
            1,
            1,
            scale_type=scale_type,
            metadata={"source": "ticks", "count": len(numeric_ticks)},
        )
