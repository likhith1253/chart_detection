from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .calibration import AxisCalibration
from .geometry import BarGeometry


@dataclass(frozen=True)
class BarSeries:
    series_name: str
    bars: list[BarGeometry]
    orientation: str


class ValueExtractor:
    def estimate_values(
        self,
        bars: list[BarGeometry],
        calibration: AxisCalibration | None = None,
        axis_scale: str = "linear",
        orientation: str = "vertical",
    ) -> list[float]:
        if not bars:
            return []
        calibration = calibration or AxisCalibration(0.0, 0.0, 1.0, 1.0, 1, 1, scale_type=axis_scale)
        values: list[float] = []
        for bar in bars:
            if orientation == "horizontal":
                pixel_value = bar.centroid_x if bar.centroid_x is not None else bar.x2
                values.append(round(abs(calibration.pixel_to_value(float(pixel_value), axis="x")), 4))
            else:
                pixel_value = bar.centroid_y if bar.centroid_y is not None else bar.y1
                values.append(round(abs(calibration.pixel_to_value(float(pixel_value), axis="y")), 4))
        if axis_scale == "categorical":
            return [round(i + 1.0, 4) for i in range(len(values))]
        return values

