from __future__ import annotations

from dataclasses import dataclass

from .geometry import DetectedBox, OutlierPoint


@dataclass(frozen=True)
class ValidationResult:
    valid: bool
    warnings: list[str]


class BoxPlotValidator:
    def validate(self, boxes: list[DetectedBox], orientation: str, plot_area: tuple[int, int, int, int] | None = None) -> ValidationResult:
        warnings: list[str] = []
        if orientation not in {"vertical", "horizontal", "grouped", "multiple"}:
            warnings.append("invalid_orientation")
        if not boxes:
            warnings.append("no_boxplots_detected")
        for box in boxes:
            if box.quartiles.lower_quartile > box.quartiles.median or box.quartiles.median > box.quartiles.upper_quartile:
                warnings.append("quartile_ordering_invalid")
                break
            if box.whiskers.lower > box.quartiles.lower_quartile or box.whiskers.upper < box.quartiles.upper_quartile:
                warnings.append("invalid_whiskers")
                break
        if plot_area is not None:
            x1, y1, x2, y2 = plot_area
            for box in boxes:
                gx1, gy1, gx2, gy2 = box.geometry.x1, box.geometry.y1, box.geometry.x2, box.geometry.y2
                if gx2 < x1 or gx1 > x2 or gy2 < y1 or gy1 > y2:
                    warnings.append("broken_geometry")
                    break
        seen = set()
        for box in boxes:
            key = (box.geometry.x1, box.geometry.y1, box.geometry.x2, box.geometry.y2)
            if key in seen:
                warnings.append("duplicate_boxes")
                break
            seen.add(key)
            for outlier in box.outliers:
                if not self._valid_outlier(outlier, box):
                    warnings.append("invalid_outliers")
                    break
        return ValidationResult(valid=not warnings, warnings=warnings)

    def _valid_outlier(self, outlier: OutlierPoint, box: DetectedBox) -> bool:
        return box.geometry.x1 - 50 <= outlier.x <= box.geometry.x2 + 50 and box.geometry.y1 - 50 <= outlier.y <= box.geometry.y2 + 50

