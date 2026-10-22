from __future__ import annotations

from dataclasses import dataclass

from .geometry import BarGeometry


@dataclass(frozen=True)
class ValidationResult:
    valid: bool
    warnings: list[str]


class BarValidator:
    def validate(
        self,
        bars: list[BarGeometry],
        labels: list[str],
        orientation: str,
        plot_area: tuple[int, int, int, int] | None = None,
    ) -> ValidationResult:
        warnings: list[str] = []
        if labels and len(labels) != len(bars):
            warnings.append("number_of_bars_does_not_match_number_of_labels")
        if orientation not in {"vertical", "horizontal", "mixed"}:
            warnings.append("invalid_orientation")
        if any(bar.width <= 0 or bar.height <= 0 for bar in bars):
            warnings.append("invalid_bar_dimensions")
        if len(bars) > 1:
            sorted_bars = sorted(bars, key=lambda b: (b.x1, b.y1))
            for left, right in zip(sorted_bars, sorted_bars[1:]):
                if not (left.x2 <= right.x1 or left.y2 <= right.y1):
                    warnings.append("overlapping_bars_detected")
                    break
        if plot_area is not None:
            px1, py1, px2, py2 = plot_area
            for bar in bars:
                if bar.x2 < px1 or bar.x1 > px2 or bar.y2 < py1 or bar.y1 > py2:
                    warnings.append("out_of_plot_detection")
                    break
        seen = set()
        for bar in bars:
            key = (bar.x1, bar.y1, bar.x2, bar.y2, bar.orientation)
            if key in seen:
                warnings.append("duplicate_detections")
                break
            seen.add(key)
        if orientation not in {bar.orientation for bar in bars} and bars:
            warnings.append("inconsistent_orientation")
        return ValidationResult(valid=not warnings, warnings=warnings)

