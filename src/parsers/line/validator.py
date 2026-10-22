from __future__ import annotations

from dataclasses import dataclass

from .geometry import CurveGeometry, DetectedPoint, DetectedSeries


@dataclass(frozen=True)
class ValidationResult:
    valid: bool
    warnings: list[str]


class LineValidator:
    def validate(
        self,
        curves: list[CurveGeometry],
        points: list[DetectedPoint],
        series: list[DetectedSeries],
        orientation: str,
        plot_area: tuple[int, int, int, int] | None = None,
    ) -> ValidationResult:
        warnings: list[str] = []
        if orientation not in {"vertical", "horizontal", "mixed"}:
            warnings.append("invalid_orientation")
        if not curves and not points:
            warnings.append("no_line_elements_detected")
        seen = set()
        for p in points:
            key = (round(p.x, 2), round(p.y, 2), p.series_id)
            if key in seen:
                warnings.append("duplicate_points")
                break
            seen.add(key)
        if plot_area is not None:
            x1, y1, x2, y2 = plot_area
            for p in points:
                if p.x < x1 or p.x > x2 or p.y < y1 or p.y > y2:
                    warnings.append("out_of_plot_points")
                    break
        for curve in curves:
            if len(curve.points) < 2:
                warnings.append("invalid_curves")
                break
        if series and any(not s.points for s in series):
            warnings.append("broken_polylines")
        return ValidationResult(valid=not warnings, warnings=warnings)

