from __future__ import annotations

from dataclasses import dataclass

from .geometry import DetectedPoint, MarkerGeometry, ScatterSeries


@dataclass(frozen=True)
class ValidationResult:
    valid: bool
    warnings: list[str]


class ScatterValidator:
    def validate(
        self,
        markers: list[MarkerGeometry],
        points: list[DetectedPoint],
        series: list[ScatterSeries],
        plot_area: tuple[int, int, int, int] | None = None,
    ) -> ValidationResult:
        warnings: list[str] = []
        if not markers:
            warnings.append("no_scatter_markers_detected")
        seen = set()
        for p in points:
            key = (round(p.x, 2), round(p.y, 2), p.series_id)
            if key in seen:
                warnings.append("duplicate_detections")
                break
            seen.add(key)
            if p.x < 0 or p.y < 0:
                warnings.append("invalid_centroids")
                break
        if plot_area is not None:
            x1, y1, x2, y2 = plot_area
            if any(p.x < x1 or p.x > x2 or p.y < y1 or p.y > y2 for p in points):
                warnings.append("out_of_plot_detections")
        if series and any(not s.points for s in series):
            warnings.append("series_consistency")
        if any(m.bbox[2] <= m.bbox[0] or m.bbox[3] <= m.bbox[1] for m in markers):
            warnings.append("invalid_centroids")
        return ValidationResult(valid=not warnings, warnings=warnings)

