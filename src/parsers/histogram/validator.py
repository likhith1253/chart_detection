from __future__ import annotations

from dataclasses import dataclass

from .geometry import DetectedBin, HistogramDistribution


@dataclass(frozen=True)
class ValidationResult:
    valid: bool
    warnings: list[str]


class HistogramValidator:
    def validate(self, bins: list[DetectedBin], distribution: HistogramDistribution, plot_area: tuple[int, int, int, int] | None = None) -> ValidationResult:
        warnings: list[str] = []
        if not bins:
            warnings.append("missing_bins")
        seen = set()
        for bin_ in bins:
            g = bin_.geometry
            key = (g.x1, g.y1, g.x2, g.y2)
            if key in seen:
                warnings.append("duplicate_bins")
                break
            seen.add(key)
            if g.width <= 0:
                warnings.append("invalid_widths")
            if g.x2 <= g.x1 or g.y2 <= g.y1:
                warnings.append("invalid_widths")
        if len(bins) > 1:
            sorted_bins = sorted(bins, key=lambda b: b.geometry.x1)
            for a, b in zip(sorted_bins, sorted_bins[1:]):
                if a.geometry.x2 > b.geometry.x1:
                    warnings.append("overlapping_bins")
                    break
        if distribution.continuity < 0.15:
            warnings.append("broken_distributions")
        if plot_area is not None:
            x1, y1, x2, y2 = plot_area
            if any(b.geometry.x1 < x1 or b.geometry.x2 > x2 or b.geometry.y1 < y1 or b.geometry.y2 > y2 for b in bins):
                warnings.append("out_of_plot_detections")
        return ValidationResult(valid=not warnings, warnings=warnings)

