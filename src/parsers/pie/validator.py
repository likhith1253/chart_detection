from __future__ import annotations

from dataclasses import dataclass

from .geometry import DetectedSlice


@dataclass(frozen=True)
class ValidationResult:
    valid: bool
    warnings: list[str]


class PieValidator:
    def validate(self, slices: list[DetectedSlice], center: tuple[float, float] | None = None) -> ValidationResult:
        warnings: list[str] = []
        if not slices:
            warnings.append("missing_slices")
        if center is not None:
            centers = [s.geometry.centroid for s in slices]
            if any(abs(cx - center[0]) > 30 or abs(cy - center[1]) > 30 for cx, cy in centers):
                warnings.append("center_consistency")
        ordered = sorted(slices, key=lambda s: s.geometry.start_angle)
        for a, b in zip(ordered, ordered[1:]):
            if a.geometry.end_angle > b.geometry.start_angle + 1.5:
                warnings.append("overlapping_slices")
                break
        angle_sum = sum(s.geometry.sweep_angle for s in slices)
        if not (300.0 <= angle_sum <= 420.0):
            warnings.append("angle_consistency")
        if any(s.geometry.sweep_angle <= 0 for s in slices):
            warnings.append("invalid_slice_geometry")
        return ValidationResult(valid=not warnings, warnings=warnings)

