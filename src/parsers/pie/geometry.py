from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class SliceGeometry:
    start_angle: float
    end_angle: float
    sweep_angle: float
    radius: float
    inner_radius: float
    centroid: tuple[float, float]
    confidence: float = 0.0
    label: str = ""
    slice_id: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class DetectedSlice:
    geometry: SliceGeometry
    score: float
    exploded: bool = False


@dataclass(frozen=True)
class PieSeries:
    center: tuple[float, float]
    radius: float
    inner_radius: float
    slices: list[DetectedSlice]
    chart_type: str = "pie"
    metadata: dict[str, Any] = field(default_factory=dict)

