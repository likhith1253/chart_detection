from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class BinGeometry:
    x1: int
    y1: int
    x2: int
    y2: int
    width: int
    height: int
    area: int
    centroid: tuple[float, float]
    confidence: float = 0.0
    bin_index: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class DetectedBin:
    geometry: BinGeometry
    score: float
    merged_from: list[int] = field(default_factory=list)


@dataclass(frozen=True)
class HistogramDistribution:
    bins: list[DetectedBin]
    continuity: float
    symmetry: float
    skewness: float
    density: float
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class DistributionStatistics:
    mean_width: float
    width_variance: float
    mean_height: float
    height_variance: float
    bin_count: int
    continuity: float
    symmetry: float
    skewness: float
    density: float
    metadata: dict[str, Any] = field(default_factory=dict)

