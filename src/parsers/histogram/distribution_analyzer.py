from __future__ import annotations

from dataclasses import dataclass, field
from math import fabs
from typing import Any

from .geometry import DetectedBin, DistributionStatistics, HistogramDistribution


@dataclass
class DistributionAnalyzer:
    metadata: dict[str, Any] = field(default_factory=dict)

    def analyze(self, bins: list[DetectedBin]) -> HistogramDistribution:
        if not bins:
            return HistogramDistribution([], 0.0, 0.0, 0.0, 0.0, metadata={"empty": True})
        widths = [b.geometry.width for b in bins]
        heights = [b.geometry.height for b in bins]
        continuity = self._continuity(bins)
        symmetry = self._symmetry(heights)
        skewness = self._skewness(heights)
        density = sum(heights) / max(1.0, sum(widths) * max(heights))
        return HistogramDistribution(bins=bins, continuity=continuity, symmetry=symmetry, skewness=skewness, density=density, metadata=self.statistics(bins).__dict__)

    def statistics(self, bins: list[DetectedBin]) -> DistributionStatistics:
        if not bins:
            return DistributionStatistics(0.0, 0.0, 0.0, 0.0, 0, 0.0, 0.0, 0.0, 0.0)
        widths = [b.geometry.width for b in bins]
        heights = [b.geometry.height for b in bins]
        mw = sum(widths) / len(widths)
        mh = sum(heights) / len(heights)
        width_var = sum((w - mw) ** 2 for w in widths) / len(widths)
        height_var = sum((h - mh) ** 2 for h in heights) / len(heights)
        return DistributionStatistics(
            mean_width=mw,
            width_variance=width_var,
            mean_height=mh,
            height_variance=height_var,
            bin_count=len(bins),
            continuity=self._continuity(bins),
            symmetry=self._symmetry(heights),
            skewness=self._skewness(heights),
            density=sum(heights) / max(1.0, sum(widths) * max(heights)),
            metadata={"sources": "distribution_analyzer"},
        )

    def _continuity(self, bins: list[DetectedBin]) -> float:
        if len(bins) < 2:
            return 1.0
        sorted_bins = sorted(bins, key=lambda b: b.geometry.x1)
        gaps = []
        for a, b in zip(sorted_bins, sorted_bins[1:]):
            gap = max(0, b.geometry.x1 - a.geometry.x2)
            gaps.append(gap)
        avg_gap = sum(gaps) / len(gaps)
        return max(0.0, 1.0 - avg_gap / max(1.0, sum(b.geometry.width for b in bins) / len(bins)))

    def _symmetry(self, heights: list[int]) -> float:
        if not heights:
            return 0.0
        mid = len(heights) // 2
        left = heights[:mid]
        right = heights[mid + (len(heights) % 2):]
        if not left or not right:
            return 0.5
        score = 1.0 - abs((sum(left) / len(left)) - (sum(right) / len(right))) / max(1.0, max(heights))
        return max(0.0, min(1.0, score))

    def _skewness(self, heights: list[int]) -> float:
        if len(heights) < 3:
            return 0.0
        mean = sum(heights) / len(heights)
        m2 = sum((h - mean) ** 2 for h in heights) / len(heights)
        m3 = sum((h - mean) ** 3 for h in heights) / len(heights)
        if m2 == 0:
            return 0.0
        return float(m3 / (m2 ** 1.5))

