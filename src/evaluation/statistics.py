from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class StatisticSummary:
    mean: float
    median: float
    std: float
    ci_low: float
    ci_high: float


class StatisticalAnalyzer:
    def summary(self, values: list[float], confidence: float = 0.95) -> StatisticSummary:
        if not values:
            return StatisticSummary(0.0, 0.0, 0.0, 0.0, 0.0)
        arr = np.asarray(values, dtype=float)
        mean = float(arr.mean())
        median = float(np.median(arr))
        std = float(arr.std(ddof=1)) if len(arr) > 1 else 0.0
        low, high = self.bootstrap_ci(arr.tolist(), confidence=confidence)
        return StatisticSummary(mean, median, std, low, high)

    def paired_difference(self, a: list[float], b: list[float]) -> StatisticSummary:
        n = min(len(a), len(b))
        diffs = [a[i] - b[i] for i in range(n)]
        return self.summary(diffs)

    def bootstrap_ci(self, values: list[float], confidence: float = 0.95, iterations: int = 500, seed: int = 42) -> tuple[float, float]:
        if not values:
            return 0.0, 0.0
        rng = np.random.default_rng(seed)
        arr = np.asarray(values, dtype=float)
        samples = [float(rng.choice(arr, size=len(arr), replace=True).mean()) for _ in range(iterations)]
        alpha = (1 - confidence) / 2
        return float(np.quantile(samples, alpha)), float(np.quantile(samples, 1 - alpha))

