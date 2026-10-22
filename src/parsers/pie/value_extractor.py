from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .geometry import DetectedSlice


@dataclass
class ValueExtractor:
    metadata: dict[str, Any] = field(default_factory=dict)

    def estimate(self, slices: list[DetectedSlice], *, absolute_total: float | None = None) -> list[dict[str, float]]:
        total = sum(max(0.0, s.geometry.sweep_angle) for s in slices) or 1.0
        payload: list[dict[str, float]] = []
        for s in slices:
            pct = max(0.0, s.geometry.sweep_angle) / total
            payload.append(
                {
                    "percentage": round(pct * 100.0, 4),
                    "proportion": round(pct, 6),
                    "relative_value": round(pct * 1.0, 6),
                    "absolute_value": round(pct * absolute_total, 4) if absolute_total is not None else -1.0,
                }
            )
        return payload

