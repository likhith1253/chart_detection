from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .geometry import MarkerGeometry


@dataclass
class ClusterDetector:
    eps: float = 18.0
    min_samples: int = 1
    metadata: dict[str, Any] = field(default_factory=dict)

    def detect(self, markers: list[MarkerGeometry]) -> list[list[MarkerGeometry]]:
        if not markers:
            return []
        points = np.array([m.centroid for m in markers], dtype=float)
        clusters: list[list[MarkerGeometry]] = []
        used = set()
        for idx, marker in enumerate(markers):
            if idx in used:
                continue
            cluster = [marker]
            used.add(idx)
            for j in range(idx + 1, len(markers)):
                if j in used:
                    continue
                if np.linalg.norm(points[idx] - points[j]) <= self.eps:
                    cluster.append(markers[j])
                    used.add(j)
            clusters.append(cluster)
        return clusters

