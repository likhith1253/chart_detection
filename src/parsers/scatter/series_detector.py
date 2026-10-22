from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .geometry import DetectedPoint, MarkerGeometry, ScatterSeries


@dataclass
class SeriesDetector:
    metadata: dict[str, Any] = field(default_factory=dict)

    def detect(self, clusters: list[list[MarkerGeometry]]) -> list[ScatterSeries]:
        series: list[ScatterSeries] = []
        for idx, cluster in enumerate(clusters):
            points = [
                DetectedPoint(x=float(marker.centroid[0]), y=float(marker.centroid[1]), confidence=marker.confidence, marker_shape=marker.shape, series_id=idx)
                for marker in cluster
            ]
            series.append(ScatterSeries(series_id=idx, markers=cluster, points=points, confidence=self._confidence(cluster), label=f"series_{idx + 1}", metadata={"cluster_size": len(cluster)}))
        return series

    def _confidence(self, cluster: list[MarkerGeometry]) -> float:
        if not cluster:
            return 0.0
        return min(0.99, sum(m.confidence for m in cluster) / len(cluster))

