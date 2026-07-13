"""
Runtime stage profiling utilities.
"""

from __future__ import annotations

import json
import threading
import time
from collections import defaultdict
from contextlib import contextmanager
from pathlib import Path
from typing import Dict, Iterable, Iterator


class RuntimeProfiler:
    """Collects timing statistics per stage across all images."""

    def __init__(self, stage_names: Iterable[str]):
        self.stage_names = list(stage_names)
        self._totals = defaultdict(float)
        self._counts = defaultdict(int)
        self._lock = threading.Lock()

    @contextmanager
    def track(self, stage: str) -> Iterator[None]:
        start = time.perf_counter()
        try:
            yield
        finally:
            duration = time.perf_counter() - start
            self.record(stage, duration)

    def record(self, stage: str, seconds: float) -> None:
        with self._lock:
            self._totals[stage] += float(seconds)
            self._counts[stage] += 1

    def summarize(self) -> Dict[str, Dict[str, float]]:
        with self._lock:
            summary: Dict[str, Dict[str, float]] = {}
            for stage in sorted(set(self.stage_names + list(self._totals.keys()))):
                total = float(self._totals.get(stage, 0.0))
                count = int(self._counts.get(stage, 0))
                summary[stage] = {
                    "total_seconds": total,
                    "count": count,
                    "mean_seconds": total / count if count else 0.0,
                }
            summary["overall"] = {
                "total_seconds": float(sum(self._totals.values())),
                "stage_count": len(self._totals),
            }
            return summary

    def save(self, path: str | Path) -> Path:
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(self.summarize(), indent=2), encoding="utf-8")
        return output_path
