from __future__ import annotations

from dataclasses import dataclass, field
from time import perf_counter
from typing import Any

import tracemalloc


@dataclass
class PipelineMetrics:
    stage_timings: dict[str, float] = field(default_factory=dict)
    detected_charts: int = 0
    parsed_charts: int = 0
    total_runs: int = 0
    successful_runs: int = 0
    _started: float = field(default_factory=perf_counter, init=False, repr=False)

    def __post_init__(self) -> None:
        if not tracemalloc.is_tracing():
            tracemalloc.start()

    def record_stage(self, name: str, elapsed: float) -> None:
        self.stage_timings[name] = self.stage_timings.get(name, 0.0) + float(elapsed)

    def snapshot(self) -> dict[str, Any]:
        current, peak = tracemalloc.get_traced_memory()
        total_runtime = perf_counter() - self._started
        success_rate = (self.successful_runs / self.total_runs) if self.total_runs else 0.0
        return {
            "runtime_sec": total_runtime,
            "memory_current_bytes": current,
            "memory_peak_bytes": peak,
            "stage_timings": dict(self.stage_timings),
            "detected_charts": self.detected_charts,
            "parsed_charts": self.parsed_charts,
            "total_runs": self.total_runs,
            "successful_runs": self.successful_runs,
            "success_rate": success_rate,
        }

