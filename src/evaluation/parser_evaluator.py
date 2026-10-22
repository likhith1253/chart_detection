from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from .comparison import GroundTruthComparator
from .ground_truth import GroundTruthInstance
from .statistics import StatisticalAnalyzer


@dataclass
class ParserEvaluationRecord:
    parser_name: str
    chart_type: str
    chart_id: str
    metrics: dict[str, float]
    runtime_seconds: float
    memory_megabytes: float
    metadata: dict[str, Any] = field(default_factory=dict)


class ParserEvaluator:
    def __init__(self) -> None:
        self.comparator = GroundTruthComparator()
        self.stats = StatisticalAnalyzer()

    def evaluate(self, parser_name: str, chart_type: str, prediction: dict[str, Any], ground_truth: GroundTruthInstance, runtime_seconds: float = 0.0, memory_megabytes: float = 0.0) -> ParserEvaluationRecord:
        result = self.comparator.compare(parser_name, chart_type, prediction, ground_truth)
        return ParserEvaluationRecord(parser_name, chart_type, ground_truth.chart_id, result.metrics, runtime_seconds, memory_megabytes, result.details)

