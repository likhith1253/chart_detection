from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from src.detection.runtime.prediction_record import DetectionPrediction
from src.parser_engine.parser_result import ParserResult


@dataclass
class PipelineResult:
    detected_charts: list[DetectionPrediction] = field(default_factory=list)
    classified_charts: list[dict[str, Any]] = field(default_factory=list)
    parser_outputs: list[ParserResult] = field(default_factory=list)
    structured_json: dict[str, Any] = field(default_factory=dict)
    integration_report: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    runtime_metrics: dict[str, Any] = field(default_factory=dict)
