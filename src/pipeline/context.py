from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from PIL import Image

from src.detection.runtime.prediction_record import DetectionPrediction
from src.detection.runtime.cropper import ChartCrop
from src.router.result import RoutingResult
from src.parser_engine.parser_result import ParserResult


@dataclass
class PipelineContext:
    document: str | Path | None = None
    page: Image.Image | None = None
    page_path: Path | None = None
    detections: list[DetectionPrediction] = field(default_factory=list)
    raw_detection_inputs: list[Any] = field(default_factory=list)
    classifier_outputs: list[dict[str, Any]] = field(default_factory=list)
    routing_outputs: list[RoutingResult] = field(default_factory=list)
    parser_outputs: list[ParserResult] = field(default_factory=list)
    crops: list[ChartCrop] = field(default_factory=list)
    timings: dict[str, float] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
