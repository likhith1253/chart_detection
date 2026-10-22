from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class PipelineConfiguration:
    output_dir: Path = Path("results/pipeline")
    save_crops: bool = True
    save_debug_images: bool = True
    save_json: bool = True
    save_logs: bool = True
    save_metrics: bool = True
    classifier_confidence_threshold: float = 0.0
    routing_confidence_threshold: float = 0.5
    extra: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        if not 0.0 <= self.classifier_confidence_threshold <= 1.0:
            raise ValueError("classifier_confidence_threshold must be in [0, 1]")
        if not 0.0 <= self.routing_confidence_threshold <= 1.0:
            raise ValueError("routing_confidence_threshold must be in [0, 1]")

