from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class GroundTruthInstance:
    chart_type: str
    chart_id: str
    boxes: list[dict[str, Any]] = field(default_factory=list)
    points: list[dict[str, Any]] = field(default_factory=list)
    values: list[float] = field(default_factory=list)
    categories: list[str] = field(default_factory=list)
    series: list[dict[str, Any]] = field(default_factory=list)
    ocr_text: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

