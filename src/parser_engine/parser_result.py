from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class ParserResult:
    success: bool
    chart_type: str
    confidence: float
    structured_data: Any
    warnings: list[str] = field(default_factory=list)
    execution_time: float = 0.0
    provenance: dict[str, Any] = field(default_factory=dict)

