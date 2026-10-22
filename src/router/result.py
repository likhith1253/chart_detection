from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class RoutingResult:
    selected_parser: str
    candidate_parsers: list[str]
    confidence: float
    routing_reason: str
    fallback_used: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)

