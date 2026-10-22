from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class AxisTick:
    text: str
    value: float | str
    confidence: float
    bounding_box: tuple[int, int, int, int] | None = None
    rotation: float = 0.0
    source: str = "unknown"
    is_numeric: bool = False


@dataclass(frozen=True)
class TickParseResult:
    ticks: list[AxisTick] = field(default_factory=list)
    numeric_ticks: list[AxisTick] = field(default_factory=list)
    categorical_ticks: list[AxisTick] = field(default_factory=list)
    ordered: bool = True
    numeric: bool = False
    axis: str = "y"
    metadata: dict[str, Any] = field(default_factory=dict)
