from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class LegendText:
    text: str
    confidence: float = 0.0

