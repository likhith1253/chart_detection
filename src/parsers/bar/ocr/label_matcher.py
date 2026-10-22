from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ..geometry import BarGeometry
from .ocr_interface import OCRResult


@dataclass(frozen=True)
class CategoryMatch:
    label: str
    bar_index: int
    confidence: float
    source: str = "ocr"
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class CategoryMatcher:
    def match(
        self,
        bars: list[BarGeometry],
        ocr_results: list[OCRResult],
        orientation: str,
    ) -> list[CategoryMatch]:
        labels = [r.text for r in ocr_results if str(r.text).strip()]
        matches: list[CategoryMatch] = []
        if not bars:
            return matches
        if not labels:
            labels = [f"category_{i+1}" for i in range(len(bars))]
        for idx, bar in enumerate(bars):
            label = labels[idx % len(labels)]
            matches.append(CategoryMatch(label=label, bar_index=idx, confidence=0.5 if ocr_results else 0.0, metadata={"orientation": orientation}))
        return matches

