from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .axis_text import AxisTick, TickParseResult
from .ocr_interface import OCRResult
from .text_cleaner import TextCleaner


@dataclass
class TickParser:
    cleaner: TextCleaner = TextCleaner()

    def parse(self, results: list[OCRResult], axis: str = "y") -> TickParseResult:
        ticks: list[AxisTick] = []
        numeric_ticks: list[AxisTick] = []
        categorical_ticks: list[AxisTick] = []
        for res in results:
            cleaned = self.cleaner.normalize_numeric(res.text)
            numeric = self._to_number(cleaned)
            tick = AxisTick(
                text=self.cleaner.clean(res.text),
                value=numeric if numeric is not None else self.cleaner.clean(res.text),
                confidence=float(res.confidence),
                bounding_box=res.bounding_box,
                rotation=float(res.rotation),
                source=res.source,
                is_numeric=numeric is not None,
            )
            if numeric is not None:
                numeric_ticks.append(tick)
            else:
                categorical_ticks.append(tick)
            ticks.append(tick)
        ordered = self._is_ordered(ticks)
        return TickParseResult(
            ticks=ticks,
            numeric_ticks=numeric_ticks,
            categorical_ticks=categorical_ticks,
            ordered=ordered,
            numeric=bool(numeric_ticks),
            axis=axis,
            metadata={"count": len(ticks), "numeric_count": len(numeric_ticks), "categorical_count": len(categorical_ticks)},
        )

    def _to_number(self, text: str) -> float | None:
        try:
            return float(text)
        except Exception:
            return None

    def _is_ordered(self, ticks: list[AxisTick]) -> bool:
        numeric_values = [t.value for t in ticks if isinstance(t.value, (int, float))]
        return numeric_values == sorted(numeric_values) if numeric_values else True
