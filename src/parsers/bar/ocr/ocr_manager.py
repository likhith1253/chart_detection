from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from PIL import Image

from .ocr_interface import OCRBackend, OCRResult
from .text_cleaner import TextCleaner
from .title_text import TitleText
from .legend_text import LegendText
from .axis_text import AxisTick


@dataclass
class OCRManager:
    backend: OCRBackend | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    cleaner: TextCleaner = field(default_factory=TextCleaner)

    def recognize(self, image: Image.Image, *, metadata: dict[str, Any] | None = None) -> list[OCRResult]:
        if self.backend is None:
            from src.ocr import get_ocr_backend

            self.backend = get_ocr_backend()
        if self.backend is None:
            return []
        merged = {**self.metadata, **dict(metadata or {})}
        return list(self.backend.recognize(image, metadata=merged))

    def extract_title(self, image: Image.Image, *, metadata: dict[str, Any] | None = None) -> TitleText:
        results = self.recognize(image, metadata=metadata)
        if not results:
            return TitleText(text="", confidence=0.0)
        best = max(results, key=lambda r: r.confidence)
        return TitleText(text=self.cleaner.clean(best.text), confidence=float(best.confidence))

    def extract_legend(self, image: Image.Image, *, metadata: dict[str, Any] | None = None) -> list[LegendText]:
        return [LegendText(text=self.cleaner.clean(r.text), confidence=float(r.confidence)) for r in self.recognize(image, metadata=metadata) if self.cleaner.clean(r.text)]

    def extract_ticks(self, image: Image.Image, *, axis: str = "y", metadata: dict[str, Any] | None = None):
        from .tick_parser import TickParser

        return TickParser().parse(self.recognize(image, metadata=metadata), axis=axis)
