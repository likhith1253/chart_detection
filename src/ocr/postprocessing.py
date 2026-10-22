from __future__ import annotations

import re
from dataclasses import replace

from .interfaces import OCRResult


class OCRPostprocessor:
    def filter_confidence(self, results: list[OCRResult], threshold: float) -> list[OCRResult]:
        return [r for r in results if r.confidence >= threshold]

    def deduplicate(self, results: list[OCRResult]) -> list[OCRResult]:
        deduped: list[OCRResult] = []
        seen: set[tuple[str, tuple[int, int, int, int] | None]] = set()
        for result in results:
            key = (result.text.strip().lower(), result.bounding_box)
            if key in seen:
                continue
            seen.add(key)
            deduped.append(result)
        return deduped

    def merge_lines(self, results: list[OCRResult]) -> list[OCRResult]:
        if not results:
            return []
        merged: list[OCRResult] = []
        for result in sorted(results, key=lambda r: (r.bounding_box[1] if r.bounding_box else 0, r.bounding_box[0] if r.bounding_box else 0)):
            if merged and self._same_line(merged[-1], result):
                prev = merged[-1]
                merged[-1] = replace(prev, text=f"{prev.text} {result.text}".strip(), confidence=max(prev.confidence, result.confidence))
            else:
                merged.append(result)
        return merged

    def normalize_numeric(self, results: list[OCRResult]) -> list[OCRResult]:
        out: list[OCRResult] = []
        for r in results:
            text = r.text.strip()
            if re.fullmatch(r"[\d\.,:%$OoIlSB\-+]+", text):
                text = text.translate(str.maketrans({"O": "0", "o": "0", "I": "1", "l": "1", "S": "5", "B": "8"}))
            out.append(replace(r, text=text))
        return out

    def normalize_axis_ticks(self, results: list[OCRResult]) -> list[OCRResult]:
        return [replace(r, text=re.sub(r"\s+", "", r.text)) for r in results]

    def postprocess(self, results: list[OCRResult], confidence_threshold: float = 0.3) -> list[OCRResult]:
        results = self.filter_confidence(results, confidence_threshold)
        results = self.deduplicate(results)
        results = self.merge_lines(results)
        results = self.normalize_numeric(results)
        return self.normalize_axis_ticks(results)

    @staticmethod
    def _same_line(a: OCRResult, b: OCRResult) -> bool:
        if a.bounding_box is None or b.bounding_box is None:
            return False
        return abs(a.bounding_box[1] - b.bounding_box[1]) <= 8
