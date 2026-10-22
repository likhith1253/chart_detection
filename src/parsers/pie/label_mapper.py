from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from src.parsers.bar.ocr.ocr_interface import OCRResult

from .geometry import DetectedSlice


@dataclass(frozen=True)
class SliceLabelMatch:
    label: str
    slice_id: int
    confidence: float
    source: str = "ocr"
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class LabelMapper:
    metadata: dict[str, Any] = field(default_factory=dict)

    def map(self, slices: list[DetectedSlice], ocr_results: list[OCRResult]) -> list[SliceLabelMatch]:
        labels = [r.text for r in ocr_results if str(r.text).strip()]
        if not labels:
            labels = [f"slice_{i+1}" for i in range(len(slices))]
        matches: list[SliceLabelMatch] = []
        for idx, s in enumerate(slices):
            matches.append(SliceLabelMatch(label=labels[idx % len(labels)], slice_id=idx, confidence=0.5 if ocr_results else 0.0, metadata={"sweep_angle": s.geometry.sweep_angle}))
        return matches

