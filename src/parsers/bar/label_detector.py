from __future__ import annotations

from dataclasses import dataclass

from PIL import Image


@dataclass(frozen=True)
class CategoryLabel:
    text: str
    index: int
    bbox: tuple[int, int, int, int] | None = None
    confidence: float = 0.0


class LabelDetector:
    def detect_title(self, image: Image.Image) -> str:
        return ""

    def detect_categories(self, image: Image.Image, bar_count: int) -> list[CategoryLabel]:
        return [CategoryLabel(text=f"category_{i+1}", index=i, confidence=0.0) for i in range(bar_count)]

    def detect_axis_labels(self, image: Image.Image) -> dict[str, str]:
        return {"x_axis_label": "", "y_axis_label": "", "title": ""}

    def detect_legend_labels(self, image: Image.Image) -> list[str]:
        return []

