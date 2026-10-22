from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from PIL import Image


@dataclass(frozen=True)
class OCRHook:
    title: str = ""
    x_axis_label: str = ""
    y_axis_label: str = ""
    category_labels: list[str] | None = None
    legend_labels: list[str] | None = None


class OCRProvider(Protocol):
    def extract_title(self, image: Image.Image) -> str: ...
    def extract_axis_labels(self, image: Image.Image) -> tuple[str, str]: ...
    def extract_category_labels(self, image: Image.Image, count: int) -> list[str]: ...
    def extract_legend_labels(self, image: Image.Image) -> list[str]: ...

