from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

from PIL import Image


@dataclass(frozen=True)
class OCRResult:
    text: str
    confidence: float
    bounding_box: tuple[int, int, int, int] | None = None
    rotation: float = 0.0
    source: str = "unknown"
    metadata: dict[str, Any] | None = None


class OCRBackend(ABC):
    @abstractmethod
    def recognize(self, image: Image.Image, *, metadata: dict[str, Any] | None = None) -> list[OCRResult]:
        raise NotImplementedError

