from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class OCRBackendConfig:
    backend: str = "paddleocr"
    language: str = "en"
    device: str = "auto"
    batch_size: int = 1
    confidence_threshold: float = 0.3
    merge_text_boxes: bool = True
    enable_rotation: bool = True
    cache_enabled: bool = True
    preprocess_enabled: bool = True
    postprocess_enabled: bool = True
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class OCRConfiguration:
    backend: OCRBackendConfig = field(default_factory=OCRBackendConfig)
    seed: int = 42

