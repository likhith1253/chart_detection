from __future__ import annotations

from functools import lru_cache

from .configuration import OCRBackendConfig
from .ocr_cache import OCRCache
from .paddle_backend import PaddleOCRBackend


class OCRFactory:
    def create(self, config: OCRBackendConfig | None = None):
        cfg = config or OCRBackendConfig()
        if cfg.backend.lower() == "paddleocr":
            return PaddleOCRBackend(cfg, cache=OCRCache())
        raise ValueError(f"Unsupported OCR backend: {cfg.backend}")


@lru_cache(maxsize=1)
def get_ocr_backend() -> PaddleOCRBackend:
    return PaddleOCRBackend()

