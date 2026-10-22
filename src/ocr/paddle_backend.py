from __future__ import annotations

import logging
import os
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from .configuration import OCRBackendConfig
from .ocr_cache import OCRCache
from .interfaces import OCRBackend, OCRResult
from .postprocessing import OCRPostprocessor
from .preprocessing import OCRPreprocessor

logger = logging.getLogger(__name__)
os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")


class PaddleOCRBackend(OCRBackend):
    def __init__(self, config: OCRBackendConfig | None = None, cache: OCRCache | None = None, preprocessor: OCRPreprocessor | None = None, postprocessor: OCRPostprocessor | None = None) -> None:
        self.config = config or OCRBackendConfig()
        self.cache = cache or OCRCache()
        self.preprocessor = preprocessor or OCRPreprocessor()
        self.postprocessor = postprocessor or OCRPostprocessor()
        self._reader = None
        self._load_model()

    def _load_model(self) -> None:
        try:
            from paddleocr import PaddleOCR

            use_gpu = self.config.device == "gpu"
            self._reader = PaddleOCR(lang=self.config.language, use_angle_cls=True, use_gpu=use_gpu, show_log=False)
        except Exception as exc:
            logger.warning("PaddleOCR backend initialization failed: %s", exc)
            self._reader = None

    def recognize(self, image: Image.Image, *, metadata: dict[str, Any] | None = None) -> list[OCRResult]:
        metadata = dict(metadata or {})
        key = None
        if self.config.cache_enabled:
            key = self._cache_key(image, metadata)
            cached = self.cache.get(key)
            if cached is not None:
                return [OCRResult(**item) for item in cached]
        arr = np.asarray(image.convert("RGB"))
        if self.config.preprocess_enabled:
            arr = self.preprocessor.process(arr, size=None)
        outputs = self._run(arr)
        if self.config.postprocess_enabled:
            outputs = self.postprocessor.postprocess(outputs, confidence_threshold=self.config.confidence_threshold)
        if key is not None:
            self.cache.set(key, [self._serialize_result(r) for r in outputs])
        return outputs

    def recognize_batch(self, images: list[Image.Image], *, metadata: dict[str, Any] | None = None) -> list[list[OCRResult]]:
        return [self.recognize(image, metadata=metadata) for image in images]

    def _run(self, arr: np.ndarray) -> list[OCRResult]:
        if self._reader is None:
            return []
        try:
            result = self._reader.ocr(arr, cls=self.config.enable_rotation)
        except Exception:
            try:
                result = self._reader.ocr(np.ascontiguousarray(arr))
            except Exception as exc:
                logger.warning("PaddleOCR inference failed: %s", exc)
                return []
        return self._parse_result(result)

    def _parse_result(self, result: Any) -> list[OCRResult]:
        parsed: list[OCRResult] = []
        if not isinstance(result, list):
            return parsed
        for block in result:
            if not isinstance(block, list):
                continue
            for item in block:
                if not isinstance(item, list) or len(item) < 2:
                    continue
                bbox, (text, conf) = item[0], item[1]
                pts = np.asarray(bbox, dtype=float)
                x1, y1, x2, y2 = int(pts[:, 0].min()), int(pts[:, 1].min()), int(pts[:, 0].max()), int(pts[:, 1].max())
                parsed.append(OCRResult(text=str(text), confidence=float(conf), bounding_box=(x1, y1, x2, y2), source="paddleocr"))
        return parsed

    def _cache_key(self, image: Image.Image, metadata: dict[str, Any]) -> str:
        import hashlib

        digest = hashlib.sha256(image.tobytes())
        digest.update(str(image.size).encode("utf-8"))
        digest.update(self.config.language.encode("utf-8"))
        digest.update(str(metadata).encode("utf-8"))
        return digest.hexdigest()

    @staticmethod
    def _serialize_result(result: OCRResult) -> dict[str, Any]:
        return {
            "text": result.text,
            "confidence": result.confidence,
            "bounding_box": list(result.bounding_box) if result.bounding_box else None,
            "rotation": result.rotation,
            "source": result.source,
            "metadata": result.metadata,
        }
