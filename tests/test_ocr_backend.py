from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from src.ocr import OCRBackendConfig, OCRCache, OCRFactory, OCRPostprocessor, OCRPreprocessor, PaddleOCRBackend, get_ocr_backend
from src.parsers.bar.ocr import OCRResult, OCRManager


class StubReader:
    def __init__(self) -> None:
        self.calls = 0

    def ocr(self, arr, cls=True):
        self.calls += 1
        return [
            [
                [[[0, 0], [10, 0], [10, 10], [0, 10]], ("100", 0.98)],
                [[[12, 0], [20, 0], [20, 10], [12, 10]], ("100", 0.97)],
            ]
        ]


def test_factory_creates_backend():
    backend = OCRFactory().create(OCRBackendConfig(backend="paddleocr"))
    assert isinstance(backend, PaddleOCRBackend)
    assert get_ocr_backend() is not None


def test_preprocessing_and_postprocessing():
    pre = OCRPreprocessor()
    img = np.full((32, 32, 3), 255, dtype=np.uint8)
    processed = pre.process(img, size=(16, 16))
    assert processed.shape == (16, 16)
    post = OCRPostprocessor()
    results = [
        OCRResult(text="100", confidence=0.8, bounding_box=(0, 0, 10, 10)),
        OCRResult(text="100", confidence=0.9, bounding_box=(0, 0, 10, 10)),
        OCRResult(text="A", confidence=0.1, bounding_box=(1, 1, 2, 2)),
    ]
    cleaned = post.postprocess(results, confidence_threshold=0.3)
    assert len(cleaned) == 1
    assert cleaned[0].text == "100"


def test_backend_cache_and_manager(tmp_path: Path):
    backend = PaddleOCRBackend(OCRBackendConfig(cache_enabled=True), cache=OCRCache(tmp_path / "cache"))
    backend._reader = StubReader()
    image = Image.new("RGB", (40, 40), "white")
    first = backend.recognize(image)
    second = backend.recognize(image)
    assert first and second
    assert backend._reader.calls == 1
    mgr = OCRManager(backend=backend)
    texts = mgr.recognize(image)
    assert texts

