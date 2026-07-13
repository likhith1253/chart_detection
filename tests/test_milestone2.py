from pathlib import Path

import cv2
import numpy as np

from src.pipeline.config import OCRConfig, PreprocessingConfig
from src.pipeline.milestone2 import Milestone2Pipeline
from src.preprocessing.image_preprocessor import ImagePreprocessor


def _make_test_image(path: Path) -> None:
    img = np.full((240, 320, 3), 255, dtype=np.uint8)
    cv2.rectangle(img, (40, 80), (75, 200), (0, 0, 0), -1)
    cv2.rectangle(img, (100, 120), (135, 200), (0, 0, 0), -1)
    cv2.line(img, (30, 210), (300, 210), (0, 0, 0), 3)
    cv2.line(img, (30, 40), (30, 210), (0, 0, 0), 3)
    cv2.circle(img, (230, 120), 18, (0, 0, 0), -1)
    cv2.imwrite(str(path), img)


def test_preprocessing_pipeline(tmp_path):
    path = tmp_path / "chart.png"
    _make_test_image(path)
    cfg = PreprocessingConfig(
        enabled=True,
        resize_width=160,
        resize_height=120,
        grayscale=True,
        clahe=False,
        adaptive_threshold=True,
        denoise=True,
        morphology_cleanup=True,
        deskew=False,
        edge_detection=True,
    )
    pre = ImagePreprocessor(cfg)
    out = pre.process_path(str(path))
    assert out is not None
    assert out.shape == (120, 160)


def test_segmentation_and_features(tmp_path):
    path = tmp_path / "chart.png"
    _make_test_image(path)
    pipe = Milestone2Pipeline(ocr_config=OCRConfig(enabled_engines=[]))
    seg = pipe.segment(str(path))
    assert isinstance(seg, dict)
    fv = pipe.extract_features(str(path))
    assert len(fv.feature_names) == len(fv.feature_values)
    assert len(fv.feature_names) >= 25
    assert "bar_count" in fv.feature_names
    assert "ocr_confidence" in fv.feature_names
    assert all(np.isfinite(v) for v in fv.feature_values)


def test_real_sample_image():
    pipe = Milestone2Pipeline(ocr_config=OCRConfig(enabled_engines=[]))
    sample = Path("data/datasets/plotqa/PlotQA-master/images/3_vbar_categorical.png")
    fv = pipe.extract_features(str(sample))
    assert len(fv.feature_names) >= 25
    assert fv.metadata["image_path"].endswith("3_vbar_categorical.png")
