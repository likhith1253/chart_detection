from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import numpy as np

from src.ocr.ocr_engine import OCREngine
from src.pipeline.config import OCRConfig, PreprocessingConfig
from src.preprocessing.image_preprocessor import ImagePreprocessor
from src.segmentation.chart_element_segmenter import ChartElementSegmenter


@dataclass
class FeatureVector:
    feature_names: List[str]
    feature_values: List[float]
    metadata: Dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> Dict[str, Any]:
        return dict(zip(self.feature_names, self.feature_values))


class Milestone2Pipeline:
    def __init__(
        self,
        preprocessing_config: Optional[PreprocessingConfig] = None,
        ocr_config: Optional[OCRConfig] = None,
    ) -> None:
        self.preprocessor = ImagePreprocessor(preprocessing_config or PreprocessingConfig())
        self.segmenter = ChartElementSegmenter()
        self.ocr = OCREngine(config=ocr_config or OCRConfig(), preprocessor=self.preprocessor)

    def preprocess(self, image: np.ndarray) -> np.ndarray:
        return self.preprocessor.process(image)

    def segment(self, image_path: str) -> Dict[str, List]:
        return self.segmenter.segment_elements(image_path)

    def extract_features(self, image_path: str) -> FeatureVector:
        image = cv2.imread(image_path)
        if image is None:
            raise FileNotFoundError(image_path)
        processed = self.preprocess(image)
        seg = self.segmenter.segment_elements(image=image)
        ocr_result = self.ocr.run_ocr(image_path)
        features = self._build_features(image, processed, seg, ocr_result)
        names = list(features.keys())
        return FeatureVector(
            feature_names=names,
            feature_values=[float(features[name]) for name in names],
            metadata={
                "image_path": str(Path(image_path)),
                "preprocessing": self.preprocessor.config.__dict__,
                "segmentation_counts": {k: len(v) if isinstance(v, list) else v for k, v in seg.items()},
                "ocr_enabled": bool(self.ocr.easyocr_reader is not None or self.ocr.paddleocr_reader is not None),
            },
        )

    def _build_features(
        self,
        image: np.ndarray,
        processed: np.ndarray,
        seg: Dict[str, List],
        ocr: Dict[str, Any],
    ) -> Dict[str, float]:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
        h, w = gray.shape[:2]
        area = max(1.0, float(h * w))
        edges = cv2.Canny(gray, 50, 150)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        bars = seg.get("bars", [])
        lines = seg.get("lines", [])
        points = seg.get("points", [])
        axes = seg.get("axes", [])
        circles = seg.get("circles_hough", [])
        slices = seg.get("pie_slices", [])
        text_tokens = ocr.get("cleaned_text", [])

        feat = {
            "image_width": float(w),
            "image_height": float(h),
            "aspect_ratio": float(w / max(1.0, h)),
            "gray_mean": float(np.mean(gray)) / 255.0,
            "gray_std": float(np.std(gray)) / 255.0,
            "edge_density": float(np.count_nonzero(edges)) / area,
            "contour_count": float(len(contours)),
            "axis_count": float(len(axes)),
            "bar_count": float(len(bars)),
            "line_count": float(len(lines)),
            "point_count": float(len(points)),
            "circle_count": float(len(circles)),
            "pie_slice_count": float(len(slices)),
            "ocr_token_count": float(len(text_tokens)),
            "ocr_confidence": float(ocr.get("confidence", 0.0)),
            "ocr_text_length": float(sum(len(t) for t in text_tokens)),
        }
        feat.update(self._shape_features(gray, bars, lines, points, circles, slices, axes))
        feat.update(self._histogram_features(gray, bars, axes))
        feat.update(self._ocr_shape_features(text_tokens))
        feat.update(self._image_texture_features(processed))
        return feat

    def _shape_features(self, gray, bars, lines, points, circles, slices, axes):
        h, w = gray.shape[:2]
        bar_heights = [b[3] for b in bars] or [0]
        bar_widths = [b[2] for b in bars] or [0]
        line_lengths = [float((ln[2] - ln[0]) ** 2 + (ln[3] - ln[1]) ** 2) ** 0.5 for ln in lines] or [0.0]
        point_density = float(len(points)) / max(1.0, h * w)
        return {
            "bar_height_mean": float(np.mean(bar_heights)),
            "bar_width_mean": float(np.mean(bar_widths)),
            "bar_width_var": float(np.var(bar_widths)),
            "bar_height_var": float(np.var(bar_heights)),
            "bar_aspect_mean": float(np.mean([b[3] / max(1.0, b[2]) for b in bars])) if bars else 0.0,
            "line_span_mean": float(np.mean(line_lengths)),
            "point_density": point_density,
            "circle_density": float(len(circles)) / max(1.0, h * w),
            "pie_slice_ratio": float(len(slices)) / max(1.0, len(circles) + len(slices)),
            "axis_horizontal_ratio": float(sum(1 for a in axes if abs(a[3] - a[1]) < abs(a[2] - a[0]))) / max(1.0, len(axes)),
            "connected_component_balance": float(len(bars) + len(points) + len(lines)) / max(1.0, len(axes) + len(circles) + len(slices)),
            "scatter_cluster_proxy": float(len(points) / max(1.0, len(lines) + 1)),
        }

    def _histogram_features(self, gray, bars, axes):
        if not bars:
            return {
                "hist_bin_count": 0.0,
                "hist_bin_width_var": 0.0,
                "hist_gap_ratio": 0.0,
                "hist_alignment_score": 0.0,
                "hist_touching_ratio": 0.0,
            }
        widths = np.array([b[2] for b in bars], dtype=float)
        xs = np.array([b[0] for b in bars], dtype=float)
        bottoms = np.array([b[1] + b[3] for b in bars], dtype=float)
        gaps = np.diff(np.sort(xs)) if len(xs) > 1 else np.array([0.0])
        touching = float(np.sum(gaps < np.mean(widths) * 0.2)) / max(1.0, len(gaps))
        return {
            "hist_bin_count": float(len(bars)),
            "hist_bin_width_var": float(np.var(widths)),
            "hist_gap_ratio": float(np.mean(gaps) / (np.mean(widths) + 1e-6)),
            "hist_alignment_score": float(1.0 - min(1.0, np.std(bottoms) / max(1.0, gray.shape[0]))),
            "hist_touching_ratio": touching,
        }

    def _ocr_shape_features(self, tokens: List[str]) -> Dict[str, float]:
        numeric = sum(1 for t in tokens if any(ch.isdigit() for ch in t))
        alpha = sum(1 for t in tokens if any(ch.isalpha() for ch in t))
        return {
            "ocr_numeric_token_ratio": float(numeric) / max(1.0, len(tokens)),
            "ocr_alpha_token_ratio": float(alpha) / max(1.0, len(tokens)),
            "ocr_unique_token_ratio": float(len(set(tokens))) / max(1.0, len(tokens)),
        }

    def _image_texture_features(self, processed: np.ndarray) -> Dict[str, float]:
        gray = processed if processed.ndim == 2 else cv2.cvtColor(processed, cv2.COLOR_BGR2GRAY)
        hist = cv2.calcHist([gray], [0], None, [32], [0, 256]).flatten()
        hist = hist / hist.sum() if hist.sum() else hist
        entropy = float(-(hist * np.log2(hist + 1e-9)).sum())
        return {
            "processed_mean": float(np.mean(gray)) / 255.0,
            "processed_std": float(np.std(gray)) / 255.0,
            "processed_entropy": entropy,
            "processed_nonzero_ratio": float(np.count_nonzero(gray)) / max(1.0, gray.size),
        }
