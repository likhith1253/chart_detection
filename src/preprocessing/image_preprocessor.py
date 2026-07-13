"""
Configurable image preprocessing stages for OCR and downstream analysis.
"""

from __future__ import annotations

import cv2
import numpy as np

from src.pipeline.config import PreprocessingConfig


class ImagePreprocessor:
    """Applies preprocessing steps controlled by YAML configuration."""

    def __init__(self, config: PreprocessingConfig):
        self.config = config

    def process(self, image: np.ndarray) -> np.ndarray:
        if image is None:
            return image
        if not self.config.enabled:
            return image

        work = image.copy()
        target_w = int(getattr(self.config, "resize_width", 0) or 0)
        target_h = int(getattr(self.config, "resize_height", 0) or 0)
        if target_w > 0 and target_h > 0:
            work = cv2.resize(work, (target_w, target_h), interpolation=cv2.INTER_AREA)
        if self.config.grayscale and work.ndim == 3:
            work = cv2.cvtColor(work, cv2.COLOR_BGR2GRAY)

        if self.config.clahe:
            if work.ndim == 3:
                work = cv2.cvtColor(work, cv2.COLOR_BGR2GRAY)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            work = clahe.apply(work)

        if self.config.adaptive_threshold:
            if work.ndim == 3:
                work = cv2.cvtColor(work, cv2.COLOR_BGR2GRAY)
            work = cv2.adaptiveThreshold(
                work,
                255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY,
                21,
                8,
            )

        if self.config.denoise:
            if work.ndim == 2:
                work = cv2.medianBlur(work, 3)
            else:
                work = cv2.bilateralFilter(work, 5, 50, 50)

        if self.config.morphology_cleanup:
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            work = cv2.morphologyEx(work, cv2.MORPH_OPEN, kernel)
            work = cv2.morphologyEx(work, cv2.MORPH_CLOSE, kernel)

        if getattr(self.config, "edge_detection", False):
            if work.ndim == 3:
                work = cv2.cvtColor(work, cv2.COLOR_BGR2GRAY)
            work = cv2.Canny(work, 50, 150)

        if self.config.deskew:
            work = self._deskew(work)

        return work

    def process_path(self, image_path: str) -> np.ndarray | None:
        image = cv2.imread(image_path)
        if image is None:
            return None
        return self.process(image)

    @staticmethod
    def _deskew(image: np.ndarray) -> np.ndarray:
        gray = image
        if gray.ndim == 3:
            gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)

        coords = np.column_stack(np.where(gray > 0))
        if coords.size == 0:
            return image

        angle = cv2.minAreaRect(coords)[-1]
        if angle < -45:
            angle = -(90 + angle)
        else:
            angle = -angle

        if abs(angle) < 0.5 or abs(angle) > 20:
            return image

        h, w = gray.shape[:2]
        matrix = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
        return cv2.warpAffine(
            image,
            matrix,
            (w, h),
            flags=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_REPLICATE,
        )
