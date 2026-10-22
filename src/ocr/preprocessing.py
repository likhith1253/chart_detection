from __future__ import annotations

import cv2
import numpy as np
from PIL import Image


class OCRPreprocessor:
    def adaptive_threshold(self, image: np.ndarray) -> np.ndarray:
        gray = self._gray(image)
        return cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 21, 8)

    def denoise(self, image: np.ndarray) -> np.ndarray:
        return cv2.medianBlur(image, 3) if image.ndim == 2 else cv2.bilateralFilter(image, 5, 50, 50)

    def deskew(self, image: np.ndarray) -> np.ndarray:
        gray = self._gray(image)
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
        return cv2.warpAffine(image, matrix, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

    def enhance_contrast(self, image: np.ndarray) -> np.ndarray:
        gray = self._gray(image)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        return clahe.apply(gray)

    def resize(self, image: np.ndarray, width: int, height: int) -> np.ndarray:
        return cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)

    def process(self, image: np.ndarray, size: tuple[int, int] | None = None) -> np.ndarray:
        work = image.copy()
        if size:
            work = self.resize(work, size[0], size[1])
        work = self.enhance_contrast(work)
        work = self.denoise(work)
        work = self.adaptive_threshold(work)
        work = self.deskew(work)
        return work

    def process_image(self, image: Image.Image) -> np.ndarray:
        return self.process(np.asarray(image.convert("RGB")))

    @staticmethod
    def _gray(image: np.ndarray) -> np.ndarray:
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image

