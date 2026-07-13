"""
Debug Visualizer — overlays detected chart elements on images.
Draws bounding boxes for bars, points, axes, and pie slices.
Saves annotated images to results/debug/.
"""

import cv2
import numpy as np
import logging
from pathlib import Path
from typing import Dict, List

logger = logging.getLogger(__name__)


class DebugVisualizer:
    """Overlays segmentation results on chart images for visual debugging."""

    # Colors (BGR)
    COLOR_BAR = (0, 255, 0)       # Green
    COLOR_POINT = (255, 0, 0)     # Blue
    COLOR_AXIS = (0, 0, 255)      # Red
    COLOR_PIE = (255, 255, 0)     # Cyan
    COLOR_LINE = (0, 165, 255)    # Orange

    def __init__(self, output_dir: Path = None):
        import config
        self.output_dir = output_dir or config.DEBUG_DIR
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def save_debug_image(self, image_path: str, segmentation: Dict,
                         image_name: str = None) -> str:
        """
        Draw segmentation overlays on the image and save to debug dir.

        Args:
            image_path: Path to original image.
            segmentation: Dict from ChartElementSegmenter.segment_elements().
            image_name: Optional filename for the debug image.

        Returns:
            Path to saved debug image, or empty string on failure.
        """
        try:
            img = cv2.imread(image_path)
            if img is None:
                return ""

            overlay = img.copy()

            # ── Draw bars ──────────────────────────────────────
            for bar in segmentation.get("bars", []):
                if len(bar) == 4:
                    x, y, w, h = bar
                    cv2.rectangle(overlay, (x, y), (x + w, y + h),
                                  self.COLOR_BAR, 2)
                    cv2.putText(overlay, "bar", (x, y - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                                self.COLOR_BAR, 1)

            # ── Draw points ────────────────────────────────────
            for pt in segmentation.get("points", []):
                if len(pt) == 2:
                    cx, cy = pt
                    cv2.circle(overlay, (cx, cy), 5, self.COLOR_POINT, -1)

            # ── Draw axes ──────────────────────────────────────
            for axis in segmentation.get("axes", []):
                if len(axis) == 4:
                    x1, y1, x2, y2 = axis
                    cv2.line(overlay, (x1, y1), (x2, y2),
                             self.COLOR_AXIS, 2)
                    cv2.putText(overlay, "axis", (x1, y1 - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                                self.COLOR_AXIS, 1)

            # ── Draw pie slices ─────────────────────────────────
            for contour in segmentation.get("pie_slices", []):
                if isinstance(contour, np.ndarray):
                    cv2.drawContours(overlay, [contour], -1,
                                     self.COLOR_PIE, 2)

            # ── Draw lines ──────────────────────────────────────
            for line in segmentation.get("lines", []):
                if len(line) == 4:
                    x1, y1, x2, y2 = line
                    cv2.line(overlay, (x1, y1), (x2, y2),
                             self.COLOR_LINE, 2)

            # ── Blend and save ─────────────────────────────────
            result = cv2.addWeighted(img, 0.5, overlay, 0.5, 0)

            if image_name is None:
                image_name = Path(image_path).stem + "_debug.png"
            else:
                image_name = Path(image_name).stem + "_debug.png"

            save_path = self.output_dir / image_name
            cv2.imwrite(str(save_path), result)
            return str(save_path)

        except Exception as e:
            logger.error(f"Debug visualization failed for {image_path}: {e}")
            return ""
