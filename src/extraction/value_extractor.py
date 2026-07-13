"""
Value extractor for chart images.
Extracts approximate numeric values from bar, line, scatter, and pie charts.
Uses OpenCV contour analysis and segmentation results.
"""

import cv2
import numpy as np
import logging
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class ValueExtractor:
    """Extracts numeric values from chart images using OpenCV heuristics."""

    MIN_BAR_AREA = 300
    MIN_ASPECT_RATIO = 0.8

    def extract_values(self, image_path: str, chart_type: str,
                       labels: Optional[List[str]] = None) -> Dict:
        """
        Dispatcher: extracts values based on chart type.

        Args:
            image_path: Path to chart image.
            chart_type: One of bar_chart, histogram, line_chart, scatter_plot, pie_chart.
            labels: Optional category labels from OCR.

        Returns:
            Dictionary of extracted data appropriate to chart type.
        """
        try:
            if chart_type in ("bar_chart", "histogram"):
                return self.extract_bar_values(image_path, labels)
            elif chart_type == "line_chart":
                return self.extract_line_values(image_path)
            elif chart_type == "scatter_plot":
                return self.extract_scatter_values(image_path)
            elif chart_type == "pie_chart":
                return self.extract_pie_values(image_path)
            else:
                return {}
        except Exception as e:
            logger.error(f"Value extraction failed for {image_path}: {e}")
            return {}

    # ──────────────────────────────────────────────────────────────
    # Bar Chart Value Extraction
    # ──────────────────────────────────────────────────────────────
    def extract_bar_values(self, image_path: str,
                           labels: Optional[List[str]] = None) -> Dict[str, float]:
        """
        Extracts bar values from a bar chart image.
        Detects bar tops relative to baseline axis.
        """
        try:
            img = cv2.imread(image_path)
            if img is None:
                return {}

            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            h_img, w_img = gray.shape[:2]

            from src.segmentation.chart_element_segmenter import ChartElementSegmenter
            segmenter = ChartElementSegmenter()
            segmented = segmenter.segment_elements(image_path)

            bars = segmented.get("bars", [])
            axes = segmented.get("axes", [])

            if not bars:
                return {}

            bars.sort(key=lambda b: b[0])
            values = self._compute_values(bars, h_img, axes)
            result = self._map_labels(values, labels)
            return result

        except Exception as e:
            logger.error(f"Bar value extraction failed for {image_path}: {e}")
            return {}

    # ──────────────────────────────────────────────────────────────
    # Line Chart Value Extraction
    # ──────────────────────────────────────────────────────────────
    def extract_line_values(self, image_path: str) -> Dict:
        """
        Extracts line inflection points from a line chart.
        Returns detected y-values at sampled x-positions.
        """
        try:
            img = cv2.imread(image_path)
            if img is None:
                return {}

            h, w = img.shape[:2]
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            edges = cv2.Canny(blurred, 30, 100)

            # Sample x positions across chart area (avoiding margins)
            margin_x = int(w * 0.15)
            margin_y = int(h * 0.1)
            sample_xs = np.linspace(margin_x, w - margin_x, 20).astype(int)

            data_points = []
            for sx in sample_xs:
                col = edges[margin_y:h - margin_y, sx]
                ys = np.where(col > 0)[0]
                if len(ys) > 0:
                    # Take the median edge position as the data point
                    y_val = int(np.median(ys)) + margin_y
                    # Normalize to 0-100 scale (invert since higher pixel = lower value)
                    norm_val = round((1.0 - y_val / h) * 100, 1)
                    data_points.append({
                        "x_pixel": int(sx),
                        "y_pixel": y_val,
                        "normalized_value": norm_val
                    })

            return {
                "type": "line_chart",
                "data_points": data_points,
                "num_points": len(data_points)
            }

        except Exception as e:
            logger.error(f"Line value extraction failed: {e}")
            return {}

    # ──────────────────────────────────────────────────────────────
    # Scatter Plot Value Extraction
    # ──────────────────────────────────────────────────────────────
    def extract_scatter_values(self, image_path: str) -> Dict:
        """
        Extracts point coordinates from a scatter plot.
        """
        try:
            img = cv2.imread(image_path)
            if img is None:
                return {}

            h, w = img.shape[:2]
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            _, thresh = cv2.threshold(gray, 220, 255, cv2.THRESH_BINARY_INV)
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            clean = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
            contours, _ = cv2.findContours(clean, cv2.RETR_EXTERNAL,
                                           cv2.CHAIN_APPROX_SIMPLE)

            points = []
            for c in contours:
                area = cv2.contourArea(c)
                if 10 < area < 1500:
                    peri = cv2.arcLength(c, True)
                    if peri > 0:
                        circularity = 4 * np.pi * area / (peri * peri)
                        if circularity > 0.5:
                            M = cv2.moments(c)
                            if M["m00"] != 0:
                                cx = int(M["m10"] / M["m00"])
                                cy = int(M["m01"] / M["m00"])
                                # Normalize coordinates to 0-100
                                norm_x = round(cx / w * 100, 1)
                                norm_y = round((1.0 - cy / h) * 100, 1)
                                points.append({
                                    "x_pixel": cx,
                                    "y_pixel": cy,
                                    "x_normalized": norm_x,
                                    "y_normalized": norm_y
                                })

            return {
                "type": "scatter_plot",
                "points": points,
                "num_points": len(points)
            }

        except Exception as e:
            logger.error(f"Scatter value extraction failed: {e}")
            return {}

    # ──────────────────────────────────────────────────────────────
    # Pie Chart Value Extraction
    # ──────────────────────────────────────────────────────────────
    def extract_pie_values(self, image_path: str) -> Dict:
        """
        Estimates pie slice angles from a pie chart image.
        """
        try:
            img = cv2.imread(image_path)
            if img is None:
                return {}

            h, w = img.shape[:2]
            img_area = h * w

            # Use HSV to separate colored wedges
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            sat = hsv[:, :, 1]
            color_mask = cv2.inRange(sat, 25, 255)
            kernel = np.ones((5, 5), np.uint8)
            clean = cv2.morphologyEx(color_mask, cv2.MORPH_OPEN, kernel)

            contours, _ = cv2.findContours(clean, cv2.RETR_EXTERNAL,
                                           cv2.CHAIN_APPROX_SIMPLE)

            # Find the center of the pie (largest colored region centroid)
            slices = []
            total_area = 0

            for c in contours:
                area = cv2.contourArea(c)
                if area > img_area * 0.01 and area < img_area * 0.8:
                    slices.append({"contour": c, "area": area})
                    total_area += area

            if not slices or total_area == 0:
                return {}

            # Estimate angles proportionally to area
            result_slices = []
            for i, s in enumerate(slices):
                angle = round(s["area"] / total_area * 360, 1)
                pct = round(s["area"] / total_area * 100, 1)
                result_slices.append({
                    "slice_index": i,
                    "estimated_angle_deg": angle,
                    "estimated_percentage": pct,
                    "area_pixels": int(s["area"])
                })

            return {
                "type": "pie_chart",
                "slices": result_slices,
                "num_slices": len(result_slices)
            }

        except Exception as e:
            logger.error(f"Pie value extraction failed: {e}")
            return {}

    # ──────────────────────────────────────────────────────────────
    # Internal helpers
    # ──────────────────────────────────────────────────────────────
    def _compute_values(self, bars: List[Tuple], img_height: int,
                        axes: List[List[int]] = None) -> List[float]:
        """Computes normalized values for each bar based on height."""
        if not bars:
            return []

        baseline = None
        if axes:
            horizontal_axes = [line for line in axes if abs(line[1] - line[3]) < 20]
            if horizontal_axes:
                baseline = max(max(l[1], l[3]) for l in horizontal_axes)

        if baseline is None:
            baseline = max(b[1] + b[3] for b in bars)

        top_line = min(b[1] for b in bars)
        chart_height = baseline - top_line
        if chart_height <= 0:
            chart_height = 1

        values = []
        for (x, y, w, h) in bars:
            normalized = (h / chart_height) * 100.0
            values.append(round(normalized, 1))

        return values

    def _map_labels(self, values: List[float],
                    labels: Optional[List[str]] = None) -> Dict[str, float]:
        """Maps extracted values to labels."""
        result = {}
        if labels and len(labels) >= len(values):
            for i, val in enumerate(values):
                result[labels[i]] = val
        else:
            for i, val in enumerate(values):
                label = f"Bar_{i + 1}"
                if labels and i < len(labels):
                    label = labels[i]
                result[label] = val
        return result
