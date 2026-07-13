"""
Chart Element Segmenter — Research-Grade.
Detects chart elements using:
  - Morphological filtering (open/close)
  - Hough line detection (axis separation)
  - Connected component filtering
  - Contour size thresholds
  - Circular Hough transform (pie charts)
  - DBSCAN point clustering (scatter plots)
  - Filters for tiny contours, axis lines, text boxes
"""

import cv2
import numpy as np
import logging
from typing import Dict, List, Tuple

logger = logging.getLogger(__name__)

# Min area thresholds
_MIN_CONTOUR_AREA = 200
_MIN_BAR_AREA = 400
_MIN_POINT_AREA = 15
_MAX_POINT_AREA = 1200
_MIN_PIE_FRACTION = 0.015
_MAX_PIE_FRACTION = 0.75


class ChartElementSegmenter:
    """Research-grade chart element segmenter with noise filtering."""

    def segment_elements(self, image_path: str | None = None, image: np.ndarray | None = None) -> Dict[str, List]:
        """
        Segments the chart into explicit sub-components with noise filtering.

        Returns:
            Dictionary containing bounded structures:
            {
                "plot_area": [(x, y, w, h)],
                "bars": [(x, y, w, h), ...],
                "points": [(cx, cy), ...],
                "lines": [[x1, y1, x2, y2], ...],
                "pie_slices": [contour, ...],
                "axes": [[x1, y1, x2, y2], ...],
                "circles_hough": [(cx, cy, r), ...],
                "point_clusters": [[(cx, cy), ...], ...],
            }
        """
        result = {
            "plot_area": [],
            "bars": [],
            "points": [],
            "lines": [],
            "pie_slices": [],
            "axes": [],
            "circles_hough": [],
            "point_clusters": [],
        }

        try:
            if image is not None:
                img = image.copy()
            else:
                img = cv2.imread(image_path or "")
            if img is None:
                logger.error(f"Failed to load image: {image_path}")
                return result

            h_img, w_img = img.shape[:2]
            img_area = h_img * w_img

            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)

            # ── 1. Morphological preprocessing ──────────────────
            kernel_small = np.ones((3, 3), np.uint8)
            kernel_med = np.ones((5, 5), np.uint8)

            # Adaptive threshold for structure detection
            _, thresh_otsu = cv2.threshold(
                blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
            )
            # Open to remove noise, close to fill gaps
            morph_clean = cv2.morphologyEx(thresh_otsu, cv2.MORPH_OPEN, kernel_small)
            morph_clean = cv2.morphologyEx(morph_clean, cv2.MORPH_CLOSE, kernel_med)

            # ── 2. Detect Axes via Hough Line Detection ─────────
            edges = cv2.Canny(blurred, 50, 150)
            axis_lines_h, axis_lines_v = self._detect_axes_hough(
                edges, h_img, w_img
            )

            if axis_lines_h:
                axis_lines_h.sort(key=lambda l: max(l[1], l[3]), reverse=True)
                result["axes"].append(axis_lines_h[0])
            if axis_lines_v:
                axis_lines_v.sort(key=lambda l: min(l[0], l[2]))
                result["axes"].append(axis_lines_v[0])

            # Build axis exclusion zones
            axis_zones = self._build_axis_zones(result["axes"], h_img, w_img)

            # ── 3. Detect Plot Area ─────────────────────────────
            plot_contours, _ = cv2.findContours(
                cv2.dilate(edges, kernel_small, iterations=1),
                cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            largest_rect = None
            max_area = 0
            for c in plot_contours:
                x, y, w, h = cv2.boundingRect(c)
                area = w * h
                if img_area * 0.1 < area < img_area * 0.95:
                    if area > max_area:
                        max_area = area
                        largest_rect = (x, y, w, h)

            if largest_rect:
                result["plot_area"].append(largest_rect)
            else:
                mx, my = int(w_img * 0.1), int(h_img * 0.1)
                result["plot_area"].append((mx, my, w_img - 2 * mx, h_img - 2 * my))

            # ── 4. Connected Component Filtering for Bars ───────
            cc_output = cv2.connectedComponentsWithStats(morph_clean, connectivity=8)
            n_labels, labels_map, stats, centroids = cc_output

            for i in range(1, n_labels):  # skip background
                x, y, w, h = stats[i, cv2.CC_STAT_LEFT], stats[i, cv2.CC_STAT_TOP], \
                              stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT]
                area = stats[i, cv2.CC_STAT_AREA]

                # Filter tiny contours
                if area < _MIN_BAR_AREA or area > img_area * 0.5:
                    continue

                # Filter axis-overlapping shapes
                if self._overlaps_axis(x, y, w, h, axis_zones):
                    continue

                # Filter text boxes (small height, wide)
                if w > 0 and h > 0:
                    aspect = float(h) / float(w)
                    fill_ratio = area / (w * h)

                    # Text boxes: wide and short with low fill
                    if aspect < 0.3 and fill_ratio < 0.4:
                        continue

                    # Bar-like: tall rectangles with decent fill
                    if aspect > 0.6 and fill_ratio > 0.35:
                        result["bars"].append((x, y, w, h))

            # ── 5. Scatter Points with Clustering ───────────────
            _, dot_thresh = cv2.threshold(gray, 215, 255, cv2.THRESH_BINARY_INV)
            dot_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            dot_clean = cv2.morphologyEx(dot_thresh, cv2.MORPH_OPEN, dot_kernel)

            # Filter margin areas
            margin_y = int(h_img * 0.12)
            margin_x = int(w_img * 0.1)
            dot_roi = dot_clean.copy()
            dot_roi[:margin_y, :] = 0
            dot_roi[h_img - margin_y:, :] = 0
            dot_roi[:, :margin_x] = 0
            dot_roi[:, w_img - margin_x:] = 0

            dot_contours, _ = cv2.findContours(
                dot_roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )

            raw_points = []
            for c in dot_contours:
                area = cv2.contourArea(c)
                if _MIN_POINT_AREA < area < _MAX_POINT_AREA:
                    peri = cv2.arcLength(c, True)
                    if peri > 0:
                        circularity = 4 * np.pi * area / (peri * peri)
                        if circularity > 0.55:
                            M = cv2.moments(c)
                            if M["m00"] != 0:
                                cx = int(M["m10"] / M["m00"])
                                cy = int(M["m01"] / M["m00"])
                                # Skip if on axis
                                if not self._overlaps_axis(cx - 2, cy - 2, 4, 4, axis_zones):
                                    raw_points.append((cx, cy))

            result["points"] = raw_points

            # DBSCAN-style clustering for scatter points
            if len(raw_points) > 3:
                result["point_clusters"] = self._cluster_points(
                    raw_points, eps=max(w_img, h_img) * 0.05
                )

            # ── 6. Circular Hough Transform for Pie Charts ──────
            gray_blur = cv2.GaussianBlur(gray, (9, 9), 2)
            circles = cv2.HoughCircles(
                gray_blur, cv2.HOUGH_GRADIENT,
                dp=1.2,
                minDist=int(min(h_img, w_img) * 0.3),
                param1=100, param2=50,
                minRadius=int(min(h_img, w_img) * 0.1),
                maxRadius=int(min(h_img, w_img) * 0.48)
            )
            if circles is not None:
                for c in np.round(circles[0]).astype(int):
                    result["circles_hough"].append((int(c[0]), int(c[1]), int(c[2])))

            # ── 7. Pie Slice Detection (color-based) ────────────
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            sat = hsv[:, :, 1]
            val = hsv[:, :, 2]
            color_mask = cv2.inRange(sat, 25, 255)
            bright_mask = cv2.inRange(val, 50, 255)
            wedge_mask = cv2.bitwise_and(color_mask, bright_mask)
            wedge_clean = cv2.morphologyEx(wedge_mask, cv2.MORPH_OPEN, kernel_med)
            wedge_clean = cv2.morphologyEx(wedge_clean, cv2.MORPH_CLOSE, kernel_med)

            pie_contours, _ = cv2.findContours(
                wedge_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )

            for c in pie_contours:
                area = cv2.contourArea(c)
                if area > img_area * _MIN_PIE_FRACTION and area < img_area * _MAX_PIE_FRACTION:
                    x, y, w, h = cv2.boundingRect(c)
                    fill_ratio = area / (w * h) if w * h > 0 else 0
                    hull = cv2.convexHull(c)
                    hull_area = cv2.contourArea(hull)
                    solidity = float(area) / hull_area if hull_area > 0 else 0
                    if 0.25 < fill_ratio < 0.95 and solidity > 0.75:
                        result["pie_slices"].append(c)

            # ── 8. Chart Lines (filtered: no axis lines) ────────
            line_edges = cv2.Canny(blurred, 30, 100)
            chart_lines = cv2.HoughLinesP(
                line_edges, 1, np.pi / 180,
                threshold=40,
                minLineLength=int(w_img * 0.05),
                maxLineGap=15
            )
            if chart_lines is not None:
                for line in chart_lines:
                    x1, y1, x2, y2 = line[0]
                    angle = abs(np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi)
                    length = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
                    # Diagonal lines only (not axis-aligned), reasonably long
                    if (10 < angle < 80 or 100 < angle < 170) and length > w_img * 0.05:
                        # Check not overlapping axis zones
                        mid_x = (x1 + x2) // 2
                        mid_y = (y1 + y2) // 2
                        if not self._overlaps_axis(mid_x - 3, mid_y - 3, 6, 6, axis_zones):
                            result["lines"].append([x1, y1, x2, y2])

            return result

        except Exception as e:
            logger.error(f"Segmentation failed for {image_path}: {e}")
            return result

    # ── Helper: Hough-based axis detection ──────────────────
    def _detect_axes_hough(self, edges, h_img, w_img):
        """Detect horizontal and vertical axis lines using Hough transform."""
        lines = cv2.HoughLinesP(
            edges, 1, np.pi / 180,
            threshold=80,
            minLineLength=int(max(w_img, h_img) * 0.2),
            maxLineGap=20
        )
        horizontal = []
        vertical = []
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                angle = abs(np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi)
                length = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
                if (angle < 5 or angle > 175) and length > w_img * 0.25:
                    horizontal.append([x1, y1, x2, y2])
                elif 85 < angle < 95 and length > h_img * 0.25:
                    vertical.append([x1, y1, x2, y2])
        return horizontal, vertical

    # ── Helper: Build axis exclusion zones ──────────────────
    @staticmethod
    def _build_axis_zones(axes, h_img, w_img):
        """Build rectangular exclusion zones around detected axes."""
        zones = []
        margin = 12
        for axis in axes:
            if len(axis) == 4:
                x1, y1, x2, y2 = axis
                zones.append((
                    min(x1, x2) - margin,
                    min(y1, y2) - margin,
                    abs(x2 - x1) + 2 * margin,
                    abs(y2 - y1) + 2 * margin,
                ))
        return zones

    # ── Helper: Check axis overlap ──────────────────────────
    @staticmethod
    def _overlaps_axis(x, y, w, h, axis_zones):
        """Check if a bounding box overlaps any axis exclusion zone."""
        for (ax, ay, aw, ah) in axis_zones:
            if (x < ax + aw and x + w > ax and y < ay + ah and y + h > ay):
                # Overlap area must be significant
                ox = max(0, min(x + w, ax + aw) - max(x, ax))
                oy = max(0, min(y + h, ay + ah) - max(y, ay))
                overlap_area = ox * oy
                box_area = w * h if w * h > 0 else 1
                if overlap_area / box_area > 0.5:
                    return True
        return False

    # ── Helper: DBSCAN-style point clustering ───────────────
    @staticmethod
    def _cluster_points(points, eps=30.0, min_samples=2):
        """Simple DBSCAN-style clustering without sklearn dependency."""
        if not points:
            return []

        pts = np.array(points, dtype=float)
        n = len(pts)
        visited = [False] * n
        clusters = []

        for i in range(n):
            if visited[i]:
                continue
            visited[i] = True
            cluster = [i]
            queue = [i]

            while queue:
                current = queue.pop(0)
                dists = np.sqrt(np.sum((pts - pts[current]) ** 2, axis=1))
                neighbors = np.where(dists < eps)[0]

                if len(neighbors) >= min_samples:
                    for nb in neighbors:
                        if not visited[nb]:
                            visited[nb] = True
                            cluster.append(nb)
                            queue.append(nb)

            if len(cluster) >= min_samples:
                clusters.append([points[j] for j in cluster])

        return clusters
