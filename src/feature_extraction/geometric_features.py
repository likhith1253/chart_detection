"""
Geometric Feature Extraction module - research-grade.
Extracts structural features used by heuristic and ML models.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Tuple

import cv2
import numpy as np
from scipy.stats import entropy as scipy_entropy

logger = logging.getLogger(__name__)


class GeometricFeatureExtractor:
    """Extracts comprehensive geometric features from chart images."""

    @staticmethod
    def _cluster_points(points: List[Tuple[float, float]], eps: float) -> int:
        """Lightweight connected-components clustering with euclidean threshold."""
        if not points:
            return 0
        pts = np.asarray(points, dtype=float)
        n = len(pts)
        visited = np.zeros(n, dtype=bool)
        clusters = 0

        for idx in range(n):
            if visited[idx]:
                continue
            clusters += 1
            stack = [idx]
            visited[idx] = True
            while stack:
                current = stack.pop()
                dists = np.sqrt(np.sum((pts - pts[current]) ** 2, axis=1))
                neighbors = np.where((dists <= eps) & (~visited))[0]
                for nb in neighbors:
                    visited[nb] = True
                    stack.append(int(nb))
        return int(clusters)

    @staticmethod
    def _edge_orientation_entropy(gray: np.ndarray) -> float:
        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        mag = cv2.magnitude(gx, gy)
        ori = cv2.phase(gx, gy, angleInDegrees=False)
        mask = mag > np.percentile(mag, 70)
        if np.count_nonzero(mask) < 100:
            return 0.0
        values = ori[mask].flatten()
        hist, _ = np.histogram(values, bins=18, range=(0.0, 2 * np.pi), density=False)
        hist = hist.astype(float)
        if hist.sum() <= 0:
            return 0.0
        probs = hist / hist.sum()
        ent = scipy_entropy(probs + 1e-12, base=2)
        return float(ent / np.log2(18))

    @staticmethod
    def _axis_numeric_density(gray: np.ndarray) -> float:
        """Approximate numeric label density near axis bands using connected components."""
        h, w = gray.shape[:2]
        if h < 32 or w < 32:
            return 0.0

        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        bottom_band = binary[int(h * 0.78) : h, :]
        left_band = binary[:, : int(w * 0.22)]

        def _count_digits(region: np.ndarray) -> int:
            contours, _ = cv2.findContours(region, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            count = 0
            for cnt in contours:
                x, y, bw, bh = cv2.boundingRect(cnt)
                area = bw * bh
                if area < 20 or area > 1200:
                    continue
                ar = bw / float(bh + 1e-6)
                if 0.15 <= ar <= 1.5 and 6 <= bh <= 80:
                    count += 1
            return count

        digits = _count_digits(bottom_band) + _count_digits(left_band)
        norm = max(1.0, np.sqrt(float(h * h + w * w)) * 0.08)
        return float(min(1.0, digits / norm))

    def extract(self, image_path: str) -> Dict[str, float]:
        """Extract geometric features from a chart image."""
        features = {
            "bar_count": 0,
            "bar_width_variance": 0.0,
            "bar_spacing_variance": 0.0,
            "line_count": 0,
            "point_density": 0.0,
            "circle_count": 0,
            "edge_density": 0.0,
            "axis_orientation": 0.0,
            "contour_count": 0,
            "texture_entropy": 0.0,
            "bar_height_variance": 0.0,
            "bar_alignment_score": 0.0,
            "bin_adjacency_ratio": 0.0,
            "width_uniformity": 0.0,
            "polyline_score": 0.0,
            "slope_variance": 0.0,
            "scatter_spatial_var": 0.0,
            "nn_density": 0.0,
            "hough_circle_count": 0,
            "circular_symmetry": 0.0,
            "radial_density": 0.0,
            "tick_density": 0.0,
            "axis_confidence": 0.0,
            "fill_ratio_mean": 0.0,
            "aspect_ratio_var": 0.0,
            "color_diversity": 0.0,
            "saturation_mean": 0.0,
            "brightness_variance": 0.0,
            # New research features
            "line_continuity_score": 0.0,
            "scatter_cluster_count": 0.0,
            "polyline_vertex_count": 0.0,
            "bar_gap_ratio": 0.0,
            "histogram_uniformity_score": 0.0,
            "axis_numeric_density": 0.0,
            "edge_orientation_entropy": 0.0,
            # Extra disambiguation helpers
            "line_connectivity_score": 0.0,
            "slope_continuity_score": 0.0,
            "cluster_compactness": 0.0,
        }

        try:
            img = cv2.imread(image_path)
            if img is None:
                logger.error("GeometricFeatures: failed to load %s", image_path)
                return features

            h, w = img.shape[:2]
            img_area = float(max(1, h * w))
            diag = float(np.sqrt(h**2 + w**2))
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

            # Global image/texture features
            edges = cv2.Canny(blurred, 50, 150)
            features["edge_density"] = float(np.count_nonzero(edges)) / img_area
            features["edge_orientation_entropy"] = self._edge_orientation_entropy(gray)

            hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).flatten()
            hist = hist / hist.sum() if hist.sum() > 0 else hist
            features["texture_entropy"] = float(scipy_entropy(hist + 1e-12, base=2))

            hue_hist = cv2.calcHist([hsv], [0], None, [18], [0, 180]).flatten()
            features["color_diversity"] = float(np.count_nonzero(hue_hist > img_area * 0.005))
            features["saturation_mean"] = float(np.mean(hsv[:, :, 1])) / 255.0
            features["brightness_variance"] = float(np.var(hsv[:, :, 2])) / (255.0**2)
            features["axis_numeric_density"] = self._axis_numeric_density(gray)

            # Contours
            _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            kernel = np.ones((3, 3), np.uint8)
            morph = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
            contours, _ = cv2.findContours(morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            features["contour_count"] = int(len(contours))

            bar_widths: List[float] = []
            bar_heights: List[float] = []
            bar_xs: List[float] = []
            bar_bottoms: List[float] = []
            scatter_pts: List[Tuple[float, float]] = []
            fill_ratios: List[float] = []
            aspect_ratios: List[float] = []
            point_count = 0

            for contour in contours:
                area = cv2.contourArea(contour)
                if area < 200 or area > img_area * 0.5:
                    continue

                x, y, cw, ch = cv2.boundingRect(contour)
                peri = cv2.arcLength(contour, True)
                if peri <= 0:
                    continue

                aspect = float(ch) / float(cw + 1e-6)
                circularity = 4 * np.pi * area / (peri * peri)
                fill = area / float(max(1, cw * ch))
                fill_ratios.append(fill)
                aspect_ratios.append(aspect)

                if aspect > 0.8 and circularity < 0.7 and fill > 0.4:
                    bar_widths.append(float(cw))
                    bar_heights.append(float(ch))
                    bar_xs.append(float(x))
                    bar_bottoms.append(float(y + ch))

                if circularity > 0.6:
                    if area < 1500:
                        point_count += 1
                        m = cv2.moments(contour)
                        if m["m00"] != 0:
                            scatter_pts.append((float(m["m10"] / m["m00"]), float(m["m01"] / m["m00"])))
                    else:
                        features["circle_count"] += 1

            features["bar_count"] = int(len(bar_widths))
            features["point_density"] = float(point_count) / img_area
            if fill_ratios:
                features["fill_ratio_mean"] = float(np.mean(fill_ratios))
            if len(aspect_ratios) > 1:
                features["aspect_ratio_var"] = float(np.var(aspect_ratios))

            if len(bar_widths) > 1:
                features["bar_width_variance"] = float(np.var(bar_widths))
                features["bar_height_variance"] = float(np.var(bar_heights))
                sorted_xs = sorted(bar_xs)
                spacings = [sorted_xs[i + 1] - sorted_xs[i] for i in range(len(sorted_xs) - 1)]
                if spacings:
                    spacing_arr = np.asarray(spacings, dtype=float)
                    features["bar_spacing_variance"] = float(np.var(spacing_arr))
                    avg_w = float(np.mean(bar_widths))
                    gap_mean = float(np.mean(spacing_arr))
                    features["bar_gap_ratio"] = float(gap_mean / (avg_w + 1e-6))
                    touching = sum(1 for s in spacing_arr.tolist() if s < avg_w * 0.15)
                    features["bin_adjacency_ratio"] = float(touching / len(spacing_arr))

                    gap_cv = float(np.std(spacing_arr)) / (gap_mean + 1e-6)
                    width_cv = float(np.std(bar_widths)) / (avg_w + 1e-6)
                    features["width_uniformity"] = 1.0 - min(1.0, width_cv)
                    features["histogram_uniformity_score"] = float(
                        np.clip(0.55 * (1.0 - min(1.0, width_cv)) + 0.45 * (1.0 - min(1.0, gap_cv)), 0.0, 1.0)
                    )

                if len(bar_bottoms) > 1:
                    features["bar_alignment_score"] = 1.0 - min(1.0, float(np.std(bar_bottoms)) / float(h))

            if len(scatter_pts) > 2:
                pts = np.asarray(scatter_pts, dtype=float)
                features["scatter_spatial_var"] = float(np.var(pts[:, 0]) + np.var(pts[:, 1])) / img_area

                nn_dists = []
                for i in range(len(pts)):
                    dists = np.sqrt(np.sum((pts - pts[i]) ** 2, axis=1))
                    dists[i] = np.inf
                    nn_dists.append(float(np.min(dists)))
                if nn_dists:
                    avg_nn = float(np.mean(nn_dists))
                    density = 1.0 - min(1.0, avg_nn / (diag * 0.1 + 1e-6))
                    features["nn_density"] = density
                    features["cluster_compactness"] = density

                eps = max(w, h) * 0.05
                features["scatter_cluster_count"] = float(self._cluster_points(scatter_pts, eps=eps))

            lines = cv2.HoughLinesP(
                edges,
                1,
                np.pi / 180,
                threshold=50,
                minLineLength=int(w * 0.1),
                maxLineGap=10,
            )

            h_count = 0
            v_count = 0
            diag_count = 0
            slopes: List[float] = []
            line_lengths: List[float] = []
            segments: List[Tuple[int, int, int, int]] = []

            if lines is not None:
                for entry in lines:
                    x1, y1, x2, y2 = entry[0]
                    angle = abs(np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi)
                    length = float(np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2))
                    if length < w * 0.05:
                        continue
                    segments.append((x1, y1, x2, y2))
                    line_lengths.append(length)
                    if angle < 10 or angle > 170:
                        h_count += 1
                    elif 80 < angle < 100:
                        v_count += 1
                    else:
                        diag_count += 1
                        slopes.append(float((y2 - y1) / (x2 - x1 + 1e-6)))

            features["line_count"] = int(diag_count)
            features["line_continuity_score"] = float(sum(line_lengths) / max(1.0, diag))
            features["polyline_vertex_count"] = float(len(segments))

            total_axis = h_count + v_count
            if total_axis > 0:
                features["axis_orientation"] = float(h_count) / float(total_axis)
                features["axis_confidence"] = min(1.0, float(total_axis * 50.0 / max(1.0, diag)))
            else:
                features["axis_orientation"] = 0.5

            if len(segments) > 1:
                sorted_segments = sorted(segments, key=lambda s: s[0])
                chains = 0
                for idx in range(1, len(sorted_segments)):
                    p1 = (sorted_segments[idx - 1][2], sorted_segments[idx - 1][3])
                    p2 = (sorted_segments[idx][0], sorted_segments[idx][1])
                    if np.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2) < w * 0.1:
                        chains += 1
                connectivity = chains / float(max(1, len(sorted_segments) - 1))
                features["polyline_score"] = connectivity
                features["line_connectivity_score"] = connectivity

            if len(slopes) > 1:
                slopes_arr = np.asarray(slopes, dtype=float)
                var = float(np.var(slopes_arr))
                features["slope_variance"] = var
                features["slope_continuity_score"] = 1.0 - min(1.0, var / (abs(float(np.mean(slopes_arr))) + 1.0))

            short_lines = cv2.HoughLinesP(
                edges,
                1,
                np.pi / 180,
                threshold=15,
                minLineLength=3,
                maxLineGap=3,
            )
            tick_count = 0
            if short_lines is not None:
                for entry in short_lines:
                    x1, y1, x2, y2 = entry[0]
                    length = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
                    if 3 < length < max(w, h) * 0.05:
                        tick_count += 1
            features["tick_density"] = float(tick_count) / (diag * 0.01 + 1e-6)

            gray_blur = cv2.GaussianBlur(gray, (9, 9), 2)
            circles = cv2.HoughCircles(
                gray_blur,
                cv2.HOUGH_GRADIENT,
                dp=1.2,
                minDist=int(min(h, w) * 0.3),
                param1=100,
                param2=50,
                minRadius=int(min(h, w) * 0.1),
                maxRadius=int(min(h, w) * 0.48),
            )
            if circles is not None:
                features["hough_circle_count"] = int(len(circles[0]))
                for cx, cy, _ in circles[0]:
                    dist = np.sqrt((cx - w / 2) ** 2 + (cy - h / 2) ** 2)
                    sym = 1.0 - min(1.0, dist / (min(w, h) * 0.3))
                    features["circular_symmetry"] = max(features["circular_symmetry"], float(sym))

            cx, cy = w // 2, h // 2
            r_max = min(w, h) // 3
            if r_max > 10:
                mask = np.zeros_like(gray)
                cv2.circle(mask, (cx, cy), r_max, 255, -1)
                inner = np.zeros_like(gray)
                cv2.circle(inner, (cx, cy), r_max // 3, 255, -1)
                ring = cv2.subtract(mask, inner)
                ring_edges = cv2.bitwise_and(edges, ring)
                ring_area = np.count_nonzero(ring)
                if ring_area > 0:
                    features["radial_density"] = float(np.count_nonzero(ring_edges)) / float(ring_area)

            return features

        except Exception as exc:
            logger.error("Geometric feature extraction failed: %s", exc)
            return features
