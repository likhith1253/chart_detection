"""
Chart Feature Extraction Module — Research-Grade.
Computes 28+ structural metrics from chart images organized by chart type:
  Bar: height variance, alignment score, adjacency score
  Histogram: bin adjacency, spacing uniformity, bin continuity
  Line: polyline continuity, slope change frequency, vertex count
  Scatter: clustering coefficient, spatial variance, NN density
  Pie: arc detection, circular symmetry, radial edge density
  Axis: detection confidence, orientation, tick density
Plus original metrics: rectangles, circles, spacing, entropy, etc.
"""

import cv2
import numpy as np
import logging
from typing import Dict

logger = logging.getLogger(__name__)


class ChartFeatureExtractor:
    """Computes 28+ structural metrics from chart images."""

    def extract_features(self, img_path: str) -> dict:
        """
        Extracts comprehensive structural metrics from a chart image.

        Returns:
            Dictionary containing 28+ float/int metrics.
        """
        metrics = {
            # Original metrics
            "rectangle_count": 0,
            "vertical_rectangle_ratio": 0.0,
            "bar_spacing_ratio": 0.0,
            "rectangle_width_variance": 0.0,
            "circle_count": 0,
            "avg_circularity": 0.0,
            "scatter_density": 0.0,
            "line_continuity_score": 0.0,
            "contour_area_entropy": 0.0,
            "contour_area_variance": 0.0,
            # Bar features
            "bar_height_variance": 0.0,
            "bar_alignment_score": 0.0,
            "bar_adjacency_score": 0.0,
            # Histogram features
            "bin_adjacency": 0.0,
            "bar_spacing_uniformity": 0.0,
            "bin_continuity": 0.0,
            # Line chart features
            "polyline_continuity": 0.0,
            "slope_change_frequency": 0.0,
            "vertex_count": 0,
            # Scatter features
            "point_clustering_coefficient": 0.0,
            "spatial_variance": 0.0,
            "nearest_neighbor_density": 0.0,
            # Pie features
            "arc_detection_score": 0.0,
            "circular_symmetry": 0.0,
            "radial_edge_density": 0.0,
            # Axis features
            "axis_detection_confidence": 0.0,
            "axis_orientation": 0.0,
            "tick_density": 0.0,
        }

        try:
            img = cv2.imread(img_path)
            if img is None:
                logger.error(f"Feature Extractor: Failed to load {img_path}")
                return metrics

            h_img, w_img = img.shape[:2]
            img_area = h_img * w_img

            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)

            # ── Color/brightness segmentation ────────────────────
            sat = hsv[:, :, 1]
            val = hsv[:, :, 2]
            color_mask = cv2.inRange(sat, 25, 255)
            bright_mask = cv2.inRange(val, 50, 255)
            combined = cv2.bitwise_and(color_mask, bright_mask)

            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
            combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel)
            combined = cv2.morphologyEx(combined, cv2.MORPH_OPEN, kernel)

            contours, _ = cv2.findContours(
                combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )

            rectangles = []
            rect_widths = []
            rect_heights = []
            rect_bottoms = []
            circularity_scores = []

            for c in contours:
                area = cv2.contourArea(c)
                if area < 100 or area > img_area * 0.4:
                    continue

                peri = cv2.arcLength(c, True)
                circularity = (4 * np.pi * area / (peri * peri)) if peri > 0 else 0
                x, y, w, h = cv2.boundingRect(c)
                bbox_ar = float(h) / float(w) if w > 0 else 1.0
                approx = cv2.approxPolyDP(c, 0.04 * peri, True)
                vertices = len(approx)

                # Circle detection
                if circularity > 0.6 and area > img_area * 0.02 and 0.4 < bbox_ar < 2.5:
                    metrics["circle_count"] += 1
                    circularity_scores.append(circularity)
                elif area < 1500 and circularity > 0.5:
                    if (y > h_img * 0.15 and y + h < h_img * 0.85 and
                            x > w_img * 0.10 and x + w < w_img * 0.90):
                        metrics["circle_count"] += 1
                        circularity_scores.append(circularity)
                # Rectangle detection
                elif 3 <= vertices <= 8:
                    fill_ratio = area / (w * h) if (w * h) > 0 else 0
                    if fill_ratio > 0.4:
                        rectangles.append((x, y, w, h))
                        rect_widths.append(w)
                        rect_heights.append(h)
                        rect_bottoms.append(y + h)

                        if w > w_img * 0.3 and h > h_img * 0.15:
                            pass  # merged histogram mass

            # ── Structural rectangle metrics ─────────────────────
            metrics["rectangle_count"] = len(rectangles)
            vertical_rects = 0

            if rectangles:
                for (x, y, w, h) in rectangles:
                    if float(h) / float(w) > 1.0:
                        vertical_rects += 1
                metrics["vertical_rectangle_ratio"] = vertical_rects / len(rectangles)
                metrics["rectangle_width_variance"] = float(np.var(rect_widths)) if len(rect_widths) > 1 else 0.0

                sorted_rects = sorted(rectangles, key=lambda r: r[0])
                gaps = []
                avg_width = float(np.mean(rect_widths))
                for i in range(1, len(sorted_rects)):
                    gap = sorted_rects[i][0] - (sorted_rects[i - 1][0] + sorted_rects[i - 1][2])
                    if gap >= 0:
                        gaps.append(gap)

                if gaps and avg_width > 0:
                    metrics["bar_spacing_ratio"] = float(np.mean(gaps)) / avg_width

                # ── BAR FEATURES ─────────────────────────────────
                if len(rect_heights) > 1:
                    metrics["bar_height_variance"] = float(np.var(rect_heights))

                # Alignment score: how well bar bottoms align (low = good alignment)
                if len(rect_bottoms) > 1:
                    metrics["bar_alignment_score"] = 1.0 - min(1.0, float(np.std(rect_bottoms)) / h_img)

                # Adjacency score: uniformity of gaps
                if gaps:
                    mean_gap = np.mean(gaps)
                    if mean_gap > 0:
                        metrics["bar_adjacency_score"] = 1.0 - min(1.0, float(np.std(gaps)) / mean_gap)
                    else:
                        metrics["bar_adjacency_score"] = 1.0

                # ── HISTOGRAM FEATURES ───────────────────────────
                # Bin adjacency: fraction of adjacent bars that are touching
                touching = sum(1 for g in gaps if g < avg_width * 0.1) if gaps else 0
                metrics["bin_adjacency"] = touching / len(gaps) if gaps else 0.0

                # Spacing uniformity
                if gaps and len(gaps) > 1:
                    metrics["bar_spacing_uniformity"] = 1.0 - min(1.0, float(np.std(gaps)) / (np.mean(gaps) + 1e-6))

                # Bin continuity: width uniformity of bars
                if len(rect_widths) > 1:
                    cv_w = float(np.std(rect_widths)) / (np.mean(rect_widths) + 1e-6)
                    metrics["bin_continuity"] = 1.0 - min(1.0, cv_w)

            # ── Grayscale scatter detection ──────────────────────
            _, dot_thresh = cv2.threshold(gray, 220, 255, cv2.THRESH_BINARY_INV)
            margin_y = int(h_img * 0.15)
            margin_x = int(w_img * 0.12)
            chart_roi = dot_thresh[margin_y:h_img - margin_y, margin_x:w_img - margin_x]

            scatter_points = []
            dot_areas = []

            if chart_roi.size > 0:
                dot_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
                chart_roi = cv2.morphologyEx(chart_roi, cv2.MORPH_OPEN, dot_kernel)
                dot_contours, _ = cv2.findContours(
                    chart_roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                )

                for dc in dot_contours:
                    da = cv2.contourArea(dc)
                    if 20 < da < 1500:
                        dot_areas.append(da)
                        dp = cv2.arcLength(dc, True)
                        if dp > 0:
                            dc_circ = 4 * np.pi * da / (dp * dp)
                            if dc_circ > 0.4:
                                metrics["circle_count"] += 1
                                circularity_scores.append(dc_circ)
                                M = cv2.moments(dc)
                                if M["m00"] != 0:
                                    cx = int(M["m10"] / M["m00"]) + margin_x
                                    cy = int(M["m01"] / M["m00"]) + margin_y
                                    scatter_points.append((cx, cy))

            if circularity_scores:
                metrics["avg_circularity"] = float(np.mean(circularity_scores))
            metrics["scatter_density"] = float(metrics["circle_count"]) / img_area

            # ── SCATTER FEATURES ─────────────────────────────────
            if len(scatter_points) > 2:
                pts = np.array(scatter_points, dtype=float)
                metrics["spatial_variance"] = float(np.var(pts[:, 0]) + np.var(pts[:, 1])) / img_area

                # Nearest-neighbor density
                nn_dists = []
                for i in range(len(pts)):
                    dists = np.sqrt(np.sum((pts - pts[i]) ** 2, axis=1))
                    dists[i] = np.inf
                    nn_dists.append(np.min(dists))
                if nn_dists:
                    avg_nn = np.mean(nn_dists)
                    diag = np.sqrt(h_img ** 2 + w_img ** 2)
                    metrics["nearest_neighbor_density"] = 1.0 - min(1.0, avg_nn / (diag * 0.1))

                # Clustering coefficient (fraction in clusters)
                eps = max(w_img, h_img) * 0.05
                clustered = 0
                for i in range(len(pts)):
                    dists = np.sqrt(np.sum((pts - pts[i]) ** 2, axis=1))
                    neighbors = np.sum(dists < eps) - 1
                    if neighbors >= 2:
                        clustered += 1
                metrics["point_clustering_coefficient"] = clustered / len(pts)

            # ── Contour area entropy & variance ──────────────────
            all_contour_areas = []
            for c in contours:
                area = cv2.contourArea(c)
                if 100 <= area <= img_area * 0.4:
                    all_contour_areas.append(area)
            all_contour_areas.extend(dot_areas)

            if all_contour_areas:
                areas_arr = np.array(all_contour_areas, dtype=float)
                metrics["contour_area_variance"] = float(np.var(areas_arr))
                total_area = np.sum(areas_arr)
                if total_area > 0:
                    probs = areas_arr / total_area
                    probs = probs[probs > 0]
                    entropy = -np.sum(probs * np.log2(probs))
                    metrics["contour_area_entropy"] = float(entropy)

            # ── LINE CHART FEATURES ──────────────────────────────
            edges = cv2.Canny(blurred, 50, 150)
            lines = cv2.HoughLinesP(
                edges, 1, np.pi / 180,
                threshold=50, minLineLength=int(w_img * 0.12), maxLineGap=10
            )

            img_diagonal = np.sqrt(h_img ** 2 + w_img ** 2)
            line_len_sum = 0
            line_segments = []
            slope_changes = []

            if lines is not None:
                for line in lines:
                    x1, y1, x2, y2 = line[0]
                    angle = abs(np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi)
                    if 10 < angle < 80 or 100 < angle < 170:
                        line_len = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
                        line_len_sum += line_len
                        slope = (y2 - y1) / (x2 - x1 + 1e-6)
                        line_segments.append((x1, y1, x2, y2, slope))

            metrics["line_continuity_score"] = float(line_len_sum / img_diagonal) if img_diagonal > 0 else 0.0

            # Polyline continuity: chain connected segments
            if len(line_segments) > 1:
                sorted_segs = sorted(line_segments, key=lambda s: s[0])
                chain_count = 0
                for i in range(1, len(sorted_segs)):
                    end_prev = (sorted_segs[i - 1][2], sorted_segs[i - 1][3])
                    start_curr = (sorted_segs[i][0], sorted_segs[i][1])
                    dist = np.sqrt((end_prev[0] - start_curr[0]) ** 2 + (end_prev[1] - start_curr[1]) ** 2)
                    if dist < w_img * 0.1:
                        chain_count += 1
                metrics["polyline_continuity"] = chain_count / (len(sorted_segs) - 1)

                # Slope change frequency
                slopes = [s[4] for s in sorted_segs]
                changes = sum(1 for i in range(1, len(slopes)) if abs(slopes[i] - slopes[i - 1]) > 0.2)
                metrics["slope_change_frequency"] = changes / len(slopes)

            # Vertex count (approximate polylines)
            metrics["vertex_count"] = len(line_segments)

            # ── PIE CHART FEATURES ───────────────────────────────
            # Arc detection via HoughCircles
            gray_blur = cv2.GaussianBlur(gray, (9, 9), 2)
            hough_circles = cv2.HoughCircles(
                gray_blur, cv2.HOUGH_GRADIENT,
                dp=1.2, minDist=int(min(h_img, w_img) * 0.3),
                param1=100, param2=50,
                minRadius=int(min(h_img, w_img) * 0.1),
                maxRadius=int(min(h_img, w_img) * 0.48)
            )
            if hough_circles is not None:
                metrics["arc_detection_score"] = min(1.0, len(hough_circles[0]) * 0.5)

                # Circular symmetry: check if image center is near circle center
                for circ in hough_circles[0]:
                    cx, cy, r = circ
                    center_dist = np.sqrt((cx - w_img / 2) ** 2 + (cy - h_img / 2) ** 2)
                    sym_score = 1.0 - min(1.0, center_dist / (min(w_img, h_img) * 0.3))
                    metrics["circular_symmetry"] = max(metrics["circular_symmetry"], sym_score)

            # Radial edge density
            center_x, center_y = w_img // 2, h_img // 2
            r_max = min(w_img, h_img) // 3
            if r_max > 10:
                mask = np.zeros_like(gray)
                cv2.circle(mask, (center_x, center_y), r_max, 255, -1)
                inner_mask = np.zeros_like(gray)
                cv2.circle(inner_mask, (center_x, center_y), r_max // 3, 255, -1)
                ring_mask = cv2.subtract(mask, inner_mask)
                ring_edges = cv2.bitwise_and(edges, ring_mask)
                ring_area = np.count_nonzero(ring_mask)
                if ring_area > 0:
                    metrics["radial_edge_density"] = float(np.count_nonzero(ring_edges)) / ring_area

            # ── AXIS FEATURES ────────────────────────────────────
            all_lines = cv2.HoughLinesP(
                edges, 1, np.pi / 180,
                threshold=80, minLineLength=int(max(w_img, h_img) * 0.2), maxLineGap=20
            )
            h_axes = 0
            v_axes = 0
            total_axis_len = 0

            if all_lines is not None:
                for line in all_lines:
                    x1, y1, x2, y2 = line[0]
                    angle = abs(np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi)
                    length = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
                    if (angle < 5 or angle > 175) and length > w_img * 0.25:
                        h_axes += 1
                        total_axis_len += length
                    elif 85 < angle < 95 and length > h_img * 0.25:
                        v_axes += 1
                        total_axis_len += length

            total_axes = h_axes + v_axes
            if total_axes > 0:
                metrics["axis_detection_confidence"] = min(1.0, total_axis_len / (img_diagonal * 0.5))
                metrics["axis_orientation"] = float(h_axes) / total_axes
            else:
                metrics["axis_orientation"] = 0.5

            # Tick density: short perpendicular marks near axes
            tick_lines = cv2.HoughLinesP(
                edges, 1, np.pi / 180,
                threshold=20, minLineLength=3, maxLineGap=3
            )
            tick_count = 0
            if tick_lines is not None:
                for line in tick_lines:
                    x1, y1, x2, y2 = line[0]
                    length = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
                    if 3 < length < max(w_img, h_img) * 0.05:
                        tick_count += 1
            metrics["tick_density"] = tick_count / (img_diagonal * 0.01 + 1e-6)

            # ── Histogram merged bins edge case ──────────────────
            for (x, y, w, h) in rectangles:
                if w > w_img * 0.3 and h > h_img * 0.15:
                    metrics["rectangle_count"] = max(metrics["rectangle_count"], 10)
                    metrics["bar_spacing_ratio"] = 0.0
                    metrics["rectangle_width_variance"] = 0.0
                    metrics["vertical_rectangle_ratio"] = 1.0

            return metrics

        except Exception as e:
            logger.error(f"Feature extraction failed for {img_path}: {e}")
            return metrics
