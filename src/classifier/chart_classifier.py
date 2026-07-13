"""
Chart classifier using OpenCV heuristics, structural features, and disambiguation rules.
Supports: bar_chart, histogram, line_chart, scatter_plot, pie_chart.
"""

from __future__ import annotations

import logging

import cv2

from src.feature_extraction.geometric_features import GeometricFeatureExtractor
from src.features.chart_feature_extractor import ChartFeatureExtractor
from src.segmentation.chart_element_segmenter import ChartElementSegmenter

logger = logging.getLogger(__name__)


class ChartClassifier:
    """Classifies chart images into one of five chart types."""

    def __init__(self):
        self.feature_extractor = ChartFeatureExtractor()
        self.geo_extractor = GeometricFeatureExtractor()
        self.segmenter = ChartElementSegmenter()

    def classify_chart(self, image_path: str) -> dict:
        result = {
            "chart_type": "unknown",
            "metrics": {},
            "geo_features": {},
            "segmentation_counts": {},
            "title": "",
            "x_axis_label": "",
            "y_axis_label": "",
            "summary": "",
        }

        try:
            img = cv2.imread(image_path)
            if img is None:
                logger.error("Failed to load image: %s", image_path)
                result["summary"] = "Failed to load image"
                return result

            metrics = self.feature_extractor.extract_features(image_path)
            geo = self.geo_extractor.extract(image_path)
            seg = self.segmenter.segment_elements(image_path)

            result["metrics"] = metrics
            result["geo_features"] = geo
            result["segmentation_counts"] = {
                "bars": len(seg.get("bars", [])),
                "points": len(seg.get("points", [])),
                "lines": len(seg.get("lines", [])),
                "pie_slices": len(seg.get("pie_slices", [])),
                "axes": len(seg.get("axes", [])),
                "circles_hough": len(seg.get("circles_hough", [])),
                "point_clusters": len(seg.get("point_clusters", [])),
            }

            chart_type = self._classify_combined(metrics, geo, result["segmentation_counts"])
            result["chart_type"] = chart_type
            result["summary"] = f"Classified as {chart_type} using combined features"

        except Exception as exc:
            logger.error("Chart classification failed for %s: %s", image_path, exc)
            result["summary"] = f"Classification error: {exc}"

        return result

    def _classify_combined(self, m: dict, geo: dict, seg: dict) -> str:
        rects = m.get("rectangle_count", 0)
        spacing = m.get("bar_spacing_ratio", 0.0)
        circles = m.get("circle_count", 0)
        circ_score = m.get("avg_circularity", 0.0)
        density = m.get("scatter_density", 0.0)
        line_cont = m.get("line_continuity_score", 0.0)

        bar_height_var = m.get("bar_height_variance", 0.0)
        bar_align = m.get("bar_alignment_score", 0.0)
        bar_adj = m.get("bar_adjacency_score", 0.0)
        bin_adj = m.get("bin_adjacency", 0.0)
        spacing_unif = m.get("bar_spacing_uniformity", 0.0)
        bin_cont = m.get("bin_continuity", 0.0)
        polyline = m.get("polyline_continuity", 0.0)
        slope_freq = m.get("slope_change_frequency", 0.0)
        vertex_ct = m.get("vertex_count", 0)
        clust_coeff = m.get("point_clustering_coefficient", 0.0)
        nn_dens = m.get("nearest_neighbor_density", 0.0)
        arc_det = m.get("arc_detection_score", 0.0)
        circ_sym = m.get("circular_symmetry", 0.0)
        radial_dens = m.get("radial_edge_density", 0.0)

        bar_count = geo.get("bar_count", 0)
        g_line_count = geo.get("line_count", 0)
        point_density = geo.get("point_density", 0.0)
        g_circles = geo.get("circle_count", 0)
        edge_density = geo.get("edge_density", 0.0)
        hough_circles = geo.get("hough_circle_count", 0)
        g_circ_sym = geo.get("circular_symmetry", 0.0)
        g_bin_adj = geo.get("bin_adjacency_ratio", 0.0)
        g_width_unif = geo.get("width_uniformity", 0.0)
        line_cont_geo = geo.get("line_continuity_score", 0.0)
        line_connectivity = geo.get("line_connectivity_score", geo.get("polyline_score", 0.0))
        slope_continuity = geo.get("slope_continuity_score", 0.0)
        cluster_compactness = geo.get("cluster_compactness", geo.get("nn_density", 0.0))
        scatter_cluster_count = geo.get("scatter_cluster_count", 0.0)
        bar_gap_ratio = geo.get("bar_gap_ratio", 0.0)
        hist_uniformity = geo.get("histogram_uniformity_score", 0.0)
        axis_numeric_density = geo.get("axis_numeric_density", 0.0)

        seg_bars = seg.get("bars", 0)
        seg_points = seg.get("points", 0)
        seg_lines = seg.get("lines", 0)
        seg_pie = seg.get("pie_slices", 0)
        seg_hough = seg.get("circles_hough", 0)
        seg_clusters = seg.get("point_clusters", 0)

        scores = {
            "bar_chart": 0.0,
            "histogram": 0.0,
            "scatter_plot": 0.0,
            "pie_chart": 0.0,
            "line_chart": 0.0,
        }

        histogram_score = self._compute_histogram_score(
            bin_adj,
            g_bin_adj,
            spacing_unif,
            bin_cont,
            g_width_unif,
            spacing,
            rects,
            seg_bars,
        )
        bar_score = self._compute_bar_score(spacing, bar_adj, bar_height_var, rects, seg_bars, bar_count)

        if rects >= 2 and spacing > 0.1:
            scores["bar_chart"] += 2.8
        if 2 <= bar_count <= 12:
            scores["bar_chart"] += 2.2
        if 2 <= seg_bars <= 12:
            scores["bar_chart"] += 1.8
        if spacing > 0.28:
            scores["bar_chart"] += 1.0
        if bar_align > 0.8:
            scores["bar_chart"] += 0.8
        if bar_gap_ratio > 0.18:
            scores["bar_chart"] += 1.2
        if hist_uniformity < 0.55:
            scores["bar_chart"] += 0.8
        scores["bar_chart"] += bar_score * 2.0

        if rects > 5 and spacing < 0.15:
            scores["histogram"] += 3.6
        if bin_adj > 0.5:
            scores["histogram"] += 2.7
        if g_bin_adj > 0.5:
            scores["histogram"] += 2.1
        if bin_cont > 0.7 or hist_uniformity > 0.7:
            scores["histogram"] += 2.3
        if bar_gap_ratio < 0.12:
            scores["histogram"] += 1.2
        if axis_numeric_density > 0.25:
            scores["histogram"] += 1.0
        if bar_count > 5 or seg_bars > 5:
            scores["histogram"] += 1.2
        scores["histogram"] += histogram_score * 2.8

        if density > 0.00005 and circles > 15:
            scores["scatter_plot"] += 3.0
        if point_density > 0.00003:
            scores["scatter_plot"] += 2.0
        if seg_points > 15:
            scores["scatter_plot"] += 2.4
        if clust_coeff > 0.3:
            scores["scatter_plot"] += 1.6
        if nn_dens > 0.5 or cluster_compactness > 0.45:
            scores["scatter_plot"] += 1.6
        if seg_clusters > 1 or scatter_cluster_count > 2:
            scores["scatter_plot"] += 1.4
        if line_connectivity > 0.45 or slope_continuity > 0.45:
            scores["scatter_plot"] -= 1.6

        if circles >= 1 and circ_score > 0.6:
            scores["pie_chart"] += 3.0
        if seg_pie >= 1:
            scores["pie_chart"] += 3.0
        if g_circles >= 1 and seg_bars < 3:
            scores["pie_chart"] += 1.5
        if arc_det > 0.3:
            scores["pie_chart"] += 2.0
        if circ_sym > 0.5 or g_circ_sym > 0.5:
            scores["pie_chart"] += 2.0
        if radial_dens > 0.05:
            scores["pie_chart"] += 1.5
        if seg_hough >= 1 or hough_circles >= 1:
            scores["pie_chart"] += 1.8

        if line_cont > 0.05 or line_cont_geo > 0.05:
            scores["line_chart"] += 3.0
        if g_line_count > 3:
            scores["line_chart"] += 1.8
        if seg_lines > 5:
            scores["line_chart"] += 2.0
        if line_connectivity > 0.35:
            scores["line_chart"] += 1.7
        if slope_continuity > 0.35:
            scores["line_chart"] += 1.2
        if polyline > 0.3:
            scores["line_chart"] += 1.4
        if slope_freq > 0.2:
            scores["line_chart"] += 0.8
        if vertex_ct > 5:
            scores["line_chart"] += 0.7
        if edge_density > 0.05 and seg_bars < 3:
            scores["line_chart"] += 0.8
        if seg_points > 20 and cluster_compactness > 0.45:
            scores["line_chart"] -= 1.7

        best = max(scores, key=scores.get)
        if scores[best] < 1.0:
            return self._classify_fallback(m)
        return best

    def _compute_histogram_score(self, bin_adj, g_bin_adj, spacing_unif, bin_cont, g_width_unif, spacing, rects, seg_bars):
        """Histogram detection score using contiguous and uniform bin cues."""
        score = 0.0
        count = 0

        adj = max(bin_adj, g_bin_adj)
        score += adj
        count += 1

        unif = max(g_width_unif, bin_cont)
        score += unif
        count += 1

        if spacing < 0.1:
            score += 1.0
        elif spacing < 0.2:
            score += 0.5
        count += 1

        if rects > 8 or seg_bars > 8:
            score += 1.0
        elif rects > 5 or seg_bars > 5:
            score += 0.5
        count += 1

        return score / count if count > 0 else 0.0

    def _compute_bar_score(self, spacing, bar_adj, bar_height_var, rects, seg_bars, bar_count):
        """Bar chart detection score using separated bars and moderate count cues."""
        score = 0.0
        count = 0

        if spacing > 0.15:
            score += 1.0
        elif spacing > 0.05:
            score += 0.5
        count += 1

        if 2 <= rects <= 12 or 2 <= seg_bars <= 12 or 2 <= bar_count <= 12:
            score += 1.0
        count += 1

        if bar_height_var > 100:
            score += 0.5
        count += 1

        return score / count if count > 0 else 0.0

    def _classify_fallback(self, m: dict) -> str:
        rects = m.get("rectangle_count", 0)
        circles = m.get("circle_count", 0)
        line_cont = m.get("line_continuity_score", 0.0)

        if rects >= 2:
            return "bar_chart"
        if circles >= 10:
            return "scatter_plot"
        if circles >= 1:
            return "pie_chart"
        if line_cont > 0.03:
            return "line_chart"
        return "bar_chart"
