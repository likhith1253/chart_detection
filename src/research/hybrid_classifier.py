"""
Hybrid chart classification with confidence-weighted voting.
"""

from __future__ import annotations

import itertools
import logging
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

logger = logging.getLogger(__name__)


def _bounded(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return float(max(lo, min(hi, value)))


class HybridChartClassifier:
    """Combine heuristic, ML, CNN, and YOLO structural predictions."""

    DEFAULT_WEIGHTS = {
        "ml": 0.5,
        "cnn": 0.3,
        "yolo": 0.15,
        "heuristic": 0.05,
    }

    @staticmethod
    def _disambiguate_hist_bar(row: pd.Series, pred: str, confidence: float, source: str) -> Tuple[str, float]:
        if pred not in ("histogram", "bar_chart"):
            return pred, confidence

        # Do not override strong ML/CNN predictions.
        if source in ("ml", "cnn") and confidence >= 0.6:
            return pred, confidence
        if confidence >= 0.85:
            return pred, confidence

        hist_score = float(row.get("histogram_score", 0.0))
        bar_score = float(row.get("bar_chart_score", 0.0))
        yolo_bars = float(row.get("yolo_bars", 0))
        yolo_lines = float(row.get("yolo_line_segments", 0))

        hist_boost = 0.0
        bar_boost = 0.0

        if hist_score > bar_score + 0.25:
            hist_boost += 0.12
        if bar_score > hist_score + 0.25:
            bar_boost += 0.12
        if source in ("heuristic", "yolo") and yolo_bars >= 8 and yolo_lines < 6:
            hist_boost += 0.08
        if source in ("heuristic", "yolo") and 2 <= yolo_bars <= 7:
            bar_boost += 0.06

        if hist_boost > bar_boost:
            return "histogram", _bounded(confidence + hist_boost)
        if bar_boost > hist_boost:
            return "bar_chart", _bounded(confidence + bar_boost)
        return pred, confidence

    @staticmethod
    def _vote_row(row: pd.Series, weights: Dict[str, float]) -> Tuple[str, float]:
        vote_scores: Dict[str, float] = {}

        candidates = [
            ("heuristic_prediction", "heuristic_confidence", "heuristic"),
            ("ml_prediction", "ml_confidence", "ml"),
            ("cnn_prediction", "cnn_confidence", "cnn"),
            ("yolo_prediction", "yolo_confidence", "yolo"),
        ]

        for pred_col, conf_col, key in candidates:
            pred = row.get(pred_col, "unknown")
            conf = float(row.get(conf_col, 0.0))
            pred, conf = HybridChartClassifier._disambiguate_hist_bar(row, str(pred), conf, source=key)
            if pred in ("unknown", "", "nan"):
                continue
            score = weights.get(key, 0.0) * _bounded(conf, 0.0, 1.0)
            vote_scores[pred] = vote_scores.get(pred, 0.0) + score

        if not vote_scores:
            return "unknown", 0.0

        best_label = max(vote_scores, key=vote_scores.get)
        sorted_scores = sorted(vote_scores.values(), reverse=True)
        margin = sorted_scores[0] - sorted_scores[1] if len(sorted_scores) > 1 else sorted_scores[0]
        confidence = _bounded(0.5 + margin)
        return best_label, confidence

    def apply(self, df: pd.DataFrame, weights: Dict[str, float] | None = None) -> pd.DataFrame:
        weights = weights or self.DEFAULT_WEIGHTS
        preds = []
        confs = []
        for _, row in df.iterrows():
            pred, conf = self._vote_row(row, weights)
            preds.append(pred)
            confs.append(conf)
        out = df.copy()
        out["hybrid_prediction"] = preds
        out["hybrid_confidence"] = confs
        return out

    @staticmethod
    def evaluate(df: pd.DataFrame, pred_col: str = "hybrid_prediction") -> Dict[str, float]:
        eval_df = df[df["true_chart_type"] != "unknown"].copy()
        y_true = eval_df["true_chart_type"]
        y_pred = eval_df[pred_col]
        return {
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "precision": float(precision_score(y_true, y_pred, average="weighted", zero_division=0)),
            "recall": float(recall_score(y_true, y_pred, average="weighted", zero_division=0)),
            "f1_score": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        }

    def optimize_weights(
        self,
        df: pd.DataFrame,
        target_accuracy: float = 0.90,
    ) -> Tuple[pd.DataFrame, Dict]:
        grid = np.arange(0.1, 0.81, 0.1)
        best_weights = self.DEFAULT_WEIGHTS.copy()
        best_metrics = {"accuracy": -1.0}
        best_df = df.copy()

        for ml_w, cnn_w, yolo_w, heur_w in itertools.product(grid, grid, grid, grid):
            total = ml_w + cnn_w + yolo_w + heur_w
            if abs(total - 1.0) > 1e-6:
                continue
            weights = {
                "ml": float(ml_w),
                "cnn": float(cnn_w),
                "yolo": float(yolo_w),
                "heuristic": float(heur_w),
            }
            pred_df = self.apply(df, weights=weights)
            metrics = self.evaluate(pred_df, pred_col="hybrid_prediction")
            if metrics["accuracy"] > best_metrics["accuracy"]:
                best_weights = weights
                best_metrics = metrics
                best_df = pred_df
            if metrics["accuracy"] >= target_accuracy:
                logger.info("Hybrid target achieved with weights: %s", weights)
                return pred_df, {"weights": weights, "metrics": metrics}

        logger.info("Best hybrid accuracy %.4f with weights %s", best_metrics["accuracy"], best_weights)
        return best_df, {"weights": best_weights, "metrics": best_metrics}
