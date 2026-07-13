"""
Research visualization generator for evaluation and runtime analyses.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


class ResearchPlotGenerator:
    """Generates required plots for publication-ready reporting."""

    def __init__(self, output_dir: str | Path):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        sns.set_theme(style="whitegrid")

    def generate(
        self,
        df: pd.DataFrame,
        metrics: Dict,
        runtime_profile: Dict,
        ml_results: Optional[Dict] = None,
    ) -> int:
        count = 0
        count += int(self._plot_confusion_matrix(metrics))
        count += int(self._plot_ocr_confidence(df))
        count += int(self._plot_runtime_breakdown(runtime_profile))
        count += int(self._plot_chart_type_distribution(df))
        count += int(self._plot_success_rates(metrics))
        count += int(self._plot_feature_importance(ml_results or {}))
        return count

    def _save(self, fig: plt.Figure, filename: str) -> bool:
        path = self.output_dir / filename
        fig.tight_layout()
        fig.savefig(path, dpi=180)
        plt.close(fig)
        return True

    def _plot_confusion_matrix(self, metrics: Dict) -> bool:
        cls = metrics.get("classification", {})
        labels = cls.get("labels", [])
        matrix = cls.get("confusion_matrix", [])
        if not labels or not matrix:
            return False
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels, ax=ax)
        ax.set_title("Confusion Matrix")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        return self._save(fig, "confusion_matrix.png")

    def _plot_ocr_confidence(self, df: pd.DataFrame) -> bool:
        if "ocr_confidence" not in df.columns:
            return False
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.histplot(df["ocr_confidence"].fillna(0.0), bins=30, kde=True, ax=ax, color="#257179")
        ax.set_title("OCR Confidence Distribution")
        ax.set_xlabel("Confidence")
        ax.set_ylabel("Count")
        return self._save(fig, "ocr_confidence_distribution.png")

    def _plot_runtime_breakdown(self, runtime_profile: Dict) -> bool:
        if not runtime_profile:
            return False
        rows = []
        for stage, payload in runtime_profile.items():
            if stage == "overall":
                continue
            rows.append((stage, float(payload.get("mean_seconds", 0.0))))
        if not rows:
            return False
        data = pd.DataFrame(rows, columns=["stage", "mean_seconds"]).sort_values("mean_seconds", ascending=False)
        fig, ax = plt.subplots(figsize=(9, 5))
        sns.barplot(
            data=data,
            x="mean_seconds",
            y="stage",
            hue="stage",
            legend=False,
            ax=ax,
            palette="crest",
        )
        ax.set_title("Runtime Breakdown (Mean Seconds per Image)")
        ax.set_xlabel("Mean Seconds")
        ax.set_ylabel("Pipeline Stage")
        return self._save(fig, "runtime_breakdown.png")

    def _plot_chart_type_distribution(self, df: pd.DataFrame) -> bool:
        if "predicted_chart_type" not in df.columns:
            return False
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.countplot(data=df, x="predicted_chart_type", order=df["predicted_chart_type"].value_counts().index, ax=ax)
        ax.set_title("Chart Type Distribution")
        ax.set_xlabel("Predicted Chart Type")
        ax.set_ylabel("Count")
        ax.tick_params(axis="x", rotation=20)
        return self._save(fig, "chart_type_distribution.png")

    def _plot_success_rates(self, metrics: Dict) -> bool:
        ocr = metrics.get("ocr", {})
        val = metrics.get("value_extraction", {})
        cls = metrics.get("classification", {})
        rows = [
            ("Classification Accuracy", float(cls.get("accuracy", 0.0))),
            ("OCR Coverage", float(ocr.get("ocr_coverage", 0.0))),
            ("Token Recall", float(ocr.get("token_recall", 0.0))),
            ("Value Extraction Success", float(val.get("extraction_success_rate", 0.0))),
        ]
        fig, ax = plt.subplots(figsize=(9, 4))
        data = pd.DataFrame(rows, columns=["metric", "value"])
        sns.barplot(
            data=data,
            x="metric",
            y="value",
            hue="metric",
            legend=False,
            ax=ax,
            palette="mako",
        )
        ax.set_ylim(0, 1.0)
        ax.set_title("Success Rate Comparison")
        ax.set_ylabel("Score")
        ax.tick_params(axis="x", rotation=15)
        return self._save(fig, "success_rate_comparison.png")

    def _plot_feature_importance(self, ml_results: Dict) -> bool:
        features = ml_results.get("feature_importance", [])
        if not features:
            return False
        top = features[:15]
        data = pd.DataFrame(top)
        if data.empty or "feature" not in data.columns or "importance" not in data.columns:
            return False
        fig, ax = plt.subplots(figsize=(9, 6))
        sns.barplot(data=data, y="feature", x="importance", ax=ax, palette="viridis")
        ax.set_title("Feature Importance")
        ax.set_xlabel("Importance")
        ax.set_ylabel("Feature")
        return self._save(fig, "feature_importance.png")

    def dump_runtime_profile(self, runtime_profile: Dict, output_path: str | Path) -> Path:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(runtime_profile, indent=2), encoding="utf-8")
        return path
