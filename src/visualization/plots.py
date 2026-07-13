"""
Visualization Module — Research-Grade.
Generates research plots:
  - Chart type distribution
  - OCR confidence comparison / distribution
  - Processing time distribution
  - Dataset composition
  - Feature distribution
  - Confusion matrix heatmap
  - Feature importance bar chart
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class PlotGenerator:
    """Generates research plots from experiment results."""

    def __init__(self, output_dir: Path):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        sns.set_theme(style="whitegrid", palette="muted")

    def generate_all_plots(self, csv_path: Path, ml_results: dict = None) -> int:
        """Generate all research plots. Returns count of plots saved."""
        csv_path = Path(csv_path)
        if not csv_path.exists():
            logger.error(f"CSV not found: {csv_path}")
            return 0

        count = 0
        try:
            df = pd.read_csv(csv_path)

            if self._plot_type_distribution(df):
                count += 1
            if self._plot_ocr_accuracy(df):
                count += 1
            if self._plot_ocr_confidence_distribution(df):
                count += 1
            if self._plot_time_distribution(df):
                count += 1
            if self._plot_dataset_composition(df):
                count += 1
            if self._plot_feature_distribution(df):
                count += 1
            if self._plot_confusion_matrix(df):
                count += 1
            if ml_results and self._plot_feature_importance(ml_results):
                count += 1

            logger.info(f"Generated {count} plots in {self.output_dir}")

        except Exception as e:
            logger.error(f"Plot generation failed: {e}")

        return count

    def _plot_type_distribution(self, df: pd.DataFrame) -> bool:
        """Bar chart of predicted chart types."""
        try:
            if "predicted_chart_type" not in df.columns:
                return False
            plt.figure(figsize=(10, 6))
            order = df["predicted_chart_type"].value_counts().index
            ax = sns.countplot(data=df, x="predicted_chart_type", order=order,
                               palette="viridis")
            plt.title("Distribution of Predicted Chart Types", fontsize=14, fontweight="bold")
            plt.xlabel("Chart Type", fontsize=12)
            plt.ylabel("Count", fontsize=12)
            for p in ax.patches:
                ax.annotate(f"{int(p.get_height())}",
                            (p.get_x() + p.get_width() / 2., p.get_height()),
                            ha='center', va='bottom', fontsize=10, fontweight='bold')
            plt.tight_layout()
            plt.savefig(self.output_dir / "chart_type_distribution.png", dpi=200)
            plt.close()
            return True
        except Exception as e:
            logger.error(f"Type distribution plot failed: {e}")
            plt.close()
            return False

    def _plot_ocr_accuracy(self, df: pd.DataFrame) -> bool:
        """Box plot of OCR engine comparison."""
        try:
            cols = [c for c in ["easyocr_confidence", "paddleocr_confidence",
                                "best_ocr_accuracy"] if c in df.columns]
            if not cols:
                return False
            plt.figure(figsize=(10, 6))
            melt_df = df[cols].melt(var_name="OCR Engine", value_name="Confidence")
            sns.boxplot(data=melt_df, x="OCR Engine", y="Confidence", palette="Set2")
            plt.title("OCR Confidence Comparison", fontsize=14, fontweight="bold")
            plt.ylabel("Confidence Score", fontsize=12)
            plt.ylim(0, 1.1)
            plt.tight_layout()
            plt.savefig(self.output_dir / "ocr_accuracy_comparison.png", dpi=200)
            plt.close()
            return True
        except Exception as e:
            logger.error(f"OCR accuracy plot failed: {e}")
            plt.close()
            return False

    def _plot_ocr_confidence_distribution(self, df: pd.DataFrame) -> bool:
        """Histogram of OCR confidence scores."""
        try:
            cols = [c for c in ["easyocr_confidence", "paddleocr_confidence"] if c in df.columns]
            if not cols:
                return False
            fig, axes = plt.subplots(1, len(cols), figsize=(6 * len(cols), 5))
            if len(cols) == 1:
                axes = [axes]
            colors = ["steelblue", "coral"]
            for ax, col, color in zip(axes, cols, colors):
                data = df[col].dropna()
                data = data[data > 0]
                if len(data) > 0:
                    sns.histplot(data=data, bins=30, kde=True, ax=ax, color=color, alpha=0.7)
                    ax.set_title(col.replace("_", " ").title(), fontsize=12)
                    ax.set_xlabel("Confidence", fontsize=11)
                    ax.axvline(data.mean(), color="red", linestyle="--",
                               label=f"Mean: {data.mean():.3f}")
                    ax.legend()
            plt.suptitle("OCR Confidence Distribution", fontsize=14, fontweight="bold")
            plt.tight_layout()
            plt.savefig(self.output_dir / "ocr_confidence_distribution.png", dpi=200)
            plt.close()
            return True
        except Exception as e:
            logger.error(f"OCR confidence distribution plot failed: {e}")
            plt.close()
            return False

    def _plot_time_distribution(self, df: pd.DataFrame) -> bool:
        """Histogram + KDE of processing times."""
        try:
            if "processing_time_sec" not in df.columns:
                return False
            plt.figure(figsize=(10, 6))
            sns.histplot(data=df, x="processing_time_sec", bins=30, kde=True,
                         color="steelblue", alpha=0.7)
            plt.title("Pipeline Processing Time Distribution", fontsize=14, fontweight="bold")
            plt.xlabel("Time (seconds)", fontsize=12)
            plt.ylabel("Frequency", fontsize=12)
            mean_t = df["processing_time_sec"].mean()
            plt.axvline(mean_t, color="red", linestyle="--", linewidth=1.5,
                        label=f"Mean: {mean_t:.2f}s")
            plt.legend(fontsize=11)
            plt.tight_layout()
            plt.savefig(self.output_dir / "processing_time_distribution.png", dpi=200)
            plt.close()
            return True
        except Exception as e:
            logger.error(f"Time distribution plot failed: {e}")
            plt.close()
            return False

    def _plot_dataset_composition(self, df: pd.DataFrame) -> bool:
        """Pie chart of dataset sources."""
        try:
            if "dataset_source" not in df.columns:
                return False
            plt.figure(figsize=(8, 8))
            counts = df["dataset_source"].value_counts()
            colors = sns.color_palette("pastel", len(counts))
            plt.pie(counts.values, labels=counts.index, autopct="%1.1f%%",
                    colors=colors, startangle=140, shadow=True)
            plt.title("Dataset Composition", fontsize=14, fontweight="bold")
            plt.tight_layout()
            plt.savefig(self.output_dir / "dataset_composition.png", dpi=200)
            plt.close()
            return True
        except Exception as e:
            logger.error(f"Dataset composition plot failed: {e}")
            plt.close()
            return False

    def _plot_feature_distribution(self, df: pd.DataFrame) -> bool:
        """KDE plots for key geometric features."""
        try:
            features = [c for c in df.columns if c.startswith("feature_")]
            core = ["feature_bar_count", "feature_edge_density",
                     "feature_texture_entropy", "feature_point_density"]
            present = [f for f in core if f in features]
            if len(present) < 2:
                present = features[:4]
            if len(present) < 2:
                return False

            fig, axes = plt.subplots(1, len(present), figsize=(5 * len(present), 5))
            if len(present) == 1:
                axes = [axes]
            for ax, feat in zip(axes, present):
                sns.histplot(data=df, x=feat, kde=True, ax=ax, color="coral", alpha=0.6)
                ax.set_title(feat.replace("feature_", "").replace("_", " ").title(),
                             fontsize=11)
            plt.suptitle("Geometric Feature Distributions", fontsize=14, fontweight="bold")
            plt.tight_layout()
            plt.savefig(self.output_dir / "feature_distribution.png", dpi=200)
            plt.close()
            return True
        except Exception as e:
            logger.error(f"Feature distribution plot failed: {e}")
            plt.close()
            return False

    def _plot_confusion_matrix(self, df: pd.DataFrame) -> bool:
        """Heatmap confusion matrix."""
        try:
            if "true_chart_type" not in df.columns or "predicted_chart_type" not in df.columns:
                return False
            mask = df["true_chart_type"] != "unknown"
            if mask.sum() < 10:
                return False

            y_true = df.loc[mask, "true_chart_type"]
            y_pred = df.loc[mask, "predicted_chart_type"]
            labels = sorted(set(y_true.tolist() + y_pred.tolist()))

            from sklearn.metrics import confusion_matrix
            cm = confusion_matrix(y_true, y_pred, labels=labels)

            plt.figure(figsize=(10, 8))
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                        xticklabels=labels, yticklabels=labels)
            plt.title("Classification Confusion Matrix", fontsize=14, fontweight="bold")
            plt.xlabel("Predicted", fontsize=12)
            plt.ylabel("True", fontsize=12)
            plt.tight_layout()
            plt.savefig(self.output_dir / "confusion_matrix.png", dpi=200)
            plt.close()
            return True
        except Exception as e:
            logger.error(f"Confusion matrix plot failed: {e}")
            plt.close()
            return False

    def _plot_feature_importance(self, ml_results: dict) -> bool:
        """Bar chart of feature importances from RandomForest."""
        try:
            feat_imp = ml_results.get("feature_importance", [])
            if not feat_imp:
                return False

            names = [f["feature"] for f in feat_imp[:15]]
            values = [f["importance"] for f in feat_imp[:15]]

            plt.figure(figsize=(10, 8))
            colors = sns.color_palette("viridis", len(names))
            bars = plt.barh(range(len(names)), values, color=colors)
            plt.yticks(range(len(names)), names)
            plt.xlabel("Importance", fontsize=12)
            plt.title("Top Feature Importances (RandomForest)", fontsize=14, fontweight="bold")
            plt.gca().invert_yaxis()
            plt.tight_layout()
            plt.savefig(self.output_dir / "feature_importance.png", dpi=200)
            plt.close()
            return True
        except Exception as e:
            logger.error(f"Feature importance plot failed: {e}")
            plt.close()
            return False
