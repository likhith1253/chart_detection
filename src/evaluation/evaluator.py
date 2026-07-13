"""
Pipeline Evaluator — Research-Grade.
Computes comprehensive research metrics:
  - Classification accuracy / precision / recall / F1
  - Confusion matrix
  - OCR accuracy statistics
  - Value extraction accuracy
  - Chart type distribution
  - Processing time statistics
Saves all results to results/research_metrics/.
"""

import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)

logger = logging.getLogger(__name__)


class PipelineEvaluator:
    """Accumulates per-image metrics and generates research reports."""

    def __init__(self, output_dir: Path = None):
        import config
        self.output_dir = output_dir or config.METRICS_DIR
        self.research_dir = config.RESEARCH_METRICS_DIR
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.research_dir.mkdir(parents=True, exist_ok=True)
        self.records: List[Dict] = []

    def log_result(self, record: Dict):
        """Append a single image result record."""
        self.records.append(record)

    def save(self, filename: str = "experiment_results.csv"):
        """Write accumulated records to CSV."""
        csv_path = self.output_dir / filename
        df = pd.DataFrame(self.records)
        df.to_csv(csv_path, index=False)
        logger.info(f"Saved {len(self.records)} records to {csv_path}")
        return csv_path

    # ─── Aggregate metrics ────────────────────────────────────

    def compute_summary(self) -> Dict:
        """Compute aggregate summary statistics."""
        if not self.records:
            return {}

        df = pd.DataFrame(self.records)
        summary = {
            "total_images": len(df),
            "avg_processing_time": float(df["processing_time_sec"].mean()) if "processing_time_sec" in df.columns else 0,
            "classification_accuracy": 0.0,
            "ocr_avg_text_length": 0.0,
            "value_extraction_success_rate": 0.0,
            "chart_type_distribution": {},
        }

        if "classification_correct" in df.columns:
            summary["classification_accuracy"] = float(df["classification_correct"].mean())
        if "predicted_chart_type" in df.columns:
            summary["chart_type_distribution"] = df["predicted_chart_type"].value_counts().to_dict()
        if "ocr_text_length" in df.columns:
            summary["ocr_avg_text_length"] = float(df["ocr_text_length"].mean())
        if "value_extraction_success" in df.columns:
            summary["value_extraction_success_rate"] = float(df["value_extraction_success"].mean())

        return summary

    # ─── Research Metrics ─────────────────────────────────────

    def compute_research_metrics(self, df: pd.DataFrame) -> Dict:
        """
        Compute comprehensive research metrics and save to results/research_metrics/.
        """
        metrics = {
            "dataset_size": len(df),
            "classification": {},
            "ocr": {},
            "value_extraction": {},
            "chart_type_distribution": {},
            "processing_time": {},
            "runtime_stages": {},
        }

        # ── Classification metrics ──
        if "true_chart_type" in df.columns and "predicted_chart_type" in df.columns:
            mask = df["true_chart_type"] != "unknown"
            if mask.sum() > 0:
                y_true = df.loc[mask, "true_chart_type"]
                y_pred = df.loc[mask, "predicted_chart_type"]

                metrics["classification"]["accuracy"] = round(
                    accuracy_score(y_true, y_pred), 4
                )
                metrics["classification"]["precision"] = round(
                    precision_score(y_true, y_pred, average="weighted", zero_division=0), 4
                )
                metrics["classification"]["recall"] = round(
                    recall_score(y_true, y_pred, average="weighted", zero_division=0), 4
                )
                metrics["classification"]["f1_score"] = round(
                    f1_score(y_true, y_pred, average="weighted", zero_division=0), 4
                )

                # Confusion matrix
                labels = sorted(set(y_true.tolist() + y_pred.tolist()))
                cm = confusion_matrix(y_true, y_pred, labels=labels)
                metrics["classification"]["confusion_matrix"] = cm.tolist()
                metrics["classification"]["class_labels"] = labels

                # Per-class report
                report = classification_report(
                    y_true, y_pred, labels=labels,
                    zero_division=0, output_dict=True
                )
                metrics["classification"]["per_class"] = report

        # ── OCR metrics ──
        if "easyocr_confidence" in df.columns:
            metrics["ocr"]["avg_easyocr_confidence"] = round(
                df["easyocr_confidence"].mean(), 4
            )
        if "paddleocr_confidence" in df.columns:
            metrics["ocr"]["avg_paddleocr_confidence"] = round(
                df["paddleocr_confidence"].mean(), 4
            )
        if "best_ocr_accuracy" in df.columns:
            metrics["ocr"]["avg_best_confidence"] = round(
                df["best_ocr_accuracy"].mean(), 4
            )
        if "ocr_text_length" in df.columns:
            metrics["ocr"]["avg_text_length"] = round(
                df["ocr_text_length"].mean(), 2
            )
            metrics["ocr"]["median_text_length"] = round(
                df["ocr_text_length"].median(), 2
            )

        # ── Value extraction ──
        if "value_extraction_success" in df.columns:
            metrics["value_extraction"]["overall_success_rate"] = round(
                df["value_extraction_success"].mean() * 100, 2
            )
            # Per chart type
            if "predicted_chart_type" in df.columns:
                per_type = {}
                for ct in df["predicted_chart_type"].unique():
                    mask = df["predicted_chart_type"] == ct
                    rate = df.loc[mask, "value_extraction_success"].mean() * 100
                    per_type[ct] = round(rate, 2)
                metrics["value_extraction"]["per_chart_type"] = per_type

        # ── Chart type distribution ──
        if "predicted_chart_type" in df.columns:
            dist = df["predicted_chart_type"].value_counts().to_dict()
            metrics["chart_type_distribution"] = dist

        # ── Processing time stats ──
        if "processing_time_sec" in df.columns:
            times = df["processing_time_sec"]
            metrics["processing_time"] = {
                "mean": round(float(times.mean()), 4),
                "median": round(float(times.median()), 4),
                "std": round(float(times.std()), 4),
                "min": round(float(times.min()), 4),
                "max": round(float(times.max()), 4),
                "total": round(float(times.sum()), 2),
            }

        stage_cols = [
            "image_loading_time",
            "feature_extraction_time",
            "yolo_detection_time",
            "ocr_time",
            "value_extraction_time",
            "classification_time",
            "total_pipeline_time",
        ]
        available_stage_cols = [c for c in stage_cols if c in df.columns]
        if available_stage_cols:
            metrics["runtime_stages"] = {
                c: round(float(pd.to_numeric(df[c], errors="coerce").fillna(0.0).mean()), 6)
                for c in available_stage_cols
            }

        # Save to JSON
        metrics_path = self.research_dir / "research_metrics.json"
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=2, default=str)
        logger.info(f"Research metrics saved to {metrics_path}")

        # Save confusion matrix CSV
        if "confusion_matrix" in metrics.get("classification", {}):
            cm = metrics["classification"]["confusion_matrix"]
            labels = metrics["classification"]["class_labels"]
            cm_df = pd.DataFrame(cm, index=labels, columns=labels)
            cm_path = self.research_dir / "confusion_matrix.csv"
            cm_df.to_csv(cm_path)
            logger.info(f"Confusion matrix saved to {cm_path}")

        # Save summary CSV
        summary_rows = []
        summary_rows.append({"metric": "dataset_size", "value": metrics["dataset_size"]})
        for k, v in metrics.get("classification", {}).items():
            if isinstance(v, (int, float)):
                summary_rows.append({"metric": f"classification_{k}", "value": v})
        for k, v in metrics.get("ocr", {}).items():
            summary_rows.append({"metric": f"ocr_{k}", "value": v})
        for k, v in metrics.get("value_extraction", {}).items():
            if isinstance(v, (int, float)):
                summary_rows.append({"metric": f"value_extraction_{k}", "value": v})
        for k, v in metrics.get("processing_time", {}).items():
            summary_rows.append({"metric": f"time_{k}", "value": v})

        summary_df = pd.DataFrame(summary_rows)
        summary_path = self.research_dir / "system_metrics.csv"
        summary_df.to_csv(summary_path, index=False)
        logger.info(f"System metrics saved to {summary_path}")

        return metrics
