"""
Extended evaluation metrics for research-grade reporting.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


class ResearchEvaluator:
    """Computes classification, OCR, value extraction, and grouped metrics."""

    def evaluate(self, df: pd.DataFrame) -> Dict[str, Any]:
        metrics: Dict[str, Any] = {
            "dataset_size": int(len(df)),
            "classification": {},
            "ocr": {},
            "value_extraction": {},
            "per_chart_type_accuracy": {},
            "per_dataset_source_accuracy": {},
        }
        if df.empty:
            return metrics

        self._classification_metrics(df, metrics)
        self._ocr_metrics(df, metrics)
        self._value_metrics(df, metrics)
        self._grouped_accuracy(df, metrics)
        return metrics

    def _classification_metrics(self, df: pd.DataFrame, metrics: Dict[str, Any]) -> None:
        if not {"true_chart_type", "predicted_chart_type"}.issubset(df.columns):
            return
        valid = df[df["true_chart_type"] != "unknown"].copy()
        if valid.empty:
            return

        y_true = valid["true_chart_type"].astype(str)
        y_pred = valid["predicted_chart_type"].astype(str)
        labels = sorted(set(y_true.tolist() + y_pred.tolist()))
        cm = confusion_matrix(y_true, y_pred, labels=labels)

        metrics["classification"] = {
            "accuracy": _safe_float(accuracy_score(y_true, y_pred)),
            "precision": _safe_float(precision_score(y_true, y_pred, average="weighted", zero_division=0)),
            "recall": _safe_float(recall_score(y_true, y_pred, average="weighted", zero_division=0)),
            "f1_score": _safe_float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
            "labels": labels,
            "confusion_matrix": cm.tolist(),
        }

    def _ocr_metrics(self, df: pd.DataFrame, metrics: Dict[str, Any]) -> None:
        ocr: Dict[str, Any] = {}
        if "ocr_confidence" in df.columns:
            ocr["average_ocr_confidence"] = _safe_float(df["ocr_confidence"].fillna(0.0).mean())
        elif "best_ocr_confidence" in df.columns:
            ocr["average_ocr_confidence"] = _safe_float(df["best_ocr_confidence"].fillna(0.0).mean())
        else:
            ocr["average_ocr_confidence"] = 0.0

        if "ocr_token_recall" in df.columns:
            ocr["token_recall"] = _safe_float(df["ocr_token_recall"].fillna(0.0).mean())
        elif "ensemble_token_count" in df.columns and "union_token_count" in df.columns:
            union = df["union_token_count"].replace(0, np.nan)
            recalls = (df["ensemble_token_count"] / union).fillna(0.0)
            ocr["token_recall"] = _safe_float(recalls.mean())
        else:
            ocr["token_recall"] = 0.0

        if "ocr_text_length" in df.columns:
            ocr["ocr_coverage"] = _safe_float((df["ocr_text_length"].fillna(0) > 0).mean())
        else:
            ocr["ocr_coverage"] = 0.0

        if "paddle_success_rate" in df.columns:
            ocr["paddle_success_rate"] = _safe_float(df["paddle_success_rate"].fillna(0.0).iloc[-1])
        if "easyocr_success_rate" in df.columns:
            ocr["easyocr_success_rate"] = _safe_float(df["easyocr_success_rate"].fillna(0.0).iloc[-1])
        if "fallback_rate" in df.columns:
            ocr["fallback_rate"] = _safe_float(df["fallback_rate"].fillna(0.0).iloc[-1])
        metrics["ocr"] = ocr

    def _value_metrics(self, df: pd.DataFrame, metrics: Dict[str, Any]) -> None:
        extraction = {}
        if "value_extraction_success" in df.columns:
            extraction["extraction_success_rate"] = _safe_float(df["value_extraction_success"].fillna(0).mean())
        else:
            extraction["extraction_success_rate"] = 0.0

        # Proxy numeric extraction quality when exact numeric ground truth is unavailable.
        if "numeric_extraction_accuracy" in df.columns:
            extraction["numeric_extraction_accuracy"] = _safe_float(
                df["numeric_extraction_accuracy"].fillna(0.0).mean()
            )
        elif "numeric_values_found" in df.columns and "numeric_values_total" in df.columns:
            total = df["numeric_values_total"].sum()
            found = df["numeric_values_found"].sum()
            extraction["numeric_extraction_accuracy"] = float(found / total) if total else 0.0
        else:
            extraction["numeric_extraction_accuracy"] = 0.0

        metrics["value_extraction"] = extraction

    def _grouped_accuracy(self, df: pd.DataFrame, metrics: Dict[str, Any]) -> None:
        if "classification_correct" in df.columns and "true_chart_type" in df.columns:
            by_type = df.groupby("true_chart_type")["classification_correct"].mean().fillna(0.0).to_dict()
            metrics["per_chart_type_accuracy"] = {str(k): _safe_float(v) for k, v in by_type.items()}
        elif "classification_correct" in df.columns and "predicted_chart_type" in df.columns:
            by_type = (
                df.groupby("predicted_chart_type")["classification_correct"].mean().fillna(0.0).to_dict()
            )
            metrics["per_chart_type_accuracy"] = {str(k): _safe_float(v) for k, v in by_type.items()}

        if "classification_correct" in df.columns and "dataset_source" in df.columns:
            by_source = df.groupby("dataset_source")["classification_correct"].mean().fillna(0.0).to_dict()
            metrics["per_dataset_source_accuracy"] = {str(k): _safe_float(v) for k, v in by_source.items()}

    def save(self, metrics: Dict[str, Any], output_path: str | Path) -> Path:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        return path
