from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import pandas as pd

from .model import AbstractDetector, DetectionPrediction
from .utils import BoundingBox, load_image


@dataclass(frozen=True)
class DetectionMetrics:
    mAP: float
    precision: float
    recall: float
    mean_iou: float


def _iou(a: Sequence[int], b: Sequence[int]) -> float:
    ax1, ay1, ax2, ay2 = map(float, a)
    bx1, by1, bx2, by2 = map(float, b)
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter_w = max(0.0, ix2 - ix1)
    inter_h = max(0.0, iy2 - iy1)
    inter = inter_w * inter_h
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


class DetectionEvaluator:
    def __init__(self, detector: AbstractDetector, iou_threshold: float = 0.5):
        self.detector = detector
        self.iou_threshold = iou_threshold

    def evaluate_dataframe(self, df: pd.DataFrame) -> DetectionMetrics:
        required = {"image_path", "bbox"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"Evaluation dataframe missing required columns: {sorted(missing)}")

        tp = fp = fn = 0
        ious: list[float] = []
        for _, row in df.iterrows():
            image = load_image(row["image_path"])
            preds = self.detector.predict(image)
            gt = [int(float(v)) for v in str(row["bbox"]).split(",")]
            if preds:
                best = max((_iou(pred.bbox, gt), pred) for pred in preds)[1]
                score = _iou(best.bbox, gt)
                ious.append(score)
                if score >= self.iou_threshold:
                    tp += 1
                    fp += max(0, len(preds) - 1)
                else:
                    fp += len(preds)
                    fn += 1
            else:
                fn += 1

        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        mean_iou = sum(ious) / len(ious) if ious else 0.0
        map_score = precision * recall
        return DetectionMetrics(mAP=map_score, precision=precision, recall=recall, mean_iou=mean_iou)

    def export_metrics(
        self,
        metrics: DetectionMetrics,
        output_dir: str | Path,
        per_class: pd.DataFrame | None = None,
        confusion_matrix: pd.DataFrame | None = None,
    ) -> dict[str, Path]:
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        json_path = out_dir / "metrics.json"
        json_path.write_text(
            pd.Series(
                {
                    "mAP": metrics.mAP,
                    "precision": metrics.precision,
                    "recall": metrics.recall,
                    "iou": metrics.mean_iou,
                }
            ).to_json(indent=2),
            encoding="utf-8",
        )

        csv_path = out_dir / "metrics.csv"
        pd.DataFrame(
            [
                {"metric": "mAP", "value": metrics.mAP},
                {"metric": "precision", "value": metrics.precision},
                {"metric": "recall", "value": metrics.recall},
                {"metric": "iou", "value": metrics.mean_iou},
            ]
        ).to_csv(csv_path, index=False)

        paths = {"json": json_path, "csv": csv_path}

        if per_class is not None:
            per_class_path = out_dir / "per_class_metrics.csv"
            per_class.to_csv(per_class_path, index=False)
            paths["per_class_csv"] = per_class_path

        if confusion_matrix is not None:
            cm_json = out_dir / "confusion_matrix.json"
            cm_csv = out_dir / "confusion_matrix.csv"
            confusion_matrix.to_json(cm_json, orient="split", indent=2)
            confusion_matrix.to_csv(cm_csv)
            paths["confusion_matrix_json"] = cm_json
            paths["confusion_matrix_csv"] = cm_csv

        return paths
