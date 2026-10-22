from __future__ import annotations

import csv
from pathlib import Path

import pandas as pd
import pytest
from PIL import Image

from src.detection import BenchmarkDetectionDatasetBuilder, DetectorTrainingConfig, YoloTrainingBridge, load_manifest
from src.detection.evaluate import DetectionMetrics


def _write_image(path: Path, size: tuple[int, int] = (32, 24)) -> None:
    Image.new("RGB", size, color="white").save(path)


@pytest.mark.skipif(
    not (Path("data") / "datasets" / "benchmark_manifest.csv").exists(),
    reason="needs the local (git-ignored) benchmark manifest",
)
def test_manifest_survey_reports_current_datasets() -> None:
    df = pd.read_csv(Path("data") / "datasets" / "benchmark_manifest.csv")
    datasets = set(df["dataset"].astype(str).str.lower().unique())
    assert {"chartqa", "plotqa", "dvqa", "figureqa"}.issubset(datasets)


def test_training_bridge_exports_artifacts(tmp_path) -> None:
    img1 = tmp_path / "train.png"
    img2 = tmp_path / "validation.png"
    _write_image(img1)
    _write_image(img2)

    manifest_csv = tmp_path / "benchmark_manifest.csv"
    with manifest_csv.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=["image_id", "dataset", "image_path", "chart_type", "split", "license", "width", "height"],
        )
        writer.writeheader()
        writer.writerow(
            {
                "image_id": "img1",
                "dataset": "chartqa",
                "image_path": str(img1),
                "chart_type": "bar",
                "split": "train",
                "license": "unknown",
                "width": 32,
                "height": 24,
            }
        )
        writer.writerow(
            {
                "image_id": "img2",
                "dataset": "figureqa",
                "image_path": str(img2),
                "chart_type": "line",
                "split": "validation",
                "license": "unknown",
                "width": 32,
                "height": 24,
            }
        )
        writer.writerow(
            {
                "image_id": "img3",
                "dataset": "dvqa",
                "image_path": str(img1),
                "chart_type": "pie",
                "split": "test",
                "license": "unknown",
                "width": 32,
                "height": 24,
            }
        )

    class StubTrainer:
        def __init__(self):
            self.calls = []

        def train(self, **kwargs):
            self.calls.append(kwargs)
            return {"trained": True}

    class StubEvaluator:
        def evaluate_dataframe(self, df):
            assert not df.empty
            return DetectionMetrics(mAP=0.7, precision=0.8, recall=0.6, mean_iou=0.5)

    bridge = YoloTrainingBridge(
        DetectorTrainingConfig(
            manifest_csv=manifest_csv,
            output_dir=tmp_path / "out",
            epochs=3,
            batch_size=2,
            model_name="yolov11n.pt",
        ),
        trainer_factory=lambda cfg: StubTrainer(),
        evaluator=StubEvaluator(),
    )
    state = bridge.train()

    assert state.best_metric == 0.7
    assert state.latest_checkpoint is not None and state.latest_checkpoint.exists()
    assert state.best_checkpoint is not None and state.best_checkpoint.exists()
    assert (tmp_path / "out" / "training_history.csv").exists()
    assert (tmp_path / "out" / "epoch_metrics.csv").exists()
    assert (tmp_path / "out" / "metrics.json").exists()
    assert (tmp_path / "out" / "manifest_statistics.json").exists()
    assert (tmp_path / "out" / "manifest_validation_report.json").exists()


def test_dataset_builder_reads_manifest_splits(tmp_path) -> None:
    img = tmp_path / "page.png"
    _write_image(img)
    csv_path = tmp_path / "manifest.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=["image_id", "dataset", "image_path", "chart_type", "split", "license", "width", "height"],
        )
        writer.writeheader()
        writer.writerow(
            {
                "image_id": "x1",
                "dataset": "chartqa",
                "image_path": str(img),
                "chart_type": "bar",
                "split": "train",
                "license": "unknown",
                "width": 32,
                "height": 24,
            }
        )

    dataset = BenchmarkDetectionDatasetBuilder(csv_path).build("train")
    assert len(dataset.samples) == 1
    assert dataset.samples[0].image_id == "x1"
    assert dataset.annotations[0].label == "bar"
