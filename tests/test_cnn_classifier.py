from __future__ import annotations

import numpy as np
import pandas as pd

from src.research.cnn_classifier import CNNChartClassifier


def test_confusion_figure_array_uses_buffer_rgba(tmp_path) -> None:
    classifier = CNNChartClassifier(output_dir=tmp_path / "cnn")
    arr = classifier._confusion_figure_array([[3, 1], [0, 4]])

    assert isinstance(arr, np.ndarray)
    assert arr.dtype == np.uint8
    assert arr.ndim == 3
    assert arr.shape[2] == 3
    assert arr.shape[0] > 0
    assert arr.shape[1] > 0


def test_training_csvs_are_written_with_rows(tmp_path) -> None:
    classifier = CNNChartClassifier(output_dir=tmp_path / "cnn")
    history = [
        {"epoch": i + 1, "train_loss": float(i), "val_accuracy": 0.5 + i * 0.01}
        for i in range(6)
    ]
    epoch_rows = [
        {"epoch": i + 1, "train_loss": float(i), "val_accuracy": 0.5 + i * 0.01}
        for i in range(6)
    ]
    system_rows = [
        {"epoch": i + 1, "cpu_usage": 10.0 + i, "ram_usage": 20.0 + i}
        for i in range(6)
    ]

    classifier._write_training_csvs(history, epoch_rows, system_rows)

    for filename in ["training_history.csv", "epoch_metrics.csv", "training_metrics.csv"]:
        df = pd.read_csv(tmp_path / "cnn" / filename)
        assert len(df) == 6
        assert not df.empty
