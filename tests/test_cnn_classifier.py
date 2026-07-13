from __future__ import annotations

import numpy as np

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
