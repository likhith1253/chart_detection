from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


class EvaluationVisualizer:
    def error_histogram(self, values: list[float], output_path: str | Path) -> Path:
        return self._save_plot(values, output_path, kind="hist")

    def confusion_matrix(self, matrix: list[list[float]], labels: list[str], output_path: str | Path) -> Path:
        fig, ax = plt.subplots(figsize=(6, 5))
        arr = np.asarray(matrix, dtype=float)
        im = ax.imshow(arr, cmap="Blues")
        ax.set_xticks(range(len(labels)), labels=labels, rotation=45, ha="right")
        ax.set_yticks(range(len(labels)), labels=labels)
        fig.colorbar(im, ax=ax)
        return self._finalize(fig, output_path)

    def scatter_plot(self, x: list[float], y: list[float], output_path: str | Path) -> Path:
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.scatter(x, y, alpha=0.7)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Ground Truth")
        return self._finalize(fig, output_path)

    def performance_curve(self, x: list[float], y: list[float], output_path: str | Path) -> Path:
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.plot(x, y, marker="o")
        ax.set_xlabel("Threshold")
        ax.set_ylabel("Performance")
        return self._finalize(fig, output_path)

    def _save_plot(self, values: list[float], output_path: str | Path, kind: str) -> Path:
        fig, ax = plt.subplots(figsize=(6, 5))
        if kind == "hist":
            ax.hist(values, bins=min(20, max(5, len(values))))
        return self._finalize(fig, output_path)

    def _finalize(self, fig: Any, output_path: str | Path) -> Path:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.tight_layout()
        fig.savefig(path)
        plt.close(fig)
        return path

