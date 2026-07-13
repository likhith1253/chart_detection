"""
Visualization module for chart research results.
Generates matplotlib plots for OCR comparison, processing time, and chart type distribution.
"""

import logging
from pathlib import Path
from typing import List, Dict, Any

import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for CPU-only environments
import matplotlib.pyplot as plt
import numpy as np

logger = logging.getLogger(__name__)


class ResultVisualizer:
    """Generates research-grade visualizations from pipeline results."""

    def __init__(self, output_dir: str | Path):
        """
        Args:
            output_dir: Directory to save plot images.
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def generate_all(self, pipeline_results: List[Dict[str, Any]]) -> None:
        """
        Generates all visualization plots from pipeline results.

        Args:
            pipeline_results: List of per-image result dicts from the pipeline.
        """
        if not pipeline_results:
            logger.warning("No results to visualize.")
            return

        try:
            self.plot_ocr_accuracy(pipeline_results)
        except Exception as e:
            logger.error(f"Failed to generate OCR accuracy plot: {e}")

        try:
            self.plot_processing_time(pipeline_results)
        except Exception as e:
            logger.error(f"Failed to generate processing time plot: {e}")

        try:
            self.plot_chart_type_distribution(pipeline_results)
        except Exception as e:
            logger.error(f"Failed to generate chart type distribution plot: {e}")

        logger.info(f"Visualizations saved to {self.output_dir}")

    def plot_ocr_accuracy(self, results: List[Dict[str, Any]]) -> None:
        """
        Generates a grouped bar chart comparing EasyOCR vs PaddleOCR accuracy.
        """
        image_names = []
        easyocr_scores = []
        paddleocr_scores = []

        for r in results:
            image_names.append(r.get("image_name", "unknown"))
            scores = r.get("evaluation_scores", {})
            easyocr_scores.append(scores.get("easyocr_score", 0.0))
            paddleocr_scores.append(scores.get("paddleocr_score", 0.0))

        x = np.arange(len(image_names))
        width = 0.35

        fig, ax = plt.subplots(figsize=(10, 6))
        bars1 = ax.bar(x - width / 2, easyocr_scores, width, label='EasyOCR', color='#4C72B0', alpha=0.85)
        bars2 = ax.bar(x + width / 2, paddleocr_scores, width, label='PaddleOCR', color='#DD8452', alpha=0.85)

        ax.set_xlabel('Image', fontsize=12)
        ax.set_ylabel('Accuracy Score', fontsize=12)
        ax.set_title('OCR Engine Accuracy Comparison', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(image_names, rotation=45, ha='right', fontsize=9)
        ax.set_ylim(0, 1.1)
        ax.legend()
        ax.grid(axis='y', linestyle='--', alpha=0.4)

        # Add value labels on bars
        for bar in bars1:
            height = bar.get_height()
            if height > 0:
                ax.text(bar.get_x() + bar.get_width() / 2., height + 0.02,
                        f'{height:.2f}', ha='center', va='bottom', fontsize=8)
        for bar in bars2:
            height = bar.get_height()
            if height > 0:
                ax.text(bar.get_x() + bar.get_width() / 2., height + 0.02,
                        f'{height:.2f}', ha='center', va='bottom', fontsize=8)

        plt.tight_layout()
        path = self.output_dir / "ocr_accuracy_comparison.png"
        plt.savefig(path, dpi=150)
        plt.close()
        logger.info(f"Saved: {path}")

    def plot_processing_time(self, results: List[Dict[str, Any]]) -> None:
        """
        Generates a bar chart showing processing time per image.
        """
        image_names = []
        times = []

        for r in results:
            image_names.append(r.get("image_name", "unknown"))
            times.append(r.get("processing_time_sec", 0.0))

        if all(t == 0 for t in times):
            logger.info("No processing time data available, skipping plot.")
            return

        fig, ax = plt.subplots(figsize=(10, 6))
        colors = ['#55A868' if t < np.mean(times) else '#C44E52' for t in times]
        bars = ax.bar(image_names, times, color=colors, alpha=0.85)

        ax.set_xlabel('Image', fontsize=12)
        ax.set_ylabel('Processing Time (seconds)', fontsize=12)
        ax.set_title('Pipeline Processing Time per Image', fontsize=14, fontweight='bold')
        ax.set_xticklabels(image_names, rotation=45, ha='right', fontsize=9)
        ax.grid(axis='y', linestyle='--', alpha=0.4)

        # Add value labels
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2., height + 0.05,
                    f'{height:.2f}s', ha='center', va='bottom', fontsize=9)

        plt.tight_layout()
        path = self.output_dir / "processing_time_comparison.png"
        plt.savefig(path, dpi=150)
        plt.close()
        logger.info(f"Saved: {path}")

    def plot_chart_type_distribution(self, results: List[Dict[str, Any]]) -> None:
        """
        Generates a pie chart showing the distribution of detected chart types.
        """
        type_counts: Dict[str, int] = {}
        for r in results:
            ct = r.get("classification", {}).get("chart_type", "unknown")
            type_counts[ct] = type_counts.get(ct, 0) + 1

        labels = list(type_counts.keys())
        sizes = list(type_counts.values())
        colors = ['#4C72B0', '#55A868', '#C44E52', '#8172B2', '#CCB974', '#64B5CD']

        fig, ax = plt.subplots(figsize=(8, 6))
        wedges, texts, autotexts = ax.pie(
            sizes, labels=labels, autopct='%1.1f%%',
            colors=colors[:len(labels)],
            startangle=140, pctdistance=0.85
        )

        # Style
        for text in texts:
            text.set_fontsize(11)
        for autotext in autotexts:
            autotext.set_fontsize(10)
            autotext.set_fontweight('bold')

        ax.set_title('Chart Type Distribution', fontsize=14, fontweight='bold')

        plt.tight_layout()
        path = self.output_dir / "chart_type_distribution.png"
        plt.savefig(path, dpi=150)
        plt.close()
        logger.info(f"Saved: {path}")
