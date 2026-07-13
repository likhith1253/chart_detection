"""
Benchmark Experiment Framework.
Runs the full pipeline across the dataset and records high-level metrics
(classification, OCR, features, processing time) to CSV.
"""

import sys
import time
import pandas as pd
import logging
from pathlib import Path
from tqdm import tqdm

project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import config
from src.classifier.chart_classifier import ChartClassifier
from src.ocr.ocr_engine import OCREngine
from src.utils.evaluator import OCREvaluator
from main import load_ground_truth

logger = logging.getLogger(__name__)


class BenchmarkRunner:
    """Runs tests on dataset to analyze capability over time."""

    def __init__(self):
        self.classifier = ChartClassifier()
        self.ocr_engine = OCREngine()
        self.evaluator = OCREvaluator()
        
        self.results_dir = config.RESULT_DIR / "metrics"
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.output_csv = self.results_dir / "experiment_results.csv"

    def run_benchmark(self):
        """Runs the pipeline on all images and logs metrics."""
        images = (
            list(config.RAW_IMAGE_DIR.glob("*.png")) +
            list(config.RAW_IMAGE_DIR.glob("*.jpg")) +
            list(config.RAW_IMAGE_DIR.glob("*.jpeg"))
        )

        if not images:
            print(f"No images found in {config.RAW_IMAGE_DIR}")
            return

        print(f"Starting benchmark on {len(images)} images...")
        
        records = []

        for img_path in tqdm(images, desc="Benchmarking"):
            img_str = str(img_path)
            img_name = img_path.name
            start_time = time.time()
            
            # Ground truth for chart type accuracy
            true_chart_type = "unknown"
            if "bar" in img_name.lower(): true_chart_type = "bar_chart"
            elif "pie" in img_name.lower(): true_chart_type = "pie_chart"
            elif "line" in img_name.lower(): true_chart_type = "line_chart"
            elif "scatter" in img_name.lower(): true_chart_type = "scatter_plot"
            elif "histogram" in img_name.lower(): true_chart_type = "histogram"
            
            # 1. Classification & Features
            try:
                cls_res = self.classifier.classify_chart(img_str)
                pred_chart_type = cls_res.get("chart_type", "unknown")
                metrics = cls_res.get("metrics", {})
            except Exception as e:
                pred_chart_type = "unknown"
                metrics = {}
                logger.error(f"Classification failed on {img_name}: {e}")
                
            # Accuracy computation
            classification_accuracy = 1.0 if pred_chart_type == true_chart_type else 0.0

            # 2. OCR
            try:
                ocr_res = self.ocr_engine.run_ocr(img_str)
                easy_text = OCREngine.clean_text(ocr_res.get("easyocr_text", []))
                paddle_text = OCREngine.clean_text(ocr_res.get("paddleocr_text", []))
            except Exception as e:
                easy_text = []
                paddle_text = []
                logger.error(f"OCR failed on {img_name}: {e}")

            # 3. OCR Evaluation
            gt = load_ground_truth(img_name)
            try:
                scores = self.evaluator.evaluate(easy_text, paddle_text, gt)
                ocr_acc = max(scores.get("easyocr_score", 0), scores.get("paddleocr_score", 0))
            except Exception as e:
                ocr_acc = 0.0
                logger.error(f"Evaluation failed on {img_name}: {e}")
                
            elapsed = time.time() - start_time
            
            # Record
            record = {
                "image_name": img_name,
                "true_chart_type": true_chart_type,
                "predicted_chart_type": pred_chart_type,
                "classification_accuracy": classification_accuracy,
                "best_ocr_accuracy": ocr_acc,
                "processing_time_sec": elapsed,
            }
            # Add dynamic metrics directly into record
            for k, v in metrics.items():
                record[f"feature_{k}"] = v
                
            records.append(record)
            
        # Save to CSV
        df = pd.DataFrame(records)
        df.to_csv(self.output_csv, index=False)
        print(f"\nSaved {len(records)} benchmark records to {self.output_csv}")
        
        # Print summary
        acc = df["classification_accuracy"].mean()
        print(f"Overall Classification Accuracy: {acc*100:.1f}%")

if __name__ == "__main__":
    runner = BenchmarkRunner()
    runner.run_benchmark()
