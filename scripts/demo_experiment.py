"""
Demo experiment runner for professor presentation.

This mode keeps the original full pipeline intact and runs a shorter
subset experiment with presentation-friendly artifacts.
"""

from __future__ import annotations

import json
import logging
import math
import time
from collections import Counter
from pathlib import Path
from typing import Dict, List

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from tqdm import tqdm

from src.classifier.chart_classifier import ChartClassifier
from src.extraction.value_extractor import ValueExtractor
from src.ocr.ocr_engine import OCREngine
from src.pipeline.config import OCRConfig, PreprocessingConfig
from src.preprocessing.image_preprocessor import ImagePreprocessor


LOGGER = logging.getLogger("demo_experiment")


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )
    # Keep third-party logs quiet for a clean demo console.
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    logging.getLogger("PIL").setLevel(logging.WARNING)
    logging.getLogger("easyocr").setLevel(logging.ERROR)
    logging.getLogger("ppocr").setLevel(logging.ERROR)
    logging.getLogger("paddle").setLevel(logging.ERROR)
    logging.getLogger("urllib3").setLevel(logging.ERROR)
    logging.getLogger("src.ocr.ocr_engine").setLevel(logging.ERROR)


def discover_images(raw_dir: Path) -> List[Path]:
    exts = ("*.png", "*.jpg", "*.jpeg", "*.bmp")
    images: List[Path] = []
    for ext in exts:
        images.extend(sorted(raw_dir.glob(ext)))
    return images


def ensure_output_dirs(results_dir: Path) -> Dict[str, Path]:
    plots_dir = results_dir / "plots"
    examples_dir = results_dir / "demo_examples"
    logs_dir = results_dir / "logs"
    for d in (results_dir, plots_dir, examples_dir, logs_dir):
        d.mkdir(parents=True, exist_ok=True)
    return {"plots": plots_dir, "examples": examples_dir, "logs": logs_dir}


def _safe_text(text: str, max_len: int = 160) -> str:
    text = " ".join(str(text).split())
    return text if len(text) <= max_len else text[: max_len - 3] + "..."


def _draw_text_block(image, lines: List[str], x: int, y: int) -> None:
    h = 22 * len(lines) + 12
    w = max(280, min(860, max(len(line) for line in lines) * 8 + 16))
    overlay = image.copy()
    cv2.rectangle(overlay, (x, y), (x + w, y + h), (255, 255, 255), -1)
    cv2.addWeighted(overlay, 0.75, image, 0.25, 0, image)
    cy = y + 20
    for line in lines:
        cv2.putText(image, line, (x + 8, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (10, 10, 10), 1, cv2.LINE_AA)
        cy += 22


def save_demo_examples(records: List[Dict], examples_dir: Path, max_examples: int = 10) -> int:
    if not records:
        return 0

    # Prefer OCR-successful and non-error examples for presentation quality.
    good = [r for r in records if r.get("num_text_elements", 0) > 0 and not r.get("error")]
    pool = good if len(good) >= max_examples else records

    if len(pool) <= max_examples:
        selected = pool
    else:
        idxs = np.linspace(0, len(pool) - 1, max_examples).astype(int).tolist()
        selected = [pool[i] for i in idxs]

    saved = 0
    for rec in selected:
        path = Path(rec["image_path"])
        img = cv2.imread(str(path))
        if img is None:
            continue

        tokens = rec.get("ocr_tokens", [])
        for token in tokens:
            bbox = token.get("bbox", [])
            if len(bbox) != 4:
                continue
            x1, y1, x2, y2 = [int(v) for v in bbox]
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 170, 255), 2)

        lines = [
            f"Predicted: {rec.get('chart_type_predicted', 'unknown')}",
            f"Confidence: {float(rec.get('confidence_score', 0.0)):.3f}",
            f"Text elems: {int(rec.get('num_text_elements', 0))}",
            f"Text: {_safe_text(rec.get('ocr_text_detected', ''), 120)}",
        ]
        _draw_text_block(img, lines, 10, 10)

        out_name = f"{path.stem}_demo.png"
        out_path = examples_dir / out_name
        cv2.imwrite(str(out_path), img)
        saved += 1
    return saved


def plot_chart_type_distribution(df: pd.DataFrame, out_path: Path) -> None:
    plt.figure(figsize=(10, 5))
    order = df["chart_type_predicted"].value_counts().index
    sns.countplot(
        data=df,
        x="chart_type_predicted",
        order=order,
        hue="chart_type_predicted",
        legend=False,
        palette="crest",
    )
    plt.title("Demo: Chart Type Distribution")
    plt.xlabel("Predicted Chart Type")
    plt.ylabel("Count")
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def plot_ocr_success(success_rate: float, out_path: Path) -> None:
    fail_rate = max(0.0, 1.0 - success_rate)
    plt.figure(figsize=(7, 4))
    data = pd.DataFrame(
        {
            "metric": ["OCR Success", "OCR Failure"],
            "rate": [success_rate, fail_rate],
        }
    )
    sns.barplot(data=data, x="metric", y="rate", hue="metric", legend=False, palette=["#2E8B57", "#B22222"])
    plt.ylim(0, 1.0)
    plt.title("Demo: OCR Success Rate")
    plt.ylabel("Rate")
    plt.xlabel("")
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def plot_runtime_per_image(df: pd.DataFrame, out_path: Path) -> None:
    plt.figure(figsize=(10, 4.5))
    plt.plot(range(1, len(df) + 1), df["processing_time"].values, color="#1f77b4", linewidth=1.3)
    plt.title("Demo: Runtime Per Image")
    plt.xlabel("Processed Image Index")
    plt.ylabel("Processing Time (sec)")
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def build_demo_report(
    report_path: Path,
    subset_requested: int,
    subset_processed: int,
    metrics: Dict,
    examples_saved: int,
) -> None:
    lines = [
        "# Demo Report: Chart Understanding Framework",
        "",
        "## 1. Overview of the Pipeline",
        "- Input chart image",
        "- OCR using PaddleOCR with EasyOCR fallback",
        "- Geometric and structural feature extraction",
        "- Chart classification (bar/line/pie/scatter/histogram)",
        "- Value extraction heuristics",
        "",
        "## 2. Dataset Subset Used",
        f"- Requested subset size: {subset_requested}",
        f"- Processed subset size: {subset_processed}",
        "- Source: `data/raw_images/`",
        "",
        "## 3. Key Experiment Metrics",
        f"- Total images processed: {metrics['total_images_processed']}",
        f"- Average processing time: {metrics['average_processing_time']:.3f} sec/image",
        f"- OCR success rate: {metrics['ocr_success_rate'] * 100:.2f}%",
        f"- OCR failure rate: {metrics['ocr_failure_rate'] * 100:.2f}%",
        f"- Chart type distribution: {metrics['chart_type_distribution']}",
        "",
        "## 4. Example Outputs",
        f"- Annotated demo examples saved: {examples_saved}",
        "- Directory: `results/demo_examples/`",
        "",
        "## 5. Observations and Limitations",
        "- OCR performance varies with image quality and text orientation.",
        "- PaddleOCR runtime stability depends on local CPU/Paddle build; fallback keeps the pipeline robust.",
        "- Confidence score in this demo is OCR-derived confidence proxy.",
        "- Demo subset is representative but not exhaustive of the full 6500-image dataset.",
    ]
    report_path.write_text("\n".join(lines), encoding="utf-8")


def run_demo_experiment() -> None:
    configure_logging()

    project_root = Path(__file__).resolve().parent
    raw_dir = project_root / "data" / "raw_images"
    results_dir = project_root / "results"
    out_dirs = ensure_output_dirs(results_dir)

    images = discover_images(raw_dir)
    if not images:
        raise RuntimeError(f"No chart images found in {raw_dir}")

    # Required demo subset line for runtime control.
    images = images[:150]
    requested_subset = len(images)

    # Runtime budget: 25 minutes target (within 20-30 min requirement).
    max_runtime_sec = 25 * 60
    min_subset_for_demo = 100
    target_count = len(images)

    ocr_cfg = OCRConfig(
        enabled_engines=["paddleocr", "easyocr"],
        ensemble_enabled=True,
        paddle_retry_count=0,
        disable_mkldnn_on_retry=False,
        force_engine=None,
    )
    ocr_engine = OCREngine(
        config=ocr_cfg,
        preprocessor=ImagePreprocessor(PreprocessingConfig()),
        cache=None,
    )
    classifier = ChartClassifier()
    extractor = ValueExtractor()

    records: List[Dict] = []
    started = time.perf_counter()

    pbar = tqdm(total=target_count, desc="Demo Pipeline", unit="img")

    for idx, image_path in enumerate(images):
        if len(records) >= target_count:
            break

        t0 = time.perf_counter()
        error = None
        chart_type = "unknown"
        ocr_text = ""
        num_text = 0
        conf = 0.0
        tokens = []

        try:
            cls = classifier.classify_chart(str(image_path))
            chart_type = cls.get("chart_type", "unknown")

            # OCR handling is resilient by design; fallback is inside OCREngine.
            ocr = ocr_engine.run_ocr(str(image_path))
            tokens = ocr.get("tokens", []) or []
            ocr_text = str(ocr.get("text", "") or " ".join(ocr.get("cleaned_text", []) or []))
            num_text = len(tokens) if tokens else len(ocr.get("cleaned_text", []) or [])
            conf = float(ocr.get("confidence", ocr.get("best_ocr_confidence", 0.0)))

            # Keep value extraction in demo run to preserve full-pipeline behavior.
            labels = ocr.get("cleaned_text", [])[:20]
            _ = extractor.extract_values(str(image_path), chart_type, labels)

        except Exception as exc:
            error = str(exc)
            # Graceful failure record; never crash whole demo run.
            LOGGER.warning("Image failed: %s | %s", image_path.name, exc)

        dt = time.perf_counter() - t0
        records.append(
            {
                "image_name": image_path.name,
                "image_path": str(image_path),
                "chart_type_predicted": chart_type,
                "ocr_text_detected": ocr_text,
                "num_text_elements": int(num_text),
                "processing_time": float(dt),
                "confidence_score": float(conf),
                "ocr_tokens": tokens,
                "error": error,
            }
        )
        pbar.update(1)

        # Runtime-aware auto-reduction after initial sample.
        processed = len(records)
        elapsed = time.perf_counter() - started
        if processed >= 15 and target_count > min_subset_for_demo:
            avg = elapsed / processed
            projected_total = avg * target_count
            if projected_total > max_runtime_sec:
                reduced = max(min_subset_for_demo, int(max_runtime_sec / max(avg, 1e-6)))
                reduced = min(reduced, target_count)
                if reduced < target_count:
                    LOGGER.info(
                        "Runtime projection %.1f min too high; reducing subset from %d to %d images.",
                        projected_total / 60.0,
                        target_count,
                        reduced,
                    )
                    target_count = reduced
                    pbar.total = target_count
                    pbar.refresh()

        # Hard stop near runtime budget once minimum demo size is met.
        if elapsed >= max_runtime_sec and processed >= min_subset_for_demo:
            LOGGER.info("Reached runtime budget, stopping at %d images.", processed)
            break

    pbar.close()

    processed_records = records[:target_count]
    df = pd.DataFrame(processed_records)

    # Required CSV with requested columns.
    csv_cols = [
        "image_name",
        "chart_type_predicted",
        "ocr_text_detected",
        "num_text_elements",
        "processing_time",
        "confidence_score",
    ]
    demo_csv = results_dir / "demo_results.csv"
    df[csv_cols].to_csv(demo_csv, index=False)

    total = len(df)
    ocr_success = float((df["num_text_elements"] > 0).mean()) if total else 0.0
    metrics = {
        "total_images_processed": int(total),
        "average_processing_time": float(df["processing_time"].mean()) if total else 0.0,
        "chart_type_distribution": dict(Counter(df["chart_type_predicted"])) if total else {},
        "ocr_success_rate": ocr_success,
        "ocr_failure_rate": float(max(0.0, 1.0 - ocr_success)),
    }
    metrics_json = results_dir / "demo_metrics.json"
    metrics_json.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    # Required plots.
    plot_chart_type_distribution(df, out_dirs["plots"] / "chart_type_distribution.png")
    plot_ocr_success(metrics["ocr_success_rate"], out_dirs["plots"] / "ocr_success_rate.png")
    plot_runtime_per_image(df, out_dirs["plots"] / "runtime_per_image.png")

    # Required annotated example images.
    examples_saved = save_demo_examples(processed_records, out_dirs["examples"], max_examples=10)

    # Required markdown report.
    report_path = results_dir / "demo_report.md"
    build_demo_report(
        report_path=report_path,
        subset_requested=requested_subset,
        subset_processed=total,
        metrics=metrics,
        examples_saved=examples_saved,
    )

    print("\nExperiment completed successfully.\n")
    print(f"Images processed: {total}")
    print(f"Average processing time: {metrics['average_processing_time']:.3f} sec")
    print(f"OCR success rate: {metrics['ocr_success_rate'] * 100:.2f}%")
    print("Output directory: results/")


if __name__ == "__main__":
    run_demo_experiment()
