"""
Unified research-grade chart understanding pipeline.
"""

from __future__ import annotations

import json
import logging
import math
import os
import re
import threading
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd

from src.caching.ocr_cache import OCRCache
from src.classifier.chart_classifier import ChartClassifier
from src.classifier.ml_classifier import MLChartClassifier
from src.dataset.dataset_manager import DatasetManager
from src.dataset.versioning import DatasetImage, build_dataset_manifest, discover_images
from src.detection.yolo_detector import ChartDetectionService
from src.evaluation.research_evaluator import ResearchEvaluator
from src.experiments.ablation import AblationRunner
from src.extraction.value_extractor import ValueExtractor
from src.feature_extraction.geometric_features import GeometricFeatureExtractor
from src.ocr.ocr_engine import OCREngine
from src.pipeline.config import PipelineConfig
from src.pipeline.reproducibility import (
    build_experiment_snapshot,
    save_experiment_snapshot,
    set_global_seed,
)
from src.preprocessing.image_preprocessor import ImagePreprocessor
from src.profiling.runtime_profiler import RuntimeProfiler
from src.visualization.research_plots import ResearchPlotGenerator

logger = logging.getLogger(__name__)


def _extract_numeric_values(payload: Any) -> List[float]:
    values: List[float] = []
    if isinstance(payload, dict):
        for value in payload.values():
            values.extend(_extract_numeric_values(value))
    elif isinstance(payload, list):
        for value in payload:
            values.extend(_extract_numeric_values(value))
    else:
        if isinstance(payload, (int, float)):
            values.append(float(payload))
        elif isinstance(payload, str):
            matches = re.findall(r"[-+]?\d*\.?\d+", payload)
            values.extend([float(m) for m in matches if m])
    return values


class ResearchPipeline:
    """Main orchestrator for scalable and reproducible experiments."""

    STAGES = [
        "image_loading_time",
        "ocr_time",
        "yolo_detection_time",
        "feature_extraction_time",
        "classification_time",
        "value_extraction_time",
        "total_pipeline_time",
    ]

    def __init__(self, config: PipelineConfig):
        self.cfg = config
        self._thread_local = threading.local()
        self._runtime_profiler = RuntimeProfiler(self.STAGES)

        self.results_dir = Path(self.cfg.outputs.results_dir)
        self.logs_dir = Path(self.cfg.outputs.logs_dir)
        self.plots_dir = Path(self.cfg.outputs.plots_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        self.plots_dir.mkdir(parents=True, exist_ok=True)

        self.dataset_manager = DatasetManager(min_size=self.cfg.dataset.min_size)
        self.preprocessor = ImagePreprocessor(self.cfg.preprocessing)
        self.cache = OCRCache(self.cfg.cache.root_dir, self.cfg.cache.ocr_subdir) if self.cfg.cache.enabled else None
        if self.cache and self.cfg.cache.invalidate:
            self.cache.invalidate()

        self.ocr_engine = OCREngine(self.cfg.ocr, self.preprocessor, self.cache)
        self.detector = ChartDetectionService(imgsz=512, device="cpu")
        self.plot_generator = ResearchPlotGenerator(self.plots_dir)
        self.evaluator = ResearchEvaluator()

    def _get_worker_components(self) -> Tuple[GeometricFeatureExtractor, ChartClassifier, ValueExtractor]:
        if not hasattr(self._thread_local, "geo"):
            self._thread_local.geo = GeometricFeatureExtractor()
            self._thread_local.classifier = ChartClassifier()
            self._thread_local.value_extractor = ValueExtractor()
        return self._thread_local.geo, self._thread_local.classifier, self._thread_local.value_extractor

    def _worker_count(self) -> int:
        if not self.cfg.parallel.enabled:
            return 1
        if self.cfg.parallel.worker_count and self.cfg.parallel.worker_count > 0:
            return int(self.cfg.parallel.worker_count)
        cpu = os.cpu_count() or 4
        return max(1, min(self.cfg.parallel.max_worker_cap, cpu))

    def _process_one(self, item: DatasetImage) -> Dict[str, Any]:
        geo_extractor, classifier, value_extractor = self._get_worker_components()
        image_path = str(item.path)
        record: Dict[str, Any] = {
            "image_name": item.name,
            "image_path": str(item.path.resolve()),
            "dataset_source": item.source,
            "true_chart_type": item.chart_type,
            "predicted_chart_type": "unknown",
            "classification_correct": 0,
            "ocr_confidence": 0.0,
            "ocr_text_length": 0,
            "value_extraction_success": 0,
            "numeric_extraction_accuracy": 0.0,
            "numeric_values_found": 0,
            "numeric_values_total": 0,
            "union_token_count": 0,
            "ensemble_token_count": 0,
            "ocr_token_recall": 0.0,
            "error": None,
        }
        for stage in self.STAGES:
            record[stage] = 0.0

        start_total = time.perf_counter()
        try:
            t0 = time.perf_counter()
            img = cv2.imread(image_path)
            record["image_loading_time"] = time.perf_counter() - t0
            self._runtime_profiler.record("image_loading_time", record["image_loading_time"])
            if img is None:
                raise RuntimeError("Failed to load image")

            t1 = time.perf_counter()
            ocr_res = self.ocr_engine.run_ocr(image_path, invalidate_cache=self.cfg.cache.invalidate)
            record["ocr_time"] = time.perf_counter() - t1
            self._runtime_profiler.record("ocr_time", record["ocr_time"])

            t2 = time.perf_counter()
            yolo_counts = self.detector.infer_counts(image_path)
            record["yolo_detection_time"] = time.perf_counter() - t2
            self._runtime_profiler.record("yolo_detection_time", record["yolo_detection_time"])

            t3 = time.perf_counter()
            geo = geo_extractor.extract(image_path)
            record["feature_extraction_time"] = time.perf_counter() - t3
            self._runtime_profiler.record("feature_extraction_time", record["feature_extraction_time"])

            t4 = time.perf_counter()
            cls = classifier.classify_chart(image_path)
            record["classification_time"] = time.perf_counter() - t4
            self._runtime_profiler.record("classification_time", record["classification_time"])

            chart_type = cls.get("chart_type", "unknown")
            record["predicted_chart_type"] = chart_type
            record["classification_correct"] = int(chart_type == item.chart_type)
            record["classification_summary"] = cls.get("summary", "")
            record["segmentation_counts"] = cls.get("segmentation_counts", {})
            record["yolo_counts"] = yolo_counts
            record["geo_features"] = geo

            t5 = time.perf_counter()
            labels = ocr_res.get("cleaned_text", [])[:20]
            values = value_extractor.extract_values(image_path, chart_type, labels)
            record["value_extraction_time"] = time.perf_counter() - t5
            self._runtime_profiler.record("value_extraction_time", record["value_extraction_time"])

            numeric_values = _extract_numeric_values(values)
            token_numeric = re.findall(r"[-+]?\d*\.?\d+", ocr_res.get("text", ""))
            found = len(numeric_values)
            total_numeric = len(token_numeric)

            easy_tokens = {str(t).strip().lower() for t in ocr_res.get("easyocr_text", []) if str(t).strip()}
            paddle_tokens = {str(t).strip().lower() for t in ocr_res.get("paddleocr_text", []) if str(t).strip()}
            union_tokens = easy_tokens | paddle_tokens
            ensemble_tokens = {str(t.get("text", "")).strip().lower() for t in ocr_res.get("tokens", []) if str(t.get("text", "")).strip()}

            record["ocr_output"] = ocr_res
            record["extracted_values"] = values
            record["value_extraction_success"] = int(bool(values))
            record["easyocr_confidence"] = float(ocr_res.get("easyocr_confidence", 0.0))
            record["paddleocr_confidence"] = float(ocr_res.get("paddleocr_confidence", 0.0))
            record["ocr_confidence"] = float(ocr_res.get("confidence", ocr_res.get("best_ocr_confidence", 0.0)))
            record["ocr_text_length"] = len(str(ocr_res.get("text", "")))
            record["numeric_values_found"] = found
            record["numeric_values_total"] = total_numeric
            record["numeric_extraction_accuracy"] = float(found / total_numeric) if total_numeric else float(bool(found))
            record["union_token_count"] = len(union_tokens)
            record["ensemble_token_count"] = len(ensemble_tokens)
            record["ocr_token_recall"] = float(len(ensemble_tokens) / len(union_tokens)) if union_tokens else 0.0

            rates = self.ocr_engine.get_success_rates()
            record["paddle_success_rate"] = rates["paddle_success_rate"]
            record["easyocr_success_rate"] = rates["easyocr_success_rate"]
            record["fallback_rate"] = rates["fallback_rate"]
            record["ocr_cache_hits"] = rates.get("cache_hits", 0)
            record["ocr_cache_misses"] = rates.get("cache_misses", 0)

        except Exception as exc:
            record["error"] = str(exc)
            logger.warning("Image failed: %s | %s", item.name, exc, extra={"event": "image_failure", "image_name": item.name})

        record["total_pipeline_time"] = time.perf_counter() - start_total
        record["processing_time_sec"] = record["total_pipeline_time"]
        self._runtime_profiler.record("total_pipeline_time", record["total_pipeline_time"])
        return record

    def _run_parallel(self, items: List[DatasetImage]) -> List[Dict[str, Any]]:
        workers = self._worker_count()
        logger.info("Parallel pipeline workers: %s", workers, extra={"event": "parallel_start"})
        if workers <= 1:
            return [self._process_one(item) for item in items]

        results: List[Dict[str, Any]] = []
        executor_type = self.cfg.parallel.executor_type.lower().strip()
        executor_cls = ThreadPoolExecutor
        if executor_type == "process":
            executor_cls = ProcessPoolExecutor

        # Process mode is not used with OCR engines due non-picklable model handles.
        if executor_cls is ProcessPoolExecutor:
            logger.warning("ProcessPool requested; falling back to ThreadPool for OCR safety")
            executor_cls = ThreadPoolExecutor

        with executor_cls(max_workers=workers) as executor:
            futures = {executor.submit(self._process_one, item): item for item in items}
            for future in as_completed(futures):
                item = futures[future]
                try:
                    results.append(future.result())
                except Exception as exc:
                    logger.error("Unhandled worker exception for %s: %s", item.name, exc)
                    results.append(
                        {
                            "image_name": item.name,
                            "image_path": str(item.path.resolve()),
                            "dataset_source": item.source,
                            "true_chart_type": item.chart_type,
                            "predicted_chart_type": "unknown",
                            "classification_correct": 0,
                            "error": str(exc),
                            "processing_time_sec": 0.0,
                        }
                    )
        return results

    def _records_to_dataframe(self, records: List[Dict[str, Any]]) -> pd.DataFrame:
        flat_rows: List[Dict[str, Any]] = []
        for rec in records:
            row = {k: v for k, v in rec.items() if k not in {"geo_features", "yolo_counts", "segmentation_counts", "ocr_output", "extracted_values"}}
            for key, value in (rec.get("geo_features", {}) or {}).items():
                row[f"feature_{key}"] = value
            for key, value in (rec.get("yolo_counts", {}) or {}).items():
                row[f"yolo_{key}"] = value
            row["best_ocr_confidence"] = rec.get("ocr_confidence", 0.0)
            row["best_ocr_accuracy"] = rec.get("ocr_confidence", 0.0)
            flat_rows.append(row)
        return pd.DataFrame(flat_rows)

    def _save_pipeline_output(self, records: List[Dict[str, Any]]) -> Path:
        output_path = Path(self.cfg.outputs.pipeline_output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(records, indent=2, default=str), encoding="utf-8")
        return output_path

    def _save_experiment_csv(self, df: pd.DataFrame) -> Path:
        path = Path(self.cfg.outputs.experiment_csv)
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(path, index=False)
        return path

    def _generate_report(
        self,
        metrics: Dict[str, Any],
        runtime_profile: Dict[str, Any],
        manifest: Dict[str, Any],
        output_path: str | Path,
        ablation_path: Optional[Path],
    ) -> Path:
        cls = metrics.get("classification", {})
        ocr = metrics.get("ocr", {})
        ext = metrics.get("value_extraction", {})
        lines = [
            "# Research Report",
            "",
            "## Experiment Description",
            f"- Dataset images: {manifest.get('total_images', 0)}",
            f"- Dataset fingerprint: `{manifest.get('fingerprint', 'unknown')}`",
            f"- Parallel workers: {self._worker_count()}",
            f"- OCR engines: {', '.join(self.cfg.ocr.enabled_engines)}",
            f"- OCR ensemble enabled: {self.cfg.ocr.ensemble_enabled}",
            "",
            "## Dataset Statistics",
            f"- Source distribution: {manifest.get('source_counts', {})}",
            f"- Chart type distribution: {manifest.get('chart_type_counts', {})}",
            "",
            "## Classification Performance",
            f"- Accuracy: {float(cls.get('accuracy', 0.0)):.4f}",
            f"- Precision: {float(cls.get('precision', 0.0)):.4f}",
            f"- Recall: {float(cls.get('recall', 0.0)):.4f}",
            f"- F1 score: {float(cls.get('f1_score', 0.0)):.4f}",
            "",
            "## OCR Performance",
            f"- Average confidence: {float(ocr.get('average_ocr_confidence', 0.0)):.4f}",
            f"- Token recall: {float(ocr.get('token_recall', 0.0)):.4f}",
            f"- OCR coverage: {float(ocr.get('ocr_coverage', 0.0)):.4f}",
            f"- Paddle success rate: {float(ocr.get('paddle_success_rate', 0.0)):.4f}",
            f"- EasyOCR success rate: {float(ocr.get('easyocr_success_rate', 0.0)):.4f}",
            f"- Fallback rate: {float(ocr.get('fallback_rate', 0.0)):.4f}",
            "",
            "## Value Extraction Results",
            f"- Numeric extraction accuracy: {float(ext.get('numeric_extraction_accuracy', 0.0)):.4f}",
            f"- Extraction success rate: {float(ext.get('extraction_success_rate', 0.0)):.4f}",
            "",
            "## Runtime Analysis",
        ]

        for stage, payload in runtime_profile.items():
            if stage == "overall":
                continue
            lines.append(f"- {stage}: mean={float(payload.get('mean_seconds', 0.0)):.6f}s")

        lines.extend(
            [
                "",
                "## Grouped Accuracy",
                f"- Per chart type accuracy: {metrics.get('per_chart_type_accuracy', {})}",
                f"- Per dataset source accuracy: {metrics.get('per_dataset_source_accuracy', {})}",
            ]
        )
        if ablation_path is not None:
            lines.append(f"- Ablation table: `{ablation_path.as_posix()}`")

        report_path = Path(output_path)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text("\n".join(lines), encoding="utf-8")
        return report_path

    def _train_ml_models(self, experiment_csv: Path) -> Dict[str, Any]:
        try:
            clf = MLChartClassifier(models_dir=self.results_dir / "models")
            return clf.train_and_evaluate(experiment_csv) or {}
        except Exception as exc:
            logger.warning("ML model training skipped: %s", exc)
            return {}

    def _execute_experiment(
        self,
        images: List[DatasetImage],
        run_name: str = "main",
    ) -> Tuple[List[Dict[str, Any]], pd.DataFrame, Dict[str, Any], Dict[str, Any]]:
        logger.info("Starting experiment run: %s", run_name, extra={"event": "run_start"})
        records = self._run_parallel(images)
        df = self._records_to_dataframe(records)
        metrics = self.evaluator.evaluate(df)
        runtime_profile = self._runtime_profiler.summarize()
        if not df.empty:
            metrics["mean_pipeline_time_sec"] = float(df["total_pipeline_time"].mean())
        else:
            metrics["mean_pipeline_time_sec"] = 0.0
        return records, df, metrics, runtime_profile

    def run(self) -> Dict[str, Any]:
        set_global_seed(self.cfg.seed)
        self.dataset_manager.ensure_dataset()

        manifest = build_dataset_manifest(
            self.cfg.dataset.raw_images_dir,
            self.cfg.dataset.manifest_path,
            self.cfg.dataset.split_ratio,
            self.cfg.dataset.split_seed,
        )
        images = discover_images(self.cfg.dataset.raw_images_dir)
        if self.cfg.dataset.max_images > 0:
            images = images[: self.cfg.dataset.max_images]
        if not images:
            raise RuntimeError(f"No images found in {self.cfg.dataset.raw_images_dir}")

        snapshot = build_experiment_snapshot(self.cfg, manifest.get("fingerprint", "unknown"))
        save_experiment_snapshot(snapshot, self.cfg.outputs.experiment_config)

        records, df, metrics, runtime_profile = self._execute_experiment(images, run_name="main")
        pipeline_output_path = self._save_pipeline_output(records)
        experiment_csv = self._save_experiment_csv(df)
        research_metrics_path = self.evaluator.save(metrics, self.cfg.outputs.research_metrics)
        runtime_profile_path = self._runtime_profiler.save(self.cfg.outputs.runtime_profile)

        ml_results = self._train_ml_models(experiment_csv)
        plot_count = self.plot_generator.generate(df, metrics, runtime_profile, ml_results)

        ablation_path = None
        if self.cfg.run_ablation and self.cfg.ablation.enabled and self.cfg.ablation.variants:
            runner = AblationRunner(self.cfg, self.cfg.ablation.variants)

            def _exec_variant(name: str, cfg_variant: PipelineConfig) -> Dict[str, Any]:
                # Keep ablations computationally practical by sampling if dataset is large.
                sample_size = min(len(images), max(200, min(1000, int(math.ceil(len(images) * 0.2)))))
                sampled = images[:sample_size]
                sub_pipeline = ResearchPipeline(cfg_variant)
                _, df_v, metrics_v, _ = sub_pipeline._execute_experiment(sampled, run_name=name)
                if not df_v.empty:
                    metrics_v["mean_pipeline_time_sec"] = float(df_v["total_pipeline_time"].mean())
                return metrics_v

            rows = runner.run(_exec_variant)
            ablation_path = runner.save_table(rows, self.cfg.outputs.ablation_csv)

        report_path = self._generate_report(
            metrics=metrics,
            runtime_profile=runtime_profile,
            manifest=manifest,
            output_path=self.cfg.outputs.research_report,
            ablation_path=ablation_path,
        )

        summary = {
            "pipeline_output": str(pipeline_output_path),
            "experiment_csv": str(experiment_csv),
            "research_metrics": str(research_metrics_path),
            "runtime_profile": str(runtime_profile_path),
            "experiment_config": str(self.cfg.outputs.experiment_config),
            "research_report": str(report_path),
            "plots_dir": str(self.plots_dir),
            "plot_count": int(plot_count),
            "ablation_csv": str(ablation_path) if ablation_path else None,
        }
        logger.info("Research pipeline completed")
        return summary
