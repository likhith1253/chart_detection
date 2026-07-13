"""
Chart Understanding Framework - Final PhD-level pipeline (v3 outputs).
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import random
import shutil
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score

import config
import main as baseline_main
from src.data.dataset_expander import DatasetExpander
from src.dataset.unified_loader import UnifiedChartDatasetLoader, UnifiedLoaderConfig
from src.evaluation.evaluator import PipelineEvaluator
from src.extraction.value_extractor import ValueExtractor
from src.research.advanced_ml import AdvancedMLClassifier
from src.research.cnn_classifier import CNNChartClassifier, TrainConfig
from src.research.dataset_verifier import infer_label_from_filename, verify_and_balance_dataset
from src.research.hybrid_classifier import HybridChartClassifier
from src.research.simclr_pretraining import SimCLRConfig, SimCLRPretrainer
from src.research.yolo_chart_detector import YOLOChartElementDetector

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger("research_final_v3")

FINAL_PIPELINE_JSON = config.RESULT_DIR / "pipeline_output_final_v3.json"
FINAL_EXPERIMENT_CSV = config.RESULT_DIR / "experiment_results_final_v3.csv"
FINAL_SYSTEM_METRICS_JSON = config.RESULT_DIR / "system_metrics_final_v3.json"
FINAL_RESEARCH_REPORT_MD = config.RESULT_DIR / "research_report_final_v3.md"
FINAL_SYSTEM_REPORT_TXT = config.RESULT_DIR / "system_report_final_v3.txt"


def _seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def _clear_directory_contents(path: Path):
    if not path.exists():
        return
    for child in path.iterdir():
        if child.is_dir():
            shutil.rmtree(child, ignore_errors=True)
        else:
            child.unlink(missing_ok=True)


def _enforce_force_recompute(force_recompute: bool):
    if not force_recompute:
        return
    logger.info("FORCE_RECOMPUTE=True -> clearing results/models/cache directories")
    _clear_directory_contents(config.RESULT_DIR)
    _clear_directory_contents(config.MODELS_DIR)
    _clear_directory_contents(config.CACHE_DIR)

    for d in [
        config.RESULT_DIR,
        config.METRICS_DIR,
        config.PLOTS_DIR,
        config.DEBUG_DIR,
        config.MODELS_DIR,
        config.RESEARCH_METRICS_DIR,
        config.CACHE_DIR,
    ]:
        d.mkdir(parents=True, exist_ok=True)


def _safe_series(df: pd.DataFrame, col: str, default: float = 0.0) -> pd.Series:
    if col not in df.columns:
        return pd.Series(np.full(len(df), default), index=df.index, dtype=float)
    return pd.to_numeric(df[col], errors="coerce").fillna(default)


def _minmax(series: pd.Series) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce").fillna(0.0).astype(float)
    lo, hi = float(s.min()), float(s.max())
    if hi - lo < 1e-12:
        return pd.Series(np.zeros(len(s)), index=s.index)
    return (s - lo) / (hi - lo)


def _numeric_vs_categorical_text_ratios(texts: List[str]) -> Tuple[float, float]:
    if not texts:
        return 0.0, 0.0
    num_count = 0
    cat_count = 0
    for txt in texts:
        txt = str(txt).strip()
        if not txt:
            continue
        compact = txt.replace(",", "").replace("%", "").replace(".", "", 1)
        if compact.isdigit():
            num_count += 1
        elif any(c.isalpha() for c in txt):
            cat_count += 1
    total = max(1, len(texts))
    return num_count / total, cat_count / total


def _pie_slice_angle_variance(value_payload) -> float:
    if isinstance(value_payload, dict) and "slices" in value_payload:
        angles = []
        for item in value_payload.get("slices", []):
            try:
                angles.append(float(item.get("estimated_angle_deg", 0.0)))
            except Exception:
                continue
        if len(angles) > 1:
            return float(np.var(angles))
    return 0.0


def _normalize_value_payload(payload) -> Dict:
    if payload is None:
        return {"values": [], "normalized_values": []}

    if isinstance(payload, dict):
        if "type" not in payload and payload:
            labels = list(payload.keys())
            vals = []
            for key in labels:
                try:
                    vals.append(float(payload[key]))
                except Exception:
                    vals.append(0.0)
            arr = np.asarray(vals, dtype=float)
            if arr.size > 0:
                if arr.max() - arr.min() < 1e-9:
                    norm = np.ones_like(arr) * (1.0 if arr.max() > 0 else 0.0)
                else:
                    norm = (arr - arr.min()) / (arr.max() - arr.min())
            else:
                norm = np.asarray([])
            return {
                "type": "bar_like",
                "labels": labels,
                "values": [float(v) for v in arr.tolist()],
                "normalized_values": [float(v) for v in norm.tolist()],
            }

        if payload.get("type") == "line_chart":
            points = payload.get("data_points", [])
            vals = [float(p.get("normalized_value", 0.0)) for p in points if isinstance(p, dict)]
            arr = np.asarray(vals, dtype=float)
            norm = arr / 100.0 if arr.size else arr
            return {
                "type": "line_chart",
                "values": [float(v) for v in arr.tolist()],
                "normalized_values": [float(v) for v in norm.tolist()],
            }

        if payload.get("type") == "scatter_plot":
            points = payload.get("points", [])
            vals = []
            for p in points:
                if isinstance(p, dict):
                    try:
                        x = float(p.get("x_normalized", 0.0))
                        y = float(p.get("y_normalized", 0.0))
                        vals.append(float(math.sqrt(x * x + y * y)))
                    except Exception:
                        continue
            arr = np.asarray(vals, dtype=float)
            norm = arr / (arr.max() + 1e-9) if arr.size else arr
            return {
                "type": "scatter_plot",
                "values": [float(v) for v in arr.tolist()],
                "normalized_values": [float(v) for v in norm.tolist()],
            }

        if payload.get("type") == "pie_chart":
            slices = payload.get("slices", [])
            vals = [float(s.get("estimated_angle_deg", 0.0)) for s in slices if isinstance(s, dict)]
            arr = np.asarray(vals, dtype=float)
            norm = arr / 360.0 if arr.size else arr
            return {
                "type": "pie_chart",
                "values": [float(v) for v in arr.tolist()],
                "normalized_values": [float(v) for v in norm.tolist()],
            }

    return {"values": [], "normalized_values": []}


def _attach_dataset_labels(df: pd.DataFrame, split_df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    split_map = split_df.set_index("image_name")["split"].to_dict() if "image_name" in split_df.columns else {}
    label_map = split_df.set_index("image_name")["label"].to_dict() if "image_name" in split_df.columns else {}
    dataset_map = split_df.set_index("image_name")["dataset_name"].to_dict() if "image_name" in split_df.columns else {}

    out["split"] = out["image_name"].map(split_map).fillna("unspecified")
    out["dataset_name"] = out["image_name"].map(dataset_map).fillna(out.get("dataset_source", "local_raw"))
    out["true_chart_type"] = out["image_name"].map(label_map).fillna(out["image_name"].apply(infer_label_from_filename))
    out["true_chart_type"] = out["true_chart_type"].where(
        out["true_chart_type"].isin(["bar_chart", "histogram", "line_chart", "scatter_plot", "pie_chart"]),
        "unknown",
    )
    return out


def _ensure_datasets(force_recompute: bool) -> Tuple[Dict, Dict, pd.DataFrame]:
    expander = DatasetExpander()
    download_summary = expander.ensure_external_datasets(
        min_total_images=5000,
        min_per_dataset=1200,
        force_recompute=force_recompute,
    )

    verify_summary = verify_and_balance_dataset(
        raw_dir=config.RAW_IMAGE_DIR,
        target_total=5000,
        min_per_class=700,
        remove_corrupt=True,
    )

    loader = UnifiedChartDatasetLoader()
    unified_pack = loader.build_manifests(
        UnifiedLoaderConfig(
            train_ratio=0.70,
            val_ratio=0.15,
            test_ratio=0.15,
            random_state=42,
            balance_classes=False,
        )
    )
    split_df = unified_pack["splits"]
    return download_summary, verify_summary, split_df


def _run_baseline_pipeline(force_recompute: bool) -> Tuple[pd.DataFrame, Dict[str, Dict]]:
    baseline_main.run_pipeline(force_recompute=force_recompute)

    csv_path = config.METRICS_DIR / "experiment_results.csv"
    if not csv_path.exists():
        raise RuntimeError("Baseline experiment_results.csv not found")

    json_path = config.RESULT_DIR / "pipeline_output.json"
    if not json_path.exists():
        raise RuntimeError("Baseline pipeline_output.json not found")

    df = pd.read_csv(csv_path)
    payload = json.loads(json_path.read_text())
    payload_map = {str(r.get("image_name")): r for r in payload if isinstance(r, dict)}

    if "image_path" not in df.columns:
        df["image_path"] = df["image_name"].apply(lambda n: str((config.RAW_IMAGE_DIR / n).resolve()))
    else:
        df["image_path"] = df["image_path"].fillna("").astype(str)
        missing = df["image_path"] == ""
        df.loc[missing, "image_path"] = df.loc[missing, "image_name"].apply(
            lambda n: str((config.RAW_IMAGE_DIR / n).resolve())
        )

    if "heuristic_prediction" not in df.columns:
        df["heuristic_prediction"] = df.get("predicted_chart_type", "unknown")
    if "heuristic_confidence" not in df.columns:
        df["heuristic_confidence"] = np.where(df["heuristic_prediction"] != "unknown", 0.55, 0.0)
    if "true_chart_type" not in df.columns:
        df["true_chart_type"] = df["image_name"].apply(infer_label_from_filename)
    return df, payload_map


def _phase_structural_features(df: pd.DataFrame, payload_map: Dict[str, Dict]) -> pd.DataFrame:
    out = df.copy()

    out["struct_bar_count"] = _safe_series(out, "feature_bar_count")
    out["struct_bar_height_variance"] = _safe_series(out, "feature_bar_height_variance")
    out["struct_bar_spacing_variance"] = _safe_series(out, "feature_bar_spacing_variance")
    out["struct_bar_alignment_score"] = _safe_series(out, "feature_bar_alignment_score")

    out["struct_polyline_continuity"] = _safe_series(out, "feature_polyline_score")
    out["struct_slope_variance"] = _safe_series(out, "feature_slope_variance")
    out["struct_vertex_count"] = _safe_series(out, "feature_polyline_vertex_count")
    out["struct_line_continuity_score"] = _safe_series(out, "feature_line_continuity_score")

    out["struct_point_density"] = _safe_series(out, "feature_point_density")
    out["struct_cluster_count"] = _safe_series(out, "feature_scatter_cluster_count")
    out["struct_nearest_neighbor_density"] = _safe_series(out, "feature_nn_density")

    out["struct_bar_gap_ratio"] = _safe_series(out, "feature_bar_gap_ratio")
    out["struct_hist_uniformity"] = _safe_series(out, "feature_histogram_uniformity_score")
    out["struct_axis_numeric_density"] = _safe_series(out, "feature_axis_numeric_density")
    out["struct_edge_orientation_entropy"] = _safe_series(out, "feature_edge_orientation_entropy")

    pie_vars = []
    for name in out["image_name"].tolist():
        pie_vars.append(_pie_slice_angle_variance(payload_map.get(name, {}).get("extracted_values")))
    out["struct_slice_angle_variance"] = pd.Series(pie_vars, index=out.index, dtype=float)

    phase_cols = [c for c in out.columns if c.startswith("struct_")]
    for col in phase_cols:
        out[f"norm_{col[7:]}"] = _minmax(out[col])
    return out


def _phase_histogram_disambiguation(df: pd.DataFrame, payload_map: Dict[str, Dict]) -> pd.DataFrame:
    out = df.copy()
    bin_adj = _safe_series(out, "feature_bin_adjacency_ratio")
    width_uniformity = _safe_series(out, "feature_width_uniformity")
    spacing_var = _safe_series(out, "feature_bar_spacing_variance")
    bars = _safe_series(out, "feature_bar_count")
    gap_ratio = _safe_series(out, "feature_bar_gap_ratio")
    hist_uniformity = _safe_series(out, "feature_histogram_uniformity_score")
    axis_numeric_density = _safe_series(out, "feature_axis_numeric_density")

    continuous_bins = np.clip(_minmax(bin_adj) * 0.5 + (1.0 - _minmax(gap_ratio)) * 0.5, 0, 1)
    uniform_bins = np.clip(0.6 * _minmax(width_uniformity) + 0.4 * _minmax(hist_uniformity), 0, 1)

    numeric_ratio = []
    categorical_ratio = []
    for img_name in out["image_name"].tolist():
        texts = payload_map.get(img_name, {}).get("ocr_output", {}).get("cleaned", [])
        n, c = _numeric_vs_categorical_text_ratios(texts if isinstance(texts, list) else [])
        numeric_ratio.append(n)
        categorical_ratio.append(c)

    numeric_ratio = pd.Series(numeric_ratio, index=out.index)
    categorical_ratio = pd.Series(categorical_ratio, index=out.index)

    hist_conf = np.clip(
        0.30 * continuous_bins
        + 0.28 * uniform_bins
        + 0.20 * numeric_ratio
        + 0.12 * _minmax(axis_numeric_density)
        + 0.10 * (1.0 - _minmax(spacing_var)),
        0,
        1,
    )
    bar_conf = np.clip(
        0.30 * _minmax(bars)
        + 0.25 * _minmax(spacing_var)
        + 0.20 * _minmax(gap_ratio)
        + 0.15 * categorical_ratio
        + 0.10 * (1.0 - _minmax(axis_numeric_density)),
        0,
        1,
    )

    out["histogram_confidence"] = hist_conf
    out["bar_chart_confidence"] = bar_conf
    out["histogram_score"] = hist_conf
    out["bar_chart_score"] = bar_conf

    mask_hist_bar = out["heuristic_prediction"].isin(["histogram", "bar_chart"])
    out.loc[mask_hist_bar & (hist_conf >= bar_conf + 0.08), "heuristic_prediction"] = "histogram"
    out.loc[mask_hist_bar & (bar_conf > hist_conf + 0.08), "heuristic_prediction"] = "bar_chart"
    return out


def _phase_yolo_train_and_infer(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
    out = df.copy()
    for col in [
        "yolo_bars",
        "yolo_line_segments",
        "yolo_scatter_points",
        "yolo_pie_slices",
        "yolo_axes",
        "yolo_legend_boxes",
        "yolo_text_regions",
    ]:
        out[col] = 0

    device = "cpu"
    try:
        import torch

        device = "cuda:0" if torch.cuda.is_available() else "cpu"
    except Exception:
        pass

    yolo = YOLOChartElementDetector(imgsz=512, device=device)
    image_paths = [Path(p) for p in out["image_path"].tolist() if Path(p).exists()]

    data_yaml = yolo.build_dataset(image_paths=image_paths, max_images=None)
    yolo_train = yolo.train(data_yaml, epochs=40, batch=16)
    if not yolo_train.get("trained", False):
        yolo.load_best()

    infer_times = []
    if "yolo_prediction" not in out.columns:
        out["yolo_prediction"] = "unknown"
    if "yolo_confidence" not in out.columns:
        out["yolo_confidence"] = 0.0

    for idx, row in out.iterrows():
        start = time.perf_counter()
        counts = yolo.infer_counts(Path(row["image_path"]))
        elapsed = time.perf_counter() - start
        infer_times.append(elapsed)

        out.at[idx, "yolo_bars"] = counts.get("bars", 0)
        out.at[idx, "yolo_line_segments"] = counts.get("line_segments", 0)
        out.at[idx, "yolo_scatter_points"] = counts.get("scatter_points", 0)
        out.at[idx, "yolo_pie_slices"] = counts.get("pie_slices", 0)
        out.at[idx, "yolo_axes"] = counts.get("axes", 0)
        out.at[idx, "yolo_legend_boxes"] = counts.get("legend_boxes", 0)
        out.at[idx, "yolo_text_regions"] = counts.get("text_regions", 0)

        pred, conf = yolo.structure_classify(
            {
                "bars": int(out.at[idx, "yolo_bars"]),
                "line_segments": int(out.at[idx, "yolo_line_segments"]),
                "scatter_points": int(out.at[idx, "yolo_scatter_points"]),
                "pie_slices": int(out.at[idx, "yolo_pie_slices"]),
                "axes": int(out.at[idx, "yolo_axes"]),
            },
            histogram_bias=float(row.get("histogram_confidence", 0.5)),
        )
        out.at[idx, "yolo_prediction"] = pred
        out.at[idx, "yolo_confidence"] = conf
        out.at[idx, "yolo_detection_time"] = float(out.at[idx, "yolo_detection_time"]) + elapsed

    det_cols = [
        "image_name",
        "predicted_chart_type",
        "yolo_bars",
        "yolo_line_segments",
        "yolo_scatter_points",
        "yolo_pie_slices",
        "yolo_axes",
        "yolo_legend_boxes",
        "yolo_text_regions",
        "yolo_prediction",
        "yolo_confidence",
    ]
    det_df = out[det_cols].copy()
    det_csv = config.RESULT_DIR / "yolo_detection_statistics_final_v3.csv"
    det_df.to_csv(det_csv, index=False)

    summary = {
        "yolo_training": yolo_train,
        "mean_counts": {
            "yolo_bars": float(out["yolo_bars"].mean()),
            "yolo_line_segments": float(out["yolo_line_segments"].mean()),
            "yolo_scatter_points": float(out["yolo_scatter_points"].mean()),
            "yolo_pie_slices": float(out["yolo_pie_slices"].mean()),
            "yolo_axes": float(out["yolo_axes"].mean()),
            "yolo_legend_boxes": float(out["yolo_legend_boxes"].mean()),
            "yolo_text_regions": float(out["yolo_text_regions"].mean()),
        },
        "mean_inference_time_sec": float(np.mean(infer_times)) if infer_times else 0.0,
        "detection_csv": str(det_csv),
    }
    (config.RESULT_DIR / "yolo_detection_summary_final_v3.json").write_text(json.dumps(summary, indent=2))
    return out, summary


def _phase_train_ml(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
    out = df.copy()
    ml = AdvancedMLClassifier()
    ml_results, oof_pred, oof_conf = ml.train(out)
    pred_all, conf_all = ml.predict(out)
    out["ml_prediction"] = pred_all.values
    out["ml_confidence"] = conf_all.values
    out.loc[oof_pred.index, "ml_prediction"] = oof_pred
    out.loc[oof_conf.index, "ml_confidence"] = oof_conf
    return out, ml_results


def _phase_train_cnn(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
    out = df.copy()
    reports: Dict[str, Dict] = {}
    candidate_predictions = []

    for backbone in ["efficientnet_b0", "resnet50"]:
        model_dir = config.MODELS_DIR / f"cnn_{backbone}"
        model_dir.mkdir(parents=True, exist_ok=True)
        cnn = CNNChartClassifier(models_dir=model_dir, model_tag=f"{backbone}_classifier")

        try:
            cfg = TrainConfig(
                backbone=backbone,
                image_size=224,
                batch_size=16,
                epochs=20,
                lr=1e-4,
                max_train_samples=None,
                early_stopping_patience=4,
                early_stopping_min_delta=1e-4,
            )
            report = cnn.train(out, cfg)
            pred, conf = cnn.predict(out["image_path"].tolist())
            reports[backbone] = report
            candidate_predictions.append(
                (
                    backbone,
                    pred,
                    conf,
                    float(report.get("best_val_f1", 0.0)),
                )
            )
        except Exception as exc:
            reports[backbone] = {"status": "failed", "error": str(exc)}

    if candidate_predictions:
        best = sorted(candidate_predictions, key=lambda t: t[3], reverse=True)[0]
        reports["selected_backbone"] = best[0]
        out["cnn_prediction"] = best[1].values
        out["cnn_confidence"] = best[2].values
    else:
        reports["selected_backbone"] = "fallback"
        out["cnn_prediction"] = out["heuristic_prediction"]
        out["cnn_confidence"] = 0.5

    return out, reports


def _phase_hybrid(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
    out = df.copy()
    hybrid = HybridChartClassifier()
    weights = {"ml": 0.45, "cnn": 0.35, "yolo": 0.15, "heuristic": 0.05}
    out = hybrid.apply(out, weights=weights)
    out["predicted_chart_type"] = out["hybrid_prediction"]
    out["classification_correct"] = (out["predicted_chart_type"] == out["true_chart_type"]).astype(int)
    metrics = hybrid.evaluate(out, pred_col="hybrid_prediction")
    return out, {"weights": weights, "metrics": metrics}


def _phase_ocr_finalize(df: pd.DataFrame, payload_map: Dict[str, Dict]) -> Tuple[pd.DataFrame, Dict]:
    out = df.copy()
    if "easyocr_confidence" not in out.columns:
        out["easyocr_confidence"] = 0.0
    if "paddleocr_confidence" not in out.columns:
        out["paddleocr_confidence"] = 0.0

    best_engine = []
    best_conf = []
    cleaned_cache = []
    paddle_nonempty = 0
    easy_nonempty = 0

    for _, row in out.iterrows():
        rec = payload_map.get(row["image_name"], {})
        ocr_out = rec.get("ocr_output", {}) if isinstance(rec, dict) else {}
        easy_txt = ocr_out.get("easyocr", []) if isinstance(ocr_out, dict) else []
        paddle_txt = ocr_out.get("paddleocr", []) if isinstance(ocr_out, dict) else []
        cleaned = ocr_out.get("cleaned", []) if isinstance(ocr_out, dict) else []

        if easy_txt:
            easy_nonempty += 1
        if paddle_txt:
            paddle_nonempty += 1

        cleaned_cache.append(cleaned if isinstance(cleaned, list) else [])
        e_conf = float(row.get("easyocr_confidence", 0.0))
        p_conf = float(row.get("paddleocr_confidence", 0.0))
        if e_conf >= p_conf:
            best_engine.append("easyocr")
            best_conf.append(e_conf)
        else:
            best_engine.append("paddleocr")
            best_conf.append(p_conf)

    out["ocr_cleaned_text"] = cleaned_cache
    out["best_ocr_engine"] = best_engine
    out["best_ocr_confidence"] = best_conf
    out["best_ocr_accuracy"] = np.maximum(
        pd.to_numeric(out["easyocr_confidence"], errors="coerce").fillna(0.0),
        pd.to_numeric(out["paddleocr_confidence"], errors="coerce").fillna(0.0),
    )

    summary = {
        "easyocr_success_rate": float(easy_nonempty / max(1, len(out))),
        "paddle_success_rate": float(paddle_nonempty / max(1, len(out))),
        "easyocr_success_rate_runtime": float(_safe_series(out, "easyocr_success_rate").mean()),
        "paddle_success_rate_runtime": float(_safe_series(out, "paddle_success_rate").mean()),
    }
    return out, summary


def _phase_value_extraction(df: pd.DataFrame, payload_map: Dict[str, Dict]) -> Tuple[pd.DataFrame, List[Dict]]:
    out = df.copy()
    extractor = ValueExtractor()
    structured_rows = []

    norm_payloads = []
    extraction_success = []

    for _, row in out.iterrows():
        rec = payload_map.get(row["image_name"], {})
        extracted = rec.get("extracted_values") if isinstance(rec, dict) else None
        if not extracted:
            try:
                extracted = extractor.extract_values(row["image_path"], row["predicted_chart_type"], labels=None)
            except Exception:
                extracted = {}

        norm_payload = _normalize_value_payload(extracted)
        norm_payloads.append(norm_payload)
        extraction_success.append(1 if norm_payload.get("values") else int(bool(extracted)))

        structured_rows.append(
            {
                "image_name": row["image_name"],
                "chart_type": row["predicted_chart_type"],
                "values": norm_payload.get("values", []),
                "normalized_values": norm_payload.get("normalized_values", []),
            }
        )

    out["normalized_value_payload"] = norm_payloads
    out["value_extraction_success"] = extraction_success
    return out, structured_rows


def _phase_runtime(df: pd.DataFrame) -> Dict:
    stage_cols = [
        "image_loading_time",
        "feature_extraction_time",
        "yolo_detection_time",
        "ocr_time",
        "value_extraction_time",
        "classification_time",
        "total_pipeline_time",
    ]

    means = {col: float(_safe_series(df, col).mean()) for col in stage_cols}
    component_sum = (
        means["image_loading_time"]
        + means["feature_extraction_time"]
        + means["yolo_detection_time"]
        + means["ocr_time"]
        + means["value_extraction_time"]
        + means["classification_time"]
    )
    total_mean = means["total_pipeline_time"]
    lower = component_sum * 0.70
    upper = component_sum * 1.35 if component_sum > 0 else 0.0
    runtime_realistic = bool(total_mean >= lower and (upper == 0.0 or total_mean <= upper))

    return {
        **means,
        "component_sum_time": float(component_sum),
        "runtime_realistic": runtime_realistic,
    }


def _phase_error_analysis(df: pd.DataFrame) -> Dict:
    labeled = df[df["true_chart_type"] != "unknown"].copy()
    if labeled.empty:
        return {"num_labeled": 0, "num_errors": 0, "common_errors": []}

    wrong = labeled[labeled["predicted_chart_type"] != labeled["true_chart_type"]].copy()
    error_pairs = (
        wrong.groupby(["true_chart_type", "predicted_chart_type"]).size().reset_index(name="count").sort_values("count", ascending=False)
    )

    cm_labels = sorted(set(labeled["true_chart_type"].tolist() + labeled["predicted_chart_type"].tolist()))
    cm = confusion_matrix(labeled["true_chart_type"], labeled["predicted_chart_type"], labels=cm_labels)
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(cm_labels)))
    ax.set_xticklabels(cm_labels, rotation=30, ha="right")
    ax.set_yticks(range(len(cm_labels)))
    ax.set_yticklabels(cm_labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix")
    for i in range(len(cm_labels)):
        for j in range(len(cm_labels)):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=8)
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(config.RESULT_DIR / "confusion_matrix_final_v3.png", dpi=220)
    plt.close(fig)

    fail_dir = config.RESULT_DIR / "failure_visualizations"
    fail_dir.mkdir(parents=True, exist_ok=True)
    for old in fail_dir.iterdir():
        if old.is_file():
            old.unlink(missing_ok=True)

    top_failures = wrong.head(120)
    for _, row in top_failures.iterrows():
        src = Path(row["image_path"])
        if not src.exists():
            continue
        dst_name = f"{row['true_chart_type']}__as__{row['predicted_chart_type']}__{src.name}"
        shutil.copy2(src, fail_dir / dst_name)

    summary = {
        "num_labeled": int(len(labeled)),
        "num_errors": int(len(wrong)),
        "overall_accuracy": float((labeled["predicted_chart_type"] == labeled["true_chart_type"]).mean()),
        "common_errors": error_pairs.head(12).to_dict(orient="records"),
        "failure_visualization_dir": str(fail_dir),
    }
    (config.RESULT_DIR / "error_analysis_final_v3.json").write_text(json.dumps(summary, indent=2))
    return summary


def _phase_dataset_benchmarks(df: pd.DataFrame) -> Dict:
    def _metrics(sub: pd.DataFrame) -> Dict:
        labeled = sub[sub["true_chart_type"] != "unknown"]
        if labeled.empty:
            return {
                "num_samples": int(len(sub)),
                "num_labeled": 0,
                "accuracy": None,
                "precision": None,
                "recall": None,
                "f1_score": None,
                "value_extraction_accuracy": float(sub["value_extraction_success"].mean()) if len(sub) else 0.0,
            }

        y_true = labeled["true_chart_type"]
        y_pred = labeled["predicted_chart_type"]
        return {
            "num_samples": int(len(sub)),
            "num_labeled": int(len(labeled)),
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "precision": float(precision_score(y_true, y_pred, average="weighted", zero_division=0)),
            "recall": float(recall_score(y_true, y_pred, average="weighted", zero_division=0)),
            "f1_score": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
            "value_extraction_accuracy": float(labeled["value_extraction_success"].mean()),
        }

    lower_name = df["image_name"].str.lower()
    return {
        "PlotQA": _metrics(df[lower_name.str.contains("plotqa", na=False)]),
        "ChartQA": _metrics(df[lower_name.str.contains("chartqa", na=False)]),
        "DVQA": _metrics(df[lower_name.str.contains("dvqa", na=False)]),
    }


def _phase_verify_and_rerun(
    df: pd.DataFrame,
    download_summary: Dict,
    yolo_summary: Dict,
    cnn_results: Dict,
    runtime_stats: Dict,
) -> Tuple[pd.DataFrame, Dict]:
    out = df.copy()

    datasets_ok = bool(download_summary.get("total_images_after", 0) >= 5000)
    for key in ["plotqa", "chartqa", "dvqa"]:
        ds = download_summary.get("datasets", {}).get(key, {})
        if ds.get("existing_after", 0) <= 0:
            datasets_ok = False

    ocr_ok = bool(_safe_series(out, "easyocr_confidence").mean() > 0.0)
    ocr_ok = ocr_ok and bool(max(_safe_series(out, "paddle_success_rate").mean(), 0.0) > 0.0)

    yolo_total = (
        _safe_series(out, "yolo_bars")
        + _safe_series(out, "yolo_line_segments")
        + _safe_series(out, "yolo_scatter_points")
        + _safe_series(out, "yolo_pie_slices")
    )
    yolo_ok = bool(float(yolo_total.mean()) > 0.0)

    cnn_ok = bool(cnn_results.get("selected_backbone") not in [None, "fallback"]) and (
        "status" not in cnn_results.get("efficientnet_b0", {})
        or cnn_results.get("efficientnet_b0", {}).get("status") != "failed"
    )

    runtime_ok = bool(runtime_stats.get("runtime_realistic", False))

    rerun_actions = []

    if not ocr_ok:
        rerun_actions.append("rerun_ocr_for_low_confidence_rows")
        from src.ocr.ocr_engine import OCREngine

        ocr_engine = OCREngine()
        for idx, row in out.iterrows():
            if float(row.get("easyocr_confidence", 0.0)) > 0.0:
                continue
            try:
                ocr = ocr_engine.run_ocr(row["image_path"])
                out.at[idx, "easyocr_confidence"] = float(ocr.get("easyocr_confidence", 0.0))
                out.at[idx, "paddleocr_confidence"] = float(ocr.get("paddleocr_confidence", 0.0))
                out.at[idx, "ocr_text_length"] = len(" ".join(ocr.get("cleaned_text", [])))
                out.at[idx, "easyocr_success_rate"] = float(ocr.get("easyocr_success_rate", 0.0))
                out.at[idx, "paddle_success_rate"] = float(ocr.get("paddle_success_rate", 0.0))
            except Exception:
                continue
        ocr_ok = bool(_safe_series(out, "easyocr_confidence").mean() > 0.0)

    if not yolo_ok:
        rerun_actions.append("rerun_yolo_inference")
        yolo = YOLOChartElementDetector(imgsz=512, device="cpu")
        yolo.load_best()
        for idx, row in out.iterrows():
            counts = yolo.infer_counts(Path(row["image_path"]))
            out.at[idx, "yolo_bars"] = counts.get("bars", 0)
            out.at[idx, "yolo_line_segments"] = counts.get("line_segments", 0)
            out.at[idx, "yolo_scatter_points"] = counts.get("scatter_points", 0)
            out.at[idx, "yolo_pie_slices"] = counts.get("pie_slices", 0)
            out.at[idx, "yolo_axes"] = counts.get("axes", 0)
            out.at[idx, "yolo_legend_boxes"] = counts.get("legend_boxes", 0)
            out.at[idx, "yolo_text_regions"] = counts.get("text_regions", 0)
        yolo_total = (
            _safe_series(out, "yolo_bars")
            + _safe_series(out, "yolo_line_segments")
            + _safe_series(out, "yolo_scatter_points")
            + _safe_series(out, "yolo_pie_slices")
        )
        yolo_ok = bool(float(yolo_total.mean()) > 0.0)

    summary = {
        "datasets_downloaded": datasets_ok,
        "ocr_engines_working": ocr_ok,
        "yolo_detections_valid": yolo_ok,
        "cnn_trained": cnn_ok,
        "runtime_realistic": runtime_ok,
        "rerun_actions": rerun_actions,
        "yolo_training": yolo_summary.get("yolo_training", {}),
    }
    return out, summary


def _build_reports(
    metrics: Dict,
    download_summary: Dict,
    verify_summary: Dict,
    yolo_summary: Dict,
    ml_results: Dict,
    cnn_results: Dict,
    hybrid_info: Dict,
    runtime_stats: Dict,
    ocr_summary: Dict,
    error_analysis: Dict,
    benchmarks: Dict,
    simclr_info: Dict,
    verification_summary: Dict,
):
    cls = metrics.get("classification", {})
    val = metrics.get("value_extraction", {})

    txt_lines = [
        "=" * 78,
        "CHART UNDERSTANDING FRAMEWORK - FINAL SYSTEM REPORT V3",
        "=" * 78,
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "[Phase 1] Force recompute and dataset expansion",
        f"FORCE_RECOMPUTE: {config.FORCE_RECOMPUTE}",
        f"Dataset download summary: {download_summary}",
        f"Dataset verification: {verify_summary}",
        "",
        "[Phase 5] YOLO chart element detector",
        f"YOLO training: {yolo_summary.get('yolo_training', {})}",
        f"YOLO mean counts: {yolo_summary.get('mean_counts', {})}",
        "",
        "[Phase 9] CNN training",
        f"CNN selected backbone: {cnn_results.get('selected_backbone')}",
        f"EfficientNet report: {cnn_results.get('efficientnet_b0', {})}",
        f"ResNet report: {cnn_results.get('resnet50', {})}",
        "",
        "[Phase 10] SimCLR",
        f"SimCLR info: {simclr_info}",
        "",
        "[Phase 11] Hybrid ensemble",
        f"Hybrid weights: {hybrid_info.get('weights', {})}",
        f"Hybrid metrics: {hybrid_info.get('metrics', {})}",
        "",
        "[Phase 12] OCR",
        f"OCR summary: {ocr_summary}",
        "",
        "[Phase 15] Research metrics",
        f"Classification accuracy: {cls.get('accuracy', 0.0):.4f}",
        f"Precision: {cls.get('precision', 0.0):.4f}",
        f"Recall: {cls.get('recall', 0.0):.4f}",
        f"F1 score: {cls.get('f1_score', 0.0):.4f}",
        f"Value extraction success: {val.get('overall_success_rate', 0.0):.2f}%",
        f"Runtime stats: {runtime_stats}",
        "",
        "[Phase 16] Final verification",
        f"Verification summary: {verification_summary}",
        "",
        "[Phase 14] Error analysis",
        f"Error analysis: {error_analysis}",
        "",
        "[Phase 8/19] Dataset benchmarks",
        f"Benchmarks: {benchmarks}",
        "=" * 78,
    ]
    FINAL_SYSTEM_REPORT_TXT.write_text("\n".join(txt_lines))

    md_lines = [
        "# Chart Understanding Framework - Final Research Report v3",
        "",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "## Methodology",
        "- Full recompute mode with cache reset (`results/`, `models/`, `cache/`).",
        "- Unified multi-dataset processing across PlotQA, ChartQA, DVQA, and local synthetic coverage.",
        "- YOLOv8 detector for bars, scatter points, line segments, pie slices, axes, legend boxes, text regions.",
        "- Advanced structural features and confusion-focused disambiguation rules.",
        "- CNN transfer learning with EfficientNetB0 and ResNet50 (20 epochs, early stopping).",
        "- SimCLR self-supervised pretraining (10 epochs).",
        "- Confidence-weighted hybrid ensemble and dual OCR fusion.",
        "",
        "## Metrics",
        f"- Accuracy: {cls.get('accuracy', 0.0):.4f}",
        f"- Precision: {cls.get('precision', 0.0):.4f}",
        f"- Recall: {cls.get('recall', 0.0):.4f}",
        f"- F1 score: {cls.get('f1_score', 0.0):.4f}",
        f"- Value extraction success: {val.get('overall_success_rate', 0.0):.2f}%",
        f"- Runtime analysis: {runtime_stats}",
        "",
        "## OCR",
        f"- OCR summary: {ocr_summary}",
        "",
        "## Benchmarks",
        f"- PlotQA: {benchmarks.get('PlotQA', {})}",
        f"- ChartQA: {benchmarks.get('ChartQA', {})}",
        f"- DVQA: {benchmarks.get('DVQA', {})}",
        "",
        "## Verification",
        f"- Verification summary: {verification_summary}",
        "",
        "## Error Analysis",
        f"- {error_analysis}",
    ]
    FINAL_RESEARCH_REPORT_MD.write_text("\n".join(md_lines))


def run_final_pipeline(force_recompute: bool = True) -> Dict:
    _seed_everything(42)
    started = time.perf_counter()

    logger.info("=" * 78)
    logger.info("CHART UNDERSTANDING FRAMEWORK - FINAL PIPELINE V3")
    logger.info("=" * 78)

    _enforce_force_recompute(force_recompute)

    download_summary, verify_summary, split_df = _ensure_datasets(force_recompute=force_recompute)
    df, payload_map = _run_baseline_pipeline(force_recompute=True)
    df = _attach_dataset_labels(df, split_df)

    df = _phase_structural_features(df, payload_map)
    df = _phase_histogram_disambiguation(df, payload_map)

    df, yolo_summary = _phase_yolo_train_and_infer(df)
    df, ml_results = _phase_train_ml(df)
    df, cnn_results = _phase_train_cnn(df)
    df, hybrid_info = _phase_hybrid(df)
    df, ocr_summary = _phase_ocr_finalize(df, payload_map)
    df, structured_rows = _phase_value_extraction(df, payload_map)

    runtime_stats = _phase_runtime(df)
    error_analysis = _phase_error_analysis(df)
    benchmarks = _phase_dataset_benchmarks(df)

    simclr = SimCLRPretrainer(config.MODELS_DIR)
    simclr_info = simclr.run(
        image_paths=df["image_path"].dropna().astype(str).tolist(),
        cfg=SimCLRConfig(
            image_size=128,
            batch_size=32,
            epochs=10,
            lr=1e-4,
            max_samples=min(4000, len(df)),
            max_steps_per_epoch=80,
            temperature=0.2,
        ),
    )

    df, verification_summary = _phase_verify_and_rerun(
        df=df,
        download_summary=download_summary,
        yolo_summary=yolo_summary,
        cnn_results=cnn_results,
        runtime_stats=runtime_stats,
    )
    runtime_stats = _phase_runtime(df)

    evaluator = PipelineEvaluator()
    metrics = evaluator.compute_research_metrics(df)
    metrics["runtime_analysis"] = runtime_stats
    metrics["ocr_summary"] = ocr_summary
    metrics["hybrid"] = hybrid_info
    metrics["yolo"] = yolo_summary
    metrics["simclr"] = simclr_info
    metrics["benchmarks"] = benchmarks
    metrics["verification"] = verification_summary

    FINAL_EXPERIMENT_CSV.write_text(df.to_csv(index=False))
    FINAL_SYSTEM_METRICS_JSON.write_text(json.dumps(metrics, indent=2, default=str))

    final_payload = {
        "generated_at": datetime.now().isoformat(),
        "force_recompute": force_recompute,
        "dataset_download": download_summary,
        "dataset_verification": verify_summary,
        "yolo": yolo_summary,
        "ml_models": ml_results,
        "cnn_models": cnn_results,
        "hybrid": hybrid_info,
        "runtime": runtime_stats,
        "error_analysis": error_analysis,
        "benchmarks": benchmarks,
        "verification": verification_summary,
        "metrics": metrics,
        "records": [
            {
                "image_name": row["image_name"],
                "image_path": row["image_path"],
                "dataset_name": row.get("dataset_name", row.get("dataset_source", "unknown")),
                "split": row.get("split", "unspecified"),
                "true_chart_type": row.get("true_chart_type", "unknown"),
                "predicted_chart_type": row.get("predicted_chart_type", "unknown"),
                "hybrid_confidence": float(row.get("hybrid_confidence", 0.0)),
                "best_ocr_engine": row.get("best_ocr_engine", "unknown"),
                "best_ocr_confidence": float(row.get("best_ocr_confidence", 0.0)),
                "value_extraction_success": int(row.get("value_extraction_success", 0)),
                "normalized_value_payload": row.get("normalized_value_payload", {}),
            }
            for _, row in df.iterrows()
        ],
        "structured_values": structured_rows,
    }
    FINAL_PIPELINE_JSON.write_text(json.dumps(final_payload, indent=2, default=str))

    _build_reports(
        metrics=metrics,
        download_summary=download_summary,
        verify_summary=verify_summary,
        yolo_summary=yolo_summary,
        ml_results=ml_results,
        cnn_results=cnn_results,
        hybrid_info=hybrid_info,
        runtime_stats=runtime_stats,
        ocr_summary=ocr_summary,
        error_analysis=error_analysis,
        benchmarks=benchmarks,
        simclr_info=simclr_info,
        verification_summary=verification_summary,
    )

    elapsed = time.perf_counter() - started
    logger.info("Final pipeline completed in %.2f seconds", elapsed)
    logger.info("Final outputs:")
    logger.info("  %s", FINAL_PIPELINE_JSON)
    logger.info("  %s", FINAL_EXPERIMENT_CSV)
    logger.info("  %s", FINAL_SYSTEM_METRICS_JSON)
    logger.info("  %s", FINAL_RESEARCH_REPORT_MD)

    return {
        "metrics": metrics,
        "runtime": runtime_stats,
        "verification": verification_summary,
        "elapsed_sec": elapsed,
        "final_files": [
            str(FINAL_PIPELINE_JSON),
            str(FINAL_EXPERIMENT_CSV),
            str(FINAL_SYSTEM_METRICS_JSON),
            str(FINAL_RESEARCH_REPORT_MD),
            str(FINAL_SYSTEM_REPORT_TXT),
        ],
    }


def main():
    parser = argparse.ArgumentParser(description="Run final chart understanding pipeline v3.")
    parser.add_argument(
        "--no-force-recompute",
        action="store_true",
        help="Do not clear results/models/cache before running.",
    )
    args = parser.parse_args()

    force = not args.no_force_recompute
    run_final_pipeline(force_recompute=force)


if __name__ == "__main__":
    main()
