"""
Typed configuration loader for reproducible research experiments.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml


@dataclass
class OCRConfig:
    enabled_engines: List[str] = field(default_factory=lambda: ["paddleocr", "easyocr"])
    ensemble_enabled: bool = True
    paddle_retry_count: int = 1
    disable_mkldnn_on_retry: bool = True
    force_engine: Optional[str] = None


@dataclass
class PreprocessingConfig:
    enabled: bool = True
    resize_width: int = 640
    resize_height: int = 480
    grayscale: bool = True
    clahe: bool = True
    adaptive_threshold: bool = True
    denoise: bool = True
    morphology_cleanup: bool = True
    deskew: bool = True
    edge_detection: bool = True


@dataclass
class CacheConfig:
    enabled: bool = True
    root_dir: str = "results/cache"
    ocr_subdir: str = "ocr"
    invalidate: bool = False


@dataclass
class ParallelConfig:
    enabled: bool = True
    worker_count: int = 0  # 0 => auto
    max_worker_cap: int = 32
    executor_type: str = "thread"


@dataclass
class DatasetConfig:
    raw_images_dir: str = "data/raw_images"
    datasets_dir: str = "data/datasets"
    manifest_path: str = "data/dataset_manifest.json"
    metadata_csv: str = "data/datasets/dataset_metadata.csv"
    verified_labels_csv: str = "data/datasets/verified_labels.csv"
    min_size: int = 6500
    split_seed: int = 42
    split_ratio: Dict[str, float] = field(
        default_factory=lambda: {"train": 0.7, "val": 0.15, "test": 0.15}
    )
    max_images: int = 0  # 0 => all


@dataclass
class OutputConfig:
    results_dir: str = "results"
    logs_dir: str = "logs"
    plots_dir: str = "results/plots"
    pipeline_output: str = "results/pipeline_output.json"
    experiment_csv: str = "results/experiment_results.csv"
    research_metrics: str = "results/research_metrics.json"
    runtime_profile: str = "results/runtime_profile.json"
    experiment_config: str = "results/experiment_config.json"
    research_report: str = "results/research_report.md"
    ablation_csv: str = "results/ablation_comparison.csv"


@dataclass
class LoggingConfig:
    level: str = "INFO"
    structured_json: bool = True
    file_name: str = "progress.log"


@dataclass
class AblationConfig:
    enabled: bool = True
    variants: List[Dict[str, Any]] = field(
        default_factory=lambda: [
            {"name": "easyocr_only", "ocr_force_engine": "easyocr", "preprocessing_enabled": True},
            {"name": "paddleocr_only", "ocr_force_engine": "paddleocr", "preprocessing_enabled": True},
            {"name": "ocr_ensemble", "ocr_force_engine": None, "preprocessing_enabled": True},
            {"name": "ensemble_no_preprocessing", "ocr_force_engine": None, "preprocessing_enabled": False},
        ]
    )


@dataclass
class PipelineConfig:
    seed: int = 42
    batch_size: int = 64
    run_ablation: bool = True
    ocr: OCRConfig = field(default_factory=OCRConfig)
    preprocessing: PreprocessingConfig = field(default_factory=PreprocessingConfig)
    cache: CacheConfig = field(default_factory=CacheConfig)
    parallel: ParallelConfig = field(default_factory=ParallelConfig)
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    outputs: OutputConfig = field(default_factory=OutputConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    ablation: AblationConfig = field(default_factory=AblationConfig)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _deep_update(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_update(merged[key], value)
        else:
            merged[key] = value
    return merged


def _to_config(data: Dict[str, Any]) -> PipelineConfig:
    cfg = PipelineConfig()

    ocr_data = data.get("ocr", {})
    prep_data = data.get("preprocessing", {})
    cache_data = data.get("cache", {})
    par_data = data.get("parallel", {})
    ds_data = data.get("dataset", {})
    out_data = data.get("outputs", {})
    log_data = data.get("logging", {})
    abl_data = data.get("ablation", {})

    cfg.seed = int(data.get("seed", cfg.seed))
    cfg.batch_size = int(data.get("batch_size", cfg.batch_size))
    cfg.run_ablation = bool(data.get("run_ablation", cfg.run_ablation))

    cfg.ocr = OCRConfig(**_deep_update(asdict(cfg.ocr), ocr_data))
    cfg.preprocessing = PreprocessingConfig(
        **_deep_update(asdict(cfg.preprocessing), prep_data)
    )
    cfg.cache = CacheConfig(**_deep_update(asdict(cfg.cache), cache_data))
    cfg.parallel = ParallelConfig(**_deep_update(asdict(cfg.parallel), par_data))
    cfg.dataset = DatasetConfig(**_deep_update(asdict(cfg.dataset), ds_data))
    cfg.outputs = OutputConfig(**_deep_update(asdict(cfg.outputs), out_data))
    cfg.logging = LoggingConfig(**_deep_update(asdict(cfg.logging), log_data))
    cfg.ablation = AblationConfig(**_deep_update(asdict(cfg.ablation), abl_data))
    return cfg


def load_pipeline_config(config_path: str | Path = "configs/pipeline.yaml") -> PipelineConfig:
    path = Path(config_path)
    default_cfg = PipelineConfig()
    if not path.exists():
        return default_cfg

    with path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}

    merged = _deep_update(default_cfg.to_dict(), raw)
    return _to_config(merged)
