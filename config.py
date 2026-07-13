import os
from pathlib import Path

# Automatically detect project root based on this file's location
PROJECT_ROOT = Path(__file__).parent.resolve()

# Directories
RAW_IMAGE_DIR = PROJECT_ROOT / "data" / "raw_images"
DATASETS_DIR = PROJECT_ROOT / "data" / "datasets"
RESULT_DIR = PROJECT_ROOT / "results"
CACHE_DIR = RESULT_DIR / "cache"
GROUND_TRUTH_DIR = RESULT_DIR / "ground_truth"
PLOTS_DIR = RESULT_DIR / "plots"
METRICS_DIR = RESULT_DIR / "metrics"
DEBUG_DIR = RESULT_DIR / "debug"
MODELS_DIR = PROJECT_ROOT / "models"
RESEARCH_METRICS_DIR = RESULT_DIR / "research_metrics"

# Ensure directories exist
for d in [
    RAW_IMAGE_DIR,
    DATASETS_DIR,
    RESULT_DIR,
    CACHE_DIR,
    GROUND_TRUTH_DIR,
    PLOTS_DIR,
    METRICS_DIR,
    DEBUG_DIR,
    MODELS_DIR,
    RESEARCH_METRICS_DIR,
]:
    d.mkdir(parents=True, exist_ok=True)

# Chart types
CHART_TYPES = [
    "bar_chart",
    "line_chart",
    "pie_chart",
    "scatter_plot",
    "histogram",
]

# OCR settings
OCR_LANG = "en"

# Multiprocessing
MAX_WORKERS = max(1, os.cpu_count() - 1) if os.cpu_count() else 2

# Pipeline controls
FORCE_RECOMPUTE = True

# Dataset targets - expanded for final research run
MIN_DATASET_SIZE = 5000
SYNTHETIC_TARGETS = {
    "bar": 1500,
    "line": 1000,
    "scatter": 1000,
    "pie": 750,
    "hist": 750,
}
