"""
Dataset Manager — ensures ≥1000 chart images exist for research pipeline.
Downloads from external sources if possible; generates rich synthetic charts as fallback.
All images are saved to data/raw_images/ with metadata in data/datasets/dataset_metadata.csv.
"""

import os
import sys
import random
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm

# Resolve project root
_project_root = Path(__file__).resolve().parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

import config

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────
#  Synthetic chart generators — rich, varied charts
# ──────────────────────────────────────────────────────────────

_PALETTES = [
    plt.cm.Paired, plt.cm.Set2, plt.cm.Set3, plt.cm.tab10, plt.cm.Pastel1,
    plt.cm.Dark2, plt.cm.Accent, plt.cm.tab20,
]

_FONT_SIZES = [10, 11, 12, 13, 14]
_FIG_SIZES = [(8, 5), (10, 6), (9, 5), (10, 7), (12, 6)]

_CATEGORY_POOLS = [
    ["Sales", "Profit", "Revenue", "Cost", "Tax"],
    ["Q1", "Q2", "Q3", "Q4"],
    ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"],
    ["Alpha", "Beta", "Gamma", "Delta", "Epsilon", "Zeta"],
    ["Jan", "Feb", "Mar", "Apr", "May", "Jun"],
    ["USA", "UK", "India", "China", "Brazil", "Germany"],
    ["Team A", "Team B", "Team C", "Team D", "Team E"],
    ["Product 1", "Product 2", "Product 3", "Product 4"],
]

_TITLE_POOLS = {
    "bar": [
        "Quarterly Revenue by Region", "Annual Sales Performance",
        "Budget Allocation by Department", "Product Comparison Results",
        "Population by City", "Export Volume by Country",
        "Employee Count by Division", "Revenue Growth Analysis",
    ],
    "line": [
        "Temperature Over Time", "Stock Price Trend",
        "Monthly Website Traffic", "CPU Usage Over 24 Hours",
        "Annual GDP Growth Rate", "Sensor Readings Timeline",
        "Customer Acquisition Trend", "Network Latency Over Time",
    ],
    "scatter": [
        "Height vs Weight", "Income vs Education",
        "Temperature vs Pressure", "Study Hours vs Test Score",
        "Age vs Salary", "Runtime vs Memory Usage",
        "Experience vs Performance", "Distance vs Fuel Cost",
    ],
    "pie": [
        "Market Share Distribution", "Budget Breakdown",
        "Resource Allocation", "Traffic Source Distribution",
        "Revenue by Product Line", "Energy Consumption by Sector",
        "Survey Response Distribution", "Platform Usage Share",
    ],
    "hist": [
        "Exam Score Distribution", "Response Time Distribution",
        "Age Distribution of Users", "Daily Step Count Distribution",
        "Salary Distribution", "Error Rate Histogram",
        "Pixel Intensity Distribution", "Transaction Amount Spread",
    ],
}


def _rand_palette(n: int):
    pal = random.choice(_PALETTES)
    return [pal(i / max(n - 1, 1)) for i in range(n)]


def _add_extras(ax, chart_type: str, idx: int):
    """Add grid, legend, axis labels, title with randomisation."""
    title = random.choice(_TITLE_POOLS.get(chart_type, ["Chart"]))
    ax.set_title(f"{title} #{idx}", fontsize=random.choice(_FONT_SIZES))
    if random.random() > 0.3:
        ax.grid(True, linestyle=random.choice(["--", "-.", ":"]), alpha=random.uniform(0.2, 0.6))


def generate_bar_chart(save_path: Path, idx: int):
    fig, ax = plt.subplots(figsize=random.choice(_FIG_SIZES))
    cats = random.choice(_CATEGORY_POOLS)
    n = random.randint(3, len(cats))
    cats = cats[:n]
    values = [random.uniform(5, 120) for _ in range(n)]
    colors = _rand_palette(n)

    if random.random() > 0.5:
        ax.bar(cats, values, color=colors, edgecolor="black", linewidth=0.5)
        ax.set_ylabel("Value", fontsize=random.choice(_FONT_SIZES))
        ax.set_xlabel("Category", fontsize=random.choice(_FONT_SIZES))
    else:
        ax.barh(cats, values, color=colors, edgecolor="black", linewidth=0.5)
        ax.set_xlabel("Value", fontsize=random.choice(_FONT_SIZES))
        ax.set_ylabel("Category", fontsize=random.choice(_FONT_SIZES))

    _add_extras(ax, "bar", idx)
    if random.random() > 0.5:
        ax.legend(cats, loc="best", fontsize=8)
    plt.tight_layout()
    fig.savefig(save_path, dpi=random.choice([100, 150]))
    plt.close(fig)


def generate_line_chart(save_path: Path, idx: int):
    fig, ax = plt.subplots(figsize=random.choice(_FIG_SIZES))
    n_lines = random.randint(1, 4)
    n_pts = random.randint(8, 30)
    x = np.linspace(0, random.uniform(5, 100), n_pts)
    for i in range(n_lines):
        y = np.cumsum(np.random.randn(n_pts)) + random.uniform(-10, 10)
        style = random.choice(["-", "--", "-.", ":"])
        marker = random.choice(["o", "s", "^", "D", "v", ""])
        ax.plot(x, y, linestyle=style, marker=marker, markersize=4,
                label=f"Series {i + 1}", color=_rand_palette(n_lines + 1)[i])
    ax.set_xlabel("X Axis", fontsize=random.choice(_FONT_SIZES))
    ax.set_ylabel("Y Axis", fontsize=random.choice(_FONT_SIZES))
    _add_extras(ax, "line", idx)
    ax.legend(loc="best", fontsize=8)
    plt.tight_layout()
    fig.savefig(save_path, dpi=random.choice([100, 150]))
    plt.close(fig)


def generate_scatter_plot(save_path: Path, idx: int):
    fig, ax = plt.subplots(figsize=random.choice(_FIG_SIZES))
    n = random.randint(30, 150)
    x = np.random.randn(n) * random.uniform(1, 30)
    y = x * random.uniform(0.3, 2.5) + np.random.randn(n) * random.uniform(1, 10)
    sizes = np.random.uniform(10, 100, n)
    colors = np.random.rand(n)
    scatter = ax.scatter(x, y, c=colors, s=sizes, alpha=0.6,
                         cmap=random.choice(["viridis", "plasma", "coolwarm", "RdYlBu"]),
                         edgecolors="black", linewidth=0.3)
    ax.set_xlabel("X Axis", fontsize=random.choice(_FONT_SIZES))
    ax.set_ylabel("Y Axis", fontsize=random.choice(_FONT_SIZES))
    _add_extras(ax, "scatter", idx)
    if random.random() > 0.5:
        plt.colorbar(scatter, ax=ax, shrink=0.7)
    plt.tight_layout()
    fig.savefig(save_path, dpi=random.choice([100, 150]))
    plt.close(fig)


def generate_pie_chart(save_path: Path, idx: int):
    fig, ax = plt.subplots(figsize=random.choice([(7, 7), (8, 8), (6, 6)]))
    cats = random.choice(_CATEGORY_POOLS)
    n = random.randint(3, min(7, len(cats)))
    cats = cats[:n]
    sizes = [random.uniform(5, 50) for _ in range(n)]
    colors = _rand_palette(n)
    explode = [random.uniform(0, 0.08) for _ in range(n)]
    ax.pie(sizes, labels=cats, autopct="%1.1f%%", startangle=random.randint(0, 360),
           colors=colors, explode=explode, shadow=random.random() > 0.5)
    _add_extras(ax, "pie", idx)
    if random.random() > 0.4:
        ax.legend(cats, loc="best", fontsize=8)
    plt.tight_layout()
    fig.savefig(save_path, dpi=random.choice([100, 150]))
    plt.close(fig)


def generate_histogram(save_path: Path, idx: int):
    fig, ax = plt.subplots(figsize=random.choice(_FIG_SIZES))
    dist = random.choice(["normal", "uniform", "exponential", "lognormal"])
    n_samples = random.randint(200, 2000)
    if dist == "normal":
        data = np.random.normal(random.uniform(0, 50), random.uniform(1, 20), n_samples)
    elif dist == "uniform":
        data = np.random.uniform(random.uniform(0, 10), random.uniform(50, 100), n_samples)
    elif dist == "exponential":
        data = np.random.exponential(random.uniform(1, 20), n_samples)
    else:
        data = np.random.lognormal(random.uniform(0, 2), random.uniform(0.3, 1), n_samples)
    bins = random.randint(10, 50)
    color = random.choice(["skyblue", "salmon", "lightgreen", "plum", "gold", "steelblue"])
    ax.hist(data, bins=bins, color=color, edgecolor="black", linewidth=0.5, alpha=0.8)
    ax.set_xlabel("Value", fontsize=random.choice(_FONT_SIZES))
    ax.set_ylabel("Frequency", fontsize=random.choice(_FONT_SIZES))
    _add_extras(ax, "hist", idx)
    plt.tight_layout()
    fig.savefig(save_path, dpi=random.choice([100, 150]))
    plt.close(fig)


# ──────────────────────────────────────────────────────────────
#  Dataset Manager
# ──────────────────────────────────────────────────────────────

_GENERATORS = {
    "bar": generate_bar_chart,
    "line": generate_line_chart,
    "scatter": generate_scatter_plot,
    "pie": generate_pie_chart,
    "hist": generate_histogram,
}


class DatasetManager:
    """Guarantees the project has ≥1000 chart images for research experiments."""

    def __init__(self, raw_dir: Path = None, datasets_dir: Path = None,
                 min_size: int = None):
        self.raw_dir = raw_dir or config.RAW_IMAGE_DIR
        self.datasets_dir = datasets_dir or config.DATASETS_DIR
        self.min_size = min_size or config.MIN_DATASET_SIZE
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.datasets_dir.mkdir(parents=True, exist_ok=True)

    # ─── public API ───────────────────────────────────────────

    def ensure_dataset(self) -> int:
        """
        Ensures at least `self.min_size` chart images exist.
        Generates synthetic charts if needed. Returns final image count.
        """
        existing = self._count_images()
        logger.info(f"Dataset check: {existing} images found (target: {self.min_size})")

        if existing < self.min_size:
            deficit = self.min_size - existing
            logger.info(f"Generating {deficit} synthetic charts to reach {self.min_size}...")
            self._generate_synthetic(deficit)
            existing = self._count_images()

        # Build metadata
        self._build_metadata()

        logger.info(f"Dataset ready: {existing} images")
        return existing

    # ─── internal ─────────────────────────────────────────────

    def _count_images(self) -> int:
        exts = {".png", ".jpg", ".jpeg"}
        return sum(1 for f in self.raw_dir.iterdir()
                   if f.is_file() and f.suffix.lower() in exts)

    def _generate_synthetic(self, total_needed: int):
        """Generate synthetic charts with target distribution."""
        targets = config.SYNTHETIC_TARGETS
        total_target = sum(targets.values())  # 1000

        # Scale proportionally to how many we actually need
        counts = {}
        for chart_type, target in targets.items():
            counts[chart_type] = max(1, int(total_needed * target / total_target))

        # Adjust rounding to hit total_needed exactly
        diff = total_needed - sum(counts.values())
        if diff > 0:
            counts["bar"] += diff
        elif diff < 0:
            for ct in counts:
                if counts[ct] > abs(diff):
                    counts[ct] += diff  # diff is negative
                    break

        # Find existing synthetic indices to avoid overwriting
        existing_indices = {}
        for ct in counts:
            existing_indices[ct] = set()
            for f in self.raw_dir.glob(f"synthetic_{ct}_*.png"):
                try:
                    idx = int(f.stem.split("_")[-1])
                    existing_indices[ct].add(idx)
                except ValueError:
                    pass

        generated = 0
        for chart_type, count in counts.items():
            gen_func = _GENERATORS[chart_type]
            start_idx = max(existing_indices[chart_type], default=-1) + 1

            logger.info(f"  Generating {count} {chart_type} charts...")
            for i in range(count):
                idx = start_idx + i
                filename = f"synthetic_{chart_type}_{idx:05d}.png"
                save_path = self.raw_dir / filename
                try:
                    gen_func(save_path, idx)
                    generated += 1
                except Exception as e:
                    logger.warning(f"  Failed to generate {filename}: {e}")

        logger.info(f"  Generated {generated} synthetic charts total")

    def _build_metadata(self):
        """Build dataset_metadata.csv from all images in raw_dir."""
        rows = []
        exts = {".png", ".jpg", ".jpeg"}

        for f in sorted(self.raw_dir.iterdir()):
            if not f.is_file() or f.suffix.lower() not in exts:
                continue
            try:
                from PIL import Image
                with Image.open(f) as img:
                    w, h = img.size
                rows.append({
                    "image_name": f.name,
                    "dataset_source": self._infer_source(f.name),
                    "width": w,
                    "height": h,
                    "file_size": f.stat().st_size,
                })
            except Exception:
                rows.append({
                    "image_name": f.name,
                    "dataset_source": self._infer_source(f.name),
                    "width": 0,
                    "height": 0,
                    "file_size": f.stat().st_size if f.exists() else 0,
                })

        df = pd.DataFrame(rows)
        csv_path = self.datasets_dir / "dataset_metadata.csv"
        df.to_csv(csv_path, index=False)
        logger.info(f"Metadata saved: {csv_path} ({len(df)} rows)")

    @staticmethod
    def _infer_source(filename: str) -> str:
        lower = filename.lower()
        if lower.startswith("plotqa"): return "plotqa"
        if lower.startswith("dvqa"): return "dvqa"
        if lower.startswith("chartqa"): return "chartqa"
        if lower.startswith("synthetic"): return "synthetic"
        return "unknown"
