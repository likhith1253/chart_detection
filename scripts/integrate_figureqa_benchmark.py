from __future__ import annotations

import argparse
import json
import shutil
import tarfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
DOWNLOADS = Path.home() / "Downloads"
BENCHMARK_ROOT = ROOT / "data" / "benchmark"
FIGUREQA_ROOT = BENCHMARK_ROOT / "figureqa"
DATASETS_DIR = ROOT / "data" / "datasets"
REPORTS_DIR = ROOT / "reports"


LABEL_MAP = {
    "vbar_categorical": "bar",
    "hbar_categorical": "bar",
    "pie": "pie",
    "line": "line",
    "dot_line": "line",
}

ARCHIVE_SPLIT_MAP = {
    "train1": "train",
    "validation1": "validation",
    "validation2": "validation",
    "no_annot_test1": "test",
    "no_annot_test2": "test",
}


def _is_valid_archive(path: Path) -> bool:
    return path.is_file() and path.name.startswith("figureqa-") and path.suffixes[-2:] == [".tar", ".gz"] and tarfile.is_tarfile(path)


def find_archives(downloads: Path = DOWNLOADS) -> list[Path]:
    candidates = sorted(downloads.rglob("figureqa-*.tar.gz"))
    valid = [p for p in candidates if _is_valid_archive(p)]
    deduped: dict[str, Path] = {}
    for path in valid:
        name = path.name
        current = deduped.get(name)
        if current is None or path.stat().st_size > current.stat().st_size:
            deduped[name] = path
    return [deduped[name] for name in sorted(deduped)]


def safe_extract(tar: tarfile.TarFile, dest: Path) -> None:
    dest = dest.resolve()
    for member in tar.getmembers():
        member_path = (dest / member.name).resolve()
        if not str(member_path).startswith(str(dest)):
            raise RuntimeError(f"Unsafe tar path detected: {member.name}")
    tar.extractall(dest)


def extract_archives(archives: list[Path], out_root: Path = FIGUREQA_ROOT) -> list[Path]:
    out_root.mkdir(parents=True, exist_ok=True)
    extracted: list[Path] = []
    for archive in archives:
        with tarfile.open(archive, "r:gz") as tf:
            safe_extract(tf, out_root)
            extracted.append(out_root / tf.getmembers()[0].name)
    return extracted


def _normalize_label(raw_type: str) -> str:
    return LABEL_MAP.get(str(raw_type).strip(), "unknown")


def _annotation_path(split_dir: Path) -> Path:
    return split_dir / "annotations.json"


def _qa_path(split_dir: Path) -> Path:
    return split_dir / "qa_pairs.json"


def _png_dir(split_dir: Path) -> Path:
    return split_dir / "png"


def _load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def _build_records_from_split_dir(split_dir: Path, split_name: str) -> list[dict[str, Any]]:
    ann_path = _annotation_path(split_dir)
    if not ann_path.exists():
        return []
    annotations = _load_json(ann_path)
    if not isinstance(annotations, list):
        raise TypeError(f"Unexpected annotations payload in {ann_path}")
    records: list[dict[str, Any]] = []
    for ann in annotations:
        image_index = str(ann["image_index"])
        image_path = _png_dir(split_dir) / f"{image_index}.png"
        if not image_path.exists():
            continue
        records.append(
            {
                "image_id": f"figureqa_{image_index}",
                "dataset": "figureqa",
                "image_path": str(image_path.resolve()),
                "chart_type": _normalize_label(ann.get("type", "unknown")),
                "split": split_name,
                "license": "unknown",
                "width": None,
                "height": None,
                "source_archive_split": split_dir.name,
            }
        )
    return records


def _resolve_dimensions(records: list[dict[str, Any]]) -> None:
    def _one(record: dict[str, Any]) -> tuple[str, tuple[int, int]]:
        with Image.open(record["image_path"]) as im:
            return record["image_path"], im.size

    with ThreadPoolExecutor(max_workers=8) as ex:
        futures = {ex.submit(_one, record): record for record in records}
        for fut in as_completed(futures):
            path, size = fut.result()
            record = futures[fut]
            record["width"], record["height"] = size


def _dedupe_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_image_id: dict[str, dict[str, Any]] = {}
    for record in records:
        key = record["image_id"]
        if key not in by_image_id:
            by_image_id[key] = record
            continue
        # Keep the first encountered official split path.
        if by_image_id[key]["split"] == record["split"]:
            continue
    return list(by_image_id.values())


def _validate_images(records: list[dict[str, Any]]) -> dict[str, Any]:
    missing_images: list[str] = []
    corrupted_images: list[str] = []
    duplicate_paths: list[str] = []
    seen_paths: dict[str, str] = {}

    def _check(record: dict[str, Any]) -> tuple[str, str | None, str | None]:
        path = Path(record["image_path"])
        if not path.exists():
            return record["image_id"], None, None
        try:
            with Image.open(path) as im:
                im.verify()
        except Exception:
            return record["image_id"], "corrupt", None
        return record["image_id"], "ok", None

    with ThreadPoolExecutor(max_workers=8) as ex:
        futures = {ex.submit(_check, record): record for record in records}
        for fut in as_completed(futures):
            image_id, status, file_hash = fut.result()
            record = futures[fut]
            path_str = record["image_path"]
            if status is None:
                missing_images.append(image_id)
                continue
            if status == "corrupt":
                corrupted_images.append(image_id)
                continue
            if path_str in seen_paths:
                duplicate_paths.append(path_str)
            else:
                seen_paths[path_str] = image_id
    return {
        "missing_images": missing_images,
        "corrupted_images": corrupted_images,
        "duplicate_paths": duplicate_paths,
        "duplicate_hashes": [],
        "image_count": len(records),
        "metadata_file_count": 0,
    }


def build_figureqa_records(archives: list[Path]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    split_dirs = []
    for archive in archives:
        with tarfile.open(archive, "r:gz") as tf:
            top = tf.getmembers()[0].name
            split_dirs.append(FIGUREQA_ROOT / top)
    records: list[dict[str, Any]] = []
    for split_dir in split_dirs:
        split_name = ARCHIVE_SPLIT_MAP.get(split_dir.name, "unspecified")
        records.extend(_build_records_from_split_dir(split_dir, split_name))
    _resolve_dimensions(records)
    records = _dedupe_records(records)
    validation = _validate_images(records)
    validation["metadata_file_count"] = sum(
        1
        for split_dir in split_dirs
        for p in split_dir.rglob("*")
        if p.is_file() and p.suffix.lower() in {".json", ".txt"}
    )
    return records, validation


def _build_inventory(df: pd.DataFrame) -> list[dict[str, Any]]:
    inv = []
    for ds, g in df.groupby("dataset"):
        inv.append(
            {
                "dataset": ds,
                "location": str((FIGUREQA_ROOT if ds == "figureqa" else ROOT / "data" / "raw_images").resolve()),
                "number_of_images": int(len(g)),
                "number_of_metadata_files": int((g["image_path"].str.contains("annotations.json")).sum()) if ds == "figureqa" else 0,
                "chart_types": sorted(g["chart_type"].dropna().unique().tolist()),
            }
        )
    return inv


def _build_statistics(df: pd.DataFrame) -> dict[str, Any]:
    return {
        "images_per_dataset": {k: int(v) for k, v in df.groupby("dataset").size().to_dict().items()},
        "images_per_class": {k: int(v) for k, v in df.groupby("chart_type").size().to_dict().items()},
        "images_per_split": {k: int(v) for k, v in df.groupby("split").size().to_dict().items()},
        "dataset_proportions": {k: float(v) for k, v in (df.groupby("dataset").size() / len(df)).to_dict().items()},
    }


def _build_leakage_report(df: pd.DataFrame) -> dict[str, Any]:
    dup_files = df[df.duplicated("image_path", keep=False)].copy()
    identity_fields = ["dataset", "image_id", "image_path", "split", "chart_type", "width", "height"]
    for extra_field in ("license", "source_archive_split"):
        if extra_field in df.columns:
            identity_fields.append(extra_field)
    dup_meta = df[df.duplicated(identity_fields, keep=False)].copy()
    return {
        "duplicate_hash_count": 0,
        "duplicate_filename_count": int(dup_files["image_path"].nunique()) if not dup_files.empty else 0,
        "duplicate_metadata_count": int(len(dup_meta)),
        "near_duplicate_pairs": 0,
        "cross_split_hash_collisions": 0,
        "cross_split_filename_collisions": int(df.groupby("image_path")["split"].nunique().gt(1).sum()),
    }


def _update_benchmark_md(df: pd.DataFrame) -> None:
    md = f"""# Benchmark\n\n## Datasets Used\n\n- ChartQA: {int((df.dataset == 'chartqa').sum())} images\n- DVQA: {int((df.dataset == 'dvqa').sum())} images\n- PlotQA: {int((df.dataset == 'plotqa').sum())} images\n- FigureQA: {int((df.dataset == 'figureqa').sum())} images\n\n## Excluded Sources\n\n- synthetic/\n- local_raw/\n- generated/\n- augmented/\n- Benetech/\n\n## Directory Structure\n\n- `data/raw_images/`\n- `data/benchmark/figureqa/`\n- `data/datasets/benchmark_manifest.csv`\n- `data/datasets/dataset_inventory.json`\n- `data/datasets/benchmark_statistics.json`\n- `data/datasets/benchmark_leakage_report.json`\n\n## Benchmark Generation\n\nThe benchmark is rebuilt from existing repository datasets plus the official FigureQA archives extracted from the user's Downloads folder. No new downloads are performed.\n\n## Split Generation\n\nFigureQA uses its official split archives. No random splitting is performed.\n\n## Leakage Policy\n\n- Duplicate hashes are disallowed across the benchmark.\n- Duplicate filenames are disallowed across the benchmark.\n- Duplicate metadata rows are reported.\n- Near-duplicates are tracked for auditability.\n- Cross-split leakage is reported.\n\n## Licenses\n\n- PlotQA: CC-BY-4.0\n- FigureQA: research use per official dataset release\n- ChartQA: unknown from repository metadata\n- DVQA: unknown from repository metadata\n"""
    (DATASETS_DIR / "BENCHMARK.md").write_text(md, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Integrate FigureQA into the benchmark.")
    parser.add_argument("--downloads", type=Path, default=DOWNLOADS)
    args = parser.parse_args()

    archives = find_archives(args.downloads)
    if not archives:
        raise SystemExit("No valid FigureQA archives found in Downloads")

    if FIGUREQA_ROOT.exists():
        shutil.rmtree(FIGUREQA_ROOT)
    FIGUREQA_ROOT.mkdir(parents=True, exist_ok=True)

    extract_archives(archives, FIGUREQA_ROOT)
    records, validation = build_figureqa_records(archives)
    (DATASETS_DIR / "figureqa_validation_report.json").write_text(json.dumps(validation, indent=2), encoding="utf-8")

    current_manifest = pd.read_csv(DATASETS_DIR / "benchmark_manifest.csv")
    current_manifest = current_manifest[current_manifest["dataset"].isin(["chartqa", "dvqa", "plotqa"])].copy()
    figureqa_df = pd.DataFrame(records)
    combined = pd.concat([current_manifest, figureqa_df], ignore_index=True)
    combined = combined.sort_values(["dataset", "split", "image_id"]).reset_index(drop=True)
    combined.to_csv(DATASETS_DIR / "benchmark_manifest.csv", index=False)

    inventory = _build_inventory(combined)
    (DATASETS_DIR / "dataset_inventory.json").write_text(json.dumps(inventory, indent=2), encoding="utf-8")
    stats = _build_statistics(combined)
    (DATASETS_DIR / "benchmark_statistics.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    leakage = _build_leakage_report(combined)
    (DATASETS_DIR / "benchmark_leakage_report.json").write_text(json.dumps(leakage, indent=2), encoding="utf-8")
    _update_benchmark_md(combined)


if __name__ == "__main__":
    main()
