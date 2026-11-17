"""High-level dataset and sequence generation."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import cv2
import numpy as np

from fluidsim.config import SimConfig
from fluidsim.groundtruth import (
    CsvAppender,
    build_record,
    instance_mask,
    summarise_dataset,
    write_json,
    write_yolo,
)
from fluidsim.render import render_scene
from fluidsim.scene import build_scene
from fluidsim.viz import annotate


@dataclass
class OutputOptions:
    masks: bool = False
    yolo: bool = False
    previews: bool = False
    video: bool = False


def _prepare(out_dir: Path, opts: OutputOptions) -> None:
    (out_dir / "images").mkdir(parents=True, exist_ok=True)
    (out_dir / "labels").mkdir(exist_ok=True)
    if opts.masks:
        (out_dir / "masks").mkdir(exist_ok=True)
    if opts.yolo:
        (out_dir / "yolo").mkdir(exist_ok=True)
    if opts.previews:
        (out_dir / "previews").mkdir(exist_ok=True)


def _save_frame(out_dir: Path, name: str, scene, seed: int, opts: OutputOptions, frame=None):
    result = render_scene(scene, seed)
    record = build_record(scene, f"images/{name}.png", seed, frame)
    cv2.imwrite(str(out_dir / "images" / f"{name}.png"), result.image)
    write_json(record, out_dir / "labels" / f"{name}.json")
    if opts.masks:
        cv2.imwrite(str(out_dir / "masks" / f"{name}.png"), instance_mask(scene))
    if opts.yolo:
        write_yolo(record, out_dir / "yolo" / f"{name}.txt")
    if opts.previews:
        cv2.imwrite(str(out_dir / "previews" / f"{name}.jpg"), annotate(result.image, record))
    return result, record


def _write_meta(out_dir: Path, cfg: SimConfig, records, elapsed: float, kind: str) -> None:
    meta = {
        "kind": kind,
        "generator": "fluidsim",
        "elapsed_s": round(elapsed, 2),
        "config": cfg.to_dict(),
        "summary": summarise_dataset(records),
    }
    (out_dir / "dataset.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")


def generate_dataset(
    cfg: SimConfig, n_images: int, out_dir: Path, opts: OutputOptions,
    progress: Callable[[str], None] = print,
) -> list[dict]:
    out_dir = Path(out_dir)
    _prepare(out_dir, opts)
    csv = CsvAppender(out_dir / "annotations.csv")
    records = []
    t0 = time.perf_counter()
    try:
        for k in range(n_images):
            seed = cfg.seed * 1_000_003 + k
            scene = build_scene(cfg, np.random.default_rng(seed))
            t = time.perf_counter()
            _, rec = _save_frame(out_dir, f"img_{k:05d}", scene, seed, opts)
            csv.add(rec)
            records.append(rec)
            iso = rec["ground_truth_iso"]
            progress(
                f"[{k + 1}/{n_images}] img_{k:05d}: {len(scene.particles)} particles, "
                f"{len(scene.bubbles)} bubbles, GT ISO {iso['iso4406_code']} ({time.perf_counter() - t:.1f}s)"
            )
    finally:
        csv.close()
    _write_meta(out_dir, cfg, records, time.perf_counter() - t0, "images")
    return records


def generate_sequence(
    cfg: SimConfig, n_frames: int, out_dir: Path, opts: OutputOptions,
    progress: Callable[[str], None] = print,
) -> list[dict]:
    """Frames of the same scene advected by the plane Poiseuille flow (persistent object ids)."""
    out_dir = Path(out_dir)
    _prepare(out_dir, opts)
    csv = CsvAppender(out_dir / "annotations.csv")
    rng = np.random.default_rng(cfg.seed)
    scene = build_scene(cfg, rng)
    dt = cfg.flow.frame_interval_ms / 1000
    writer = None
    records = []
    t0 = time.perf_counter()
    try:
        for f in range(n_frames):
            seed = cfg.seed * 1_000_003 + f
            result, rec = _save_frame(out_dir, f"frame_{f:05d}", scene, seed, opts, frame=f)
            csv.add(rec)
            records.append(rec)
            if opts.video:
                img = result.image if result.image.dtype == np.uint8 else (result.image >> 8).astype(np.uint8)
                if writer is None:
                    h, w = img.shape
                    fps = 1000 / cfg.flow.frame_interval_ms
                    writer = cv2.VideoWriter(str(out_dir / "sequence.avi"), cv2.VideoWriter_fourcc(*"FFV1"), fps, (w, h), False)
                writer.write(img)
            progress(f"[{f + 1}/{n_frames}] frame_{f:05d} t={scene.time_s * 1000:.0f} ms")
            scene.advance(dt)
    finally:
        csv.close()
        if writer is not None:
            writer.release()
    _write_meta(out_dir, cfg, records, time.perf_counter() - t0, "sequence")
    return records
