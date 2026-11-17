"""Ground-truth records: per-image JSON, dataset CSV, YOLO labels and instance masks."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from fluidsim.config import SimConfig
from fluidsim.iso4406 import CHANNELS_UM, iso_code_string
from fluidsim.scene import Bubble, Particle, Scene

SCHEMA_VERSION = "1.1"
CLASS_IDS = {"particle": 0, "bubble": 1}

CSV_FIELDS = [
    "image", "frame", "id", "class", "x_px", "y_px", "x_um", "y_um", "z_um",
    "size_um", "ecd_um", "bbox_x0", "bbox_y0", "bbox_x1", "bbox_y1",
    "inside_reference_square", "gt4", "gt6", "gt14", "vx_um_s", "vy_um_s",
]


def _to_px(cfg: SimConfig, x_um: float, y_um: float) -> tuple[float, float]:
    """Pixel coordinates with (0, 0) at the centre of the top-left pixel."""
    p = cfg.image.pixel_size_um
    return x_um / p - 0.5, y_um / p - 0.5


def object_record(scene: Scene, obj: Particle | Bubble) -> dict[str, Any]:
    cfg = scene.config
    x_px, y_px = _to_px(cfg, obj.x_um, obj.y_um)
    if isinstance(obj, Particle):
        v = obj.vertices_abs()
        xs, ys = v[:, 0], v[:, 1]
        cls = "particle"
        extra = {
            "transmission": round(obj.transmission, 4),
            "outline_px": [[round(a, 2), round(b, 2)] for a, b in (np.stack(_to_px(cfg, xs, ys), 1))],
        }
    else:
        r = obj.diameter_um / 2
        xs = np.array([obj.x_um - r, obj.x_um + r])
        ys = np.array([obj.y_um - r, obj.y_um + r])
        cls = "bubble"
        extra = {}
    bx0, by0 = _to_px(cfg, xs.min(), ys.min())
    bx1, by1 = _to_px(cfg, xs.max(), ys.max())
    vx, vy = scene.flow_velocity(obj.z_um)
    size = obj.size_um
    rec = {
        "id": obj.id,
        "class": cls,
        "x_px": round(x_px, 3),
        "y_px": round(y_px, 3),
        "x_um": round(obj.x_um, 3),
        "y_um": round(obj.y_um, 3),
        "z_um": round(obj.z_um, 3),
        "size_um": round(size, 3),
        "ecd_um": round(obj.ecd_um, 3),
        "bbox_px": [round(bx0, 2), round(by0, 2), round(bx1, 2), round(by1, 2)],
        "inside_reference_square": bool(scene.square.inside_interior(obj.x_um, obj.y_um)),
        "iso_channels": {f">{int(c)}um": bool(size > c) for c in CHANNELS_UM},
        "velocity_um_s": [round(vx, 3), round(vy, 3)],
    }
    rec.update(extra)
    return rec


def iso_summary(scene: Scene) -> dict[str, Any]:
    """Ground-truth ISO 4406 result: solid particles inside the square interior only."""
    cfg = scene.config
    volume_ml = cfg.square_interior_volume_ml
    inside = [p for p in scene.particles if scene.square.inside_interior(p.x_um, p.y_um)]
    counts = [sum(1 for p in inside if p.size_um > c) for c in CHANNELS_UM]
    per_ml = [c / volume_ml for c in counts]
    return {
        "volume_ml": volume_ml,
        "counts": {f">{int(c)}um": n for c, n in zip(CHANNELS_UM, counts)},
        "counts_per_ml": {f">{int(c)}um": round(v, 3) for c, v in zip(CHANNELS_UM, per_ml)},
        "iso4406_code": iso_code_string(per_ml),
        "bubbles_inside_square": sum(1 for b in scene.bubbles if scene.square.inside_interior(b.x_um, b.y_um)),
    }


def build_record(scene: Scene, image_name: str, seed: int, frame: int | None = None) -> dict[str, Any]:
    cfg = scene.config
    sq = scene.square
    corners_px = [list(map(lambda v: round(v, 3), _to_px(cfg, x, y))) for x, y in sq.corners_um()]
    p = cfg.image.pixel_size_um
    return {
        "schema_version": SCHEMA_VERSION,
        "image": image_name,
        "frame": frame,
        "time_s": round(scene.time_s, 6),
        "seed": seed,
        "config_fingerprint": cfg.fingerprint(),
        "image_size_px": [cfg.image.width_px, cfg.image.height_px],
        "bit_depth": cfg.image.bit_depth,
        "pixel_size_um": p,
        "film_thickness_um": cfg.cell.film_thickness_um,
        "focus_depth_um": cfg.focus_depth_um,
        "optics": {
            "n_liquid": cfg.optics.n_liquid,
            "n_bubble": cfg.optics.n_bubble,
            "objective_na": cfg.optics.objective_na,
            "condenser_na": cfg.optics.condenser_na,
            "critical_angle_deg": round(cfg.critical_angle_deg, 3),
        },
        "reference_square": {
            "side_um": sq.side_um,
            "line_width_um": sq.line_width_um,
            "rotation_deg": round(sq.rotation_deg, 4),
            "center_px": list(map(lambda v: round(v, 3), _to_px(cfg, *sq.center_um))),
            "outer_corners_px": corners_px,
            "side_px": round(sq.side_um / p, 3),
            "um_per_px": p,
        },
        "ground_truth_iso": iso_summary(scene),
        "objects": [object_record(scene, o) for o in scene.objects()],
    }


def write_json(record: dict[str, Any], path: Path) -> None:
    path.write_text(json.dumps(record, indent=2), encoding="utf-8")


def csv_rows(record: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for o in record["objects"]:
        ch = o["iso_channels"]
        rows.append({
            "image": record["image"], "frame": record["frame"], "id": o["id"], "class": o["class"],
            "x_px": o["x_px"], "y_px": o["y_px"], "x_um": o["x_um"], "y_um": o["y_um"], "z_um": o["z_um"],
            "size_um": o["size_um"], "ecd_um": o["ecd_um"],
            "bbox_x0": o["bbox_px"][0], "bbox_y0": o["bbox_px"][1],
            "bbox_x1": o["bbox_px"][2], "bbox_y1": o["bbox_px"][3],
            "inside_reference_square": o["inside_reference_square"],
            "gt4": ch[">4um"], "gt6": ch[">6um"], "gt14": ch[">14um"],
            "vx_um_s": o["velocity_um_s"][0], "vy_um_s": o["velocity_um_s"][1],
        })
    return rows


class CsvAppender:
    def __init__(self, path: Path):
        self.path = path
        self._fh = open(path, "w", newline="", encoding="utf-8")
        self._writer = csv.DictWriter(self._fh, fieldnames=CSV_FIELDS)
        self._writer.writeheader()

    def add(self, record: dict[str, Any]) -> None:
        self._writer.writerows(csv_rows(record))

    def close(self) -> None:
        self._fh.close()


def write_yolo(record: dict[str, Any], path: Path) -> None:
    w, h = record["image_size_px"]
    lines = []
    for o in record["objects"]:
        x0, y0, x1, y1 = o["bbox_px"]
        x0, x1 = max(x0, 0), min(x1, w - 1)
        y0, y1 = max(y0, 0), min(y1, h - 1)
        if x1 <= x0 or y1 <= y0:
            continue
        cx, cy = (x0 + x1) / 2 / w, (y0 + y1) / 2 / h
        lines.append(f"{CLASS_IDS[o['class']]} {cx:.6f} {cy:.6f} {(x1 - x0) / w:.6f} {(y1 - y0) / h:.6f}")
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def instance_mask(scene: Scene) -> np.ndarray:
    """uint16 mask: 0 = background, object id + 1 elsewhere (bubbles drawn over particles)."""
    cfg = scene.config
    p = cfg.image.pixel_size_um
    mask = np.zeros((cfg.image.height_px, cfg.image.width_px), dtype=np.uint16)
    shift = 4  # sub-pixel precision for cv2 drawing
    scale = (1 << shift) / p
    for part in scene.particles:
        pts = np.round(part.vertices_abs() * scale - (1 << shift) / 2).astype(np.int32)
        cv2.fillPoly(mask, [pts], int(part.id + 1), lineType=cv2.LINE_8, shift=shift)
    for b in scene.bubbles:
        c = (int(round((b.x_um / p - 0.5) * (1 << shift))), int(round((b.y_um / p - 0.5) * (1 << shift))))
        cv2.circle(mask, c, int(round(b.diameter_um / 2 / p * (1 << shift))), int(b.id + 1), -1, cv2.LINE_8, shift)
    return mask


def summarise_dataset(records: list[dict[str, Any]]) -> dict[str, Any]:
    n_p = sum(1 for r in records for o in r["objects"] if o["class"] == "particle")
    n_b = sum(1 for r in records for o in r["objects"] if o["class"] == "bubble")
    sizes = [o["size_um"] for r in records for o in r["objects"] if o["class"] == "particle"]
    return {
        "images": len(records),
        "particles": n_p,
        "bubbles": n_b,
        "particle_size_um": {
            "min": round(min(sizes), 2) if sizes else None,
            "median": round(float(np.median(sizes)), 2) if sizes else None,
            "max": round(max(sizes), 2) if sizes else None,
        },
        "iso_codes": [r["ground_truth_iso"]["iso4406_code"] for r in records],
        "mean_bubbles_per_image": round(n_b / max(len(records), 1), 2),
        "log10_particles_per_image": round(math.log10(max(n_p, 1) / max(len(records), 1)), 3),
    }
