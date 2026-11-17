"""Annotated previews of generated images (ground truth drawn on top)."""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np

COLORS = {"particle": (40, 40, 230), "bubble": (230, 160, 30)}  # BGR


def to_display(image: np.ndarray) -> np.ndarray:
    if image.dtype != np.uint8:
        image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR) if image.ndim == 2 else image.copy()


def annotate(image: np.ndarray, record: dict[str, Any], max_side: int = 2000) -> np.ndarray:
    out = to_display(image)
    sq = record["reference_square"]
    pts = np.array(sq["outer_corners_px"], dtype=np.float32).reshape(-1, 1, 2)
    cv2.polylines(out, [np.round(pts).astype(np.int32)], True, (60, 200, 60), 3)
    thick = max(2, out.shape[1] // 1500)
    for o in record["objects"]:
        x0, y0, x1, y1 = (int(round(v)) for v in o["bbox_px"])
        pad = 4
        cv2.rectangle(out, (x0 - pad, y0 - pad), (x1 + pad, y1 + pad), COLORS[o["class"]], thick)
    scale = min(1.0, max_side / max(out.shape[:2]))
    if scale < 1.0:
        out = cv2.resize(out, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    iso = record["ground_truth_iso"]
    text = f"GT ISO 4406 {iso['iso4406_code']}  particles>4um in square: {iso['counts']['>4um']}"
    cv2.putText(out, text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(out, text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)
    return out


def crop_gallery(image: np.ndarray, record: dict[str, Any], n: int = 12, tile: int = 96) -> np.ndarray:
    """Side-by-side crops of the largest particles and bubbles, for quick visual checks."""
    disp = to_display(image)
    rows = []
    for cls in ("particle", "bubble"):
        objs = sorted((o for o in record["objects"] if o["class"] == cls), key=lambda o: -o["size_um"])[:n]
        tiles = []
        for o in objs:
            x0, y0, x1, y1 = o["bbox_px"]
            half = int(max(x1 - x0, y1 - y0) * 0.8) + 6
            cx, cy = int(o["x_px"]), int(o["y_px"])
            c = disp[max(cy - half, 0): cy + half, max(cx - half, 0): cx + half]
            if c.size == 0:
                continue
            tiles.append(cv2.resize(c, (tile, tile), interpolation=cv2.INTER_NEAREST))
        tiles += [np.full((tile, tile, 3), 255, np.uint8)] * (n - len(tiles))
        rows.append(np.hstack(tiles))
    return np.vstack(rows)
