"""2-D particle outlines and size metrics.

Solid contaminant particles (wear metal, silica, fibres) are modelled as thin,
opaque, irregular flakes. Their outline is a star-shaped polygon whose radius is
perturbed by a few random Fourier harmonics, stretched to an aspect ratio and
rotated. Size follows the microscope convention of ISO 4407: the longest
dimension (maximum Feret diameter) of the projected outline.
"""

from __future__ import annotations

import math

import numpy as np

MAX_VERTICES = 48


def polygon_area(v: np.ndarray) -> float:
    x, y = v[:, 0], v[:, 1]
    return 0.5 * abs(float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))


def max_feret(v: np.ndarray) -> float:
    """Longest distance between any two vertices (exact for polygons)."""
    d = v[:, None, :] - v[None, :, :]
    return float(np.sqrt((d**2).sum(-1)).max())


def equivalent_circle_diameter(area: float) -> float:
    return 2.0 * math.sqrt(area / math.pi)


def irregular_polygon(
    rng: np.random.Generator,
    size_um: float,
    aspect: float = 1.5,
    roughness: float = 0.18,
    n_vertices: int = 28,
) -> np.ndarray:
    """Random star-shaped outline centred at the origin, scaled so max Feret == size_um."""
    if not 3 <= n_vertices <= MAX_VERTICES:
        raise ValueError(f"n_vertices must be in [3, {MAX_VERTICES}]")
    theta = np.sort(rng.uniform(0, 2 * math.pi, n_vertices))
    # jitter the angles but keep them ordered so the outline never self-intersects
    radius = np.ones(n_vertices)
    for k in range(2, 6):
        amp = roughness * rng.normal() / k
        radius += amp * np.cos(k * theta + rng.uniform(0, 2 * math.pi))
    radius = np.clip(radius, 0.35, None)
    pts = np.stack([radius * np.cos(theta) * aspect, radius * np.sin(theta)], axis=1)
    rot = rng.uniform(0, math.pi)
    c, s = math.cos(rot), math.sin(rot)
    pts = pts @ np.array([[c, s], [-s, c]])
    pts -= pts.mean(axis=0)
    return pts * (size_um / max_feret(pts))
