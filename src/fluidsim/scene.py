"""Scene construction: reference square, particle population, bubbles and flow."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from fluidsim.config import SimConfig
from fluidsim.geometry import (
    equivalent_circle_diameter,
    irregular_polygon,
    max_feret,
    polygon_area,
)
from fluidsim.iso4406 import CHANNELS_UM, code_midpoint, parse_code_string


@dataclass
class ReferenceSquare:
    center_um: tuple[float, float]
    side_um: float  # outer edge length
    line_width_um: float
    rotation_deg: float

    def corners_um(self) -> np.ndarray:
        """Outer corners, clockwise from top-left in the square's own frame."""
        a = math.radians(self.rotation_deg)
        c, s = math.cos(a), math.sin(a)
        half = self.side_um / 2
        local = np.array([[-half, -half], [half, -half], [half, half], [-half, half]])
        rot = np.array([[c, -s], [s, c]])
        return local @ rot.T + np.asarray(self.center_um)

    def to_local(self, x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        a = math.radians(self.rotation_deg)
        c, s = math.cos(a), math.sin(a)
        dx = np.asarray(x) - self.center_um[0]
        dy = np.asarray(y) - self.center_um[1]
        return c * dx + s * dy, -s * dx + c * dy

    def inside_interior(self, x, y, margin_um: float = 0.0) -> np.ndarray:
        u, v = self.to_local(x, y)
        lim = self.side_um / 2 - self.line_width_um - margin_um
        return (np.abs(u) < lim) & (np.abs(v) < lim)


@dataclass
class Particle:
    id: int
    x_um: float
    y_um: float
    z_um: float
    outline_um: np.ndarray  # (N, 2) relative to centre
    transmission: float

    @property
    def size_um(self) -> float:
        return max_feret(self.outline_um)

    @property
    def ecd_um(self) -> float:
        return equivalent_circle_diameter(polygon_area(self.outline_um))

    def vertices_abs(self) -> np.ndarray:
        return self.outline_um + np.array([self.x_um, self.y_um])

    def radius_um(self) -> float:
        return float(np.sqrt((self.outline_um**2).sum(1)).max())


@dataclass
class Bubble:
    id: int
    x_um: float
    y_um: float
    z_um: float
    diameter_um: float

    @property
    def size_um(self) -> float:
        return self.diameter_um

    @property
    def ecd_um(self) -> float:
        return self.diameter_um

    def radius_um(self) -> float:
        return self.diameter_um / 2


@dataclass
class Scene:
    config: SimConfig
    square: ReferenceSquare
    particles: list[Particle] = field(default_factory=list)
    bubbles: list[Bubble] = field(default_factory=list)
    time_s: float = 0.0

    def objects(self):
        return [*self.particles, *self.bubbles]

    def flow_velocity(self, z_um: float) -> tuple[float, float]:
        """Plane Poiseuille profile between the plates: v(z) = 4 vmax z (h - z) / h^2."""
        h = self.config.cell.film_thickness_um
        v = 4 * self.config.flow.max_velocity_um_s * z_um * (h - z_um) / (h * h)
        a = math.radians(self.config.flow.direction_deg)
        return v * math.cos(a), v * math.sin(a)

    def advance(self, dt_s: float) -> None:
        """Advect every object with the local flow velocity, wrapping around the field."""
        fw, fh = self.config.fov_um
        for obj in self.objects():
            vx, vy = self.flow_velocity(obj.z_um)
            obj.x_um = (obj.x_um + vx * dt_s) % fw
            obj.y_um = (obj.y_um + vy * dt_s) % fh
        self.time_s += dt_s


def target_counts_per_ml(cfg: SimConfig) -> tuple[float, float, float]:
    pop = cfg.population
    if pop.counts_per_ml is not None:
        n4, n6, n14 = (float(v) for v in pop.counts_per_ml)
    elif pop.target_iso_code:
        n4, n6, n14 = (code_midpoint(c) for c in parse_code_string(pop.target_iso_code))
    else:
        return 0.0, 0.0, 0.0
    if not n4 >= n6 >= n14 >= 0:
        raise ValueError("cumulative counts must satisfy N(>4) >= N(>6) >= N(>14) >= 0")
    return n4, n6, n14


def size_distribution_nodes(
    counts: tuple[float, float, float], size_min: float, size_max: float
) -> tuple[np.ndarray, np.ndarray]:
    """Nodes (d, N(>d)) of the piecewise power-law cumulative distribution through
    the three ISO channel counts. Below 4 um the 4 -> 6 um slope is extended down to
    size_min; above 14 um the 6 -> 14 um slope is extended up to size_max."""
    n4, n6, n14 = (max(c, 1e-9) for c in counts)
    lo_slope = math.log(n6 / n4) / math.log(6 / 4)
    hi_slope = math.log(n14 / n6) / math.log(14 / 6)
    d = [CHANNELS_UM[0], CHANNELS_UM[1], CHANNELS_UM[2], size_max]
    n = [n4, n6, n14, n14 * (size_max / 14) ** hi_slope]
    if size_min < CHANNELS_UM[0]:
        d.insert(0, size_min)
        n.insert(0, n4 * (size_min / 4) ** lo_slope)
    return np.array(d), np.array(n)


def sample_particle_sizes(
    rng: np.random.Generator, n: int, counts: tuple[float, float, float],
    size_max: float, size_min: float = 4.0,
) -> np.ndarray:
    """Inverse-CDF sampling of ``n`` sizes in [size_min, size_max] from the
    piecewise power-law cumulative distribution (see size_distribution_nodes)."""
    d, cum = size_distribution_nodes(counts, size_min, size_max)
    frac = cum / cum[0]  # fraction of particles larger than d
    u = rng.uniform(frac[-1], 1.0, n)  # exclude particles above size_max
    # log F(d) decreases monotonically with d, so invert it by interpolation
    return np.exp(np.interp(np.log(u), np.log(frac)[::-1], np.log(d)[::-1]))


def build_scene(cfg: SimConfig, rng: np.random.Generator) -> Scene:
    fw, fh = cfg.fov_um
    cell = cfg.cell
    pop = cfg.population
    h = cell.film_thickness_um

    square = ReferenceSquare(
        center_um=(
            fw / 2 + rng.uniform(-1, 1) * cell.square_offset_um_max,
            fh / 2 + rng.uniform(-1, 1) * cell.square_offset_um_max,
        ),
        side_um=cell.reference_square_mm * 1000,
        line_width_um=cell.square_line_width_um,
        rotation_deg=rng.uniform(-1, 1) * cell.square_rotation_deg_max,
    )
    scene = Scene(config=cfg, square=square)

    fov_volume_ml = (fw * 1e-4) * (fh * 1e-4) * (h * 1e-4)
    counts = target_counts_per_ml(cfg)
    size_min = pop.particle_size_min_um
    if counts[0] > 0:
        _, cum = size_distribution_nodes(counts, size_min, pop.particle_size_max_um)
        n_particles = int(rng.poisson(cum[0] * fov_volume_ml))
    else:
        n_particles = 0
    sizes = (
        sample_particle_sizes(rng, n_particles, counts, pop.particle_size_max_um, size_min)
        if n_particles else []
    )
    next_id = 0
    for size in sizes:
        aspect = rng.uniform(*pop.particle_aspect_range)
        outline = irregular_polygon(rng, float(size), aspect, pop.particle_roughness)
        scene.particles.append(
            Particle(
                id=next_id,
                x_um=rng.uniform(0, fw),
                y_um=rng.uniform(0, fh),
                z_um=rng.uniform(0.03 * h, 0.97 * h),
                outline_um=outline,
                transmission=rng.uniform(*pop.particle_transmission_range),
            )
        )
        next_id += 1

    lo, hi = pop.bubble_diameter_range_um
    n_bubbles = int(rng.poisson(pop.bubble_count)) if pop.bubble_count > 0 else 0
    for _ in range(n_bubbles):
        # log-uniform diameters: small bubbles are more common
        dia = float(math.exp(rng.uniform(math.log(lo), math.log(hi))))
        r = dia / 2
        z = h - r - 1.0 if pop.bubbles_rise_to_top else rng.uniform(r + 1.0, h - r - 1.0)
        scene.bubbles.append(
            Bubble(id=next_id, x_um=rng.uniform(0, fw), y_um=rng.uniform(0, fh), z_um=z, diameter_um=dia)
        )
        next_id += 1
    return scene
