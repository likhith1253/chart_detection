"""Typed simulation configuration loaded from YAML with dotted-key overrides."""

from __future__ import annotations

import copy
import hashlib
import json
import math
from dataclasses import asdict, dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass
class OpticsConfig:
    wavelength_nm: float = 550.0
    objective_na: float = 0.25  # collection numerical aperture (10x objective)
    condenser_na: float = 0.12  # illumination cone, condenser partly stopped down
    n_liquid: float = 1.47  # mineral hydraulic oil
    n_bubble: float = 1.00  # air
    focus_depth_um: float | None = None  # focal plane height in the film; None = mid-film
    rays_per_pixel: int = 32
    max_ray_depth: int = 8  # reflection/refraction events followed per primary ray
    min_ray_weight: float = 1e-3


@dataclass
class CellConfig:
    film_thickness_um: float = 200.0
    reference_square_mm: float = 10.0  # outer edge length of the printed square
    square_line_width_um: float = 60.0
    square_transmission: float = 0.03  # chrome print on the background glass
    square_rotation_deg_max: float = 2.0
    square_offset_um_max: float = 150.0


@dataclass
class ImageConfig:
    pixel_size_um: float = 2.0  # object-space sampling
    width_px: int = 5600
    height_px: int = 5600
    bit_depth: int = 8  # 8 or 16


@dataclass
class SensorConfig:
    full_well_e: float = 12_000.0
    read_noise_e: float = 6.0
    background_level: float = 0.72  # mean illumination as fraction of full scale
    vignetting: float = 0.10  # relative fall-off at the image corner
    illumination_gradient: float = 0.04  # linear tilt across the field
    exposure_ms: float = 0.5


@dataclass
class PopulationConfig:
    target_iso_code: str | None = "19/17/14"
    # explicit counts per mL for >4, >6, >14 um; overrides target_iso_code when set
    counts_per_ml: list[float] | None = None
    # sub-4 um particles exist in real oil; without them measurement scatter only
    # ever pushes borderline particles below the 4 um threshold, never above it
    particle_size_min_um: float = 2.0
    particle_size_max_um: float = 150.0
    particle_aspect_range: list[float] = field(default_factory=lambda: [1.0, 2.2])
    particle_roughness: float = 0.18
    particle_transmission_range: list[float] = field(default_factory=lambda: [0.02, 0.18])
    bubble_count: int = 30
    bubble_diameter_range_um: list[float] = field(default_factory=lambda: [12.0, 160.0])
    bubbles_rise_to_top: bool = False


@dataclass
class FlowConfig:
    max_velocity_um_s: float = 2_000.0  # centre-line velocity, plane Poiseuille profile
    direction_deg: float = 0.0
    frame_interval_ms: float = 40.0


@dataclass
class SimConfig:
    seed: int = 0
    optics: OpticsConfig = field(default_factory=OpticsConfig)
    cell: CellConfig = field(default_factory=CellConfig)
    image: ImageConfig = field(default_factory=ImageConfig)
    sensor: SensorConfig = field(default_factory=SensorConfig)
    population: PopulationConfig = field(default_factory=PopulationConfig)
    flow: FlowConfig = field(default_factory=FlowConfig)

    # ---- derived quantities -------------------------------------------------
    @property
    def focus_depth_um(self) -> float:
        f = self.optics.focus_depth_um
        return self.cell.film_thickness_um / 2 if f is None else f

    @property
    def fov_um(self) -> tuple[float, float]:
        p = self.image.pixel_size_um
        return self.image.width_px * p, self.image.height_px * p

    @property
    def square_interior_volume_ml(self) -> float:
        inner_mm = self.cell.reference_square_mm - 2 * self.cell.square_line_width_um / 1000
        area_cm2 = (inner_mm / 10) ** 2
        return area_cm2 * self.cell.film_thickness_um * 1e-4

    @property
    def critical_angle_deg(self) -> float:
        return math.degrees(math.asin(self.optics.n_bubble / self.optics.n_liquid))

    def validate(self) -> None:
        o, c, im = self.optics, self.cell, self.image
        if not 0 < o.condenser_na < o.n_liquid:
            raise ValueError("condenser_na must be in (0, n_liquid)")
        if not 0 < o.objective_na < 1:
            raise ValueError("objective_na must be in (0, 1)")
        if o.n_bubble >= o.n_liquid:
            raise ValueError("edge glow needs n_bubble < n_liquid")
        if im.bit_depth not in (8, 16):
            raise ValueError("bit_depth must be 8 or 16")
        if o.rays_per_pixel < 1:
            raise ValueError("rays_per_pixel must be >= 1")
        lo, hi = self.population.bubble_diameter_range_um
        if not 0 < self.population.particle_size_min_um <= 4.0:
            raise ValueError("particle_size_min_um must be in (0, 4]")
        if not 0 < lo <= hi < c.film_thickness_um:
            raise ValueError("bubble diameters must be positive and smaller than the film thickness")
        half_extent = c.reference_square_mm * 1000 / 2
        rot = math.radians(c.square_rotation_deg_max)
        needed = 2 * (half_extent * (math.cos(rot) + math.sin(rot)) + c.square_offset_um_max)
        fw, fh = self.fov_um
        if min(fw, fh) < needed:
            raise ValueError(
                f"field of view {fw:.0f}x{fh:.0f} um cannot contain the {c.reference_square_mm} mm "
                f"reference square (needs >= {needed:.0f} um); raise width/height or pixel size"
            )
        if not 0 <= self.focus_depth_um <= c.film_thickness_um:
            raise ValueError("focus_depth_um must lie inside the film")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def fingerprint(self) -> str:
        blob = json.dumps(self.to_dict(), sort_keys=True).encode()
        return hashlib.sha256(blob).hexdigest()[:12]


def _merge(dc: Any, data: dict[str, Any], path: str = "") -> None:
    names = {f.name for f in fields(dc)}
    for key, value in data.items():
        if key not in names:
            raise KeyError(f"unknown config key '{path}{key}'")
        current = getattr(dc, key)
        if is_dataclass(current):
            if not isinstance(value, dict):
                raise TypeError(f"'{path}{key}' must be a mapping")
            _merge(current, value, f"{path}{key}.")
        else:
            setattr(dc, key, value)


def _parse_override(item: str) -> tuple[list[str], Any]:
    if "=" not in item:
        raise ValueError(f"override must look like key.sub=value, got {item!r}")
    key, raw = item.split("=", 1)
    return key.strip().split("."), yaml.safe_load(raw)


def load_config(path: str | Path | None = None, overrides: list[str] | None = None) -> SimConfig:
    """Load a YAML config on top of the defaults, then apply ``a.b=value`` overrides."""
    cfg = SimConfig()
    if path is not None:
        with open(path, encoding="utf-8") as fh:
            data = yaml.safe_load(fh) or {}
        _merge(cfg, data)
    for item in overrides or []:
        keys, value = _parse_override(item)
        nested: dict[str, Any] = {keys[-1]: value}
        for k in reversed(keys[:-1]):
            nested = {k: nested}
        _merge(cfg, nested)
    cfg.validate()
    return cfg


def clone(cfg: SimConfig) -> SimConfig:
    return copy.deepcopy(cfg)
