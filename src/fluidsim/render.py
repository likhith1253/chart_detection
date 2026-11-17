"""Image formation: illumination, reference square, ray-traced objects, PSF and sensor."""

from __future__ import annotations

import math
from dataclasses import dataclass

import cv2
import numpy as np
from scipy.special import erf

from fluidsim.config import SimConfig
from fluidsim.geometry import MAX_VERTICES
from fluidsim.raycaster import render_patch
from fluidsim.scene import Bubble, Particle, Scene

ROW_CHUNK = 512


@dataclass
class RenderResult:
    image: np.ndarray  # uint8 or uint16 sensor output
    irradiance: np.ndarray  # float32, noise-free, fraction of full scale
    transmission: np.ndarray  # float32, objects + square only (no illumination)


def _cone_sines(cfg: SimConfig) -> tuple[float, float]:
    n = cfg.optics.n_liquid
    return cfg.optics.condenser_na / n, cfg.optics.objective_na / n


def defocus_sigma_um(cfg: SimConfig, dz_um: float) -> float:
    """Gaussian sigma approximating the blur disc of a point dz away from focus."""
    s, _ = _cone_sines(cfg)
    radius = abs(dz_um) * s / math.sqrt(1 - s * s)
    return radius / 2.0


def illumination_field(cfg: SimConfig, rng: np.random.Generator) -> np.ndarray:
    """Smooth Koehler illumination with vignetting and a linear tilt."""
    s = cfg.sensor
    h, w = cfg.image.height_px, cfg.image.width_px
    small = 128
    yy, xx = np.mgrid[0:small, 0:small].astype(np.float64)
    u = (xx + 0.5) / small * 2 - 1
    v = (yy + 0.5) / small * 2 - 1
    r2 = (u * u + v * v) / 2
    angle = rng.uniform(0, 2 * math.pi)
    tilt = s.illumination_gradient * (u * math.cos(angle) + v * math.sin(angle)) / 2
    field = s.background_level * (1 - s.vignetting * r2) * (1 + tilt)
    return cv2.resize(field.astype(np.float32), (w, h), interpolation=cv2.INTER_CUBIC)


def square_transmission(cfg: SimConfig, scene: Scene) -> np.ndarray:
    """Analytic, defocused rendering of the printed square outline on the bottom glass."""
    cell, im = cfg.cell, cfg.image
    p = im.pixel_size_um
    sigma = math.hypot(defocus_sigma_um(cfg, cfg.focus_depth_um), p / math.sqrt(12))
    half = scene.square.side_um / 2
    inner = half - scene.square.line_width_um
    out = np.ones((im.height_px, im.width_px), dtype=np.float32)
    xs = (np.arange(im.width_px) + 0.5) * p
    k = 1.0 / (math.sqrt(2) * sigma)
    for r0 in range(0, im.height_px, ROW_CHUNK):
        ys = (np.arange(r0, min(r0 + ROW_CHUNK, im.height_px)) + 0.5) * p
        X, Y = np.meshgrid(xs, ys)
        u, v = scene.square.to_local(X, Y)
        cheb = np.maximum(np.abs(u), np.abs(v))
        # coverage of the band  inner < cheb < half, blurred by the defocus kernel
        cov = 0.5 * (erf((half - cheb) * k) - erf((inner - cheb) * k))
        out[r0 : r0 + ys.size] = 1.0 - (1.0 - cell.square_transmission) * cov
    return out


def _object_extent_um(cfg: SimConfig, obj) -> float:
    blur = 3 * defocus_sigma_um(cfg, obj.z_um - cfg.focus_depth_um)
    return obj.radius_um() * 1.6 + blur + 4 * cfg.image.pixel_size_um


def _clusters(cfg: SimConfig, objs: list) -> list[tuple[list, tuple[float, float, float, float]]]:
    """Group objects whose padded bounding boxes overlap so each group renders as one patch."""
    boxes = []
    for o in objs:
        e = _object_extent_um(cfg, o)
        boxes.append([o.x_um - e, o.y_um - e, o.x_um + e, o.y_um + e])
    parent = list(range(len(objs)))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    order = sorted(range(len(objs)), key=lambda i: boxes[i][0])
    active: list[int] = []
    for i in order:
        active = [j for j in active if boxes[j][2] >= boxes[i][0]]
        for j in active:
            if boxes[j][1] <= boxes[i][3] and boxes[i][1] <= boxes[j][3]:
                parent[find(i)] = find(j)
        active.append(i)
    groups: dict[int, list[int]] = {}
    for i in range(len(objs)):
        groups.setdefault(find(i), []).append(i)
    result = []
    for members in groups.values():
        b = np.array([boxes[i] for i in members])
        result.append(([objs[i] for i in members], (b[:, 0].min(), b[:, 1].min(), b[:, 2].max(), b[:, 3].max())))
    # merged boxes can overlap again after union; merge until stable
    changed = True
    while changed:
        changed = False
        for a in range(len(result)):
            for c in range(a + 1, len(result)):
                ba, bc = result[a][1], result[c][1]
                if ba[0] <= bc[2] and bc[0] <= ba[2] and ba[1] <= bc[3] and bc[1] <= ba[3]:
                    merged = (min(ba[0], bc[0]), min(ba[1], bc[1]), max(ba[2], bc[2]), max(ba[3], bc[3]))
                    result[a] = (result[a][0] + result[c][0], merged)
                    del result[c]
                    changed = True
                    break
            if changed:
                break
    return result


def _pack(particles: list[Particle], bubbles: list[Bubble]):
    spheres = np.array([[b.x_um, b.y_um, b.z_um, b.diameter_um / 2] for b in bubbles], dtype=np.float64).reshape(-1, 4)
    k = len(particles)
    poly_v = np.zeros((k, MAX_VERTICES, 2))
    poly_n = np.zeros(k, dtype=np.int64)
    poly_bbox = np.zeros((k, 4))
    poly_z = np.zeros(k)
    poly_t = np.zeros(k)
    for i, p in enumerate(particles):
        v = p.vertices_abs()
        poly_v[i, : len(v)] = v
        poly_n[i] = len(v)
        poly_bbox[i] = [v[:, 0].min(), v[:, 1].min(), v[:, 0].max(), v[:, 1].max()]
        poly_z[i] = p.z_um
        poly_t[i] = p.transmission
    return spheres, poly_bbox, poly_z, poly_t, poly_v, poly_n


def _motion_blur(patch: np.ndarray, cfg: SimConfig, scene: Scene, objs) -> np.ndarray:
    z = float(np.mean([o.z_um for o in objs]))
    vx, vy = scene.flow_velocity(z)
    length_px = math.hypot(vx, vy) * cfg.sensor.exposure_ms * 1e-3 / cfg.image.pixel_size_um
    if length_px < 0.5:
        return patch
    n = int(math.ceil(length_px)) * 2 + 1
    kernel = np.zeros((n, n), np.float32)
    c = n // 2
    ang = math.atan2(vy, vx)
    for t in np.linspace(-length_px / 2, length_px / 2, 4 * n):
        kernel[int(round(c + t * math.sin(ang))), int(round(c + t * math.cos(ang)))] += 1
    kernel /= kernel.sum()
    return 1.0 + cv2.filter2D(patch - 1.0, -1, kernel, borderType=cv2.BORDER_CONSTANT)


def object_transmission(cfg: SimConfig, scene: Scene, seed: int) -> np.ndarray:
    """Ray-traced transmission map of all particles and bubbles (1.0 = empty liquid)."""
    im, o = cfg.image, cfg.optics
    p = im.pixel_size_um
    out = np.ones((im.height_px, im.width_px), dtype=np.float32)
    sin_cond, sin_obj = _cone_sines(cfg)
    for n, (objs, (bx0, by0, bx1, by1)) in enumerate(_clusters(cfg, scene.objects())):
        j0 = max(0, int(math.floor(bx0 / p)))
        i0 = max(0, int(math.floor(by0 / p)))
        j1 = min(im.width_px, int(math.ceil(bx1 / p)))
        i1 = min(im.height_px, int(math.ceil(by1 / p)))
        if j1 <= j0 or i1 <= i0:
            continue
        parts = [x for x in objs if isinstance(x, Particle)]
        bubs = [x for x in objs if isinstance(x, Bubble)]
        packed = _pack(parts, bubs)
        patch = render_patch(
            j0 * p, i0 * p, j1 - j0, i1 - i0, p, o.rays_per_pixel,
            cfg.focus_depth_um, cfg.cell.film_thickness_um, o.n_liquid, o.n_bubble,
            sin_cond, sin_obj, *packed, o.max_ray_depth, o.min_ray_weight, seed * 100_003 + n,
        ).astype(np.float32)
        patch = _motion_blur(patch, cfg, scene, objs)
        out[i0:i1, j0:j1] = patch
    return out


def psf_sigma_px(cfg: SimConfig) -> float:
    """Gaussian approximation of the Airy disc: sigma = 0.21 lambda / NA."""
    lam_um = cfg.optics.wavelength_nm / 1000
    return 0.21 * lam_um / cfg.optics.objective_na / cfg.image.pixel_size_um


def apply_sensor(cfg: SimConfig, irradiance: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    s = cfg.sensor
    electrons = rng.poisson(np.clip(irradiance, 0, None) * s.full_well_e).astype(np.float32)
    electrons += rng.normal(0, s.read_noise_e, size=electrons.shape).astype(np.float32)
    max_dn = 2**cfg.image.bit_depth - 1
    dn = np.clip(np.rint(electrons / s.full_well_e * max_dn), 0, max_dn)
    return dn.astype(np.uint8 if cfg.image.bit_depth == 8 else np.uint16)


def render_scene(scene: Scene, seed: int) -> RenderResult:
    cfg = scene.config
    rng = np.random.default_rng(seed)
    trans = square_transmission(cfg, scene) * object_transmission(cfg, scene, seed)
    irr = illumination_field(cfg, rng) * trans
    sigma = psf_sigma_px(cfg)
    if sigma > 0.15:
        irr = cv2.GaussianBlur(irr, (0, 0), sigmaX=sigma, sigmaY=sigma)
    return RenderResult(image=apply_sensor(cfg, irr, rng), irradiance=irr, transmission=trans)
