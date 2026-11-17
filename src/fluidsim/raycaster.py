"""Numba ray caster for a liquid film between two glass plates.

Coordinate system (micrometres): x to the right, y down (image convention),
z along the optical axis, z = 0 at the top surface of the background
(lower) glass plate, z = h at the lower surface of the cover glass. Light
from the condenser travels in +z.

Each output pixel is sampled with ``spp`` rays. A sample picks a point on the
focal plane inside the pixel and a direction inside the illumination cone; the
straight line through that point is the ray an empty cell would deliver to
that pixel. The ray is then traced through the scene:

* opaque particles are thin flakes at a given depth: the ray weight is
  multiplied by the flake transmission when it crosses the flake;
* air bubbles are spheres with refractive index ``n_bubble``; at every surface
  hit the ray splits into a reflected and a refracted branch weighted by the
  unpolarised Fresnel coefficients. Rays striking the bubble from the oil
  beyond the critical angle are totally internally reflected - near the rim
  they are deflected only slightly and stay inside the objective aperture,
  which is what produces the bright edge glow; at intermediate impact
  parameters they leave the aperture, producing the dark ring.

A ray leaving through the cover glass is collected if its angle fits inside
the objective NA. It is imaged where its (straight) path in the liquid crosses
the focal plane, and its weight is splatted into that pixel. A pixel with no
objects therefore receives exactly 1.0 - the output is a transmission map.
"""

from __future__ import annotations

import math

import numba as nb
import numpy as np

EPS = 1e-6


@nb.njit(cache=True, inline="always")
def _xorshift(state: np.ndarray) -> float:
    x = state[0]
    x ^= (x << np.uint64(13)) & np.uint64(0xFFFFFFFFFFFFFFFF)
    x ^= x >> np.uint64(7)
    x ^= (x << np.uint64(17)) & np.uint64(0xFFFFFFFFFFFFFFFF)
    state[0] = x
    return (x >> np.uint64(11)) * (1.0 / 9007199254740992.0)


@nb.njit(cache=True)
def fresnel_unpolarised(cos_i: float, n1: float, n2: float) -> tuple[float, float]:
    """Return (reflectance, cos_t). cos_t < 0 signals total internal reflection."""
    eta = n1 / n2
    sin_t2 = eta * eta * (1.0 - cos_i * cos_i)
    if sin_t2 >= 1.0:
        return 1.0, -1.0
    cos_t = math.sqrt(1.0 - sin_t2)
    rs = (n1 * cos_i - n2 * cos_t) / (n1 * cos_i + n2 * cos_t)
    rp = (n1 * cos_t - n2 * cos_i) / (n1 * cos_t + n2 * cos_i)
    return 0.5 * (rs * rs + rp * rp), cos_t


@nb.njit(cache=True)
def _point_in_poly(px: float, py: float, verts: np.ndarray, n: int) -> bool:
    inside = False
    j = n - 1
    for i in range(n):
        xi, yi = verts[i, 0], verts[i, 1]
        xj, yj = verts[j, 0], verts[j, 1]
        if (yi > py) != (yj > py):
            xc = xi + (py - yi) * (xj - xi) / (yj - yi)
            if px < xc:
                inside = not inside
        j = i
    return inside


@nb.njit(cache=True)
def _sphere_hit(ox, oy, oz, dx, dy, dz, cx, cy, cz, r, inside):
    lx, ly, lz = ox - cx, oy - cy, oz - cz
    b = lx * dx + ly * dy + lz * dz
    c = lx * lx + ly * ly + lz * lz - r * r
    disc = b * b - c
    if disc < 0.0:
        return -1.0
    sq = math.sqrt(disc)
    if inside:
        t = -b + sq
        return t if t > EPS else -1.0
    t = -b - sq
    if t > EPS:
        return t
    return -1.0


@nb.njit(cache=True)
def _trace_sample(
    xf, yf, dx, dy, dz, z_focus, h, n_liq, n_bub, sin_obj_liq,
    spheres, poly_bbox, poly_z, poly_t, poly_v, poly_n,
    max_depth, min_w, stack,
    out, x0, y0, pix, w0,
):
    """Trace one primary sample; splat collected energy into ``out``."""
    ny, nx = out.shape
    # start on the bottom plate, on the line through (xf, yf, z_focus)
    ox = xf - dx / dz * z_focus
    oy = yf - dy / dz * z_focus
    oz = 0.0
    sp = 0
    stack[0, 0] = ox; stack[0, 1] = oy; stack[0, 2] = oz
    stack[0, 3] = dx; stack[0, 4] = dy; stack[0, 5] = dz
    stack[0, 6] = w0; stack[0, 7] = -1.0; stack[0, 8] = 0.0
    sp = 1
    n_sph = spheres.shape[0]
    n_poly = poly_z.shape[0]
    while sp > 0:
        sp -= 1
        ox = stack[sp, 0]; oy = stack[sp, 1]; oz = stack[sp, 2]
        dx = stack[sp, 3]; dy = stack[sp, 4]; dz = stack[sp, 5]
        w = stack[sp, 6]; medium = int(stack[sp, 7]); depth = int(stack[sp, 8])
        while True:
            t_best = 1e30
            kind = 0  # 1 top exit, 2 bottom loss, 3 flake, 4 sphere
            idx = -1
            if medium < 0:
                if dz > 0.0:
                    t = (h - oz) / dz
                    if t < t_best:
                        t_best = t; kind = 1
                elif dz < 0.0:
                    t = -oz / dz
                    if t < t_best:
                        t_best = t; kind = 2
                for k in range(n_sph):
                    t = _sphere_hit(ox, oy, oz, dx, dy, dz, spheres[k, 0], spheres[k, 1],
                                    spheres[k, 2], spheres[k, 3], False)
                    if t > 0.0 and t < t_best:
                        t_best = t; kind = 4; idx = k
                if dz != 0.0:
                    for k in range(n_poly):
                        t = (poly_z[k] - oz) / dz
                        if t > EPS and t < t_best:
                            px = ox + t * dx
                            py = oy + t * dy
                            if (poly_bbox[k, 0] <= px <= poly_bbox[k, 2]
                                    and poly_bbox[k, 1] <= py <= poly_bbox[k, 3]):
                                if _point_in_poly(px, py, poly_v[k], poly_n[k]):
                                    t_best = t; kind = 3; idx = k
            else:
                t = _sphere_hit(ox, oy, oz, dx, dy, dz, spheres[medium, 0], spheres[medium, 1],
                                spheres[medium, 2], spheres[medium, 3], True)
                if t < 0.0:
                    break  # numerical miss - drop the ray
                t_best = t; kind = 4; idx = medium

            if kind == 0 or kind == 2:
                break
            if kind == 1:
                sin_t = math.sqrt(max(0.0, 1.0 - dz * dz))
                if sin_t <= sin_obj_liq:
                    s = (z_focus - oz) / dz
                    lx = ox + s * dx
                    ly = oy + s * dy
                    j = int(math.floor((lx - x0) / pix))
                    i = int(math.floor((ly - y0) / pix))
                    if 0 <= i < ny and 0 <= j < nx:
                        out[i, j] += w
                break
            hx = ox + t_best * dx
            hy = oy + t_best * dy
            hz = oz + t_best * dz
            if kind == 3:
                w *= poly_t[idx]
                if w < min_w:
                    break
                ox, oy, oz = hx, hy, hz
                continue
            # sphere interface
            r = spheres[idx, 3]
            nx_ = (hx - spheres[idx, 0]) / r
            ny_ = (hy - spheres[idx, 1]) / r
            nz_ = (hz - spheres[idx, 2]) / r
            if medium < 0:
                n1 = n_liq; n2 = n_bub
            else:
                n1 = n_bub; n2 = n_liq
                nx_ = -nx_; ny_ = -ny_; nz_ = -nz_
            cos_i = -(dx * nx_ + dy * ny_ + dz * nz_)
            if cos_i < 0.0:
                cos_i = 0.0
            refl, cos_t = fresnel_unpolarised(cos_i, n1, n2)
            rdx = dx + 2.0 * cos_i * nx_
            rdy = dy + 2.0 * cos_i * ny_
            rdz = dz + 2.0 * cos_i * nz_
            depth += 1
            if cos_t >= 0.0 and depth < max_depth:
                wt = w * (1.0 - refl)
                if wt >= min_w and sp < stack.shape[0]:
                    eta = n1 / n2
                    k_ = eta * cos_i - cos_t
                    stack[sp, 0] = hx; stack[sp, 1] = hy; stack[sp, 2] = hz
                    stack[sp, 3] = eta * dx + k_ * nx_
                    stack[sp, 4] = eta * dy + k_ * ny_
                    stack[sp, 5] = eta * dz + k_ * nz_
                    stack[sp, 6] = wt
                    stack[sp, 7] = -1.0 if medium >= 0 else float(idx)
                    stack[sp, 8] = float(depth)
                    sp += 1
            w *= refl
            if w < min_w or depth >= max_depth:
                break
            ox, oy, oz = hx, hy, hz
            dx, dy, dz = rdx, rdy, rdz
            # reflected branch stays in the same medium


@nb.njit(cache=True, parallel=True)
def render_patch(
    x0: float, y0: float, nx: int, ny: int, pix: float, spp: int,
    z_focus: float, h: float, n_liq: float, n_bub: float,
    sin_cond_liq: float, sin_obj_liq: float,
    spheres: np.ndarray, poly_bbox: np.ndarray, poly_z: np.ndarray,
    poly_t: np.ndarray, poly_v: np.ndarray, poly_n: np.ndarray,
    max_depth: int, min_w: float, seed: int,
) -> np.ndarray:
    """Transmission map of a rectangular patch whose top-left corner is (x0, y0) um."""
    n_chunks = nb.get_num_threads()
    buf = np.zeros((n_chunks, ny, nx))
    w0 = 1.0 / spp
    side = int(math.ceil(math.sqrt(spp)))
    for c in nb.prange(n_chunks):
        state = np.empty(1, dtype=np.uint64)
        state[0] = np.uint64(seed) * np.uint64(6364136223846793005) + np.uint64(c * 7919 + 1442695040888963407)
        if state[0] == 0:
            state[0] = np.uint64(88172645463325252)
        stack = np.empty((64, 9))
        out = buf[c]
        for i in range(c, ny, n_chunks):
            for j in range(nx):
                for s in range(spp):
                    # stratified jitter inside the pixel
                    su = (s % side + _xorshift(state)) / side
                    sv = ((s // side) % side + _xorshift(state)) / side
                    xf = x0 + (j + su) * pix
                    yf = y0 + (i + sv) * pix
                    # uniform sampling of the illumination disk (in sin-space)
                    rr = sin_cond_liq * math.sqrt(_xorshift(state))
                    ph = 2.0 * math.pi * _xorshift(state)
                    dx = rr * math.cos(ph)
                    dy = rr * math.sin(ph)
                    dz = math.sqrt(1.0 - rr * rr)
                    _trace_sample(xf, yf, dx, dy, dz, z_focus, h, n_liq, n_bub, sin_obj_liq,
                                  spheres, poly_bbox, poly_z, poly_t, poly_v, poly_n,
                                  max_depth, min_w, stack, out, x0, y0, pix, w0)
    return buf.sum(axis=0)
