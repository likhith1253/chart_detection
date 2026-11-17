import math

import numpy as np
import pytest

from fluidsim.config import load_config
from fluidsim.geometry import irregular_polygon, max_feret
from fluidsim.iso4406 import code_midpoint, iso_code, iso_code_string, parse_code_string
from fluidsim.raycaster import fresnel_unpolarised
from fluidsim.render import object_transmission
from fluidsim.scene import Bubble, Particle, ReferenceSquare, Scene, sample_particle_sizes


def small_cfg(**over):
    # unit scenes are tiny patches, so set fields directly instead of going through
    # load_config's full-field validation (which requires the whole 10 mm square)
    cfg = load_config()
    for key, value in over.items():
        section, name = key.split(".")
        setattr(getattr(cfg, section), name, value)
    cfg.image.width_px, cfg.image.height_px = 120, 120
    return cfg


@pytest.mark.parametrize("count,code", [(0, 0), (0.01, 0), (0.011, 1), (1300, 17), (1300.5, 18), (2500, 18), (5000, 19), (3e6, 29)])
def test_iso_table_boundaries(count, code):
    assert iso_code(count) == code


def test_iso_string_and_midpoints():
    assert iso_code_string([3535, 912, 113]) == "19/17/14"
    for c in (5, 14, 19, 23):
        assert iso_code(code_midpoint(c)) == c
    with pytest.raises(ValueError):
        parse_code_string("14/17/19")


@pytest.mark.parametrize("n1,n2", [(1.47, 1.0), (1.0, 1.47), (1.0, 1.52)])
def test_fresnel_bounds_and_normal_incidence(n1, n2):
    r0, cos_t = fresnel_unpolarised(1.0, n1, n2)
    assert r0 == pytest.approx(((n1 - n2) / (n1 + n2)) ** 2, rel=1e-9)
    assert cos_t == pytest.approx(1.0)
    for cos_i in np.linspace(0.01, 1, 25):
        r, _ = fresnel_unpolarised(cos_i, n1, n2)
        assert 0.0 <= r <= 1.0


def test_total_internal_reflection_beyond_critical_angle():
    crit = math.asin(1.0 / 1.47)
    r_in, _ = fresnel_unpolarised(math.cos(crit - 0.01), 1.47, 1.0)
    r_out, cos_t = fresnel_unpolarised(math.cos(crit + 0.01), 1.47, 1.0)
    assert r_in < 1.0
    assert r_out == 1.0 and cos_t < 0


def test_polygon_scaled_to_max_feret():
    rng = np.random.default_rng(0)
    for size in (4.0, 6.5, 14.0, 80.0):
        v = irregular_polygon(rng, size, aspect=2.0)
        assert max_feret(v) == pytest.approx(size, rel=1e-9)


def test_size_sampler_reproduces_channel_ratios():
    rng = np.random.default_rng(1)
    counts = (3535.0, 912.0, 113.0)
    s = sample_particle_sizes(rng, 200_000, counts, 150.0)
    assert s.min() >= 4.0 and s.max() <= 150.0
    assert (s > 6).mean() == pytest.approx(912 / 3535, rel=0.03)
    assert (s > 14).mean() == pytest.approx(113 / 3535, rel=0.06)


def test_empty_cell_transmits_everything():
    cfg = small_cfg()
    scene = Scene(cfg, ReferenceSquare((0, 0), 10_000, 60, 0))
    t = object_transmission(cfg, scene, seed=0)
    assert np.allclose(t, 1.0)


def _radial_profile(t, cx, cy, pix):
    yy, xx = np.mgrid[0:t.shape[0], 0:t.shape[1]]
    r = np.hypot((xx + 0.5) * pix - cx, (yy + 0.5) * pix - cy)
    return r, t


def test_bubble_has_bright_centre_dark_ring_and_edge_glow_particle_does_not():
    cfg = small_cfg(**{"optics.rays_per_pixel": 64, "image.pixel_size_um": 1.0})
    sq = ReferenceSquare((0, 0), 10_000, 60, 0)
    bubble = Bubble(0, 60.0, 60.0, 100.0, 80.0)
    t_b = object_transmission(cfg, Scene(cfg, sq, bubbles=[bubble]), seed=1)
    r, t = _radial_profile(t_b, 60.0, 60.0, 1.0)
    centre = t[r < 3].mean()
    ring = t[(r > 15) & (r < 35)].mean()
    rim = t[(r > 39) & (r < 41.5)].max()
    assert centre > 0.6, "air bubble acts as a lens - bright centre expected"
    assert ring < 0.1, "total internal reflection should darken the outer annulus"
    assert rim > 1.0, "grazing reflections should create an edge glow brighter than background"

    rng = np.random.default_rng(3)
    part = Particle(1, 60.0, 60.0, 100.0, irregular_polygon(rng, 60.0, 1.0), 0.05)
    t_p = object_transmission(cfg, Scene(cfg, sq, particles=[part]), seed=1)
    assert t_p.max() <= 1.0 + 1e-6, "opaque particles only absorb, they cannot glow"
    assert t_p[55:65, 55:65].mean() < 0.1


def test_energy_is_conserved_or_lost_never_created():
    cfg = small_cfg(**{"optics.rays_per_pixel": 32})
    sq = ReferenceSquare((0, 0), 10_000, 60, 0)
    t = object_transmission(cfg, Scene(cfg, sq, bubbles=[Bubble(0, 240.0, 240.0, 100.0, 120.0)]), seed=2)
    assert t.sum() <= t.size + 1e-6
