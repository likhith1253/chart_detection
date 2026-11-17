import json

import cv2
import numpy as np
import pytest

from fluidsim.cli import main
from fluidsim.config import load_config
from fluidsim.groundtruth import build_record, instance_mask
from fluidsim.render import render_scene
from fluidsim.scene import build_scene

TINY = [
    "image.pixel_size_um=5.0", "image.width_px=2240", "image.height_px=2240",
    "optics.rays_per_pixel=4", "population.target_iso_code=21/19/16", "population.bubble_count=10",
]


@pytest.fixture(scope="module")
def tiny_cfg():
    return load_config(overrides=TINY)


def test_config_rejects_field_smaller_than_square():
    with pytest.raises(ValueError, match="reference square"):
        load_config(overrides=["image.width_px=1000"])


def test_unknown_key_is_an_error():
    with pytest.raises(KeyError):
        load_config(overrides=["optics.not_a_key=1"])


def test_scene_is_deterministic(tiny_cfg):
    a = build_scene(tiny_cfg, np.random.default_rng(5))
    b = build_scene(tiny_cfg, np.random.default_rng(5))
    assert [p.size_um for p in a.particles] == [p.size_um for p in b.particles]
    assert a.square.rotation_deg == b.square.rotation_deg


def test_ground_truth_consistency(tiny_cfg):
    scene = build_scene(tiny_cfg, np.random.default_rng(2))
    rec = build_record(scene, "x.png", seed=2)
    objs = rec["objects"]
    assert len(objs) == len(scene.particles) + len(scene.bubbles)
    assert {o["class"] for o in objs} == {"particle", "bubble"}
    inside = [o for o in objs if o["class"] == "particle" and o["inside_reference_square"]]
    assert rec["ground_truth_iso"]["counts"][">4um"] == sum(o["size_um"] > 4 for o in inside)
    assert rec["ground_truth_iso"]["counts"][">14um"] == sum(o["size_um"] > 14 for o in inside)
    side = np.linalg.norm(np.subtract(*rec["reference_square"]["outer_corners_px"][:2]))
    assert side == pytest.approx(10_000 / 5.0, rel=1e-6)


def test_render_and_mask(tiny_cfg):
    scene = build_scene(tiny_cfg, np.random.default_rng(4))
    res = render_scene(scene, seed=4)
    assert res.image.dtype == np.uint8 and res.image.shape == (2240, 2240)
    mask = instance_mask(scene)
    big = max(scene.bubbles, key=lambda b: b.diameter_um)
    cx, cy = int(big.x_um / 5.0), int(big.y_um / 5.0)
    if 0 <= cx < 2240 and 0 <= cy < 2240:
        assert mask[cy, cx] == big.id + 1


def test_cli_generate_and_sequence(tmp_path):
    out = tmp_path / "ds"
    assert main(["generate", "-o", str(out), "-n", "1", "--yolo", "--masks", "--previews", *sum((["--set", s] for s in TINY), [])]) == 0
    rec = json.loads((out / "labels" / "img_00000.json").read_text())
    assert rec["schema_version"] and (out / "images" / "img_00000.png").exists()
    assert cv2.imread(str(out / "masks" / "img_00000.png"), cv2.IMREAD_UNCHANGED).dtype == np.uint16
    assert (out / "annotations.csv").read_text().count("\n") == len(rec["objects"]) + 1

    seq = tmp_path / "seq"
    args = ["sequence", "-o", str(seq), "--frames", "2", *sum((["--set", s] for s in TINY), [])]
    assert main(args) == 0
    f0 = json.loads((seq / "labels" / "frame_00000.json").read_text())
    f1 = json.loads((seq / "labels" / "frame_00001.json").read_text())
    moved = [(a["x_um"], b["x_um"]) for a, b in zip(f0["objects"], f1["objects"]) if a["velocity_um_s"][0] > 100]
    assert moved and all(a != b for a, b in moved)
    assert [o["id"] for o in f0["objects"]] == [o["id"] for o in f1["objects"]]


def test_cli_iso():
    assert main(["iso", "70", "18", "2", "--volume-ml", "0.0195"]) == 0
