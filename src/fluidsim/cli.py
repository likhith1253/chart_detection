"""Command-line interface: ``fluidsim generate | sequence | info | iso``."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from fluidsim import __version__
from fluidsim.config import load_config
from fluidsim.generator import OutputOptions, generate_dataset, generate_sequence
from fluidsim.iso4406 import iso_code_string
from fluidsim.render import defocus_sigma_um, psf_sigma_px
from fluidsim.scene import target_counts_per_ml


def _common(p: argparse.ArgumentParser) -> None:
    p.add_argument("-c", "--config", type=Path, help="YAML config (defaults are used for missing keys)")
    p.add_argument("--set", dest="overrides", action="append", default=[], metavar="KEY=VALUE",
                   help="override a config value, e.g. --set image.pixel_size_um=3 (repeatable)")
    p.add_argument("--seed", type=int, help="random seed (overrides config)")


def _outputs(p: argparse.ArgumentParser) -> None:
    p.add_argument("-o", "--out", type=Path, required=True, help="output directory")
    p.add_argument("--masks", action="store_true", help="write 16-bit instance masks")
    p.add_argument("--yolo", action="store_true", help="write YOLO-format box labels (0=particle, 1=bubble)")
    p.add_argument("--previews", action="store_true", help="write annotated JPEG previews")


def _load(args):
    overrides = list(args.overrides)
    if args.seed is not None:
        overrides.append(f"seed={args.seed}")
    return load_config(args.config, overrides)


def cmd_info(args) -> int:
    cfg = _load(args)
    fw, fh = cfg.fov_um
    counts = target_counts_per_ml(cfg)
    vol = cfg.square_interior_volume_ml
    info = {
        "field_of_view_mm": [round(fw / 1000, 3), round(fh / 1000, 3)],
        "pixel_size_um": cfg.image.pixel_size_um,
        "counting_volume_ml": round(vol, 5),
        "target_counts_per_ml": [round(c, 1) for c in counts],
        "expected_particles_in_square": [round(c * vol, 1) for c in counts],
        "target_code": iso_code_string(counts) if any(counts) else None,
        "critical_angle_deg": round(cfg.critical_angle_deg, 2),
        "psf_sigma_px": round(psf_sigma_px(cfg), 3),
        "lateral_resolution_um": round(0.61 * cfg.optics.wavelength_nm / 1000 / cfg.optics.objective_na, 2),
        "defocus_sigma_at_film_edge_um": round(defocus_sigma_um(cfg, cfg.focus_depth_um), 2),
        "depth_of_field_um": round(cfg.optics.wavelength_nm / 1000 * cfg.optics.n_liquid / cfg.optics.objective_na**2, 1),
        "config_fingerprint": cfg.fingerprint(),
    }
    print(json.dumps(info, indent=2))
    return 0


def cmd_generate(args) -> int:
    cfg = _load(args)
    opts = OutputOptions(masks=args.masks, yolo=args.yolo, previews=args.previews)
    generate_dataset(cfg, args.count, args.out, opts)
    print(f"done -> {args.out}")
    return 0


def cmd_sequence(args) -> int:
    cfg = _load(args)
    opts = OutputOptions(masks=args.masks, yolo=args.yolo, previews=args.previews, video=args.video)
    generate_sequence(cfg, args.frames, args.out, opts)
    print(f"done -> {args.out}")
    return 0


def cmd_iso(args) -> int:
    counts = [args.n4, args.n6, args.n14]
    if args.volume_ml:
        counts = [c / args.volume_ml for c in counts]
    print(iso_code_string(counts))
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="fluidsim", description="Optical physics image simulator for ISO 4406 particle counting")
    p.add_argument("--version", action="version", version=f"fluidsim {__version__}")
    sub = p.add_subparsers(dest="command", required=True)

    g = sub.add_parser("generate", help="generate independent static images with ground truth")
    _common(g); _outputs(g)
    g.add_argument("-n", "--count", type=int, default=1, help="number of images")
    g.set_defaults(func=cmd_generate)

    s = sub.add_parser("sequence", help="generate a time sequence of a flowing film")
    _common(s); _outputs(s)
    s.add_argument("--frames", type=int, default=25)
    s.add_argument("--video", action="store_true", help="also write a lossless AVI of the sequence")
    s.set_defaults(func=cmd_sequence)

    i = sub.add_parser("info", help="print derived optical and counting quantities for a config")
    _common(i)
    i.set_defaults(func=cmd_info)

    c = sub.add_parser("iso", help="ISO 4406 code from counts")
    c.add_argument("n4", type=float); c.add_argument("n6", type=float); c.add_argument("n14", type=float)
    c.add_argument("--volume-ml", type=float, help="treat counts as raw counts in this volume")
    c.set_defaults(func=cmd_iso)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except (ValueError, KeyError, TypeError, FileNotFoundError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
