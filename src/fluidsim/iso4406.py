"""ISO 4406:2021 cleanliness code table.

The code for each size channel (>4 um(c), >6 um(c), >14 um(c)) is the scale
number whose range contains the particle count per millilitre. Ranges are
"more than lower bound, up to and including upper bound".
"""

from __future__ import annotations

import math
from typing import Sequence

# (scale number, more than, up to and including) - counts per mL
ISO4406_TABLE: tuple[tuple[int, float, float], ...] = (
    (0, 0.0, 0.01),
    (1, 0.01, 0.02),
    (2, 0.02, 0.04),
    (3, 0.04, 0.08),
    (4, 0.08, 0.16),
    (5, 0.16, 0.32),
    (6, 0.32, 0.64),
    (7, 0.64, 1.3),
    (8, 1.3, 2.5),
    (9, 2.5, 5.0),
    (10, 5.0, 10.0),
    (11, 10.0, 20.0),
    (12, 20.0, 40.0),
    (13, 40.0, 80.0),
    (14, 80.0, 160.0),
    (15, 160.0, 320.0),
    (16, 320.0, 640.0),
    (17, 640.0, 1_300.0),
    (18, 1_300.0, 2_500.0),
    (19, 2_500.0, 5_000.0),
    (20, 5_000.0, 10_000.0),
    (21, 10_000.0, 20_000.0),
    (22, 20_000.0, 40_000.0),
    (23, 40_000.0, 80_000.0),
    (24, 80_000.0, 160_000.0),
    (25, 160_000.0, 320_000.0),
    (26, 320_000.0, 640_000.0),
    (27, 640_000.0, 1_300_000.0),
    (28, 1_300_000.0, 2_500_000.0),
)

CHANNELS_UM: tuple[float, float, float] = (4.0, 6.0, 14.0)


def iso_code(count_per_ml: float) -> int:
    """Scale number for a single channel. Counts above the table return 29 (">28")."""
    if count_per_ml < 0 or math.isnan(count_per_ml):
        raise ValueError(f"count per mL must be non-negative, got {count_per_ml}")
    for code, _lo, hi in ISO4406_TABLE:
        if count_per_ml <= hi:
            return code
    return 29


def code_range(code: int) -> tuple[float, float]:
    """(more than, up to and including) counts per mL for a scale number."""
    for c, lo, hi in ISO4406_TABLE:
        if c == code:
            return lo, hi
    raise ValueError(f"ISO 4406 scale number must be 0..28, got {code}")


def code_midpoint(code: int) -> float:
    """Geometric centre of a scale number's range (counts per mL)."""
    lo, hi = code_range(code)
    if lo == 0:
        return hi / 2
    return math.sqrt(lo * hi)


def iso_code_string(counts_per_ml: Sequence[float]) -> str:
    """Format the three-channel code, e.g. ``19/17/14``."""
    if len(counts_per_ml) != 3:
        raise ValueError("expected three counts: >4, >6 and >14 um")
    parts = []
    for c in counts_per_ml:
        code = iso_code(c)
        parts.append(">28" if code == 29 else str(code))
    return "/".join(parts)


def parse_code_string(code: str) -> tuple[int, int, int]:
    """Parse ``"19/17/14"`` into ``(19, 17, 14)``."""
    parts = [int(p) for p in code.strip().split("/")]
    if len(parts) != 3:
        raise ValueError(f"ISO 4406 code needs three numbers, got {code!r}")
    if not parts[0] >= parts[1] >= parts[2]:
        raise ValueError(f"ISO 4406 codes must be non-increasing with size, got {code!r}")
    for p in parts:
        code_range(p)
    return parts[0], parts[1], parts[2]
