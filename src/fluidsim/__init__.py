"""fluidsim - physically based synthetic image generator for ISO 4406 particle counting.

Simulates a thin film of liquid sandwiched between two glass plates, containing
opaque solid particles and spherical air bubbles, imaged in transmitted
(bright-field) light. Bubbles are ray traced with Fresnel reflection/refraction
so that the characteristic edge glow (total internal reflection at the
liquid -> air interface) emerges from the physics instead of being painted on.
Every image comes with an exact ground-truth record.
"""

from fluidsim.config import SimConfig, load_config
from fluidsim.iso4406 import iso_code, iso_code_string
from fluidsim.scene import Scene, build_scene
from fluidsim.render import render_scene

__all__ = [
    "SimConfig",
    "load_config",
    "iso_code",
    "iso_code_string",
    "Scene",
    "build_scene",
    "render_scene",
]

__version__ = "1.0.0"
