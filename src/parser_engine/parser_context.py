from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from PIL import Image


@dataclass(frozen=True)
class ParserContext:
    image: Image.Image
    metadata: dict[str, Any] = field(default_factory=dict)
    classifier_output: dict[str, Any] = field(default_factory=dict)
    routing_information: dict[str, Any] = field(default_factory=dict)
    ocr_hooks: dict[str, Any] = field(default_factory=dict)
    axis_hooks: dict[str, Any] = field(default_factory=dict)
    legend_hooks: dict[str, Any] = field(default_factory=dict)

