from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class OCRLanguageProfile:
    language: str
    scripts: tuple[str, ...] = ("Latin",)
    numeric_mode: bool = False

