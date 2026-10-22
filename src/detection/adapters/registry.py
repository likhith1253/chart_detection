from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .base_adapter import DatasetAdapter


@dataclass
class AdapterRegistry:
    _adapters: dict[str, DatasetAdapter] = field(default_factory=dict)

    def register(self, name: str, adapter: DatasetAdapter) -> None:
        self._adapters[name.lower()] = adapter

    def get(self, name: str) -> DatasetAdapter:
        key = name.lower()
        if key not in self._adapters:
            raise KeyError(f"Unknown detection dataset adapter: {name}")
        return self._adapters[key]

    def known_formats(self) -> tuple[str, ...]:
        return tuple(sorted(self._adapters.keys()))


_REGISTRY = AdapterRegistry()


def get_adapter_registry() -> AdapterRegistry:
    return _REGISTRY

