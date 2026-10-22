from __future__ import annotations

from dataclasses import dataclass, field

from .base_parser import BaseChartParser


@dataclass
class ParserRegistry:
    _parsers: dict[str, BaseChartParser] = field(default_factory=dict)

    def register(self, parser: BaseChartParser) -> None:
        name = getattr(parser, "parser_name", parser.__class__.__name__).lower()
        self._parsers[name] = parser

    def get(self, name: str) -> BaseChartParser:
        key = name.lower()
        if key not in self._parsers:
            raise KeyError(f"Unknown parser: {name}")
        return self._parsers[key]

    def names(self) -> tuple[str, ...]:
        return tuple(sorted(self._parsers))

    def has(self, name: str) -> bool:
        return name.lower() in self._parsers


_REGISTRY = ParserRegistry()


def get_parser_registry() -> ParserRegistry:
    return _REGISTRY

