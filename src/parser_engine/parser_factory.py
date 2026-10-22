from __future__ import annotations

from dataclasses import dataclass, field

from src.router.base_parser import BaseChartParser
from src.router.registry import ParserRegistry, get_parser_registry

from .exceptions import ParserInitializationError


@dataclass
class ParserFactory:
    registry: ParserRegistry = field(default_factory=get_parser_registry)

    def create(self, parser_name: str) -> BaseChartParser:
        try:
            return self.registry.get(parser_name)
        except KeyError as exc:
            raise ParserInitializationError(f"Parser is not registered: {parser_name}") from exc
