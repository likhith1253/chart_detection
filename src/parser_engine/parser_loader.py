from __future__ import annotations

from dataclasses import dataclass, field

from src.router.registry import ParserRegistry, get_parser_registry


@dataclass
class ParserLoader:
    registry: ParserRegistry = field(default_factory=get_parser_registry)

    def available(self) -> tuple[str, ...]:
        return self.registry.names()
