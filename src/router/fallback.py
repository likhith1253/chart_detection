from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class UnknownParser:
    reason: str = "confidence_below_threshold"


@dataclass(frozen=True)
class MultipleParserExecution:
    parsers: list[str]


@dataclass(frozen=True)
class RejectPrediction:
    reason: str


@dataclass(frozen=True)
class FallbackRouter:
    confidence_threshold: float = 0.5

    def choose(self, selected_parser: str | None, confidence: float, candidates: list[str]) -> Any:
        if confidence < self.confidence_threshold:
            if len(candidates) > 1:
                return MultipleParserExecution(parsers=candidates)
            return UnknownParser()
        if not selected_parser:
            return RejectPrediction(reason="no_parser_selected")
        return selected_parser

