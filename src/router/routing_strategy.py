from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any


class RoutingStrategyType(str, Enum):
    HIGHEST_CONFIDENCE = "highest_confidence"
    RULE_BASED = "rule_based"
    MULTI_PARSER_VOTING = "multi_parser_voting"
    ORACLE = "oracle"


@dataclass(frozen=True)
class RoutingStrategy:
    strategy_type: RoutingStrategyType = RoutingStrategyType.HIGHEST_CONFIDENCE
    confidence_threshold: float = 0.5
    candidate_limit: int = 3

    def select(self, classifier_name: str | None, confidence_scores: dict[str, float], available_parsers: list[str], metadata: dict[str, Any] | None = None) -> tuple[list[str], str]:
        metadata = metadata or {}
        if not available_parsers:
            return [], "no_registered_parsers"
        if self.strategy_type == RoutingStrategyType.ORACLE:
            target = str(metadata.get("oracle_parser") or classifier_name or available_parsers[0]).lower()
            return [target] if target in available_parsers else [available_parsers[0]], "oracle_routing"
        if self.strategy_type == RoutingStrategyType.RULE_BASED and classifier_name:
            target = classifier_name.lower()
            if target in available_parsers:
                return [target], "rule_based_classifier_match"
        ranked = sorted(((name, confidence_scores.get(name, 0.0)) for name in available_parsers), key=lambda x: x[1], reverse=True)
        candidates = [name for name, score in ranked[: self.candidate_limit] if score > 0.0]
        if not candidates:
            candidates = [ranked[0][0]]
        return candidates, "highest_confidence_ranking"

