from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import yaml

from .routing_strategy import RoutingStrategyType


@dataclass(frozen=True)
class RouterConfig:
    strategy: RoutingStrategyType = RoutingStrategyType.HIGHEST_CONFIDENCE
    confidence_threshold: float = 0.5
    candidate_limit: int = 3


def load_router_config(path: str | Path | None = None, overrides: Mapping[str, Any] | None = None) -> RouterConfig:
    data: dict[str, Any] = {}
    if path is not None:
        loaded = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        if not isinstance(loaded, dict):
            raise TypeError("Router config must be a mapping")
        data.update(loaded)
    data.update(dict(overrides or {}))
    return RouterConfig(
        strategy=RoutingStrategyType(str(data.get("strategy", RoutingStrategyType.HIGHEST_CONFIDENCE))),
        confidence_threshold=float(data.get("confidence_threshold", 0.5)),
        candidate_limit=int(data.get("candidate_limit", 3)),
    )

