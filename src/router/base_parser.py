from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from PIL import Image


class BaseChartParser(ABC):
    parser_name: str = "base"

    @abstractmethod
    def parse(self, image: Image.Image, *, metadata: dict[str, Any] | None = None) -> Any:
        raise NotImplementedError

    @abstractmethod
    def validate(self, parsed_output: Any) -> bool:
        raise NotImplementedError

    @abstractmethod
    def export(self, parsed_output: Any, output_path: str | Path) -> Path:
        raise NotImplementedError

