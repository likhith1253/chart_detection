from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd


class Leaderboard:
    def build(self, rows: list[dict[str, Any]], output_path: str | Path) -> Path:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).sort_values(by=["overall_score", "parser_name"], ascending=[False, True]).to_csv(path, index=False)
        return path

