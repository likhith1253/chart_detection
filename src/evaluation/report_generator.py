from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt


class ReportGenerator:
    def json_report(self, payload: dict[str, Any], output_path: str | Path) -> Path:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
        return path

    def csv_report(self, rows: list[dict[str, Any]], output_path: str | Path) -> Path:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_csv(path, index=False)
        return path

    def pdf_summary(self, title: str, lines: list[str], output_path: str | Path) -> Path:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with PdfPages(path) as pdf:
            fig = plt.figure(figsize=(8.5, 11))
            fig.text(0.08, 0.95, title, fontsize=16, weight="bold")
            y = 0.9
            for line in lines:
                fig.text(0.08, y, line, fontsize=10)
                y -= 0.03
            pdf.savefig(fig)
            plt.close(fig)
        return path

