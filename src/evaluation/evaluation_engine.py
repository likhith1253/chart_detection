from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from .leaderboard import Leaderboard
from .parser_evaluator import ParserEvaluationRecord, ParserEvaluator
from .report_generator import ReportGenerator


@dataclass
class EvaluationRun:
    records: list[ParserEvaluationRecord] = field(default_factory=list)
    configuration: dict[str, Any] = field(default_factory=dict)


class EvaluationEngine:
    def __init__(self, output_dir: str | Path) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.evaluator = ParserEvaluator()
        self.reporter = ReportGenerator()
        self.leaderboard = Leaderboard()

    def add(self, record: ParserEvaluationRecord) -> None:
        self.run.records.append(record)

    def evaluate(self, parser_name: str, chart_type: str, prediction: dict[str, Any], ground_truth: Any, runtime_seconds: float = 0.0, memory_megabytes: float = 0.0) -> ParserEvaluationRecord:
        record = self.evaluator.evaluate(parser_name, chart_type, prediction, ground_truth, runtime_seconds, memory_megabytes)
        self.run.records.append(record)
        return record

    def aggregate(self) -> dict[str, Any]:
        rows = [asdict(r) for r in self.run.records]
        df = pd.DataFrame(rows)
        if df.empty:
            return {"records": [], "overall": {}, "per_parser": {}}
        overall = {}
        metric_cols = [c for c in ["runtime_seconds", "memory_megabytes"] if c in df.columns]
        for col in metric_cols:
            overall[col] = float(df[col].mean())
        for metric in sorted({m for r in self.run.records for m in r.metrics}):
            overall[metric] = float(df["metrics"].apply(lambda d: d.get(metric, 0.0)).mean())
        per_parser: dict[str, dict[str, float]] = {}
        for parser_name, group in df.groupby("parser_name"):
            metric_df = pd.DataFrame(list(group["metrics"]))
            per_parser[str(parser_name)] = {
                str(col): float(metric_df[col].mean()) for col in metric_df.columns
            }
        return {"records": rows, "overall": overall, "per_parser": per_parser}

    def save(self, dataset: str, seed: int, software_versions: dict[str, str] | None = None, configuration: dict[str, Any] | None = None) -> dict[str, Path]:
        software_versions = software_versions or {}
        configuration = configuration or {}
        agg = self.aggregate()
        rows = []
        for record in self.run.records:
            row = {"parser_name": record.parser_name, "chart_type": record.chart_type, "chart_id": record.chart_id, "runtime_seconds": record.runtime_seconds, "memory_megabytes": record.memory_megabytes, **record.metrics}
            rows.append(row)
        df = pd.DataFrame(rows)
        json_path = self.output_dir / "evaluation_report.json"
        csv_path = self.output_dir / "evaluation_report.csv"
        leaderboard_path = self.output_dir / "leaderboard.csv"
        pdf_path = self.output_dir / "summary.pdf"
        self.reporter.json_report({"dataset": dataset, "seed": seed, "software_versions": software_versions, "configuration": configuration, **agg}, json_path)
        df.to_csv(csv_path, index=False)
        leaderboard_rows = [{"parser_name": p, "overall_score": float(v.get("parsing_success_rate", 0.0) + v.get("geometry_iou", 0.0) + v.get("value_mae", 0.0) * -1.0)} for p, v in agg.get("per_parser", {}).items()]
        self.leaderboard.build(leaderboard_rows, leaderboard_path)
        self.reporter.pdf_summary("Evaluation Summary", [f"dataset: {dataset}", f"seed: {seed}", f"records: {len(rows)}"], pdf_path)
        return {"json": json_path, "csv": csv_path, "leaderboard": leaderboard_path, "pdf": pdf_path}

    @property
    def run(self) -> EvaluationRun:
        if not hasattr(self, "_run"):
            self._run = EvaluationRun()
        return self._run
