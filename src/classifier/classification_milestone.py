from __future__ import annotations

import json
import time
import os
import logging
import hashlib
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from src.pipeline.config import OCRConfig, PreprocessingConfig
from src.pipeline.milestone2 import Milestone2Pipeline

try:
    from xgboost import XGBClassifier
except Exception:  # pragma: no cover
    XGBClassifier = None


@dataclass
class ClassificationArtifacts:
    best_model: str
    feature_names: List[str]
    label_order: List[str]
    metrics: Dict[str, Any]


class ClassificationMilestone:
    def __init__(self, output_dir: str | Path = "results/classification") -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.model_dir = self.output_dir / "models"
        self.model_dir.mkdir(exist_ok=True)
        self.pipeline = Milestone2Pipeline(
            preprocessing_config=PreprocessingConfig(),
            ocr_config=OCRConfig(enabled_engines=[]),
        )
        self.logger = logging.getLogger(self.__class__.__name__)
        if not logging.getLogger().handlers:
            logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    def _fingerprint(self, split_csv: str | Path, df: pd.DataFrame) -> str:
        payload = {
            "split_csv": str(Path(split_csv).resolve()),
            "rows": int(len(df)),
            "labels": sorted(df["label"].unique().tolist()),
            "preprocessing": asdict(self.pipeline.preprocessor.config),
            "feature_version": "milestone2_v1",
        }
        return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()

    def _cache_paths(self) -> Dict[str, Path]:
        return {
            "csv": self.output_dir / "feature_cache.csv",
            "meta": self.output_dir / "feature_cache_metadata.json",
            "completed": self.output_dir / "completed_models.json",
        }

    def _load_feature_cache(self, df: pd.DataFrame, split_csv: str | Path) -> Optional[pd.DataFrame]:
        paths = self._cache_paths()
        fp = self._fingerprint(split_csv, df)
        if not paths["csv"].exists() or not paths["meta"].exists():
            return None
        meta = json.loads(paths["meta"].read_text())
        if meta.get("dataset_fingerprint") != fp:
            return None
        return pd.read_csv(paths["csv"])

    def _save_feature_cache(self, df: pd.DataFrame, split_csv: str | Path, feats: pd.DataFrame) -> None:
        paths = self._cache_paths()
        feats.to_csv(paths["csv"], index=False)
        meta = {
            "feature_names": [c for c in feats.columns if c not in {"image_path", "label"}],
            "dataset_fingerprint": self._fingerprint(split_csv, df),
            "preprocessing_config_hash": hashlib.sha256(json.dumps(asdict(self.pipeline.preprocessor.config), sort_keys=True).encode("utf-8")).hexdigest(),
            "feature_extraction_version": "milestone2_v1",
            "timestamp": time.time(),
        }
        paths["meta"].write_text(json.dumps(meta, indent=2))

    def _load_completed(self) -> Dict[str, bool]:
        path = self._cache_paths()["completed"]
        if path.exists():
            try:
                return json.loads(path.read_text())
            except Exception:
                return {}
        state = {}
        for name in ["logistic_regression", "svm_rbf", "random_forest", "xgboost"]:
            state[name] = (self.model_dir / f"{name}.joblib").exists()
        return state

    def _save_completed(self, state: Dict[str, bool]) -> None:
        self._cache_paths()["completed"].write_text(json.dumps(state, indent=2, sort_keys=True))

    def _load_split(self, split_csv: str | Path) -> pd.DataFrame:
        df = pd.read_csv(split_csv)
        df = df[df["label"].isin(["bar_chart", "histogram", "line_chart", "scatter_plot", "pie_chart"])].copy()
        return df

    def _vectorize(self, df: pd.DataFrame) -> pd.DataFrame:
        cache = self._load_feature_cache(df, self._current_split_csv)
        if cache is not None:
            self.logger.info("Loaded cached features: %s", self._cache_paths()["csv"])
            return cache
        rows = []
        total = len(df)
        for _, row in df.iterrows():
            idx = len(rows) + 1
            if idx == 1 or idx % 250 == 0 or idx == total:
                self.logger.info("Feature extraction %d / %d", idx, total)
            fv = self.pipeline.extract_features(row["image_path"])
            data = fv.as_dict()
            data["image_path"] = row["image_path"]
            data["label"] = row["label"]
            rows.append(data)
        out = pd.DataFrame(rows)
        self._save_feature_cache(df, self._current_split_csv, out)
        return out

    def _models(self) -> Dict[str, Any]:
        models = {
            "logistic_regression": Pipeline([
                ("scaler", StandardScaler()),
                ("clf", LogisticRegression(max_iter=2000, class_weight="balanced", random_state=42, multi_class="auto")),
            ]),
            "svm_rbf": Pipeline([
                ("scaler", StandardScaler()),
                ("clf", CalibratedClassifierCV(SVC(kernel="rbf", C=10.0, gamma="scale", class_weight="balanced", random_state=42), cv=3)),
            ]),
            "random_forest": RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1, class_weight="balanced"),
        }
        if XGBClassifier is not None:
            models["xgboost"] = XGBClassifier(
                n_estimators=300,
                max_depth=5,
                learning_rate=0.08,
                subsample=0.9,
                colsample_bytree=0.9,
                objective="multi:softprob",
                eval_metric="mlogloss",
                random_state=42,
                n_jobs=max(1, (os.cpu_count() or 4) - 1),
            )
        return models

    def run(self, split_csv: str | Path) -> Dict[str, Any]:
        import os
        start = time.perf_counter()
        self._current_split_csv = str(split_csv)
        self.logger.info("Loading dataset split manifest")
        df = self._load_split(split_csv)
        self.logger.info("Extracting features for %d rows", len(df))
        feats = self._vectorize(df)
        feature_cols = [c for c in feats.columns if c not in {"image_path", "label"}]
        feats = feats.dropna(axis=1, how="all")
        label_order = sorted(df["label"].unique().tolist())
        label_to_int = {label: idx for idx, label in enumerate(label_order)}
        int_to_label = {idx: label for label, idx in label_to_int.items()}

        train_df = feats[df["split"] == "train"].copy()
        val_df = feats[df["split"] == "validation"].copy()
        test_df = feats[df["split"] == "test"].copy()
        X_train, y_train = train_df[feature_cols], df.loc[df["split"] == "train", "label"]
        X_val, y_val = val_df[feature_cols], df.loc[df["split"] == "validation", "label"]
        X_test, y_test = test_df[feature_cols], df.loc[df["split"] == "test", "label"]
        y_train_i = y_train.map(label_to_int)
        y_val_i = y_val.map(label_to_int)
        y_test_i = y_test.map(label_to_int)

        results: Dict[str, Any] = {"models": {}, "label_order": label_order}
        completed = self._load_completed()
        best_name, best_score, best_model = None, -1.0, None
        for name, model in self._models().items():
            try:
                if completed.get(name):
                    self.logger.info("Skipping completed model %s", name)
                    model = joblib.load(self.model_dir / f"{name}.joblib")
                else:
                    self.logger.info("Training %s", name)
                    t0 = time.perf_counter()
                    model.fit(X_train, y_train_i if name == "xgboost" else y_train)
                    train_time = time.perf_counter() - t0
                    self.logger.info("%s fit in %.2fs", name, train_time)
                    self._save_model(model, name, feature_cols)
                    completed[name] = True
                    self._save_completed(completed)
                val_pred = model.predict(X_val)
                test_pred = model.predict(X_test)
                if name == "xgboost":
                    val_pred = pd.Series(val_pred).map(int_to_label).to_numpy()
                    test_pred = pd.Series(test_pred).map(int_to_label).to_numpy()
                val_proba = model.predict_proba(X_val) if hasattr(model, "predict_proba") else None
                test_proba = model.predict_proba(X_test) if hasattr(model, "predict_proba") else None
                score = f1_score(y_val, val_pred, average="macro")
                results["models"][name] = self._metrics(y_test, test_pred, label_order)
                results["models"][name]["val_macro_f1"] = round(float(score), 4)
                self.logger.info("%s val_macro_f1=%.4f", name, score)
                if score > best_score:
                    best_score, best_name, best_model = score, name, model
                    best_test_pred, best_test_proba = test_pred, test_proba
            except Exception as exc:
                self.logger.warning("Skipping %s after failure: %s", name, exc)
                completed[name] = False
                self._save_completed(completed)
                results["models"][name] = {"status": "failed", "error": str(exc)}

        results["best_model"] = best_name
        results["overall"] = self._metrics(y_test, best_test_pred, label_order)
        results["overall"]["best_val_macro_f1"] = round(float(best_score), 4)
        results["feature_names"] = feature_cols
        self.logger.info("Evaluating best model and writing reports")
        self._write_outputs(results, X_test, y_test, best_model, best_name, feature_cols, best_test_proba)
        results["training_time_sec"] = round(time.perf_counter() - start, 4)
        (self.output_dir / "metrics.json").write_text(json.dumps(results, indent=2, default=str))
        return results

    def _metrics(self, y_true, y_pred, labels):
        cm = confusion_matrix(y_true, y_pred, labels=labels)
        report = classification_report(y_true, y_pred, labels=labels, output_dict=True, zero_division=0)
        per_class = {labels[i]: round(float(cm[i, i]) / max(1, int(cm[i].sum())), 4) for i in range(len(labels))}
        return {
            "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
            "precision_macro": round(float(precision_score(y_true, y_pred, average="macro", zero_division=0)), 4),
            "recall_macro": round(float(recall_score(y_true, y_pred, average="macro", zero_division=0)), 4),
            "macro_f1": round(float(f1_score(y_true, y_pred, average="macro", zero_division=0)), 4),
            "weighted_f1": round(float(f1_score(y_true, y_pred, average="weighted", zero_division=0)), 4),
            "confusion_matrix": cm.tolist(),
            "per_class_accuracy": per_class,
            "classification_report": report,
        }

    def _save_model(self, model, name, feature_cols):
        joblib.dump(model, self.model_dir / f"{name}.joblib")
        (self.model_dir / f"{name}_features.json").write_text(json.dumps(feature_cols, indent=2))

    def _write_outputs(self, results, X_test, y_test, model, best_name, feature_cols, proba):
        metrics_path = self.output_dir / "metrics.json"
        metrics_path.write_text(json.dumps(results, indent=2, default=str))
        report_df = pd.DataFrame(results["overall"]["classification_report"]).T
        report_df.to_csv(self.output_dir / "classification_report.csv")
        pd.DataFrame(results["overall"]["confusion_matrix"]).to_csv(self.output_dir / "confusion_matrix.csv", index=False)
        if model is not None:
            self._feature_importance(model, feature_cols).to_csv(self.output_dir / "feature_importance.csv", index=False)
        self._save_confusion_png(results["overall"]["confusion_matrix"], results["label_order"])
        self._save_misclassified(X_test, y_test, model, proba)

    def _feature_importance(self, model, feature_cols):
        if hasattr(model, "feature_importances_"):
            imp = model.feature_importances_
        elif hasattr(model, "named_steps") and "clf" in model.named_steps and hasattr(model.named_steps["clf"], "coef_"):
            imp = np.mean(np.abs(model.named_steps["clf"].coef_), axis=0)
        else:
            return pd.DataFrame(columns=["feature", "importance"])
        return pd.DataFrame({"feature": feature_cols, "importance": imp}).sort_values("importance", ascending=False)

    def _save_confusion_png(self, cm, labels):
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(6, 5))
        im = ax.imshow(cm, cmap="Blues")
        ax.set_xticks(range(len(labels)), labels=labels, rotation=45, ha="right")
        ax.set_yticks(range(len(labels)), labels=labels)
        fig.colorbar(im, ax=ax)
        fig.tight_layout()
        fig.savefig(self.output_dir / "confusion_matrix.png", dpi=200)
        plt.close(fig)

    def _save_misclassified(self, X_test, y_test, model, proba):
        if model is None:
            return
        pred = model.predict(X_test)
        if proba is None and hasattr(model, "predict_proba"):
            proba = model.predict_proba(X_test)
        conf = np.max(proba, axis=1) if proba is not None else np.ones(len(pred))
        wrong = np.where(pred != np.asarray(y_test))[0]
        rows = [{"index": int(i), "true": y_test.iloc[i], "pred": pred[i], "confidence": float(conf[i])} for i in wrong]
        pd.DataFrame(rows).sort_values("confidence", ascending=False).head(20).to_csv(self.output_dir / "top20_misclassified.csv", index=False)
