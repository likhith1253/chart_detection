"""
Advanced classical ML training for chart classification.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_predict, train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC

import config

logger = logging.getLogger(__name__)


class AdvancedMLClassifier:
    """Train RF/SVM/GB/XGB with tuning, CV, and OOF outputs."""

    EXTRA_COLUMNS = [
        "yolo_bars",
        "yolo_line_segments",
        "yolo_scatter_points",
        "yolo_pie_slices",
        "yolo_axes",
        "yolo_legend_boxes",
        "yolo_text_regions",
        "yolo_text_blocks",
        "histogram_score",
        "bar_chart_score",
    ]

    def __init__(self, models_dir: Path | None = None):
        self.models_dir = models_dir or config.MODELS_DIR
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.best_model = None
        self.best_model_name = None
        self.feature_columns: List[str] = []

    def _feature_columns(self, df: pd.DataFrame) -> List[str]:
        features = [c for c in df.columns if c.startswith("feature_")]
        features.extend([c for c in self.EXTRA_COLUMNS if c in df.columns])
        unique = sorted(set(features))
        return unique

    @staticmethod
    def _model_candidates():
        models = {
            "RandomForest": (
                RandomForestClassifier(class_weight="balanced", random_state=42, n_jobs=-1),
                {
                    "n_estimators": [200, 400],
                    "max_depth": [20, None],
                    "min_samples_split": [2, 4],
                },
            ),
            "SVM": (
                SVC(probability=True, class_weight="balanced", random_state=42),
                {
                    "C": [5.0, 10.0],
                    "kernel": ["rbf"],
                    "gamma": ["scale", "auto"],
                },
            ),
            "GradientBoosting": (
                GradientBoostingClassifier(random_state=42),
                {
                    "n_estimators": [150, 250],
                    "learning_rate": [0.05, 0.1],
                    "max_depth": [3],
                },
            ),
        }

        try:
            from xgboost import XGBClassifier

            models["XGBoost"] = (
                XGBClassifier(
                    random_state=42,
                    eval_metric="mlogloss",
                    objective="multi:softprob",
                    tree_method="hist",
                    n_jobs=-1,
                ),
                {
                    "n_estimators": [300, 500],
                    "max_depth": [4, 6],
                    "learning_rate": [0.05, 0.1],
                    "subsample": [0.9],
                    "colsample_bytree": [0.9],
                },
            )
        except Exception as exc:
            logger.warning("XGBoost unavailable: %s", exc)

        return models

    def train(self, df: pd.DataFrame) -> Tuple[Dict, pd.Series, pd.Series]:
        """
        Train all models and return:
        - results dict
        - OOF predicted labels for best model
        - OOF confidence for best model
        """
        work_df = df[df["true_chart_type"] != "unknown"].copy()
        self.feature_columns = self._feature_columns(work_df)
        if len(self.feature_columns) < 5:
            raise RuntimeError("Insufficient feature columns for ML training")

        X = work_df[self.feature_columns].fillna(0).to_numpy()
        y_raw = work_df["true_chart_type"].to_numpy()
        y = self.label_encoder.fit_transform(y_raw)

        X_scaled = self.scaler.fit_transform(X)
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.2, random_state=42, stratify=y
        )
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

        all_results: Dict[str, Dict] = {}
        best_f1 = -1.0
        best_estimator = None

        feature_importance = []

        for name, (model, param_grid) in self._model_candidates().items():
            logger.info("Tuning %s...", name)
            search = GridSearchCV(
                estimator=model,
                param_grid=param_grid,
                cv=cv,
                scoring="f1_weighted",
                n_jobs=-1,
                verbose=0,
            )
            search.fit(X_train, y_train)
            est = search.best_estimator_
            pred = est.predict(X_test)

            acc = accuracy_score(y_test, pred)
            prec = precision_score(y_test, pred, average="weighted", zero_division=0)
            rec = recall_score(y_test, pred, average="weighted", zero_division=0)
            f1 = f1_score(y_test, pred, average="weighted", zero_division=0)

            all_results[name] = {
                "accuracy": round(float(acc), 4),
                "precision": round(float(prec), 4),
                "recall": round(float(rec), 4),
                "f1_score": round(float(f1), 4),
                "cv_best_score": round(float(search.best_score_), 4),
                "best_params": search.best_params_,
            }

            model_path = self.models_dir / f"{name.lower()}_tuned.joblib"
            joblib.dump(est, model_path)

            if name == "RandomForest" and hasattr(est, "feature_importances_"):
                fi = sorted(
                    zip(self.feature_columns, est.feature_importances_),
                    key=lambda x: x[1],
                    reverse=True,
                )
                feature_importance = [
                    {"feature": str(feat).replace("feature_", ""), "importance": round(float(val), 6)}
                    for feat, val in fi
                ]

            if f1 > best_f1:
                best_f1 = f1
                best_estimator = est
                self.best_model_name = name

        if best_estimator is None:
            raise RuntimeError("Failed to train any ML model")

        self.best_model = best_estimator
        joblib.dump(self.best_model, self.models_dir / "best_ml_model.joblib")
        joblib.dump(self.scaler, self.models_dir / "best_ml_scaler.joblib")
        joblib.dump(self.label_encoder, self.models_dir / "best_ml_label_encoder.joblib")
        (self.models_dir / "best_ml_features.json").write_text(json.dumps(self.feature_columns, indent=2))

        # OOF predictions for robust dataset-wide evaluation
        oof_indices = work_df.index
        oof_pred_idx = cross_val_predict(self.best_model, X_scaled, y, cv=cv, method="predict")
        if hasattr(self.best_model, "predict_proba"):
            oof_proba = cross_val_predict(self.best_model, X_scaled, y, cv=cv, method="predict_proba")
            oof_conf = oof_proba.max(axis=1)
        else:
            oof_conf = np.ones(len(oof_pred_idx), dtype=float) * 0.5

        oof_pred_labels = pd.Series(
            self.label_encoder.inverse_transform(oof_pred_idx),
            index=oof_indices,
            name="ml_prediction",
        )
        oof_conf_series = pd.Series(oof_conf, index=oof_indices, name="ml_confidence")

        output = {
            "best_model": self.best_model_name,
            "models": all_results,
            "feature_columns": self.feature_columns,
            "feature_importance": feature_importance,
        }
        results_path = self.models_dir / "advanced_ml_results.json"
        results_path.write_text(json.dumps(output, indent=2))
        logger.info("Advanced ML results written to %s", results_path)
        return output, oof_pred_labels, oof_conf_series

    def predict(self, df: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
        """Predict labels and confidence for dataframe rows."""
        if self.best_model is None:
            raise RuntimeError("Model is not loaded/trained")
        X = df[self.feature_columns].fillna(0).to_numpy()
        X_scaled = self.scaler.transform(X)
        pred_idx = self.best_model.predict(X_scaled)
        if hasattr(self.best_model, "predict_proba"):
            proba = self.best_model.predict_proba(X_scaled)
            conf = proba.max(axis=1)
        else:
            conf = np.ones(len(pred_idx), dtype=float) * 0.5
        labels = self.label_encoder.inverse_transform(pred_idx)
        return (
            pd.Series(labels, index=df.index, name="ml_prediction"),
            pd.Series(conf, index=df.index, name="ml_confidence"),
        )
