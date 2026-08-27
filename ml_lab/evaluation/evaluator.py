"""
evaluator.py

Orchestrates evaluation for classification, regression, and clustering
models. Also handles cross-validation, baseline comparison (Dummy
models), and feature importance extraction.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score

from ml_lab.evaluation.classification_metrics import (
    compute_classification_metrics,
    compute_confusion_matrix,
    compute_classification_report,
)
from ml_lab.evaluation.regression_metrics import compute_regression_metrics
from ml_lab.evaluation.clustering_metrics import compute_clustering_metrics


class ModelEvaluator:
    """Real, calculated evaluation of trained pipelines. No fabricated metrics."""

    # ------------------------------------------------------------------ #
    # Classification
    # ------------------------------------------------------------------ #
    def evaluate_classification(self, pipeline, X_test, y_test) -> dict:
        y_pred = pipeline.predict(X_test)
        y_proba = None
        if hasattr(pipeline, "predict_proba"):
            try:
                y_proba = pipeline.predict_proba(X_test)
            except Exception:  # noqa: BLE001
                y_proba = None

        metrics = compute_classification_metrics(y_test, y_pred, y_proba)
        cm, labels = compute_confusion_matrix(y_test, y_pred)
        report = compute_classification_report(y_test, y_pred)

        return {
            "metrics": metrics,
            "confusion_matrix": cm,
            "labels": labels,
            "report": report,
            "y_pred": y_pred,
            "y_proba": y_proba,
        }

    # ------------------------------------------------------------------ #
    # Regression
    # ------------------------------------------------------------------ #
    def evaluate_regression(self, pipeline, X_test, y_test) -> dict:
        y_pred = pipeline.predict(X_test)
        metrics = compute_regression_metrics(y_test, y_pred)
        return {"metrics": metrics, "y_pred": y_pred}

    # ------------------------------------------------------------------ #
    # Clustering
    # ------------------------------------------------------------------ #
    def evaluate_clustering(self, pipeline, X) -> dict:
        model = pipeline.named_steps["model"]
        X_transformed = pipeline.named_steps["preprocessor"].transform(X)
        labels = model.labels_ if hasattr(model, "labels_") else pipeline.predict(X)
        inertia = getattr(model, "inertia_", None)
        metrics = compute_clustering_metrics(X_transformed, labels, inertia)
        return {"metrics": metrics, "labels": labels}

    # ------------------------------------------------------------------ #
    # Cross validation
    # ------------------------------------------------------------------ #
    def cross_validate(
        self, pipeline, X, y, folds: int = 5, problem_type: str = "classification"
    ) -> dict:
        scoring = "accuracy" if problem_type == "classification" else "r2"
        if problem_type == "classification":
            splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=42)
        else:
            splitter = KFold(n_splits=folds, shuffle=True, random_state=42)

        scores = cross_val_score(pipeline, X, y, cv=splitter, scoring=scoring)
        return {
            "fold_scores": scores.tolist(),
            "mean": float(np.mean(scores)),
            "std": float(np.std(scores)),
            "scoring": scoring,
        }

    # ------------------------------------------------------------------ #
    # Baseline comparison
    # ------------------------------------------------------------------ #
    def baseline_comparison(
        self, X_train, y_train, X_test, y_test, problem_type: str
    ) -> dict:
        if problem_type == "classification":
            baseline = DummyClassifier(strategy="most_frequent", random_state=42)
            baseline.fit(X_train, y_train)
            baseline_pred = baseline.predict(X_test)
            baseline_metrics = compute_classification_metrics(y_test, baseline_pred)
            key = "accuracy"
        else:
            baseline = DummyRegressor(strategy="mean")
            baseline.fit(X_train, y_train)
            baseline_pred = baseline.predict(X_test)
            baseline_metrics = compute_regression_metrics(y_test, baseline_pred)
            key = "r2"

        return {"baseline_metrics": baseline_metrics, "primary_metric_key": key}

    # ------------------------------------------------------------------ #
    # Feature importance
    # ------------------------------------------------------------------ #
    def feature_importance(self, pipeline, feature_names: list[str] | None = None) -> dict | None:
        model = pipeline.named_steps.get("model")
        preprocessor = pipeline.named_steps.get("preprocessor")

        try:
            output_names = list(preprocessor.get_feature_names_out())
        except Exception:  # noqa: BLE001
            output_names = feature_names or []

        importances = None
        if hasattr(model, "feature_importances_"):
            importances = model.feature_importances_
        elif hasattr(model, "coef_"):
            coef = model.coef_
            importances = np.abs(coef[0]) if coef.ndim > 1 else np.abs(coef)

        if importances is None or len(importances) != len(output_names):
            return None

        importance_dict = dict(zip(output_names, importances.tolist()))
        # Normalize to sum to 1 for readability
        total = sum(importance_dict.values())
        if total > 0:
            importance_dict = {k: v / total for k, v in importance_dict.items()}
        return dict(sorted(importance_dict.items(), key=lambda x: x[1], reverse=True))
