"""
hyperparameter_tuning.py

Real hyperparameter search using GridSearchCV or RandomizedSearchCV over
the model registry's predefined search spaces. Search spaces are kept
small by default so tuning stays fast; sensible defaults avoid runaway
search times on typical demo-sized datasets.
"""
from __future__ import annotations

import time
from dataclasses import dataclass

import pandas as pd
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.pipeline import Pipeline

from ml_lab.models.model_registry import get_model
from ml_lab.preprocessing.pipeline import build_preprocessor


@dataclass
class TuningResult:
    best_params: dict
    best_score: float
    best_estimator: Pipeline
    search_time_seconds: float
    cv_results_summary: list


class HyperparameterTuner:
    def tune(
        self,
        model_key: str,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        numerical_features: list[str],
        categorical_features: list[str],
        problem_type: str,
        method: str = "grid",
        cv_folds: int = 5,
        scoring: str | None = None,
        n_iter: int = 10,
    ) -> TuningResult:
        spec = get_model(model_key)
        search_space = spec.search_space

        if not search_space:
            raise ValueError(
                f"'{spec.display_name}' has no tunable hyperparameters defined."
            )

        preprocessor = build_preprocessor(
            numerical_features=numerical_features,
            categorical_features=categorical_features,
        )
        estimator = spec.build()
        pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("model", estimator)])

        # Prefix search space keys for the pipeline step
        param_grid = {f"model__{k}": v for k, v in search_space.items()}

        default_scoring = "accuracy" if problem_type == "classification" else "r2"
        scoring = scoring or default_scoring

        start = time.perf_counter()
        if method == "random":
            search = RandomizedSearchCV(
                pipeline, param_distributions=param_grid, n_iter=n_iter,
                cv=cv_folds, scoring=scoring, random_state=42, n_jobs=-1,
            )
        else:
            search = GridSearchCV(
                pipeline, param_grid=param_grid, cv=cv_folds, scoring=scoring, n_jobs=-1,
            )

        search.fit(X_train, y_train)
        elapsed = time.perf_counter() - start

        best_params = {k.replace("model__", ""): v for k, v in search.best_params_.items()}

        cv_summary = []
        results = search.cv_results_
        for i in range(len(results["mean_test_score"])):
            cv_summary.append({
                "params": {k.replace("model__", ""): v for k, v in results["params"][i].items()},
                "mean_test_score": float(results["mean_test_score"][i]),
                "std_test_score": float(results["std_test_score"][i]),
            })
        cv_summary.sort(key=lambda x: x["mean_test_score"], reverse=True)

        return TuningResult(
            best_params=best_params,
            best_score=float(search.best_score_),
            best_estimator=search.best_estimator_,
            search_time_seconds=elapsed,
            cv_results_summary=cv_summary[:10],
        )
