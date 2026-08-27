"""
trainer.py

Core model-training workflow. Builds a preprocessing + estimator pipeline,
fits it on the training split only (never on test data), and returns the
fitted pipeline plus timing information. All results are computed for real
— there is no simulated or hard-coded output anywhere in this module.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass

import pandas as pd
from sklearn.pipeline import Pipeline

from ml_lab.models.model_registry import ModelSpec, get_model
from ml_lab.preprocessing.pipeline import build_preprocessor

logger = logging.getLogger("ml_lab.trainer")


@dataclass
class TrainingResult:
    pipeline: Pipeline
    model_spec: ModelSpec
    training_time_seconds: float
    params_used: dict


class ModelTrainer:
    """Fits a full preprocessing + estimator pipeline on training data."""

    def train(
        self,
        model_key: str,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        numerical_features: list[str],
        categorical_features: list[str],
        scaler: str = "standard",
        encoding: str = "onehot",
        param_overrides: dict | None = None,
    ) -> TrainingResult:
        spec = get_model(model_key)
        param_overrides = param_overrides or {}

        preprocessor = build_preprocessor(
            numerical_features=numerical_features,
            categorical_features=categorical_features,
            scaler=scaler,
            encoding=encoding,
        )
        estimator = spec.build(**param_overrides)

        pipeline = Pipeline(steps=[
            ("preprocessor", preprocessor),
            ("model", estimator),
        ])

        start = time.perf_counter()
        pipeline.fit(X_train, y_train)
        elapsed = time.perf_counter() - start

        logger.info(
            "Trained '%s' on %d samples in %.3fs", spec.display_name, len(X_train), elapsed
        )

        return TrainingResult(
            pipeline=pipeline,
            model_spec=spec,
            training_time_seconds=elapsed,
            params_used={**spec.default_params, **param_overrides},
        )

    def train_clustering(
        self,
        model_key: str,
        X: pd.DataFrame,
        numerical_features: list[str],
        categorical_features: list[str],
        scaler: str = "standard",
        encoding: str = "onehot",
        param_overrides: dict | None = None,
    ) -> TrainingResult:
        spec = get_model(model_key)
        param_overrides = param_overrides or {}

        preprocessor = build_preprocessor(
            numerical_features=numerical_features,
            categorical_features=categorical_features,
            scaler=scaler,
            encoding=encoding,
        )
        estimator = spec.build(**param_overrides)

        pipeline = Pipeline(steps=[
            ("preprocessor", preprocessor),
            ("model", estimator),
        ])

        start = time.perf_counter()
        pipeline.fit(X)
        elapsed = time.perf_counter() - start

        return TrainingResult(
            pipeline=pipeline,
            model_spec=spec,
            training_time_seconds=elapsed,
            params_used={**spec.default_params, **param_overrides},
        )
