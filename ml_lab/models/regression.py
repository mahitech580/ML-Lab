"""regression.py - convenience accessors for regression models."""
from __future__ import annotations
from ml_lab.models.model_registry import get_models_for_task, ModelSpec


def list_regression_models() -> list[ModelSpec]:
    return get_models_for_task("regression")
