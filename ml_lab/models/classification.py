"""classification.py - convenience accessors for classification models."""
from __future__ import annotations
from ml_lab.models.model_registry import get_models_for_task, ModelSpec


def list_classification_models() -> list[ModelSpec]:
    return get_models_for_task("classification")
