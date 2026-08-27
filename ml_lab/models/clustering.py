"""clustering.py - convenience accessors for clustering models."""
from __future__ import annotations
from ml_lab.models.model_registry import get_models_for_task, ModelSpec


def list_clustering_models() -> list[ModelSpec]:
    return get_models_for_task("clustering")
