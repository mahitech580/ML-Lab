"""
clustering_metrics.py

Real clustering metric calculations using scikit-learn.
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import silhouette_score


def compute_clustering_metrics(X, labels, inertia: float | None = None) -> dict:
    unique_labels = np.unique(labels)
    n_clusters = len(unique_labels[unique_labels != -1]) if -1 in unique_labels else len(unique_labels)

    metrics = {
        "n_clusters": int(n_clusters),
        "inertia": float(inertia) if inertia is not None else None,
        "silhouette_score": None,
    }

    if n_clusters >= 2 and n_clusters < len(X):
        try:
            metrics["silhouette_score"] = float(silhouette_score(X, labels))
        except Exception:  # noqa: BLE001
            metrics["silhouette_score"] = None

    cluster_sizes = {}
    for label in unique_labels:
        cluster_sizes[str(label)] = int(np.sum(labels == label))
    metrics["cluster_sizes"] = cluster_sizes

    return metrics
