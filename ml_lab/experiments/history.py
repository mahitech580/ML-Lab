"""
history.py

Comparison utilities across logged experiments.
"""
from __future__ import annotations

from ml_lab.experiments.tracker import Experiment


def compare_experiments(experiments: list[Experiment], metric_key: str = "accuracy") -> list[dict]:
    rows = []
    for exp in experiments:
        value = exp.metrics.get(metric_key)
        rows.append({
            "experiment_id": exp.experiment_id,
            "model": exp.model,
            "dataset": exp.dataset,
            "timestamp": exp.timestamp,
            metric_key: value,
        })
    rows.sort(key=lambda r: (r[metric_key] is None, r[metric_key]), reverse=True)
    return rows
