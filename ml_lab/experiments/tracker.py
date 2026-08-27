"""
tracker.py

Persists experiment records as JSON files under experiments/. Each
training run creates one record capturing the dataset, problem type,
model, parameters, metrics, and timing so later analysis or comparison
is possible without re-running anything.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path

import config

logger = logging.getLogger("ml_lab.experiments")


@dataclass
class Experiment:
    experiment_id: str
    timestamp: str
    dataset: str
    problem_type: str
    target: str
    features: list
    preprocessing: dict
    model: str
    parameters: dict
    metrics: dict
    training_time_seconds: float
    notes: str = ""


class ExperimentTracker:
    def __init__(self, directory: Path | None = None):
        self.directory = directory or config.EXPERIMENTS_DIR
        self.directory.mkdir(parents=True, exist_ok=True)

    def _next_id(self) -> str:
        existing = sorted(self.directory.glob("experiment_*.json"))
        if not existing:
            return "experiment_001"
        last_num = max(int(p.stem.split("_")[1]) for p in existing)
        return f"experiment_{last_num + 1:03d}"

    def log_experiment(
        self,
        dataset: str,
        problem_type: str,
        target: str,
        features: list,
        preprocessing: dict,
        model: str,
        parameters: dict,
        metrics: dict,
        training_time_seconds: float,
        notes: str = "",
    ) -> Experiment:
        exp = Experiment(
            experiment_id=self._next_id(),
            timestamp=datetime.now().isoformat(timespec="seconds"),
            dataset=dataset,
            problem_type=problem_type,
            target=target,
            features=features,
            preprocessing=preprocessing,
            model=model,
            parameters=_json_safe(parameters),
            metrics=_json_safe(metrics),
            training_time_seconds=round(training_time_seconds, 4),
            notes=notes,
        )
        path = self.directory / f"{exp.experiment_id}.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(asdict(exp), f, indent=2)
        logger.info("Logged experiment %s", exp.experiment_id)
        return exp

    def list_experiments(self) -> list[Experiment]:
        results = []
        for path in sorted(self.directory.glob("experiment_*.json")):
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            results.append(Experiment(**data))
        return results

    def get_experiment(self, experiment_id: str) -> Experiment | None:
        path = self.directory / f"{experiment_id}.json"
        if not path.exists():
            return None
        with open(path, "r", encoding="utf-8") as f:
            return Experiment(**json.load(f))

    def delete_experiment(self, experiment_id: str) -> bool:
        path = self.directory / f"{experiment_id}.json"
        if path.exists():
            path.unlink()
            return True
        return False


def _json_safe(obj):
    """Recursively convert numpy/pandas scalars to plain Python types."""
    import numpy as np
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj
