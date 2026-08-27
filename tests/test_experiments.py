"""Tests for experiment tracking (logging, listing, deleting)."""
import json

from ml_lab.experiments.tracker import ExperimentTracker


def test_log_and_list_experiment(tmp_path):
    tracker = ExperimentTracker(directory=tmp_path)
    exp = tracker.log_experiment(
        dataset="iris", problem_type="classification", target="species",
        features=["a", "b"], preprocessing={"scaler": "standard"},
        model="Random Forest", parameters={"n_estimators": 100},
        metrics={"accuracy": 0.95}, training_time_seconds=0.5,
    )
    assert exp.experiment_id == "experiment_001"
    experiments = tracker.list_experiments()
    assert len(experiments) == 1
    assert experiments[0].dataset == "iris"


def test_experiment_ids_increment(tmp_path):
    tracker = ExperimentTracker(directory=tmp_path)
    for _ in range(3):
        tracker.log_experiment(
            dataset="d", problem_type="classification", target="t",
            features=[], preprocessing={}, model="m", parameters={},
            metrics={}, training_time_seconds=0.1,
        )
    ids = [e.experiment_id for e in tracker.list_experiments()]
    assert ids == ["experiment_001", "experiment_002", "experiment_003"]


def test_delete_experiment(tmp_path):
    tracker = ExperimentTracker(directory=tmp_path)
    exp = tracker.log_experiment(
        dataset="d", problem_type="classification", target="t",
        features=[], preprocessing={}, model="m", parameters={},
        metrics={}, training_time_seconds=0.1,
    )
    assert tracker.delete_experiment(exp.experiment_id) is True
    assert tracker.list_experiments() == []


def test_experiment_json_is_serializable(tmp_path):
    import numpy as np
    tracker = ExperimentTracker(directory=tmp_path)
    tracker.log_experiment(
        dataset="d", problem_type="classification", target="t",
        features=[], preprocessing={}, model="m",
        parameters={"n_estimators": np.int64(100)},
        metrics={"accuracy": np.float64(0.9)}, training_time_seconds=0.1,
    )
    path = tmp_path / "experiment_001.json"
    with open(path) as f:
        data = json.load(f)  # should not raise
    assert data["metrics"]["accuracy"] == 0.9
