"""Tests for classification, regression, and clustering metrics."""
import numpy as np
import pytest

from ml_lab.evaluation.classification_metrics import compute_classification_metrics
from ml_lab.evaluation.regression_metrics import compute_regression_metrics
from ml_lab.evaluation.clustering_metrics import compute_clustering_metrics


def test_classification_metrics_perfect_predictions():
    y_true = [0, 1, 0, 1, 1]
    y_pred = [0, 1, 0, 1, 1]
    metrics = compute_classification_metrics(y_true, y_pred)
    assert metrics["accuracy"] == 1.0
    assert metrics["f1"] == 1.0


def test_classification_metrics_imperfect_predictions():
    y_true = [0, 1, 0, 1, 1]
    y_pred = [0, 0, 0, 1, 1]
    metrics = compute_classification_metrics(y_true, y_pred)
    assert 0.0 < metrics["accuracy"] < 1.0


def test_regression_metrics_perfect_predictions():
    y_true = [1.0, 2.0, 3.0, 4.0]
    y_pred = [1.0, 2.0, 3.0, 4.0]
    metrics = compute_regression_metrics(y_true, y_pred)
    assert metrics["mae"] == pytest.approx(0.0)
    assert metrics["r2"] == pytest.approx(1.0)


def test_regression_metrics_known_error():
    y_true = [1.0, 2.0, 3.0]
    y_pred = [2.0, 2.0, 2.0]
    metrics = compute_regression_metrics(y_true, y_pred)
    assert metrics["mae"] == pytest.approx(2.0 / 3.0)


def test_clustering_metrics_basic():
    X = np.array([[0, 0], [0, 1], [10, 10], [10, 11]])
    labels = np.array([0, 0, 1, 1])
    metrics = compute_clustering_metrics(X, labels, inertia=1.5)
    assert metrics["n_clusters"] == 2
    assert metrics["inertia"] == 1.5
    assert metrics["silhouette_score"] is not None
    assert metrics["silhouette_score"] > 0.5  # well-separated clusters
