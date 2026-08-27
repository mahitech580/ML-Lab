"""Tests for the preprocessing pipeline and feature selection modules."""
import numpy as np
import pandas as pd
import pytest

from ml_lab.preprocessing.pipeline import build_preprocessor
from ml_lab.preprocessing import feature_selection as fs


@pytest.fixture
def sample_df():
    return pd.DataFrame({
        "num1": [1.0, 2.0, np.nan, 4.0, 5.0],
        "num2": [10, 20, 30, 40, 50],
        "cat1": ["a", "b", "a", "b", "a"],
        "target": [0, 1, 0, 1, 0],
    })


def test_build_preprocessor_fits_and_transforms(sample_df):
    preprocessor = build_preprocessor(["num1", "num2"], ["cat1"])
    X = sample_df[["num1", "num2", "cat1"]]
    transformed = preprocessor.fit_transform(X)
    assert transformed.shape[0] == 5
    # num1, num2 scaled (2 cols) + cat1 one-hot (2 categories) = 4 columns
    assert transformed.shape[1] == 4


def test_build_preprocessor_no_leakage_on_transform_only(sample_df):
    preprocessor = build_preprocessor(["num1", "num2"], ["cat1"])
    X_train = sample_df[["num1", "num2", "cat1"]].iloc[:3]
    X_test = sample_df[["num1", "num2", "cat1"]].iloc[3:]
    preprocessor.fit(X_train)
    # Should not raise, and should use train statistics only
    transformed_test = preprocessor.transform(X_test)
    assert transformed_test.shape[0] == 2


def test_build_preprocessor_requires_features():
    with pytest.raises(ValueError):
        build_preprocessor([], [])


def test_variance_threshold_selection():
    df = pd.DataFrame({
        "constant": [1, 1, 1, 1],
        "varying": [1, 2, 3, 4],
    })
    _, kept = fs.variance_threshold_selection(df, threshold=0.0)
    assert "varying" in kept
    assert "constant" not in kept


def test_correlation_filter_removes_redundant_feature():
    df = pd.DataFrame({
        "a": [1, 2, 3, 4, 5],
        "b": [2, 4, 6, 8, 10],  # perfectly correlated with a
        "c": [5, 1, 4, 2, 3],
    })
    _, kept = fs.correlation_filter(df, threshold=0.9)
    assert len(kept) == 2  # one of a/b dropped, c kept


def test_select_k_best_classification():
    df = pd.DataFrame({
        "f1": [1, 2, 3, 4, 5, 6],
        "f2": [6, 5, 4, 3, 2, 1],
        "f3": [1, 1, 1, 1, 1, 1],
    })
    y = pd.Series([0, 0, 0, 1, 1, 1])
    _, selected = fs.select_k_best(df, y, k=2, problem_type="classification")
    assert len(selected) == 2
