"""
feature_selection.py

Feature selection utilities: variance threshold, correlation-based
filtering, SelectKBest (f-statistic), and mutual information.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.feature_selection import (
    SelectKBest,
    VarianceThreshold,
    f_classif,
    f_regression,
    mutual_info_classif,
    mutual_info_regression,
)


def variance_threshold_selection(
    X: pd.DataFrame, threshold: float = 0.0
) -> tuple[pd.DataFrame, list[str]]:
    selector = VarianceThreshold(threshold=threshold)
    numeric_X = X.select_dtypes(include="number")
    if numeric_X.empty:
        return X, list(X.columns)
    selector.fit(numeric_X)
    kept = numeric_X.columns[selector.get_support()].tolist()
    non_numeric = [c for c in X.columns if c not in numeric_X.columns]
    final_cols = kept + non_numeric
    return X[final_cols], final_cols


def correlation_filter(
    X: pd.DataFrame, threshold: float = 0.9
) -> tuple[pd.DataFrame, list[str]]:
    """Remove one feature from each pair whose absolute correlation exceeds threshold."""
    numeric_X = X.select_dtypes(include="number")
    if numeric_X.shape[1] < 2:
        return X, list(X.columns)
    corr_matrix = numeric_X.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [col for col in upper.columns if any(upper[col] > threshold)]
    remaining = [c for c in X.columns if c not in to_drop]
    return X[remaining], remaining


def select_k_best(
    X: pd.DataFrame, y: pd.Series, k: int, problem_type: str = "classification"
) -> tuple[pd.DataFrame, list[str]]:
    numeric_X = X.select_dtypes(include="number").fillna(0)
    k = min(k, numeric_X.shape[1])
    score_func = f_classif if problem_type == "classification" else f_regression
    selector = SelectKBest(score_func=score_func, k=k)
    selector.fit(numeric_X, y)
    selected = numeric_X.columns[selector.get_support()].tolist()
    return X[selected], selected


def mutual_information_ranking(
    X: pd.DataFrame, y: pd.Series, problem_type: str = "classification"
) -> pd.Series:
    numeric_X = X.select_dtypes(include="number").fillna(0)
    if numeric_X.empty:
        return pd.Series(dtype=float)
    func = mutual_info_classif if problem_type == "classification" else mutual_info_regression
    scores = func(numeric_X, y, random_state=42)
    return pd.Series(scores, index=numeric_X.columns).sort_values(ascending=False)
