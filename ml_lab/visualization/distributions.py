"""
distributions.py

Helpers for summarizing feature distributions (used by the EDA menu to
decide which plot types make sense for a given column).
"""
from __future__ import annotations

import pandas as pd


def is_plottable_numeric(df: pd.DataFrame, column: str) -> bool:
    return pd.api.types.is_numeric_dtype(df[column]) and df[column].notna().sum() > 1


def is_plottable_categorical(df: pd.DataFrame, column: str, max_categories: int = 30) -> bool:
    return df[column].nunique(dropna=True) <= max_categories


def summarize_distribution(df: pd.DataFrame, column: str) -> dict:
    series = df[column].dropna()
    if pd.api.types.is_numeric_dtype(series):
        return {
            "type": "numeric",
            "mean": float(series.mean()),
            "median": float(series.median()),
            "std": float(series.std()),
            "min": float(series.min()),
            "max": float(series.max()),
        }
    counts = series.value_counts(normalize=True).head(10)
    return {"type": "categorical", "top_categories": counts.to_dict()}
