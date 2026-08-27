"""
missing_values.py

Convenience wrappers around SimpleImputer for quick, column-level
missing-value handling outside the full ColumnTransformer pipeline.
"""
from __future__ import annotations

import pandas as pd
from sklearn.impute import SimpleImputer


def impute_numeric(df: pd.DataFrame, columns: list[str], strategy: str = "median") -> pd.DataFrame:
    df = df.copy()
    imputer = SimpleImputer(strategy=strategy)
    df[columns] = imputer.fit_transform(df[columns])
    return df


def impute_categorical(df: pd.DataFrame, columns: list[str], strategy: str = "most_frequent") -> pd.DataFrame:
    df = df.copy()
    imputer = SimpleImputer(strategy=strategy)
    df[columns] = imputer.fit_transform(df[columns])
    return df


def missing_value_summary(df: pd.DataFrame) -> pd.DataFrame:
    missing = df.isna().sum()
    pct = (missing / len(df) * 100).round(2)
    summary = pd.DataFrame({"missing_count": missing, "missing_pct": pct})
    return summary[summary["missing_count"] > 0].sort_values("missing_count", ascending=False)
