"""
scaling.py

Standalone numeric scaling helpers.
"""
from __future__ import annotations

import pandas as pd
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler

SCALERS = {
    "standard": StandardScaler,
    "minmax": MinMaxScaler,
    "robust": RobustScaler,
}


def scale_columns(df: pd.DataFrame, columns: list[str], method: str = "standard"):
    df = df.copy()
    scaler_cls = SCALERS.get(method, StandardScaler)
    scaler = scaler_cls()
    df[columns] = scaler.fit_transform(df[columns])
    return df, scaler
