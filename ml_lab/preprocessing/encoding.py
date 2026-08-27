"""
encoding.py

Standalone categorical encoding helpers (used outside the full
ColumnTransformer pipeline, e.g. for quick EDA transforms).
"""
from __future__ import annotations

import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, OrdinalEncoder


def one_hot_encode(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    return pd.get_dummies(df, columns=columns, dummy_na=False)


def ordinal_encode(df: pd.DataFrame, columns: list[str]) -> tuple[pd.DataFrame, dict]:
    df = df.copy()
    encoders = {}
    for col in columns:
        enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
        df[[col]] = enc.fit_transform(df[[col]].astype(str))
        encoders[col] = enc
    return df, encoders


def label_encode_target(y: pd.Series) -> tuple["pd.Series", LabelEncoder]:
    encoder = LabelEncoder()
    encoded = encoder.fit_transform(y.astype(str))
    return pd.Series(encoded, index=y.index, name=y.name), encoder
