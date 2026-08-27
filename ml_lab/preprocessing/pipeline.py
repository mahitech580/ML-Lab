"""
pipeline.py

Builds a leakage-safe scikit-learn preprocessing pipeline (ColumnTransformer)
for numerical and categorical features. The transformer must be fit ONLY on
training data; callers are responsible for calling fit_transform on X_train
and transform (never fit) on X_test.
"""
from __future__ import annotations

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    MinMaxScaler,
    OneHotEncoder,
    OrdinalEncoder,
    RobustScaler,
    StandardScaler,
)

SCALER_MAP = {
    "standard": StandardScaler,
    "minmax": MinMaxScaler,
    "robust": RobustScaler,
}

NUMERIC_IMPUTE_STRATEGIES = {"mean", "median", "most_frequent", "constant"}


def build_preprocessor(
    numerical_features: list[str],
    categorical_features: list[str],
    scaler: str = "standard",
    encoding: str = "onehot",
    numeric_impute_strategy: str = "median",
    categorical_impute_strategy: str = "most_frequent",
) -> ColumnTransformer:
    """Construct a ColumnTransformer that scales numeric features and
    encodes categorical features. Safe to fit_transform on train and
    transform-only on test/new data.
    """
    transformers = []

    if numerical_features:
        scaler_cls = SCALER_MAP.get(scaler, StandardScaler)
        numeric_pipeline = Pipeline(steps=[
            ("imputer", SimpleImputer(strategy=numeric_impute_strategy)),
            ("scaler", scaler_cls()),
        ])
        transformers.append(("num", numeric_pipeline, numerical_features))

    if categorical_features:
        if encoding == "ordinal":
            encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
        else:
            encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        categorical_pipeline = Pipeline(steps=[
            ("imputer", SimpleImputer(strategy=categorical_impute_strategy)),
            ("encoder", encoder),
        ])
        transformers.append(("cat", categorical_pipeline, categorical_features))

    if not transformers:
        raise ValueError("No numerical or categorical features provided to preprocessor.")

    return ColumnTransformer(transformers=transformers, remainder="drop")


def get_output_feature_names(preprocessor: ColumnTransformer) -> list[str]:
    """Best-effort retrieval of feature names after transformation."""
    try:
        return list(preprocessor.get_feature_names_out())
    except Exception:  # noqa: BLE001
        return []
