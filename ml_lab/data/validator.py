"""
validator.py

Validates datasets and target columns before training, and performs
lightweight, heuristic data-leakage detection.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass
class ValidationResult:
    is_valid: bool
    errors: list = field(default_factory=list)
    warnings: list = field(default_factory=list)


class DataValidator:
    """Validation checks used prior to model training."""

    def validate_dataset(self, df: pd.DataFrame) -> ValidationResult:
        errors, warnings = [], []
        if df is None or df.empty:
            errors.append("Dataset is empty.")
            return ValidationResult(False, errors, warnings)
        if df.shape[0] < 10:
            warnings.append("Dataset has fewer than 10 rows; results may be unreliable.")
        if df.shape[1] < 2:
            errors.append("Dataset must contain at least 2 columns (features + target).")
        return ValidationResult(len(errors) == 0, errors, warnings)

    def validate_target(
        self, df: pd.DataFrame, target: str, problem_type: str
    ) -> ValidationResult:
        errors, warnings = [], []
        if target not in df.columns:
            errors.append(f"Target column '{target}' does not exist in the dataset.")
            return ValidationResult(False, errors, warnings)

        series = df[target].dropna()
        n_unique = series.nunique()

        if len(series) == 0:
            errors.append("Target column contains only missing values.")
            return ValidationResult(False, errors, warnings)

        if problem_type == "classification":
            if n_unique < 2:
                errors.append(
                    "The selected target column contains only one class. "
                    "Classification requires at least two classes."
                )
            elif n_unique > 50:
                warnings.append(
                    f"Target has {n_unique} unique values — this looks more like a "
                    "regression target than a classification target."
                )
            class_counts = series.value_counts(normalize=True)
            if len(class_counts) >= 2 and class_counts.iloc[0] > 0.97:
                warnings.append(
                    "Target classes are extremely imbalanced (>97% majority class)."
                )
        elif problem_type == "regression":
            if not pd.api.types.is_numeric_dtype(series):
                errors.append("Regression requires a numeric target column.")
            elif n_unique < 5:
                warnings.append(
                    "Target has very few unique numeric values; consider classification instead."
                )

        n_samples = df.shape[0]
        if n_samples < 20:
            errors.append("Insufficient samples to train a reliable model (need at least 20).")

        return ValidationResult(len(errors) == 0, errors, warnings)

    def detect_leakage(
        self, df: pd.DataFrame, target: str, feature_columns: list[str]
    ) -> list[str]:
        """Heuristic leakage detection. Never removes anything automatically."""
        warnings = []
        if target in feature_columns:
            warnings.append(f"Target column '{target}' is also included as a feature.")

        if target not in df.columns:
            return warnings

        target_series = df[target]
        is_numeric_target = pd.api.types.is_numeric_dtype(target_series)

        for col in feature_columns:
            if col == target or col not in df.columns:
                continue
            col_series = df[col]
            try:
                if is_numeric_target and pd.api.types.is_numeric_dtype(col_series):
                    corr = np.corrcoef(
                        col_series.fillna(col_series.median()),
                        pd.to_numeric(target_series, errors="coerce").fillna(0),
                    )[0, 1]
                    if abs(corr) > 0.95:
                        warnings.append(
                            f"Feature '{col}' has suspiciously high correlation "
                            f"({corr:.2f}) with target '{target}'."
                        )
            except Exception:  # noqa: BLE001 - heuristic only, never fatal
                continue
        return warnings
