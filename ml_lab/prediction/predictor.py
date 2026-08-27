"""
predictor.py

Handles custom, single-record predictions using a trained pipeline.
Automatically detects which raw input features the pipeline expects
and builds a single-row dataframe from user-supplied values before
calling predict(). Never hard-codes an output — every prediction is
computed by the fitted model.
"""
from __future__ import annotations

import pandas as pd


class PredictionEngine:
    def __init__(self, pipeline, feature_columns: list[str], numerical_features: list[str],
                 categorical_features: list[str]):
        self.pipeline = pipeline
        self.feature_columns = feature_columns
        self.numerical_features = set(numerical_features)
        self.categorical_features = set(categorical_features)

    def expected_inputs(self) -> list[tuple[str, str]]:
        """Return (column_name, expected_type) pairs for prompting the user."""
        result = []
        for col in self.feature_columns:
            kind = "numeric" if col in self.numerical_features else "text"
            result.append((col, kind))
        return result

    def build_input_row(self, values: dict) -> pd.DataFrame:
        row = {col: values.get(col) for col in self.feature_columns}
        return pd.DataFrame([row])

    def predict(self, values: dict) -> dict:
        X_new = self.build_input_row(values)
        prediction = self.pipeline.predict(X_new)[0]

        result = {"prediction": prediction, "confidence": None, "probabilities": None}

        if hasattr(self.pipeline, "predict_proba"):
            try:
                proba = self.pipeline.predict_proba(X_new)[0]
                classes = self.pipeline.named_steps["model"].classes_
                result["probabilities"] = dict(zip(classes.tolist(), proba.tolist()))
                result["confidence"] = float(max(proba))
            except Exception:  # noqa: BLE001
                pass

        return result
