"""
regression_metrics.py

Real regression metric calculations using scikit-learn.
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def compute_regression_metrics(y_true, y_pred) -> dict:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    mae = float(mean_absolute_error(y_true, y_pred))
    mse = float(mean_squared_error(y_true, y_pred))
    rmse = float(np.sqrt(mse))
    r2 = float(r2_score(y_true, y_pred))

    # MAPE - guard against division by zero
    nonzero_mask = y_true != 0
    if nonzero_mask.any():
        mape = float(
            np.mean(np.abs((y_true[nonzero_mask] - y_pred[nonzero_mask]) / y_true[nonzero_mask])) * 100
        )
    else:
        mape = None

    return {"mae": mae, "mse": mse, "rmse": rmse, "r2": r2, "mape": mape}
