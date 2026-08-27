"""
classification_metrics.py

Real classification metric calculations using scikit-learn. Supports
both binary and multiclass problems.
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    classification_report,
)


def compute_classification_metrics(y_true, y_pred, y_proba=None) -> dict:
    labels = np.unique(np.concatenate([np.asarray(y_true), np.asarray(y_pred)]))
    is_binary = len(labels) == 2
    average = "binary" if is_binary else "macro"

    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, average=average, zero_division=0,
                                            pos_label=labels[-1] if is_binary else None)
                            if is_binary else
                            precision_score(y_true, y_pred, average=average, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, average=average, zero_division=0,
                                      pos_label=labels[-1] if is_binary else None)
                        if is_binary else
                        recall_score(y_true, y_pred, average=average, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, average=average, zero_division=0,
                              pos_label=labels[-1] if is_binary else None)
                    if is_binary else
                    f1_score(y_true, y_pred, average=average, zero_division=0)),
    }

    if y_proba is not None:
        try:
            if is_binary:
                proba_positive = y_proba[:, 1] if y_proba.ndim == 2 else y_proba
                metrics["roc_auc"] = float(roc_auc_score(y_true, proba_positive))
            else:
                metrics["roc_auc"] = float(
                    roc_auc_score(y_true, y_proba, multi_class="ovr", average="macro")
                )
        except Exception:  # noqa: BLE001
            metrics["roc_auc"] = None
    else:
        metrics["roc_auc"] = None

    return metrics


def compute_confusion_matrix(y_true, y_pred):
    labels = sorted(np.unique(np.concatenate([np.asarray(y_true), np.asarray(y_pred)])).tolist())
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    return cm, labels


def compute_classification_report(y_true, y_pred) -> str:
    return classification_report(y_true, y_pred, zero_division=0)
