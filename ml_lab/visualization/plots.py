"""
plots.py

Matplotlib/Seaborn plot generation. All plots are saved as PNG files
under generated_plots/ with automatically generated, descriptive
filenames. Uses the non-interactive Agg backend so it works headlessly
in any terminal environment.
"""
from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

import config

logger = logging.getLogger("ml_lab.visualization")

sns.set_theme(style="whitegrid")


def _timestamped_path(prefix: str) -> Path:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return config.PLOTS_DIR / f"{prefix}_{ts}.png"


def plot_histogram(df: pd.DataFrame, column: str) -> Path:
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.histplot(df[column].dropna(), kde=True, ax=ax, color="#4C72B0")
    ax.set_title(f"Distribution of {column}")
    path = _timestamped_path(f"distribution_{column}")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def plot_boxplot(df: pd.DataFrame, column: str, by: str | None = None) -> Path:
    fig, ax = plt.subplots(figsize=(8, 5))
    if by and by in df.columns:
        sns.boxplot(x=by, y=column, data=df, ax=ax)
    else:
        sns.boxplot(y=df[column], ax=ax)
    ax.set_title(f"Box Plot of {column}" + (f" by {by}" if by else ""))
    path = _timestamped_path(f"boxplot_{column}")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def plot_scatter(df: pd.DataFrame, x: str, y: str, hue: str | None = None) -> Path:
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.scatterplot(data=df, x=x, y=y, hue=hue if hue in df.columns else None, ax=ax)
    ax.set_title(f"{x} vs {y}")
    path = _timestamped_path(f"scatter_{x}_vs_{y}")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def plot_bar(df: pd.DataFrame, column: str) -> Path:
    fig, ax = plt.subplots(figsize=(8, 5))
    counts = df[column].value_counts().head(20)
    sns.barplot(x=counts.values, y=counts.index.astype(str), ax=ax, orient="h")
    ax.set_title(f"Bar Chart of {column}")
    ax.set_xlabel("Count")
    path = _timestamped_path(f"bar_{column}")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def plot_line(df: pd.DataFrame, x: str, y: str) -> Path:
    fig, ax = plt.subplots(figsize=(8, 5))
    ordered = df.sort_values(x)
    ax.plot(ordered[x], ordered[y], color="#55A868")
    ax.set_title(f"{y} over {x}")
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    path = _timestamped_path(f"line_{y}_over_{x}")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def plot_count(df: pd.DataFrame, column: str) -> Path:
    fig, ax = plt.subplots(figsize=(8, 5))
    order = df[column].value_counts().index
    sns.countplot(data=df, x=column, order=order, ax=ax)
    ax.set_title(f"Count Plot of {column}")
    plt.xticks(rotation=30, ha="right")
    path = _timestamped_path(f"countplot_{column}")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def plot_correlation_heatmap(df: pd.DataFrame) -> Path | None:
    numeric_df = df.select_dtypes(include="number")
    if numeric_df.shape[1] < 2:
        return None
    fig, ax = plt.subplots(figsize=(9, 7))
    sns.heatmap(numeric_df.corr(), annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
    ax.set_title("Correlation Matrix")
    path = _timestamped_path("correlation_matrix")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def plot_confusion_matrix(cm: np.ndarray, labels: list) -> Path:
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels,
                yticklabels=labels, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix")
    path = _timestamped_path("confusion_matrix")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def plot_roc_curve(y_true, y_proba) -> Path | None:
    from sklearn.metrics import roc_curve, auc
    try:
        fpr, tpr, _ = roc_curve(y_true, y_proba)
        roc_auc = auc(fpr, tpr)
    except Exception:  # noqa: BLE001
        return None
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(fpr, tpr, label=f"ROC curve (AUC = {roc_auc:.3f})", color="#C44E52")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve")
    ax.legend()
    path = _timestamped_path("roc_curve")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def plot_precision_recall_curve(y_true, y_proba) -> Path | None:
    from sklearn.metrics import precision_recall_curve
    try:
        precision, recall, _ = precision_recall_curve(y_true, y_proba)
    except Exception:  # noqa: BLE001
        return None
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(recall, precision, color="#8172B2")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curve")
    path = _timestamped_path("precision_recall_curve")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def plot_actual_vs_predicted(y_true, y_pred) -> Path:
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(y_true, y_pred, alpha=0.5, color="#4C72B0")
    lims = [min(min(y_true), min(y_pred)), max(max(y_true), max(y_pred))]
    ax.plot(lims, lims, linestyle="--", color="gray")
    ax.set_xlabel("Actual")
    ax.set_ylabel("Predicted")
    ax.set_title("Actual vs Predicted")
    path = _timestamped_path("actual_vs_predicted")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def plot_residuals(y_true, y_pred) -> Path:
    residuals = np.asarray(y_true) - np.asarray(y_pred)
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(y_pred, residuals, alpha=0.5, color="#DD8452")
    ax.axhline(0, linestyle="--", color="gray")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Residual")
    ax.set_title("Residual Plot")
    path = _timestamped_path("residual_plot")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def plot_feature_importance(importance: dict, top_n: int = 15) -> Path:
    items = list(importance.items())[:top_n]
    names = [i[0] for i in items][::-1]
    values = [i[1] for i in items][::-1]
    fig, ax = plt.subplots(figsize=(8, max(4, len(names) * 0.4)))
    ax.barh(names, values, color="#55A868")
    ax.set_title("Feature Importance")
    ax.set_xlabel("Relative Importance")
    path = _timestamped_path("feature_importance")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def plot_cluster_scatter(X_2d: np.ndarray, labels: np.ndarray) -> Path:
    fig, ax = plt.subplots(figsize=(8, 6))
    scatter = ax.scatter(X_2d[:, 0], X_2d[:, 1], c=labels, cmap="tab10", alpha=0.7)
    ax.set_title("Cluster Assignment (2D projection)")
    legend = ax.legend(*scatter.legend_elements(), title="Cluster")
    ax.add_artist(legend)
    path = _timestamped_path("cluster_scatter")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path
