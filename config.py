"""
config.py
Centralized configuration for ML Lab.

All paths, defaults, and constant settings used across the application
live here so nothing is hard-coded in individual modules.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

# --------------------------------------------------------------------------- #
# Base paths (cross-platform, resolved relative to this file)
# --------------------------------------------------------------------------- #
BASE_DIR: Path = Path(__file__).resolve().parent

DATASETS_DIR: Path = BASE_DIR / "datasets"
SAVED_MODELS_DIR: Path = BASE_DIR / "saved_models"
EXPERIMENTS_DIR: Path = BASE_DIR / "experiments"
REPORTS_DIR: Path = BASE_DIR / "reports"
PLOTS_DIR: Path = BASE_DIR / "generated_plots"
LOG_FILE: Path = BASE_DIR / "ml_lab.log"

for _dir in (DATASETS_DIR, SAVED_MODELS_DIR, EXPERIMENTS_DIR, REPORTS_DIR, PLOTS_DIR):
    _dir.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------------------------------- #
# Built-in datasets
# --------------------------------------------------------------------------- #
BUILTIN_DATASETS = {
    "iris": {
        "path": DATASETS_DIR / "iris.csv",
        "target": "species",
        "description": "Classic flower classification dataset (150 rows).",
    },
    "house_prices": {
        "path": DATASETS_DIR / "house_prices.csv",
        "target": "price",
        "description": "Synthetic housing price regression dataset.",
    },
    "customer_churn": {
        "path": DATASETS_DIR / "customer_churn.csv",
        "target": "churn",
        "description": "Synthetic telecom customer churn classification dataset.",
    },
}

# --------------------------------------------------------------------------- #
# Reproducibility
# --------------------------------------------------------------------------- #
RANDOM_SEED: int = 42

# --------------------------------------------------------------------------- #
# Train / test split options
# --------------------------------------------------------------------------- #
SPLIT_OPTIONS = {
    "60/40": 0.4,
    "70/30": 0.3,
    "75/25": 0.25,
    "80/20": 0.2,
    "90/10": 0.1,
}
DEFAULT_SPLIT: str = "80/20"

# --------------------------------------------------------------------------- #
# Cross validation
# --------------------------------------------------------------------------- #
CV_FOLD_OPTIONS = [3, 5, 10]
DEFAULT_CV_FOLDS: int = 5

# --------------------------------------------------------------------------- #
# Large dataset warning threshold
# --------------------------------------------------------------------------- #
LARGE_DATASET_ROW_THRESHOLD: int = 100_000

# --------------------------------------------------------------------------- #
# Supported model keys (used by the model registry)
# --------------------------------------------------------------------------- #
CLASSIFICATION_MODELS = [
    "logistic_regression",
    "knn_classifier",
    "decision_tree_classifier",
    "random_forest_classifier",
    "gradient_boosting_classifier",
    "gaussian_nb",
    "svm_classifier",
]

REGRESSION_MODELS = [
    "linear_regression",
    "ridge_regression",
    "lasso_regression",
    "decision_tree_regressor",
    "random_forest_regressor",
    "gradient_boosting_regressor",
    "knn_regressor",
]

CLUSTERING_MODELS = [
    "kmeans",
    "dbscan",
    "agglomerative",
]

# --------------------------------------------------------------------------- #
# Application metadata
# --------------------------------------------------------------------------- #
APP_NAME = "ML Lab"
APP_TAGLINE = "Interactive Machine Learning Laboratory"
APP_VERSION = "1.0.0"


@dataclass
class AppState:
    """Mutable, in-memory state shared across the running session."""

    dataset_name: str | None = None
    original_dataset: object = None
    working_dataset: object = None
    processed_dataset: object = None
    target_column: str | None = None
    problem_type: str | None = None  # "classification" | "regression" | "clustering"
    numerical_features: list = field(default_factory=list)
    categorical_features: list = field(default_factory=list)
    feature_columns: list = field(default_factory=list)

    X_train: object = None
    X_test: object = None
    y_train: object = None
    y_test: object = None
    split_label: str | None = None

    preprocessor: object = None
    trained_model: object = None
    trained_model_name: str | None = None
    last_metrics: dict = field(default_factory=dict)
    last_feature_importance: dict = field(default_factory=dict)
    last_training_time: float | None = None

    comparison_results: list = field(default_factory=list)

    def status(self) -> str:
        if self.trained_model is not None:
            return "MODEL TRAINED"
        if self.working_dataset is not None:
            return "DATASET READY"
        return "NO DATASET"
