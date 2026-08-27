"""
model_registry.py

Centralized registry of every supported ML algorithm: classification,
regression, and clustering. Each entry carries the estimator class,
default parameters, a short description, and a small hyperparameter
search space used by the tuning module. Keeping this centralized avoids
hard-coding model choices throughout the rest of the application.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from sklearn.cluster import DBSCAN, AgglomerativeClustering, KMeans
from sklearn.ensemble import (
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge, Lasso
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor


@dataclass
class ModelSpec:
    key: str
    display_name: str
    category: str  # "Linear" | "Tree" | "Ensemble" | "Instance-based" | "Probabilistic" | "Kernel" | "Clustering"
    task: str  # "classification" | "regression" | "clustering"
    estimator_factory: Callable[..., Any]
    default_params: dict = field(default_factory=dict)
    search_space: dict = field(default_factory=dict)
    description: str = ""

    def build(self, **overrides) -> Any:
        params = {**self.default_params, **overrides}
        return self.estimator_factory(**params)


_RANDOM_STATE = 42

REGISTRY: dict[str, ModelSpec] = {}


def _register(spec: ModelSpec) -> None:
    REGISTRY[spec.key] = spec


# --------------------------------------------------------------------------- #
# Classification models
# --------------------------------------------------------------------------- #
_register(ModelSpec(
    key="logistic_regression",
    display_name="Logistic Regression",
    category="Linear",
    task="classification",
    estimator_factory=LogisticRegression,
    default_params={"max_iter": 1000, "random_state": _RANDOM_STATE},
    search_space={"C": [0.01, 0.1, 1.0, 10.0], "penalty": ["l2"]},
    description="Linear model estimating class probabilities via the logistic function.",
))
_register(ModelSpec(
    key="knn_classifier",
    display_name="K-Nearest Neighbors",
    category="Instance-based",
    task="classification",
    estimator_factory=KNeighborsClassifier,
    default_params={"n_neighbors": 5},
    search_space={"n_neighbors": [3, 5, 7, 9, 11], "weights": ["uniform", "distance"]},
    description="Classifies a point by majority vote among its nearest neighbors.",
))
_register(ModelSpec(
    key="decision_tree_classifier",
    display_name="Decision Tree",
    category="Tree",
    task="classification",
    estimator_factory=DecisionTreeClassifier,
    default_params={"random_state": _RANDOM_STATE},
    search_space={"max_depth": [3, 5, 10, None], "min_samples_split": [2, 5, 10]},
    description="Splits the feature space with a sequence of decision rules.",
))
_register(ModelSpec(
    key="random_forest_classifier",
    display_name="Random Forest",
    category="Ensemble",
    task="classification",
    estimator_factory=RandomForestClassifier,
    default_params={"n_estimators": 200, "random_state": _RANDOM_STATE},
    search_space={"n_estimators": [100, 200, 300], "max_depth": [None, 10, 20],
                  "min_samples_split": [2, 5, 10]},
    description="An ensemble of decision trees trained on bootstrapped samples.",
))
_register(ModelSpec(
    key="gradient_boosting_classifier",
    display_name="Gradient Boosting",
    category="Ensemble",
    task="classification",
    estimator_factory=GradientBoostingClassifier,
    default_params={"random_state": _RANDOM_STATE},
    search_space={"n_estimators": [100, 200], "learning_rate": [0.01, 0.1, 0.2],
                  "max_depth": [2, 3, 5]},
    description="Sequentially builds trees that correct the errors of prior trees.",
))
_register(ModelSpec(
    key="gaussian_nb",
    display_name="Gaussian Naive Bayes",
    category="Probabilistic",
    task="classification",
    estimator_factory=GaussianNB,
    default_params={},
    search_space={"var_smoothing": [1e-9, 1e-8, 1e-7]},
    description="Probabilistic classifier assuming Gaussian feature distributions.",
))
_register(ModelSpec(
    key="svm_classifier",
    display_name="Support Vector Machine",
    category="Kernel",
    task="classification",
    estimator_factory=SVC,
    default_params={"probability": True, "random_state": _RANDOM_STATE},
    search_space={"C": [0.1, 1, 10], "kernel": ["linear", "rbf"]},
    description="Finds the maximum-margin boundary between classes.",
))

# --------------------------------------------------------------------------- #
# Regression models
# --------------------------------------------------------------------------- #
_register(ModelSpec(
    key="linear_regression",
    display_name="Linear Regression",
    category="Linear",
    task="regression",
    estimator_factory=LinearRegression,
    default_params={},
    search_space={},
    description="Fits a linear relationship between features and the target.",
))
_register(ModelSpec(
    key="ridge_regression",
    display_name="Ridge Regression",
    category="Linear",
    task="regression",
    estimator_factory=Ridge,
    default_params={"random_state": _RANDOM_STATE},
    search_space={"alpha": [0.1, 1.0, 10.0, 100.0]},
    description="Linear regression with L2 regularization to reduce overfitting.",
))
_register(ModelSpec(
    key="lasso_regression",
    display_name="Lasso Regression",
    category="Linear",
    task="regression",
    estimator_factory=Lasso,
    default_params={"random_state": _RANDOM_STATE, "max_iter": 5000},
    search_space={"alpha": [0.001, 0.01, 0.1, 1.0]},
    description="Linear regression with L1 regularization; can zero out weak features.",
))
_register(ModelSpec(
    key="decision_tree_regressor",
    display_name="Decision Tree Regressor",
    category="Tree",
    task="regression",
    estimator_factory=DecisionTreeRegressor,
    default_params={"random_state": _RANDOM_STATE},
    search_space={"max_depth": [3, 5, 10, None], "min_samples_split": [2, 5, 10]},
    description="Regression tree splitting the feature space to minimize error.",
))
_register(ModelSpec(
    key="random_forest_regressor",
    display_name="Random Forest Regressor",
    category="Ensemble",
    task="regression",
    estimator_factory=RandomForestRegressor,
    default_params={"n_estimators": 200, "random_state": _RANDOM_STATE},
    search_space={"n_estimators": [100, 200, 300], "max_depth": [None, 10, 20]},
    description="Ensemble of regression trees averaged for a stable prediction.",
))
_register(ModelSpec(
    key="gradient_boosting_regressor",
    display_name="Gradient Boosting Regressor",
    category="Ensemble",
    task="regression",
    estimator_factory=GradientBoostingRegressor,
    default_params={"random_state": _RANDOM_STATE},
    search_space={"n_estimators": [100, 200], "learning_rate": [0.01, 0.1, 0.2],
                  "max_depth": [2, 3, 5]},
    description="Sequentially boosted regression trees.",
))
_register(ModelSpec(
    key="knn_regressor",
    display_name="KNN Regressor",
    category="Instance-based",
    task="regression",
    estimator_factory=KNeighborsRegressor,
    default_params={"n_neighbors": 5},
    search_space={"n_neighbors": [3, 5, 7, 9, 11]},
    description="Predicts by averaging the target of the nearest neighbors.",
))

# --------------------------------------------------------------------------- #
# Clustering models
# --------------------------------------------------------------------------- #
_register(ModelSpec(
    key="kmeans",
    display_name="K-Means",
    category="Clustering",
    task="clustering",
    estimator_factory=KMeans,
    default_params={"n_clusters": 3, "n_init": 10, "random_state": _RANDOM_STATE},
    search_space={},
    description="Partitions data into k clusters by minimizing within-cluster variance.",
))
_register(ModelSpec(
    key="dbscan",
    display_name="DBSCAN",
    category="Clustering",
    task="clustering",
    estimator_factory=DBSCAN,
    default_params={"eps": 0.5, "min_samples": 5},
    search_space={},
    description="Density-based clustering that can find arbitrarily shaped clusters and noise.",
))
_register(ModelSpec(
    key="agglomerative",
    display_name="Agglomerative Clustering",
    category="Clustering",
    task="clustering",
    estimator_factory=AgglomerativeClustering,
    default_params={"n_clusters": 3},
    search_space={},
    description="Hierarchical, bottom-up clustering that merges the closest clusters.",
))


def get_models_for_task(task: str) -> list[ModelSpec]:
    return [spec for spec in REGISTRY.values() if spec.task == task]


def get_model(key: str) -> ModelSpec:
    if key not in REGISTRY:
        raise KeyError(f"Unknown model key '{key}'.")
    return REGISTRY[key]
