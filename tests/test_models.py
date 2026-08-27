"""Tests for the model registry and trainer."""
import pandas as pd
import pytest
from sklearn.model_selection import train_test_split

from ml_lab.data.dataset_manager import DatasetManager
from ml_lab.models.model_registry import get_model, get_models_for_task, REGISTRY
from ml_lab.models.trainer import ModelTrainer


def test_registry_has_all_expected_categories():
    tasks = {spec.task for spec in REGISTRY.values()}
    assert "classification" in tasks
    assert "regression" in tasks
    assert "clustering" in tasks


def test_get_model_unknown_key_raises():
    with pytest.raises(KeyError):
        get_model("not_a_real_model")


def test_get_models_for_task_classification():
    models = get_models_for_task("classification")
    assert len(models) >= 5
    keys = {m.key for m in models}
    assert "random_forest_classifier" in keys
    assert "logistic_regression" in keys


@pytest.fixture
def iris_split():
    manager = DatasetManager()
    df = manager.load_builtin("iris")
    X = df.drop(columns=["species"])
    y = df["species"]
    return train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)


def test_trainer_trains_real_classifier(iris_split):
    X_train, X_test, y_train, y_test = iris_split
    trainer = ModelTrainer()
    numerical = X_train.columns.tolist()
    result = trainer.train(
        "random_forest_classifier", X_train, y_train, numerical, []
    )
    assert result.pipeline is not None
    predictions = result.pipeline.predict(X_test)
    assert len(predictions) == len(X_test)
    # A random forest on Iris should do meaningfully better than chance (3 classes)
    accuracy = (predictions == y_test.values).mean()
    assert accuracy > 0.5


def test_trainer_respects_param_overrides(iris_split):
    X_train, X_test, y_train, y_test = iris_split
    trainer = ModelTrainer()
    numerical = X_train.columns.tolist()
    result = trainer.train(
        "random_forest_classifier", X_train, y_train, numerical, [],
        param_overrides={"n_estimators": 17},
    )
    assert result.pipeline.named_steps["model"].n_estimators == 17
