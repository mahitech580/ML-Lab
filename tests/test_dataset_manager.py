"""Tests for the dataset manager module."""
import pandas as pd
import pytest

from ml_lab.data.dataset_manager import DatasetManager, DatasetLoadError
import config


def test_load_builtin_iris():
    manager = DatasetManager()
    df = manager.load_builtin("iris")
    assert df.shape[0] == 150
    assert "species" in df.columns


def test_load_builtin_unknown_key():
    manager = DatasetManager()
    with pytest.raises(DatasetLoadError):
        manager.load_builtin("not_a_real_dataset")


def test_load_custom_missing_file(tmp_path):
    manager = DatasetManager()
    with pytest.raises(DatasetLoadError):
        manager.load_custom(str(tmp_path / "does_not_exist.csv"))


def test_load_custom_empty_file(tmp_path):
    empty_file = tmp_path / "empty.csv"
    empty_file.write_text("")
    manager = DatasetManager()
    with pytest.raises(DatasetLoadError):
        manager.load_custom(str(empty_file))


def test_load_custom_valid_csv(tmp_path):
    path = tmp_path / "sample.csv"
    pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]}).to_csv(path, index=False)
    manager = DatasetManager()
    df = manager.load_custom(str(path))
    assert df.shape == (3, 2)


def test_get_info_requires_loaded_dataset():
    manager = DatasetManager()
    with pytest.raises(DatasetLoadError):
        manager.get_info()


def test_column_type_breakdown():
    manager = DatasetManager()
    manager.load_builtin("iris")
    breakdown = manager.column_type_breakdown()
    assert "sepal_length" in breakdown["numerical"]
    assert "species" in breakdown["categorical"]


def test_detect_target_candidates():
    manager = DatasetManager()
    manager.load_builtin("customer_churn")
    candidates = manager.detect_target_candidates()
    assert "churn" in candidates
