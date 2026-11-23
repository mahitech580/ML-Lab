"""Tests for the automated end-to-end ML pipeline."""
from ml_lab.data.dataset_manager import DatasetManager
from ml_lab.pipeline.auto_pipeline import AutoPipeline


def test_pipeline_runs_end_to_end_classification(tmp_path, monkeypatch):
    import config
    monkeypatch.setattr(config, "SAVED_MODELS_DIR", tmp_path / "models")
    monkeypatch.setattr(config, "EXPERIMENTS_DIR", tmp_path / "experiments")
    monkeypatch.setattr(config, "REPORTS_DIR", tmp_path / "reports")

    manager = DatasetManager()
    df = manager.load_builtin("iris")

    pipeline = AutoPipeline()
    result = pipeline.run(
        df=df, dataset_name="iris", target="species", problem_type="classification"
    )

    assert not result.errors
    assert result.best_model_name
    assert "accuracy" in result.best_metrics
    assert result.best_metrics["accuracy"] > 0.5
    assert result.model_path
    assert result.report_path
    assert result.experiment_id


def test_pipeline_runs_end_to_end_regression(tmp_path, monkeypatch):
    import config
    monkeypatch.setattr(config, "SAVED_MODELS_DIR", tmp_path / "models")
    monkeypatch.setattr(config, "EXPERIMENTS_DIR", tmp_path / "experiments")
    monkeypatch.setattr(config, "REPORTS_DIR", tmp_path / "reports")

    manager = DatasetManager()
    df = manager.load_builtin("house_prices")

    pipeline = AutoPipeline()
    result = pipeline.run(
        df=df, dataset_name="house_prices", target="price", problem_type="regression"
    )

    assert not result.errors
    assert "r2" in result.best_metrics
    assert result.best_metrics["r2"] > 0.5


def test_pipeline_reports_error_on_invalid_target(tmp_path, monkeypatch):
    import config
    monkeypatch.setattr(config, "SAVED_MODELS_DIR", tmp_path / "models")
    monkeypatch.setattr(config, "EXPERIMENTS_DIR", tmp_path / "experiments")
    monkeypatch.setattr(config, "REPORTS_DIR", tmp_path / "reports")

    manager = DatasetManager()
    df = manager.load_builtin("iris")

    pipeline = AutoPipeline()
    result = pipeline.run(
        df=df, dataset_name="iris", target="not_a_column", problem_type="classification"
    )
    assert result.errors


def test_pipeline_includes_string_and_boolean_features(tmp_path, monkeypatch):
    import joblib
    import config

    monkeypatch.setattr(config, "SAVED_MODELS_DIR", tmp_path / "models")
    monkeypatch.setattr(config, "EXPERIMENTS_DIR", tmp_path / "experiments")
    monkeypatch.setattr(config, "REPORTS_DIR", tmp_path / "reports")

    rows = 40
    frame = pd.DataFrame({
        "record_id": range(rows),
        "segment": pd.Series(["north", "south"] * 20, dtype="string"),
        "active": pd.Series([True, False] * 20, dtype="boolean"),
        "target": pd.Series(["yes", "no"] * 20, dtype="string"),
    })

    result = AutoPipeline().run(
        df=frame,
        dataset_name="categorical-features",
        target="target",
        problem_type="classification",
        candidate_models=["decision_tree_classifier"],
    )

    assert not result.errors, result.errors
    pipeline = joblib.load(result.model_path)
    categorical_columns = next(
        columns
        for name, _transformer, columns in pipeline.named_steps["preprocessor"].transformers
        if name == "cat"
    )
    assert categorical_columns == ["segment", "active"]


def test_pipeline_rejects_constant_regression_target(tmp_path, monkeypatch):
    import config

    monkeypatch.setattr(config, "SAVED_MODELS_DIR", tmp_path / "models")
    monkeypatch.setattr(config, "EXPERIMENTS_DIR", tmp_path / "experiments")
    monkeypatch.setattr(config, "REPORTS_DIR", tmp_path / "reports")

    frame = pd.DataFrame({
        "feature": range(30),
        "target": [12.5] * 30,
    })

    result = AutoPipeline().run(
        df=frame,
        dataset_name="constant-target",
        target="target",
        problem_type="regression",
    )

    assert result.errors
    assert "at least two distinct target values" in result.errors[0]
