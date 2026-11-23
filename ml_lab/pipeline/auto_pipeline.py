"""
auto_pipeline.py

Runs a complete, automated end-to-end ML workflow: load -> validate ->
clean -> preprocess -> split -> train baseline models -> evaluate ->
compare -> select best -> save model -> log experiment -> generate report.

Every step reports real, calculated status back to the caller so the UI
can render live progress.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import pandas as pd
from sklearn.model_selection import train_test_split

import config
from ml_lab.data.cleaner import DataCleaner
from ml_lab.data.validator import DataValidator
from ml_lab.evaluation.evaluator import ModelEvaluator
from ml_lab.experiments.tracker import ExperimentTracker
from ml_lab.models.trainer import ModelTrainer
from ml_lab.models_storage.model_manager import ModelManager
from ml_lab.reports.report_generator import ReportGenerator


@dataclass
class PipelineResult:
    steps_completed: list = field(default_factory=list)
    best_model_name: str = ""
    best_metrics: dict = field(default_factory=dict)
    comparison_table: list = field(default_factory=list)
    model_path: str = ""
    report_path: str = ""
    experiment_id: str = ""
    errors: list = field(default_factory=list)


DEFAULT_CLASSIFICATION_CANDIDATES = [
    "logistic_regression", "decision_tree_classifier", "random_forest_classifier",
]
DEFAULT_REGRESSION_CANDIDATES = [
    "linear_regression", "decision_tree_regressor", "random_forest_regressor",
]


class AutoPipeline:
    def __init__(self, progress_callback: Callable[[str], None] | None = None):
        self.progress_callback = progress_callback or (lambda msg: None)
        self.cleaner = DataCleaner()
        self.validator = DataValidator()
        self.trainer = ModelTrainer()
        self.evaluator = ModelEvaluator()
        self.tracker = ExperimentTracker()
        self.model_manager = ModelManager()
        self.report_generator = ReportGenerator()

    def _report(self, result: PipelineResult, message: str) -> None:
        result.steps_completed.append(message)
        self.progress_callback(message)

    def run(
        self,
        df: pd.DataFrame,
        dataset_name: str,
        target: str,
        problem_type: str,
        test_size: float = 0.2,
        primary_metric: str | None = None,
        candidate_models: list[str] | None = None,
    ) -> PipelineResult:
        result = PipelineResult()

        # 1. Dataset already loaded by caller
        self._report(result, "Dataset loaded")

        # 2. Validate
        validation = self.validator.validate_dataset(df)
        if not validation.is_valid:
            result.errors.extend(validation.errors)
            return result
        target_validation = self.validator.validate_target(df, target, problem_type)
        if not target_validation.is_valid:
            result.errors.extend(target_validation.errors)
            return result
        self._report(result, "Data validated")

        # 3. Clean: drop duplicates + constant columns, impute missing values
        working = self.cleaner.remove_duplicates(df)
        constant_features = [
            column
            for column in working.columns
            if column != target and working[column].nunique(dropna=True) <= 1
        ]
        working = self.cleaner.remove_columns(working, constant_features)
        working = self.cleaner.handle_missing_values(working, strategy="median")
        working = self.cleaner.handle_missing_values(working, strategy="mode")
        self._report(result, "Missing values handled")

        # 4. Identify features
        feature_df = working.drop(columns=[target])
        numerical_features = feature_df.select_dtypes(include="number").columns.tolist()
        categorical_features = [
            column
            for column in feature_df.columns
            if (
                pd.api.types.is_object_dtype(feature_df[column].dtype)
                or pd.api.types.is_string_dtype(feature_df[column].dtype)
                or isinstance(feature_df[column].dtype, pd.CategoricalDtype)
                or pd.api.types.is_bool_dtype(feature_df[column].dtype)
            )
        ]
        feature_columns = numerical_features + categorical_features
        self._report(result, "Features processed")

        if not feature_columns:
            result.errors.append("No usable feature columns were found after cleaning.")
            return result

        X = working[feature_columns]
        y = working[target]

        # 5. Split
        stratify = y if problem_type == "classification" else None
        try:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=config.RANDOM_SEED, stratify=stratify
            )
        except ValueError:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=config.RANDOM_SEED
            )
        self._report(result, "Train/test split")

        # 6-8. Train baseline models
        candidates = candidate_models or (
            DEFAULT_CLASSIFICATION_CANDIDATES if problem_type == "classification"
            else DEFAULT_REGRESSION_CANDIDATES
        )

        trained = []
        for model_key in candidates:
            try:
                training_result = self.trainer.train(
                    model_key, X_train, y_train, numerical_features, categorical_features
                )
                if problem_type == "classification":
                    eval_result = self.evaluator.evaluate_classification(
                        training_result.pipeline, X_test, y_test
                    )
                else:
                    eval_result = self.evaluator.evaluate_regression(
                        training_result.pipeline, X_test, y_test
                    )
                trained.append((training_result, eval_result))
                self._report(result, f"{training_result.model_spec.display_name} trained")
            except Exception as exc:  # noqa: BLE001
                result.errors.append(f"Failed to train {model_key}: {exc}")

        if not trained:
            result.errors.append("No models were successfully trained.")
            return result

        self._report(result, "Models evaluated")

        # 9. Compare and select best
        metric_key = primary_metric or ("f1" if problem_type == "classification" else "r2")
        comparison_table = []
        for training_result, eval_result in trained:
            row = {"model": training_result.model_spec.display_name, **eval_result["metrics"]}
            comparison_table.append(row)
        result.comparison_table = comparison_table

        best_training, best_eval = max(
            trained, key=lambda pair: pair[1]["metrics"].get(metric_key) or float("-inf")
        )
        result.best_model_name = best_training.model_spec.display_name
        result.best_metrics = best_eval["metrics"]
        self._report(result, "Best model selected")

        # 10. Save model
        feature_importance = self.evaluator.feature_importance(best_training.pipeline)
        model_path, _ = self.model_manager.save_model(
            pipeline=best_training.pipeline,
            model_name=best_training.model_spec.display_name,
            dataset=dataset_name,
            target=target,
            features=feature_columns,
            parameters=best_training.params_used,
            metrics=best_eval["metrics"],
            problem_type=problem_type,
        )
        result.model_path = str(model_path)
        self._report(result, "Model saved")

        # 11. Log experiment
        experiment = self.tracker.log_experiment(
            dataset=dataset_name,
            problem_type=problem_type,
            target=target,
            features=feature_columns,
            preprocessing={"scaler": "standard", "encoding": "onehot"},
            model=best_training.model_spec.display_name,
            parameters=best_training.params_used,
            metrics=best_eval["metrics"],
            training_time_seconds=best_training.training_time_seconds,
            notes="Generated by AutoPipeline",
        )
        result.experiment_id = experiment.experiment_id
        self._report(result, "Experiment logged")

        # 12. Generate report
        report_path = self.report_generator.generate(
            dataset_name=dataset_name,
            problem_type=problem_type,
            model_name=best_training.model_spec.display_name,
            n_train=len(X_train),
            n_test=len(X_test),
            features=feature_columns,
            metrics=best_eval["metrics"],
            parameters=best_training.params_used,
            feature_importance=feature_importance,
        )
        result.report_path = str(report_path)
        self._report(result, "Report generated")

        return result
