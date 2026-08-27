"""
menu.py

The main menu controller. Routes the user to every feature area of
ML Lab and holds the orchestration logic that connects the UI to the
underlying data/model/evaluation modules. Business logic itself lives
in the respective modules; this file is primarily responsible for
navigation, prompting, and rendering results.
"""
from __future__ import annotations

import logging
import time

import pandas as pd
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn
from sklearn.model_selection import train_test_split

import config
from config import AppState
from ml_lab.academy.lessons import LESSONS
from ml_lab.academy.questions import all_categories, all_difficulties, get_questions
from ml_lab.data.cleaner import DataCleaner, DataQualityAnalyzer
from ml_lab.data.dataset_manager import DatasetLoadError, DatasetManager
from ml_lab.data.statistics import StatisticsEngine
from ml_lab.data.validator import DataValidator
from ml_lab.evaluation.evaluator import ModelEvaluator
from ml_lab.experiments.history import compare_experiments
from ml_lab.experiments.tracker import ExperimentTracker
from ml_lab.models.model_registry import get_models_for_task, get_model
from ml_lab.models.trainer import ModelTrainer
from ml_lab.models_storage.model_manager import ModelManager
from ml_lab.pipeline.auto_pipeline import AutoPipeline
from ml_lab.prediction.predictor import PredictionEngine
from ml_lab.preprocessing import feature_selection as fsmod
from ml_lab.reports.report_generator import ReportGenerator
from ml_lab.tuning.hyperparameter_tuning import HyperparameterTuner
from ml_lab.ui import display, prompts
from ml_lab.ui.display import console
from ml_lab.visualization import plots as viz

logger = logging.getLogger("ml_lab.menu")

MAIN_MENU_OPTIONS = [
    ("1", "Dataset Studio"),
    ("2", "Exploratory Data Analysis"),
    ("3", "Data Preprocessing"),
    ("4", "Feature Engineering & Selection"),
    ("5", "Model Training"),
    ("6", "Hyperparameter Tuning"),
    ("7", "Model Evaluation"),
    ("8", "Model Comparison"),
    ("9", "Prediction"),
    ("10", "Experiment Tracking"),
    ("11", "Model Manager"),
    ("12", "Generate Report"),
    ("13", "ML Academy"),
    ("14", "Run Complete Pipeline"),
    ("0", "Exit"),
]


class MenuController:
    def __init__(self) -> None:
        self.state = AppState()
        self.dataset_manager = DatasetManager()
        self.validator = DataValidator()
        self.cleaner = DataCleaner()
        self.quality_analyzer = DataQualityAnalyzer()
        self.stats_engine = StatisticsEngine()
        self.trainer = ModelTrainer()
        self.evaluator = ModelEvaluator()
        self.tuner = HyperparameterTuner()
        self.tracker = ExperimentTracker()
        self.model_manager = ModelManager()
        self.report_generator = ReportGenerator()

    # ------------------------------------------------------------------ #
    # Main loop
    # ------------------------------------------------------------------ #
    def run(self) -> None:
        while True:
            console.rule(style="dim")
            display.show_status_bar(self.state)
            display.show_menu("MAIN MENU", MAIN_MENU_OPTIONS)
            choice = prompts.ask_choice(
                "Select an option", [k for k, _ in MAIN_MENU_OPTIONS], default="1"
            )
            try:
                self._dispatch(choice)
            except KeyboardInterrupt:
                raise
            except Exception as exc:  # noqa: BLE001
                logger.exception("Unhandled error in menu dispatch")
                display.show_error(f"An unexpected error occurred: {exc}")
            if choice == "0":
                break

    def _dispatch(self, choice: str) -> None:
        handlers = {
            "1": self.dataset_studio,
            "2": self.eda_menu,
            "3": self.preprocessing_menu,
            "4": self.feature_engineering_menu,
            "5": self.model_training_menu,
            "6": self.hyperparameter_tuning_menu,
            "7": self.model_evaluation_menu,
            "8": self.model_comparison_menu,
            "9": self.prediction_menu,
            "10": self.experiment_tracking_menu,
            "11": self.model_manager_menu,
            "12": self.generate_report_menu,
            "13": self.academy_menu,
            "14": self.run_pipeline_menu,
            "0": self.exit_app,
        }
        handlers[choice]()

    def exit_app(self) -> None:
        console.print("[bold cyan]Thank you for using ML Lab. Goodbye![/bold cyan]")

    # ------------------------------------------------------------------ #
    # 1. Dataset Studio
    # ------------------------------------------------------------------ #
    def dataset_studio(self) -> None:
        options = [
            ("1", "Load Iris (built-in)"),
            ("2", "Load House Prices (built-in)"),
            ("3", "Load Customer Churn (built-in)"),
            ("4", "Load custom CSV"),
            ("5", "Preview dataset (head/tail)"),
            ("6", "Column information"),
            ("7", "Descriptive statistics"),
            ("8", "Data quality report"),
            ("9", "Set target column"),
            ("0", "Back to main menu"),
        ]
        display.show_menu("DATASET STUDIO", options)
        choice = prompts.ask_choice("Select an option", [k for k, _ in options], default="0")

        if choice == "1":
            self._load_builtin("iris")
        elif choice == "2":
            self._load_builtin("house_prices")
        elif choice == "3":
            self._load_builtin("customer_churn")
        elif choice == "4":
            self._load_custom()
        elif choice == "5":
            self._preview_dataset()
        elif choice == "6":
            self._column_info()
        elif choice == "7":
            self._descriptive_stats()
        elif choice == "8":
            self._data_quality_report()
        elif choice == "9":
            self._set_target()

    def _load_builtin(self, key: str) -> None:
        try:
            df = self.dataset_manager.load_builtin(key)
        except DatasetLoadError as exc:
            display.show_error(str(exc))
            return
        self._commit_loaded_dataset(key, df)

    def _load_custom(self) -> None:
        path = prompts.ask_text("Enter CSV path")
        try:
            df = self.dataset_manager.load_custom(path)
        except DatasetLoadError as exc:
            display.show_error(str(exc))
            return
        self._commit_loaded_dataset(self.dataset_manager.name, df)

    def _commit_loaded_dataset(self, name: str, df: pd.DataFrame) -> None:
        if df.shape[0] > config.LARGE_DATASET_ROW_THRESHOLD:
            if not prompts.ask_confirm(
                f"Dataset contains {df.shape[0]:,} rows. Some operations may take longer. Continue?"
            ):
                return

        self.state = AppState()
        self.state.dataset_name = name
        self.state.original_dataset = df.copy()
        self.state.working_dataset = df.copy()

        info = self.dataset_manager.get_info()
        display.show_success(
            f"Loaded '{name}' — {info.rows} rows, {info.columns} columns, "
            f"{info.memory_mb} MB, {info.missing_values} missing values, "
            f"{info.duplicate_rows} duplicate rows."
        )

        candidates = self.dataset_manager.detect_target_candidates()
        breakdown = self.dataset_manager.column_type_breakdown()
        console.print(
            f"[bold]Numerical:[/bold] {len(breakdown['numerical'])}  "
            f"[bold]Categorical:[/bold] {len(breakdown['categorical'])}  "
            f"[bold]Datetime:[/bold] {len(breakdown['datetime'])}  "
            f"[bold]Binary:[/bold] {len(breakdown['binary'])}"
        )
        if candidates:
            console.print(f"[bold]Potential target columns:[/bold] {', '.join(candidates)}")

        if name in config.BUILTIN_DATASETS:
            self.state.target_column = config.BUILTIN_DATASETS[name]["target"]
            console.print(f"[dim]Default target set to '{self.state.target_column}'.[/dim]")

    def _preview_dataset(self) -> None:
        if not self._require_dataset():
            return
        n = prompts.ask_int("How many rows to preview", default=5, minimum=1, maximum=100)
        mode = prompts.ask_choice("Head or tail?", ["head", "tail"], default="head")
        df = self.state.working_dataset
        subset = df.head(n) if mode == "head" else df.tail(n)
        display.show_dataframe(subset, title=f"{mode.title()} ({n} rows)")

    def _column_info(self) -> None:
        if not self._require_dataset():
            return
        info_df = self.stats_engine.column_info(self.state.working_dataset)
        display.show_dataframe(info_df, title="Column Information")

    def _descriptive_stats(self) -> None:
        if not self._require_dataset():
            return
        stats_df = self.stats_engine.describe(self.state.working_dataset)
        if stats_df.empty:
            display.show_warning("No numerical columns to describe.")
            return
        display.show_dataframe(stats_df.round(3).reset_index(), title="Descriptive Statistics")

    def _data_quality_report(self) -> None:
        if not self._require_dataset():
            return
        report = self.quality_analyzer.analyze(self.state.working_dataset)
        lines = []
        for issue in report.issues:
            icon = "\u2713" if issue.severity == "ok" else "\u26a0"
            lines.append(f"{icon} [{issue.category}] {issue.message}")
        display.console.print("\n".join(lines))

    def _set_target(self) -> None:
        if not self._require_dataset():
            return
        columns = self.state.working_dataset.columns.tolist()
        target = prompts.ask_choice("Select target column", columns)
        self.state.target_column = target
        display.show_success(f"Target column set to '{target}'.")

    # ------------------------------------------------------------------ #
    # 2. Exploratory Data Analysis
    # ------------------------------------------------------------------ #
    def eda_menu(self) -> None:
        if not self._require_dataset():
            return
        options = [
            ("1", "Histogram"),
            ("2", "Box plot"),
            ("3", "Scatter plot"),
            ("4", "Bar chart"),
            ("5", "Line chart"),
            ("6", "Count plot"),
            ("7", "Correlation heatmap + top correlations"),
            ("8", "Outlier detection"),
            ("0", "Back"),
        ]
        display.show_menu("EXPLORATORY DATA ANALYSIS", options)
        choice = prompts.ask_choice("Select an option", [k for k, _ in options], default="0")
        df = self.state.working_dataset
        numeric_cols = df.select_dtypes(include="number").columns.tolist()
        all_cols = df.columns.tolist()

        try:
            if choice == "1":
                col = prompts.ask_choice("Select numeric column", numeric_cols)
                path = viz.plot_histogram(df, col)
                display.show_success(f"Histogram saved to {path}")
            elif choice == "2":
                col = prompts.ask_choice("Select numeric column", numeric_cols)
                path = viz.plot_boxplot(df, col)
                display.show_success(f"Box plot saved to {path}")
            elif choice == "3":
                x = prompts.ask_choice("X-axis column", numeric_cols)
                y = prompts.ask_choice("Y-axis column", numeric_cols)
                path = viz.plot_scatter(df, x, y)
                display.show_success(f"Scatter plot saved to {path}")
            elif choice == "4":
                col = prompts.ask_choice("Select column", all_cols)
                path = viz.plot_bar(df, col)
                display.show_success(f"Bar chart saved to {path}")
            elif choice == "5":
                x = prompts.ask_choice("X-axis column", numeric_cols)
                y = prompts.ask_choice("Y-axis column", numeric_cols)
                path = viz.plot_line(df, x, y)
                display.show_success(f"Line chart saved to {path}")
            elif choice == "6":
                col = prompts.ask_choice("Select categorical column", all_cols)
                path = viz.plot_count(df, col)
                display.show_success(f"Count plot saved to {path}")
            elif choice == "7":
                self._correlation_analysis()
            elif choice == "8":
                self._outlier_detection()
        except Exception as exc:  # noqa: BLE001
            display.show_error(f"Could not generate plot: {exc}")

    def _correlation_analysis(self) -> None:
        df = self.state.working_dataset
        method = prompts.ask_choice("Correlation method", ["pearson", "spearman"], default="pearson")
        pairs = self.stats_engine.top_correlations(df, method=method, top_n=8)
        if not pairs:
            display.show_warning("Not enough numeric columns for correlation analysis.")
            return
        lines = [f"{a} \u2194 {b} : {v:.3f}" for a, b, v in pairs]
        console.print("[bold]TOP CORRELATIONS[/bold]\n" + "\n".join(lines))
        path = viz.plot_correlation_heatmap(df)
        if path:
            display.show_success(f"Correlation heatmap saved to {path}")

    def _outlier_detection(self) -> None:
        df = self.state.working_dataset
        numeric_cols = df.select_dtypes(include="number").columns.tolist()
        if not numeric_cols:
            display.show_warning("No numeric columns available.")
            return
        col = prompts.ask_choice("Select numeric column", numeric_cols)
        method = prompts.ask_choice("Method", ["iqr", "zscore"], default="iqr")
        if method == "iqr":
            mask, lower, upper = self.cleaner.detect_outliers_iqr(df, col)
        else:
            mask = self.cleaner.detect_outliers_zscore(df, col)
            lower = upper = None
        n_out = int(mask.sum())
        pct = n_out / len(df) * 100
        display.show_info(
            f"'{col}': {n_out} outliers detected ({pct:.2f}% of rows)"
            + (f", bounds=[{lower:.3f}, {upper:.3f}]" if lower is not None else "")
        )
        if n_out == 0:
            return
        action = prompts.ask_choice("Action", ["keep", "remove", "cap"], default="keep")
        if action == "remove":
            self.state.working_dataset = self.cleaner.remove_outliers(df, col, method=method)
            display.show_success(f"Removed {n_out} outlier rows from working dataset.")
        elif action == "cap":
            self.state.working_dataset = self.cleaner.cap_outliers(df, col)
            display.show_success(f"Capped (winsorized) outliers in '{col}'.")

    # ------------------------------------------------------------------ #
    # 3. Data Preprocessing / cleaning
    # ------------------------------------------------------------------ #
    def preprocessing_menu(self) -> None:
        if not self._require_dataset():
            return
        options = [
            ("1", "Remove duplicate rows"),
            ("2", "Remove columns"),
            ("3", "Rename columns"),
            ("4", "Handle missing values"),
            ("5", "Convert column data type"),
            ("6", "Remove constant columns"),
            ("7", "Reset to original dataset"),
            ("0", "Back"),
        ]
        display.show_menu("DATA PREPROCESSING / CLEANING", options)
        choice = prompts.ask_choice("Select an option", [k for k, _ in options], default="0")
        df = self.state.working_dataset

        if choice == "1":
            before = len(df)
            self.state.working_dataset = self.cleaner.remove_duplicates(df)
            display.show_success(f"Removed {before - len(self.state.working_dataset)} duplicate rows.")
        elif choice == "2":
            cols = prompts.ask_text("Enter column names to remove (comma-separated)")
            col_list = [c.strip() for c in cols.split(",") if c.strip()]
            if prompts.ask_confirm(f"Remove columns {col_list}?"):
                self.state.working_dataset = self.cleaner.remove_columns(df, col_list)
                display.show_success("Columns removed.")
        elif choice == "3":
            old = prompts.ask_choice("Column to rename", df.columns.tolist())
            new = prompts.ask_text(f"New name for '{old}'")
            self.state.working_dataset = self.cleaner.rename_columns(df, {old: new})
            display.show_success(f"Renamed '{old}' to '{new}'.")
        elif choice == "4":
            self._handle_missing_values_menu()
        elif choice == "5":
            col = prompts.ask_choice("Column to convert", df.columns.tolist())
            dtype = prompts.ask_choice("Target type", ["numeric", "string", "category", "datetime"])
            self.state.working_dataset = self.cleaner.convert_dtype(df, col, dtype)
            display.show_success(f"Converted '{col}' to {dtype}.")
        elif choice == "6":
            before_cols = set(df.columns)
            self.state.working_dataset = self.cleaner.remove_constant_columns(df)
            removed = before_cols - set(self.state.working_dataset.columns)
            display.show_success(f"Removed constant columns: {', '.join(removed) if removed else 'none found'}.")
        elif choice == "7":
            if prompts.ask_confirm("Reset working dataset to the originally loaded data?"):
                self.state.working_dataset = self.state.original_dataset.copy()
                display.show_success("Working dataset reset.")

    def _handle_missing_values_menu(self) -> None:
        df = self.state.working_dataset
        strategy = prompts.ask_choice(
            "Missing value strategy",
            ["mean", "median", "mode", "ffill", "bfill", "drop_rows", "drop_columns"],
        )
        cols_input = prompts.ask_text("Columns to apply to (comma-separated, blank = all)", default="")
        columns = [c.strip() for c in cols_input.split(",") if c.strip()] or None
        self.state.working_dataset = self.cleaner.handle_missing_values(df, strategy, columns)
        display.show_success(f"Applied '{strategy}' strategy for missing values.")

    # ------------------------------------------------------------------ #
    # 4. Feature engineering & selection
    # ------------------------------------------------------------------ #
    def feature_engineering_menu(self) -> None:
        if not self._require_dataset():
            return
        options = [
            ("1", "Log transform a column"),
            ("2", "Create ratio feature (A / B)"),
            ("3", "Create interaction feature (A * B)"),
            ("4", "Binning (equal-width buckets)"),
            ("5", "Polynomial feature (x^2)"),
            ("6", "Feature selection: variance threshold"),
            ("7", "Feature selection: correlation filter"),
            ("8", "Feature selection: SelectKBest"),
            ("9", "Feature selection: mutual information ranking"),
            ("0", "Back"),
        ]
        display.show_menu("FEATURE ENGINEERING & SELECTION", options)
        choice = prompts.ask_choice("Select an option", [k for k, _ in options], default="0")
        df = self.state.working_dataset
        numeric_cols = df.select_dtypes(include="number").columns.tolist()

        import numpy as np

        if choice == "1":
            col = prompts.ask_choice("Column to log-transform", numeric_cols)
            new_col = f"log_{col}"
            df = df.copy()
            df[new_col] = np.log1p(df[col].clip(lower=0))
            self.state.working_dataset = df
            display.show_success(f"Created '{new_col}' = log(1 + {col}). This compresses "
                                  f"large values and can help skewed distributions.")
        elif choice == "2":
            a = prompts.ask_choice("Numerator column", numeric_cols)
            b = prompts.ask_choice("Denominator column", numeric_cols)
            new_col = f"{a}_per_{b}"
            df = df.copy()
            df[new_col] = df[a] / df[b].replace(0, pd.NA)
            self.state.working_dataset = df
            display.show_success(f"Created ratio feature '{new_col}' = {a} / {b}.")
        elif choice == "3":
            a = prompts.ask_choice("First column", numeric_cols)
            b = prompts.ask_choice("Second column", numeric_cols)
            new_col = f"{a}_x_{b}"
            df = df.copy()
            df[new_col] = df[a] * df[b]
            self.state.working_dataset = df
            display.show_success(f"Created interaction feature '{new_col}' = {a} * {b}.")
        elif choice == "4":
            col = prompts.ask_choice("Column to bin", numeric_cols)
            bins = prompts.ask_int("Number of bins", default=4, minimum=2, maximum=20)
            new_col = f"{col}_bin"
            df = df.copy()
            df[new_col] = pd.cut(df[col], bins=bins, labels=False)
            self.state.working_dataset = df
            display.show_success(f"Created '{new_col}' with {bins} equal-width bins.")
        elif choice == "5":
            col = prompts.ask_choice("Column to square", numeric_cols)
            new_col = f"{col}_squared"
            df = df.copy()
            df[new_col] = df[col] ** 2
            self.state.working_dataset = df
            display.show_success(f"Created polynomial feature '{new_col}' = {col}^2.")
        elif choice in ("6", "7", "8", "9"):
            self._feature_selection(choice)

    def _feature_selection(self, choice: str) -> None:
        if not self.state.target_column:
            display.show_warning("Set a target column first (Dataset Studio > Set target column).")
            return
        df = self.state.working_dataset
        target = self.state.target_column
        X = df.drop(columns=[target])
        y = df[target]

        if choice == "6":
            threshold = prompts.ask_float("Variance threshold", default=0.0)
            _, kept = fsmod.variance_threshold_selection(X, threshold)
            console.print(f"[bold]Kept {len(kept)} features:[/bold] {', '.join(kept)}")
        elif choice == "7":
            threshold = prompts.ask_float("Correlation threshold", default=0.9)
            _, kept = fsmod.correlation_filter(X, threshold)
            console.print(f"[bold]Kept {len(kept)} features (correlation < {threshold}):[/bold] {', '.join(kept)}")
        elif choice == "8":
            k = prompts.ask_int("Number of top features to select", default=5, minimum=1)
            problem_type = "classification" if y.nunique() <= 20 and y.dtype == object else "regression"
            problem_type = prompts.ask_choice("Problem type", ["classification", "regression"], default=problem_type)
            _, selected = fsmod.select_k_best(X, y, k, problem_type)
            console.print(f"[bold]Top {len(selected)} features:[/bold] {', '.join(selected)}")
        elif choice == "9":
            problem_type = prompts.ask_choice("Problem type", ["classification", "regression"])
            ranking = fsmod.mutual_information_ranking(X, y, problem_type)
            if ranking.empty:
                display.show_warning("No numeric features available for mutual information.")
                return
            display.show_dataframe(ranking.round(4).reset_index().rename(
                columns={"index": "feature", 0: "mutual_information"}), title="Mutual Information Ranking")

    # ------------------------------------------------------------------ #
    # 5. Model Training
    # ------------------------------------------------------------------ #
    def model_training_menu(self) -> None:
        if not self._require_dataset():
            return
        if not self._ensure_target():
            return

        problem_type = prompts.ask_choice(
            "Problem type", ["classification", "regression", "clustering"],
            default=self.state.problem_type or "classification",
        )
        self.state.problem_type = problem_type

        df = self.state.working_dataset
        validation = self.validator.validate_target(df, self.state.target_column, problem_type)
        for w in validation.warnings:
            display.show_warning(w)
        if not validation.is_valid:
            for e in validation.errors:
                display.show_error(e)
            return

        feature_columns = [c for c in df.columns if c != self.state.target_column]
        leakage_warnings = self.validator.detect_leakage(df, self.state.target_column, feature_columns)
        for w in leakage_warnings:
            display.show_warning(w)

        numerical = df[feature_columns].select_dtypes(include="number").columns.tolist()
        categorical = df[feature_columns].select_dtypes(include=["object", "category"]).columns.tolist()

        models = get_models_for_task(problem_type)
        model_options = [(str(i + 1), spec.display_name) for i, spec in enumerate(models)]
        display.show_menu(f"AVAILABLE {problem_type.upper()} MODELS", model_options)
        idx = prompts.ask_choice("Select a model", [k for k, _ in model_options])
        spec = models[int(idx) - 1]

        console.print(
            f"[bold]MODEL CONFIGURATION[/bold]\nModel: {spec.display_name}\n"
            f"Target: {self.state.target_column}\n"
            f"Features: {', '.join(feature_columns)}"
        )

        if problem_type == "clustering":
            self._train_clustering(spec, df, feature_columns, numerical, categorical)
            return

        split_label = prompts.ask_split()
        test_size = config.SPLIT_OPTIONS[split_label]
        seed = prompts.ask_int("Random state", default=config.RANDOM_SEED)

        X = df[feature_columns]
        y = df[self.state.target_column]
        stratify = y if problem_type == "classification" else None
        try:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=seed, stratify=stratify
            )
        except ValueError as exc:
            display.show_warning(f"Could not stratify split ({exc}); splitting without stratification.")
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=seed)

        console.print(f"Training samples: {len(X_train)}\nTesting samples: {len(X_test)}")

        with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"),
                      BarColumn(), console=console) as progress:
            task = progress.add_task("Preparing data...", total=None)
            progress.update(task, description="Applying preprocessing & training model...")
            training_result = self.trainer.train(
                spec.key, X_train, y_train, numerical, categorical
            )
            progress.update(task, description="Evaluating model...")

        if problem_type == "classification":
            eval_result = self.evaluator.evaluate_classification(training_result.pipeline, X_test, y_test)
        else:
            eval_result = self.evaluator.evaluate_regression(training_result.pipeline, X_test, y_test)

        self.state.trained_model = training_result.pipeline
        self.state.trained_model_name = spec.display_name
        self.state.X_train, self.state.X_test = X_train, X_test
        self.state.y_train, self.state.y_test = y_train, y_test
        self.state.split_label = split_label
        self.state.feature_columns = feature_columns
        self.state.numerical_features = numerical
        self.state.categorical_features = categorical
        self.state.last_metrics = eval_result["metrics"]
        self.state.last_training_time = training_result.training_time_seconds

        importance = self.evaluator.feature_importance(training_result.pipeline)
        self.state.last_feature_importance = importance or {}

        display.show_success(f"'{spec.display_name}' trained in {training_result.training_time_seconds:.3f}s")
        display.show_key_value_table("Test Metrics", eval_result["metrics"])

        # Log experiment automatically
        exp = self.tracker.log_experiment(
            dataset=self.state.dataset_name, problem_type=problem_type,
            target=self.state.target_column, features=feature_columns,
            preprocessing={"scaler": "standard", "encoding": "onehot"},
            model=spec.display_name, parameters=training_result.params_used,
            metrics=eval_result["metrics"], training_time_seconds=training_result.training_time_seconds,
        )
        console.print(f"[dim]Experiment logged as {exp.experiment_id}[/dim]")

    def _train_clustering(self, spec, df, feature_columns, numerical, categorical) -> None:
        if spec.key == "kmeans":
            n_clusters = prompts.ask_int("Number of clusters", default=3, minimum=2, maximum=20)
            max_iter = prompts.ask_int("Max iterations", default=300, minimum=10)
            seed = prompts.ask_int("Random state", default=config.RANDOM_SEED)
            overrides = {"n_clusters": n_clusters, "max_iter": max_iter, "random_state": seed}
        else:
            overrides = {}

        X = df[feature_columns]
        training_result = self.trainer.train_clustering(
            spec.key, X, numerical, categorical, param_overrides=overrides
        )
        eval_result = self.evaluator.evaluate_clustering(training_result.pipeline, X)

        self.state.trained_model = training_result.pipeline
        self.state.trained_model_name = spec.display_name
        self.state.feature_columns = feature_columns
        self.state.numerical_features = numerical
        self.state.categorical_features = categorical
        self.state.last_metrics = eval_result["metrics"]

        display.show_success(f"'{spec.display_name}' trained in {training_result.training_time_seconds:.3f}s")
        display.show_key_value_table("Clustering Metrics", {
            k: v for k, v in eval_result["metrics"].items() if k != "cluster_sizes"
        })
        console.print(f"[bold]Cluster sizes:[/bold] {eval_result['metrics']['cluster_sizes']}")

    # ------------------------------------------------------------------ #
    # 6. Hyperparameter Tuning
    # ------------------------------------------------------------------ #
    def hyperparameter_tuning_menu(self) -> None:
        if not self._require_dataset() or not self._ensure_target():
            return
        problem_type = self.state.problem_type or prompts.ask_choice(
            "Problem type", ["classification", "regression"]
        )
        self.state.problem_type = problem_type
        df = self.state.working_dataset
        feature_columns = [c for c in df.columns if c != self.state.target_column]
        numerical = df[feature_columns].select_dtypes(include="number").columns.tolist()
        categorical = df[feature_columns].select_dtypes(include=["object", "category"]).columns.tolist()

        models = get_models_for_task(problem_type)
        model_options = [(str(i + 1), spec.display_name) for i, spec in enumerate(models)]
        display.show_menu(f"TUNABLE {problem_type.upper()} MODELS", model_options)
        idx = prompts.ask_choice("Select a model", [k for k, _ in model_options])
        spec = models[int(idx) - 1]

        if not spec.search_space:
            display.show_warning(f"'{spec.display_name}' has no tunable hyperparameters defined.")
            return

        method = prompts.ask_choice("Search method", ["grid", "random"], default="grid")
        folds = prompts.ask_cv_folds()

        X = df[feature_columns]
        y = df[self.state.target_column]

        console.print("[dim]Running hyperparameter search... this may take a moment.[/dim]")
        try:
            result = self.tuner.tune(
                spec.key, X, y, numerical, categorical, problem_type, method=method, cv_folds=folds
            )
        except Exception as exc:  # noqa: BLE001
            display.show_error(f"Tuning failed: {exc}")
            return

        console.print("[bold]BEST PARAMETERS[/bold]")
        for k, v in result.best_params.items():
            console.print(f"  {k}: {v}")
        console.print(f"\n[bold]BEST CV SCORE:[/bold] {result.best_score:.4f}")
        console.print(f"[dim]Search time: {result.search_time_seconds:.2f}s[/dim]")

        if prompts.ask_confirm("Use this tuned model as the current trained model?"):
            self.state.trained_model = result.best_estimator
            self.state.trained_model_name = f"{spec.display_name} (tuned)"
            self.state.feature_columns = feature_columns
            self.state.numerical_features = numerical
            self.state.categorical_features = categorical
            display.show_success("Tuned model set as the active trained model.")

    # ------------------------------------------------------------------ #
    # 7. Model Evaluation
    # ------------------------------------------------------------------ #
    def model_evaluation_menu(self) -> None:
        if not self._require_trained_model():
            return
        problem_type = self.state.problem_type
        if problem_type == "classification":
            eval_result = self.evaluator.evaluate_classification(
                self.state.trained_model, self.state.X_test, self.state.y_test
            )
            display.show_key_value_table("Classification Metrics", eval_result["metrics"])
            console.print(eval_result["report"])
            path = viz.plot_confusion_matrix(eval_result["confusion_matrix"], eval_result["labels"])
            display.show_success(f"Confusion matrix saved to {path}")
            if eval_result["y_proba"] is not None and len(eval_result["labels"]) == 2:
                roc_path = viz.plot_roc_curve(self.state.y_test, eval_result["y_proba"][:, 1])
                pr_path = viz.plot_precision_recall_curve(self.state.y_test, eval_result["y_proba"][:, 1])
                if roc_path:
                    display.show_success(f"ROC curve saved to {roc_path}")
                if pr_path:
                    display.show_success(f"Precision-recall curve saved to {pr_path}")
            if self.state.last_feature_importance:
                display.show_feature_importance(self.state.last_feature_importance)
                fi_path = viz.plot_feature_importance(self.state.last_feature_importance)
                display.show_success(f"Feature importance plot saved to {fi_path}")

        elif problem_type == "regression":
            eval_result = self.evaluator.evaluate_regression(
                self.state.trained_model, self.state.X_test, self.state.y_test
            )
            display.show_key_value_table("Regression Metrics", eval_result["metrics"])
            path1 = viz.plot_actual_vs_predicted(self.state.y_test, eval_result["y_pred"])
            path2 = viz.plot_residuals(self.state.y_test, eval_result["y_pred"])
            display.show_success(f"Actual vs predicted plot saved to {path1}")
            display.show_success(f"Residual plot saved to {path2}")
            if self.state.last_feature_importance:
                display.show_feature_importance(self.state.last_feature_importance)

        else:
            display.show_key_value_table("Clustering Metrics", {
                k: v for k, v in self.state.last_metrics.items() if k != "cluster_sizes"
            })

        if prompts.ask_confirm("Run cross-validation too?", default=False):
            self._run_cross_validation()

        if prompts.ask_confirm("Compare against a baseline model?", default=False) and problem_type in (
            "classification", "regression"
        ):
            self._baseline_comparison()

    def _run_cross_validation(self) -> None:
        folds = prompts.ask_cv_folds()
        X = pd.concat([self.state.X_train, self.state.X_test])
        y = pd.concat([self.state.y_train, self.state.y_test])
        result = self.evaluator.cross_validate(
            self.state.trained_model, X, y, folds=folds, problem_type=self.state.problem_type
        )
        console.print("[bold]CROSS VALIDATION[/bold]")
        for i, score in enumerate(result["fold_scores"], 1):
            console.print(f"Fold {i}: {score:.4f}")
        console.print(f"\nMean: {result['mean']:.4f}\nStd: {result['std']:.4f}")

    def _baseline_comparison(self) -> None:
        result = self.evaluator.baseline_comparison(
            self.state.X_train, self.state.y_train, self.state.X_test, self.state.y_test,
            self.state.problem_type,
        )
        key = result["primary_metric_key"]
        baseline_val = result["baseline_metrics"].get(key)
        model_val = self.state.last_metrics.get(key)
        console.print(f"Baseline {key}: {baseline_val:.4f}")
        console.print(f"{self.state.trained_model_name} {key}: {model_val:.4f}")
        if baseline_val is not None and model_val is not None:
            diff = (model_val - baseline_val) * 100
            console.print(f"Improvement: {diff:+.1f} percentage points")

    # ------------------------------------------------------------------ #
    # 8. Model Comparison
    # ------------------------------------------------------------------ #
    def model_comparison_menu(self) -> None:
        if not self._require_dataset() or not self._ensure_target():
            return
        problem_type = prompts.ask_choice(
            "Problem type", ["classification", "regression"],
            default=self.state.problem_type if self.state.problem_type in ("classification", "regression") else "classification",
        )
        self.state.problem_type = problem_type

        df = self.state.working_dataset
        target_validation = self.validator.validate_target(df, self.state.target_column, problem_type)
        for w in target_validation.warnings:
            display.show_warning(w)
        if not target_validation.is_valid:
            for e in target_validation.errors:
                display.show_error(e)
            return

        feature_columns = [c for c in df.columns if c != self.state.target_column]
        numerical = df[feature_columns].select_dtypes(include="number").columns.tolist()
        categorical = df[feature_columns].select_dtypes(include=["object", "category"]).columns.tolist()

        models = get_models_for_task(problem_type)
        model_options = [(str(i + 1), spec.display_name) for i, spec in enumerate(models)]
        display.show_menu("SELECT MODELS TO COMPARE", model_options)
        console.print("[dim]Enter comma-separated numbers, e.g. 1,3,4[/dim]")
        raw = prompts.ask_text("Models to compare", default=",".join(k for k, _ in model_options))
        selected_idx = [i.strip() for i in raw.split(",") if i.strip()]
        selected_specs = [models[int(i) - 1] for i in selected_idx if i.isdigit() and 1 <= int(i) <= len(models)]

        if not selected_specs:
            display.show_warning("No valid models selected.")
            return

        split_label = prompts.ask_split()
        test_size = config.SPLIT_OPTIONS[split_label]
        X = df[feature_columns]
        y = df[self.state.target_column]
        stratify = y if problem_type == "classification" else None
        try:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=config.RANDOM_SEED, stratify=stratify
            )
        except ValueError:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=config.RANDOM_SEED
            )

        results_table = []
        best_row = None
        best_metric_val = float("-inf")
        metric_key = "f1" if problem_type == "classification" else "r2"

        with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"), console=console) as progress:
            task = progress.add_task("Training models...", total=len(selected_specs))
            for spec in selected_specs:
                progress.update(task, description=f"Training {spec.display_name}...")
                training_result = self.trainer.train(spec.key, X_train, y_train, numerical, categorical)
                if problem_type == "classification":
                    eval_result = self.evaluator.evaluate_classification(training_result.pipeline, X_test, y_test)
                else:
                    eval_result = self.evaluator.evaluate_regression(training_result.pipeline, X_test, y_test)
                row = {"Model": spec.display_name, **eval_result["metrics"]}
                results_table.append(row)
                if (eval_result["metrics"].get(metric_key) or float("-inf")) > best_metric_val:
                    best_metric_val = eval_result["metrics"].get(metric_key) or float("-inf")
                    best_row = row
                progress.advance(task)

        self.state.comparison_results = results_table
        df_results = pd.DataFrame(results_table).round(4)
        display.show_dataframe(df_results, title="MODEL COMPARISON")
        if best_row:
            console.print(
                f"\n[bold green]Best model by {metric_key}: "
                f"{best_row['Model']} ({best_row.get(metric_key):.4f})[/bold green]"
            )
        console.print("[dim]Note: no single model is universally best; results are specific "
                      "to this dataset and split.[/dim]")

    # ------------------------------------------------------------------ #
    # 9. Prediction
    # ------------------------------------------------------------------ #
    def prediction_menu(self) -> None:
        if not self._require_trained_model():
            return
        if self.state.problem_type == "clustering":
            display.show_warning("Custom prediction is not supported for clustering models.")
            return

        engine = PredictionEngine(
            self.state.trained_model, self.state.feature_columns,
            self.state.numerical_features, self.state.categorical_features,
        )
        console.print("[bold]CUSTOM PREDICTION[/bold]")
        values = {}
        for col, kind in engine.expected_inputs():
            if kind == "numeric":
                values[col] = prompts.ask_float(col)
            else:
                values[col] = prompts.ask_text(col)

        result = engine.predict(values)
        if self.state.problem_type == "classification":
            console.print(f"\n[bold]Predicted class:[/bold] {result['prediction']}")
            if result["confidence"] is not None:
                console.print(f"[bold]Confidence:[/bold] {result['confidence'] * 100:.1f}%")
        else:
            console.print(f"\n[bold]Predicted value:[/bold] {result['prediction']:.4f}")

    # ------------------------------------------------------------------ #
    # 10. Experiment Tracking
    # ------------------------------------------------------------------ #
    def experiment_tracking_menu(self) -> None:
        options = [
            ("1", "List experiments"),
            ("2", "View experiment"),
            ("3", "Compare experiments"),
            ("4", "Delete experiment"),
            ("0", "Back"),
        ]
        display.show_menu("EXPERIMENT TRACKING", options)
        choice = prompts.ask_choice("Select an option", [k for k, _ in options], default="0")
        experiments = self.tracker.list_experiments()

        if choice == "1":
            if not experiments:
                display.show_info("No experiments logged yet.")
                return
            rows = [{"id": e.experiment_id, "model": e.model, "dataset": e.dataset,
                     "timestamp": e.timestamp} for e in experiments]
            display.show_dataframe(pd.DataFrame(rows), title="Experiments")
        elif choice == "2":
            if not experiments:
                display.show_info("No experiments logged yet.")
                return
            ids = [e.experiment_id for e in experiments]
            exp_id = prompts.ask_choice("Experiment ID", ids)
            exp = self.tracker.get_experiment(exp_id)
            console.print_json(data=exp.__dict__)
        elif choice == "3":
            if not experiments:
                display.show_info("No experiments logged yet.")
                return
            metric_key = prompts.ask_text("Metric to compare by", default="accuracy")
            comparison = compare_experiments(experiments, metric_key)
            display.show_dataframe(pd.DataFrame(comparison), title=f"Comparison by {metric_key}")
        elif choice == "4":
            if not experiments:
                display.show_info("No experiments logged yet.")
                return
            ids = [e.experiment_id for e in experiments]
            exp_id = prompts.ask_choice("Experiment ID to delete", ids)
            if prompts.ask_confirm(f"Delete {exp_id}? This cannot be undone."):
                self.tracker.delete_experiment(exp_id)
                display.show_success(f"{exp_id} deleted.")

    # ------------------------------------------------------------------ #
    # 11. Model Manager
    # ------------------------------------------------------------------ #
    def model_manager_menu(self) -> None:
        options = [
            ("1", "Save current trained model"),
            ("2", "List saved models"),
            ("3", "Load a saved model"),
            ("4", "Delete a saved model"),
            ("0", "Back"),
        ]
        display.show_menu("MODEL MANAGER", options)
        choice = prompts.ask_choice("Select an option", [k for k, _ in options], default="0")

        if choice == "1":
            if not self._require_trained_model():
                return
            model_path, meta_path = self.model_manager.save_model(
                pipeline=self.state.trained_model,
                model_name=self.state.trained_model_name,
                dataset=self.state.dataset_name,
                target=self.state.target_column or "",
                features=self.state.feature_columns,
                parameters={},
                metrics=self.state.last_metrics,
                problem_type=self.state.problem_type or "",
            )
            display.show_success(f"Model saved to {model_path}")
        elif choice == "2":
            models = self.model_manager.list_models()
            if not models:
                display.show_info("No saved models found.")
                return
            rows = [{"name": m["model_name"], "dataset": m["dataset"],
                     "created_at": m["created_at"]} for m in models]
            display.show_dataframe(pd.DataFrame(rows), title="Saved Models")
        elif choice == "3":
            models = self.model_manager.list_models()
            if not models:
                display.show_info("No saved models found.")
                return
            names = [m["model_path"] for m in models]
            labels = [f"{m['model_name']} ({m['created_at']})" for m in models]
            selection = prompts.ask_choice("Select a model", labels)
            chosen = models[labels.index(selection)]
            pipeline = self.model_manager.load_model(chosen["model_path"])
            self.state.trained_model = pipeline
            self.state.trained_model_name = chosen["model_name"]
            self.state.target_column = chosen["target"]
            self.state.feature_columns = chosen["features"]
            self.state.problem_type = chosen["problem_type"]
            self.state.last_metrics = chosen["metrics"]
            display.show_success(f"Loaded '{chosen['model_name']}' as the active model.")
        elif choice == "4":
            models = self.model_manager.list_models()
            if not models:
                display.show_info("No saved models found.")
                return
            labels = [f"{m['model_name']} ({m['created_at']})" for m in models]
            selection = prompts.ask_choice("Select a model to delete", labels)
            chosen = models[labels.index(selection)]
            if prompts.ask_confirm("Delete this model permanently?"):
                self.model_manager.delete_model(chosen["model_path"])
                display.show_success("Model deleted.")

    # ------------------------------------------------------------------ #
    # 12. Generate Report
    # ------------------------------------------------------------------ #
    def generate_report_menu(self) -> None:
        if not self._require_trained_model():
            return
        fmt = prompts.ask_choice("Report format", ["md", "txt"], default="md")
        baseline = None
        if self.state.problem_type in ("classification", "regression") and self.state.X_test is not None:
            baseline = self.evaluator.baseline_comparison(
                self.state.X_train, self.state.y_train, self.state.X_test, self.state.y_test,
                self.state.problem_type,
            )["baseline_metrics"]

        path = self.report_generator.generate(
            dataset_name=self.state.dataset_name or "unknown",
            problem_type=self.state.problem_type or "unknown",
            model_name=self.state.trained_model_name or "unknown",
            n_train=len(self.state.X_train) if self.state.X_train is not None else 0,
            n_test=len(self.state.X_test) if self.state.X_test is not None else 0,
            features=self.state.feature_columns,
            metrics=self.state.last_metrics,
            parameters={},
            feature_importance=self.state.last_feature_importance or None,
            baseline_metrics=baseline,
            fmt=fmt,
        )
        display.show_success(f"Report generated at {path}")

    # ------------------------------------------------------------------ #
    # 13. ML Academy
    # ------------------------------------------------------------------ #
    def academy_menu(self) -> None:
        options = [("1", "Browse lessons"), ("2", "Take a quiz"), ("0", "Back")]
        display.show_menu("ML ACADEMY", options)
        choice = prompts.ask_choice("Select an option", [k for k, _ in options], default="0")
        if choice == "1":
            self._browse_lessons()
        elif choice == "2":
            self._take_quiz()

    def _browse_lessons(self) -> None:
        lesson_options = [(str(i + 1), lesson.title) for i, lesson in enumerate(LESSONS)]
        display.show_menu("LESSONS", lesson_options)
        idx = prompts.ask_choice("Select a lesson", [k for k, _ in lesson_options])
        lesson = LESSONS[int(idx) - 1]
        console.print(f"\n[bold cyan]{lesson.title}[/bold cyan]\n")
        console.print(f"[bold]Simple explanation:[/bold] {lesson.simple_explanation}\n")
        console.print(f"[bold]Technical explanation:[/bold] {lesson.technical_explanation}\n")
        console.print(f"[bold]Example:[/bold] {lesson.example}\n")
        console.print(f"[bold]When to use:[/bold] {lesson.when_to_use}\n")
        console.print(f"[bold]Common mistakes:[/bold] {lesson.common_mistakes}\n")
        console.print(f"[bold]Interview question:[/bold] {lesson.interview_question}\n")

    def _take_quiz(self) -> None:
        categories = all_categories()
        cat_choice = prompts.ask_choice("Category", ["All"] + categories, default="All")
        diff_choice = prompts.ask_choice("Difficulty", ["All"] + all_difficulties(), default="All")
        n = prompts.ask_int("Number of questions", default=5, minimum=1, maximum=20)

        cats = None if cat_choice == "All" else [cat_choice]
        diffs = None if diff_choice == "All" else [diff_choice]
        questions = get_questions(categories=cats, difficulties=diffs, n=n)

        if not questions:
            display.show_warning("No questions match that filter.")
            return

        score = 0
        for i, q in enumerate(questions, 1):
            console.print(f"\n[bold]Q{i}. {q.question}[/bold]")
            option_labels = [f"{chr(65+j)}) {opt}" for j, opt in enumerate(q.options)]
            for label in option_labels:
                console.print(label)
            answer = prompts.ask_choice(
                "Your answer", [chr(65 + j) for j in range(len(q.options))]
            )
            correct_letter = chr(65 + q.correct_index)
            if answer == correct_letter:
                score += 1
                display.show_success(f"Correct! {q.explanation}")
            else:
                display.show_error(f"Incorrect. Correct answer: {correct_letter}) "
                                    f"{q.options[q.correct_index]}. {q.explanation}")

        pct = score / len(questions) * 100
        console.print(f"\n[bold]FINAL SCORE:[/bold] {score}/{len(questions)} ({pct:.1f}%)")

    # ------------------------------------------------------------------ #
    # 14. Run Complete Pipeline
    # ------------------------------------------------------------------ #
    def run_pipeline_menu(self) -> None:
        if not self._require_dataset() or not self._ensure_target():
            return
        problem_type = prompts.ask_choice(
            "Problem type", ["classification", "regression"], default="classification"
        )
        split_label = prompts.ask_split()
        test_size = config.SPLIT_OPTIONS[split_label]

        console.print("[bold]ML PIPELINE[/bold]")
        pipeline = AutoPipeline(progress_callback=lambda msg: console.print(f"[green][\u2713][/green] {msg}"))
        result = pipeline.run(
            df=self.state.working_dataset,
            dataset_name=self.state.dataset_name or "dataset",
            target=self.state.target_column,
            problem_type=problem_type,
            test_size=test_size,
        )

        if result.errors:
            for e in result.errors:
                display.show_error(e)
            return

        console.print(f"\n[bold green]Pipeline completed successfully.[/bold green]")
        console.print(f"Best model: {result.best_model_name}")
        display.show_key_value_table("Best Model Metrics", result.best_metrics)
        console.print(f"Model saved to: {result.model_path}")
        console.print(f"Report saved to: {result.report_path}")
        console.print(f"Experiment logged as: {result.experiment_id}")

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #
    def _require_dataset(self) -> bool:
        if self.state.working_dataset is None:
            display.show_error("No dataset loaded. Please load one from Dataset Studio first.")
            return False
        return True

    def _require_trained_model(self) -> bool:
        if self.state.trained_model is None:
            display.show_error("No trained model available. Train a model first.")
            return False
        return True

    def _ensure_target(self) -> bool:
        if not self._require_dataset():
            return False
        if not self.state.target_column:
            columns = self.state.working_dataset.columns.tolist()
            target = prompts.ask_choice("Select target column", columns)
            self.state.target_column = target
        return True
