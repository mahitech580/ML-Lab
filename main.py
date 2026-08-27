#!/usr/bin/env python3
"""
main.py

Entry point for ML Lab — an interactive, terminal-based Machine
Learning laboratory built entirely in Python.

Usage:
    python main.py                              Interactive mode
    python main.py --dataset datasets/iris.csv  Load a dataset and show info, then enter interactive mode
    python main.py --pipeline datasets/iris.csv  Run the full automated pipeline on a dataset and exit
    python main.py --info datasets/iris.csv      Print dataset info and exit
"""
from __future__ import annotations

import argparse
import logging
import sys

import config


def configure_logging() -> None:
    logging.basicConfig(
        filename=str(config.LOG_FILE),
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )


def check_dependencies() -> bool:
    required = ["numpy", "pandas", "sklearn", "scipy", "matplotlib", "seaborn", "joblib", "rich"]
    missing = []
    for module in required:
        try:
            __import__(module)
        except ImportError:
            missing.append(module)
    if missing:
        print(f"Missing required packages: {', '.join(missing)}")
        print("Install them with: pip install -r requirements.txt")
        return False
    return True


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ML Lab", description="Interactive Machine Learning Laboratory (Python-only, terminal UI)."
    )
    parser.add_argument("--dataset", metavar="CSV_PATH", help="Load a CSV dataset on startup.")
    parser.add_argument("--pipeline", metavar="CSV_PATH",
                         help="Run the full automated pipeline on the given CSV and exit.")
    parser.add_argument("--info", metavar="CSV_PATH", help="Print dataset info for the given CSV and exit.")
    parser.add_argument("--target", metavar="COLUMN", help="Target column (used with --pipeline).")
    parser.add_argument("--problem-type", choices=["classification", "regression"],
                         default="classification", help="Problem type (used with --pipeline).")
    return parser


def run_info_mode(path: str) -> int:
    from ml_lab.data.dataset_manager import DatasetManager, DatasetLoadError
    from ml_lab.ui import display

    manager = DatasetManager()
    try:
        manager.load_custom(path)
    except DatasetLoadError as exc:
        display.show_error(str(exc))
        return 1
    info = manager.get_info()
    display.console.print(f"[bold]Dataset:[/bold] {info.name}")
    display.console.print(f"[bold]Rows:[/bold] {info.rows}")
    display.console.print(f"[bold]Columns:[/bold] {info.columns}")
    display.console.print(f"[bold]Memory (MB):[/bold] {info.memory_mb}")
    display.console.print(f"[bold]Numerical columns:[/bold] {', '.join(info.numerical_columns)}")
    display.console.print(f"[bold]Categorical columns:[/bold] {', '.join(info.categorical_columns)}")
    display.console.print(f"[bold]Missing values:[/bold] {info.missing_values}")
    display.console.print(f"[bold]Duplicate rows:[/bold] {info.duplicate_rows}")
    return 0


def run_pipeline_mode(path: str, target: str | None, problem_type: str) -> int:
    from ml_lab.data.dataset_manager import DatasetManager, DatasetLoadError
    from ml_lab.pipeline.auto_pipeline import AutoPipeline
    from ml_lab.ui import display

    manager = DatasetManager()
    try:
        df = manager.load_custom(path)
    except DatasetLoadError as exc:
        display.show_error(str(exc))
        return 1

    if not target:
        candidates = manager.detect_target_candidates()
        if not candidates:
            display.show_error("Could not auto-detect a target column. Please pass --target.")
            return 1
        target = candidates[0]
        display.console.print(f"[dim]Auto-selected target column: {target}[/dim]")

    pipeline = AutoPipeline(progress_callback=lambda msg: display.console.print(f"[\u2713] {msg}"))
    result = pipeline.run(df=df, dataset_name=manager.name, target=target, problem_type=problem_type)

    if result.errors:
        for e in result.errors:
            display.show_error(e)
        return 1

    display.show_success(f"Pipeline completed. Best model: {result.best_model_name}")
    display.show_key_value_table("Best Model Metrics", result.best_metrics)
    display.console.print(f"Model saved to: {result.model_path}")
    display.console.print(f"Report saved to: {result.report_path}")
    return 0


def run_interactive_mode(initial_dataset: str | None) -> int:
    from ml_lab.ui.menu import MenuController
    from ml_lab.ui import display

    display.show_banner()
    controller = MenuController()

    if initial_dataset:
        try:
            df = controller.dataset_manager.load_custom(initial_dataset)
            controller._commit_loaded_dataset(controller.dataset_manager.name, df)
        except Exception as exc:  # noqa: BLE001
            display.show_error(f"Could not load initial dataset: {exc}")

    try:
        controller.run()
    except KeyboardInterrupt:
        display.console.print("\n[yellow]Interrupted. Exiting ML Lab.[/yellow]")
    except Exception as exc:  # noqa: BLE001
        logging.getLogger("ml_lab.main").exception("Fatal error in interactive mode")
        display.show_error(f"A fatal error occurred and ML Lab must close: {exc}")
        return 1
    return 0


def main() -> int:
    configure_logging()
    logger = logging.getLogger("ml_lab.main")
    logger.info("ML Lab starting up (version %s)", config.APP_VERSION)

    if not check_dependencies():
        return 1

    parser = build_arg_parser()
    args = parser.parse_args()

    if args.info:
        return run_info_mode(args.info)
    if args.pipeline:
        return run_pipeline_mode(args.pipeline, args.target, args.problem_type)

    return run_interactive_mode(args.dataset)


if __name__ == "__main__":
    sys.exit(main())
