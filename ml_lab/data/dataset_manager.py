"""
dataset_manager.py

Handles loading of built-in and custom datasets, and exposes basic
structural information about the currently loaded dataset.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

import config

logger = logging.getLogger("ml_lab.data")


class DatasetLoadError(Exception):
    """Raised when a dataset cannot be loaded or is unusable."""


@dataclass
class DatasetInfo:
    name: str
    path: str
    rows: int
    columns: int
    memory_mb: float
    numerical_columns: list
    categorical_columns: list
    datetime_columns: list
    missing_values: int
    duplicate_rows: int


class DatasetManager:
    """Loads datasets (built-in or custom CSV) and tracks basic metadata."""

    def __init__(self) -> None:
        self.name: str | None = None
        self.path: str | None = None
        self.dataframe: pd.DataFrame | None = None

    # ------------------------------------------------------------------ #
    # Loading
    # ------------------------------------------------------------------ #
    def load_builtin(self, key: str) -> pd.DataFrame:
        if key not in config.BUILTIN_DATASETS:
            raise DatasetLoadError(f"Unknown built-in dataset '{key}'.")
        entry = config.BUILTIN_DATASETS[key]
        path = Path(entry["path"])
        df = self._read_csv(path)
        self.name = key
        self.path = str(path)
        self.dataframe = df
        logger.info("Loaded built-in dataset '%s' with shape %s", key, df.shape)
        return df

    def load_custom(self, path_str: str) -> pd.DataFrame:
        path = Path(path_str.strip().strip('"').strip("'")).expanduser()
        df = self._read_csv(path)
        self.name = path.stem
        self.path = str(path)
        self.dataframe = df
        logger.info("Loaded custom dataset '%s' with shape %s", path, df.shape)
        return df

    @staticmethod
    def _read_csv(path: Path) -> pd.DataFrame:
        if not path.exists():
            raise DatasetLoadError(f"File not found: {path}")
        if not path.is_file():
            raise DatasetLoadError(f"Path is not a file: {path}")
        if path.suffix.lower() not in (".csv", ".tsv", ".txt"):
            raise DatasetLoadError(
                f"Unsupported file extension '{path.suffix}'. Expected a CSV file."
            )
        if path.stat().st_size == 0:
            raise DatasetLoadError("The file is empty.")

        sep = "\t" if path.suffix.lower() == ".tsv" else ","
        try:
            df = pd.read_csv(path, sep=sep, encoding="utf-8")
        except UnicodeDecodeError:
            try:
                df = pd.read_csv(path, sep=sep, encoding="latin-1")
            except Exception as exc:  # noqa: BLE001
                raise DatasetLoadError(f"Encoding error while reading file: {exc}") from exc
        except pd.errors.EmptyDataError as exc:
            raise DatasetLoadError("The CSV file has no columns/data.") from exc
        except pd.errors.ParserError as exc:
            raise DatasetLoadError(f"Malformed CSV file: {exc}") from exc
        except Exception as exc:  # noqa: BLE001
            raise DatasetLoadError(f"Could not read file: {exc}") from exc

        if df.shape[0] == 0:
            raise DatasetLoadError("The dataset contains no rows.")
        if df.shape[1] == 0:
            raise DatasetLoadError("The dataset contains no columns.")
        return df

    # ------------------------------------------------------------------ #
    # Inspection
    # ------------------------------------------------------------------ #
    def get_info(self) -> DatasetInfo:
        if self.dataframe is None:
            raise DatasetLoadError("No dataset is currently loaded.")
        df = self.dataframe
        numerical = df.select_dtypes(include="number").columns.tolist()
        categorical = df.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
        datetime_cols = df.select_dtypes(include=["datetime", "datetimetz"]).columns.tolist()

        return DatasetInfo(
            name=self.name or "unknown",
            path=self.path or "",
            rows=df.shape[0],
            columns=df.shape[1],
            memory_mb=round(df.memory_usage(deep=True).sum() / (1024 ** 2), 3),
            numerical_columns=numerical,
            categorical_columns=categorical,
            datetime_columns=datetime_cols,
            missing_values=int(df.isna().sum().sum()),
            duplicate_rows=int(df.duplicated().sum()),
        )

    def detect_target_candidates(self) -> list[str]:
        """Heuristically suggest likely target columns."""
        if self.dataframe is None:
            return []
        df = self.dataframe
        candidates = []
        name_hints = ("target", "label", "class", "churn", "price", "outcome", "status", "species", "y")
        for col in df.columns:
            lower = col.lower()
            if any(hint in lower for hint in name_hints):
                candidates.append(col)
        # Also suggest low-cardinality object/int columns as classification targets
        for col in df.columns:
            if col in candidates:
                continue
            nunique = df[col].nunique(dropna=True)
            if df[col].dtype == object and 1 < nunique <= 20:
                candidates.append(col)
        return candidates

    def column_type_breakdown(self) -> dict:
        """Classify each column as numerical / categorical / datetime / binary."""
        if self.dataframe is None:
            return {}
        df = self.dataframe
        breakdown = {"numerical": [], "categorical": [], "datetime": [], "binary": []}
        for col in df.columns:
            series = df[col]
            nunique = series.nunique(dropna=True)
            if nunique == 2:
                breakdown["binary"].append(col)
            elif pd.api.types.is_datetime64_any_dtype(series):
                breakdown["datetime"].append(col)
            elif pd.api.types.is_numeric_dtype(series):
                breakdown["numerical"].append(col)
            else:
                breakdown["categorical"].append(col)
        return breakdown
