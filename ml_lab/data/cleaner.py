"""
cleaner.py

Data-quality analysis and data-cleaning operations. Cleaning operations
always work on an explicit copy of the dataframe and return a new
dataframe rather than mutating in place, so the caller decides when to
commit changes to the working dataset.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass
class QualityIssue:
    category: str
    message: str
    severity: str  # "ok" | "warning"


@dataclass
class QualityReport:
    issues: list = field(default_factory=list)

    def add(self, category: str, message: str, severity: str = "warning") -> None:
        self.issues.append(QualityIssue(category, message, severity))

    def warnings(self) -> list[QualityIssue]:
        return [i for i in self.issues if i.severity == "warning"]


class DataQualityAnalyzer:
    """Produces an automated data-quality report with recommendations."""

    def analyze(self, df: pd.DataFrame) -> QualityReport:
        report = QualityReport()

        # Missing values
        missing = df.isna().sum()
        missing = missing[missing > 0]
        if missing.empty:
            report.add("missing_values", "No missing values detected.", "ok")
        else:
            for col, count in missing.items():
                pct = count / len(df) * 100
                report.add(
                    "missing_values",
                    f"'{col}': {count} missing values ({pct:.1f}%). "
                    f"Consider imputation (median/mode) or dropping if severe.",
                )

        # Duplicates
        dup_count = df.duplicated().sum()
        if dup_count == 0:
            report.add("duplicates", "No duplicate rows.", "ok")
        else:
            report.add("duplicates", f"{dup_count} duplicate rows found. Consider removing them.")

        # Constant columns
        constant_cols = [c for c in df.columns if df[c].nunique(dropna=True) <= 1]
        if constant_cols:
            report.add(
                "constant_columns",
                f"Constant columns (no variance): {', '.join(constant_cols)}. "
                "These carry no predictive signal and can be dropped.",
            )
        else:
            report.add("constant_columns", "No constant columns.", "ok")

        # High cardinality
        high_card = [
            c for c in df.select_dtypes(include="object").columns
            if df[c].nunique(dropna=True) > max(50, int(0.5 * len(df)))
        ]
        if high_card:
            report.add(
                "high_cardinality",
                f"High-cardinality categorical columns: {', '.join(high_card)}. "
                "Consider dropping, hashing, or grouping rare categories.",
            )

        # Outliers (IQR based) on numeric columns
        numeric_cols = df.select_dtypes(include="number").columns
        outlier_cols = []
        for col in numeric_cols:
            series = df[col].dropna()
            if series.empty:
                continue
            q1, q3 = series.quantile(0.25), series.quantile(0.75)
            iqr = q3 - q1
            if iqr == 0:
                continue
            lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
            n_out = ((series < lower) | (series > upper)).sum()
            if n_out > 0:
                outlier_cols.append((col, n_out))
        if outlier_cols:
            for col, n_out in outlier_cols:
                report.add(
                    "outliers",
                    f"'{col}' contains {n_out} potential outlier(s) (IQR method).",
                )
        else:
            report.add("outliers", "No significant outliers detected.", "ok")

        # Zero-variance / near-zero-variance numeric features
        for col in numeric_cols:
            std = df[col].std()
            if std is not None and std == 0:
                report.add("zero_variance", f"'{col}' has zero variance.")

        return report


class DataCleaner:
    """Cleaning operations that operate on copies of a dataframe."""

    def remove_duplicates(self, df: pd.DataFrame) -> pd.DataFrame:
        return df.drop_duplicates().reset_index(drop=True)

    def remove_columns(self, df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
        return df.drop(columns=[c for c in columns if c in df.columns])

    def remove_rows(self, df: pd.DataFrame, indices: list[int]) -> pd.DataFrame:
        return df.drop(index=[i for i in indices if i in df.index]).reset_index(drop=True)

    def rename_columns(self, df: pd.DataFrame, mapping: dict) -> pd.DataFrame:
        return df.rename(columns=mapping)

    def remove_constant_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        constant_cols = [c for c in df.columns if df[c].nunique(dropna=True) <= 1]
        return df.drop(columns=constant_cols)

    def convert_dtype(self, df: pd.DataFrame, column: str, dtype: str) -> pd.DataFrame:
        df = df.copy()
        if dtype == "numeric":
            df[column] = pd.to_numeric(df[column], errors="coerce")
        elif dtype == "string":
            df[column] = df[column].astype(str)
        elif dtype == "category":
            df[column] = df[column].astype("category")
        elif dtype == "datetime":
            df[column] = pd.to_datetime(df[column], errors="coerce")
        return df

    def handle_missing_values(
        self, df: pd.DataFrame, strategy: str, columns: list[str] | None = None
    ) -> pd.DataFrame:
        """
        strategy: 'mean' | 'median' | 'mode' | 'ffill' | 'bfill' | 'drop_rows' | 'drop_columns'
        """
        df = df.copy()
        columns = columns or df.columns.tolist()

        if strategy == "drop_rows":
            return df.dropna(subset=columns).reset_index(drop=True)
        if strategy == "drop_columns":
            cols_with_na = [c for c in columns if df[c].isna().any()]
            return df.drop(columns=cols_with_na)

        for col in columns:
            if col not in df.columns or not df[col].isna().any():
                continue
            if strategy == "mean" and pd.api.types.is_numeric_dtype(df[col]):
                df[col] = df[col].fillna(df[col].mean())
            elif strategy == "median" and pd.api.types.is_numeric_dtype(df[col]):
                df[col] = df[col].fillna(df[col].median())
            elif strategy == "mode":
                mode_val = df[col].mode(dropna=True)
                if not mode_val.empty:
                    df[col] = df[col].fillna(mode_val.iloc[0])
            elif strategy == "ffill":
                df[col] = df[col].ffill()
            elif strategy == "bfill":
                df[col] = df[col].bfill()
        return df

    def detect_outliers_iqr(self, df: pd.DataFrame, column: str) -> tuple[pd.Series, float, float]:
        series = df[column].dropna()
        q1, q3 = series.quantile(0.25), series.quantile(0.75)
        iqr = q3 - q1
        lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        mask = (df[column] < lower) | (df[column] > upper)
        return mask.fillna(False), lower, upper

    def detect_outliers_zscore(
        self, df: pd.DataFrame, column: str, threshold: float = 3.0
    ) -> pd.Series:
        series = df[column]
        mean, std = series.mean(), series.std()
        if std == 0 or np.isnan(std):
            return pd.Series(False, index=df.index)
        z = (series - mean) / std
        return z.abs() > threshold

    def remove_outliers(self, df: pd.DataFrame, column: str, method: str = "iqr") -> pd.DataFrame:
        if method == "iqr":
            mask, _, _ = self.detect_outliers_iqr(df, column)
        else:
            mask = self.detect_outliers_zscore(df, column)
        return df.loc[~mask].reset_index(drop=True)

    def cap_outliers(self, df: pd.DataFrame, column: str) -> pd.DataFrame:
        """Winsorize using the IQR bounds."""
        df = df.copy()
        _, lower, upper = self.detect_outliers_iqr(df, column)
        df[column] = df[column].clip(lower=lower, upper=upper)
        return df
