"""
statistics.py

Descriptive statistics and correlation analysis utilities.
"""
from __future__ import annotations

import pandas as pd
from scipy import stats as scipy_stats


class StatisticsEngine:
    """Computes descriptive statistics for numerical columns."""

    def describe(self, df: pd.DataFrame) -> pd.DataFrame:
        numeric_df = df.select_dtypes(include="number")
        if numeric_df.empty:
            return pd.DataFrame()

        rows = []
        for col in numeric_df.columns:
            series = numeric_df[col].dropna()
            if series.empty:
                continue
            rows.append({
                "column": col,
                "count": series.count(),
                "mean": series.mean(),
                "median": series.median(),
                "std": series.std(),
                "min": series.min(),
                "max": series.max(),
                "q1": series.quantile(0.25),
                "q3": series.quantile(0.75),
                "variance": series.var(),
                "skewness": scipy_stats.skew(series) if len(series) > 2 else float("nan"),
                "kurtosis": scipy_stats.kurtosis(series) if len(series) > 2 else float("nan"),
            })
        return pd.DataFrame(rows).set_index("column") if rows else pd.DataFrame()

    def column_info(self, df: pd.DataFrame) -> pd.DataFrame:
        rows = []
        for col in df.columns:
            series = df[col]
            rows.append({
                "column": col,
                "dtype": str(series.dtype),
                "non_null": int(series.notna().sum()),
                "missing": int(series.isna().sum()),
                "unique": int(series.nunique(dropna=True)),
            })
        return pd.DataFrame(rows)

    def correlation_matrix(self, df: pd.DataFrame, method: str = "pearson") -> pd.DataFrame:
        numeric_df = df.select_dtypes(include="number")
        if numeric_df.shape[1] < 2:
            return pd.DataFrame()
        return numeric_df.corr(method=method)

    def top_correlations(
        self, df: pd.DataFrame, method: str = "pearson", top_n: int = 5
    ) -> list[tuple[str, str, float]]:
        corr = self.correlation_matrix(df, method=method)
        if corr.empty:
            return []
        pairs = []
        cols = corr.columns.tolist()
        for i in range(len(cols)):
            for j in range(i + 1, len(cols)):
                val = corr.iloc[i, j]
                if pd.notna(val):
                    pairs.append((cols[i], cols[j], float(val)))
        pairs.sort(key=lambda x: abs(x[2]), reverse=True)
        return pairs[:top_n]
