"""
correlation.py

Thin wrapper connecting the statistics engine's correlation utilities
to the visualization plots module, so the UI layer has one place to
call for "analyze + plot correlations".
"""
from __future__ import annotations

import pandas as pd

from ml_lab.data.statistics import StatisticsEngine
from ml_lab.visualization.plots import plot_correlation_heatmap


def analyze_and_plot(df: pd.DataFrame, method: str = "pearson", top_n: int = 5):
    engine = StatisticsEngine()
    top_pairs = engine.top_correlations(df, method=method, top_n=top_n)
    plot_path = plot_correlation_heatmap(df)
    return top_pairs, plot_path
