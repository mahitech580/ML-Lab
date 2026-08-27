"""
display.py

Rich-based display helpers: panels, tables, headers, and status
messages, kept consistent across the whole application.
"""
from __future__ import annotations

import pandas as pd
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

import config

console = Console()


def show_banner() -> None:
    banner = Text(justify="center")
    banner.append(f"{config.APP_NAME}\n", style="bold cyan")
    banner.append(config.APP_TAGLINE, style="dim")
    console.print(Panel(banner, border_style="cyan", expand=True))


def show_status_bar(state) -> None:
    dataset = state.dataset_name or "None"
    rows = state.working_dataset.shape[0] if state.working_dataset is not None else "-"
    cols = state.working_dataset.shape[1] if state.working_dataset is not None else "-"
    target = state.target_column or "-"
    model = state.trained_model_name or "-"
    status = state.status()

    table = Table.grid(padding=(0, 2))
    table.add_column(style="bold")
    table.add_column()
    table.add_row("Dataset:", str(dataset))
    table.add_row("Rows:", str(rows))
    table.add_row("Columns:", str(cols))
    table.add_row("Target:", str(target))
    table.add_row("Model:", str(model))
    table.add_row("Status:", f"[green]{status}[/green]" if status != "NO DATASET" else f"[yellow]{status}[/yellow]")
    console.print(Panel(table, title="Session", border_style="blue"))


def show_menu(title: str, options: list[tuple[str, str]]) -> None:
    table = Table(title=title, show_header=False, border_style="magenta")
    table.add_column("Key", style="bold cyan", width=4)
    table.add_column("Option")
    for key, label in options:
        table.add_row(key, label)
    console.print(table)


def show_success(message: str) -> None:
    console.print(Panel(message, title="Success", border_style="green"))


def show_error(message: str) -> None:
    console.print(Panel(message, title="Error", border_style="red"))


def show_warning(message: str) -> None:
    console.print(Panel(message, title="Warning", border_style="yellow"))


def show_info(message: str) -> None:
    console.print(Panel(message, title="Info", border_style="blue"))


def show_dataframe(df: pd.DataFrame, title: str = "", max_rows: int = 20) -> None:
    table = Table(title=title, border_style="cyan", show_lines=False)
    for col in df.columns:
        table.add_column(str(col))
    for _, row in df.head(max_rows).iterrows():
        table.add_row(*[str(v) for v in row.tolist()])
    console.print(table)


def show_key_value_table(title: str, data: dict) -> None:
    table = Table(title=title, border_style="cyan")
    table.add_column("Metric", style="bold")
    table.add_column("Value")
    for k, v in data.items():
        if v is None:
            continue
        v_str = f"{v:.4f}" if isinstance(v, float) else str(v)
        table.add_row(str(k), v_str)
    console.print(table)


def show_bar_row(label: str, value: float, max_value: float, width: int = 30) -> str:
    filled = int(width * (value / max_value)) if max_value else 0
    return f"{label:<20} {'█' * filled}{' ' * (width - filled)} {value:.3f}"

def show_feature_importance(importance: dict, top_n: int = 10) -> None:
    items = list(importance.items())[:top_n]
    if not items:
        return
    max_val = max(v for _, v in items)
    console.print(Panel.fit("\n".join(
        show_bar_row(name, val, max_val) for name, val in items
    ), title="Feature Importance", border_style="green"))
