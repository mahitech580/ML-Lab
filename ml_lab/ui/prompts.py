"""
prompts.py

Input helpers built on rich.prompt with validation, so invalid input
never crashes the application.
"""
from __future__ import annotations

from rich.prompt import Confirm, Prompt

import config
from ml_lab.ui.display import console, show_error


def ask_choice(message: str, choices: list[str], default: str | None = None) -> str:
    while True:
        answer = Prompt.ask(message, choices=choices, default=default, show_choices=True)
        if answer in choices:
            return answer
        show_error(f"Invalid choice. Please pick one of: {', '.join(choices)}")


def ask_text(message: str, default: str | None = None) -> str:
    return Prompt.ask(message, default=default)


def ask_int(message: str, default: int | None = None, minimum: int | None = None,
            maximum: int | None = None) -> int:
    while True:
        raw = Prompt.ask(message, default=str(default) if default is not None else None)
        try:
            value = int(raw)
        except (TypeError, ValueError):
            show_error("Please enter a valid whole number.")
            continue
        if minimum is not None and value < minimum:
            show_error(f"Value must be at least {minimum}.")
            continue
        if maximum is not None and value > maximum:
            show_error(f"Value must be at most {maximum}.")
            continue
        return value


def ask_float(message: str, default: float | None = None) -> float:
    while True:
        raw = Prompt.ask(message, default=str(default) if default is not None else None)
        try:
            return float(raw)
        except (TypeError, ValueError):
            show_error("Please enter a valid number.")


def ask_confirm(message: str, default: bool = True) -> bool:
    return Confirm.ask(message, default=default)


def ask_split() -> str:
    return ask_choice(
        f"Enter train/test split", list(config.SPLIT_OPTIONS.keys()), default=config.DEFAULT_SPLIT
    )


def ask_cv_folds() -> int:
    choice = ask_choice(
        "Select number of CV folds", [str(f) for f in config.CV_FOLD_OPTIONS],
        default=str(config.DEFAULT_CV_FOLDS),
    )
    return int(choice)


def pause() -> None:
    Prompt.ask("\n[dim]Press Enter to continue[/dim]", default="")
