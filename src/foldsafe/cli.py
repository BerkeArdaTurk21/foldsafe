"""Command-line interface: `foldsafe check <paths>`."""

from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Annotated

import typer

from foldsafe import __version__
from foldsafe.analyzer import analyze_file
from foldsafe.report import exit_code, render_json, render_text

SUPPORTED_SUFFIXES = {".py", ".ipynb"}
SKIPPED_DIRS = {".git", ".venv", "venv", "__pycache__", ".ipynb_checkpoints", "node_modules"}

app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    help="Catch data leakage in ML code before your metrics lie to you.",
)


class OutputFormat(str, Enum):
    text = "text"
    json = "json"


def _version_callback(value: bool) -> None:
    if value:
        typer.echo(f"foldsafe {__version__}")
        raise typer.Exit()


@app.callback()
def main(
    version: Annotated[
        bool,
        typer.Option("--version", callback=_version_callback, is_eager=True, help="Show version."),
    ] = False,
) -> None:
    """foldsafe: static data-leakage checks for Python scripts and Jupyter notebooks."""


def collect_files(paths: list[Path]) -> list[Path]:
    """Expand files and directories into a sorted list of .py / .ipynb files."""
    found: set[Path] = set()
    for path in paths:
        if path.is_file():
            if path.suffix in SUPPORTED_SUFFIXES:
                found.add(path)
            continue
        for candidate in path.rglob("*"):
            # Only folders inside the given path count: a project may itself live under venv/.
            if SKIPPED_DIRS.intersection(candidate.relative_to(path).parts):
                continue
            if candidate.is_file() and candidate.suffix in SUPPORTED_SUFFIXES:
                found.add(candidate)
    return sorted(found)


@app.command()
def check(
    paths: Annotated[
        list[Path],
        typer.Argument(exists=True, help="Files or directories to analyze."),
    ],
    output_format: Annotated[
        OutputFormat,
        typer.Option("--format", help="Output format: text for people, json for tools."),
    ] = OutputFormat.text,
) -> None:
    """Analyze files for data leakage.

    Exit codes: 0 no leak found, 1 leak found, 2 a file could not be analyzed.
    """
    results = [analyze_file(path) for path in collect_files(paths)]
    render = render_json if output_format is OutputFormat.json else render_text
    typer.echo(render(results))
    raise typer.Exit(exit_code(results))
