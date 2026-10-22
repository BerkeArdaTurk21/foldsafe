"""Command-line interface: `foldsafe check <paths>`."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from foldsafe import __version__

SUPPORTED_SUFFIXES = {".py", ".ipynb"}
SKIPPED_DIRS = {".git", ".venv", "venv", "__pycache__", ".ipynb_checkpoints", "node_modules"}

app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    help="Catch data leakage in ML code before your metrics lie to you.",
)


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
            if any(part in SKIPPED_DIRS for part in candidate.parts):
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
) -> None:
    """Analyze files for data leakage. Exits with code 1 when leakage is found."""
    files = collect_files(paths)
    # Rules arrive in M2; for now this only reports what would be analyzed.
    typer.echo(f"foldsafe {__version__}: {len(files)} file(s) found, 0 rules active yet.")
