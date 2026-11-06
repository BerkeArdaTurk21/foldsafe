"""Analysis entry points: source or files in, findings out. Code is parsed, never executed."""

from __future__ import annotations

import ast
import tokenize
from collections.abc import Callable
from dataclasses import dataclass, replace
from pathlib import Path

from foldsafe.findings import Finding
from foldsafe.flow import run_flow
from foldsafe.notebook import read_notebook
from foldsafe.rules import CHECKERS


@dataclass(frozen=True)
class FileResult:
    path: str
    findings: tuple[Finding, ...] = ()
    error: str | None = None  # the file could not be read or parsed at all
    skipped_cells: tuple[int, ...] = ()  # notebook cells that are not valid Python


def analyze_source(
    source: str, path: str = "<string>", describe_line: Callable[[int], str] | None = None
) -> list[Finding]:
    """Parse source, run the flow pass once, then every registered checker over its events."""
    state = run_flow(ast.parse(source, filename=path))
    if describe_line is not None:
        state.describe_line = describe_line
    findings = {finding for check in CHECKERS.values() for finding in check(state, path)}
    return sorted(findings, key=lambda finding: (finding.path, finding.line, finding.rule))


def analyze_file(path: Path) -> FileResult:
    """Analyze a .py or .ipynb file. A file that cannot be read is reported, never raised."""
    name = str(path)
    try:
        if path.suffix == ".ipynb":
            return _analyze_notebook(path.read_text(encoding="utf-8"), name)
        with tokenize.open(path) as handle:  # honours encoding declarations and a BOM
            return FileResult(name, tuple(analyze_source(handle.read(), name)))
    except SyntaxError as error:
        return FileResult(name, error=f"invalid Python on line {error.lineno}: {error.msg}")
    except (OSError, UnicodeDecodeError, ValueError) as error:
        return FileResult(name, error=str(error))


def _analyze_notebook(text: str, path: str) -> FileResult:
    notebook = read_notebook(text)
    findings = []
    for finding in analyze_source(notebook.source, path, notebook.describe_line):
        cell, line = notebook.locate(finding.line)
        findings.append(replace(finding, cell=cell, line=line))
    ordered = sorted(findings, key=lambda finding: (finding.cell, finding.line, finding.rule))
    return FileResult(path, tuple(ordered), skipped_cells=notebook.skipped_cells)
