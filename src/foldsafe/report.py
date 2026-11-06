"""Render analysis results as text for people or JSON for tools."""

from __future__ import annotations

import json
import textwrap
from collections.abc import Sequence

from foldsafe import __version__
from foldsafe.analyzer import FileResult
from foldsafe.findings import Finding, Severity
from foldsafe.rules import ACTIVE_RULES

WIDTH = 90


def exit_code(results: Sequence[FileResult]) -> int:
    """1 when a leak is found, 2 when nothing was found but a file could not be analyzed."""
    if any(result.findings for result in results):
        return 1
    return 2 if any(result.error for result in results) else 0


def render_text(results: Sequence[FileResult]) -> str:
    blocks = [_finding_block(finding) for result in results for finding in result.findings]
    notes = [_problem_line(result) for result in results if result.error or result.skipped_cells]
    parts = [*blocks, "\n".join(notes), _summary(results)]
    return "\n\n".join(part for part in parts if part)


def render_json(results: Sequence[FileResult]) -> str:
    findings = [finding for result in results for finding in result.findings]
    errors, warnings = _count_by_severity(findings)
    report = {
        "version": __version__,
        "rules_checked": sorted(ACTIVE_RULES),
        "files_checked": len(results),
        "findings": [_finding_dict(finding) for finding in findings],
        "problems": [
            {"path": result.path, "error": result.error, "skipped_cells": result.skipped_cells}
            for result in results
            if result.error or result.skipped_cells
        ],
        "summary": {"errors": errors, "warnings": warnings},
    }
    return json.dumps(report, indent=2)


def location(finding: Finding) -> str:
    if finding.cell is not None:
        return f"{finding.path}:cell {finding.cell}:{finding.line}"
    return f"{finding.path}:{finding.line}"


def _finding_block(finding: Finding) -> str:
    lines = [f"{location(finding)}: {finding.rule} [{finding.severity.value}]"]
    lines.append(_wrapped("  why: ", finding.message))
    if finding.fix:
        lines.append(_wrapped("  fix: ", finding.fix))
    return "\n".join(lines)


def _wrapped(label: str, text: str) -> str:
    return textwrap.fill(
        text,
        width=WIDTH,
        initial_indent=label,
        subsequent_indent=" " * len(label),
        break_long_words=False,
        break_on_hyphens=False,
    )


def _problem_line(result: FileResult) -> str:
    if result.error:
        return f"could not analyze {result.path}: {result.error}"
    cells = ", ".join(str(cell) for cell in result.skipped_cells)
    return f"note: {result.path}: skipped cell {cells} (not valid Python)"


def _summary(results: Sequence[FileResult]) -> str:
    findings = [finding for result in results for finding in result.findings]
    files = f"{len(results)} file{'' if len(results) == 1 else 's'}"
    rules = ", ".join(sorted(ACTIVE_RULES))
    if not findings:
        return f"No leaks found in {files}. Rules checked: {rules}."
    errors, warnings = _count_by_severity(findings)
    leaky_files = sum(1 for result in results if result.findings)
    total = len(findings)
    return (
        f"Found {total} possible leak{'' if total == 1 else 's'} in {leaky_files} of {files} "
        f"({errors} error{'' if errors == 1 else 's'}, "
        f"{warnings} warning{'' if warnings == 1 else 's'}). Rules checked: {rules}."
    )


def _count_by_severity(findings: Sequence[Finding]) -> tuple[int, int]:
    errors = sum(1 for finding in findings if finding.severity is Severity.ERROR)
    return errors, len(findings) - errors


def _finding_dict(finding: Finding) -> dict[str, object]:
    return {
        "rule": finding.rule,
        "severity": finding.severity.value,
        "path": finding.path,
        "cell": finding.cell,
        "line": finding.line,
        "message": finding.message,
        "fix": finding.fix,
    }
