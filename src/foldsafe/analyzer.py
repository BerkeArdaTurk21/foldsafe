"""Analysis entry point: Python source in, findings out. Code is parsed, never executed."""

from __future__ import annotations

import ast

from foldsafe.findings import Finding
from foldsafe.flow import run_flow
from foldsafe.rules import CHECKERS


def analyze_source(source: str, path: str = "<string>") -> list[Finding]:
    """Parse source, run the flow pass once, then every registered checker over its events."""
    state = run_flow(ast.parse(source, filename=path))
    findings = {finding for check in CHECKERS.values() for finding in check(state, path)}
    return sorted(findings, key=lambda finding: (finding.path, finding.line, finding.rule))
