"""Analysis entry point: Python source in, findings out. Code is parsed, never executed."""

from __future__ import annotations

import ast

from foldsafe.findings import Finding
from foldsafe.rules import CHECKERS


def analyze_source(source: str, path: str = "<string>") -> list[Finding]:
    """Parse source and run every registered checker over it."""
    tree = ast.parse(source, filename=path)
    findings = [finding for check in CHECKERS.values() for finding in check(tree, path)]
    return sorted(findings, key=lambda finding: (finding.path, finding.line, finding.rule))
