"""docs/rules.md must stay in sync with the rule catalog."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from foldsafe.rules import ACTIVE_RULES, RULES

DOCS = Path(__file__).resolve().parents[1] / "docs" / "rules.md"


def rule_sections() -> dict[str, str]:
    """Rule code -> its section text (heading line included)."""
    parts = re.split(r"^(?=## FS\d{3} )", DOCS.read_text(encoding="utf-8"), flags=re.MULTILINE)
    return {part[3:8]: part for part in parts if part.startswith("## FS")}


def test_docs_have_one_section_per_rule_in_order():
    assert list(rule_sections()) == sorted(RULES)


@pytest.mark.parametrize("code", sorted(RULES))
def test_rule_section_matches_the_catalog(code: str):
    section = rule_sections()[code]
    rule = RULES[code]
    assert section.splitlines()[0] == f"## {code} · {rule.title}"
    assert f"Severity: {rule.severity.value}" in section
    status = "active" if code in ACTIVE_RULES else "planned"
    assert f"Status: {status}" in section
    # A leaky example marked with the rule, then a fix.
    assert f"# {code}" in section
    assert "**Fix.**" in section
