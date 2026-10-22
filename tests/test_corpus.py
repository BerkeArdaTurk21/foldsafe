"""The labeled example corpus is the analyzer's ground truth (see examples/README.md)."""

from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

from foldsafe.analyzer import analyze_source
from foldsafe.rules import ACTIVE_RULES, RULES

EXAMPLES_DIR = Path(__file__).resolve().parent.parent / "examples"
MARKER = re.compile(r"#\s*foldsafe-expect:\s*(.+)$")
MIN_EXAMPLES_PER_KIND = 2

LEAKY = sorted((EXAMPLES_DIR / "leaky").glob("*.py"))
CLEAN = sorted((EXAMPLES_DIR / "clean").glob("*.py"))
ALL_EXAMPLES = LEAKY + CLEAN


def example_id(path: Path) -> str:
    return f"{path.parent.name}/{path.name}"


def primary_rule(path: Path) -> str:
    """The rule an example illustrates, taken from its file name prefix."""
    return path.name.split("_", 1)[0].upper()


def expected_findings(path: Path) -> set[tuple[int, str]]:
    """(line, rule) pairs declared with `# foldsafe-expect:` markers."""
    expected = set()
    lines = path.read_text(encoding="utf-8").splitlines()
    for lineno, line in enumerate(lines, start=1):
        match = MARKER.search(line)
        if match:
            expected.update((lineno, code.strip()) for code in match.group(1).split(","))
    return expected


def test_corpus_is_not_empty():
    assert LEAKY
    assert CLEAN


@pytest.mark.parametrize("path", ALL_EXAMPLES, ids=example_id)
def test_example_is_valid_python(path: Path):
    ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


@pytest.mark.parametrize("path", LEAKY, ids=example_id)
def test_leaky_example_marks_its_own_rule(path: Path):
    codes = {code for _, code in expected_findings(path)}
    assert codes, "leaky example has no foldsafe-expect marker"
    assert codes <= set(RULES), f"unknown rule code(s): {codes - set(RULES)}"
    assert primary_rule(path) in codes


@pytest.mark.parametrize("path", CLEAN, ids=example_id)
def test_clean_example_has_no_markers(path: Path):
    assert primary_rule(path) in RULES
    assert expected_findings(path) == set()


@pytest.mark.parametrize("code", sorted(RULES))
def test_every_rule_has_enough_examples(code: str):
    leaky = [path for path in LEAKY if primary_rule(path) == code]
    clean = [path for path in CLEAN if primary_rule(path) == code]
    assert len(leaky) >= MIN_EXAMPLES_PER_KIND, f"{code}: {len(leaky)} leaky example(s)"
    assert len(clean) >= MIN_EXAMPLES_PER_KIND, f"{code}: {len(clean)} clean example(s)"


@pytest.mark.parametrize("path", ALL_EXAMPLES, ids=example_id)
def test_analyzer_matches_corpus(path: Path):
    # Only active rules are compared, on every file, so an active rule that fires
    # on another rule's clean example still fails here.
    source = path.read_text(encoding="utf-8")
    actual = {
        (finding.line, finding.rule)
        for finding in analyze_source(source, str(path))
        if finding.rule in ACTIVE_RULES
    }
    expected = {(line, code) for line, code in expected_findings(path) if code in ACTIVE_RULES}
    assert actual == expected
