"""Unit tests for the rule checkers, beyond what the example corpus covers."""

from __future__ import annotations

import textwrap

from foldsafe.analyzer import analyze_source
from foldsafe.findings import Finding, Severity

HEADER = """\
import pandas as pd
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import StandardScaler
df = pd.read_csv("train.csv")
X = df.drop(columns=["target"])
y = df["target"]
"""
FIRST_BODY_LINE = HEADER.count("\n") + 1


def findings(body: str, rule: str) -> list[Finding]:
    source = HEADER + textwrap.dedent(body)
    return [finding for finding in analyze_source(source, "script.py") if finding.rule == rule]


def lines(found: list[Finding]) -> list[int]:
    return [finding.line - FIRST_BODY_LINE for finding in found]


# -------------------------------------------------------------------- FS001


def test_fs001_reports_the_fit_line_once_for_repeated_splits():
    found = findings(
        """
        X_scaled = StandardScaler().fit_transform(X)
        X_a, X_b = train_test_split(X_scaled, random_state=0)
        X_c, X_d = train_test_split(X_scaled, random_state=1)
        """,
        "FS001",
    )
    assert lines(found) == [1]
    assert found[0].severity is Severity.ERROR
    assert found[0].path == "script.py"
    assert "train/test split" in found[0].message


def test_fs001_manual_fold_loop_counts_as_a_split():
    found = findings(
        """
        X_scaled = StandardScaler().fit_transform(X)
        for train_idx, test_idx in KFold(5).split(X_scaled):
            pass
        """,
        "FS001",
    )
    assert lines(found) == [1]


def test_fs001_ignores_a_full_data_fit_that_is_never_split():
    found = findings(
        """
        X_train, X_test, y_train, y_test = train_test_split(X, y)
        final_scaler = StandardScaler().fit(X)
        X_all = final_scaler.transform(X)
        """,
        "FS001",
    )
    assert found == []


def test_fs001_ignores_fits_on_training_rows():
    found = findings(
        """
        X_train, X_test = train_test_split(X)
        scaler = StandardScaler().fit(X_train)
        X_train_scaled = scaler.transform(X_train)
        X_c, X_d = train_test_split(X_train_scaled)
        """,
        "FS001",
    )
    assert found == []
