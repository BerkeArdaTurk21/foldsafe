"""Unit tests for the rule checkers, beyond what the example corpus covers."""

from __future__ import annotations

import textwrap

from foldsafe.analyzer import analyze_source
from foldsafe.findings import Finding, Severity

HEADER = """\
import pandas as pd
from sklearn.feature_selection import SelectKBest
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV, KFold, cross_val_score, train_test_split
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


# -------------------------------------------------------------------- FS002


def test_fs002_wins_over_fs001_and_fs003_for_selectors():
    source = """
        X_best = SelectKBest(k=5).fit_transform(X, y)
        X_a, X_b = train_test_split(X_best)
        scores = cross_val_score(LogisticRegression(), X_best, y)
        """
    assert lines(findings(source, "FS002")) == [1]
    assert findings(source, "FS001") == []
    assert findings(source, "FS003") == []


def test_fs002_selection_on_training_rows_is_fine():
    found = findings(
        """
        X_train, X_test, y_train, y_test = train_test_split(X, y)
        selector = SelectKBest(k=5).fit(X_train, y_train)
        X_test_best = selector.transform(X_test)
        """,
        "FS002",
    )
    assert found == []


def test_fs002_selection_that_is_never_evaluated_is_not_reported():
    assert findings("X_best = SelectKBest(k=5).fit_transform(X, y)", "FS002") == []


# -------------------------------------------------------------------- FS003


def test_fs003_reports_each_cv_call_and_names_the_fit_line():
    found = findings(
        """
        X_scaled = StandardScaler().fit_transform(X)
        first = cross_val_score(LogisticRegression(), X_scaled, y)
        search = GridSearchCV(LogisticRegression(), {"C": [1, 10]})
        search.fit(X_scaled, y)
        """,
        "FS003",
    )
    assert lines(found) == [2, 4]
    assert f"line {FIRST_BODY_LINE + 1}" in found[0].message


def test_fs003_defers_to_fs001_when_the_same_data_is_also_split():
    source = """
        X_scaled = StandardScaler().fit_transform(X)
        X_a, X_b = train_test_split(X_scaled)
        scores = cross_val_score(LogisticRegression(), X_scaled, y)
        """
    assert lines(findings(source, "FS001")) == [1]
    assert findings(source, "FS003") == []


def test_fs003_pipeline_in_cross_validation_is_fine():
    found = findings(
        "scores = cross_val_score(make_pipeline(StandardScaler(), LogisticRegression()), X, y)",
        "FS003",
    )
    assert found == []
