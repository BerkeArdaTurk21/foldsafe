"""Hard cases: reassignment, control flow blocks and function or class bodies."""

from __future__ import annotations

import textwrap

from foldsafe.analyzer import analyze_source
from foldsafe.findings import Finding

HEADER = """\
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler
df = pd.read_csv("train.csv")
X = df.drop(columns=["target"])
y = df["target"]
"""
FIRST_BODY_LINE = HEADER.count("\n") + 1


def rule_lines(body: str) -> list[tuple[str, int]]:
    """(rule, body line) pairs; body lines count from 1 after a leading newline."""
    found: list[Finding] = analyze_source(HEADER + textwrap.dedent(body), "script.py")
    return [(finding.rule, finding.line - FIRST_BODY_LINE) for finding in found]


# ------------------------------------------------------------- reassignment


def test_reassigning_x_keeps_the_taint():
    body = """
        X = StandardScaler().fit_transform(X)
        X_train, X_test, y_train, y_test = train_test_split(X, y)
        """
    assert rule_lines(body) == [("FS001", 1)]


def test_reassigning_after_the_split_is_fine():
    body = """
        X_train, X_test, y_train, y_test = train_test_split(X, y)
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        """
    assert rule_lines(body) == []


# ------------------------------------------------------------------- blocks


def test_fit_inside_if_branch_is_followed():
    body = """
        use_scaling = True
        if use_scaling:
            X = StandardScaler().fit_transform(X)
        scores = cross_val_score(LogisticRegression(), X, y)
        """
    assert rule_lines(body) == [("FS003", 4)]


def test_fit_inside_loop_and_with_and_try_blocks():
    body = """
        for seed in range(3):
            with open("log.txt", "a") as log:
                try:
                    X_s = StandardScaler().fit_transform(X)
                    X_a, X_b = train_test_split(X_s, random_state=seed)
                except ValueError:
                    pass
        """
    assert rule_lines(body) == [("FS001", 4)]


# ---------------------------------------------------------- function bodies


def test_leak_inside_main_function_is_found():
    body = """
        def main():
            data = pd.read_csv("train.csv")
            features = data.drop(columns=["target"])
            features = StandardScaler().fit_transform(features)
            X_train, X_test = train_test_split(features)

        if __name__ == "__main__":
            main()
        """
    assert rule_lines(body) == [("FS001", 4)]


def test_function_sees_module_variables_defined_before_it():
    body = """
        X_scaled = StandardScaler().fit_transform(X)

        def evaluate():
            return cross_val_score(LogisticRegression(), X_scaled, y)
        """
    assert rule_lines(body) == [("FS003", 4)]


def test_parameters_shadow_module_variables():
    body = """
        X_scaled = StandardScaler().fit_transform(X)

        def evaluate(X_scaled, y):
            return cross_val_score(LogisticRegression(), X_scaled, y)
        """
    assert rule_lines(body) == []


def test_local_assignments_do_not_leak_into_module_scope():
    body = """
        def prepare():
            X = StandardScaler().fit_transform(df)
            return X

        X_train, X_test = train_test_split(X)
        """
    assert rule_lines(body) == []


def test_methods_nested_and_async_functions_are_analyzed():
    body = """
        class Experiment:
            def run(self):
                features = StandardScaler().fit_transform(X)
                return train_test_split(features)

        def outer():
            def inner():
                return cross_val_score(LogisticRegression(), StandardScaler().fit_transform(X), y)

        async def job():
            Z = StandardScaler().fit_transform(X)
            parts = train_test_split(Z)
        """
    assert rule_lines(body) == [("FS001", 3), ("FS003", 8), ("FS001", 11)]
