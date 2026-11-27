"""FS005: a statistic of all rows that reaches a split or cross-validation."""

from __future__ import annotations

import textwrap

import pytest

from foldsafe.analyzer import analyze_source

HEADER = """\
import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score, train_test_split
df = pd.read_csv("train.csv")
"""


def found(body: str) -> list[tuple[str, int]]:
    source = HEADER + textwrap.dedent(body)
    return [(f.rule, f.line) for f in analyze_source(source)]


@pytest.mark.parametrize(
    "statement",
    [
        'df["Age"] = df["Age"].fillna(df["Age"].median())',
        'df["z"] = (df["x"] - df["x"].mean()) / df["x"].std()',
        'df["rate"] = df.groupby("Ticket")["Survived"].transform("mean")',
        'df["share"] = df["x"] / df["x"].sum()',
        'df["scaled"] = (df["x"] - df["x"].min()) / (df["x"].max() - df["x"].min())',
        'df["centered"] = df["x"] - np.mean(df["x"], axis=0)',
        'df["freq"] = df["city"].map(df["city"].value_counts())',
    ],
)
def test_full_frame_statistic_before_split_is_reported(statement):
    assert found(statement + "\ntrain, test = train_test_split(df)\n") == [("FS005", 5)]


def test_statistic_before_cross_validation_is_reported():
    body = """\
    X = df[["a", "b"]]
    X = X.fillna(X.mean())
    print(cross_val_score(None, X, df["y"], cv=5))
    """
    assert found(body) == [("FS005", 6)]


def test_reported_where_the_statistic_is_computed():
    body = """\
    age_median = df["Age"].median()
    df["Age"] = df["Age"].fillna(age_median)
    train, test = train_test_split(df)
    """
    assert found(body) == [("FS005", 5)]


@pytest.mark.parametrize(
    "body",
    [
        # Exploration: printed, never kept in the data.
        'print(df["Age"].median())\nprint(df.groupby("Pclass")["Survived"].mean())\n'
        "train, test = train_test_split(df)\n",
        # Row-wise: every row uses only its own values.
        'df["total"] = df[["a", "b"]].sum(axis=1)\n'
        'df["avg"] = df[["a", "b"]].mean(axis="columns")\n'
        'df["peak"] = np.max(df[["a", "b"]], axis=1)\n'
        'df["both"] = df[["a", "b"]].sum(1)\n'
        "train, test = train_test_split(df)\n",
        # Statistic of the training rows only.
        'train, test = train_test_split(df)\nmed = train["Age"].median()\n'
        'train["Age"] = train["Age"].fillna(med)\ntest["Age"] = test["Age"].fillna(med)\n',
        # Kept, but the data is never split or cross-validated (final model on all rows).
        'df["Age"] = df["Age"].fillna(df["Age"].median())\n',
        # A statistic of a separate test file.
        'test = pd.read_csv("test.csv")\ntest["Age"] = test["Age"].fillna(test["Age"].median())\n'
        "a, b = train_test_split(df)\n",
    ],
    ids=["exploration", "row-wise", "train-only", "no-split", "test-file"],
)
def test_clean_statistics_are_not_reported(body):
    assert found(body) == []


def test_train_plus_test_statistic_stays_fs004():
    body = """\
    test = pd.read_csv("test.csv")
    full = pd.concat([df, test])
    df["Age"] = df["Age"].fillna(full["Age"].median())
    train, valid = train_test_split(df)
    """
    assert [rule for rule, _ in found(body)] == ["FS004"]
