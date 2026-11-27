"""Unit tests for following calls into functions defined in the same file, and for data
loaded through the project's own helpers."""

from __future__ import annotations

import ast
import textwrap

import pytest

from foldsafe.analyzer import analyze_source
from foldsafe.events import CVEvent, FitEvent, SplitEvent
from foldsafe.flow import FlowState, run_flow
from foldsafe.values import DataValue, EstimatorValue, Origin, Taint


def flow(source: str) -> FlowState:
    return run_flow(ast.parse(textwrap.dedent(source)))


def rules(source: str) -> list[tuple[str, int]]:
    return [(f.rule, f.line) for f in analyze_source(textwrap.dedent(source))]


HEADER = """\
import pandas as pd
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler
df = pd.read_csv("train.csv")
"""


# ----------------------------------------------------------------- calls


def test_return_value_carries_the_origin():
    state = flow(
        """
        import pandas as pd
        def load():
            return pd.read_csv("test.csv")
        frame = load()
        """
    )
    assert state.env["frame"] == DataValue(Origin.TEST_FILE)


def test_positional_keyword_and_default_arguments_are_bound():
    state = flow(
        HEADER
        + """
def pick(first, second=None, *, third=None):
    return second if third is None else third
a = pick(None, df)
b = pick(None, third=df)
c = pick(df)
"""
    )
    assert state.env["a"] == DataValue(Origin.FULL)
    assert state.env["b"] == DataValue(Origin.FULL)
    assert "c" not in state.env  # second and third keep their None defaults


def test_parameters_shadow_globals_of_the_same_name():
    state = flow(
        HEADER
        + """
X_train, X_test = train_test_split(df)
def first_rows(df):
    return df.head()
part = first_rows(X_test)
"""
    )
    assert state.env["part"] == DataValue(Origin.TEST)


def test_a_returned_tuple_keeps_its_parts_apart():
    state = flow(
        HEADER
        + """
X_train, X_test = train_test_split(df)
def scale(train_part, test_part):
    scaler = StandardScaler().fit(train_part)
    return scaler.transform(train_part), scaler.transform(test_part)
X_train, X_test = scale(X_train, X_test)
"""
    )
    assert state.env["X_train"] == DataValue(Origin.TRAIN)
    assert state.env["X_test"] == DataValue(Origin.TEST)


def test_returns_with_different_origins_are_not_guessed():
    state = flow(
        HEADER
        + """
X_train, X_test = train_test_split(df)
def choose(flag):
    if flag:
        return X_train
    return X_test
chosen = choose(True)
both = choose(False)
"""
    )
    assert "chosen" not in state.env and "both" not in state.env


def test_returns_with_the_same_origin_merge_their_taints():
    state = flow(
        HEADER
        + """
def prepare(frame, scaled):
    if scaled:
        return StandardScaler().fit_transform(frame)
    return frame
out = prepare(df, True)
"""
    )
    assert state.env["out"] == DataValue(Origin.FULL, frozenset({Taint(8, "preprocess")}))


def test_events_inside_a_called_function_are_recorded_with_the_arguments():
    state = flow(
        HEADER
        + """
def fit_scaler(frame):
    return StandardScaler().fit(frame)
scaler = fit_scaler(df)
"""
    )
    fits = [e for e in state.events if isinstance(e, FitEvent) and e.data is not None]
    assert [(fit.line, fit.data.origin) for fit in fits] == [(7, Origin.FULL)]
    assert isinstance(state.env["scaler"], EstimatorValue)
    assert state.env["scaler"].fit_taint == Taint(7, "preprocess")


def test_nested_helpers_are_followed():
    state = flow(
        HEADER
        + """
def inner(frame):
    return StandardScaler().fit_transform(frame)
def outer(frame):
    return inner(frame)
X = outer(df)
X_train, X_test = train_test_split(X)
"""
    )
    splits = [e for e in state.events if isinstance(e, SplitEvent)]
    assert splits[0].inputs[0].taints == {Taint(7, "preprocess")}


def test_recursion_stops():
    state = flow(
        HEADER
        + """
def forever(frame):
    return forever(frame)
out = forever(df)
"""
    )
    assert "out" not in state.env


def test_call_depth_is_limited():
    chain = "\n".join(f"def f{i}(frame):\n    return f{i + 1}(frame)" for i in range(6))
    state = flow(
        HEADER + chain + "\ndef f6(frame):\n    return frame\nout = f0(df)\nnear = f4(df)\n"
    )
    assert "out" not in state.env  # deeper than MAX_CALL_DEPTH: not followed
    assert state.env["near"] == DataValue(Origin.FULL)


def test_a_rebound_name_is_no_longer_a_function():
    state = flow(
        HEADER
        + """
def load():
    return pd.read_csv("a.csv")
load = None
out = load()
"""
    )
    assert "out" not in state.env


def test_an_imported_name_replaces_a_local_function():
    state = flow(
        """
        import pandas as pd
        def read_csv(path):
            return None
        from pandas import read_csv
        out = read_csv("a.csv")
        """
    )
    assert state.env["out"] == DataValue(Origin.FULL)


def test_methods_and_lambdas_are_not_followed():
    state = flow(
        HEADER
        + """
class Prep:
    def run(self, frame):
        return frame
run = lambda frame: frame
a = Prep().run(df)
b = run(df)
"""
    )
    assert "a" not in state.env and "b" not in state.env


def test_star_arguments_bind_keywords_only():
    state = flow(
        HEADER
        + """
def pick(first=None, second=None):
    return second
args = [df]
a = pick(*args, second=df)
b = pick(*args)
"""
    )
    assert state.env["a"] == DataValue(Origin.FULL)
    assert "b" not in state.env


def test_cross_validation_inside_a_helper_sees_the_argument():
    state = flow(
        HEADER
        + """
X = df.drop(columns="y")
def evaluate(model, data=None):
    return cross_val_score(model, X if data is None else data, df["y"])
evaluate(None, data=StandardScaler().fit_transform(X))
"""
    )
    reached = [e for e in state.events if isinstance(e, CVEvent) and e.data and e.data.taints]
    assert reached and reached[0].data.taints == {Taint(9, "preprocess")}


def test_helper_leak_is_reported_once_at_the_fit_line():
    assert rules(
        HEADER
        + """
def preprocess(frame):
    return StandardScaler().fit_transform(frame)
X = preprocess(df)
Y = preprocess(df)
X_train, X_test = train_test_split(X)
"""
    ) == [("FS001", 7)]


def test_helper_called_only_on_training_rows_is_clean():
    assert (
        rules(
            HEADER
            + """
def preprocess(frame):
    return StandardScaler().fit_transform(frame)
X_train, X_test = train_test_split(df)
X_train = preprocess(X_train)
"""
        )
        == []
    )


# ------------------------------------------------------- own data loaders


@pytest.mark.parametrize(
    ("call", "origin"),
    [
        ('load_raw("data/train.csv")', Origin.FULL),
        ('load_raw(DATA_DIR / "test.csv")', Origin.TEST_FILE),
        ('load_raw(path="data/test.parquet")', Origin.TEST_FILE),
        ('load_raw(f"{DATA_DIR}/train.csv.gz")', Origin.FULL),
        ('load_raw("sales.xlsx", sheet="2024")', Origin.FULL),
    ],
)
def test_own_loader_with_a_data_file_is_a_source(call, origin):
    state = flow(f"from mypkg.data import load_raw\nDATA_DIR = 'data'\nframe = {call}\n")
    assert state.env["frame"] == DataValue(origin)


@pytest.mark.parametrize(
    "source",
    [
        'from pathlib import Path\nframe = Path("train.csv")',
        'import os\nframe = os.path.join("data", "train.csv")',
        'import joblib\nframe = joblib.load("model.pkl")',
        'from mypkg.config import load\nframe = load("settings.yaml")',
        'from mypkg.io import download\nframe = download("https://example.com/report.pdf")',
        'frame = open("train.csv")',
    ],
)
def test_other_calls_with_paths_are_not_sources(source):
    assert "frame" not in flow(source).env


def test_tuple_returns_merge_part_by_part():
    state = flow(
        HEADER
        + """
def prepare(frame, scaled):
    if scaled:
        return StandardScaler().fit_transform(frame), frame
    return frame, frame
a, b = prepare(df, True)
def pair(flag):
    if flag:
        return df, StandardScaler()
    return df, df
r, s = pair(True)
"""
    )
    assert state.env["a"] == DataValue(Origin.FULL, frozenset({Taint(8, "preprocess")}))
    assert state.env["b"] == DataValue(Origin.FULL)
    assert state.env["r"] == DataValue(Origin.FULL)
    assert "s" not in state.env  # an estimator in one branch, rows in the other


def test_a_tuple_and_a_single_value_are_not_guessed():
    state = flow(
        HEADER
        + """
def load(kind):
    if kind == "pair":
        return df, df
    return df
one = load("one")
"""
    )
    assert "one" not in state.env
