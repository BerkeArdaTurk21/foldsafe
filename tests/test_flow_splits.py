"""Unit tests for split detection and cross-validation events in the data-flow pass."""

from __future__ import annotations

import ast
import textwrap
from pathlib import Path

from foldsafe.events import CVEvent, SplitEvent
from foldsafe.flow import FlowState, run_flow
from foldsafe.values import DataValue, IndexValue, Origin, SearchValue, SplitterValue, Taint

EXAMPLES_DIR = Path(__file__).resolve().parent.parent / "examples"

HEADER = """
import pandas as pd
from sklearn.model_selection import (
    GridSearchCV, KFold, RepeatedStratifiedKFold, ShuffleSplit, StratifiedKFold,
    TimeSeriesSplit, cross_val_score, train_test_split,
)
df = pd.read_csv("train.csv")
X = df.drop(columns=["target"])
y = df["target"]
"""


def flow(body: str, state: FlowState | None = None) -> FlowState:
    return run_flow(ast.parse(textwrap.dedent(HEADER) + textwrap.dedent(body)), state)


def splits(state: FlowState) -> list[SplitEvent]:
    return [event for event in state.events if isinstance(event, SplitEvent)]


def cv_events(state: FlowState) -> list[CVEvent]:
    return [event for event in state.events if isinstance(event, CVEvent)]


# --------------------------------------------------------- train_test_split


def test_train_test_split_outputs_alternate_train_and_test():
    state = flow("X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)")
    assert state.env["X_train"] == DataValue(Origin.TRAIN)
    assert state.env["X_test"] == DataValue(Origin.TEST)
    assert state.env["y_train"] == DataValue(Origin.TRAIN)
    assert state.env["y_test"] == DataValue(Origin.TEST)
    (event,) = splits(state)
    assert event.kind == "train_test_split"
    assert event.shuffled is True
    assert event.inputs == (DataValue(Origin.FULL), DataValue(Origin.FULL))


def test_split_of_a_whole_frame_and_shuffle_false():
    state = flow("train_df, test_df = train_test_split(df, test_size=0.2, shuffle=False)")
    assert state.env["train_df"].origin is Origin.TRAIN
    assert state.env["test_df"].origin is Origin.TEST
    assert splits(state)[0].shuffled is False


def test_split_records_the_line_and_carries_taints_to_both_sides():
    seeded = FlowState()
    run_flow(ast.parse(textwrap.dedent(HEADER)), seeded)
    taint = Taint(4, "preprocess")
    seeded.env["X_scaled"] = DataValue(Origin.FULL, frozenset({taint}))
    source = "X_a, X_b = train_test_split(X_scaled)\n"
    state = run_flow(ast.parse(source), seeded)
    (event,) = splits(state)
    assert event.line == 1
    assert event.inputs[0].taints == {taint}
    assert state.env["X_a"].taints == {taint}
    assert state.env["X_b"].taints == {taint}


def test_split_with_untracked_input_keeps_positions():
    state = flow("a_train, a_test, y_train, y_test = train_test_split(features, y)")
    assert "a_train" not in state.env
    assert "a_test" not in state.env
    assert state.env["y_test"].origin is Origin.TEST


# ----------------------------------------------------------------- splitters


def test_splitter_objects_know_whether_they_shuffle():
    state = flow(
        """
        plain = KFold(n_splits=5)
        shuffled = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
        series = TimeSeriesSplit(n_splits=5)
        repeated = RepeatedStratifiedKFold(n_splits=5, n_repeats=3)
        random_split = ShuffleSplit(n_splits=10)
        """
    )
    assert state.env["plain"] == SplitterValue("sklearn.model_selection.KFold", False)
    assert state.env["shuffled"].shuffled is True
    assert state.env["series"].shuffled is False
    assert state.env["repeated"].shuffled is True
    assert state.env["random_split"].shuffled is True


def test_manual_fold_loop_marks_fold_rows():
    state = flow(
        """
        cv = KFold(n_splits=5, shuffle=True)
        for train_idx, test_idx in cv.split(X, y):
            X_fold_train = X.iloc[train_idx]
            X_fold_test = X.iloc[test_idx]
        """
    )
    assert state.env["train_idx"] == IndexValue(Origin.TRAIN)
    assert state.env["X_fold_train"].origin is Origin.TRAIN
    assert state.env["X_fold_test"].origin is Origin.TEST
    (event,) = splits(state)
    assert event.kind == "cv_split"
    assert event.shuffled is True


def test_enumerated_fold_loop():
    state = flow(
        """
        for fold, (tr, va) in enumerate(StratifiedKFold(5).split(X, y)):
            part = X[va]
        """
    )
    assert "fold" not in state.env
    assert state.env["part"].origin is Origin.TEST


# --------------------------------------------------------- cross-validation


def test_cross_val_score_records_x_and_splitter():
    state = flow(
        """
        cv = KFold(n_splits=5, shuffle=True, random_state=1)
        scores = cross_val_score(model, X, y, cv=cv)
        """
    )
    (event,) = cv_events(state)
    assert event.consumer == "sklearn.model_selection.cross_val_score"
    assert event.data == DataValue(Origin.FULL)
    assert event.shuffled is True
    assert "scores" not in state.env


def test_cv_call_nested_in_print_is_still_recorded():
    state = flow("print(cross_val_score(model, X=X, y=y, cv=5).mean())")
    (event,) = cv_events(state)
    assert event.data == DataValue(Origin.FULL)
    assert event.shuffled is False


def test_search_fit_and_nested_cv():
    state = flow(
        """
        search = GridSearchCV(model, {"C": [1, 10]}, cv=KFold(5, shuffle=True))
        search.fit(X, y)
        outer = cross_val_score(search, X, y, cv=StratifiedKFold(5))
        """
    )
    assert state.env["search"] == SearchValue("sklearn.model_selection.GridSearchCV", True)
    fit_event, outer_event = cv_events(state)
    assert fit_event.consumer == "sklearn.model_selection.GridSearchCV.fit"
    assert fit_event.shuffled is True
    assert outer_event.consumer == "sklearn.model_selection.cross_val_score"
    assert outer_event.shuffled is False


def test_untracked_data_still_produces_cv_event():
    state = flow("cross_val_score(model, features, labels)")
    (event,) = cv_events(state)
    assert event.data is None


# -------------------------------------------------------------------- corpus


def events_in(relative_path: str) -> tuple[list[str], FlowState]:
    source = (EXAMPLES_DIR / relative_path).read_text(encoding="utf-8")
    return source.splitlines(), run_flow(ast.parse(source))


def test_corpus_split_events_point_at_split_lines():
    lines, state = events_in("leaky/fs001_scaler_before_split.py")
    (event,) = splits(state)
    assert "train_test_split" in lines[event.line - 1]


def test_corpus_cv_events_point_at_cv_lines():
    for path in (
        "leaky/fs003_scaled_before_cross_val.py",
        "leaky/fs003_imputed_before_grid_search.py",
    ):
        lines, state = events_in(path)
        (event,) = cv_events(state)
        assert "foldsafe-expect: FS003" in lines[event.line - 1], path


def test_corpus_shuffle_flags_for_date_examples():
    _, leaky = events_in("leaky/fs008_shuffled_kfold_with_dates.py")
    assert cv_events(leaky)[0].shuffled is True
    _, clean = events_in("clean/fs008_chronological_split.py")
    assert splits(clean)[0].shuffled is False


# ---------------------------------------------------------------- statistics


def test_statistic_events_know_whether_the_result_is_kept():
    from foldsafe.events import StatEvent

    state = flow(
        """
        median_age = df["Age"].median()
        print(df["Age"].median())
        df["rate"] = df.groupby("Pclass")["target"].transform("mean")
        df.groupby("Pclass")["target"].transform(lambda s: s)
        """
    )
    stats = [event for event in state.events if isinstance(event, StatEvent)]
    assert [(event.method, event.stored) for event in stats] == [
        ("median", True),
        ("median", False),
        ("transform:mean", True),
    ]
    assert all(event.data.origin is Origin.FULL for event in stats)
