"""Unit tests for estimator recognition, fit events and taint propagation."""

from __future__ import annotations

import ast
import re
import textwrap
from pathlib import Path

import pytest

from foldsafe.catalog import estimator_kind
from foldsafe.events import CVEvent, FitEvent, SplitEvent
from foldsafe.flow import FlowState, run_flow
from foldsafe.values import DataValue, EstimatorValue, Origin, Taint

EXAMPLES_DIR = Path(__file__).resolve().parent.parent / "examples"
MARKER = re.compile(r"#\s*foldsafe-expect:\s*(.+)$")

HEADER = """\
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler
df = pd.read_csv("train.csv")
X = df.drop(columns=["target"])
y = df["target"]
"""
FIRST_BODY_LINE = HEADER.count("\n") + 1


def flow(body: str) -> FlowState:
    return run_flow(ast.parse(HEADER + textwrap.dedent(body)))


def fits(state: FlowState) -> list[FitEvent]:
    return [event for event in state.events if isinstance(event, FitEvent)]


def line(offset: int) -> int:
    """Source line of the body's n-th line (0-based)."""
    return FIRST_BODY_LINE + offset


# ------------------------------------------------------------------ catalog


@pytest.mark.parametrize(
    ("name", "kind"),
    [
        ("sklearn.preprocessing.StandardScaler", "preprocess"),
        ("sklearn.impute.SimpleImputer", "preprocess"),
        ("sklearn.compose.ColumnTransformer", "preprocess"),
        ("sklearn.feature_selection.RFE", "select"),
        ("sklearn.preprocessing.TargetEncoder", "select"),
        ("imblearn.over_sampling.SMOTE", "resample"),
        ("sklearn.pipeline.make_pipeline", "pipeline"),
        ("imblearn.pipeline.Pipeline", "pipeline"),
        ("sklearn.ensemble.RandomForestClassifier", "model"),
        ("xgboost.XGBClassifier", "model"),
        ("sklearn.feature_selection.f_classif", None),
        ("sklearn.model_selection.train_test_split", None),
        ("pandas.DataFrame", None),
    ],
)
def test_estimator_kinds(name: str, kind: str | None):
    assert estimator_kind(name) == kind


# ---------------------------------------------------------------- fit events


def test_fit_transform_on_full_data_taints_the_output():
    state = flow("X_scaled = StandardScaler().fit_transform(X)")
    taint = Taint(line(0), "preprocess")
    assert state.env["X_scaled"] == DataValue(Origin.FULL, frozenset({taint}))
    (event,) = fits(state)
    assert event == FitEvent(
        line(0),
        "sklearn.preprocessing.StandardScaler",
        "preprocess",
        "fit_transform",
        DataValue(Origin.FULL),
        False,
    )


def test_separate_fit_and_transform_carry_the_fit_line():
    state = flow(
        """
        imputer = SimpleImputer(strategy="median")
        imputer.fit(X)
        X_filled = imputer.transform(X)
        """
    )
    fit_line = line(2)
    assert state.env["imputer"].fit_origin is Origin.FULL
    assert state.env["X_filled"].taints == {Taint(fit_line, "preprocess")}


def test_fit_on_training_rows_leaves_no_taint():
    state = flow(
        """
        X_train, X_test, y_train, y_test = train_test_split(X, y)
        scaler = StandardScaler().fit(X_train)
        X_test_scaled = scaler.transform(X_test)
        """
    )
    assert state.env["scaler"] == EstimatorValue(
        "sklearn.preprocessing.StandardScaler", "preprocess", Origin.TRAIN, None
    )
    assert state.env["X_test_scaled"] == DataValue(Origin.TEST)


def test_fit_on_test_rows_is_recorded():
    state = flow(
        """
        X_train, X_test = train_test_split(X)
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.fit_transform(X_test)
        """
    )
    origins = [event.data.origin for event in fits(state)]
    assert origins == [Origin.TRAIN, Origin.TEST]
    assert state.env["scaler"].fit_origin is Origin.TEST


def test_selector_with_labels_leaves_a_select_taint():
    state = flow("X_best = SelectKBest(f_classif, k=5).fit_transform(X, y)")
    assert state.env["X_best"].taints == {Taint(line(0), "select")}
    assert fits(state)[0].uses_y is True


def test_fit_resample_taints_both_outputs():
    state = flow("X_res, y_res = SMOTE(random_state=0).fit_resample(X, y)")
    taint = Taint(line(0), "resample")
    assert state.env["X_res"].taints == {taint}
    assert state.env["y_res"].taints == {taint}
    assert fits(state)[0].method == "fit_resample"


def test_model_fit_is_recorded_without_taint():
    state = flow("model = LogisticRegression().fit(X, y)")
    assert state.env["model"].fit_taint is None
    assert fits(state)[0].kind == "model"


def test_chained_transformers_accumulate_taints():
    state = flow(
        """
        X_scaled = StandardScaler().fit_transform(X)
        X_reduced = PCA(n_components=3).fit_transform(X_scaled)
        """
    )
    assert state.env["X_reduced"].taints == {
        Taint(line(1), "preprocess"),
        Taint(line(2), "preprocess"),
    }


# ----------------------------------------------------------------- pipelines


def test_pipeline_kind_depends_on_the_last_step():
    state = flow(
        """
        with_model = make_pipeline(StandardScaler(), LogisticRegression())
        transformers = make_pipeline(SimpleImputer(), StandardScaler())
        selecting = Pipeline([("scale", StandardScaler()), ("best", SelectKBest(k=5))])
        named_model = Pipeline(steps=[("scale", StandardScaler()), ("model", LogisticRegression())])
        """
    )
    assert state.env["with_model"].kind == "pipeline"
    assert state.env["transformers"].kind == "preprocess"
    assert state.env["selecting"].kind == "select"
    assert state.env["named_model"].kind == "pipeline"


def test_transformer_pipeline_fitted_on_full_data_taints_output():
    state = flow("X_prepared = make_pipeline(SimpleImputer(), StandardScaler()).fit_transform(X)")
    assert state.env["X_prepared"].taints == {Taint(line(0), "preprocess")}


def test_model_pipeline_fit_on_full_data_leaves_no_taint():
    state = flow(
        """
        pipeline = make_pipeline(StandardScaler(), LogisticRegression())
        pipeline.fit(X, y)
        """
    )
    assert state.env["pipeline"].fit_taint is None


# -------------------------------------------------------------------- corpus


def marked_lines(path: Path, codes: set[str]) -> set[int]:
    lines = path.read_text(encoding="utf-8").splitlines()
    found = set()
    for number, text in enumerate(lines, start=1):
        match = MARKER.search(text)
        if match and codes & {code.strip() for code in match.group(1).split(",")}:
            found.add(number)
    return found


def taints_reaching_splits_or_cv(path: Path) -> set[Taint]:
    """Taints carried by data at the moment it is split or cross-validated."""
    state = run_flow(ast.parse(path.read_text(encoding="utf-8")))
    reached: set[Taint] = set()
    for event in state.events:
        if isinstance(event, SplitEvent):
            for value in event.inputs:
                reached |= value.taints
        elif isinstance(event, CVEvent) and event.data is not None:
            reached |= event.data.taints
    return reached


FIT_RULE_KINDS = {"FS001": "preprocess", "FS002": "select", "FS006": "resample"}


@pytest.mark.parametrize("code", sorted(FIT_RULE_KINDS))
def test_leaky_fit_markers_reach_a_split_or_cv(code: str):
    for path in sorted((EXAMPLES_DIR / "leaky").glob(f"{code.lower()}_*.py")):
        reached = taints_reaching_splits_or_cv(path)
        for marked in marked_lines(path, {code}):
            assert Taint(marked, FIT_RULE_KINDS[code]) in reached, f"{path.name}:{marked}"


def test_fs003_cv_calls_receive_preprocess_taints():
    for path in sorted((EXAMPLES_DIR / "leaky").glob("fs003_*.py")):
        state = run_flow(ast.parse(path.read_text(encoding="utf-8")))
        cv_lines = {
            event.line
            for event in state.events
            if isinstance(event, CVEvent)
            and event.data is not None
            and any(taint.kind == "preprocess" for taint in event.data.taints)
        }
        assert cv_lines == marked_lines(path, {"FS003"}), path.name


def test_no_clean_example_splits_or_cross_validates_tainted_data():
    for path in sorted((EXAMPLES_DIR / "clean").glob("*.py")):
        assert taints_reaching_splits_or_cv(path) == set(), path.name
