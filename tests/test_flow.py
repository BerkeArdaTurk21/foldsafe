"""Unit tests for the data-flow pass: imports, data sources, origins, derivation."""

from __future__ import annotations

import ast
import textwrap

from foldsafe.flow import FlowState, run_flow
from foldsafe.values import DataValue, Origin, Taint, merge_data, merge_origins


def flow(source: str) -> FlowState:
    return run_flow(ast.parse(textwrap.dedent(source)))


# ------------------------------------------------------------------ imports


def test_import_aliases_resolve_to_qualified_names():
    state = flow(
        """
        import numpy as np
        import sklearn.preprocessing
        from pandas import read_csv as load
        from sklearn.model_selection import train_test_split
        """
    )
    assert state.imports == {
        "np": "numpy",
        "sklearn": "sklearn",
        "load": "pandas.read_csv",
        "train_test_split": "sklearn.model_selection.train_test_split",
    }


def test_relative_imports_are_ignored():
    assert flow("from .helpers import read_csv").imports == {}


# ------------------------------------------------------------- data sources


def test_read_csv_through_alias_is_full_data():
    state = flow(
        """
        import pandas as pd
        df = pd.read_csv("train.csv")
        """
    )
    assert state.env["df"] == DataValue(Origin.FULL)


def test_file_named_test_is_test_data():
    state = flow(
        """
        from pandas import read_csv
        holdout = read_csv("data/Test_Set.csv")
        """
    )
    assert state.env["holdout"].origin is Origin.TEST


def test_numpy_and_sklearn_dataset_sources():
    state = flow(
        """
        import numpy as np
        from sklearn.datasets import load_iris
        features = np.load("features.npy")
        X, y = load_iris(return_X_y=True)
        """
    )
    assert state.env["features"].origin is Origin.FULL
    assert state.env["X"].origin is Origin.FULL
    assert state.env["y"].origin is Origin.FULL


def test_unimported_or_unknown_calls_are_not_data():
    state = flow(
        """
        import pandas as pd
        df = read_csv("train.csv")
        n = len(pd.read_csv("train.csv"))
        """
    )
    assert "df" not in state.env
    assert "n" not in state.env


# --------------------------------------------------------------- derivation


def test_columns_methods_and_arithmetic_keep_the_origin():
    state = flow(
        """
        import pandas as pd
        df = pd.read_csv("train.csv")
        X = df.drop(columns=["target"])
        y = df["target"]
        values = X.values
        z = (df["area"] - df["area"].mean()) / df["area"].std()
        """
    )
    for name in ("X", "y", "values", "z"):
        assert state.env[name].origin is Origin.FULL, name


def test_concat_of_train_and_test_files_is_combined():
    state = flow(
        """
        import pandas as pd
        train = pd.read_csv("train.csv")
        test = pd.read_csv("test.csv")
        combined = pd.concat([train, test], ignore_index=True)
        head = combined.iloc[: len(train)]
        """
    )
    assert state.env["combined"].origin is Origin.COMBINED
    assert state.env["head"].origin is Origin.COMBINED


def test_reassigning_to_untracked_value_forgets_the_variable():
    state = flow(
        """
        import pandas as pd
        df = pd.read_csv("train.csv")
        df = 3
        """
    )
    assert "df" not in state.env


def test_column_assignment_keeps_the_frame():
    state = flow(
        """
        import pandas as pd
        df = pd.read_csv("train.csv")
        df["age_filled"] = df["Age"].fillna(0)
        """
    )
    assert state.env["df"] == DataValue(Origin.FULL)


def test_blocks_are_followed_but_function_bodies_are_not():
    state = flow(
        """
        import pandas as pd
        if True:
            a = pd.read_csv("a.csv")
        for _ in range(3):
            b = pd.read_csv("b.csv")
        with open("log.txt") as handle:
            c = pd.read_csv("c.csv")
        try:
            d = pd.read_csv("d.csv")
        except OSError:
            e = pd.read_csv("e.csv")

        def load():
            hidden = pd.read_csv("f.csv")
            return hidden
        """
    )
    assert set(state.env) == {"a", "b", "c", "d", "e"}


# ------------------------------------------------------------------- values


def test_merge_origins_table():
    assert merge_origins(Origin.FULL, Origin.FULL) is Origin.FULL
    assert merge_origins(Origin.TRAIN, Origin.TEST) is Origin.COMBINED
    assert merge_origins(Origin.FULL, Origin.TEST) is Origin.COMBINED
    assert merge_origins(Origin.FULL, Origin.TRAIN) is Origin.FULL
    assert merge_origins(Origin.COMBINED, Origin.TRAIN) is Origin.COMBINED


def test_merge_data_unions_taints():
    first = DataValue(Origin.FULL, frozenset({Taint(3, "preprocess")}))
    second = DataValue(Origin.FULL, frozenset({Taint(7, "select")}))
    merged = merge_data([first, second])
    assert merged == DataValue(Origin.FULL, frozenset({Taint(3, "preprocess"), Taint(7, "select")}))
    assert merge_data([]) is None


def test_every_corpus_example_runs_through_the_flow_pass():
    from pathlib import Path

    examples = Path(__file__).resolve().parent.parent / "examples"
    for path in sorted(examples.glob("*/*.py")):
        state = run_flow(ast.parse(path.read_text(encoding="utf-8")))
        # Code inside main() leaves the module scope empty but still records events.
        assert state.env or state.events, f"nothing tracked in {path.name}"
