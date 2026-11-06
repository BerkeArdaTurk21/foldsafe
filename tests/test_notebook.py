"""Reading notebooks: IPython syntax, cell and line numbers, cells that are not Python."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from foldsafe.analyzer import analyze_file
from foldsafe.notebook import read_notebook

EXAMPLE = Path(__file__).resolve().parent.parent / "examples" / "notebooks"
IMPORTS = [
    "import pandas as pd\n",
    "from sklearn.linear_model import LogisticRegression\n",
    "from sklearn.model_selection import cross_val_score, train_test_split\n",
    "from sklearn.preprocessing import StandardScaler\n",
    'X = pd.read_csv("train.csv")\n',
    "y = X.pop('target')",
]


def code(source: str | list[str]) -> dict:
    return {"cell_type": "code", "source": source}


def notebook_text(*cells: dict) -> str:
    return json.dumps({"cells": list(cells), "nbformat": 4})


def write_notebook(tmp_path: Path, *cells: dict) -> Path:
    path = tmp_path / "analysis.ipynb"
    path.write_text(notebook_text(*cells), encoding="utf-8")
    return path


def test_cells_are_joined_in_order_and_mapped_back():
    text = notebook_text(
        code(["a = 1\n", "b = 2"]),
        {"cell_type": "markdown", "source": "# text"},
        code("c = 3"),
    )
    notebook = read_notebook(text)
    assert notebook.source == "a = 1\nb = 2\nc = 3\n"
    assert notebook.locate(1) == (1, 1)
    assert notebook.locate(2) == (1, 2)
    assert notebook.locate(3) == (3, 1)
    assert notebook.describe_line(3) == "cell 3, line 1"


def test_ipython_lines_are_blanked_and_numbering_is_kept():
    notebook = read_notebook(
        notebook_text(code(["%matplotlib inline\n", "!ls\n", "df?\n", "a = 1"]))
    )
    assert notebook.source == "\n\n\na = 1\n"
    assert notebook.locate(4) == (1, 4)
    assert notebook.skipped_cells == ()


def test_cell_magics_blank_the_whole_cell():
    notebook = read_notebook(notebook_text(code(["%%bash\n", "echo hello world\n"]), code("a = 1")))
    assert notebook.source == "\n\na = 1\n"


def test_a_cell_that_is_not_python_is_skipped_not_fatal():
    notebook = read_notebook(notebook_text(code("files = !ls"), code("a = (1"), code("b = 2")))
    assert notebook.skipped_cells == (2,)
    assert notebook.source == "\nb = 2\n"


@pytest.mark.parametrize("text", ["not json", "[1, 2]", "{}", '{"cells": 3}'])
def test_text_that_is_not_a_notebook_raises(text: str):
    with pytest.raises(ValueError):
        read_notebook(text)


def test_notebook_without_code_cells():
    assert read_notebook(notebook_text({"cell_type": "markdown", "source": "x"})).source == ""


def test_findings_point_at_cells(tmp_path: Path):
    path = write_notebook(
        tmp_path,
        code(IMPORTS),
        {"cell_type": "markdown", "source": "# Scale"},
        code(["# scale all rows\n", "S = StandardScaler().fit_transform(X)"]),
        code("X_a, X_b = train_test_split(S)"),
    )
    (finding,) = analyze_file(path).findings
    assert (finding.rule, finding.cell, finding.line) == ("FS001", 3, 2)


def test_leak_across_cells_names_the_fit_cell_in_the_message(tmp_path: Path):
    path = write_notebook(
        tmp_path,
        code(IMPORTS),
        code("S = StandardScaler().fit_transform(X)"),
        code("scores = cross_val_score(LogisticRegression(), S, y)"),
    )
    (finding,) = analyze_file(path).findings
    assert (finding.rule, finding.cell, finding.line) == ("FS003", 3, 1)
    assert "cell 2, line 1" in finding.message


def test_findings_are_ordered_by_cell(tmp_path: Path):
    path = write_notebook(
        tmp_path,
        code(IMPORTS),
        code("X_a, X_test = train_test_split(X)"),
        code("first = StandardScaler().fit(X_test)"),
        code("second = StandardScaler().fit(X_test)"),
    )
    assert [finding.cell for finding in analyze_file(path).findings] == [3, 4]


def test_skipped_cells_are_reported(tmp_path: Path):
    result = analyze_file(write_notebook(tmp_path, code("a = (1"), code("b = 2")))
    assert result.skipped_cells == (1,)
    assert result.error is None


def test_example_notebook_matches_its_marker():
    path = EXAMPLE / "leaky_scaler_before_split.ipynb"
    (finding,) = analyze_file(path).findings
    assert (finding.rule, finding.cell, finding.line) == ("FS001", 4, 2)
    cell = json.loads(path.read_text(encoding="utf-8"))["cells"][finding.cell - 1]
    assert "foldsafe-expect: FS001" in cell["source"][finding.line - 1]


# --------------------------------------------------------------- IPython magics


def test_assigning_from_shell_or_magic_only_blanks_that_line():
    notebook = read_notebook(notebook_text(code(["files = !ls\n", "env = %env PATH\n", "a = 1"])))
    assert notebook.source == "\n\na = 1\n"
    assert read_notebook(notebook_text(code("a = 1 != 2"))).source == "a = 1 != 2\n"


@pytest.mark.parametrize("magic", ["%%time", "%%timeit -n 3", "%%capture out", "  %%time"])
def test_cell_magics_with_python_bodies_keep_the_code(magic: str):
    notebook = read_notebook(notebook_text(code([f"{magic}\n", "a = 1"])))
    assert notebook.source == "\na = 1\n"


@pytest.mark.parametrize(
    "line",
    ["%time a = 1", "%timeit a = 1", "%timeit -n 3 -r 2 a = 1", "%time  a = 1"],
)
def test_timed_lines_keep_the_statement(line: str):
    assert read_notebook(notebook_text(code(line))).source == "a = 1\n"


def test_a_leak_inside_timed_code_is_found(tmp_path: Path):
    path = write_notebook(
        tmp_path,
        code(IMPORTS),
        code(["%%time\n", "S = StandardScaler().fit_transform(X)\n"]),
        code("%time X_a, X_b = train_test_split(S)"),
    )
    (finding,) = analyze_file(path).findings
    assert (finding.rule, finding.cell, finding.line) == ("FS001", 2, 2)


@pytest.mark.parametrize("magic", ["%%bash", "%%html", "%%sql", "%%", "%%  "])
def test_cell_magics_with_other_languages_blank_the_cell(magic: str):
    notebook = read_notebook(notebook_text(code([f"{magic}\n", "select * from t where"])))
    assert notebook.source == "\n\n"
    assert notebook.skipped_cells == ()
