import json
import subprocess
import sys
from pathlib import Path

from typer.testing import CliRunner

from foldsafe import __version__
from foldsafe.cli import app, collect_files

runner = CliRunner()


def test_version_flag_prints_version():
    result = runner.invoke(app, ["--version"])
    assert result.exit_code == 0
    assert __version__ in result.output


def test_check_counts_python_and_notebook_files(tmp_path: Path):
    (tmp_path / "train.py").write_text("x = 1\n")
    (tmp_path / "eda.ipynb").write_text('{"cells": []}')
    (tmp_path / "notes.md").write_text("# not code\n")

    result = runner.invoke(app, ["check", str(tmp_path)])

    assert result.exit_code == 0
    assert "No leaks found in 2 files" in result.output


def test_collect_files_skips_virtualenv_and_checkpoints(tmp_path: Path):
    (tmp_path / "model.py").write_text("x = 1\n")
    (tmp_path / ".venv" / "lib").mkdir(parents=True)
    (tmp_path / ".venv" / "lib" / "site.py").write_text("x = 1\n")
    (tmp_path / ".ipynb_checkpoints").mkdir()
    (tmp_path / ".ipynb_checkpoints" / "eda-checkpoint.ipynb").write_text("{}")

    assert collect_files([tmp_path]) == [tmp_path / "model.py"]


def test_check_rejects_missing_path(tmp_path: Path):
    result = runner.invoke(app, ["check", str(tmp_path / "missing.py")])
    assert result.exit_code != 0


def test_check_accepts_single_files_and_ignores_other_suffixes(tmp_path: Path):
    script = tmp_path / "train.py"
    script.write_text("x = 1\n")
    notes = tmp_path / "notes.md"
    notes.write_text("# not code\n")

    assert collect_files([script, notes]) == [script]


def test_module_entry_point_prints_version():
    result = subprocess.run(
        [sys.executable, "-m", "foldsafe", "--version"], capture_output=True, text=True, check=True
    )
    assert __version__ in result.stdout


# ------------------------------------------------------- output and exit codes

LEAK = """import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
X = pd.read_csv("train.csv")
S = StandardScaler().fit_transform(X)
a, b = train_test_split(S)
"""


def write(path: Path, content: str) -> Path:
    path.write_text(content, encoding="utf-8")
    return path


def test_leak_prints_finding_and_exits_1(tmp_path: Path):
    script = write(tmp_path / "train.py", LEAK)
    result = runner.invoke(app, ["check", str(script)])
    assert result.exit_code == 1
    assert f"{script}:5: FS001 [error]" in result.output
    assert "why: " in result.output
    assert "fix: " in result.output
    assert "Found 1 possible leak in 1 of 1 file (1 error, 0 warnings)" in result.output


def test_clean_code_exits_0_and_names_the_rules_checked(tmp_path: Path):
    result = runner.invoke(app, ["check", str(write(tmp_path / "ok.py", "x = 1\n"))])
    assert result.exit_code == 0
    assert "No leaks found in 1 file. Rules checked: FS001" in result.output


def test_no_files_found_is_not_an_error(tmp_path: Path):
    result = runner.invoke(app, ["check", str(tmp_path)])
    assert result.exit_code == 0
    assert "No leaks found in 0 files" in result.output


def test_unreadable_files_exit_2_but_the_rest_is_still_analyzed(tmp_path: Path):
    write(tmp_path / "broken.py", "def broken(:\n")
    write(tmp_path / "bad.ipynb", "{}")
    write(tmp_path / "ok.py", "x = 1\n")
    result = runner.invoke(app, ["check", str(tmp_path)])
    assert result.exit_code == 2
    assert "could not analyze" in result.output
    assert "broken.py: invalid Python on line 1" in result.output
    assert "bad.ipynb: not a Jupyter notebook" in result.output


def test_a_leak_outranks_an_unreadable_file(tmp_path: Path):
    write(tmp_path / "broken.py", "def broken(:\n")
    write(tmp_path / "train.py", LEAK)
    assert runner.invoke(app, ["check", str(tmp_path)]).exit_code == 1


def test_non_utf8_python_file_is_reported_not_raised(tmp_path: Path):
    (tmp_path / "latin.py").write_bytes("x = 'caf\xe9'\n".encode("latin-1"))
    assert runner.invoke(app, ["check", str(tmp_path)]).exit_code == 2


def test_encoding_declaration_is_honoured(tmp_path: Path):
    source = "# -*- coding: latin-1 -*-\nx = 'caf\xe9'\n"
    (tmp_path / "latin.py").write_bytes(source.encode("latin-1"))
    assert runner.invoke(app, ["check", str(tmp_path)]).exit_code == 0


def test_json_output(tmp_path: Path):
    script = write(tmp_path / "train.py", LEAK)
    result = runner.invoke(app, ["check", str(script), "--format", "json"])
    assert result.exit_code == 1
    report = json.loads(result.output)
    assert report["files_checked"] == 1
    assert report["summary"] == {"errors": 1, "warnings": 0}
    assert "FS001" in report["rules_checked"]
    (finding,) = report["findings"]
    assert finding["rule"] == "FS001"
    assert finding["severity"] == "error"
    assert finding["path"] == str(script)
    assert (finding["cell"], finding["line"]) == (None, 5)
    assert finding["message"]
    assert finding["fix"]
    assert report["problems"] == []


def test_json_reports_problems_and_notebook_cells(tmp_path: Path):
    write(tmp_path / "broken.py", "def broken(:\n")
    cells = [{"cell_type": "code", "source": "a = (1"}, {"cell_type": "code", "source": LEAK}]
    write(tmp_path / "eda.ipynb", json.dumps({"cells": cells}))
    report = json.loads(runner.invoke(app, ["check", str(tmp_path), "--format", "json"]).output)
    assert report["findings"][0]["cell"] == 2
    problems = {Path(problem["path"]).name: problem for problem in report["problems"]}
    assert problems["eda.ipynb"]["skipped_cells"] == [1]
    assert "invalid Python" in problems["broken.py"]["error"]


def test_text_output_shows_notebook_cells_and_skipped_cells(tmp_path: Path):
    cells = [{"cell_type": "code", "source": "a = (1"}, {"cell_type": "code", "source": LEAK}]
    notebook = write(tmp_path / "eda.ipynb", json.dumps({"cells": cells}))
    output = runner.invoke(app, ["check", str(notebook)]).output
    assert f"{notebook}:cell 2:5: FS001 [error]" in output
    assert "skipped cell 1 (not valid Python)" in output


def test_skipped_folders_only_count_inside_the_given_path(tmp_path: Path):
    project = tmp_path / "venv" / "project"
    project.mkdir(parents=True)
    write(project / "train.py", "x = 1\n")
    assert collect_files([project]) == [project / "train.py"]


def test_corpus_through_the_cli():
    examples = Path(__file__).resolve().parent.parent / "examples"
    assert runner.invoke(app, ["check", str(examples / "clean")]).exit_code == 0
    assert runner.invoke(app, ["check", str(examples / "leaky")]).exit_code == 1
    assert runner.invoke(app, ["check", str(examples / "notebooks")]).exit_code == 1
