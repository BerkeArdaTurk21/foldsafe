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
    (tmp_path / "eda.ipynb").write_text("{}")
    (tmp_path / "notes.md").write_text("# not code\n")

    result = runner.invoke(app, ["check", str(tmp_path)])

    assert result.exit_code == 0
    assert "2 file(s) found" in result.output


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
