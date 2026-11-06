"""Read Jupyter notebooks: join the code cells into one module and map lines back to cells.

Cells are joined in file order. Notebooks are often run out of order, but file order is what
"Run all" does and what a reader sees. Only the JSON is read; nothing is executed.
"""

from __future__ import annotations

import ast
import json
import re
from dataclasses import dataclass

# Cell magics whose body is still Python: only the first line is dropped.
PYTHON_CELL_MAGICS = {"time", "timeit", "capture", "prun"}
# `%time model.fit(X, y)`: the rest of the line is Python.
TIMED_LINE = re.compile(r"^(\s*)%(?:time|timeit)\s+(?:-\w+\s+(?:\d+\s+)?)*(.*)$")
# `files = !ls` and `result = %env PATH`: IPython assigning from a shell or magic call.
MAGIC_ASSIGNMENT = re.compile(r"^\s*[\w, ]+=\s*[!%]")


@dataclass(frozen=True)
class NotebookSource:
    source: str  # code cells joined; IPython-only lines are blank so numbering is kept
    locations: tuple[tuple[int, int], ...]  # per source line: (cell number, line in cell), from 1
    skipped_cells: tuple[int, ...]  # cells that are not valid Python, left out of the analysis

    def locate(self, line: int) -> tuple[int, int]:
        return self.locations[line - 1]

    def describe_line(self, line: int) -> str:
        cell, cell_line = self.locate(line)
        return f"cell {cell}, line {cell_line}"


def read_notebook(text: str) -> NotebookSource:
    """Parse notebook JSON. Raises ValueError when the text is not a v4 notebook."""
    try:
        notebook = json.loads(text)
    except json.JSONDecodeError as error:
        raise ValueError(f"not valid JSON: {error}") from error
    cells = notebook.get("cells") if isinstance(notebook, dict) else None
    if not isinstance(cells, list):
        raise ValueError("not a Jupyter notebook: it has no list of cells")

    lines: list[str] = []
    locations: list[tuple[int, int]] = []
    skipped: list[int] = []
    for number, cell in enumerate(cells, start=1):
        if not isinstance(cell, dict) or cell.get("cell_type") != "code":
            continue
        cell_lines = _python_lines(cell.get("source", ""))
        if not _is_valid_python(cell_lines):
            skipped.append(number)
            continue
        for cell_line, line in enumerate(cell_lines, start=1):
            lines.append(line)
            locations.append((number, cell_line))
    source = "\n".join(lines) + "\n" if lines else ""
    return NotebookSource(source, tuple(locations), tuple(skipped))


def _python_lines(source: str | list[str]) -> list[str]:
    """The cell's lines, with IPython-only syntax blanked: %magic, !shell, help?."""
    text = "".join(source) if isinstance(source, list) else source
    lines = text.splitlines()
    first_code = next((line.strip() for line in lines if line.strip()), "")
    if first_code.startswith("%%"):
        magic = first_code[2:].split(maxsplit=1)[0] if first_code[2:].strip() else ""
        if magic not in PYTHON_CELL_MAGICS:
            return [""] * len(lines)  # %%bash, %%html, %%sql: the whole cell is not Python
        lines = ["" if line.strip().startswith("%%") else line for line in lines]
    return [_python_line(line) for line in lines]


def _python_line(line: str) -> str:
    timed = TIMED_LINE.match(line)
    if timed:
        return timed.group(1) + timed.group(2)
    stripped = line.strip()
    if stripped.startswith(("%", "!")) or stripped.endswith("?") or MAGIC_ASSIGNMENT.match(line):
        return ""
    return line


def _is_valid_python(lines: list[str]) -> bool:
    try:
        ast.parse("\n".join(lines))
    except SyntaxError:
        return False
    return True
