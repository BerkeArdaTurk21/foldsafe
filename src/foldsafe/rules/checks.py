"""Rule checkers. Each reads the events of the flow pass and returns findings.

One leak produces one finding, from the most specific rule (examples/README.md):
the taint kind of a full-data fit decides between FS001/FS003 (preprocess) and
FS002 (select), so the rules never report the same fit twice.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator

from foldsafe.events import SplitEvent
from foldsafe.findings import Finding
from foldsafe.flow import FlowState
from foldsafe.rules.info import RULES
from foldsafe.values import Taint

Checker = Callable[[FlowState, str], Iterable[Finding]]


def finding(code: str, path: str, line: int, message: str) -> Finding:
    return Finding(code, path, line, message, RULES[code].severity)


def split_taints(state: FlowState) -> set[Taint]:
    """Taints carried by data at the moment it is split into train and test rows."""
    taints: set[Taint] = set()
    for event in state.events:
        if isinstance(event, SplitEvent):
            for value in event.inputs:
                taints |= value.taints
    return taints


def check_fs001(state: FlowState, path: str) -> Iterator[Finding]:
    """A preprocessor fitted on all rows, whose output is split afterwards."""
    leaked = {taint for taint in split_taints(state) if taint.kind == "preprocess"}
    for taint in sorted(leaked, key=lambda taint: taint.line):
        yield finding(
            "FS001",
            path,
            taint.line,
            "Preprocessor is fitted on all rows before the train/test split, so the test rows "
            "shape what it learns. Split first, then fit it on the training rows (or use a "
            "Pipeline).",
        )


# Implemented checkers, keyed by rule code. Filled in as rules land (M2, M5).
CHECKERS: dict[str, Checker] = {
    "FS001": check_fs001,
}
