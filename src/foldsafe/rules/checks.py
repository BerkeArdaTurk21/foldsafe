"""Rule checkers. Each reads the events of the flow pass and returns findings.

One leak produces one finding, from the most specific rule (examples/README.md):
the taint kind of a full-data fit decides between FS001/FS003 (preprocess) and
FS002 (select), so the rules never report the same fit twice.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator

from foldsafe.events import CVEvent, FitEvent, SplitEvent, StatEvent
from foldsafe.findings import Finding
from foldsafe.flow import FlowState
from foldsafe.rules.info import RULES
from foldsafe.values import Origin, Taint

Checker = Callable[[FlowState, str], Iterable[Finding]]


def finding(code: str, path: str, line: int, message: str, fix: str | None = None) -> Finding:
    """A finding with the rule's severity and fix, unless a more specific fix is given."""
    rule = RULES[code]
    return Finding(code, path, line, message, rule.severity, fix or rule.fix)


def short_name(qualified: str) -> str:
    """sklearn.preprocessing.StandardScaler -> StandardScaler."""
    return qualified.rsplit(".", 1)[-1]


def split_taints(state: FlowState) -> set[Taint]:
    """Taints carried by data at the moment it is split into train and test rows."""
    taints: set[Taint] = set()
    for event in state.events:
        if isinstance(event, SplitEvent):
            for value in event.inputs:
                taints |= value.taints
    return taints


def cv_taints(state: FlowState) -> set[Taint]:
    """Taints carried by data that enters cross-validation or a hyperparameter search."""
    taints: set[Taint] = set()
    for event in state.events:
        if isinstance(event, CVEvent) and event.data is not None:
            taints |= event.data.taints
    return taints


def check_fs001(state: FlowState, path: str) -> Iterator[Finding]:
    """A preprocessor fitted on all rows, whose output is split afterwards."""
    leaked = {taint for taint in split_taints(state) if taint.kind == "preprocess"}
    for taint in sorted(leaked, key=lambda taint: taint.line):
        yield finding(
            "FS001",
            path,
            taint.line,
            "Preprocessor is fitted on all rows and its output is split afterwards, so the "
            "test rows of the train/test split shaped what it learned.",
        )


def check_fs002(state: FlowState, path: str) -> Iterator[Finding]:
    """A selector that learned from y on all rows, whose output is split or cross-validated."""
    reached = split_taints(state) | cv_taints(state)
    leaked = {taint for taint in reached if taint.kind == "select"}
    for taint in sorted(leaked, key=lambda taint: taint.line):
        yield finding(
            "FS002",
            path,
            taint.line,
            "Feature selection (or target encoding) learns from the labels of all rows, "
            "including the rows its output is later evaluated on.",
        )


def check_fs003(state: FlowState, path: str) -> Iterator[Finding]:
    """Data preprocessed on all rows enters cross-validation or a search.

    Reported on the CV call, once per call. A fit whose output is also split is
    already reported as FS001 and is skipped here.
    """
    already_split = split_taints(state)
    for event in state.events:
        if not isinstance(event, CVEvent) or event.data is None:
            continue
        leaked = {
            taint
            for taint in event.data.taints
            if taint.kind == "preprocess" and taint not in already_split
        }
        if leaked:
            fit_lines = ", ".join(str(line) for line in sorted(taint.line for taint in leaked))
            yield finding(
                "FS003",
                path,
                event.line,
                f"Data preprocessed on all rows (fitted on line {fit_lines}) is cross-validated, "
                "so every validation fold was seen by the preprocessing.",
            )


# FS004 for a statistic: a pandas fix fits better than the estimator-oriented default.
STAT_FIX = (
    "Compute the statistic on the training rows and reuse it for the test rows: median = "
    'train["Age"].median(); test["Age"] = test["Age"].fillna(median). Or use SimpleImputer '
    "in a Pipeline."
)


def check_fs004(state: FlowState, path: str) -> Iterator[Finding]:
    """Test rows used for fitting, or a statistic of train + test rows that is kept."""
    reported: set[int] = set()
    for event in state.events:
        if event.line in reported:
            continue
        if isinstance(event, FitEvent) and event.data is not None:
            if event.data.origin is Origin.TEST:
                reported.add(event.line)
                yield finding(
                    "FS004",
                    path,
                    event.line,
                    f"{short_name(event.cls)} is fitted on test rows, so the evaluation data "
                    "shapes what it learns.",
                )
            elif event.data.origin is Origin.COMBINED:
                reported.add(event.line)
                yield finding(
                    "FS004",
                    path,
                    event.line,
                    f"{short_name(event.cls)} is fitted on train and test rows together, so the "
                    "evaluation data shapes what it learns.",
                )
        elif isinstance(event, StatEvent) and event.stored and event.data.origin is Origin.COMBINED:
            reported.add(event.line)
            yield finding(
                "FS004",
                path,
                event.line,
                f"A statistic ({event.method}) is computed on train and test rows combined "
                "and kept, so the test rows shape the values the model sees.",
                STAT_FIX,
            )


# Implemented checkers, keyed by rule code. Filled in as rules land (M2, M5).
CHECKERS: dict[str, Checker] = {
    "FS001": check_fs001,
    "FS002": check_fs002,
    "FS003": check_fs003,
    "FS004": check_fs004,
}
