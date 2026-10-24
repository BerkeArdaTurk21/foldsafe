"""Events the data-flow pass records for the rule checkers."""

from __future__ import annotations

from dataclasses import dataclass

from foldsafe.values import DataValue


@dataclass(frozen=True)
class SplitEvent:
    """Rows are divided into train and test: train_test_split or cv.split()."""

    line: int
    kind: str  # "train_test_split" | "cv_split"
    inputs: tuple[DataValue, ...]  # the data that was split, as it was at that moment
    shuffled: bool


@dataclass(frozen=True)
class CVEvent:
    """Data enters cross-validation: cross_val_score(...) or search.fit(X, y)."""

    line: int
    consumer: str  # qualified function name, or "<SearchClass>.fit"
    data: DataValue | None  # the X argument; None when it is not tracked
    shuffled: bool  # the cv splitter shuffles rows


Event = SplitEvent | CVEvent
