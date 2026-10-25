"""Abstract values the data-flow pass tracks for variables (see flow.py)."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class Origin(str, Enum):
    """Which rows a data value was computed from."""

    FULL = "full"  # all labeled rows, before any split
    TRAIN = "train"  # training side of a split
    TEST = "test"  # held-out side of a split, or a separate test file
    COMBINED = "combined"  # train and test rows mixed together


@dataclass(frozen=True)
class Taint:
    """A fit on full data whose output flowed into a value."""

    line: int
    kind: str  # "preprocess" | "select" | "resample" | "stat"


@dataclass(frozen=True)
class DataValue:
    """A dataset, column, array or anything derived from one."""

    origin: Origin
    taints: frozenset[Taint] = frozenset()


@dataclass(frozen=True)
class SplitterValue:
    """A cross-validation splitter object such as KFold(shuffle=True)."""

    cls: str  # qualified class name
    shuffled: bool  # rows are shuffled before they are assigned to folds


@dataclass(frozen=True)
class SearchValue:
    """A hyperparameter search object such as GridSearchCV(...)."""

    cls: str
    shuffled: bool  # its cv splitter shuffles rows


@dataclass(frozen=True)
class EstimatorValue:
    """An estimator object; `fit_*` fields are set once it has been fitted."""

    cls: str  # qualified class or factory name
    kind: str  # see catalog.py
    fit_origin: Origin | None = None  # rows it was last fitted on
    fit_taint: Taint | None = None  # left on transform output when fitted on full data


@dataclass(frozen=True)
class IndexValue:
    """Row positions of one side of a fold: `train_idx` or `test_idx`."""

    side: Origin  # TRAIN or TEST


# Transient values: they only exist while a statement is being evaluated.


@dataclass(frozen=True)
class DataTuple:
    """Several data values at once: train_test_split's parts or fit_resample's (X, y)."""

    parts: tuple[DataValue | None, ...]


@dataclass(frozen=True)
class FoldIndices:
    """What iterating over `cv.split(X)` yields: (train_idx, test_idx) pairs."""


@dataclass(frozen=True)
class Enumerated:
    """`enumerate(x)`: (counter, item) pairs."""

    item: FoldIndices


TrackedValue = DataValue | EstimatorValue | SplitterValue | SearchValue | IndexValue
Value = TrackedValue | DataTuple | FoldIndices | Enumerated


def merge_origins(first: Origin, second: Origin) -> Origin:
    """Origin of a value computed from two others."""
    if first == second:
        return first
    if Origin.COMBINED in (first, second) or Origin.TEST in (first, second):
        return Origin.COMBINED
    # FULL with TRAIN: the result still contains every labeled row.
    return Origin.FULL


def merge_data(values: list[DataValue]) -> DataValue | None:
    """Combine several data values into one; None when the list is empty."""
    if not values:
        return None
    origin = values[0].origin
    taints: frozenset[Taint] = frozenset()
    for value in values:
        origin = merge_origins(origin, value.origin)
        taints |= value.taints
    return DataValue(origin, taints)
