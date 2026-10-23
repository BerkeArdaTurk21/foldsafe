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
