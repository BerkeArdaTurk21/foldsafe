"""A finding is one leakage report tied to a source location."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class Severity(str, Enum):
    ERROR = "error"  # leakage is certain from the code alone
    WARNING = "warning"  # likely leakage; depends on intent or data


@dataclass(frozen=True)
class Finding:
    rule: str
    path: str
    line: int
    message: str  # what was seen here and why it leaks
    severity: Severity
    fix: str = ""  # how to correct it, usually with a Pipeline
