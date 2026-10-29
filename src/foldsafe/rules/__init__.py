"""Rule catalog and checker registry.

A rule is listed in RULES as soon as it is specified (with corpus examples).
It becomes active once its checker is registered in CHECKERS.
"""

from __future__ import annotations

from foldsafe.rules.checks import CHECKERS, Checker
from foldsafe.rules.info import RULES, RuleInfo

ACTIVE_RULES: frozenset[str] = frozenset(CHECKERS)

__all__ = ["ACTIVE_RULES", "CHECKERS", "RULES", "Checker", "RuleInfo"]
