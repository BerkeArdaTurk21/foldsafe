"""Rule catalog: code, title and severity of every specified rule."""

from __future__ import annotations

from dataclasses import dataclass

from foldsafe.findings import Severity


@dataclass(frozen=True)
class RuleInfo:
    code: str
    title: str
    severity: Severity


RULES: dict[str, RuleInfo] = {
    rule.code: rule
    for rule in (
        RuleInfo(
            "FS001",
            "Preprocessor fitted on the full dataset before the train/test split",
            Severity.ERROR,
        ),
        RuleInfo(
            "FS002",
            "Target-aware feature selection fitted outside the split or CV loop",
            Severity.ERROR,
        ),
        RuleInfo(
            "FS003",
            "Data preprocessed on all rows is passed into cross-validation or a search",
            Severity.ERROR,
        ),
        RuleInfo(
            "FS004",
            "Test data used for fitting, or statistics computed on train + test combined",
            Severity.ERROR,
        ),
        RuleInfo(
            "FS005",
            "pandas statistic on the full frame stored as a feature before the split",
            Severity.ERROR,
        ),
        RuleInfo(
            "FS006",
            "Oversampling applied before the split or outside an imblearn Pipeline",
            Severity.ERROR,
        ),
        RuleInfo(
            "FS007",
            "best_score_ from a hyperparameter search reported as the final score",
            Severity.WARNING,
        ),
        RuleInfo(
            "FS008",
            "Shuffled split or KFold on data that has a date column",
            Severity.WARNING,
        ),
    )
}
