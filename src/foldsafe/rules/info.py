"""Rule catalog: code, title, severity, reason and fix of every specified rule.

`why` and `fix` are one or two sentences each. Every finding carries its rule's fix
(a checker may pass a more specific one); docs/rules.md expands both with examples.
"""

from __future__ import annotations

from dataclasses import dataclass

from foldsafe.findings import Severity


@dataclass(frozen=True)
class RuleInfo:
    code: str
    title: str
    severity: Severity
    why: str  # why the pattern leaks, independent of the code at hand
    fix: str  # the Pipeline-based correction, with a short code pattern


RULES: dict[str, RuleInfo] = {
    rule.code: rule
    for rule in (
        RuleInfo(
            "FS001",
            "Preprocessor fitted on the full dataset before the train/test split",
            Severity.ERROR,
            why="The preprocessor learns its statistics (mean, scale, categories, components) "
            "from the test rows too, so the test rows are no longer unseen.",
            fix="Split first and fit on the training rows only, or put the step in a Pipeline: "
            "make_pipeline(StandardScaler(), model).fit(X_train, y_train).",
        ),
        RuleInfo(
            "FS002",
            "Target-aware feature selection fitted outside the split or CV loop",
            Severity.ERROR,
            why="The selector (or target encoder) looks at the labels of the evaluation rows, "
            "so the chosen features are tuned to the answers the model is scored on.",
            fix="Put the selector inside the Pipeline and split or cross-validate the Pipeline: "
            "cross_val_score(make_pipeline(SelectKBest(k=10), model), X, y).",
        ),
        RuleInfo(
            "FS003",
            "Data preprocessed on all rows is passed into cross-validation or a search",
            Severity.ERROR,
            why="Every validation fold was already seen by the preprocessing, so the CV score "
            "is measured on rows that shaped the features.",
            fix="Pass the preprocessing and the model to cross-validation as one Pipeline: "
            "cross_val_score(make_pipeline(StandardScaler(), model), X, y).",
        ),
        RuleInfo(
            "FS004",
            "Test data used for fitting, or statistics computed on train + test combined",
            Severity.ERROR,
            why="Whatever is learned from the test rows is no longer a fair measure of how the "
            "model handles new data.",
            fix="Fit on the training rows and only transform or predict the test rows; a "
            "Pipeline does both: pipe.fit(X_train, y_train); pipe.predict(X_test).",
        ),
        RuleInfo(
            "FS005",
            "pandas statistic on the full frame stored as a feature before the split",
            Severity.ERROR,
            why="A median, mean or group target rate of the whole frame includes the "
            "evaluation rows (and, for target means, their own labels).",
            fix="Use a transformer inside the Pipeline so the statistic is learned per fold: "
            "make_pipeline(SimpleImputer(strategy='median'), model), or TargetEncoder().",
        ),
        RuleInfo(
            "FS006",
            "Oversampling applied before the split or outside an imblearn Pipeline",
            Severity.ERROR,
            why="Oversampling copies or interpolates minority rows, so near-duplicates of the "
            "evaluation rows end up in training.",
            fix="Oversample inside an imblearn Pipeline, which resamples training folds only: "
            "imblearn.pipeline.make_pipeline(SMOTE(), model).",
        ),
        RuleInfo(
            "FS007",
            "best_score_ from a hyperparameter search reported as the final score",
            Severity.WARNING,
            why="best_score_ is the best of many settings on the same folds, so it is biased "
            "upwards; it chose the model but cannot also grade it.",
            fix="Score the tuned Pipeline on a held-out test set, or use nested CV: "
            "cross_val_score(GridSearchCV(pipe, grid), X, y).",
        ),
        RuleInfo(
            "FS008",
            "Shuffled split or KFold on data that has a date column",
            Severity.WARNING,
            why="With time-ordered rows, a shuffled split trains on the future to predict the "
            "past, which the model cannot do in production.",
            fix="Split by time: cross_val_score(pipe, X, y, cv=TimeSeriesSplit(5)), or sort by "
            "date and use train_test_split(..., shuffle=False).",
        ),
    )
}
