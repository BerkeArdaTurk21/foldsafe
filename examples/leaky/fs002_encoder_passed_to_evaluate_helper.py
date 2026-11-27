"""Leaky: a group-survival encoder learns from y on all rows, then a CV helper scores it.

The rows come from the project's own loader and reach cross-validation through a helper,
as in a Titanic notebook. Fitted inside the folds, the same feature scores lower.
"""

from pathlib import Path

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_validate

from titanic.data import load_raw
from titanic.features import GroupSurvivalEncoder

DATA_DIR = Path("data/raw")
train = load_raw(DATA_DIR / "train.csv")
X = train.drop(columns="Survived")
y = train["Survived"]
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)


def evaluate(model, data=None):
    """Score the model with 5-fold CV on the shared folds."""
    scores = cross_validate(model, X if data is None else data, y, cv=cv)
    return scores["test_score"].mean()


x_leak = GroupSurvivalEncoder().fit_transform(X, y)  # foldsafe-expect: FS002
print(evaluate(LogisticRegression(max_iter=1000), data=x_leak))
