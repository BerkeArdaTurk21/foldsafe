"""Clean: rows from the project's own loader, scored by a CV helper on a Pipeline.

The encoder is a pipeline step, so it is refitted inside every training fold.
"""

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import make_pipeline

from titanic.data import load_raw
from titanic.features import GroupSurvivalEncoder

train = load_raw("data/raw/train.csv")
X = train.drop(columns="Survived")
y = train["Survived"]
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)


def evaluate(model, data=None):
    scores = cross_validate(model, X if data is None else data, y, cv=cv)
    return scores["test_score"].mean()


print(evaluate(make_pipeline(GroupSurvivalEncoder(), LogisticRegression(max_iter=1000))))
