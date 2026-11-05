"""Leaky: a custom target encoder from the project's own module, fitted before CV.

The class is unknown to foldsafe, but it is fitted on every row and label. This is the
Titanic group-survival leak: 0.844 here, 0.841 when the encoder is inside the Pipeline.
"""

import pandas as pd
from my_project.features import GroupSurvivalEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_validate

train = pd.read_csv("train.csv")
X = train.drop(columns="Survived")
y = train["Survived"]

X_encoded = GroupSurvivalEncoder().fit_transform(X, y)  # foldsafe-expect: FS002
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_validate(LogisticRegression(max_iter=1000), X_encoded, y, cv=cv)
print(scores["test_score"].mean())
