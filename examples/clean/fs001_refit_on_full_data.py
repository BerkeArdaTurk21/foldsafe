"""Clean (near miss): after cross-validating a Pipeline, refit it on all labeled rows.

Fitting on the full training set is correct here: evaluation is already done and
no split follows. The final model predicts a separate, unlabeled test file.
"""

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")
X = train.drop(columns=["target"])
y = train["target"]

pipeline = make_pipeline(StandardScaler(), LogisticRegression())
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
print(cross_val_score(pipeline, X, y, cv=cv).mean())

pipeline.fit(X, y)
pd.Series(pipeline.predict(test)).to_csv("submission.csv", index=False)
