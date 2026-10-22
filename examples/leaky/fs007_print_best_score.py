"""Leaky (optimistic): the best of many settings on the same folds is reported as the result."""

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

df = pd.read_csv("train.csv")
X = df.drop(columns=["target"])
y = df["target"]

grid = {"max_depth": [3, 5, 8, None], "n_estimators": [100, 300]}
search = GridSearchCV(RandomForestClassifier(random_state=0), grid, cv=5)
search.fit(X, y)
print(f"Model accuracy: {search.best_score_:.3f}")  # foldsafe-expect: FS007
