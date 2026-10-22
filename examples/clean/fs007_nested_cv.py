"""Clean: nested CV; the outer loop scores the whole search procedure on unseen folds."""

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_score

df = pd.read_csv("train.csv")
X = df.drop(columns=["target"])
y = df["target"]

inner_cv = StratifiedKFold(n_splits=4, shuffle=True, random_state=0)
outer_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=1)
grid = {"max_depth": [3, 5, None]}
search = GridSearchCV(RandomForestClassifier(random_state=0), grid, cv=inner_cv)
scores = cross_val_score(search, X, y, cv=outer_cv)
print(f"Nested CV accuracy: {scores.mean():.3f} +/- {scores.std():.3f}")
