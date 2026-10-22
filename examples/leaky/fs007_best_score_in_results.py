"""Leaky (optimistic): best_score_ is saved as the model's final metric."""

import json

import pandas as pd
from sklearn.model_selection import RandomizedSearchCV
from sklearn.svm import SVC

df = pd.read_csv("train.csv")
X = df.drop(columns=["target"])
y = df["target"]

search = RandomizedSearchCV(SVC(), {"C": [0.1, 1, 10], "gamma": ["scale", 0.01]}, n_iter=5, cv=5)
search.fit(X, y)
results = {"model": "svc", "accuracy": search.best_score_}  # foldsafe-expect: FS007
with open("results.json", "w") as handle:
    json.dump(results, handle)
