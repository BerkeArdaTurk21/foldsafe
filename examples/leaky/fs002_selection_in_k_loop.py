"""Leaky: trying several k values, each selection learns from the labels of all rows.

The loop picks the k with the best CV score, but every fold's validation labels
already shaped which features were kept.
"""

import pandas as pd
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

df = pd.read_csv("genes.csv")
X = df.drop(columns=["label"])
y = df["label"]

for k in (10, 20, 50):
    X_k = SelectKBest(f_classif, k=k).fit_transform(X, y)  # foldsafe-expect: FS002
    scores = cross_val_score(LogisticRegression(max_iter=1000), X_k, y, cv=5)
    print(k, scores.mean())
