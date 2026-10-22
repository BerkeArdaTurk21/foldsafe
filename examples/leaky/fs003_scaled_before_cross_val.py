"""Leaky: rows are scaled with statistics from every fold, then cross-validated."""

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("train.csv")
X = df.drop(columns=["target"])
y = df["target"]

X_scaled = StandardScaler().fit_transform(X)
scores = cross_val_score(LogisticRegression(), X_scaled, y, cv=5)  # foldsafe-expect: FS003
print(f"CV accuracy: {scores.mean():.3f}")
