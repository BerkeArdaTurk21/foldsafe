"""Clean: the scaler is inside the Pipeline, so cross_val_score refits it per fold."""

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("train.csv")
X = df.drop(columns=["target"])
y = df["target"]

pipeline = make_pipeline(StandardScaler(), LogisticRegression())
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(pipeline, X, y, cv=cv)
print(f"CV accuracy: {scores.mean():.3f} +/- {scores.std():.3f}")
