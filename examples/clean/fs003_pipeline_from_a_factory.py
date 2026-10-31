"""Clean (near miss): the scaler is built inside a helper, but it ends up in a Pipeline."""

import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


def build_model(scale: bool):
    steps = [StandardScaler()] if scale else []
    return make_pipeline(*steps, SVC())


df = pd.read_csv("train.csv")
X = df.drop(columns=["target"])
y = df["target"]

for scale in (False, True):
    scores = cross_val_score(build_model(scale), X, y, cv=5)
    print(scale, scores.mean())
