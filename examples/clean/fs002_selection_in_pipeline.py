"""Clean: feature selection is a Pipeline step, so each CV fold selects on its own rows."""

import pandas as pd
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline

df = pd.read_csv("genes.csv")
X = df.drop(columns=["label"])
y = df["label"]

pipeline = Pipeline(
    [("select", SelectKBest(f_classif, k=20)), ("model", LogisticRegression(max_iter=1000))]
)
print(cross_val_score(pipeline, X, y, cv=5).mean())
