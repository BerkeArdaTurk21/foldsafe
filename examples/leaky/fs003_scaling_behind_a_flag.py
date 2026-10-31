"""Leaky: scaling is switched on by a config flag, then the model is cross-validated.

The scaled matrix replaces X inside the if block, so the CV call below receives
data whose statistics came from every fold.
"""

import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

CONFIG = {"scale": True}

df = pd.read_csv("train.csv")
X = df.drop(columns=["target"])
y = df["target"]

if CONFIG["scale"]:
    X = StandardScaler().fit_transform(X)

scores = cross_val_score(SVC(), X, y, cv=5)  # foldsafe-expect: FS003
print(scores.mean())
