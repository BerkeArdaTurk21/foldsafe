"""Clean (near miss): a hand-written CV loop fits the scaler on each fold's training rows.

`fit_transform` appears next to the test rows, but only the training rows reach it.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("train.csv")
X = df.drop(columns=["target"])
y = df["target"]

scores = []
for train_idx, test_idx in StratifiedKFold(n_splits=5, shuffle=True, random_state=0).split(X, y):
    scaler = StandardScaler()
    X_fold_train = scaler.fit_transform(X.iloc[train_idx])
    X_fold_test = scaler.transform(X.iloc[test_idx])
    model = LogisticRegression().fit(X_fold_train, y.iloc[train_idx])
    scores.append(model.score(X_fold_test, y.iloc[test_idx]))
print(np.mean(scores))
