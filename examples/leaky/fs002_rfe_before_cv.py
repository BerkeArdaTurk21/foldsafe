"""Leaky: RFE picks features with all labels, then CV scores the already chosen subset."""

import pandas as pd
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

df = pd.read_csv("genes.csv")
X = df.drop(columns=["label"])
y = df["label"]

selector = RFE(LogisticRegression(max_iter=1000), n_features_to_select=10)
selector.fit(X, y)  # foldsafe-expect: FS002
X_selected = selector.transform(X)

scores = cross_val_score(LogisticRegression(max_iter=1000), X_selected, y, cv=5)
print(scores.mean())
