"""Leaky: the split happens first, but the scaler is then fitted on all rows anyway.

It looks correct because the split comes before the scaler and the test rows only go
through transform(). The scaler has still seen them: the fit used X, not X_train.
"""

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("train.csv")
X = df.drop(columns=["target"])
y = df["target"]

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
scaler = StandardScaler().fit(X)  # foldsafe-expect: FS001
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = LogisticRegression().fit(X_train_scaled, y_train)
print(model.score(X_test_scaled, y_test))
