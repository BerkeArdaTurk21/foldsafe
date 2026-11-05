"""Leaky: the function form of StandardScaler looks harmless but is a fit_transform on all rows."""

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import scale

df = pd.read_csv("train.csv")
X = df.drop(columns=["target"])
y = df["target"]

X_scaled = scale(X)  # foldsafe-expect: FS001
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, random_state=0)
print(LogisticRegression().fit(X_train, y_train).score(X_test, y_test))
