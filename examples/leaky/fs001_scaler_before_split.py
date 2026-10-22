"""Leaky: the scaler learns mean/std from every row, including future test rows."""

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("train.csv")
X = df.drop(columns=["target"])
y = df["target"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)  # foldsafe-expect: FS001

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, random_state=42)
model = LogisticRegression().fit(X_train, y_train)
print(model.score(X_test, y_test))
