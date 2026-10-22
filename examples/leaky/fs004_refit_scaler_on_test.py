"""Leaky: the scaler is refitted on the test rows instead of reusing the training fit."""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler

df = pd.read_csv("train.csv")
X = df.drop(columns=["target"])
y = df["target"]

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=7)
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.fit_transform(X_test)  # foldsafe-expect: FS004

model = KNeighborsClassifier().fit(X_train_scaled, y_train)
print(model.score(X_test_scaled, y_test))
