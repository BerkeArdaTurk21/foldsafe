"""Leaky: the median used to fill Age is computed on all rows before the split."""

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

df = pd.read_csv("train.csv")
df["Age"] = df["Age"].fillna(df["Age"].median())  # foldsafe-expect: FS005

X = df[["Age", "Fare", "Pclass"]]
y = df["Survived"]
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
print(RandomForestClassifier(random_state=0).fit(X_train, y_train).score(X_test, y_test))
