"""Leaky: a hand-written z-score uses the mean and std of the whole column."""

import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

df = pd.read_csv("houses.csv")
df["area_z"] = (df["area"] - df["area"].mean()) / df["area"].std()  # foldsafe-expect: FS005

X = df[["area_z", "rooms"]]
y = df["price"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=3)
print(LinearRegression().fit(X_train, y_train).score(X_test, y_test))
