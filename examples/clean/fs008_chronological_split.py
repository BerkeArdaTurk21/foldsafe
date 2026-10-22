"""Clean: rows are sorted by date and split without shuffling, so the test set is the future."""

import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split

df = pd.read_csv("sales.csv", parse_dates=["date"]).sort_values("date")
X = df[["store", "promo", "day_of_week"]]
y = df["sales"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
print(GradientBoostingRegressor().fit(X_train, y_train).score(X_test, y_test))
