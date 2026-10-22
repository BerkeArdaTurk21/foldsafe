"""Clean: TimeSeriesSplit always validates on rows that come after the training rows."""

import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import TimeSeriesSplit, cross_val_score

df = pd.read_csv("energy.csv", parse_dates=["timestamp"]).sort_values("timestamp")
X = df[["temperature", "hour", "is_holiday"]]
y = df["load"]

cv = TimeSeriesSplit(n_splits=5)
print(cross_val_score(Ridge(), X, y, cv=cv).mean())
