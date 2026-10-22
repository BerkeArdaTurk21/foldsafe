"""Leaky (likely): shuffled KFold on time-ordered data validates on the past using the future."""

import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold, cross_val_score

df = pd.read_csv("energy.csv")
df["timestamp"] = pd.to_datetime(df["timestamp"])
X = df[["temperature", "hour", "is_holiday"]]
y = df["load"]

cv = KFold(n_splits=5, shuffle=True, random_state=0)  # foldsafe-expect: FS008
print(cross_val_score(Ridge(), X, y, cv=cv).mean())
