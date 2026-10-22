"""Leaky: train and test are concatenated and the fill value is computed on both.

A common Kaggle habit. In production the test rows do not exist yet.
"""

import pandas as pd

train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

combined = pd.concat([train, test], ignore_index=True)
combined["Age"] = combined["Age"].fillna(combined["Age"].median())  # foldsafe-expect: FS004

train = combined.iloc[: len(train)]
test = combined.iloc[len(train) :]
