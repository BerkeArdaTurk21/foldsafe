"""Clean: the fill value comes from the training file and is reused for the test file."""

import pandas as pd

train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

age_median = train["Age"].median()
train["Age"] = train["Age"].fillna(age_median)
test["Age"] = test["Age"].fillna(age_median)
