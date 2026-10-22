"""Clean: split first, then compute the fill value on the training frame only."""

import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv("train.csv")
train_df, test_df = train_test_split(df, test_size=0.2, random_state=0)

age_median = train_df["Age"].median()
train_df["Age"] = train_df["Age"].fillna(age_median)
test_df["Age"] = test_df["Age"].fillna(age_median)
