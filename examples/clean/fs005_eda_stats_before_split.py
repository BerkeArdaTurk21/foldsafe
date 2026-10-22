"""Clean (near miss): full-data statistics for exploration only, never stored as features."""

import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv("train.csv")
print(df.describe())
print("Median age:", df["Age"].median())
print(df.groupby("Pclass")["Survived"].mean())

train_df, test_df = train_test_split(df, test_size=0.2, random_state=0)
