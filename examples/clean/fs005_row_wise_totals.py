"""Clean (near miss): row-wise sums and means use only each row's own values."""

import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split

df = pd.read_csv("exam_scores.csv")
subjects = ["math", "reading", "writing"]
df["total"] = df[subjects].sum(axis=1)
df["average"] = df[subjects].mean(axis="columns")

X = df[["total", "average", "study_hours"]]
y = df["final_grade"]
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
print(GradientBoostingRegressor().fit(X_train, y_train).score(X_test, y_test))
