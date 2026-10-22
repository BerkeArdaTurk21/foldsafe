"""Leaky: a group survival rate is computed from every row's label, then cross-validated.

Each row's own label is part of its feature. On Kaggle Titanic a feature like this
looks plausible and inflates CV accuracy (see README).
"""

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

df = pd.read_csv("train.csv")
df["ticket_survival"] = df.groupby("Ticket")["Survived"].transform("mean")  # foldsafe-expect: FS005

X = df[["Pclass", "Fare", "ticket_survival"]]
y = df["Survived"]
print(cross_val_score(LogisticRegression(), X, y, cv=5).mean())
