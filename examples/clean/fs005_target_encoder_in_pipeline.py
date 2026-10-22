"""Clean: target encoding is a Pipeline step, so each fold encodes with its own labels."""

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import TargetEncoder

df = pd.read_csv("train.csv")
X = df[["Ticket", "Pclass", "Fare"]]
y = df["Survived"]

encode = ColumnTransformer([("ticket", TargetEncoder(), ["Ticket"])], remainder="passthrough")
pipeline = Pipeline([("encode", encode), ("model", LogisticRegression())])
print(cross_val_score(pipeline, X, y, cv=5).mean())
