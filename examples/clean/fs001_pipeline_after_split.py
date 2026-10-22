"""Clean: preprocessing lives inside a Pipeline that is fitted on the training rows."""

import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("train.csv")
X = df[["Age", "Fare", "Pclass"]]
y = df["Survived"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
model = Pipeline(
    [
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
        ("model", LogisticRegression()),
    ]
)
model.fit(X_train, y_train)
print(model.score(X_test, y_test))
