"""Clean: GridSearchCV tunes a Pipeline, so imputation is learned inside each fold."""

import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline

df = pd.read_csv("train.csv")
X = df[["Age", "Fare", "Pclass", "SibSp"]]
y = df["Survived"]

pipeline = Pipeline(
    [("impute", SimpleImputer(strategy="mean")), ("model", GradientBoostingClassifier())]
)
search = GridSearchCV(pipeline, {"model__max_depth": [2, 3, 4]}, cv=5)
search.fit(X, y)
print(search.best_params_)
