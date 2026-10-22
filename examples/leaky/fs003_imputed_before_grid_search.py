"""Leaky: imputation uses all rows, then GridSearchCV tunes on the imputed data."""

import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GridSearchCV

df = pd.read_csv("train.csv")
X = df[["Age", "Fare", "Pclass", "SibSp"]]
y = df["Survived"]

X_imputed = SimpleImputer(strategy="mean").fit_transform(X)
search = GridSearchCV(GradientBoostingClassifier(), {"max_depth": [2, 3, 4]}, cv=5)
search.fit(X_imputed, y)  # foldsafe-expect: FS003
print(search.best_params_)
