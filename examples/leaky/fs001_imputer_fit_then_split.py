"""Leaky: imputer medians come from all rows; fit and transform are separate calls."""

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split

df = pd.read_csv("train.csv")
X = df[["Age", "Fare", "Pclass"]]
y = df["Survived"]

imputer = SimpleImputer(strategy="median")
imputer.fit(X)  # foldsafe-expect: FS001
X_filled = imputer.transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_filled, y, test_size=0.2, random_state=0)
model = RandomForestClassifier(random_state=0).fit(X_train, y_train)
print(model.score(X_test, y_test))
