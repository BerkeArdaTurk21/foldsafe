"""Clean (near miss): these fit_transform calls learn nothing about the row values.

PolynomialFeatures only multiplies columns and LabelEncoder only numbers the class
names, so running them before the split cannot carry information from the test rows.
"""

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, PolynomialFeatures

df = pd.read_csv("train.csv")
X = PolynomialFeatures(degree=2).fit_transform(df[["age", "income"]])
y = LabelEncoder().fit_transform(df["segment"])

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
print(LogisticRegression(max_iter=1000).fit(X_train, y_train).score(X_test, y_test))
