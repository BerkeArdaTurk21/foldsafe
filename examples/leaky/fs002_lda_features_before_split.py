"""Leaky: LDA is a classifier, but here its fit_transform output is used as features.

It learns its projection from the labels of every row, including the future test rows.
"""

import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

df = pd.read_csv("train.csv")
X = df.drop(columns=["target"])
y = df["target"]

X_projected = LinearDiscriminantAnalysis().fit_transform(X, y)  # foldsafe-expect: FS002
X_train, X_test, y_train, y_test = train_test_split(X_projected, y, random_state=0)
print(RandomForestClassifier(random_state=0).fit(X_train, y_train).score(X_test, y_test))
