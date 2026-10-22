"""Leaky: SelectKBest ranks features using labels of rows that later become test rows."""

import pandas as pd
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

df = pd.read_csv("genes.csv")
X = df.drop(columns=["label"])
y = df["label"]

X_selected = SelectKBest(f_classif, k=20).fit_transform(X, y)  # foldsafe-expect: FS002
X_train, X_test, y_train, y_test = train_test_split(X_selected, y, random_state=0)
print(LogisticRegression(max_iter=1000).fit(X_train, y_train).score(X_test, y_test))
