"""Clean: the selector sees training labels only and is then applied to the test rows."""

import pandas as pd
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

df = pd.read_csv("genes.csv")
X = df.drop(columns=["label"])
y = df["label"]

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
selector = SelectKBest(mutual_info_classif, k=20)
X_train_selected = selector.fit_transform(X_train, y_train)
X_test_selected = selector.transform(X_test)

model = LogisticRegression(max_iter=1000).fit(X_train_selected, y_train)
print(model.score(X_test_selected, y_test))
