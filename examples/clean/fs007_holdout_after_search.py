"""Clean: tune on the training rows, report the score on a held-out test set."""

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, train_test_split

df = pd.read_csv("train.csv")
X = df.drop(columns=["target"])
y = df["target"]

X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=0)
search = GridSearchCV(RandomForestClassifier(random_state=0), {"max_depth": [3, 5, None]}, cv=5)
search.fit(X_train, y_train)
print("Chosen:", search.best_params_)
print(f"Held-out accuracy: {search.score(X_test, y_test):.3f}")
