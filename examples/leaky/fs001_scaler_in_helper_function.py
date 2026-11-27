"""Leaky: a preprocessing helper fits the scaler on all rows, and its output is split later."""

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def preprocess(frame):
    scaler = StandardScaler()
    return scaler.fit_transform(frame)  # foldsafe-expect: FS001


df = pd.read_csv("churn.csv")
X = preprocess(df.drop(columns="churn"))
y = df["churn"]
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
model = LogisticRegression().fit(X_train, y_train)
print(model.score(X_test, y_test))
