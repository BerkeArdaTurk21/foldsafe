"""Clean: a helper fits the scaler on the training part and returns both parts transformed."""

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def scale(train_part, test_part):
    scaler = StandardScaler().fit(train_part)
    return scaler.transform(train_part), scaler.transform(test_part)


df = pd.read_csv("churn.csv")
X = df.drop(columns="churn")
y = df["churn"]
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
X_train, X_test = scale(X_train, X_test)
model = LogisticRegression().fit(X_train, y_train)
print(model.score(X_test, y_test))
