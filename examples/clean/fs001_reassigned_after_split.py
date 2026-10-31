"""Clean (near miss): names are reused, but the scaler only ever sees training rows."""

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler


def main():
    df = pd.read_csv("train.csv")
    X = df.drop(columns=["target"])
    y = df["target"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
    scaler = RobustScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    print(LogisticRegression().fit(X_train, y_train).score(X_test, y_test))


if __name__ == "__main__":
    main()
