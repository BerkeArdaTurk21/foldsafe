"""Leaky: inside main(), X is overwritten with its scaled version and then split.

Reassigning the same name hides the leak: by the time of the split, X already
carries statistics from every row.
"""

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler


def main():
    df = pd.read_csv("train.csv")
    X = df.drop(columns=["target"])
    y = df["target"]
    X = RobustScaler().fit_transform(X)  # foldsafe-expect: FS001
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
    print(LogisticRegression().fit(X_train, y_train).score(X_test, y_test))


if __name__ == "__main__":
    main()
