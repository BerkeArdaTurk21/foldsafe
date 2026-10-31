"""Leaky: a helper refits the encoder on the test rows it was meant to only transform."""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

df = pd.read_csv("train.csv")
X = df[["Embarked", "Sex", "Pclass"]]
y = df["Survived"]
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)


def encode_splits():
    encoder = OneHotEncoder(handle_unknown="ignore")
    train_encoded = encoder.fit_transform(X_train)
    test_encoded = encoder.fit_transform(X_test)  # foldsafe-expect: FS004
    return train_encoded, test_encoded
