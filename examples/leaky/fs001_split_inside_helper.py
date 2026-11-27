"""Leaky: the data is scaled on all rows, then a helper does the train/test split."""

import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler


def make_splits(features, target):
    return train_test_split(features, target, test_size=0.2, random_state=0)


df = pd.read_csv("house_prices.csv")
X_scaled = MinMaxScaler().fit_transform(df.drop(columns="price"))  # foldsafe-expect: FS001
X_train, X_test, y_train, y_test = make_splits(X_scaled, df["price"])
print(Ridge().fit(X_train, y_train).score(X_test, y_test))
