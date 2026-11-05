"""Leaky: the scaled array is wrapped back into a DataFrame to keep the column names.

Very common in notebooks. The wrapper changes nothing: the rows were already scaled with
statistics from every row. np.log1p afterwards does not change it either.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

df = pd.read_csv("houses.csv")
X = pd.get_dummies(df.drop(columns=["price"]))
y = np.log1p(df["price"])

X_scaled = pd.DataFrame(MinMaxScaler().fit_transform(X), columns=X.columns)  # foldsafe-expect: FS001
X_scaled = np.sqrt(X_scaled)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, random_state=0)
print(Ridge().fit(X_train, y_train).score(X_test, y_test))
