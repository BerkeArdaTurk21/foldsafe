"""Clean (near miss): scaled rows are wrapped in DataFrames using the training columns.

Reading X_train.columns while wrapping the test rows does not mix any rows.
"""

import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("houses.csv")
X = df.drop(columns=["price"])
y = df["price"]

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
scaler = StandardScaler().fit(X_train)
X_train_scaled = pd.DataFrame(scaler.transform(X_train), columns=X_train.columns)
X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X_train.columns)

model = Ridge().fit(X_train_scaled, y_train)
print(model.score(X_test_scaled, y_test))
