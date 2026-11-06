"""Clean (near miss): test rows and training statistics meet, but no test rows reach training.

The fill value and the baseline both come from the training rows. A statistic is not
rows, so combining it with test rows does not mix train and test.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split

df = pd.read_csv("houses.csv")
X = df.drop(columns=["price"])
y = df["price"]
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

X_train_filled = X_train.fillna(X_train.median())
X_test_filled = X_test.fillna(X_train.median())

baseline_error = np.abs(y_test - y_train.mean()).mean()
model = Ridge().fit(X_train_filled, y_train)
print(baseline_error, model.score(X_test_filled, y_test))
