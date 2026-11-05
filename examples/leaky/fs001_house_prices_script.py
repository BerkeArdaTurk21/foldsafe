"""Leaky: a typical regression notebook turned into a script, with two leaks before the split.

The fill value comes from all rows and the scaler is fitted on all rows. The model and
the plotting code around them look completely normal.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

SEED = 42

houses = pd.read_csv("data/houses.csv")
print(houses.shape)
print(houses.isna().mean().sort_values(ascending=False).head(10))

houses = houses[houses["GrLivArea"] < 4000]
houses["LotFrontage"] = houses["LotFrontage"].fillna(houses["LotFrontage"].median())  # foldsafe-expect: FS005
houses["TotalArea"] = houses["GrLivArea"] + houses["TotalBsmtSF"]
houses["Age"] = houses["YrSold"] - houses["YearBuilt"]

features = ["LotFrontage", "TotalArea", "Age", "OverallQual", "GarageCars"]
X = houses[features]
y = np.log1p(houses["SalePrice"])

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)  # foldsafe-expect: FS001

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=SEED)

model = GradientBoostingRegressor(n_estimators=300, learning_rate=0.05, random_state=SEED)
model.fit(X_train, y_train)
predictions = model.predict(X_test)
print("MAE:", mean_absolute_error(np.expm1(y_test), np.expm1(predictions)))

plt.scatter(y_test, predictions, s=5)
plt.savefig("predictions.png")
