"""Clean (near miss): "latest" and "contest" contain the letters "test" but are not test files."""

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler

prices = pd.read_csv("data/latest_prices.csv")
features = prices[["rooms", "area"]]
labels = prices["price"]

scaler = StandardScaler().fit(features)
model = RandomForestRegressor(random_state=0).fit(scaler.transform(features), labels)

entries = pd.read_csv("data/contest_entries.csv")
print(model.predict(scaler.transform(entries[["rooms", "area"]])))
