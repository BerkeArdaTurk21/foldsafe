"""Clean: a forecast that respects time: sorted rows, chronological split, fit on the past."""

import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import TimeSeriesSplit, cross_val_score, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

demand = pd.read_csv("demand.csv", parse_dates=["timestamp"]).sort_values("timestamp")
demand["hour"] = demand["timestamp"].dt.hour
demand["lag_24"] = demand["load"].shift(24)
demand = demand.dropna()

X = demand[["hour", "temperature", "lag_24"]]
y = demand["load"]
X_past, X_future, y_past, y_future = train_test_split(X, y, test_size=0.2, shuffle=False)

scaler = StandardScaler()
X_past_scaled = scaler.fit_transform(X_past)
X_future_scaled = scaler.transform(X_future)
model = Ridge().fit(X_past_scaled, y_past)
print(mean_squared_error(y_future, model.predict(X_future_scaled)))

pipeline = make_pipeline(StandardScaler(), Ridge())
print(cross_val_score(pipeline, X_past, y_past, cv=TimeSeriesSplit(n_splits=5)).mean())
