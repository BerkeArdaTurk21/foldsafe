"""Leaky: features are centered with numpy column means of all rows, then cross-validated."""

import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVR

df = pd.read_csv("sensors.csv")
X = df[["temperature", "pressure", "humidity"]].to_numpy()
y = df["output"].to_numpy()
X_centered = X - np.mean(X, axis=0)  # foldsafe-expect: FS005
print(cross_val_score(SVR(), X_centered, y, cv=5).mean())
