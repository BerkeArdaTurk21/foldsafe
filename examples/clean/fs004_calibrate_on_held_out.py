"""Clean (near miss): the second part of the split is a calibration set, not a test set.

Fitting a calibrator on held-out rows is what it is for. Found as a false alarm in
scikit-learn's own tests; the variable name X_cal marks the rows as validation data.
"""

import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

df = pd.read_csv("train.csv")
X = df.drop(columns=["target"])
y = df["target"]

X_train, X_cal, y_train, y_cal = train_test_split(X, y, random_state=42)
model = LogisticRegression(max_iter=1000).fit(X_train, y_train)
calibrated = CalibratedClassifierCV(FrozenEstimator(model), method="isotonic").fit(X_cal, y_cal)
print(calibrated.predict_proba(X_cal)[:5])
