"""Leaky: random oversampling duplicates minority rows, then CV splits the duplicates."""

import pandas as pd
from imblearn.over_sampling import RandomOverSampler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

df = pd.read_csv("fraud.csv")
X = df.drop(columns=["is_fraud"])
y = df["is_fraud"]

sampler = RandomOverSampler(random_state=0)
X_resampled, y_resampled = sampler.fit_resample(X, y)  # foldsafe-expect: FS006
print(cross_val_score(LogisticRegression(), X_resampled, y_resampled, cv=5, scoring="f1").mean())
