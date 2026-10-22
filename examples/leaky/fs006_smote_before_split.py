"""Leaky: SMOTE synthesizes rows from all data; near-copies end up in train and test."""

import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

df = pd.read_csv("fraud.csv")
X = df.drop(columns=["is_fraud"])
y = df["is_fraud"]

X_resampled, y_resampled = SMOTE(random_state=0).fit_resample(X, y)  # foldsafe-expect: FS006
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, random_state=0)
print(RandomForestClassifier(random_state=0).fit(X_train, y_train).score(X_test, y_test))
