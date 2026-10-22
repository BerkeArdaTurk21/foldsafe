"""Clean: oversample the training rows only; the test rows keep the real class balance."""

import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

df = pd.read_csv("fraud.csv")
X = df.drop(columns=["is_fraud"])
y = df["is_fraud"]

X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=0)
X_train_resampled, y_train_resampled = SMOTE(random_state=0).fit_resample(X_train, y_train)
model = RandomForestClassifier(random_state=0).fit(X_train_resampled, y_train_resampled)
print(model.score(X_test, y_test))
