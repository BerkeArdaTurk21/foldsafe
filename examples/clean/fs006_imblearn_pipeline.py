"""Clean: imblearn's Pipeline resamples inside each training fold, never the validation fold."""

import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score

df = pd.read_csv("fraud.csv")
X = df.drop(columns=["is_fraud"])
y = df["is_fraud"]

pipeline = Pipeline([("smote", SMOTE(random_state=0)), ("model", LogisticRegression())])
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
print(cross_val_score(pipeline, X, y, cv=cv, scoring="f1").mean())
