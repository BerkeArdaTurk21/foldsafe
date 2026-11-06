"""Clean (near miss): fitted on every labeled row, then applied to a separate test file.

There is no split here. The test file was never part of the fit, so this is the correct
way to prepare a final submission.
"""

from pathlib import Path

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

DATA_DIR = Path("../input")
train = pd.read_csv(DATA_DIR / "train.csv")
test = pd.read_csv(DATA_DIR / "test.csv")
X = train.drop(columns=["Survived"])
y = train["Survived"]

scaler = StandardScaler().fit(X)
model = LogisticRegression().fit(scaler.transform(X), y)
pd.Series(model.predict(scaler.transform(test))).to_csv("submission.csv", index=False)
