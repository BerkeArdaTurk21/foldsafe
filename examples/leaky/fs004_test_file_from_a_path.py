"""Leaky: the test file is read through a pathlib path, and the scaler is refitted on it."""

from pathlib import Path

import pandas as pd
from sklearn.preprocessing import StandardScaler

DATA_DIR = Path("../input")

train = pd.read_csv(DATA_DIR / "train.csv")
test = pd.read_csv(DATA_DIR / "test.csv")
numeric = ["Age", "Fare"]

train_scaled = StandardScaler().fit_transform(train[numeric])
test_scaled = StandardScaler().fit_transform(test[numeric])  # foldsafe-expect: FS004
