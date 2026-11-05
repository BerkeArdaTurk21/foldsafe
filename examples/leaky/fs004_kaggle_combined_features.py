"""Leaky: the classic Kaggle kernel that engineers features on train and test glued together.

Statistics and encoders see the test rows. It raises leaderboard scores a little and
cannot work in production, where the test rows do not exist yet.
"""

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OrdinalEncoder

train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")
target = train["Survived"]

full_data = pd.concat([train.drop(columns="Survived"), test], ignore_index=True)

full_data["Title"] = full_data["Name"].str.extract(r" ([A-Za-z]+)\.", expand=False)
full_data["Fare"] = full_data["Fare"].fillna(full_data["Fare"].median())  # foldsafe-expect: FS004
full_data["FamilySize"] = full_data["SibSp"] + full_data["Parch"] + 1

encoder = OrdinalEncoder()
full_data[["Sex", "Title"]] = encoder.fit_transform(full_data[["Sex", "Title"]])  # foldsafe-expect: FS004

features = ["Pclass", "Sex", "Fare", "Title", "FamilySize"]
X_all = full_data[features]
X_train = X_all.iloc[: len(train)]
X_test = X_all.iloc[len(train) :]

model = RandomForestClassifier(n_estimators=500, random_state=1)
model.fit(X_train, target)
pd.DataFrame({"Survived": model.predict(X_test)}).to_csv("submission.csv", index=False)
