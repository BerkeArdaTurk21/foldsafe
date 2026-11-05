"""Clean (near miss): train and test are glued together, but nothing is learned from the rows.

FamilySize and Title are computed row by row, so no statistic crosses from test to train.
The frames are separated again before the model is fitted.
"""

import pandas as pd
from sklearn.ensemble import RandomForestClassifier

train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")
target = train["Survived"]

both = pd.concat([train.drop(columns="Survived"), test], ignore_index=True)
both["FamilySize"] = both["SibSp"] + both["Parch"] + 1
both["IsAlone"] = (both["FamilySize"] == 1).astype(int)
both["Title"] = both["Name"].str.extract(r" ([A-Za-z]+)\.", expand=False)

features = ["Pclass", "FamilySize", "IsAlone"]
X_train = both[features].iloc[: len(train)]
X_test = both[features].iloc[len(train) :]

model = RandomForestClassifier(random_state=0).fit(X_train, target)
pd.Series(model.predict(X_test)).to_csv("submission.csv", index=False)
