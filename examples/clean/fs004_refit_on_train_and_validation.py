"""Clean (near miss): after tuning, refit on train + validation; the test rows stay untouched."""

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

df = pd.read_csv("train.csv")
X = df.drop(columns=["target"])
y = df["target"]

X_rest, X_test, y_rest, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
X_train, X_val, y_train, y_val = train_test_split(X_rest, y_rest, test_size=0.25, random_state=0)

best_depth, best_score = None, 0.0
for depth in (3, 5, 8):
    model = RandomForestClassifier(max_depth=depth, random_state=0).fit(X_train, y_train)
    score = model.score(X_val, y_val)
    if score > best_score:
        best_depth, best_score = depth, score

X_final = pd.concat([X_train, X_val])
y_final = pd.concat([y_train, y_val])
final_model = RandomForestClassifier(max_depth=best_depth, random_state=0).fit(X_final, y_final)
print("Test accuracy:", final_model.score(X_test, y_test))
