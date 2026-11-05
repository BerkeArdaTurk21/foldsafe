"""Clean: an end-to-end project with functions, a ColumnTransformer and a held-out test set."""

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

NUMERIC = ["income", "debt_ratio", "age"]
CATEGORICAL = ["home_ownership", "purpose"]


def make_pipeline_for(model):
    numeric = Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())])
    prepare = ColumnTransformer(
        [("num", numeric, NUMERIC), ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL)]
    )
    return Pipeline([("prepare", prepare), ("model", model)])


def main():
    loans = pd.read_csv("loans.csv")
    print(loans.describe())
    print(loans["default"].mean())

    X = loans[NUMERIC + CATEGORICAL]
    y = loans["default"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=7)

    pipeline = make_pipeline_for(GradientBoostingClassifier(random_state=7))
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=7)
    grid = {"model__max_depth": [2, 3], "model__n_estimators": [100, 300]}
    search = GridSearchCV(pipeline, grid, cv=cv, scoring="roc_auc")
    search.fit(X_train, y_train)

    probabilities = search.predict_proba(X_test)[:, 1]
    print("Chosen:", search.best_params_)
    print("Test AUC:", roc_auc_score(y_test, probabilities))


if __name__ == "__main__":
    main()
