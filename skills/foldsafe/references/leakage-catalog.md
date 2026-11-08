# Leakage catalog: fix patterns

One section per foldsafe rule. Each shows the leaky shape and the fixed shape. The full rule
descriptions live in the repository's `docs/rules.md`.

## FS001 · Preprocessor fitted on all rows before the split

```python
# Leaky
X = StandardScaler().fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X, y)

# Fixed: split first, fit on the training rows, only transform the test rows
X_train, X_test, y_train, y_test = train_test_split(X, y)
model = make_pipeline(StandardScaler(), LogisticRegression()).fit(X_train, y_train)
model.score(X_test, y_test)
```

Also FS001: `scaler.fit(X)` on all rows, then `scaler.transform(X_test)`, even after the split.

## FS002 · Feature selection or target encoding that learned from all labels

```python
# Leaky
X_best = SelectKBest(f_classif, k=20).fit_transform(X, y)
cross_val_score(model, X_best, y)

# Fixed: selection is a Pipeline step, so every fold selects on its own rows
cross_val_score(make_pipeline(SelectKBest(f_classif, k=20), model), X, y)
```

Same for `TargetEncoder`, `RFE`, `SelectFromModel`, custom target encoders and LDA used as a
feature transformer.

## FS003 · Data preprocessed on all rows passed to cross-validation or a search

```python
# Leaky
X_scaled = StandardScaler().fit_transform(X)
GridSearchCV(SVC(), grid).fit(X_scaled, y)

# Fixed: search over the Pipeline; prefix parameters with the step name
pipeline = Pipeline([("scale", StandardScaler()), ("model", SVC())])
GridSearchCV(pipeline, {"model__C": [0.1, 1, 10]}).fit(X, y)
```

## FS004 · Test rows used for fitting, or statistics of train + test combined

```python
# Leaky
X_test_scaled = scaler.fit_transform(X_test)
both = pd.concat([train, test]); both["Age"] = both["Age"].fillna(both["Age"].median())

# Fixed: learn from training rows, reuse for test rows
X_test_scaled = scaler.transform(X_test)
median = train["Age"].median()
train["Age"] = train["Age"].fillna(median)
test["Age"] = test["Age"].fillna(median)
```

Row-by-row features on a combined frame (`FamilySize = SibSp + Parch + 1`) are fine.

## FS005 · pandas statistic of the full frame kept as a feature before the split

```python
# Leaky
df["Age"] = df["Age"].fillna(df["Age"].median())
df["ticket_rate"] = df.groupby("Ticket")["Survived"].transform("mean")

# Fixed: SimpleImputer / TargetEncoder in a Pipeline, or compute on the training rows only
pipeline = make_pipeline(
    ColumnTransformer([("age", SimpleImputer(strategy="median"), ["Age"]),
                       ("ticket", TargetEncoder(), ["Ticket"])]),
    model,
)
```

## FS006 · Oversampling before the split or outside an imblearn Pipeline

```python
# Leaky
X_res, y_res = SMOTE().fit_resample(X, y)
cross_val_score(model, X_res, y_res)

# Fixed: imblearn's Pipeline resamples training folds only
from imblearn.pipeline import Pipeline
cross_val_score(Pipeline([("smote", SMOTE()), ("model", model)]), X, y)
```

## FS007 · best_score_ reported as the final score

```python
# Leaky (optimistic): the best of many settings on the same folds
search.fit(X, y); print(search.best_score_)

# Fixed: nested CV, or a held-out test set scored once
cross_val_score(search, X, y, cv=outer_cv)
```

## FS008 · Shuffled split or KFold on time-ordered data

```python
# Leaky
train_test_split(X, y)                      # shuffles by default
KFold(n_splits=5, shuffle=True)

# Fixed: keep time order
df = df.sort_values("date")
train_test_split(X, y, shuffle=False)
TimeSeriesSplit(n_splits=5)
```
