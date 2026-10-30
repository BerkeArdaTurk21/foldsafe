# Rules

foldsafe reports one finding per leak, on one line, from the most specific rule
(see the precedence table in [`examples/README.md`](../examples/README.md)).
Every finding has three parts:

- **where**: file and line,
- **why**: what was seen on that line and why it leaks,
- **fix**: how to correct it, usually by moving the step into a scikit-learn (or imblearn) `Pipeline`.

A `Pipeline` is the general cure: it fits every step on the training rows of each split or
fold and only *transforms* the rows it is evaluated on, so nothing learned can come from them.

Severity `error` means the leak is certain from the code alone; `warning` means it is likely
but depends on intent or data. Status says whether the analyzer checks the rule yet.

Every example below is a shortened version of a script in [`examples/`](../examples/).

## FS001 · Preprocessor fitted on the full dataset before the train/test split

Severity: error · Status: active · Reported on: the `fit` / `fit_transform` call

**Why it leaks.** A scaler, imputer, encoder or PCA learns statistics (mean, scale, categories,
components) from every row it is fitted on. Fitted before the split, it has seen the test rows,
so they are no longer unseen data.

```python
X_scaled = StandardScaler().fit_transform(X)  # FS001
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y)
```

**Fix.** Split first, then fit on the training rows only; a `Pipeline` does this for you.

```python
X_train, X_test, y_train, y_test = train_test_split(X, y)
model = make_pipeline(StandardScaler(), LogisticRegression())
model.fit(X_train, y_train)
print(model.score(X_test, y_test))
```

Also reported for a manual `for train_idx, test_idx in cv.split(X_scaled)` loop.
Fitting on all rows *after* evaluation (to train the final model) is fine and not reported.

## FS002 · Target-aware feature selection fitted outside the split or CV loop

Severity: error · Status: active · Reported on: the selector's `fit` / `fit_transform` call

**Why it leaks.** `SelectKBest`, `RFE`, `SelectFromModel` and target encoders learn from `y`.
Fitted on all rows, they pick features using the labels of the very rows the model is later
scored on. This is the strongest form of preprocessing leakage, so it takes precedence over
FS001 and FS003 for the same fit.

```python
X_selected = SelectKBest(k=10).fit_transform(X, y)  # FS002
print(cross_val_score(LogisticRegression(), X_selected, y, cv=5).mean())
```

**Fix.** Put the selector inside the `Pipeline` and cross-validate (or split) the `Pipeline`,
so each fold selects on its own training rows.

```python
model = make_pipeline(SelectKBest(k=10), LogisticRegression())
print(cross_val_score(model, X, y, cv=5).mean())
```

## FS003 · Data preprocessed on all rows is passed into cross-validation or a search

Severity: error · Status: active · Reported on: the CV or search call (the message names the fit line)

**Why it leaks.** Cross-validation holds out each fold in turn, but the features were already
computed with every fold included. The CV score, and the hyperparameters a search picks with
it, are measured on rows that shaped the features.

```python
X_scaled = StandardScaler().fit_transform(X)
print(cross_val_score(SVC(), X_scaled, y, cv=5).mean())  # FS003
```

**Fix.** Give cross-validation (or `GridSearchCV`) the preprocessing and the model as one
`Pipeline`.

```python
model = make_pipeline(StandardScaler(), SVC())
print(cross_val_score(model, X, y, cv=5).mean())

search = GridSearchCV(model, {"svc__C": [0.1, 1, 10]}, cv=5).fit(X, y)
```

If the same preprocessed data is also split with `train_test_split`, the leak is reported once,
as FS001.

## FS004 · Test data used for fitting, or statistics computed on train + test combined

Severity: error · Status: active · Reported on: the `fit` on test rows, or the line that keeps the statistic

**Why it leaks.** Whatever is learned from the test rows (a refitted scaler, a median of train
and test together) is no longer a fair measure of how the model handles new data.

```python
scaler = StandardScaler().fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.fit_transform(X_test)  # FS004: refitted on test rows

full = pd.concat([train, test])
train["Age"] = train["Age"].fillna(full["Age"].median())  # FS004: median of train + test
```

**Fix.** Fit on the training rows and only `transform` / `predict` the test rows. A `Pipeline`
does exactly that.

```python
model = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), LogisticRegression())
model.fit(X_train, y_train)
print(model.score(X_test, y_test))
```

For a pandas statistic, compute it on the training rows and reuse it:

```python
median = train["Age"].median()
train["Age"] = train["Age"].fillna(median)
test["Age"] = test["Age"].fillna(median)
```

Printing a statistic of the combined data (exploration) is not reported; only a statistic that
is kept is. FS004 takes precedence over FS005.

## FS005 · pandas statistic on the full frame stored as a feature before the split

Severity: error · Status: planned (M5) · Reported on: the line that stores the statistic

**Why it leaks.** `df["Age"].fillna(df["Age"].median())`, a hand-written z-score, or a group
target rate (`groupby(...)["Survived"].transform("mean")`) uses every row, including the
evaluation rows. A target rate even puts each row's own label into its features.

```python
df["Age"] = df["Age"].fillna(df["Age"].median())  # FS005
df["ticket_survival"] = df.groupby("Ticket")["Survived"].transform("mean")  # FS005
```

**Fix.** Use a transformer inside the `Pipeline`, so the statistic is learned per fold.

```python
model = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), LogisticRegression())
encoded = make_pipeline(TargetEncoder(), LogisticRegression())  # instead of a group target mean
print(cross_val_score(model, X, y, cv=5).mean())
```

## FS006 · Oversampling applied before the split or outside an imblearn Pipeline

Severity: error · Status: planned (M5) · Reported on: the `fit_resample` call

**Why it leaks.** `SMOTE` interpolates new minority rows and `RandomOverSampler` copies them.
Done before the split, near-duplicates of the test rows land in training. Note that
scikit-learn's own `Pipeline` cannot hold a sampler; imblearn's can.

```python
X_res, y_res = SMOTE().fit_resample(X, y)  # FS006
X_train, X_test, y_train, y_test = train_test_split(X_res, y_res)
```

**Fix.** Use `imblearn.pipeline.make_pipeline`, which resamples the training folds only.

```python
from imblearn.pipeline import make_pipeline

model = make_pipeline(SMOTE(random_state=0), RandomForestClassifier(random_state=0))
print(cross_val_score(model, X, y, cv=5, scoring="f1").mean())
```

## FS007 · best_score_ from a hyperparameter search reported as the final score

Severity: warning · Status: planned (M5) · Reported on: the line that reports or stores `best_score_`

**Why it leaks.** `best_score_` is the best of many settings evaluated on the same folds. It
chose the model, so it is biased upwards and cannot also grade it.

```python
search = GridSearchCV(make_pipeline(StandardScaler(), SVC()), grid, cv=5).fit(X, y)
print(f"Accuracy: {search.best_score_:.3f}")  # FS007
```

**Fix.** Score the tuned `Pipeline` on a held-out test set, or use nested cross-validation.

```python
search = GridSearchCV(make_pipeline(StandardScaler(), SVC()), grid, cv=5)
search.fit(X_train, y_train)
print(search.score(X_test, y_test))

print(cross_val_score(search, X, y, cv=5).mean())  # nested CV
```

## FS008 · Shuffled split or KFold on data that has a date column

Severity: warning · Status: planned (M5) · Reported on: the shuffled split or `KFold(shuffle=True)` call

**Why it leaks.** When rows are ordered in time, a shuffled split trains on the future to
predict the past, which the model can never do in production. It is a warning because the
dates may not matter for the task.

```python
df = pd.read_csv("sales.csv", parse_dates=["date"])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)  # FS008
```

**Fix.** Split by time: `TimeSeriesSplit` for cross-validation, or sort by date and split
without shuffling.

```python
model = make_pipeline(StandardScaler(), Ridge())
print(cross_val_score(model, X, y, cv=TimeSeriesSplit(n_splits=5)).mean())

df = df.sort_values("date")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
```
