---
name: foldsafe
description: Find and fix data leakage in machine-learning code, where test or validation rows influence training so the score looks better than the model is. Runs the foldsafe static analyzer on scripts and notebooks, explains each finding, rewrites the code with scikit-learn Pipelines, then checks the leaks static analysis cannot see. Use when writing, reviewing or debugging model training code (scikit-learn, pandas, imblearn), before trusting or reporting a cross-validation or test score, or when a score looks too good.
---

# foldsafe: find and fix data leakage

Data leakage means information from the rows used for evaluation reached training, so the
reported score is optimistic. The code runs and nothing errors, which is why it goes unnoticed.

## 1. Run the analyzer

```bash
foldsafe check <files or folders> --format json
```

If the command is missing, install it: `python -m pip install "git+https://github.com/BerkeArdaTurk21/foldsafe"`.

- It reads code and never runs it. Do not run the user's scripts to look for leaks.
- Exit code 0: no leak found by the active rules. 1: leaks found. 2: a file could not be analyzed
  (see `problems` in the JSON; a notebook cell that is not Python is skipped and named there).
- Each finding has `rule`, `severity`, `path`, `line` (and `cell` for notebooks, counted from 1
  over all cells), `message` (why it leaks) and `fix`.
- `rules_checked` lists the active rules. Rules not listed there are not checked yet: cover
  them in step 4.

## 2. Fix each finding

For every finding, in order:

1. Tell the user in one sentence what leaks and why, in plain words, pointing at the line.
2. Apply the fix. `references/leakage-catalog.md` has a before/after pattern for every rule.
   Prefer moving the step into a `Pipeline` that is fitted on training rows only.
3. Keep everything else as it was: same model, hyperparameters, features and outputs. Change
   only what the leak requires.

A finding can be intended, for example refitting the final model on all labeled rows after
evaluation. Then leave the code, and say why the finding does not apply. Never edit code just
to silence the analyzer.

## 3. Run it again

Re-run `foldsafe check` on the same paths until it exits 0, or until every remaining finding
is explained. If the fix changes the score, the new score is the honest one: say so.

## 4. Check what static analysis cannot see

foldsafe follows code, not meaning. Read the code and the data description for these, and
report each one you find even if foldsafe exits 0:

- **Target leakage in features:** a column computed from the label, or known only after the
  outcome (refund issued, diagnosis code, days until churn).
- **Groups split across folds:** several rows per patient, customer or session. Use
  `GroupKFold` / `GroupShuffleSplit` with the group column.
- **Time order:** rows have dates and the model predicts the future. Split by time
  (`TimeSeriesSplit`, `shuffle=False` after sorting) instead of shuffling.
- **Duplicates:** identical or near-identical rows on both sides of the split.
- **Choosing by test score:** comparing models or settings on the test set, then reporting
  that score. Tune on validation folds and touch the test set once.
- **Code foldsafe does not follow:** data passed between functions, values stored on objects
  (`self.X`), helpers from the user's own modules. Trace those by hand.
- **Rules not in `rules_checked`:** check them by hand with the catalog.

## 5. Report

Give the user a short table: location, what leaked, what changed. List the step 4 checks
with what you found. Mention the score change if the code was run before and after.

## When writing new ML code

Write it leak-free from the start: split first (or cross-validate), put every step that
learns from data (scaling, imputation, encoding, feature selection, resampling) inside a
`Pipeline` or `ColumnTransformer`, tune with a search over that pipeline, and evaluate on the
test set once at the end. Then run `foldsafe check` on what you wrote.
