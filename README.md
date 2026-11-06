# foldsafe

**Catch data leakage in ML code before your metrics lie to you.**

> 🚧 **Pre-alpha, built in public.** `foldsafe check` works on scripts and notebooks and
> checks four rules today (FS001–FS004). The package is not on PyPI yet: install it from
> this repository.

## The problem

Data leakage happens when information from your test or validation rows sneaks into
training, so your score looks better than the model really is. The code runs, nothing
errors, and the number is plausible:

```python
X = StandardScaler().fit_transform(X)  # learns mean/std from ALL rows
X_train, X_test, y_train, y_test = train_test_split(X, y)  # test rows were already seen
```

The dangerous leaks are the small ones. On Kaggle Titanic, one target-derived feature
computed outside the CV loop scored **0.844**; the same feature computed correctly scored
**0.841**. Nobody would notice.

## What foldsafe tells you

```console
$ foldsafe check train.py
train.py:9: FS001 [error]
  why: Preprocessor is fitted on all rows, including the rows held out for evaluation, so
       the test rows of the train/test split shaped what it learned.
  fix: Split first and fit on the training rows only, or put the step in a Pipeline:
       make_pipeline(StandardScaler(), model).fit(X_train, y_train).

Found 1 possible leak in 1 of 1 file (1 error, 0 warnings). Rules checked: FS001, FS002, FS003, FS004.
```

Every finding has a place, a reason and a fix, and one leak gives one finding. In a
notebook the place is a cell and a line: `churn.ipynb:cell 4:2`.

## Rules

| Code | What it catches | Status |
|---|---|---|
| [FS001](docs/rules.md#fs001--preprocessor-fitted-on-the-full-dataset-before-the-traintest-split) | Scaler / imputer / encoder / PCA fitted on all rows, then the data is split | ✅ active |
| [FS002](docs/rules.md#fs002--target-aware-feature-selection-fitted-outside-the-split-or-cv-loop) | Feature selection or target encoding that learned from all labels | ✅ active |
| [FS003](docs/rules.md#fs003--data-preprocessed-on-all-rows-is-passed-into-cross-validation-or-a-search) | Data preprocessed on all rows passed to `cross_val_score` / `GridSearchCV` | ✅ active |
| [FS004](docs/rules.md#fs004--test-data-used-for-fitting-or-statistics-computed-on-train--test-combined) | Fitting on test rows, or statistics of train + test combined | ✅ active |
| [FS005](docs/rules.md#fs005--pandas-statistic-on-the-full-frame-stored-as-a-feature-before-the-split) | pandas statistics of the full frame kept as features before the split | planned |
| [FS006](docs/rules.md#fs006--oversampling-applied-before-the-split-or-outside-an-imblearn-pipeline) | SMOTE / oversampling before the split or outside an imblearn `Pipeline` | planned |
| [FS007](docs/rules.md#fs007--best_score_-from-a-hyperparameter-search-reported-as-the-final-score) | `best_score_` reported as the final score (nested CV missing) | planned |
| [FS008](docs/rules.md#fs008--shuffled-split-or-kfold-on-data-that-has-a-date-column) | Shuffled split or KFold on time-ordered data | planned |

Why each rule leaks, with a leaky and a fixed example: [`docs/rules.md`](docs/rules.md).

## How it works

- **Reads, never runs.** foldsafe parses your code with Python's `ast` module. No code is
  executed, no LLM or API key is needed, and the result is the same on every run.
- **Tracks where rows come from.** Every variable is labeled as all rows, training rows,
  test rows, or train and test mixed. A transformer fitted on all rows leaves a mark on its
  output, and the mark is traced through reassignments, loops, `if` blocks and function
  bodies to the split or cross-validation call it reaches.
- **Tested against labeled examples.** [`examples/`](examples/) holds 31 leaky scripts with
  the expected finding marked on its line, and 31 clean scripts, many of them near misses
  (exploring the full data, refitting the final model after evaluation, a hand-written fold
  loop). Every active rule finds all of its leaky examples and flags no clean one.
- **Tried on real code.** It parses scikit-learn's 671 source files in a few seconds
  without a crash. Its findings there are real patterns in test code, and the one false
  alarm this trial found (fitting a calibrator on a held-out `X_cal`) is now fixed and in
  the corpus. That is one codebase, not a measured false-positive rate.

What it does not follow yet (data passed between functions, dynamic code) is listed in
[`docs/rules.md`](docs/rules.md#what-the-analyzer-follows).

## Try it

```bash
git clone https://github.com/BerkeArdaTurk21/foldsafe
cd foldsafe
python -m venv .venv
.venv/Scripts/activate   # Windows; on macOS/Linux: source .venv/bin/activate
pip install -e .

foldsafe check path/to/project
foldsafe check notebook.ipynb --format json
```

| Exit code | Meaning |
|---|---|
| 0 | no leak found |
| 1 | leak found |
| 2 | no leak found, but a file could not be analyzed |

Exit code 1 lets CI stop a build. `--format json` is for tools and coding agents. Notebook
cells are read in file order, IPython lines (`%magic`, `!shell`) are ignored, and a cell
that is not valid Python is skipped and named in the output.

## Roadmap

| When | What |
|---|---|
| Nov 2026 | Agent skill for Claude Code, Codex and Cursor: the agent runs foldsafe, explains and fixes each finding |
| Dec 2026 | FS005–FS008 and a precision/recall benchmark on the example corpus |
| Dec 2026 | `pip install foldsafe` (0.1.0), SARIF output and a GitHub Action |

## Development

```bash
pip install -e ".[dev]"
ruff check .
ruff format --check .
pytest --cov=foldsafe --cov-branch
```

Every rule is specified by labeled examples before it is implemented. To add one, see
[`examples/README.md`](examples/README.md). Changes are listed in [`CHANGELOG.md`](CHANGELOG.md).

## License

MIT
