# foldsafe

**Catch data leakage in ML code before your metrics lie to you.**

> 🚧 **Pre-alpha, built in public.** The analyzer checks four rules today (FS001–FS004).
> The `foldsafe check` command does not print findings yet (that lands in November),
> and the package is not on PyPI yet. Use the Python API below to try it.

## The problem

Data leakage happens when information from your test or validation rows sneaks into
training, so your score looks better than the model really is. The code runs, nothing
errors, and the number is plausible:

```python
X = StandardScaler().fit_transform(X)                 # learns mean/std from ALL rows
X_train, X_test, y_train, y_test = train_test_split(X, y)   # test rows were already seen
```

The dangerous leaks are the small ones. On Kaggle Titanic, one target-derived feature
computed outside the CV loop scored **0.844**; the same feature computed correctly scored
**0.841**. Nobody would notice.

## What foldsafe tells you

```text
train.py:9: FS001 [error]
  why: Preprocessor is fitted on all rows and its output is split afterwards, so the test
       rows of the train/test split shaped what it learned.
  fix: Split first and fit on the training rows only, or put the step in a Pipeline:
       make_pipeline(StandardScaler(), model).fit(X_train, y_train).
```

Every finding has a place (file and line), a reason, and a fix. One leak gives one finding.

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
- **Tested against labeled examples.** [`examples/`](examples/) holds 22 leaky scripts with
  the expected finding marked on its line, and 21 clean scripts, many of them near misses
  (exploring the full data, refitting the final model after evaluation, a hand-written fold
  loop). Every active rule finds all of its leaky examples and flags no clean one.

What it does not follow yet (data passed between functions, dynamic code) is listed in
[`docs/rules.md`](docs/rules.md#what-the-analyzer-follows).

## Try it

```bash
git clone https://github.com/BerkeArdaTurk21/foldsafe
cd foldsafe
python -m venv .venv
.venv/Scripts/activate   # Windows; on macOS/Linux: source .venv/bin/activate
pip install -e ".[dev]"
```

```python
from pathlib import Path
from foldsafe.analyzer import analyze_source

path = Path("train.py")
for finding in analyze_source(path.read_text(), str(path)):
    print(f"{finding.path}:{finding.line}: {finding.rule}  {finding.message}")
    print(f"  fix: {finding.fix}")
```

## Roadmap

| When | What |
|---|---|
| Nov 2026 | `foldsafe check` prints findings (text and JSON) and exits non-zero on a leak; Jupyter notebook support |
| Nov 2026 | Agent skill for Claude Code, Codex and Cursor: the agent runs foldsafe, explains and fixes each finding |
| Dec 2026 | FS005–FS008 and a precision/recall benchmark on the example corpus |
| Dec 2026 | `pip install foldsafe` (0.1.0), SARIF output and a GitHub Action |

## Development

```bash
ruff check .
ruff format --check .
pytest
```

Every rule is specified by labeled examples before it is implemented. To add one, see
[`examples/README.md`](examples/README.md). Changes are listed in [`CHANGELOG.md`](CHANGELOG.md).

## License

MIT
