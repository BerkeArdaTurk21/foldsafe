# foldsafe

**Catch data leakage in ML code before your metrics lie to you.**

> 🚧 Pre-alpha. The analyzer is being built in public; no rules are active yet.

Data leakage happens when information from your validation or test data sneaks into training,
so your cross-validation score looks better than the model really is. The dangerous cases are
not the obvious ones: on Kaggle Titanic, one target-derived feature computed outside the CV loop
scored **0.844**, while the same feature done correctly scored **0.841**. Plausible enough that
nobody would notice.

foldsafe is two things:

1. **A static analyzer** (`foldsafe check`) for Python scripts and Jupyter notebooks. It reads your
   code (never runs it), needs no LLM or API key, and works in CI.
2. **An agent skill** for Claude Code, Codex and Cursor. Your coding agent runs the analyzer, explains each
   finding, rewrites the code into a leak-free `Pipeline`, and checks the semantic leaks that static
   analysis cannot see.

## Planned rules (v0.1)

| Code | What it catches |
|---|---|
| FS001 | Scaler / imputer / encoder / PCA fitted on the full dataset before the split |
| FS002 | Target-aware feature selection outside the split or CV loop |
| FS003 | Preprocessing outside a `Pipeline`, then `cross_val_score` / `GridSearchCV` |
| FS004 | Fitting on test data, or computing statistics on train + test combined |
| FS005 | pandas statistics on the full frame before the split (`fillna(df.mean())`, target means) |
| FS006 | Oversampling (SMOTE) before the split or outside an imblearn `Pipeline` |
| FS007 | `best_score_` reported as the final score (suggests nested CV) |
| FS008 | Shuffled split / KFold on data with a date column |

## Install (development)

```bash
git clone https://github.com/BerkeArdaTurk21/foldsafe
cd foldsafe
python -m venv .venv
.venv/Scripts/activate   # Windows; on macOS/Linux: source .venv/bin/activate
pip install -e ".[dev]"
foldsafe check path/to/project
```

## Development

```bash
ruff check .
ruff format --check .
pytest
```

## License

MIT
