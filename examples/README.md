# Example corpus

Labeled ground truth for the analyzer. Every file here is a small, realistic ML script.
The test suite (`tests/test_corpus.py`) runs the analyzer over all of them.

- `leaky/` scripts contain one leak each.
- `clean/` scripts are the correct version of the same task, plus *near misses*:
  code that looks like a leak but is fine (exploring the full data, refitting the final
  model on all labeled rows after evaluation). These guard against false positives.

The files are never executed. The data files they read do not exist.

## Labeling

Put a marker on the line where the finding should be reported:

```python
X_scaled = scaler.fit_transform(X)  # foldsafe-expect: FS001
```

Several codes can share a line: `# foldsafe-expect: FS001, FS005`. Clean files have no markers.

File names start with the rule they illustrate, e.g. `fs004_refit_scaler_on_test.py`.
Every rule needs at least two leaky and two clean examples.

## Which line, which rule

One leak produces one finding, from the most specific rule:

| Rule | Reported on |
|---|---|
| FS001 | the `fit` / `fit_transform` call on data that is split afterwards |
| FS002 | the `fit` of a target-aware selector (uses `y`); wins over FS001 and FS003 |
| FS003 | the CV / search call that receives data preprocessed on all rows |
| FS004 | the `fit` on test rows, or the statistic computed on train + test combined; wins over FS005 |
| FS005 | the line that stores a full-frame pandas statistic in the frame before the split or CV |
| FS006 | the `fit_resample` call before the split or CV; wins over FS003 |
| FS007 | the line that reports or stores `best_score_` |
| FS008 | the shuffled split or `KFold(shuffle=True)` call |

## Adding an example

1. Write the leaky script and its clean counterpart.
2. Mark the expected line(s) in the leaky script.
3. Run `pytest tests/test_corpus.py`. Once the rule's checker exists, the test compares
   its findings with your markers.
