# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added
- **FS003** is active: data preprocessed on all rows that enters `cross_val_score`-style
  functions or `*SearchCV.fit` is reported on the CV call, naming the fit line. When the same
  data is also split, the leak is reported once, as FS001.
- **FS002** is active: a feature selector or target encoder that learned from the labels of
  all rows, whose output is split or cross-validated, is reported on the fit line. It takes
  precedence over FS001 and FS003 for the same fit.
- **FS001** is active: a preprocessor fitted on all rows whose output is split afterwards
  (by `train_test_split` or a manual `cv.split()` loop) is reported on the fit line.
  Checkers now receive the flow pass's events (`foldsafe.rules.checks`); the rule catalog
  moved to `foldsafe.rules.info`.
- Estimator recognition (`foldsafe.catalog`): preprocessors, target-aware selectors and
  encoders, imblearn samplers, pipelines (kind taken from their last step) and models.
  `fit`, `fit_transform` and `fit_resample` are recorded as fit events; a transformer fitted
  on all rows leaves a taint on everything it outputs, so it can be traced to a later split.
- Split and cross-validation events (`foldsafe.events`): `train_test_split` outputs become
  train/test values, CV splitters remember whether they shuffle, `for train_idx, test_idx in
  cv.split(X)` loops mark fold rows, and `cross_val_score`-style calls and `*SearchCV.fit`
  are recorded with the data they receive.
- Data-flow foundation (`foldsafe.flow`, `foldsafe.values`): import/alias resolution, data
  sources (`read_csv`, `np.load`, `sklearn.datasets`), row origins (full / train / test /
  combined) and how they propagate through columns, methods, arithmetic and `concat`.
- Labeled example corpus: 18 leaky and 18 clean scripts covering FS001–FS008, with
  `# foldsafe-expect` markers and a precedence table (one finding per leak).
- Rule catalog (`foldsafe.rules.RULES`), `Finding` type, and `analyze_source()` entry point;
  no checkers are registered yet.
- Corpus tests: every example parses, markers match file names, each rule has at least two
  leaky and two clean examples, and analyzer output must equal the markers for active rules.
- Project skeleton: `pyproject.toml`, `foldsafe` CLI with `--version` and a `check` command
  that collects `.py` / `.ipynb` files (no rules active yet), tests, CI.
