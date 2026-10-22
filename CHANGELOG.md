# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added
- Labeled example corpus: 18 leaky and 18 clean scripts covering FS001–FS008, with
  `# foldsafe-expect` markers and a precedence table (one finding per leak).
- Rule catalog (`foldsafe.rules.RULES`), `Finding` type, and `analyze_source()` entry point;
  no checkers are registered yet.
- Corpus tests: every example parses, markers match file names, each rule has at least two
  leaky and two clean examples, and analyzer output must equal the markers for active rules.
- Project skeleton: `pyproject.toml`, `foldsafe` CLI with `--version` and a `check` command
  that collects `.py` / `.ipynb` files (no rules active yet), tests, CI.
