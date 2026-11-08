"""The agent skill and plugin manifests stay consistent with the package."""

from __future__ import annotations

import json
from pathlib import Path

from foldsafe import __version__
from foldsafe.rules import RULES

ROOT = Path(__file__).resolve().parent.parent
SKILL = ROOT / "skills" / "foldsafe"


def frontmatter(text: str) -> dict[str, str]:
    assert text.startswith("---\n"), "SKILL.md must start with YAML frontmatter"
    block = text.split("---\n", 2)[1]
    fields = {}
    for line in block.splitlines():
        key, _, value = line.partition(":")
        fields[key.strip()] = value.strip()
    return fields


def test_skill_frontmatter():
    fields = frontmatter((SKILL / "SKILL.md").read_text(encoding="utf-8"))
    assert fields["name"] == "foldsafe"
    assert 100 < len(fields["description"]) <= 1024


def test_skill_points_at_files_that_exist():
    text = (SKILL / "SKILL.md").read_text(encoding="utf-8")
    assert "references/leakage-catalog.md" in text
    assert (SKILL / "references" / "leakage-catalog.md").is_file()
    assert "foldsafe check" in text
    assert "--format json" in text


def test_catalog_has_a_fix_for_every_rule():
    catalog = (SKILL / "references" / "leakage-catalog.md").read_text(encoding="utf-8")
    for code in RULES:
        assert f"## {code} " in catalog, code
    assert catalog.count("# Fixed") >= len(RULES)


def test_plugin_manifests_agree():
    plugin = json.loads((ROOT / ".claude-plugin" / "plugin.json").read_text(encoding="utf-8"))
    market = json.loads((ROOT / ".claude-plugin" / "marketplace.json").read_text(encoding="utf-8"))
    assert plugin["name"] == "foldsafe"
    assert plugin["version"] == __version__
    (entry,) = market["plugins"]
    assert entry["name"] == plugin["name"]
    assert entry["source"] == "./"
