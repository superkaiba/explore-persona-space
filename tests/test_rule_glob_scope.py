"""Pin (#2233): a token-free issue script's on-demand rule load-set is code-style.md ONLY.

Guards the paths: frontmatter of .claude/rules/upload-policy.md,
.claude/rules/marker-leakage-measurement.md, .claude/rules/marker-training-recipe.md
(and trips on ANY rule re-broadening onto plain issue scripts, e.g. a re-added
scripts/issue*.py catch-all). Registration: rules-pin discovery arm (#1496) via the
literal rule paths above; deliberately NOT WORKFLOW_INVARIANT.
"""

import fnmatch
from pathlib import Path

import yaml

# Token-free sentinel: no upload/marker/train/dispatch/plot/... name token.
SENTINEL = "scripts/issue9999_aggregate_stats.py"
EXPECTED = {"code-style.md"}


def _paths(rule: Path):
    lines = rule.read_text(encoding="utf-8").split("\n")
    if not lines or lines[0].strip() != "---":
        return None  # always-on rule, no frontmatter
    end = next(i for i, ln in enumerate(lines[1:], 1) if ln.strip() == "---")
    data = yaml.safe_load("\n".join(lines[1:end]))
    return data.get("paths") if isinstance(data, dict) else None


def test_token_free_issue_script_rule_loadset_is_code_style_only():
    repo_root = Path(__file__).resolve().parents[1]
    matched = sorted(
        r.name
        for r in (repo_root / ".claude" / "rules").glob("*.md")
        if (ps := _paths(r)) and any(fnmatch.fnmatch(SENTINEL, g) for g in ps)
    )
    assert set(matched) == EXPECTED, (
        f"on-demand rule load-set for a token-free issue script is {matched}, expected "
        f"{sorted(EXPECTED)}. A rule glob has (re-)broadened onto plain issue scripts "
        f"(#2233: this costs ~len(rule) bytes of context on EVERY issue-script touch, "
        f"x1,388 scripts). If deliberate, update EXPECTED here and state the per-touch "
        f"byte cost in the commit message; if the new glob is narrow but collides with "
        f"the sentinel name, pick a different token-free SENTINEL."
    )
