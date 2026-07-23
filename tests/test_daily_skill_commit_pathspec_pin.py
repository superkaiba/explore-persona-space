"""Durability pin (#1630): /daily commit recipes stay pathspec-limited.

Pins the own-files-only commit discipline in .claude/skills/daily/SKILL.md —
staged-set audit line + trailing-pathspec commit forms — so a later editor
cannot silently regress § Commit / route-1 back to a bare `git commit -m`
that sweeps a concurrent session's staged files (incident 7dbde267f1).
Registered in scripts/select_step9c_tests.py::WORKFLOW_INVARIANT + the
tests/step9c_workflow_invariant_manifest.txt manifest.
"""

from pathlib import Path

DAILY_SKILL = Path(__file__).resolve().parents[1] / ".claude" / "skills" / "daily" / "SKILL.md"


def _text() -> str:
    return DAILY_SKILL.read_text(encoding="utf-8")


def test_commit_recipe_is_pathspec_limited():
    assert (
        "-- logs/daily/YYYY-MM-DD.md .claude/cache/nightly-consolidation-events.jsonl" in _text()
    ), "§ Commit lost its trailing-pathspec commit form (own-files-only contract, #1630)"


def test_staged_set_audit_line_present():
    assert "git diff --cached --name-only" in _text(), (
        "§ Commit lost the staged-set audit line (#1630)"
    )


def test_no_bare_daily_log_commit_form():
    for line in _text().splitlines():
        s = line.strip()
        if s.startswith("git commit") and "logs:" in s:
            assert " -- " in s, f"bare daily-log commit form regressed: {s!r} (#1630)"


def test_route1_trailing_pathspec_phrase_present():
    assert "trailing pathspecs naming exactly the file" in _text(), (
        "route-1 pathspec-commit discipline phrase missing (#1630)"
    )
