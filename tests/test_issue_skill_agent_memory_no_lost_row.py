"""Pin test for #2101 — the agent-memory MEMORY.md no-lost-row discipline.

Guards against silent removal of the no-lost-row check prose from the three
workflow surfaces that carry it: the gotchas.md entry (the canonical recipe),
the /issue SKILL.md Step 5a manual-override clause + operator echo, and the
LESSONS.md gotchas-row trigger clause. Incident: commits 038a42ec6c +
0aaf39acac (2026-07-31) manually aligned stale agent-memory MEMORY.md copies
and silently dropped 7 reconciler index rows (#2093 manifest).
"""

from __future__ import annotations

from pathlib import Path

from tests.issue_skill_source import issue_skill_text

REPO_ROOT = Path(__file__).resolve().parents[1]
SKILL_MD = REPO_ROOT / ".claude" / "skills" / "issue" / "SKILL.md"
GOTCHAS_MD = REPO_ROOT / ".claude" / "rules" / "gotchas.md"
LESSONS_MD = REPO_ROOT / ".claude" / "rules" / "LESSONS.md"


def test_no_lost_row_discipline_pinned() -> None:
    """SKILL.md, gotchas.md, and LESSONS.md all carry the no-lost-row discipline."""
    skill = issue_skill_text()

    # (i) Two DISTINCT SKILL.md anchors — the manual-override comment clause
    # and the operator echo's leading literal. Both are load-bearing: dropping
    # either the clause or the echo alone must fail this pin (a bare
    # `comm -13` grep would survive an echo removal).
    assert "no-lost-row" in skill, "SKILL.md Step 5a manual-override no-lost-row clause missing"
    assert "agent-memory: any manual re-align" in skill, (
        "SKILL.md Step 5a agent-memory operator echo missing"
    )
    # All four SKILL.md carrier sites (Step 5a comment + echo, the second
    # manual-override prose site, the Step 10d mirror) — dropping any ONE
    # site alone must fail this pin; the bare membership check above
    # survives as long as any single site remains (#2101 review round 1).
    assert skill.count("no-lost-row") >= 4, (
        "a SKILL.md no-lost-row carrier site was removed (expected >= 4 occurrences)"
    )

    # (ii) The gotchas.md entry — the canonical recipe (title token + the
    # row-set comparison command).
    gotchas = GOTCHAS_MD.read_text(encoding="utf-8")
    assert "no-lost-row check" in gotchas, "gotchas.md agent-memory index-alignments entry missing"
    assert "comm -13" in gotchas, (
        "gotchas.md no-lost-row entry lost its `comm -13` row-set comparison recipe"
    )

    # (iii) The LESSONS.md gotchas-row trigger clause — plan-time discovery.
    gotchas_rows = [
        line
        for line in LESSONS_MD.read_text(encoding="utf-8").splitlines()
        if line.startswith("- gotchas.md — ")
    ]
    assert len(gotchas_rows) == 1, "LESSONS.md gotchas row missing or duplicated"
    assert "agent-memory index alignments" in gotchas_rows[0], (
        "LESSONS.md gotchas row lost the agent-memory index-alignments trigger clause"
    )
