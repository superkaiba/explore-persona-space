"""Pin #1910 + #2174: BOTH fact-checker grain lenses survive in the workflow surface.

Protection prose with no pytest asserting its presence is silently droppable by
any later SKILL.md edit (#884/#1045/#1134). Two distinct grain lenses are pinned:

- #1910 (incident #1900): realized ROW-COUNT grain for row-grain-consuming reuse
  rows — a plan floor derived from an assumed row-count range (50-300/mix)
  crashed a launch whose reused files held exactly 20 rows/mix; the Phase 1.5
  fact-checker template carries the realized-grain (row-count-at-pin)
  verification duty, and planner.md section 10 requires the counted grain (or an
  `ungrounded — needs grain count` mark) on such rows.
- #2174 (incident #2163): EXACTNESS-claim grain — an exact-identity premise
  (byte-identity / `n_distinct_rows = 1`) grounded on a 10-shard/706-row sample
  (0.5% of 142,000 rows) became a Phase-0 full-store assert and died rc=1 on 258
  deviating rows; the Phase 1.5 template carries the EXACTNESS-CLAIM GRAIN CHECK
  (GRAIN-MISMATCH is BLOCKING), planner.md section 12 carries the
  stated-as-a-BOUND clause, and planner-section-reference.md section 12 carries
  the full sub-rule.
"""

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_skill_factchecker_carries_grain_duty():
    """The Phase 1.5 fact-checker template keeps the realized-grain duty tokens."""
    text = (REPO_ROOT / ".claude/skills/adversarial-planner/SKILL.md").read_text(encoding="utf-8")
    assert "ROW-GRAIN-CONSUMING" in text
    assert "GRAIN-CONFIRMED" in text


def test_planner_repro_card_carries_grain_clause():
    """planner.md section 10 keeps the counted-grain / ungrounded-mark clause."""
    text = (REPO_ROOT / ".claude/agents/planner.md").read_text(encoding="utf-8")
    assert "needs grain count" in text


def test_skill_factchecker_carries_exactness_grain_lens():
    """The Phase 1.5 template keeps the #2174 exactness-grain lens tokens (AC1+AC2).

    Beyond the two lens-name tokens, one token per REMEDY clause is pinned
    (round 2, NIT exactness-lens-pin-too-shallow): deleting either remedy
    from the lens text reds this pin, not just deleting the lens wholesale.
    """
    text = (REPO_ROOT / ".claude/skills/adversarial-planner/SKILL.md").read_text(encoding="utf-8")
    assert "EXACTNESS-CLAIM GRAIN CHECK" in text
    assert "GRAIN-MISMATCH" in text
    # Remedy 1: verify at full grain NOW (the lens's "Name BOTH remedies" clause).
    assert "verify at full grain NOW" in text
    # Remedy 2: restate as a bound + soften the assert.
    assert "restate the assumption as a bound" in text
    assert "soften the assert" in text


def test_planner_surfaces_carry_exactness_bound_clause():
    """planner.md keeps the bound clause; the section reference keeps the sub-rule."""
    text = (REPO_ROOT / ".claude/agents/planner.md").read_text(encoding="utf-8")
    assert "stated as a BOUND" in text
    ref = (REPO_ROOT / ".claude/rules/planner-section-reference.md").read_text(encoding="utf-8")
    assert "Exactness-claim grain" in ref
