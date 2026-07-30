"""Pin test for #1693 — planner.md §9 phase_outputs: declaration.

Guards against silent removal of the one-line planner §9 addition that gives
code-reviewer.md Step 0.69 a plan-declared phase-output artifact to grep the
dispatcher against.
"""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
PLANNER_MD = REPO_ROOT / ".claude" / "agents" / "planner.md"


def test_phase_outputs_declaration_prose_pin() -> None:
    """planner.md §9 requires a `phase_outputs:` map for multi-phase plans and
    points at code-reviewer.md Step 0.69 as the enforcing gate."""
    text = PLANNER_MD.read_text(encoding="utf-8")

    # The declaration key — distinctive enough to survive minor rewording.
    assert "phase_outputs:" in text, (
        "planner.md §9 phase_outputs: declaration missing (post-#1693 gate)"
    )

    # The sibling code-reviewer gate this addition serves.
    assert "Step 0.69" in text, "planner.md addition does not cite code-reviewer.md Step 0.69"

    # The conditional trigger — >1 phase — must be discoverable in the same
    # paragraph so a future edit cannot silently unconditionally-require it
    # (single-phase plans stay exempt).
    assert "MORE THAN ONE phase" in text or "more than one phase" in text, (
        "planner.md phase_outputs: rule missing its >1-phase conditional trigger"
    )
