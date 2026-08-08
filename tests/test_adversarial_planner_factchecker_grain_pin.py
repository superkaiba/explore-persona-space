"""Pin #1910: the fact-checker realized-grain duty survives in the workflow surface.

Protection prose with no pytest asserting its presence is silently droppable by
any later SKILL.md edit (#884/#1045/#1134). Incident #1900: a plan floor derived
from an assumed row-count range (50-300/mix) crashed a launch whose reused files
held exactly 20 rows/mix; the Phase 1.5 fact-checker template now carries an
explicit realized-grain (row-count-at-pin) verification duty for
row-grain-consuming reuse rows, and planner.md section 10 requires the counted
grain (or an `ungrounded — needs grain count` mark) on such rows.
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
