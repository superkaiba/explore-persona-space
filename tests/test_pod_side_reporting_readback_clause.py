"""Pin: pod-side-reporting.md carries the sentinel read-back clause (#1311).

Requirement 3 of the pod-side result-reporting contract tells dispatchers
that the VM poller renames posted sentinels to ``<path>.processed`` within
~one tick, so a dispatcher reading back its OWN sentinels must keep
resume/finalize state OUTSIDE the drained glob (default) or read both
filename forms bare-first (fallback). A later editor dropping the clause
silently re-opens the #1090 fu3/fu4 failure class.
"""

from pathlib import Path

RULE = Path(__file__).resolve().parents[1] / ".claude" / "rules" / "pod-side-reporting.md"


def test_readback_clause_present():
    text = RULE.read_text(encoding="utf-8")
    assert "Read-back tolerance" in text
    assert "OUTSIDE the drained glob" in text
    assert "`<path>.processed`" in text
    assert "Three requirements, no exceptions" in text
    assert "#1090" in text
