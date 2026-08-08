"""Prose pin for the #2067 cross-session pivot subsection in compute-backend-failover.md.

Asserts the new H3 subsection exists AND carries two body invariants
(the pivoter duty sentence + the UNKNOWN-refuses-as-LIVE token).
"""

from pathlib import Path


def _failover_md() -> Path:
    return (
        Path(__file__).resolve().parent.parent / ".claude" / "rules" / "compute-backend-failover.md"
    )


def test_cross_session_pivot_subsection_present() -> None:
    text = _failover_md().read_text(encoding="utf-8")

    # 1. Exact H3 header for the #2067 subsection.
    h3 = "### Cross-session pivot — resolve the owner before provisioning (#2067)"
    assert h3 in text, f"missing H3: {h3!r}"

    # 2. Duty sentence substring — a pivoting session on a task it does not own
    #    MUST first resolve the owner.
    duty = "provisions a pod on a task it does not own MUST"
    assert duty in text, f"missing duty sentence substring: {duty!r}"

    # 3. UNKNOWN-refuses-as-LIVE token in the action-table row.
    token = "treat as LIVE"
    assert token in text, f"missing UNKNOWN row token: {token!r}"
