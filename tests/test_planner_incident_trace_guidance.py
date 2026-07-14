"""Pins the planner.md SS12 incident-trace authoring guidance (#1308): a plan
that designs/modifies a detection/trigger-lane predicate must trace every
predicate arm -- including its read/ingest path -- against the motivating
incident's REAL persisted artifact, with measured values, and the predicate
must fire on that incident. planner.md sits in the lint size-WARN band, so
trim pressure is real; this pin fails loud if the paragraph is deleted, moved
out of SS12, or has its must-fire sentence / incident numbers reworded away.

Modeled on tests/test_planner_row_coverage_guidance.py (no verify_plan
execution half -- this rule has no mechanical verifier; the Phase-1.5
fact-checker stays the semantic backstop).
"""

from __future__ import annotations

from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
PLANNER = REPO / ".claude" / "agents" / "planner.md"

# Whitespace-normalized phrase pins (re-wrapping the paragraph must not break
# them; rewording the load-bearing clauses away must).
PINNED_PHRASES = (
    # the rule's title / trigger anchor
    "motivating incident's REAL artifact (#1287)",
    # the must-fire requirement
    "The predicate MUST fire on its own motivating incident",
    # the worked example stays grounded in the real measured numbers
    "825,591 B defeats the 262,144 B read cap",
    # the read/ingest-path clause (half the #1287 defect was a read-path defect)
    "including the read/ingest path",
)


def test_planner_md_has_incident_trace_guidance():
    text = PLANNER.read_text(encoding="utf-8")
    norm = " ".join(text.split())
    for phrase in PINNED_PHRASES:
        assert phrase in norm, f"pinned phrase missing from planner.md: {phrase!r}"
    # Placement (raw text): the rule lives INSIDE `### 12. Assumptions`,
    # before the `## Goal-currency guard` H2 -- not stranded elsewhere.
    # The `## ` prefix is deliberate: a prose `§ Goal-currency guard`
    # cross-reference exists earlier in the file and must not match.
    assert (
        text.index("### 12. Assumptions")
        < text.index("REAL artifact (#1287)")
        < text.index("## Goal-currency guard")
    ), "incident-trace rule is not placed inside SS12 before the Goal-currency guard H2"
