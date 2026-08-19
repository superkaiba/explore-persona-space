"""Token-presence pins for the #2248 brief-composition DISPLACEMENT clause.

Origin incident (#1336 implementation round v20, 2026-08-12): an
orchestrator brief ended in a detailed "Report back:" list and never named
the `epm:experiment-implementation` marker duty; the implementer returned
everything as Agent text, posted no marker, and code-review FAILed with
`mechanical_contract_only: true` and zero substantive blockers — one extra
round landed the durable record of work already done. The Step 5c-bis
strip cannot rescue this shape: its precondition is a PRESENT + conforming
marker, and here the missing marker IS the blocker.

The fix lives in workflow PROSE (`.claude/skills/issue/SKILL.md`), so —
mirroring the `tests/test_issue_skill_*_pin.py` prose-pin family — these
tests pin load-bearing SUBSTRINGS inside region-scoped slices of the file
(rewording survives; a silent drop of the clause does not):

(a) The Step 4b brief-composition bullet "**A brief NEVER suppresses the
    implementation marker.**" carries the DISPLACEMENT clause: both defect
    shapes (explicit skip AND displacement by a competing return
    contract), the returned-Agent-text-is-not-durable-task-state
    mechanism, the Step-5c-bis-cannot-rescue clause, and the positive duty
    ("any brief that specifies a return format names the marker duty
    alongside it").
(b) The Step 5.bis(a) vectorize-fix-round spawn sentence cross-references
    the Step 4b rule: it names the compute-deviation marker explicitly AND
    the `epm:experiment-implementation` duty alongside the re-post
    contract (that fix-round brief is itself a displacement-shaped return
    contract without the cross-reference).
"""

from __future__ import annotations

import re
from pathlib import Path

from tests.issue_skill_source import issue_skill_text

_REPO_ROOT = Path(__file__).resolve().parents[1]
_SKILL = _REPO_ROOT / ".claude" / "skills" / "issue" / "SKILL.md"


def _region(text: str, start_marker: str, end_marker: str, *, label: str) -> str:
    """Slice ``text`` between two unique-enough anchors, asserting both exist
    and are ordered — the `test_step10d_guard3.py` region-scoping pattern."""
    start = text.find(start_marker)
    end = text.find(end_marker, start + len(start_marker) if start != -1 else 0)
    assert start != -1, f"{label}: start marker not found: {start_marker!r}"
    assert end != -1, f"{label}: end marker not found: {end_marker!r}"
    assert start < end, f"{label}: start marker must precede end marker"
    return text[start:end]


def _collapsed(region: str) -> str:
    """Whitespace-collapse so pins survive prose re-wrapping."""
    return re.sub(r"\s+", " ", region)


def _suppression_bullet() -> str:
    # The BOLDED form anchors the Step 4b bullet itself; the Step 5.bis(a)
    # cross-reference quotes the rule UNbolded, so this anchor stays unique.
    return _region(
        issue_skill_text(),
        "**A brief NEVER suppresses the implementation marker.**",
        "**Marker-version discipline",
        label="SKILL.md Step 4b marker-suppression bullet",
    )


def test_displacement_clause_pinned():
    bullet = _collapsed(_suppression_bullet())
    assert "DISPLACEMENT is the same defect" in bullet, (
        "the Step 4b brief-composition bullet must cover marker DISPLACEMENT "
        "by a competing return contract, not only explicit suppression "
        "(#2248, from #1336 round v20)"
    )
    assert "returned Agent text is NOT durable task state" in bullet, (
        "the displacement clause must state WHY returned text is not a "
        "substitute — a return-only contract loses the durable record the "
        "mechanical contract keys on (#2248)"
    )
    assert "Step 5c-bis strip cannot" in bullet, (
        "the displacement clause must state that the Step 5c-bis strip "
        "cannot rescue a MISSING marker (its precondition is a present + "
        "conforming marker; #2248)"
    )
    assert "names the marker duty alongside it" in bullet, (
        "the displacement clause must carry the positive duty: any brief "
        "that specifies a return format names the marker duty alongside it "
        "(#2248)"
    )


def test_step5bis_cross_reference_names_marker_duty():
    region = _collapsed(
        _region(
            issue_skill_text(),
            "Vectorize-first signature check",
            "**Not overhead-bound**",
            label="SKILL.md Step 5.bis(a) vectorize-fix-round dispatch",
        )
    )
    assert "naming the compute-deviation marker" in region, (
        "the Step 5.bis(a) spawn sentence must name the compute-deviation "
        "marker EXPLICITLY (the bare 'the marker' read ambiguously; #2248)"
    )
    assert "A brief NEVER suppresses the implementation marker" in region, (
        "the Step 5.bis(a) spawn sentence must cross-reference the Step 4b "
        "brief-composition rule — the fix-round brief is itself a "
        "displacement-shaped return contract (#2248)"
    )
    assert "displacement included" in region, (
        "the Step 5.bis(a) cross-reference must carry the displacement extension explicitly (#2248)"
    )
    assert "`epm:experiment-implementation` duty" in region, (
        "the Step 5.bis(a) spawn sentence must name the "
        "epm:experiment-implementation duty alongside the compute-deviation "
        "re-post contract (#2248)"
    )
