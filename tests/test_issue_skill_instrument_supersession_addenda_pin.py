"""Prose pins for the #1812 instrument-supersession + scope-extension addenda duties.

Pins (a) the SKILL.md 9a-ter § Instrument-supersession + scope-extension
addenda duties block (clause (1) supersession hold-by-default; clause (2)
addenda-are-dispatches), (b) the CLAUDE.md user-chat inline free-analysis
carve-out clause mirroring it byte-identically + its canonical-block
pointer, and (c) this file's own registration in the Step-9c selector's
WORKFLOW_INVARIANT set (SKILL.md/CLAUDE.md diffs select only that set — an
unregistered pin never runs on the diffs it guards).

Incidents (#1812, both 2026-07-28): three live SAE rounds kept burning
Batch-API judge spend on labels #1773 was designed to supersede — frozen
only after the user asked twice; and "parallel + vectorized" had to be
re-stated twice before a throughput addendum landed (the compute-character
statement bound only the original dispatch).

Family precedent: tests/test_issue_skill_inline_measurement_duties.py (#1625).
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
SKILL_MD = REPO / ".claude" / "skills" / "issue" / "SKILL.md"
CLAUDE_MD = REPO / "CLAUDE.md"
SELECTOR_PY = REPO / "scripts" / "select_step9c_tests.py"

# Disambiguated header form: the CLAUDE.md canonical-block POINTER repeats the
# bare header phrase after "§ ", so the count keys on the "(REQUIRED"-suffixed
# form to avoid double-counting the pointer.
HEADER = "**Instrument-supersession + scope-extension addenda duties (REQUIRED"
CLAUSE1_ANCHOR = "(1) BEFORE dispatching any stage that spends on a measurement instrument"
CLAUSE2_ANCHOR = "(2) A mid-round SCOPE-EXTENSION ADDENDUM"
POINTER = (
    "Canonical block: SKILL.md Step 9a-ter "
    "§ Instrument-supersession + scope-extension addenda duties."
)
PIN_FILE_RELPATH = "tests/test_issue_skill_instrument_supersession_addenda_pin.py"


def _extract_clause(text: str, anchor: str, start: int) -> str:
    """Extract the clause from ``anchor`` (searched from ``start``) through its
    terminating period, whitespace-stripped.

    On the SKILL.md side each clause is one physical line (the newline
    terminates); on the CLAUDE.md side the clauses sit inline in the
    single-line carve-out bullet (the next clause marker or the
    canonical-block pointer terminates). Raises ValueError on a missing
    anchor — a hard fail, per the pin-test convention.
    """
    c = text.index(anchor, start)
    ends = [
        i
        for i in (
            text.find("\n", c),
            text.find(CLAUSE2_ANCHOR, c + 1),
            text.find(" Canonical block:", c),
        )
        if i != -1
    ]
    assert ends, f"no terminator found after anchor {anchor!r}"
    return text[c : min(ends)].strip()


def test_header_exactly_once_in_each_file():
    for path in (SKILL_MD, CLAUDE_MD):
        text = path.read_text(encoding="utf-8")
        assert text.count(HEADER) == 1, (
            f"{path}: expected exactly one header, got {text.count(HEADER)}"
        )


def test_skill_block_sits_inside_9a_ter():
    text = SKILL_MD.read_text(encoding="utf-8")
    idx = text.index(HEADER)
    # After the estimator-validity duties block, before the pod-safety block
    # (both anchors verified unique on the live tree at authoring time).
    assert (
        text.index("Inline estimator-validity + record-integrity duties")
        < idx
        < text.index("Pod-safety pre-launch signals (deviation case")
    )


def test_clause_sentences_byte_identical():
    skill = SKILL_MD.read_text(encoding="utf-8")
    claude = CLAUDE_MD.read_text(encoding="utf-8")
    s0 = skill.index(HEADER)
    c0 = claude.index(HEADER)
    for anchor in (CLAUSE1_ANCHOR, CLAUSE2_ANCHOR):
        s_clause = _extract_clause(skill, anchor, s0)
        c_clause = _extract_clause(claude, anchor, c0)
        assert s_clause == c_clause, f"clause drift across files at anchor {anchor!r}"
        # Each clause ends at its incident-citation close-paren + period.
        assert s_clause.endswith(")."), s_clause[-60:]


def test_claude_md_mirror_carries_canonical_pointer():
    text = CLAUDE_MD.read_text(encoding="utf-8")
    idx = text.index(HEADER)
    # Header (~190 chars) + clause 1 (~1150) + clause 2 (~640) + pointer sit
    # well inside a 4000-char window; headroom for wording tweaks.
    window = text[idx : idx + 4000]
    assert POINTER in window


def test_registered_in_step9c_workflow_invariant():
    spec = importlib.util.spec_from_file_location("select_step9c_tests_1812", SELECTOR_PY)
    assert spec is not None and spec.loader is not None
    sel = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(sel)
    assert PIN_FILE_RELPATH in sel.WORKFLOW_INVARIANT
