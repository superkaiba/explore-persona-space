"""Prose pins for the #1625 inline measurement-design + figure-sanity duties.

Pins (a) the SKILL.md 9a-ter § Inline measurement-design + figure-sanity
duties block (both-arms mapping statement + rendered-PNG eyeball check;
incidents #779 2026-07-14 context-only inline mapping / #958 one-arm class,
and #1112 empty-figure-presented-3x), (b) the CLAUDE.md user-chat inline
free-analysis carve-out clause mirroring it, (c) the Auto-run procedure
step-1/step-3 pointer sentences, and (d) this file's own registration in the
Step-9c selector's WORKFLOW_INVARIANT set (SKILL.md/CLAUDE.md diffs select
only that set — an unregistered pin never runs on the diffs it guards).

Family precedent: tests/test_issue_skill_trigger_dense_tag_adoption.py.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

from tests.issue_skill_source import issue_skill_text

REPO = Path(__file__).resolve().parent.parent
SKILL_MD = REPO / ".claude" / "skills" / "issue" / "SKILL.md"
CLAUDE_MD = REPO / "CLAUDE.md"
SELECTOR_PY = REPO / "scripts" / "select_step9c_tests.py"

ANCHOR = "Inline measurement-design + figure-sanity duties"
PIN_FILE_RELPATH = "tests/test_issue_skill_inline_measurement_duties.py"


def test_skill_9a_ter_duties_block_present():
    text = issue_skill_text()
    idx = text.index(ANCHOR)  # ValueError = hard fail
    # Window 2200: the drafted block MEASURES ~1912 chars from the anchor
    # (all pinned tokens sit <=1678 in; fact-checked 2026-07-23); headroom
    # for wording tweaks without letting the pin drift file-wide.
    window = text[idx : idx + 2200]
    assert "prefix-based" in window
    assert "context-based" in window
    assert "explicit stated deviation" in window
    assert "non-empty axes" in window
    assert "#958" in window and "#779" in window and "#1112" in window
    # Sits inside 9a-ter: after the compute-character block, before the
    # pod-safety block (both anchors verified unique on the live tree).
    assert (
        text.index("Compute-character pre-launch statement (REQUIRED — one paragraph")
        < idx
        < text.index("Pod-safety pre-launch signals (deviation case")
    )


def test_claude_md_carveout_duties_present():
    text = CLAUDE_MD.read_text(encoding="utf-8")
    # NOTE: "**User-chat inline free analysis**" occurs TWICE in CLAUDE.md
    # (a line-50 cross-reference inside the Follow-up bullet, then the
    # line-51 carve-out bullet itself). index() lands on the FIRST; the
    # ANCHOR search from i0 still resolves to the line-51 insert because
    # the duties phrase exists nowhere between the two occurrences.
    i0 = text.index("**User-chat inline free analysis**")
    idx = text.index(ANCHOR, i0)
    window = text[idx : idx + 1400]
    assert "prefix-based" in window and "context-based" in window
    assert "non-empty axes" in window
    # Cites the canonical block (same convention as the compute-character clause).
    assert "SKILL.md Step 9a-ter § " + ANCHOR in window


def test_step_pointer_sentences_present():
    text = issue_skill_text()
    assert "ALSO carries the both-arms line" in text  # step 1 pointer
    assert "Read each regenerated PNG" in text  # step 3 pointer
    # Both pointers sit inside the Auto-run procedure.
    proc = text.index("**Auto-run procedure.**")
    assert text.index("ALSO carries the both-arms line") > proc
    assert text.index("Read each regenerated PNG") > proc


def test_registered_in_step9c_workflow_invariant():
    spec = importlib.util.spec_from_file_location("select_step9c_tests_1625", SELECTOR_PY)
    assert spec is not None and spec.loader is not None
    sel = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(sel)
    assert PIN_FILE_RELPATH in sel.WORKFLOW_INVARIANT
