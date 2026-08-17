"""Prose pins for the #1625 inline figure-sanity duty.

Pins (a) the SKILL.md 9a-ter § Inline figure-sanity duty block (rendered-PNG
eyeball check; incident #1112 empty-figure-presented-3x), (b) the CLAUDE.md
user-chat inline free-analysis carve-out clause mirroring it, (c) the
Auto-run procedure step-3 pointer sentence, and (d) this file's own
registration in the Step-9c selector's WORKFLOW_INVARIANT set
(SKILL.md/CLAUDE.md diffs select only that set — an unregistered pin never
runs on the diffs it guards).

The both-arms (prefix+context) mapping statement this file used to pin was
retired 2026-08-12 on user order; only the figure-sanity duty remains.

Family precedent: tests/test_issue_skill_trigger_dense_tag_adoption.py.
"""

from __future__ import annotations

import importlib.util
import re
from pathlib import Path

from tests.issue_skill_source import read_workflow_doc

REPO = Path(__file__).resolve().parent.parent
SKILL_MD = REPO / ".claude" / "skills" / "issue" / "SKILL.md"
CLAUDE_MD = REPO / "CLAUDE.md"
SELECTOR_PY = REPO / "scripts" / "select_step9c_tests.py"

ANCHOR = "Inline figure-sanity duty"
PIN_FILE_RELPATH = "tests/test_issue_skill_inline_measurement_duties.py"


def _normalized(path: Path) -> str:
    """File text with whitespace runs collapsed (wrap-insensitive pins).

    SKILL.md wraps prose at ~75-78 columns, so raw-substring pins on
    multi-word fragments would break on any innocent re-wrap (same
    convention as tests/test_issue_skill_compute_pilot_fence_pin.py).
    """
    return re.sub(r"\s+", " ", read_workflow_doc(path))


def test_skill_9a_ter_duties_block_present():
    text = _normalized(SKILL_MD)
    idx = text.index(ANCHOR)  # ValueError = hard fail
    window = text[idx : idx + 1400]
    assert "non-empty axes" in window
    assert "never present or commit it" in window
    assert "#1112" in window
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
    # (a cross-reference inside the Follow-up bullet, then the carve-out
    # bullet itself). index() lands on the FIRST; the ANCHOR search from i0
    # still resolves to the carve-out insert because the duty phrase exists
    # nowhere between the two occurrences.
    i0 = text.index("**User-chat inline free analysis**")
    idx = text.index(ANCHOR, i0)
    window = text[idx : idx + 1000]
    assert "non-empty axes" in window
    # Cites the canonical block (same convention as the compute-character clause).
    assert "SKILL.md Step 9a-ter § " + ANCHOR in window


def test_step_pointer_sentences_present():
    text = _normalized(SKILL_MD)
    assert "Read each regenerated PNG" in text  # step 3 pointer
    # The pointer sits inside the Auto-run procedure.
    proc = text.index("**Auto-run procedure.**")
    assert text.index("Read each regenerated PNG") > proc


def test_registered_in_step9c_workflow_invariant():
    spec = importlib.util.spec_from_file_location("select_step9c_tests_1625", SELECTOR_PY)
    assert spec is not None and spec.loader is not None
    sel = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(sel)
    assert PIN_FILE_RELPATH in sel.WORKFLOW_INVARIANT
