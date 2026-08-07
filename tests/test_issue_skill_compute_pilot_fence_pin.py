"""Prose pins for the #1659 measured 1-cell pilot + pilot-derived fence sizing rule.

Pins (a) the SKILL.md Step 9a-ter § Compute-character pre-launch statement's
measured-pilot + fence-sizing clause (projected wall > ~15 min for a
fit/battery stage => element (1)'s per-call basis is MEASUREMENT-REQUIRED —
a 1-cell pilot through the production entrypoint, or a cited prior-issue
MEASURED figure for the SAME kernel + shape; every self-set timeout/fence
sized >= 2x the pilot-extrapolated wall; incident #1092 session f4b1d707,
2026-07-23: a guessed self-set `timeout 3000s` killed its own healthy
~25 min/cell run at EXIT=124), (b) the CLAUDE.md user-chat inline
free-analysis carve-out clause mirroring it, (c) the Step 9b cross-ref
parenthetical naming the measured-pilot / fence-sizing rule next to
"same five elements", and (d) this file's own registration in the Step-9c
selector's WORKFLOW_INVARIANT set (SKILL.md/CLAUDE.md diffs select only
that set — an unregistered pin never runs on the diffs it guards).

Assertions run on whitespace-NORMALIZED file text (the
tests/test_issue_skill_disk_routing_pin.py precedent) so prose re-wrapping
never breaks a multi-word pin; each token is still a verbatim substring of
the rule.

Family precedent: tests/test_issue_skill_inline_measurement_duties.py (#1625).
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

ANCHOR_SKILL = "Compute-character pre-launch statement (REQUIRED — one paragraph"
ANCHOR_CLAUDE = "Compute-character pre-launch statement (REQUIRED — this carve-out skips"
DUTIES = "Inline measurement-design + figure-sanity duties"
PIN_FILE_RELPATH = "tests/test_issue_skill_compute_pilot_fence_pin.py"


def _normalized(path: Path) -> str:
    """File text with all whitespace runs collapsed to single spaces.

    The pinned tokens include multi-word prose fragments; SKILL.md wraps
    prose at ~75-78 columns, so a raw-substring pin would break on any
    innocent re-wrap. Collapsing whitespace makes the pins wrap-insensitive
    while keeping them verbatim in substance.
    """
    return re.sub(r"\s+", " ", read_workflow_doc(path))


def test_skill_9a_ter_pilot_fence_clause_present() -> None:
    text = _normalized(SKILL_MD)
    lo = text.index(ANCHOR_SKILL)  # ValueError = hard fail
    hi = text.index(DUTIES)  # first occurrence = the duties block header
    assert lo < hi  # the clause sits inside the compute-character block
    window = text[lo:hi]
    for tok in (
        "1-cell",
        "production entrypoint",
        "pilot-extrapolated",
        "f4b1d707",
        "never a sizing basis",
    ):
        assert tok in window, tok
    # The two constants (>~15 min pilot trigger; >=2x fence floor). The
    # fence multiplier is pinned without the multiplication sign (ruff
    # RUF001 bans that ambiguous unicode char in Python strings; the .md
    # text spells the full ">=2x" with it).
    assert "> ~15 min" in window
    assert "≥2" in window
    # The ported rule's own alternative basis (a guess never qualifies).
    assert "cited prior-issue MEASURED figure" in window
    # Step 9b cross-ref touch (Edit 3): the teammate-loop summary names the
    # rule next to the enumeration phrase (measured +61 chars, 2026-07-24).
    assert "measured-pilot" in text[text.index("same five elements") :][:400]


def test_claude_md_carveout_pilot_fence_clause_present() -> None:
    text = _normalized(CLAUDE_MD)
    idx = text.index(ANCHOR_CLAUDE)
    # Window 6000: the farthest pinned token (f4b1d707) MEASURES 2046 chars
    # from the anchor on the live tree (2026-07-24); headroom for wording
    # tweaks without letting the pin drift file-wide.
    window = text[idx : idx + 6000]
    for tok in ("1-cell", "production entrypoint", "pilot-extrapolated", "f4b1d707"):
        assert tok in window, tok


def test_registered_in_step9c_workflow_invariant() -> None:
    spec = importlib.util.spec_from_file_location("select_step9c_tests_1659", SELECTOR_PY)
    assert spec is not None and spec.loader is not None
    sel = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(sel)
    assert PIN_FILE_RELPATH in sel.WORKFLOW_INVARIANT
