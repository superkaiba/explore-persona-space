"""Prose pins for the #2040 across-cell shard-axis + checkpoint-cadence duties.

Pins (a) the SKILL.md Step 9a-ter § Compute-character pre-launch statement's
ACROSS-CELL shard-axis duty (a many-cell battery projected > ~1h at the
stated width names its across-cell shard axis + realized width, or states
explicitly `not shardable — <one-line reason>`; WITHIN-CELL vectorization
alone does not discharge it — incident #1345: a 118-cell boundary-ablation
battery dispatched serial-across-cells on one cpu-bigmem box while the
batched-inner-loop letter of the vectorize rule was satisfied; the 4-way
reshard measured ~4x), (b) the same block's detached-launch
CHECKPOINT-CADENCE field (a detached VM-side fit/phase > ~15 min names its
intermediate-artifact write points — per phase / per cell-chunk into the
durable out-root, never only at process exit — incident #1482: a detached
fit script wrote its result JSON only at process exit; hours of in-memory
fits sat one crash from total loss and the empty output dir provoked a
missing-vs-stalled escalation), (c) the CLAUDE.md user-chat inline
free-analysis carve-out mirror of both duties, (d) the
plan-compute-sizing.md § "Teammate / mid-session box dispatches" element
list carrying both elements plus the Step 9a-ter mechanics deferral (the
third Goal surface), and (e) this file's own registration in the Step-9c
selector's WORKFLOW_INVARIANT set (SKILL.md/CLAUDE.md diffs select only
that set — an unregistered pin never runs on the diffs it guards).

Assertions run on whitespace-NORMALIZED file text (the
tests/test_issue_skill_disk_routing_pin.py precedent) so prose re-wrapping
never breaks a multi-word pin; each token is still a verbatim substring of
the rule.

Family precedent: tests/test_issue_skill_compute_pilot_fence_pin.py (#1659).
"""

from __future__ import annotations

import importlib.util
import re
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
SKILL_MD = REPO / ".claude" / "skills" / "issue" / "SKILL.md"
CLAUDE_MD = REPO / "CLAUDE.md"
SIZING_MD = REPO / ".claude" / "rules" / "plan-compute-sizing.md"
SELECTOR_PY = REPO / "scripts" / "select_step9c_tests.py"

ANCHOR_SKILL = "Compute-character pre-launch statement (REQUIRED — one paragraph"
ANCHOR_CLAUDE = "Compute-character pre-launch statement (REQUIRED — this carve-out skips"
DUTIES = "Inline measurement-design + figure-sanity duties"
DETACHED_FIELDS = "A statement covering a VM-side phase"
DETACHED_END = "Routing, auto-continue behavior, and the marker schema are unchanged"
TEAMMATE_ANCHOR = "**Teammate / mid-session box dispatches"
TEAMMATE_END = "PER-BEHAVIOR-BOXES"
PIN_FILE_RELPATH = "tests/test_issue_skill_shard_axis_checkpoint_cadence_pin.py"


def _normalized(path: Path) -> str:
    """File text with all whitespace runs collapsed to single spaces.

    The pinned tokens include multi-word prose fragments; SKILL.md wraps
    prose at ~75-78 columns, so a raw-substring pin would break on any
    innocent re-wrap. Collapsing whitespace makes the pins wrap-insensitive
    while keeping them verbatim in substance.
    """
    return re.sub(r"\s+", " ", path.read_text(encoding="utf-8"))


def test_skill_9a_ter_shard_axis_duty_present() -> None:
    text = _normalized(SKILL_MD)
    lo = text.index(ANCHOR_SKILL)  # ValueError = hard fail
    hi = text.index(DUTIES)  # first occurrence = the duties block header
    assert lo < hi  # the duty sits inside the compute-character block
    window = text[lo:hi]
    for tok in (
        "ACROSS-CELL shard axis",
        "not shardable",
        "WITHIN-CELL vectorization alone does not discharge",
        "#1345",
        # The measured reshard speedup "~4x" with the multiplication sign
        # (the .md text spells it with U+00D7; ruff RUF001 bans that
        # ambiguous unicode char as a LITERAL in Python strings, so the
        # escape form pins the identical token).
        "~4\u00d7",
    ):
        assert tok in window, tok
    # Step 9b enumeration mirror (mirror-consistency): both new rules are
    # named AFTER the "same five elements" phrase. (The 400-char
    # "measured-pilot" window stays the pilot-fence test's pin —
    # deliberately not duplicated here.)
    tail = text[text.index("same five elements") :]
    assert "across-cell shard-axis" in tail
    assert "detached checkpoint-cadence" in tail


def test_skill_9a_ter_checkpoint_cadence_field_present() -> None:
    text = _normalized(SKILL_MD)
    lo = text.index(DETACHED_FIELDS)
    hi = text.index(DETACHED_END, lo)
    assert lo < hi
    window = text[lo:hi]
    for tok in ("checkpoint cadence", "never only at process exit", "#1482"):
        assert tok in window, tok


def test_claude_md_mirror_present() -> None:
    text = _normalized(CLAUDE_MD)
    idx = text.index(ANCHOR_CLAUDE)
    # Window 7000: the farthest pinned token (#1482) MEASURES 3603 chars
    # from the anchor (token end 3608) on the live post-edit tree
    # (2026-08-09, whitespace-normalized); ~3.4k headroom for wording
    # tweaks without letting the pin drift file-wide (>= ~1k headroom per
    # the pilot-fence test's comment convention).
    window = text[idx : idx + 7000]
    for tok in (
        "ACROSS-CELL shard axis",
        "not shardable",
        "CHECKPOINT CADENCE",
        "never only at process exit",
        "#1345",
        "#1482",
    ):
        assert tok in window, tok


def test_teammate_dispatch_surface_carries_duties() -> None:
    text = _normalized(SIZING_MD)
    lo = text.index(TEAMMATE_ANCHOR)
    hi = text.index(TEAMMATE_END, lo)
    assert lo < hi
    window = text[lo:hi]
    for tok in ("ACROSS-CELL shard axis", "CHECKPOINT CADENCE", "Step 9a-ter"):
        assert tok in window, tok


def test_registered_in_step9c_workflow_invariant() -> None:
    spec = importlib.util.spec_from_file_location("select_step9c_tests_2040", SELECTOR_PY)
    assert spec is not None and spec.loader is not None
    sel = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(sel)
    assert PIN_FILE_RELPATH in sel.WORKFLOW_INVARIANT
