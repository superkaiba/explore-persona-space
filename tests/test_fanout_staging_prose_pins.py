"""Durability pins for the #2236 fan-out same-prefix staging clause.

Token-presence asserts over the two workflow-surface files carrying the
fan-out staging REVISE trigger — a later prose edit that silently drops
the enforcement text fails here (the c31 durability-pin discipline;
lineage: #884/#1045/#1134/#1395/#1541). Matching runs on
whitespace-normalized text so a future re-wrap of a pinned phrase across
physical lines cannot false-FAIL the pin (the #1541 review-round
convention). The base-escape override sentence is pinned INSIDE the
extension's own containment bounds — it is what stops item 16's
fits-quota / no-merges / kind escapes from swallowing the #2054 shape
(the round-1 plan-review blocker on this task).
"""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def _norm(text: str) -> str:
    """Collapse all whitespace runs to single spaces (wrap-proof matching)."""
    return " ".join(text.split())


def test_critic_lens_item_16_fanout_staging_extension():
    # Item 16 carries the FAN-OUT STAGING EXTENSION: trigger, REVISE
    # condition, incident citation, the named remedies, and the
    # base-escape override sentence. The containment bounds (item-16
    # heading < clause start < item-17 heading) pin the clause INSIDE
    # item 16.
    text = _norm((REPO_ROOT / ".claude" / "rules" / "critic-lens-reference.md").read_text())
    start = text.index("FAN-OUT STAGING EXTENSION (#2236, from incident #1739)")
    i16 = text.index("16. **Merge-disk budget")
    assert i16 < start  # inside item 16, not merely before item 17
    section = text[start : text.index("17. **Persona-vectors", start)]
    # The round-1 blocker pin: the override sentence sits inside THESE
    # bounds (the accumulation/mount-binding copies live earlier in item
    # 16, outside [start, item-17)), whitespace-normalized too — it wraps
    # across physical lines in the file.
    assert "escapes above do NOT cover this extension" in section
    assert "concurrent same-prefix STAGING" in section
    assert "#1739" in section
    assert "jittered start offsets" in section
    assert "plan-compute-sizing.md" in section
    # Retain-vs-pull dedupe guard: the STAGING extension is not the
    # ACCUMULATION extension (different trigger, different remedy).
    assert "FAN-OUT ACCUMULATION EXTENSION" not in section


def test_plan_compute_sizing_fanout_staging_block_names_item_16_and_c57():
    # The rule block names its lens owner AND the shipped c57 backstop
    # (the bidirectional cross-ref); the pre-#2236 "no verify_plan.py
    # backstop in v1" disclaimer is gone from THIS block (sibling blocks
    # legitimately keep theirs).
    text = _norm((REPO_ROOT / ".claude" / "rules" / "plan-compute-sizing.md").read_text())
    start = text.index("**Fan-out over the same HF prefix")
    section = text[start : text.index("**Sentinel-signaling workloads", start)]
    assert "Methodology lens item 16 FAN-OUT STAGING EXTENSION" in section
    assert "c57_fanout_prefix_staging" in section
    assert "WARN-only" in section
    assert "no verify_plan.py backstop in v1" not in section
    # The lens stays the binding gate; c57 is the early-warning net.
    assert "binding gate" in section
