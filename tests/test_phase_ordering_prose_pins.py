"""Durability pins for the #1612 phase-ordering checkpoint high-water clause.

Token-presence asserts over the two workflow-surface files carrying the
phase-ordering REVISE trigger — a later prose edit that silently drops the
enforcement text fails here (the c31 durability-pin discipline; lineage:
#884/#1045/#1134/#1395). Whitespace-normalized matching so a future re-wrap
cannot false-FAIL the pin (the test_fanout_accumulation_prose_pins.py shape).
"""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def _norm(text: str) -> str:
    """Collapse all whitespace runs to single spaces (wrap-proof matching)."""
    return " ".join(text.split())


def test_critic_lens_item_16_phase_ordering_extension():
    # Item 16 carries the PHASE-ORDERING EXTENSION: trigger, the
    # downstream-reap clause, incident citation, and the N/A escape string.
    # Containment bounds (item-16 heading < start < item-17 heading) pin the
    # clause INSIDE item 16 (the span the v2 efficiency-critic reads is
    # item-scoped — items 10/13/16).
    text = _norm((REPO_ROOT / ".claude" / "rules" / "critic-lens-reference.md").read_text())
    start = text.index("PHASE-ORDERING EXTENSION (#1612, from incident #1586 r5)")
    i16 = text.index("16. **Merge-disk budget")
    assert i16 < start  # inside item 16, not merely before item 17
    section = text[start : text.index("17. **Persona-vectors", start)]
    assert "cannot bound an upstream train-all-cells accumulation" in section
    assert "#1586" in section
    assert "N/A — no checkpoint accumulation across phases" in section


def test_plan_compute_sizing_phase_ordering_block_names_item_16():
    # The rule block names its lens owner (the bidirectional cross-ref).
    text = _norm((REPO_ROOT / ".claude" / "rules" / "plan-compute-sizing.md").read_text())
    start = text.index("**Phase-ordering checkpoint high-water")
    section = text[start : text.index("**Out-root mount binding", start)]
    assert "Methodology lens item 16 PHASE-ORDERING EXTENSION" in section
    assert "IMPLEMENTED phase ordering" in section
