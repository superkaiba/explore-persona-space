"""Durability pins for the #1541 fan-out end-of-run accumulation clause.

Token-presence asserts over the two workflow-surface files carrying the
fan-out accumulation REVISE trigger — a later prose edit that silently
drops the enforcement text fails here (the c31 durability-pin discipline;
lineage: #884/#1045/#1134/#1395). Matching runs on whitespace-normalized
text so a future re-wrap of a pinned phrase across physical lines cannot
false-FAIL the pin (#1541 review-round concern).
"""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def _norm(text: str) -> str:
    """Collapse all whitespace runs to single spaces (wrap-proof matching)."""
    return " ".join(text.split())


def test_critic_lens_item_16_fanout_accumulation_extension():
    # Item 16 carries the FAN-OUT ACCUMULATION EXTENSION: trigger, the
    # tooling-honesty clause, incident citation, and the N/A escape string.
    # The containment bounds (item-16 heading < start < item-17 heading)
    # pin the clause INSIDE item 16.
    text = _norm((REPO_ROOT / ".claude" / "rules" / "critic-lens-reference.md").read_text())
    start = text.index("FAN-OUT ACCUMULATION EXTENSION (#1541, from incident #1481)")
    i16 = text.index("16. **Merge-disk budget")
    assert i16 < start  # inside item 16, not merely before item 17
    section = text[start : text.index("17. **Persona-vectors", start)]
    assert "N/A — no per-cell retained outputs" in section
    assert "#1481" in section
    assert "clean_experiment_downloads" in section


def test_plan_compute_sizing_fanout_block_names_item_16():
    # The rule block names its lens owner (the bidirectional cross-ref).
    text = _norm((REPO_ROOT / ".claude" / "rules" / "plan-compute-sizing.md").read_text())
    start = text.index("**Fan-out end-of-run accumulated footprint")
    section = text[start : text.index("**Out-root mount binding", start)]
    assert "Methodology lens item 16 FAN-OUT ACCUMULATION EXTENSION" in section
    assert "discarded_artifacts" in section
