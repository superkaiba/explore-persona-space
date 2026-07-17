"""Durability pins for the #1444 out-root mount-binding lens clause.

Token-presence asserts over the two workflow-surface files carrying the
mount-binding REVISE trigger — a later prose edit that silently
drops the enforcement text fails here (the c31 durability-pin discipline;
lineage: #884/#1045/#1134/#1395).
"""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_critic_lens_reference_item_16_mount_binding_extension():
    # Item 16 carries the MOUNT-BINDING EXTENSION: trigger, preamble-assert
    # helper name, failover citation, and the N/A escape string. The
    # containment bounds (item-16 heading < start < item-17 heading) pin the
    # clause INSIDE item 16 — the span the v2 efficiency-critic reads is
    # item-scoped (items 10/13/16), so before-item-17 alone is not enough
    # (Statistics-critic Must-Fix, plan v3).
    text = (REPO_ROOT / ".claude" / "rules" / "critic-lens-reference.md").read_text()
    start = text.index("MOUNT-BINDING EXTENSION (#1414, from incident #1333)")
    i16 = text.index("16. **Merge-disk budget")
    assert i16 < start  # inside item 16, not merely before item 17
    section = text[start : text.index("17. **Persona-vectors", start)]
    assert "assert_out_root_headroom" in section
    assert "#1112" in section
    assert "N/A — no out-root writes" in section


def test_plan_compute_sizing_mount_block_names_item_16():
    # The rule block names its lens owner (the bidirectional cross-ref).
    text = (REPO_ROOT / ".claude" / "rules" / "plan-compute-sizing.md").read_text()
    start = text.index("**Out-root mount binding")
    section = text[start : text.index("**Sentinel-signaling", start)]
    assert "Methodology lens item 16 MOUNT-BINDING EXTENSION" in section
