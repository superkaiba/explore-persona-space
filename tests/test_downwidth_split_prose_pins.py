"""Durability pins for the #1633 down-width split prose legs.

Token-presence asserts over the three workflow-surface files carrying the
multi-arm min-width naming duty + the stall-time down-width split (the c31
durability-pin discipline; lineage: #884/#1045/#1134). Auto-selected on
plan-compute-sizing.md / compute-backend-failover.md diffs by the #1496
rules-pin discovery arm (basename substring).
"""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_plan_compute_sizing_downwidth_split_clause():
    text = (REPO_ROOT / ".claude" / "rules" / "plan-compute-sizing.md").read_text()
    start = text.index("**Multi-arm min-width + stall-time down-width split")
    section = text[start:]
    assert "MINIMUM runnable width" in section
    assert "≥ ~1 h" in section
    assert "#1112" in section
    assert "#1121 wide-FIRST walk" in section


def test_planner_spec_section9_downwidth_pointer():
    text = (REPO_ROOT / ".claude" / "agents" / "planner.md").read_text()
    assert "MINIMUM runnable width" in text
    assert "down-width split" in text


def test_compute_backend_failover_downwidth_crossref():
    text = (REPO_ROOT / ".claude" / "rules" / "compute-backend-failover.md").read_text()
    assert "Multi-arm min-width + stall-time" in text
    assert "down-width split" in text
