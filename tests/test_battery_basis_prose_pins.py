"""Durability pins for the #1395 battery-basis prose legs.

Token-presence asserts over the four workflow-surface files carrying the
binding battery-basis REVISE trigger + its MUST anchors — a later prose
edit that silently drops the enforcement text fails here (the c31
durability-pin discipline; lineage: #884/#1045/#1134).
"""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_critic_lens_reference_item_iii_battery_basis_revise():
    # Item 10(iii) carries the two-arm binding REVISE trigger: (A) an
    # asserted / FLOP-derived battery basis; (B) a pilot-gated battery
    # headline booking the naive projection (#1092).
    text = (REPO_ROOT / ".claude" / "rules" / "critic-lens-reference.md").read_text()
    item_iii = next(line for line in text.splitlines() if "**(iii)" in line)
    assert "batched does NOT exempt the basis" in item_iii
    assert "#1092" in item_iii


def test_plan_compute_sizing_per_cell_fit_phases_battery_must():
    # § Per-cell fit phases: the MUST scope names draw batteries, and the
    # pilot-gated booking presumption is MUST-level ("BOOKS >=2x").
    text = (REPO_ROOT / ".claude" / "rules" / "plan-compute-sizing.md").read_text()
    start = text.index("**Per-cell fit phases")
    section = text[start : text.index("**Store-heavy", start)]
    assert "null-draw battery" in section
    assert "BOOKS ≥2×" in section  # noqa: RUF001 — the pinned token is the rule file's real text


def test_planner_section_reference_clause_b_battery_exception():
    # §9 sizing clause (b): the FLOP-floor allowance is closed for
    # above-floor draw batteries.
    text = (REPO_ROOT / ".claude" / "rules" / "planner-section-reference.md").read_text()
    start = text.index("(b) Ground `per_call_cost`")
    clause_b = text[start : text.index("(c) ", start)]
    assert "null-draw battery above the ~15\u201330 min phase floor" in clause_b


def test_efficiency_critic_capsule_battery_basis_clause():
    # v2 capsule 2 carries the non-binding mirror clause so the capsule
    # cannot contradict the pointer-loaded item 10(iii) span.
    text = (REPO_ROOT / ".claude" / "agents" / "efficiency-critic.md").read_text()
    assert "Batched does not exempt the basis" in text
