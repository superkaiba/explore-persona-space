"""Durability pins for the #2237 fan-out pod-name collision clause.

Token-presence asserts over the workflow-surface file carrying the
fan-out POD-NAME REVISE trigger — a later prose edit that silently drops
the enforcement text fails here (the c31 durability-pin discipline;
lineage: #884/#1045/#1134/#1395/#1541/#2236). Matching runs on
whitespace-normalized text so a future re-wrap of a pinned phrase across
physical lines cannot false-FAIL the pin (the #1541 review-round
convention). Unlike the #2236 STAGING sibling there is no
plan-compute-sizing.md block to pin here — the c58 backstop
cross-reference lives inside the lens clause itself, so this file pins
ONE surface (the c58 check body is pinned separately by
tests/test_verify_plan_c58_fanout_pod_name.py).
"""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def _norm(text: str) -> str:
    """Collapse all whitespace runs to single spaces (wrap-proof matching)."""
    return " ".join(text.split())


def test_critic_lens_item_16_fanout_pod_name_extension():
    # Item 16 carries the FAN-OUT POD-NAME EXTENSION: trigger, REVISE
    # condition (incl. the --lane-suffix RunPod exclusion), the
    # prior-escapes disjointness sentence, the kind exemption, the
    # plan-time-only closer, and the c58 backstop pointer. The
    # containment bounds (item-16 heading < clause start < item-17
    # heading) pin the clause INSIDE item 16.
    text = _norm((REPO_ROOT / ".claude" / "rules" / "critic-lens-reference.md").read_text())
    start = text.index("FAN-OUT POD-NAME EXTENSION (#2237, from incident #2054)")
    i16 = text.index("16. **Merge-disk budget")
    assert i16 < start  # inside item 16, not merely before item 17
    section = text[start : text.index("17. **Persona-vectors", start)]
    # Trigger: N concurrent pods for ONE issue must mint DISTINCT names
    # on the lane the plan actually routes to.
    assert "fans `N > 1` CONCURRENT pods for ONE issue" in section
    assert "mints N DISTINCT pod names on the lane it actually routes to" in section
    # REVISE-when list — incl. that per-launch `--lane-suffix` DOES satisfy
    # a RunPod fan-out since #2145 (honored on GCP + SLURM + RunPod), while
    # a SUFFIX-LESS RunPod launch still mints the colliding per-issue name.
    assert "REVISE when the plan names NONE of" in section
    assert "`dispatch_issue.py launch --lane-suffix <slug>`" in section
    assert "honored on GCP + SLURM + RunPod since #2145" in section
    assert "`pod.py provision --name-suffix <slug>`" in section
    assert "SUFFIX-LESS RunPod launch mints the per-issue name" in section
    assert "WITHOUT distinct suffixes collide" in section
    # Prior-escapes disjointness: the earlier item-16 escape lists (the
    # accumulation/staging extensions') must not swallow this clause.
    assert "The prior extensions' escapes do NOT cover this one" in section
    # Exemption + plan-time-only closer.
    assert "`kind: infra|batch|survey` exempt" in section
    assert "Plan-time pod-naming check only, never a mid-run gate" in section
    # c58 backstop pointer: the WARN-only early-warning net; this clause
    # stays the lane-agnostic binding gate.
    assert "c58_fanout_pod_name_collision" in section
    assert "WARN-only" in section
    assert "early-warning net" in section
    assert "lane-agnostic binding gate" in section
    # Dedupe guard: the POD-NAME extension is not the STAGING or
    # ACCUMULATION extension (different trigger, different remedy).
    assert "FAN-OUT STAGING EXTENSION" not in section
    assert "FAN-OUT ACCUMULATION EXTENSION" not in section
