"""Bystander-panel size + decomposition tests for task #365.

Round-1 code-review ISSUE 5 was:

    bystanders_for(source) returns 23, plan demands 21.

The disambiguation reached in round 2:

  * The eval panel is exactly 24 personas; ``BYSTANDER_PANEL_SIZE`` = 23
    (panel size minus the source).
  * Plan v2 §6 phrasing "21 bystanders sampled from the #337/#296 source
    list" describes the NON-OCCUPATIONAL subset. The 2 sibling sources
    are bystanders for THIS cell but rotate as the source between cells.
  * The decomposition (in_domain, siblings, non_occupational) depends on
    the source — for librarian it's (0, 2, 21); for surgeon (1, 2, 20);
    for programmer (2, 2, 19).

These tests document the canonical N + assert the decomposition so the
contract cannot drift back.
"""

from __future__ import annotations

import pytest

from explore_persona_space.experiments.factor_screen_365 import (
    BYSTANDER_PANEL_SIZE,
    EVAL_PERSONAS_24,
    IN_DOMAIN_BYSTANDERS_BY_SOURCE,
    SOURCE_PERSONAS,
    bystanders_for,
    out_of_domain_bystanders_for,
)


def test_panel_size_is_24() -> None:
    """The full eval panel is exactly 24 personas (invariant)."""
    assert len(EVAL_PERSONAS_24) == 24


def test_canonical_bystander_size_is_23() -> None:
    """``BYSTANDER_PANEL_SIZE`` documents the canonical N (panel - source)."""
    assert BYSTANDER_PANEL_SIZE == 23


def test_bystanders_for_returns_exactly_panel_size() -> None:
    for src in SOURCE_PERSONAS:
        bystanders = bystanders_for(src)
        assert len(bystanders) == BYSTANDER_PANEL_SIZE
        # Source itself is excluded.
        assert src not in bystanders


def test_bystander_decomposition_matches_per_source() -> None:
    """Per-source (in_domain, siblings, non_occupational) decomposition.

    This is the round-2 disambiguation: the "21 bystanders" phrasing in plan
    v2 §6 holds strictly only for librarian (no occupational neighbours).
    """
    siblings_all = set(SOURCE_PERSONAS)
    expected = {
        "librarian": {"in_domain": 0, "siblings": 2, "non_occupational": 21},
        "surgeon": {"in_domain": 1, "siblings": 2, "non_occupational": 20},
        "programmer": {"in_domain": 2, "siblings": 2, "non_occupational": 19},
    }
    for src in SOURCE_PERSONAS:
        bystanders = bystanders_for(src)
        in_domain = sum(1 for p in bystanders if p in IN_DOMAIN_BYSTANDERS_BY_SOURCE[src])
        siblings = sum(1 for p in bystanders if p in siblings_all - {src})
        non_occ = sum(
            1
            for p in bystanders
            if p not in IN_DOMAIN_BYSTANDERS_BY_SOURCE[src] and p not in siblings_all
        )
        assert in_domain == expected[src]["in_domain"]
        assert siblings == expected[src]["siblings"]
        assert non_occ == expected[src]["non_occupational"]
        assert in_domain + siblings + non_occ == BYSTANDER_PANEL_SIZE


def test_out_of_domain_bystanders_exclude_in_domain() -> None:
    """``out_of_domain_bystanders_for`` returns BYSTANDER_PANEL_SIZE - in_domain."""
    for src in SOURCE_PERSONAS:
        ood = out_of_domain_bystanders_for(src)
        assert len(ood) == BYSTANDER_PANEL_SIZE - len(IN_DOMAIN_BYSTANDERS_BY_SOURCE[src])
        # No in-domain neighbour leaks through.
        for in_dom in IN_DOMAIN_BYSTANDERS_BY_SOURCE[src]:
            assert in_dom not in ood
        # Source itself is excluded.
        assert src not in ood


def test_bystanders_for_unknown_source_raises() -> None:
    with pytest.raises(ValueError):
        bystanders_for("not_a_real_persona")
