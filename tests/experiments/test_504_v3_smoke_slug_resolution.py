# ruff: noqa: RUF002, RUF003  # em-dash + ※ marker + × intentional
"""Task #504 round-6 v3-smoke-slug regression — cell_resolution must accept v3.

Pins the contract for the round-6 BLOCKER fix (epm:failure 2026-06-08T13:11:10Z,
``KeyError: "Unhandled #504 cell slug 'c504v3_smoke_eps2'"``): the v22
implementer added the v3 smoke slugs ``c504v3_smoke_eps{2,3}`` to
``CELL_SPECS_504``, the dispatcher, and the picker, but
``cell_resolution.negatives_for_cell_504`` only recognized the v1/v2 smoke
prefixes — every v3 smoke-cell spawn raised ``KeyError`` at cell build time.

This test asserts that:

1. ``negatives_for_cell_504`` returns ``[qwen_default, smoke_mid_band_n]`` for
   both v3 smoke slugs (parity with v1/v2 smoke cells).
2. ``arm_negatives_with_counts`` returns the same 2-persona negative set plus
   the expected (100, 100) row counts (matches v3 plan §4.1 + §11: 100
   default + 100 positioned mid-band = 200 total negs per smoke cell).
3. The v1 + v2 smoke paths still return the same shape (no regression).
4. A v3 smoke slug WITHOUT ``smoke_mid_band_n`` raises the documented
   ``ValueError`` (not the silent ``KeyError`` from the unhandled-branch).

CPU-only, sub-second; constructs no models / data and touches no GPU.
"""

from __future__ import annotations

import pytest

from explore_persona_space.experiments.contrastive_neg_geometry_504 import (
    ALWAYS_INCLUDE_NEGATIVE,
    NEG_EX_PER_PERSONA,
    PHASE0_SMOKE_SLUGS,
    PHASE0_SMOKE_SLUGS_V2,
    PHASE0_SMOKE_SLUGS_V3,
)
from explore_persona_space.experiments.contrastive_neg_geometry_504.cell_resolution import (
    arm_negatives_with_counts,
    negatives_for_cell_504,
)

# Sentinel mid-band negative persona picked by Phase 0.5 (the actual value
# is irrelevant to this test — we only care that the helper threads it
# through unchanged, alongside the qwen_default).
_SMOKE_MID_BAND_N = "scholar"


def test_v3_smoke_slugs_resolve_to_default_plus_mid_band() -> None:
    """Every v3 smoke slug returns [qwen_default, smoke_mid_band_n].

    Regression for the round-6 KeyError: prior to the cell_resolution.py fix,
    every slug in PHASE0_SMOKE_SLUGS_V3 fell through to the unhandled-branch
    raise; now they take the smoke-prefix branch with v3 included.
    """
    assert PHASE0_SMOKE_SLUGS_V3 == ("c504v3_smoke_eps2", "c504v3_smoke_eps3"), (
        f"Test pins the canonical v3 smoke slug set; got {PHASE0_SMOKE_SLUGS_V3!r}"
    )
    for slug in PHASE0_SMOKE_SLUGS_V3:
        negs = negatives_for_cell_504(
            slug,
            arm_to_positioned_n={},  # smoke cells don't consume the main-arm map
            smoke_mid_band_n=_SMOKE_MID_BAND_N,
        )
        assert negs == [ALWAYS_INCLUDE_NEGATIVE, _SMOKE_MID_BAND_N], (
            f"v3 smoke slug {slug!r} returned wrong negatives: {negs!r}"
        )


def test_v3_smoke_slugs_counts_match_plan() -> None:
    """v3 smoke arm composition: 100 default + 100 positioned = 200 total negs.

    Pins plan v3 §4.1 + §11: smoke cells match v2 composition exactly
    (NEG_EX_PER_PERSONA = 100 per persona × 2 personas = 200 total). Only
    EPOCHS varies between v3 smoke cells; the negative budget does NOT.
    """
    expected_counts = [NEG_EX_PER_PERSONA, NEG_EX_PER_PERSONA]
    assert expected_counts == [100, 100], (
        "Test pins NEG_EX_PER_PERSONA=100 — if you've changed the v3 smoke "
        "composition, update the plan and this assertion together."
    )
    for slug in PHASE0_SMOKE_SLUGS_V3:
        negs, counts = arm_negatives_with_counts(
            slug,
            arm_to_positioned_n={},
            smoke_mid_band_n=_SMOKE_MID_BAND_N,
        )
        assert negs == [ALWAYS_INCLUDE_NEGATIVE, _SMOKE_MID_BAND_N], (
            f"v3 smoke slug {slug!r} negatives mismatch: {negs!r}"
        )
        assert counts == expected_counts, (
            f"v3 smoke slug {slug!r} counts mismatch: {counts!r} (expected {expected_counts!r})"
        )
        assert sum(counts) == 200, (
            f"v3 smoke slug {slug!r} total negs {sum(counts)} != 200 (plan v3 §11 1:1 ratio)"
        )


def test_v1_and_v2_smoke_paths_unchanged() -> None:
    """No regression on the pre-existing v1 + v2 smoke prefixes.

    The round-6 fix widens the prefix tuple from ``("c504_smoke_",
    "c504v2_smoke_")`` to include ``"c504v3_smoke_"``; this asserts the
    v1/v2 paths still resolve to the same 2-persona negative set.
    """
    for slug in (*PHASE0_SMOKE_SLUGS, *PHASE0_SMOKE_SLUGS_V2):
        negs = negatives_for_cell_504(
            slug,
            arm_to_positioned_n={},
            smoke_mid_band_n=_SMOKE_MID_BAND_N,
        )
        assert negs == [ALWAYS_INCLUDE_NEGATIVE, _SMOKE_MID_BAND_N], (
            f"Regression: v1/v2 smoke slug {slug!r} returned {negs!r}"
        )


def test_v3_smoke_slug_without_mid_band_raises_value_error() -> None:
    """A v3 smoke slug with smoke_mid_band_n=None raises ValueError, not KeyError.

    Pins the failure mode of the recognition branch: the smoke-prefix branch
    catches v3 slugs, then explicitly validates that smoke_mid_band_n was
    supplied — missing it now surfaces as a documented ValueError, not the
    pre-fix silent KeyError ("Unhandled #504 cell slug") that misled callers.
    """
    for slug in PHASE0_SMOKE_SLUGS_V3:
        with pytest.raises(ValueError, match="smoke_mid_band_n"):
            negatives_for_cell_504(
                slug,
                arm_to_positioned_n={},
                smoke_mid_band_n=None,
            )
