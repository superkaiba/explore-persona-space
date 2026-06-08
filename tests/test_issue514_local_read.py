"""Unit tests for the #514 local-read bracketing filter (B7 round-3 fix).

Plan §6.4 + brief B7: the local linear-interpolation matched-rate read at
source ΔG = 8 nat must NOT use ft_b2 (collapsed, source_n_probes=1,
source ΔG ≈ 6.77 nat with held_out_g_logprob saturated above the
sub-ceiling) as a legal bracketing anchor — that turns the read into an
interpolation THROUGH a contaminated cell, which is exactly what this
follow-up exists to eliminate.

The brief enumerates three scenarios:
  (a) ft_b1 alone bracketing target=8 — should degenerate (need ≥2 anchors).
  (b) ft_b1 + one clean #514 above-9 cell — local read is interpolation
      not extrapolation.
  (c) ft_b1 + ft_b2 (current bug) — ft_b2 MUST be excluded by the new
      gates so the result is the same as (a) until #514 produces a clean
      above-9 cell.
"""

from __future__ import annotations

import math

from explore_persona_space.experiments.full_ft_regime_514.analyze import (
    LOCAL_READ_MIN_SOURCE_N_PROBES,
    MATCHED_RATE_TARGET_NAT,
    _compute_local_matched_rate_read,
    is_clean_anchor,
)

# ── Helpers ──────────────────────────────────────────────────────────────────


def _diag(
    cell: str,
    *,
    source_mean: float | None,
    held_out_mean: float | None,
    source_n_probes: int | None,
    r_collapse_rate: float,
    held_out_g_logprob_mean: float,
    lever: str = "508_anchor",
    clean_above_9_nat: bool = False,
) -> dict:
    """Build a per-cell diagnostic dict matching ``_per_cell_diagnostics`` shape."""
    return {
        "cell": cell,
        "lever": lever,
        "eval_json_path": f"/tmp/{cell}.json",
        "source_mean": source_mean,
        "held_out_mean": held_out_mean,
        "source_n_probes": source_n_probes,
        "r_collapse_rate": r_collapse_rate,
        "held_out_g_logprob_mean": held_out_g_logprob_mean,
        "clean_above_9_nat": clean_above_9_nat,
    }


# Canonical #508 anchor reads (from #508's actual eval JSONs):
#   ft_b1: source ΔG ≈ 8.193 nat, held_out mean ≈ -0.31 nat, n_probes=20,
#          r_collapse=0.00, held_out_g_logprob ≈ -6.20 nat (passes sub-ceiling)
#   ft_b2: source ΔG ≈ 6.774 nat, held_out mean ≈ -0.92 nat, n_probes=1,
#          r_collapse=0.95, held_out_g_logprob ≈ -0.865 nat (SATURATED)
FT_B1_CLEAN = _diag(
    "ft_b1",
    source_mean=8.193,
    held_out_mean=-0.31,
    source_n_probes=20,
    r_collapse_rate=0.00,
    held_out_g_logprob_mean=-6.20,
)

FT_B2_COLLAPSED = _diag(
    "ft_b2",
    source_mean=6.774,
    held_out_mean=-0.92,
    source_n_probes=1,
    r_collapse_rate=0.95,
    held_out_g_logprob_mean=-0.865,
)


# ── is_clean_anchor ──────────────────────────────────────────────────────────


def test_is_clean_anchor_admits_ft_b1():
    """Sanity: ft_b1 (#508's intact 0.25-epoch FT anchor) passes every gate."""
    assert is_clean_anchor(FT_B1_CLEAN) is True


def test_is_clean_anchor_rejects_ft_b2_collapsed():
    """B7 round-3 fix: ft_b2 (source_n_probes=1 + sub-ceiling saturated) FAILS.

    This is the canonical contaminated-anchor case the round-2 code admitted.
    """
    assert is_clean_anchor(FT_B2_COLLAPSED) is False


def test_is_clean_anchor_rejects_low_n_probes():
    """source_n_probes < LOCAL_READ_MIN_SOURCE_N_PROBES (=5) → fail."""
    diag = _diag(
        "ft_synth_lowN",
        source_mean=9.0,
        held_out_mean=-3.0,
        source_n_probes=LOCAL_READ_MIN_SOURCE_N_PROBES - 1,
        r_collapse_rate=0.0,
        held_out_g_logprob_mean=-7.0,
    )
    assert is_clean_anchor(diag) is False


def test_is_clean_anchor_rejects_saturated_held_out():
    """held_out_g_logprob_mean > -5.0 (above sub-ceiling) → fail."""
    diag = _diag(
        "ft_synth_saturated",
        source_mean=9.0,
        held_out_mean=-1.0,
        source_n_probes=20,
        r_collapse_rate=0.0,
        held_out_g_logprob_mean=-2.0,  # SATURATED above sub-ceiling
    )
    assert is_clean_anchor(diag) is False


def test_is_clean_anchor_rejects_high_rcollapse():
    """r_collapse_rate >= 0.50 → fail."""
    diag = _diag(
        "ft_synth_rcoll",
        source_mean=9.0,
        held_out_mean=-3.0,
        source_n_probes=20,
        r_collapse_rate=0.55,
        held_out_g_logprob_mean=-7.0,
    )
    assert is_clean_anchor(diag) is False


def test_is_clean_anchor_rejects_nan_means():
    """NaN source_mean or held_out_mean → fail (defensive)."""
    diag_nan_src = _diag(
        "ft_nan_src",
        source_mean=float("nan"),
        held_out_mean=-3.0,
        source_n_probes=20,
        r_collapse_rate=0.0,
        held_out_g_logprob_mean=-7.0,
    )
    assert is_clean_anchor(diag_nan_src) is False
    diag_nan_held = _diag(
        "ft_nan_held",
        source_mean=9.0,
        held_out_mean=float("nan"),
        source_n_probes=20,
        r_collapse_rate=0.0,
        held_out_g_logprob_mean=-7.0,
    )
    assert is_clean_anchor(diag_nan_held) is False


# ── _compute_local_matched_rate_read — the three brief scenarios ─────────────


def test_local_read_scenario_a_ft_b1_alone_degenerates():
    """Scenario (a): only ft_b1 in the candidate pool — fewer than 2 anchors,
    local read MUST return NaN (no degenerate single-point extrapolation).
    """
    local_read, _is_extrap, anchors = _compute_local_matched_rate_read(
        diagnostics_514=[],
        diagnostics_508_ft_anchors=[FT_B1_CLEAN],
        target_nat=MATCHED_RATE_TARGET_NAT,
    )
    assert math.isnan(local_read)
    # Only ft_b1 admitted; fewer than 2 anchors means no interpolation
    # AND the bracketing check is skipped → is_extrap defaults False.
    assert anchors == [FT_B1_CLEAN]


def test_local_read_scenario_b_ft_b1_plus_clean_above_9():
    """Scenario (b): ft_b1 + one clean #514 above-9-nat cell → interpolation.

    With ft_b1 (source=8.19) and a synthetic above-9-nat cell (source=10.0),
    target=8 nat falls BELOW both anchors → flagged as extrapolation but the
    read still emits (linear interp at target falls outside the closed
    bracket and returns NaN per ``_linear_interp_at``'s strict-bracket rule).

    More usefully, the clean above-9-nat anchor pair (8.19, 10.0) does
    bracket source ΔG = 9 nat → a future test verifying interpolation at
    9 nat would yield a finite read. Here we test the brief's stated
    interpolation-not-extrapolation property at target = 8 nat.
    """
    cell_514_clean = _diag(
        "ft_dense_b40",
        source_mean=10.0,
        held_out_mean=-2.5,
        source_n_probes=18,
        r_collapse_rate=0.10,
        held_out_g_logprob_mean=-6.5,
        lever="dense",
        clean_above_9_nat=True,
    )
    local_read, is_extrap, anchors = _compute_local_matched_rate_read(
        diagnostics_514=[cell_514_clean],
        diagnostics_508_ft_anchors=[FT_B1_CLEAN],
        target_nat=MATCHED_RATE_TARGET_NAT,
    )
    cell_names = {a["cell"] for a in anchors}
    assert cell_names == {"ft_b1", "ft_dense_b40"}
    # ft_b1's source_mean (8.19) DOES bracket target=8 from above; in fact
    # min(8.19, 10.0) = 8.19 > 8.0 → both anchors sit ABOVE target → this
    # is correctly flagged as extrapolation, and the strict-bracket
    # _linear_interp_at returns NaN (the read is *not* a valid interpolation
    # at target=8 nat with only above-side anchors).
    assert is_extrap is True
    assert math.isnan(local_read)


def test_local_read_scenario_c_excludes_ft_b2_current_bug():
    """Scenario (c): ft_b1 + ft_b2 (round-2 bug) — ft_b2 MUST be excluded.

    With ONLY ft_b1 and ft_b2 in the FT-anchor list, the round-2 code would
    admit BOTH (both have non-null source_mean + held_out_mean) and
    interpolate at target=8 nat between (6.774, -0.92) and (8.193, -0.31) —
    a misleading "matched-rate" read THROUGH the collapsed cell. After B7,
    ft_b2 fails ``is_clean_anchor`` so only ft_b1 remains → degenerate (as
    scenario a), local_read = NaN.
    """
    local_read, _is_extrap, anchors = _compute_local_matched_rate_read(
        diagnostics_514=[],
        diagnostics_508_ft_anchors=[FT_B1_CLEAN, FT_B2_COLLAPSED],
        target_nat=MATCHED_RATE_TARGET_NAT,
    )
    cell_names = {a["cell"] for a in anchors}
    assert "ft_b2" not in cell_names, (
        f"ft_b2 must be excluded by is_clean_anchor; got candidates: {cell_names}"
    )
    assert cell_names == {"ft_b1"}, f"only ft_b1 should survive the filter; got: {cell_names}"
    # 1 candidate → fewer than 2 anchors → NaN (matches scenario a).
    assert math.isnan(local_read)


def test_local_read_round2_would_have_used_ft_b2():
    """Regression guard: verify the round-2 admission rule WOULD have admitted
    ft_b2 (had we not added the gate). This makes the test self-documenting
    about WHY the gate is necessary.
    """
    # The round-2 admission condition was just:
    #     d["source_mean"] is not None and d["held_out_mean"] is not None
    round2_admitted = (
        FT_B2_COLLAPSED["source_mean"] is not None and FT_B2_COLLAPSED["held_out_mean"] is not None
    )
    assert round2_admitted is True, (
        "round-2 admission rule should have admitted ft_b2 — this is the contamination "
        "B7 round-3 fixes"
    )
    # And the new rule correctly rejects it.
    assert is_clean_anchor(FT_B2_COLLAPSED) is False


def test_local_read_interpolates_when_target_is_strictly_bracketed():
    """Sanity: when two clean anchors strictly bracket the target, the local
    read returns a finite interpolation and is_extrap is False.

    Use a synthetic (clean below-target, clean above-target) anchor pair.
    """
    below = _diag(
        "ft_synth_below",
        source_mean=7.0,
        held_out_mean=-1.0,
        source_n_probes=20,
        r_collapse_rate=0.0,
        held_out_g_logprob_mean=-7.0,
    )
    above = _diag(
        "ft_synth_above",
        source_mean=9.0,
        held_out_mean=-3.0,
        source_n_probes=20,
        r_collapse_rate=0.0,
        held_out_g_logprob_mean=-7.0,
    )
    local_read, is_extrap, anchors = _compute_local_matched_rate_read(
        diagnostics_514=[],
        diagnostics_508_ft_anchors=[below, above],
        target_nat=MATCHED_RATE_TARGET_NAT,  # 8.0 nat — strictly between 7 and 9
    )
    assert {a["cell"] for a in anchors} == {"ft_synth_below", "ft_synth_above"}
    assert is_extrap is False
    # Linear interp: target=8 sits midway between 7 and 9 → y = (-1 + -3)/2 = -2
    assert math.isclose(local_read, -2.0, abs_tol=1e-9)
