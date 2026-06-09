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
    local_read, _is_extrap, _extrap_dist, anchors = _compute_local_matched_rate_read(
        diagnostics_514=[],
        diagnostics_508_ft_anchors=[FT_B1_CLEAN],
        target_nat=MATCHED_RATE_TARGET_NAT,
    )
    assert math.isnan(local_read)
    # Only ft_b1 admitted; fewer than 2 anchors means no interpolation
    # AND the bracketing check is skipped → is_extrap defaults False.
    assert anchors == [FT_B1_CLEAN]


def test_local_read_scenario_b_ft_b1_plus_clean_above_9():
    """Scenario (b): ft_b1 + one clean #514 above-9-nat cell → extrapolation
    that emits a FINITE value (B10 round-4 pivot).

    With ft_b1 (source=8.193) and a synthetic above-9-nat cell (source=10.0),
    target=8 nat sits 0.193 nat BELOW the lower anchor → flagged as
    extrapolation. Per plan §6.4 v3 pivot the read MUST emit a finite linear
    extrapolation rather than NaN (the round-3 strict-bracket behavior
    defeated the H1 headline). The signed extrapolation distance is
    ``target - min(xs) = 8.0 - 8.193 = -0.193`` nat.

    Pre-B10 (round 3) this test asserted ``is_extrap is True AND
    math.isnan(local_read)`` — that codified the round-3 bug. The new policy
    keeps the True extrapolation flag, but replaces the NaN with the finite
    extrapolated value and adds the signed-distance assertion.
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
    local_read, is_extrap, extrap_dist, anchors = _compute_local_matched_rate_read(
        diagnostics_514=[cell_514_clean],
        diagnostics_508_ft_anchors=[FT_B1_CLEAN],
        target_nat=MATCHED_RATE_TARGET_NAT,
    )
    cell_names = {a["cell"] for a in anchors}
    assert cell_names == {"ft_b1", "ft_dense_b40"}
    # min(8.193, 10.0) = 8.193 > 8.0 → both anchors sit ABOVE target → flag
    # as extrapolation.
    assert is_extrap is True
    # B10: signed extrapolation distance = target - min(xs) = 8.0 - 8.193
    # ≈ -0.193 nat (extrapolating BELOW the lower anchor).
    assert math.isclose(extrap_dist, 8.0 - 8.193, abs_tol=1e-9)
    # Linear extrapolation from the (8.193, -0.31) → (10.0, -2.5) anchor pair
    # at target=8.0: y = -0.31 + (8.0 - 8.193) / (10.0 - 8.193) * (-2.5 - -0.31)
    slope = (-2.5 - (-0.31)) / (10.0 - 8.193)
    expected = -0.31 + (8.0 - 8.193) * slope
    assert math.isfinite(local_read)
    assert math.isclose(local_read, expected, abs_tol=1e-9)


def test_local_read_scenario_c_excludes_ft_b2_current_bug():
    """Scenario (c): ft_b1 + ft_b2 (round-2 bug) — ft_b2 MUST be excluded.

    With ONLY ft_b1 and ft_b2 in the FT-anchor list, the round-2 code would
    admit BOTH (both have non-null source_mean + held_out_mean) and
    interpolate at target=8 nat between (6.774, -0.92) and (8.193, -0.31) —
    a misleading "matched-rate" read THROUGH the collapsed cell. After B7,
    ft_b2 fails ``is_clean_anchor`` so only ft_b1 remains → degenerate (as
    scenario a), local_read = NaN.
    """
    local_read, _is_extrap, _extrap_dist, anchors = _compute_local_matched_rate_read(
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
    local_read, is_extrap, extrap_dist, anchors = _compute_local_matched_rate_read(
        diagnostics_514=[],
        diagnostics_508_ft_anchors=[below, above],
        target_nat=MATCHED_RATE_TARGET_NAT,  # 8.0 nat — strictly between 7 and 9
    )
    assert {a["cell"] for a in anchors} == {"ft_synth_below", "ft_synth_above"}
    assert is_extrap is False
    # B10: when strictly bracketed, the extrapolation distance is NaN.
    assert math.isnan(extrap_dist)
    # Linear interp: target=8 sits midway between 7 and 9 → y = (-1 + -3)/2 = -2
    assert math.isclose(local_read, -2.0, abs_tol=1e-9)


def test_local_read_admits_clean_below_9nat_514_cell():
    """Free-reanalysis fix (2026-06-08): a CLEAN #514 cell BELOW 9 nat is
    admitted as a local-read anchor even though ``clean_above_9_nat`` is False.

    The #514 candidate loop now gates on ``is_clean_anchor`` (not the
    ``clean_above_9_nat`` H1-bracketing flag), so the clean lower-LR 50%-epoch
    cell (``ft_lowlr_b50``, source ΔG ≈ 7.43 nat, 0% r-collapse) is admitted
    and pairs with #508 ``ft_b1`` (8.20 nat) to STRADDLE target=8.0 nat as a
    true interpolation. Before the fix this cell was discarded and the read
    extrapolated ~0.2 nat below ft_b1. Mirrors ft_lowlr_b50's real values.
    """
    ft_lowlr_b50 = _diag(
        "ft_lowlr_b50",
        source_mean=7.428,
        held_out_mean=3.440,
        source_n_probes=20,
        r_collapse_rate=0.0,
        held_out_g_logprob_mean=-20.42,
        lever="lowlr",
        clean_above_9_nat=False,  # below the 9-nat H1 gate — the whole point
    )
    ft_b1 = _diag(
        "ft_b1",
        source_mean=8.198,
        held_out_mean=3.590,
        source_n_probes=20,
        r_collapse_rate=0.0,
        held_out_g_logprob_mean=-20.24,
    )
    local_read, is_extrap, extrap_dist, anchors = _compute_local_matched_rate_read(
        diagnostics_514=[ft_lowlr_b50],
        diagnostics_508_ft_anchors=[ft_b1],
        target_nat=MATCHED_RATE_TARGET_NAT,  # 8.0, strictly inside [7.428, 8.198]
    )
    cell_names = {a["cell"] for a in anchors}
    assert "ft_lowlr_b50" in cell_names, (
        f"clean below-9-nat #514 cell must be admitted; got candidates: {cell_names}"
    )
    assert is_extrap is False, "target 8.0 is strictly bracketed → true interpolation"
    assert math.isnan(extrap_dist)
    # Linear interp at 8.0 between (7.428, 3.440) and (8.198, 3.590):
    #   t = (8.0 - 7.428)/(8.198 - 7.428) = 0.7429 → 3.440 + 0.7429*0.150 ≈ 3.551
    assert math.isclose(local_read, 3.551, abs_tol=5e-3)


def test_compute_matched_rate_gap_514_happy_path_and_guard():
    """Free-reanalysis (2026-06-08): `_compute_matched_rate_gap_514` computes the
    LoRA-vs-FT crossed-cluster-bootstrap gap over the CLEAN anchor set, and returns
    `{}` when an arm has <2 clean anchors. Integration test over the committed
    #514 + #508 eval JSONs (always present in the repo).
    """
    from pathlib import Path

    import pytest

    from explore_persona_space.experiments.full_ft_regime_514.analyze import (
        _compute_matched_rate_gap_514,
    )

    repo = Path(__file__).resolve().parents[1]
    e508 = repo / "eval_results" / "issue_508"
    e514 = repo / "eval_results" / "issue_514"
    if not (e508.exists() and e514.exists()):
        pytest.skip("eval JSONs not present in this checkout")

    full = sorted(e514.glob("*_seed42.json")) + [
        e508 / f"{c}_seed42.json"
        for c in ("lora_b1", "lora_b2", "lora_b3", "ft_b1", "ft_b2", "ft_b3")
    ]
    gap = _compute_matched_rate_gap_514(full)
    # Arm classification: lora_* → LoRA (all 3 clean); ft_* clean → FULLFT
    # (the collapsed/saturated ft_b2 + ft_b3 are dropped by is_clean_anchor).
    assert gap["lora_anchor_cells"] == ["lora_b1", "lora_b2", "lora_b3"]
    assert "ft_b2" not in gap["fullft_anchor_cells"]
    assert "ft_b3" not in gap["fullft_anchor_cells"]
    assert "ft_lowlr_b50" in gap["fullft_anchor_cells"]  # the newly-admitted clean sub-9 cell
    assert gap["n_replicates"] == 1000
    # At the matched 8-nat rate the methods are indistinguishable (CI spans 0).
    assert gap["gap_excludes_zero"] is False

    # Guard: <2 clean anchors in an arm → {} (one LoRA + one FT cell only).
    one_each = [e508 / "lora_b1_seed42.json", e508 / "ft_b1_seed42.json"]
    assert _compute_matched_rate_gap_514(one_each) == {}


def test_local_read_extrapolation_carries_finite_value_and_flag():
    """B10 round-4 pivot: extrapolation emits a FINITE value AND sets the
    is_extrapolation flag AND reports a signed extrapolation_distance.

    Constructs both flavors:
      - target BELOW the lower anchor (negative extrap distance)
      - target ABOVE the upper anchor (positive extrap distance)

    Verifies the determinacy gate is computed against the finite value, not
    against NaN.
    """
    below_pair_a = _diag(
        "ft_a",
        source_mean=9.0,
        held_out_mean=-2.0,
        source_n_probes=20,
        r_collapse_rate=0.0,
        held_out_g_logprob_mean=-7.0,
    )
    below_pair_b = _diag(
        "ft_b",
        source_mean=11.0,
        held_out_mean=-4.0,
        source_n_probes=20,
        r_collapse_rate=0.0,
        held_out_g_logprob_mean=-7.0,
    )
    # Target=8 sits 1 nat BELOW the lower anchor (9.0).
    local_read, is_extrap, extrap_dist, _anchors = _compute_local_matched_rate_read(
        diagnostics_514=[],
        diagnostics_508_ft_anchors=[below_pair_a, below_pair_b],
        target_nat=8.0,
    )
    assert is_extrap is True
    # Signed distance: target - min(xs) = 8.0 - 9.0 = -1.0
    assert math.isclose(extrap_dist, -1.0, abs_tol=1e-9)
    # Linear extrap from (9, -2) → (11, -4): slope=-1, y(8) = -2 + (8-9)*(-1) = -1.
    assert math.isfinite(local_read)
    assert math.isclose(local_read, -1.0, abs_tol=1e-9)

    above_pair_a = _diag(
        "ft_c",
        source_mean=5.0,
        held_out_mean=1.0,
        source_n_probes=20,
        r_collapse_rate=0.0,
        held_out_g_logprob_mean=-7.0,
    )
    above_pair_b = _diag(
        "ft_d",
        source_mean=7.0,
        held_out_mean=-1.0,
        source_n_probes=20,
        r_collapse_rate=0.0,
        held_out_g_logprob_mean=-7.0,
    )
    # Target=8 sits 1 nat ABOVE the upper anchor (7.0).
    local_read2, is_extrap2, extrap_dist2, _anchors2 = _compute_local_matched_rate_read(
        diagnostics_514=[],
        diagnostics_508_ft_anchors=[above_pair_a, above_pair_b],
        target_nat=8.0,
    )
    assert is_extrap2 is True
    # Signed distance: target - max(xs) = 8.0 - 7.0 = +1.0
    assert math.isclose(extrap_dist2, 1.0, abs_tol=1e-9)
    # Linear extrap from (5, 1) → (7, -1): slope=-1, y(8) = 1 + (8-5)*(-1) = -2.
    assert math.isfinite(local_read2)
    assert math.isclose(local_read2, -2.0, abs_tol=1e-9)
