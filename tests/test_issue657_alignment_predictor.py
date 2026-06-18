"""Issue #657 — unit guards for the alignment-predictor analysis library.

These pin the load-bearing invariants of the cross-behavior leakage bake-off so a
future refactor can't silently strip them:

  - the marker-token fail-fast in the #605 substrate loader (wrong marker_id raises);
  - the doubly-partialled partial-Spearman actually removes the partialled covariate
    (a covariate-driven correlation collapses under the double partial);
  - the prior_centroid_projection geometry-side control (M-Alts1) is computed from
    the base-rate-weighted persona-vector centroid;
  - bootstrap_over_personas resamples the BYSTANDER persona, not the cell;
  - the reliability band (M-Stats1) produces the 3-point R_low/R_expected/R_high
    sensitivity points;
  - the reliability band's EIV disattenuation actually MOVES the partial-rho across
    reliability points (the round-1 rank-invariant no-op regression) AND carries
    per-band-point CIs for partial-rho AND ΔR² (B1 + B2 round-2 blockers);
  - the H3 verdict is NOT a PASS when a band point's CI straddles 0 (B2);
  - PRIMARY_H3_BEHAVIORS is a literal frozenset excluding EM (N1);
  - the additive --trait-name CLI default reproduces #623's sycophancy verbatim.

All synthetic / CPU-only — no GPU, no HF, no model load.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from explore_persona_space.analysis import issue657_alignment_predictor as m


def _toy_persona_vectors(n_layers: int = 4, hidden: int = 8, seed: int = 0):
    """A tiny persona-vector bank: assistant (baseline) + 5 personas, (n_layers, hidden)."""
    rng = np.random.default_rng(seed)
    names = ["assistant", "p1", "p2", "p3", "p4", "p5"]
    return {
        name: torch.from_numpy(rng.standard_normal((n_layers, hidden)).astype(np.float32))
        for name in names
    }


# ── partial_spearman: the double partial removes the partialled covariate ──


def test_partial_spearman_removes_covariate():
    rng = np.random.default_rng(1)
    n = 60
    z = rng.standard_normal(n)
    # x and y are BOTH driven only by z (plus independent noise) -> raw rho strong,
    # partial rho ~ 0 once z is partialled out.
    x = z + 0.1 * rng.standard_normal(n)
    y = z + 0.1 * rng.standard_normal(n)
    raw_rho, _p, _n = m.partial_spearman(x, y, None)
    part_rho, _p2, _n2 = m.partial_spearman(x, y, z)
    assert raw_rho > 0.8, raw_rho
    assert abs(part_rho) < 0.3, part_rho


def test_partial_spearman_doubly_partialled_matrix_z():
    """The matrix-z (doubly-partialled) read removes BOTH covariate columns.

    x, y share a single driver z1; z2 is an extra (irrelevant) covariate column. The
    matrix partial on [z1, z2] must still kill the z1-driven correlation — i.e. adding
    a noise column to the partial set does not resurrect the signal. (A test where x,y
    depend on the SUM z1+z2 is NOT a valid expectation: rank-residualizing on the two
    rank columns linearly cannot remove a nonlinear-in-ranks sum dependence — that is
    a property of rank partial correlation, not a code bug.)
    """
    rng = np.random.default_rng(2)
    n = 80
    z1 = rng.standard_normal(n)
    z2 = rng.standard_normal(n)
    x = z1 + 0.1 * rng.standard_normal(n)
    y = z1 + 0.1 * rng.standard_normal(n)
    raw_rho, _, _ = m.partial_spearman(x, y, None)
    double, _, _ = m.partial_spearman(x, y, np.column_stack([z1, z2]))
    assert raw_rho > 0.8
    assert abs(double) < 0.3, double


# ── prior_centroid_projection: M-Alts1 base-rate-weighted centroid ──


def test_prior_centroid_projection_weights_by_base_rate():
    pv = _toy_persona_vectors()
    personas = ["assistant", "p1", "p2", "p3"]
    # heavy weight on p1: the centroid points toward p1's persona vector, so p1's
    # projection should be the highest among the non-baseline personas.
    base_rates = {"p1": 0.9, "p2": 0.05, "p3": 0.05, "assistant": 0.01}
    pcp = m.prior_centroid_projection(pv, base_rates, personas, layer=14)
    assert set(pcp) == set(personas)
    non_baseline = {k: v for k, v in pcp.items() if k != "assistant"}
    top = max(non_baseline, key=non_baseline.get)
    assert top == "p1", (top, non_baseline)


def test_compute_alignment_is_cosine_bounded():
    pv = _toy_persona_vectors()
    personas = ["assistant", "p1", "p2"]
    bdir = torch.from_numpy(np.ones(8, dtype=np.float32))
    align = m.compute_alignment(pv, bdir, personas, layer=14)
    # assistant is the baseline-self -> its persona vector is 0 -> NaN cosine.
    assert np.isnan(align["assistant"])
    for p in ("p1", "p2"):
        assert -1.0001 <= align[p] <= 1.0001, (p, align[p])


# ── bootstrap_over_personas: resamples the BYSTANDER, not the cell ──


def test_bootstrap_resamples_bystanders_not_cells():
    # 3 bystanders x 2 sources each = 6 cells. The stat counts DISTINCT bystanders
    # in the resampled frame; with bystander resampling it must be <= 3 (never 6).
    cells = []
    for b in ("b1", "b2", "b3"):
        for s in ("s1", "s2"):
            cells.append({"source": s, "bystander": b, "delta": 0.1, "align": 0.2})
    frame = {
        "cells": cells,
        "resolvable_bystanders": ["b1", "b2", "b3"],
        "n_resolvable": 3,
        "dropped": [],
    }

    def _distinct_bystanders(fr):
        return float(len({c["bystander"] for c in fr["cells"]}))

    out = m.bootstrap_over_personas(frame, _distinct_bystanders, b=50, seed=657)
    # the resampled frame never has more than 3 distinct bystanders
    assert out["ci_hi"] <= 3.0, out
    assert out["n_bystanders"] == 3


# ── reliability band: 3-point sensitivity (M-Stats1) ──


def test_attenuation_band_three_points():
    rng = np.random.default_rng(3)
    cells = []
    bystanders = [f"b{i}" for i in range(20)]
    base_rates = {b: float(rng.uniform(0.02, 0.13)) for b in bystanders}
    for b in bystanders:
        for s in ("s1", "s2"):
            cells.append(
                {
                    "source": s,
                    "bystander": b,
                    "delta": float(rng.uniform(-0.1, 0.7)),
                    "align": float(rng.uniform(-0.3, 0.3)),
                    "bystander_base_rate": base_rates[b],
                    "prior_centroid_projection": float(rng.uniform(-0.3, 0.3)),
                }
            )
    frame = {
        "cells": cells,
        "resolvable_bystanders": bystanders,
        "n_resolvable": len(bystanders),
        "dropped": [],
    }
    band = m.attenuation_sensitivity(frame, base_rates, behavior="refusal")
    assert band["band_applicable"] is True
    for label in ("R_observed", "R_low", "R_expected", "R_high"):
        assert label in band["points"], label
    # R_low (largest n, prior cleanest) has the LOWEST reliability-correction scale;
    # all reliabilities are in (0, 1].
    for label in ("R_low", "R_expected", "R_high"):
        rel = band["points"][label]["reliability"]
        assert 0.0 < rel <= 1.0, (label, rel)


def test_attenuation_band_skipped_for_marker():
    """Marker base rate is a log-prob, not a 0-1 rate -> binomial reliability undefined."""
    cells = [
        {
            "source": "s1",
            "bystander": f"b{i}",
            "delta": 0.1,
            "align": 0.2,
            "bystander_base_rate": -18.0,  # a log-prob, not a rate
            "prior_centroid_projection": 0.1,
        }
        for i in range(10)
    ]
    frame = {
        "cells": cells,
        "resolvable_bystanders": [f"b{i}" for i in range(10)],
        "n_resolvable": 10,
        "dropped": [],
    }
    base_rates = {f"b{i}": -18.0 for i in range(10)}
    band = m.attenuation_sensitivity(frame, base_rates, behavior="marker")
    assert band["band_applicable"] is False


# ── M-Stats1 EIV disattenuation: the band ACTUALLY MOVES the partial (B1) ──


def _eiv_band_fixture(seed: int = 42, n_bystanders: int = 30):
    """A frame where ``align`` is a CLEANER proxy of the latent leakage driver Z than
    the NOISY ``bystander_base_rate`` — exactly the regime M-Stats1 must disattenuate.

    Returns (frame, base_rates).
    """
    rng = np.random.default_rng(seed)
    bystanders = [f"b{i}" for i in range(n_bystanders)]
    z_latent = {b: float(rng.standard_normal()) for b in bystanders}
    base_rates: dict[str, float] = {}
    align_b: dict[str, float] = {}
    for b in bystanders:
        z = z_latent[b]
        # base_rate: a NOISY proxy of Z, kept in (0, 1) (a 0-1 rate so the binomial
        # reliability is defined).
        base_rates[b] = float(np.clip(0.10 + 0.05 * z + 0.04 * rng.standard_normal(), 0.01, 0.30))
        align_b[b] = float(z + 0.10 * rng.standard_normal())  # CLEAN proxy of Z
    cells = []
    for b in bystanders:
        for s in ("s1", "s2"):
            delta = 0.4 * z_latent[b] + (0.1 if s == "s1" else -0.1) + 0.1 * rng.standard_normal()
            cells.append(
                {
                    "source": s,
                    "bystander": b,
                    "delta": float(delta),
                    "align": align_b[b],
                    "bystander_base_rate": base_rates[b],
                    "prior_centroid_projection": float(
                        0.2 * z_latent[b] + 0.3 * rng.standard_normal()
                    ),
                }
            )
    frame = {
        "cells": cells,
        "resolvable_bystanders": bystanders,
        "n_resolvable": len(bystanders),
        "dropped": [],
    }
    return frame, base_rates


def test_attenuation_band_eiv_moves_partial_not_a_noop():
    """B1 regression: the reliability band must produce DISTINCT doubly-partialled rho
    across R_observed/R_low/R_expected/R_high, and the disattenuated low-reliability
    band point must move AWAY from the observed-prior point in the expected direction.

    The round-1 code did ``prior * sqrt(rel)`` then rank-residualized, which is
    rank-invariant — all four points were byte-identical to R_observed. This asserts
    the no-op is gone.
    """
    frame, base_rates = _eiv_band_fixture()
    band = m.attenuation_sensitivity(frame, base_rates, behavior="refusal", b=1, seed=7)
    assert band["band_applicable"] is True

    labels = ("R_observed", "R_low", "R_expected", "R_high")
    rhos = [band["points"][k]["doubly_partialled_rho"] for k in labels]
    for lbl, r in zip(labels, rhos, strict=True):
        assert not np.isnan(r), (lbl, r)
    # (i) the band points are NOT byte-identical (the no-op signature).
    distinct = {round(r, 9) for r in rhos}
    assert len(distinct) > 1, f"band collapsed to a single value (no-op regression): {rhos}"
    # (ii) the disattenuated low-reliability band point moves AWAY from observed-prior
    # in the expected direction (more of the noisy prior removed -> partial decreases).
    rho_obs = band["points"]["R_observed"]["doubly_partialled_rho"]
    rho_low = band["points"]["R_low"]["doubly_partialled_rho"]
    assert rho_low < rho_obs, (rho_low, rho_obs)
    # (iii) reliabilities are ordered observed >= low >= expected >= high.
    rels = [band["points"][k]["reliability"] for k in labels]
    assert all(rels[i] >= rels[i + 1] - 1e-9 for i in range(len(rels) - 1)), rels


def test_attenuation_band_carries_per_point_cis_for_rho_and_delta_r2():
    """B2 regression: every band point must carry a bootstrap CI for BOTH the
    doubly-partialled rho AND ΔR²_align-beyond-prior (the registered §7 gate reads
    the per-band CI lower bounds, not point-estimate signs)."""
    frame, base_rates = _eiv_band_fixture()
    band = m.attenuation_sensitivity(frame, base_rates, behavior="refusal", b=300, seed=7)
    assert band["ci_computed"] is True
    for label in ("R_observed", "R_low", "R_expected", "R_high"):
        pt = band["points"][label]
        assert "doubly_partialled_rho" in pt
        assert "delta_r2_align_beyond_prior" in pt
        for ci_key in ("doubly_partialled_rho_ci", "delta_r2_align_beyond_prior_ci"):
            assert ci_key in pt, (label, ci_key)
            ci = pt[ci_key]
            assert "ci_lo" in ci and "ci_hi" in ci, (label, ci_key)


def test_attenuation_band_smoke_b1_has_no_ci():
    """Smoke (b<=1) reports point estimates only, no per-point CI (matches the smoke
    fast path)."""
    frame, base_rates = _eiv_band_fixture(n_bystanders=12)
    band = m.attenuation_sensitivity(frame, base_rates, behavior="refusal", b=1, seed=7)
    assert band["ci_computed"] is False
    assert "doubly_partialled_rho_ci" not in band["points"]["R_observed"]


# ── marker-token fail-fast in the #605 substrate loader ──


def test_marker_substrate_rejects_wrong_marker_id(tmp_path: Path):
    marker_root = tmp_path / "issue_605" / "marker"
    (marker_root / "per_cell_trained").mkdir(parents=True)
    (marker_root / "per_cell_base").mkdir(parents=True)
    # one cell carrying a WRONG marker_id (bare ※ id 63680) -> must raise.
    cell = {
        "context_label": "m605_nt_comedian_1__none",
        "source_cid": "A1",
        "summary": {"mean_logp_marker": -5.0},
        "metadata": {"marker_id": 63680},
    }
    base = {"summary": {"mean_logp_marker": -18.0}, "metadata": {"marker_id": 63680}}
    name = "A1__m605_nt_comedian_1__none.json"
    (marker_root / "per_cell_trained" / name).write_text(json.dumps(cell))
    (marker_root / "per_cell_base" / name).write_text(json.dumps(base))
    with pytest.raises(ValueError, match="marker_id"):
        m.load_marker_cell_substrate(tmp_path)


def test_marker_persona_alias_mapping():
    # swe -> software_engineer; non-panel personas (formal/socratic/pirate) -> None.
    assert m._marker_persona_from_context("m605_nt_swe_1__none") == "software_engineer"
    assert m._marker_persona_from_context("m605_nt_comedian_2__explicit") == "comedian"
    assert m._marker_persona_from_context("m605_nt_formal_1__soft") is None
    assert m._marker_persona_from_context("instr_explicit_1") is None


# ── additive --trait-name default reproduces #623 sycophancy verbatim ──


def test_extract_script_trait_default_is_623_sycophancy():
    spec = importlib.util.spec_from_file_location(
        "issue623_extract_657test",
        Path(__file__).resolve().parents[1] / "scripts/issue623_extract_sycophancy_vector.py",
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    import inspect

    sig = inspect.signature(mod.generate_artifacts)
    from explore_persona_space.experiments.persona_decomp_623 import (
        SYCOPHANCY_TRAIT_DESCRIPTION,
        SYCOPHANCY_TRAIT_NAME,
    )

    assert sig.parameters["trait_name"].default == SYCOPHANCY_TRAIT_NAME
    assert sig.parameters["trait_description"].default == SYCOPHANCY_TRAIT_DESCRIPTION


# ── PRIMARY_H3_BEHAVIORS is a literal frozenset excluding EM (N1) ──


def test_primary_h3_behaviors_is_literal_frozenset_excluding_em():
    assert isinstance(m.PRIMARY_H3_BEHAVIORS, frozenset)
    assert frozenset({"sycophancy", "refusal", "marker"}) == m.PRIMARY_H3_BEHAVIORS
    assert "em" not in m.PRIMARY_H3_BEHAVIORS


# ── bake-off consumer: H3 gate + band verdict + syc-i fail-loud (B2 + N2) ──


def _load_bake_off_module():
    spec = importlib.util.spec_from_file_location(
        "issue657_run_bake_off_test",
        Path(__file__).resolve().parents[1] / "scripts/issue657_run_bake_off.py",
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _band_with_cis(point_cis: dict[str, tuple[float, float]], ci_computed: bool = True) -> dict:
    """Build a minimal band dict where each label maps to (rho_ci_lo, r2_ci_lo).

    Both CIs at a point get the same lower bound for brevity (the gate ANDs them).
    """
    points: dict[str, dict] = {}
    for label, (rho_lo, r2_lo) in point_cis.items():
        points[label] = {
            "reliability": 1.0 if label == "R_observed" else 0.5,
            "doubly_partialled_rho": 0.3,
            "delta_r2_align_beyond_prior": 0.1,
            "doubly_partialled_rho_ci": {"ci_lo": rho_lo, "ci_hi": rho_lo + 0.4},
            "delta_r2_align_beyond_prior_ci": {"ci_lo": r2_lo, "ci_hi": r2_lo + 0.2},
        }
    return {"band_applicable": True, "ci_computed": ci_computed, "points": points}


def test_band_clears_confirmed_when_all_points_clear():
    bake = _load_bake_off_module()
    band = _band_with_cis(
        {
            "R_observed": (0.20, 0.05),
            "R_low": (0.10, 0.02),
            "R_expected": (0.08, 0.01),
            "R_high": (0.05, 0.01),
        }
    )
    assert bake._band_clears(band)["verdict"] == "CONFIRMED"


def test_band_clears_indeterminate_when_only_optimistic_end_clears():
    bake = _load_bake_off_module()
    # R_low (most attenuated) straddles 0 for rho; only R_high clears both CIs.
    band = _band_with_cis(
        {
            "R_observed": (0.20, 0.05),
            "R_low": (-0.05, 0.02),
            "R_expected": (-0.02, 0.01),
            "R_high": (0.06, 0.01),
        }
    )
    assert bake._band_clears(band)["verdict"] == "INDETERMINATE"


def test_band_clears_requires_both_rho_and_delta_r2():
    bake = _load_bake_off_module()
    # rho CIs all clear, but ΔR² CI straddles 0 at R_low -> NOT confirmed.
    band = _band_with_cis(
        {
            "R_observed": (0.20, 0.05),
            "R_low": (0.10, -0.01),
            "R_expected": (0.08, 0.02),
            "R_high": (0.05, 0.01),
        }
    )
    assert bake._band_clears(band)["verdict"] != "CONFIRMED"


def test_h3_summary_not_pass_when_a_band_point_straddles_zero():
    """B2 regression: the observed-prior partial-rho clears its headline CI, but one
    reliability-band point's CI straddles 0 -> the H3 verdict must NOT be PASS."""
    bake = _load_bake_off_module()
    # band: optimistic end clears, R_low does not -> INDETERMINATE band verdict.
    band = _band_with_cis(
        {
            "R_observed": (0.20, 0.05),
            "R_low": (-0.05, 0.02),  # straddles 0
            "R_expected": (0.04, 0.01),
            "R_high": (0.06, 0.01),
        }
    )
    band_verdict = bake._band_clears(band)
    assert band_verdict["verdict"] == "INDETERMINATE"
    per_behavior = {
        "sycophancy": {
            "leakage_bake_off": {
                "in_scope": True,
                # headline R_observed CIs DO clear (the no-correction read is positive)
                "h3_doubly_partialled_rho_ci": {"ci_lo": 0.20, "ci_hi": 0.55},
                "h3_delta_r2_ci": {"ci_lo": 0.05, "ci_hi": 0.30},
                "band_verdict": band_verdict,
            }
        }
    }
    summary = bake._h3_summary(per_behavior)
    assert summary["h3_verdict"] != "PASS", summary
    assert summary["h3_verdict"] == "INDETERMINATE", summary
    assert summary["per_behavior_detail"]["sycophancy"]["passes_h3"] is False


def test_h3_summary_pass_only_when_band_confirmed():
    bake = _load_bake_off_module()
    band = _band_with_cis(
        {
            "R_observed": (0.20, 0.05),
            "R_low": (0.10, 0.02),
            "R_expected": (0.08, 0.01),
            "R_high": (0.05, 0.01),
        }
    )
    band_verdict = bake._band_clears(band)
    assert band_verdict["verdict"] == "CONFIRMED"
    per_behavior = {
        "refusal": {
            "leakage_bake_off": {
                "in_scope": True,
                "h3_doubly_partialled_rho_ci": {"ci_lo": 0.20, "ci_hi": 0.55},
                "h3_delta_r2_ci": {"ci_lo": 0.05, "ci_hi": 0.30},
                "band_verdict": band_verdict,
            }
        }
    }
    summary = bake._h3_summary(per_behavior)
    assert summary["h3_verdict"] == "PASS", summary
    assert "refusal" in summary["primary_behaviors_cleared"]


def test_syc_i_fallback_halts_in_production(tmp_path: Path):
    """N2: a missing #623 syc_i.json HALTS a non-smoke production sycophancy run
    (exit non-zero) unless --allow-syc-i-fallback is passed."""
    bake = _load_bake_off_module()
    missing = tmp_path / "nope" / "syc_i.json"
    # production, no opt-in -> SystemExit
    with pytest.raises(SystemExit):
        bake._load_da_base_rates("sycophancy", missing, smoke=False, allow_syc_i_fallback=False)
    # smoke -> warn-and-fall-back (returns None, no raise)
    assert bake._load_da_base_rates("sycophancy", missing, smoke=True) is None
    # explicit opt-in -> warn-and-fall-back
    assert (
        bake._load_da_base_rates("sycophancy", missing, smoke=False, allow_syc_i_fallback=True)
        is None
    )
    # non-sycophancy behaviors are unaffected (always None, never raise)
    assert bake._load_da_base_rates("refusal", missing, smoke=False) is None
