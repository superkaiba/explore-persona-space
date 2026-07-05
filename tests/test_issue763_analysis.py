# ruff: noqa: RUF002
"""Tests for the issue #763 analysis helpers (reliability ceiling + GLM LOCO).

Covers the two new ``src/explore_persona_space/analysis/`` modules the #763 fit
script imports — the #742-rebuild reliability ceiling (√(r_yy) split-half +
binomial) and the precision-weighted binomial GLM LOCO predictor — since #742's
modules are not on ``main`` (the rebuild branch). Behavior-focused: a clean
signal must read a HIGH ceiling + a HIGH GLM ρ; a pure-noise target must read
a LOW ceiling + a near-zero GLM ρ; degenerate inputs must return None, never
crash.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from explore_persona_space.analysis.issue_763_glm import glm_predict_loco
from explore_persona_space.analysis.issue_763_reliability import (
    compute_bracket,
    reliability_binomial_variance,
    reliability_split_half_over_probes,
)


def _make_clean_signal(n=50, h=8, seed=0):
    """v0 with a latent direction that linearly drives the rate (high decodability)."""
    rng = np.random.default_rng(seed)
    direction = rng.standard_normal(h)
    x = rng.standard_normal((n, h))
    score = x @ direction
    # logistic map to a rate in (0,1)
    rate = 1.0 / (1.0 + np.exp(-score / 2.0))
    return x, rate, direction


# ── reliability ceiling ───────────────────────────────────────────────────────


def test_split_half_clean_signal_high_ceiling():
    """A context whose per-probe scores are internally consistent reads a HIGH ceiling."""
    rng = np.random.default_rng(1)
    per_probe_by_ctx = {}
    # 30 contexts, each a true rate; per-probe = bernoulli draws around it (m=40)
    for c in range(30):
        true_rate = rng.uniform(0.1, 0.9)
        per_probe_by_ctx[f"ctx{c}"] = list(rng.binomial(1, true_rate, size=40).astype(float))
    out = reliability_split_half_over_probes(per_probe_by_ctx, seed=1)
    assert out["sqrt_r_yy"] is not None
    assert out["sqrt_r_yy"] > 0.5, out


def test_split_half_pure_noise_low_ceiling():
    """All contexts at the SAME rate (no cross-context signal) -> low/None ceiling."""
    rng = np.random.default_rng(2)
    per_probe_by_ctx = {
        f"ctx{c}": list(rng.binomial(1, 0.5, size=40).astype(float)) for c in range(30)
    }
    out = reliability_split_half_over_probes(per_probe_by_ctx, seed=2)
    # no real cross-context signal -> ceiling should be low (Spearman-Brown of ~0)
    assert out["sqrt_r_yy"] is None or out["sqrt_r_yy"] < 0.5, out


def test_split_half_too_few_contexts_returns_none():
    out = reliability_split_half_over_probes({"a": [1.0, 0.0], "b": [0.0, 1.0]}, seed=0)
    assert out["sqrt_r_yy"] is None


def _crossed_design(n_ctx=40, m=30, probe_sd=25.0, ctx_sd=3.0, noise_sd=2.0, seed=7):
    """Fully crossed per-probe scores with probe MAIN effects >> context signal.

    score(c, p) = ctx_effect(c) + probe_effect(p) + noise — the #763 E0 shape
    (same m probes under every context; per-probe difficulty dominates). True
    split-half ceiling ≈ ctx_var / (ctx_var + noise_var/(m/2)) ≈ 0.97 (√ ≈ 0.99).
    """
    rng = np.random.default_rng(seed)
    probe = rng.normal(0.0, probe_sd, m)
    ctx = rng.normal(0.0, ctx_sd, n_ctx)
    mat = ctx[:, None] + probe[None, :] + rng.normal(0.0, noise_sd, (n_ctx, m))
    return {f"c{i}": [float(v) for v in mat[i]] for i in range(n_ctx)}


def test_split_half_crossed_design_independent_clips_aligned_recovers():
    """The reliability-realign fix: on a crossed design with probe main effects
    dominating the context signal, the INDEPENDENT per-context shuffle drives the
    half correlation systematically negative (probe effects leak into the split
    noise as anti-correlated half deviations) and Spearman-Brown clips the ceiling
    to 0, while the probe-ALIGNED split (the default) recovers the true ceiling."""
    per = _crossed_design()
    ind = reliability_split_half_over_probes(per, seed=11, method="independent")
    ali = reliability_split_half_over_probes(per, seed=11)  # aligned is the default
    assert ind["method"] == "independent"
    assert ind["r_hh"] is not None and ind["r_hh"] < 0.0, ind
    assert ind["sqrt_r_yy"] == 0.0, ind  # clipped — the shipped-estimator failure mode
    assert ali["method"] == "aligned"
    assert ali["sqrt_r_yy"] is not None and ali["sqrt_r_yy"] > 0.9, ali


def test_split_half_aligned_tolerates_none_placeholders():
    """None entries (dropped judge draws) are skipped from the half means without
    breaking the positional probe alignment."""
    per = _crossed_design(seed=8)
    per["c0"][3] = None
    per["c5"][0] = None
    per["c5"][17] = None
    out = reliability_split_half_over_probes(per, seed=8)
    assert out["sqrt_r_yy"] is not None and out["sqrt_r_yy"] > 0.9, out


def test_split_half_aligned_ragged_input_raises():
    """The aligned split requires the crossed probe axis — ragged input fails loud."""
    per = _crossed_design(n_ctx=6, m=10, seed=9)
    per["c0"] = per["c0"][:7]  # break the crossed design

    with pytest.raises(ValueError, match="CROSSED"):
        reliability_split_half_over_probes(per, seed=9)


def test_split_half_seed_pinned():
    """Split draws are fully seed-pinned: same seed => identical output."""
    per = _crossed_design(seed=10)
    a = reliability_split_half_over_probes(per, seed=42)
    b = reliability_split_half_over_probes(per, seed=42)
    assert a == b


def test_compute_bracket_threads_method_and_tags_it():
    """compute_bracket threads the estimator through point + bootstrap and tags it."""
    per = _crossed_design(n_ctx=20, m=20, seed=12)
    rates = [float(np.mean(v)) for v in per.values()]
    nj = [20] * len(per)
    ali = compute_bracket(per, rates, nj, n_boot=100, seed=12)
    ind = compute_bracket(per, rates, nj, n_boot=100, seed=12, method="independent")
    assert ali["split_half_method"] == "aligned" and ali["seed"] == 12
    assert ind["split_half_method"] == "independent"
    assert ali["sqrt_r_yy"] is not None and ali["sqrt_r_yy"] > 0.9
    assert ind["sqrt_r_yy"] == 0.0  # clipped on the crossed design
    # the CI must come from the SAME estimator as the point (aligned: a real band)
    assert ali["sqrt_r_yy_ci"] is not None and ali["sqrt_r_yy_ci"][0] > 0.5


def test_binomial_variance_signal_vs_noise():
    """A high-m, high-spread set reads a HIGH ceiling; a low-spread set reads low/None."""
    # high spread, high m -> mostly signal
    rates = [0.05, 0.2, 0.4, 0.6, 0.8, 0.95] * 5
    n_judged = [200] * len(rates)
    out = reliability_binomial_variance(rates, n_judged)
    assert out["sqrt_r_yy"] is not None
    assert out["sqrt_r_yy"] > 0.8, out
    # near-constant rates -> no signal
    flat = [0.5 + 1e-6 * (i % 2) for i in range(20)]
    out2 = reliability_binomial_variance(flat, [50] * 20)
    assert out2["sqrt_r_yy"] is None or out2["sqrt_r_yy"] < 0.3, out2


def test_compute_bracket_reports_both_methods():
    rng = np.random.default_rng(3)
    per_probe_by_ctx = {}
    rates, njudged = [], []
    for c in range(30):
        tr = rng.uniform(0.1, 0.9)
        draws = rng.binomial(1, tr, size=40).astype(float)
        per_probe_by_ctx[f"ctx{c}"] = list(draws)
        rates.append(float(draws.mean()))
        njudged.append(40)
    out = compute_bracket(per_probe_by_ctx, rates, njudged, n_boot=200, seed=3)
    assert "sqrt_r_yy_split_half" in out
    assert "sqrt_r_yy_binomial" in out
    assert out["sqrt_r_yy"] is not None


# ── GLM LOCO ──────────────────────────────────────────────────────────────────


def test_glm_loco_clean_signal_high_rho():
    """A decodable rate target reads a high held-out Spearman of the GLM prediction."""
    from scipy.stats import spearmanr

    x, rate, _ = _make_clean_signal(n=50, h=8, seed=0)
    n_judged = np.full(50, 30)
    out = glm_predict_loco(x, rate, n_judged)
    pred = out["pred"]
    assert pred.shape == (50,)
    rho = spearmanr(pred, rate).correlation
    assert rho > 0.4, rho  # the latent direction is recoverable under LOCO


def test_glm_loco_pure_noise_no_systematic_positive_signal():
    """On pure noise the GLM LOCO ρ shows NO systematic POSITIVE (decodable) signal.

    A held-out LOCO predictor with a fitted intercept is NEGATIVELY biased on
    noise (the held-out point is excluded from the train mean, so a high-y
    point is predicted toward the lower train-mean → a negative rank
    correlation): across 8 independent noise targets the mean held-out ρ here
    is ≈ −0.25, NOT ~0. This is the WHY the headline rests on the shuffle-label
    null (which refits the SAME procedure per permutation and inherits the same
    negative bias, so the observed-vs-null comparison stays calibrated), NOT on
    the raw point ρ. The honest invariant the predictor must satisfy is
    one-sided: noise never manufactures a systematic POSITIVE ρ (the direction
    that would falsely read as decodability). The per-seed |ρ| stays bounded by
    the p≪n dim cap (d ≤ n_train//5).
    """
    from scipy.stats import spearmanr

    rhos = []
    for seed in range(8):
        rng = np.random.default_rng(100 + seed)
        x = rng.standard_normal((50, 8))
        rate = rng.uniform(0.1, 0.9, size=50)  # independent of x
        out = glm_predict_loco(x, rate, np.full(50, 30))
        r = spearmanr(out["pred"], rate).correlation
        if not np.isnan(r):
            rhos.append(r)
    rhos = np.asarray(rhos)
    # one-sided: noise must NOT produce a systematic positive (false-decodable) signal
    assert float(rhos.mean()) < 0.10, rhos.mean()
    # the LOO negative bias is expected + bounded (not an unbounded overfit blowup)
    assert float(rhos.min()) > -1.0 and float(np.abs(rhos).mean()) < 0.6, rhos


def test_glm_loco_reports_overdispersion_flag():
    x, rate, _ = _make_clean_signal(n=40, h=6, seed=5)
    out = glm_predict_loco(x, rate, np.full(40, 20))
    assert "overdispersion" in out
    assert isinstance(out["quasibinomial"], bool)
    assert len(out["chosen_dims"]) == 40


def test_glm_loco_smoke_tiny_slice_runs():
    """The smoke slice (n=3) must run without crashing (the §10 smoke path)."""
    rng = np.random.default_rng(9)
    x = rng.standard_normal((3, 4))
    rate = np.array([0.2, 0.5, 0.8])
    out = glm_predict_loco(x, rate, np.full(3, 5))
    assert out["pred"].shape == (3,)
    assert np.all(np.isfinite(out["pred"]))


# ── refresh-mode guardrails (deception-rubric-reanchor guard-fix round 2) ─────
#
# Offline pins for the two round-2 guard fixes on scripts/issue763_fit_predictors.py:
# 1. CONCERN reliability-refresh-default-outdir-parent-writable — BOTH refresh
#    modes require an explicit --out-dir and refuse the PARENT artifact root
#    (a flagless rerun would patch eval_results/issue_763's committed
#    fit_by_behavior/*.json + matched_predictor_results.json in place).
# 2. Mode-order revert hazard — --refresh-metadata-only re-run AFTER
#    --reliability-refresh-only preserves the round record's ALIGNED ceiling
#    fields (split_half_probe_aligned_v2) instead of reverting them to the
#    parent's legacy values.

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO / "scripts") not in sys.path:
    sys.path.insert(0, str(_REPO / "scripts"))


def _fit_module():
    """Import the fit script lazily (heavy: torch/statsmodels) so the pure-numpy
    tests above stay unaffected at collection time."""
    import issue763_fit_predictors as F

    return F


def _write(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj))


_PARENT_REC = {
    "behavior": "deception",
    "n_contexts": 2,
    "m": 60,
    "m_e0": 60,
    "reduced_power": False,
    "yield_floor": 48,
    "any_shortfall": False,
    "n_shortfall_cells": 0,
    "median_n_judged": 50.0,
    "rubric_version": "v1",
    "rho_graded_ridge": 0.5,
    "sqrt_r_yy_graded": 0.2,
    "sqrt_r_yy_graded_ci": [0.1, 0.3],
    "sqrt_r_yy": 0.2,
    "sqrt_r_yy_ci": [0.1, 0.3],
    "sqrt_r_yy_binomial": 0.9,
    "sqrt_r_yy_binary": 0.15,
    "triage_verdict": "noise_limited",
    "triage_verdict_binary": "noise_limited",
}

# The fields --reliability-refresh-only patched onto the round checkpoint (the
# aligned re-emit); a later metadata re-run must carry ALL of them verbatim.
_ALIGNED_FIELDS = {
    "sqrt_r_yy_graded": 0.55,
    "sqrt_r_yy_graded_ci": [0.4, 0.7],
    "sqrt_r_yy": 0.55,
    "sqrt_r_yy_ci": [0.4, 0.7],
    "sqrt_r_yy_binomial": 0.9,
    "sqrt_r_yy_binary": 0.5,
    "triage_verdict": "works",
    "triage_verdict_binary": "works",
    "reliability_estimator": "split_half_probe_aligned_v2",
    "reliability_e0_source": "round/E0_deception_v2.json",
    "sqrt_r_yy_graded_legacy_independent": 0.2,
    "sqrt_r_yy_graded_ci_legacy_independent": [0.1, 0.3],
    "sqrt_r_yy_binary_legacy_independent": 0.15,
    "triage_verdict_legacy_independent": "noise_limited",
    "triage_verdict_binary_legacy_independent": "noise_limited",
    "reliability_bracket_graded": {"sqrt_r_yy": 0.55, "method": "aligned"},
    "reliability_bracket_binary": {"sqrt_r_yy": 0.5, "method": "aligned"},
}

_E0 = {
    "yield_flags": {"deception": {"m_B": 60}},
    "judge_diagnostics": {"deception": {"r_jj": 0.8, "graded_binary_tracking_spearman": 0.7}},
    "e0": {"deception": {"c1": {"n_judged": 50}, "c2": {"n_judged": 50}}},
}


@pytest.mark.parametrize("flag", ["--reliability-refresh-only", "--refresh-metadata-only"])
def test_refresh_modes_require_explicit_out_dir(monkeypatch, tmp_path, flag):
    """Item 1 leg A: a FLAGLESS refresh rerun (no --out-dir) fails loud before any write."""
    F = _fit_module()
    monkeypatch.setattr(F, "EVAL_RESULTS_DIR", tmp_path / "eval_results")
    monkeypatch.setattr(
        sys, "argv", ["issue763_fit_predictors.py", "--behaviors", "deception", flag]
    )
    with pytest.raises(SystemExit, match="explicit --out-dir"):
        F.main()


@pytest.mark.parametrize("flag", ["--reliability-refresh-only", "--refresh-metadata-only"])
def test_refresh_out_dir_must_not_be_parent_root(monkeypatch, tmp_path, flag):
    """Item 1 leg B: an --out-dir resolving to the PARENT artifact root is refused."""
    F = _fit_module()
    parent_root = tmp_path / "eval_results"
    parent_root.mkdir()
    monkeypatch.setattr(F, "EVAL_RESULTS_DIR", parent_root)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "issue763_fit_predictors.py",
            "--behaviors",
            "deception",
            flag,
            "--out-dir",
            str(parent_root),
        ],
    )
    with pytest.raises(SystemExit, match="PARENT artifact root"):
        F.main()


def test_metadata_refresh_preserves_aligned_reliability_fields(monkeypatch, tmp_path):
    """Item 2: a metadata re-run AFTER the reliability refresh must NOT revert the
    round checkpoint's aligned ceiling fields to the parent's legacy values."""
    F = _fit_module()
    monkeypatch.setattr(F, "EVAL_RESULTS_DIR", tmp_path / "eval_results")
    monkeypatch.setattr(F, "load_frozen_pool_staged", lambda b: {"n_probes": 60})
    parent_dir = tmp_path / "parent" / "fit_by_behavior"
    out_dir = tmp_path / "round"
    e0_path = tmp_path / "E0_parent.json"
    _write(e0_path, _E0)
    _write(parent_dir / "deception.json", _PARENT_REC)
    round_rec = {**_PARENT_REC, "rubric_version": "v1-metadata-refresh", **_ALIGNED_FIELDS}
    _write(out_dir / "fit_by_behavior" / "deception.json", round_rec)

    ns = SimpleNamespace(behaviors=["deception"], parent_fit_dir=parent_dir)
    rc = F._refresh_metadata_only(ns, out_dir, e0_path)
    assert rc == 0

    rec = json.loads((out_dir / "fit_by_behavior" / "deception.json").read_text())
    # every aligned ceiling / verdict / provenance field survives verbatim
    for k, v in _ALIGNED_FIELDS.items():
        assert rec[k] == v, f"aligned field reverted by metadata refresh: {k}"
    # the metadata fields were still refreshed
    assert rec["rubric_version"] == "v1-metadata-refresh"
    assert rec["m"] == 60
    # the assembled output ships the preserved record
    blob = json.loads((out_dir / "matched_predictor_results.json").read_text())
    assert blob["by_behavior"]["deception"]["sqrt_r_yy"] == 0.55
    assert (
        blob["by_behavior"]["deception"]["reliability_estimator"] == "split_half_probe_aligned_v2"
    )


def test_metadata_refresh_without_aligned_round_checkpoint_unchanged(monkeypatch, tmp_path):
    """Guard scoping: with NO aligned round checkpoint the refresh behaves as before
    (plain parent + metadata patch; no reliability keys invented)."""
    F = _fit_module()
    monkeypatch.setattr(F, "EVAL_RESULTS_DIR", tmp_path / "eval_results")
    monkeypatch.setattr(F, "load_frozen_pool_staged", lambda b: {"n_probes": 60})
    parent_dir = tmp_path / "parent" / "fit_by_behavior"
    out_dir = tmp_path / "round"
    e0_path = tmp_path / "E0_parent.json"
    _write(e0_path, _E0)
    _write(parent_dir / "deception.json", _PARENT_REC)

    ns = SimpleNamespace(behaviors=["deception"], parent_fit_dir=parent_dir)
    rc = F._refresh_metadata_only(ns, out_dir, e0_path)
    assert rc == 0

    rec = json.loads((out_dir / "fit_by_behavior" / "deception.json").read_text())
    assert "reliability_estimator" not in rec
    assert rec["sqrt_r_yy"] == _PARENT_REC["sqrt_r_yy"]
    assert rec["triage_verdict"] == _PARENT_REC["triage_verdict"]
