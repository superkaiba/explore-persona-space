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

import numpy as np

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
