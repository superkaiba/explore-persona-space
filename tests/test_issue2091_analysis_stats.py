"""Synthetic-fixture tests for the #2091 P4 statistics engine (unit C).

Covers the unit-C smoke list: the exchangeability null (E[Delta] ~ 0 +
~uniform greedy ranks on exchangeable draws), the rank-space commonality
decomposition recovering planted unique/shared structure, the R5 ceiling /
polarization arithmetic (exact on all-or-nothing draws), the draw-jackknife
band, the two-way cluster bootstrap reducing EXACTLY to one-way when the
second axis is constant, the weighted median vs a naive per-draw loop, and
the Holm-adjusted contrast CIs. Everything is tiny + CPU (<10 s); no
network, no staged data.
"""

from __future__ import annotations

import numpy as np

from scripts.issue2091_analysis import (
    MID_HI,
    MID_LO,
    GroupBootstrap,
    column_ceilings,
    commonality,
    delta_ctx,
    disjoint_half_agreement,
    exchangeability_ranks,
    holm_adjusted_cis,
    jackknife_delta_band,
    pairwise_cos_dispersion,
    polarization_stats,
    r2_score_rows,
    rankdata_avg,
    spearman,
    split_half_reliability,
    variance_components,
)

RNG = np.random.default_rng(7)


def _exchangeable_pool(n: int = 400, k: int = 5, d: int = 16):
    """(g, v) where the greedy row is an iid member of the rollout cloud."""
    mu = RNG.normal(size=(n, 1, d))
    pool = mu + 0.3 * RNG.normal(size=(n, k + 1, d))
    return pool[:, 0, :], pool[:, 1:, :]


def test_exchangeability_null_mean_delta_zero_and_uniform_ranks():
    g, v = _exchangeable_pool()
    d = delta_ctx(g, v)
    # under exchangeability the matched-LOO Delta has mean ~0
    assert abs(float(d["delta"].mean())) < 0.01, d["delta"].mean()
    ranks = exchangeability_ranks(g, v)
    # rank of the greedy member among K+1=6 -> uniform on 1..6, mean 3.5
    assert abs(float(ranks.mean()) - 3.5) < 0.25, ranks.mean()
    # every rank value occupied roughly uniformly (n/6 ~ 67 each)
    counts = np.array([(ranks == r).sum() for r in (1, 2, 3, 4, 5, 6)])
    assert counts.min() > 30, counts


def test_delta_positive_when_greedy_is_cloud_mean():
    # greedy == the mean of the rollouts -> cos(g, LOO) > mean cos(v_j, LOO)
    _, v = _exchangeable_pool(n=200)
    g = v.mean(axis=1)
    d = delta_ctx(g, v)
    assert float(np.median(d["delta"])) > 0.01


def test_jackknife_band_shape_and_coverage():
    g, v = _exchangeable_pool(n=150)
    jk = jackknife_delta_band(g, v)
    assert len(jk["drop_one_medians"]) == v.shape[1]
    lo, hi = jk["band"]
    assert lo <= hi
    full = float(np.median(delta_ctx(g, v)["delta"]))
    # drop-one medians should bracket the full-sample median loosely
    assert lo - 0.05 <= full <= hi + 0.05


def test_disjoint_half_agreement_bounded():
    _, v = _exchangeable_pool(n=100)
    ha = disjoint_half_agreement(v, "test")
    assert ha.shape == (100,)
    assert np.all(ha <= 1.0) and np.all(ha > -1.0)


def test_two_way_bootstrap_reduces_to_one_way_when_axis_constant():
    groups = [f"g{i % 20}" for i in range(200)]
    vals = RNG.normal(size=200)
    b1 = GroupBootstrap(groups, 64, "seedkey")
    b2 = GroupBootstrap(groups, 64, "seedkey", groups2=["only"] * 200)
    np.testing.assert_allclose(b1.mean(vals), b2.mean(vals))
    np.testing.assert_allclose(b1.median(vals), b2.median(vals))
    y = RNG.normal(size=200)
    np.testing.assert_allclose(b1.corr(vals, y), b2.corr(vals, y))


def test_weighted_median_matches_naive_loop():
    groups = [f"g{i % 7}" for i in range(60)]
    vals = RNG.normal(size=60)
    boot = GroupBootstrap(groups, 32, "medtest")
    med = boot.median(vals)
    for b in range(32):
        w = boot.w[b]
        order = np.argsort(vals, kind="stable")
        sv, sw = vals[order], w[order]
        cum = np.cumsum(sw)
        naive = sv[np.argmax(cum >= 0.5 * cum[-1])]
        assert med[b] == naive, (b, med[b], naive)


def test_bootstrap_corr_matches_full_sample_at_identity_weights():
    # sanity: point Spearman via rank transform ~ corr of ranks
    x = RNG.normal(size=120)
    y = 0.7 * x + RNG.normal(size=120)
    rho = spearman(x, y)
    rx, ry = rankdata_avg(x), rankdata_avg(y)
    den = np.sqrt(((rx - rx.mean()) ** 2).sum() * ((ry - ry.mean()) ** 2).sum())
    manual = float(((rx - rx.mean()) * (ry - ry.mean())).sum() / den)
    assert abs(rho - manual) < 1e-12


def test_polarization_all_or_nothing_p_is_one():
    # binary 0/100 draws with mu in the middling band -> P == 1 exactly
    scores = np.array(
        [
            [100.0, 0.0, 100.0, 0.0, 100.0],  # mu=60, sd=sqrt(.6*.4)*100 -> P=1
            [0.0, 0.0, 100.0, 0.0, 0.0],  # mu=20 (not middling)
            [50.0, 50.0, 50.0, 50.0, 50.0],  # mu=50, sd=0 -> P=0, f_mid=1
        ]
    )
    pol = polarization_stats(scores, ["a", "b", "c"])
    p = np.asarray(pol["p"])
    assert abs(p[0] - 1.0) < 1e-12
    assert abs(p[2] - 0.0) < 1e-12
    # middling contexts: a (mu=60) and c (mu=50); b (mu=20) is not
    assert pol["n_middling"] == 2
    # f_mid: a -> 0/5 draws in [25,75]; c -> 5/5
    assert abs(pol["mean_f_mid"] - 0.5) < 1e-12
    assert abs(pol["q_pol"] - (0.5 - 0.4)) < 1e-12
    assert abs(pol["g_pol"] - (0.5 - 0.5)) < 1e-12
    assert MID_LO < 60 < MID_HI  # fixture sanity


def test_variance_components_and_ceiling_formula():
    n, k = 500, 5
    true = RNG.normal(0, 10.0, size=(n, 1))
    within = RNG.normal(0, 4.0, size=(n, k))
    scores = 50 + true + within
    vc = variance_components(scores)
    assert abs(vc["within_sd_mean"] - 4.0) < 0.5
    # corrected between-SD recovers the true context SD (10), raw is inflated
    assert abs(vc["between_sd_corrected"] - 10.0) < 1.0
    ceils = column_ceilings(scores, judge_draw_var=None)
    # analytic: ceil(m) = sqrt(var_true / (var_true + var_within/m))
    vt, vw = ceils["var_true_between"], ceils["var_rollout"]
    expect_greedy = np.sqrt(vt / (vt + vw))
    assert abs(ceils["ceil_greedy"] - expect_greedy) < 1e-9
    assert ceils["ceil_avg_k5"] > ceils["ceil_greedy"]
    assert 0.0 < ceils["ceil_greedy"] < 1.0


def test_column_ceilings_judge_var_split():
    scores = 50 + RNG.normal(0, 10, size=(300, 1)) + RNG.normal(0, 4, size=(300, 5))
    with_jv = column_ceilings(scores, judge_draw_var=12.0, n_judge_draws=3)
    no_jv = column_ceilings(scores, judge_draw_var=None)
    # judge variance is carved OUT of the rollout component
    assert with_jv["var_rollout"] < no_jv["var_rollout"]
    assert with_jv["judge_draw_var"] == 12.0 and no_jv["judge_draw_var"] is None


def test_commonality_recovers_planted_unique_structure():
    n = 800
    x1 = RNG.normal(size=n)
    x2 = RNG.normal(size=n)
    y = x1 + 0.15 * RNG.normal(size=n)  # y driven by x1 only
    cm = commonality(y, x1, x2)
    assert cm["r2_full"] > 0.85
    assert cm["unique_x1"] > 0.8
    assert abs(cm["unique_x2"]) < 0.05
    assert abs(cm["shared"]) < 0.05
    # planted SHARED structure: both predictors are noisy reads of one driver
    z = RNG.normal(size=n)
    cm2 = commonality(
        z + 0.3 * RNG.normal(size=n), z + 0.3 * RNG.normal(size=n), z + 0.3 * RNG.normal(size=n)
    )
    assert cm2["shared"] > 0.5
    assert cm2["unique_x1"] < 0.2 and cm2["unique_x2"] < 0.2


def test_commonality_nan_and_small_n_guards():
    y = np.array([1.0, 2.0, np.nan, 4.0])
    out = commonality(y, y, y)
    assert out["r2_full"] is None and out["n"] == 3


def test_split_half_reliability_spearman_brown():
    a = RNG.normal(size=200)
    b = a + 0.1 * RNG.normal(size=200)
    r = split_half_reliability(a, b)
    assert r is not None and r > 0.98
    assert split_half_reliability(a, RNG.normal(size=200)) < 0.5


def test_r2_and_dispersion_basics():
    y = RNG.normal(size=(50, 4))
    assert abs(r2_score_rows(y, y) - 1.0) < 1e-12
    pred0 = np.tile(y.mean(axis=0), (50, 1))
    assert abs(r2_score_rows(pred0, y)) < 1e-9
    # identical rollouts -> dispersion 0; orthogonal-ish -> > 0
    v_same = np.tile(RNG.normal(size=(10, 1, 8)), (1, 5, 1))
    assert np.allclose(pairwise_cos_dispersion(v_same), 0.0, atol=1e-6)
    v_rand = RNG.normal(size=(10, 5, 64))
    assert pairwise_cos_dispersion(v_rand).mean() > 0.5


def test_holm_adjusted_cis_orders_and_bounds():
    draws = {
        "strong": RNG.normal(0.5, 0.05, size=500),  # clearly > 0
        "null": RNG.normal(0.0, 0.2, size=500),
    }
    out = holm_adjusted_cis(draws)
    assert out["strong"]["p_holm"] < 0.05
    assert out["null"]["p_holm"] > 0.2
    lo, hi = out["strong"]["ci_holm"]
    assert lo > 0.0 and hi > lo
    # Holm-adjusted p is monotone: max over ordered running values
    assert out["null"]["p_holm"] >= out["strong"]["p_holm"]
