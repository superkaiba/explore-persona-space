"""#1775 engine pins: batched permutation nulls == serial #763 reference,
fit_press_pairs == the banked _fit_cv on complement pairs, LOO prefix means,
derangement validity, cluster-bootstrap point identity, errorbar offsets.

All synthetic + CPU + seconds-fast; no network, no store reads.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "src"))

i1775 = pytest.importorskip("issue1775_common")


def _toy(n=60, d=8, p=5, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, d))
    W = rng.standard_normal((d, p))
    Y = X @ W + 0.3 * rng.standard_normal((n, p))
    rows = [
        {"prefix_id": f"pf{i % 6}", "query_id": f"q{i % 10}", "stratum": "dense_core"}
        for i in range(n)
    ]
    return X, Y, rows


def test_fit_press_pairs_matches_fit_cv():
    """Complement pairs -> numerically identical to the banked _fit_cv engine."""
    X, Y, rows = _toy()
    folds = i1775._folds_from_manifest(rows, len(rows), group_key="prefix_id", n_folds=3)
    ref = i1775._fit_cv(X, Y, folds)
    pairs = i1775.fold_pairs(rows, len(rows), "prefix", n_folds=3)
    got, _pred, _cov = i1775.fit_press_pairs(X, Y, pairs)
    assert abs(got["r2"] - ref["r2"]) < 1e-12
    assert got["lambda_indices"] == ref["lambda_indices"]


def test_batched_null_matches_serial_reference():
    """One batched permuted draw == hsic/dcor computed serially on permuted rows."""
    X, Y, _rows = _toy(n=40)
    R = Y[:, :3]
    mats = i1775.build_dependence_matrices(X, R, device="cpu")
    obs = i1775.observed_stats(mats)
    assert abs(obs["hsic"] - i1775.hsic_statistic(X, R)) < 1e-8
    assert abs(obs["dcor"] - i1775.distance_correlation(X, R)) < 1e-6
    rng = np.random.default_rng(3)
    perm = rng.permutation(40)
    got = i1775.null_stats_batched(mats, perm[None, :])
    # serial reference on permuted residual rows: kernels recomputed from scratch
    ref_h = i1775.hsic_statistic(X, R[perm])
    ref_d = i1775.distance_correlation(X, R[perm])
    assert abs(float(got["hsic"][0]) - ref_h) < 1e-6
    assert abs(float(got["dcor"][0]) - ref_d) < 1e-5


def test_crossed_permutations_shapes_and_derangement():
    P, Q = 5, 7
    for scheme in ("prefix_block", "query_block", "within_prefix_derangement"):
        perms = i1775.crossed_permutations(P, Q, scheme, 8, seed=1)
        assert perms.shape == (8, P * Q)
        for b in range(8):
            assert sorted(perms[b].tolist()) == list(range(P * Q))
    der = i1775._batched_derangements(np.random.default_rng(0), 50, 6)
    assert not (der == np.arange(6)).any()


def test_cluster_bootstrap_point_identity():
    _X, Y, rows = _toy(n=80)
    rng = np.random.default_rng(1)
    pred_a = Y + 0.1 * rng.standard_normal(Y.shape)
    pred_b = Y + 0.4 * rng.standard_normal(Y.shape)
    groups = np.asarray([r["prefix_id"] for r in rows])
    cov = np.ones(len(rows), dtype=bool)
    out = i1775.cluster_bootstrap_delta_r2(Y, pred_a, pred_b, cov, groups, n_draws=50, seed=0)
    direct = i1775._r2(Y, pred_a) - i1775._r2(Y, pred_b)
    assert abs(out["delta_r2"] - direct) < 1e-10
    assert out["ci95_cluster"][0] <= out["delta_r2"] <= out["ci95_cluster"][1]


def test_loo_prefix_mean_and_singleton_mask():
    X = np.arange(12, dtype=np.float64).reshape(6, 2)
    prefixes = np.asarray(["a", "a", "a", "b", "b", "c"])
    out, mask = i1775._loo_prefix_mean(X, prefixes)
    np.testing.assert_allclose(out[0], X[[1, 2]].mean(0))
    np.testing.assert_allclose(out[3], X[4])
    assert mask.tolist() == [True] * 5 + [False]  # singleton prefix c masked out


def test_holm_correction_monotone():
    p = {"a": 0.01, "b": 0.02, "c": 0.5}
    adj = i1775.holm_correction(p)
    assert adj["a"] == pytest.approx(0.03)
    assert adj["b"] == pytest.approx(0.04)
    assert adj["c"] == pytest.approx(0.5)
    assert adj["a"] <= adj["b"] <= adj["c"]


def test_err_offsets_never_negative_on_inverted_ci():
    """The #547/#1335 xerr class: inverted quantile CI must clamp, then render."""
    figs = pytest.importorskip("issue1775_figures")
    lo, hi = figs._err_offsets(0.5, [0.6, 0.4])  # deliberately INVERTED bounds
    assert lo >= 0.0 and hi >= 0.0
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    ax.errorbar([0.0], [0.5], yerr=[[lo], [hi]])  # must not raise ValueError
    fig.savefig(Path("/tmp") / "i1775_errbar_smoke.png")
    plt.close(fig)


def test_doubly_fold_pairs_disjoint():
    _X, _Y, rows = _toy(n=90)
    pairs = i1775.fold_pairs(rows, len(rows), "doubly", n_folds=3)
    assert pairs, "doubly scheme produced no usable pairs"
    for tr, te in pairs:
        assert not set(tr.tolist()) & set(te.tolist())
        te_prefixes = {rows[i]["prefix_id"] for i in te}
        te_queries = {rows[i]["query_id"] for i in te}
        assert not any(rows[i]["prefix_id"] in te_prefixes for i in tr)
        assert not any(rows[i]["query_id"] in te_queries for i in tr)
