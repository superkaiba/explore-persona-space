"""Synthetic-recovery tests for the leg-10 variance-decomposition estimators.

Generates data with KNOWN L / S / W / N pieces (d=32 >= 16, n != d) and checks
each estimator recovers its piece within tolerance:

  v_A = v_C @ B  +  c_N * (cos(v_C @ P) - E[cos])  +  t * w  +  eps

with v_C ~ N(0, I_d), t ~ N(0,1) an independent whole-context factor, and
eps ~ N(0, sigma^2 I) fresh per draw. Under a symmetric input the centered
cosine is uncorrelated with every linear function of v_C, so the true pieces
are L_abs = ||B||_F^2, N_abs = c_N^2 * d_out * Var(cos(g)), W_abs = ||w||^2,
S_abs = sigma^2 * d (Var(cos(g)) = (1 + e^-2)/2 - e^-1 for g ~ N(0,1)).
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "scripts"))

import issue2569_variance_decomposition as VD  # noqa: E402

D = 32
N_TRAIN = 6000
N_EVAL = 6000
N_BANK = 1200
K_BANK = 8
SEED = 7


def _make_world(rng: np.random.Generator) -> dict:
    b = rng.standard_normal((D, D))
    b *= np.sqrt(5.0) / np.linalg.norm(b)  # L_abs = ||B||_F^2 = 5.0
    p = rng.standard_normal((D, D))
    p /= np.linalg.norm(p, axis=0, keepdims=True)  # unit columns -> g_j ~ N(0,1)
    var_cos = (1.0 + np.exp(-2.0)) / 2.0 - np.exp(-1.0)
    c_n = np.sqrt(2.0 / (D * var_cos))  # N_abs = 2.0
    w = rng.standard_normal(D)
    w *= np.sqrt(1.5) / np.linalg.norm(w)  # W_abs = 1.5
    sigma2 = 1.5 / D  # S_abs = 1.5
    total = 5.0 + 2.0 + 1.5 + 1.5
    return {
        "B": b,
        "P": p,
        "c_n": c_n,
        "w": w,
        "sigma2": sigma2,
        "true": {"L": 5.0 / total, "N": 2.0 / total, "W": 1.5 / total, "S": 1.5 / total},
        "total_abs": total,
    }


def _mu(world: dict, x: np.ndarray, t: np.ndarray) -> np.ndarray:
    nl = np.cos(x @ world["P"]) - np.exp(-0.5)
    return x @ world["B"] + world["c_n"] * nl + t[:, None] * world["w"][None, :]


def _draws(world: dict, x: np.ndarray, t: np.ndarray, k: int, rng) -> np.ndarray:
    mu = _mu(world, x, t)
    eps = rng.standard_normal((x.shape[0], k, D)) * np.sqrt(world["sigma2"])
    return mu[:, None, :] + eps


def test_estimators_recover_known_pieces() -> None:
    rng = np.random.default_rng(SEED)
    world = _make_world(rng)
    total = world["total_abs"]
    true = world["true"]

    # S from a rollout bank -----------------------------------------------------------
    xb = rng.standard_normal((N_BANK, D))
    tb = rng.standard_normal(N_BANK)
    bank = _draws(world, xb, tb, K_BANK, rng)
    wb = VD.within_between(bank)
    assert abs(wb["s_abs"] - 1.5) / 1.5 < 0.05, wb["s_abs"]
    assert abs(wb["total_abs"] - total) / total < 0.10, wb["total_abs"]

    # linear piece via a held-out lstsq map --------------------------------------------
    xtr = rng.standard_normal((N_TRAIN, D))
    ttr = rng.standard_normal(N_TRAIN)
    ytr = _draws(world, xtr, ttr, 1, rng)[:, 0, :]
    xev = rng.standard_normal((N_EVAL, D))
    tev = rng.standard_normal(N_EVAL)
    yev = _draws(world, xev, tev, 1, rng)[:, 0, :]
    a1 = np.concatenate([xtr, np.ones((N_TRAIN, 1))], axis=1)
    coef, *_ = np.linalg.lstsq(a1, ytr, rcond=None)
    resid = yev - np.concatenate([xev, np.ones((N_EVAL, 1))], axis=1) @ coef
    yc = yev - yev.mean(axis=0)
    l_est = 1.0 - float((resid**2).sum()) / float((yc**2).sum())
    assert abs(l_est - true["L"]) < 0.05, (l_est, true["L"])

    # NN intercept = S + W (raw coordinates) -------------------------------------------
    idx, d2, _dups = VD.topk_neighbors(xev.astype(np.float32), k=11, block=2048, threads=4)
    dy = yev - yev[idx[:, 0]]
    stat = 0.5 * (dy**2).sum(axis=1)
    fit = VD.binned_intercept(d2[:, 0].astype(np.float64), stat, frac=0.5, n_boot=200)
    sw_true = 1.5 + 1.5
    assert abs(fit["intercept"] - sw_true) / sw_true < 0.25, (fit["intercept"], sw_true)
    w_est_abs = fit["intercept"] - wb["s_abs"]
    assert abs(w_est_abs / total - true["W"]) < 0.10, w_est_abs / total

    # kNN regression: a finite-sample LOWER bound on L + N. At n=6000 in d=32 the
    # neighbor-mean bias is large, so the bound sits well below L + N (and can sit
    # below L itself); only the guaranteed side is asserted.
    knn = VD.knn_r2_pooled(yev, idx[:, :10], (5, 10))
    best = max(knn.values())
    assert 0.0 < best < true["L"] + true["N"] + 0.05, (best, true)

    # identity: the remainder recovers N -----------------------------------------------
    s_frac = wb["s_abs"] / total
    w_frac = w_est_abs / total
    n_est = 1.0 - l_est - s_frac - w_frac
    assert abs(n_est - true["N"]) < 0.12, (n_est, true["N"])
    assert abs(l_est + s_frac + w_frac + n_est - 1.0) < 1e-12


def test_per_direction_r2_and_whitener() -> None:
    rng = np.random.default_rng(SEED + 1)
    world = _make_world(rng)
    x = rng.standard_normal((4000, D))
    t = rng.standard_normal(4000)
    y = _draws(world, x, t, 1, rng)[:, 0, :]
    a1 = np.concatenate([x, np.ones((4000, 1))], axis=1)
    coef, *_ = np.linalg.lstsq(a1, y, rcond=None)
    resid = y - a1 @ coef
    w_dir = VD.unit(world["w"])
    b_dir = VD.unit(world["B"][:, 0] @ world["B"].T)  # a direction rich in linear signal
    r2 = VD.per_direction_r2(y, resid, np.stack([w_dir, b_dir]))
    assert r2[1] > r2[0], r2  # the linear-rich direction is better predicted
    assert np.all(r2 <= 1.0)

    sig = np.cov(x, rowvar=False)
    tr = VD.whitener(sig, shrink=1e-6)
    sig_w = tr @ sig @ tr
    assert np.abs(sig_w - np.eye(D)).max() < 0.05


def test_binned_intercept_zero_for_noiseless_linear() -> None:
    rng = np.random.default_rng(SEED + 2)
    x = rng.standard_normal((3000, D))
    y = x @ rng.standard_normal((D, D))
    idx, d2, _ = VD.topk_neighbors(x.astype(np.float32), k=2, block=1024, threads=4)
    dy = y - y[idx[:, 0]]
    stat = 0.5 * (dy**2).sum(axis=1)
    fit = VD.binned_intercept(d2[:, 0].astype(np.float64), stat, frac=0.5, n_boot=100)
    scale = float(y.var(axis=0).sum())
    assert abs(fit["intercept"]) < 0.05 * scale, (fit["intercept"], scale)


def test_memory_bounded_pair_and_knn_match_dense_reference(caplog) -> None:
    """Chunked crash-fix paths reproduce the former dense formulas exactly."""
    caplog.set_level(logging.INFO, logger="leg10")
    rng = np.random.default_rng(SEED + 3)
    y = rng.standard_normal((47, 13)).astype(np.float32)
    x = rng.standard_normal((47, 9)).astype(np.float32)
    dirs = rng.standard_normal((5, 13))
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
    idx, d2, _ = VD.topk_neighbors(x, k=8, block=11, threads=2)

    got_pair = VD.pair_stats_chunked(y, idx, d2, kth=3, dirs=dirs, row_block=7)
    dense_dy = y.astype(np.float64) - y[idx[:, 2]].astype(np.float64)
    assert np.allclose(got_pair["stat_pooled"], 0.5 * (dense_dy**2).sum(axis=1))
    assert np.allclose(got_pair["stat_dirs"], 0.5 * (dense_dy @ dirs.T) ** 2)

    got_knn = VD.knn_r2_pooled(y, idx, (2, 5, 8), row_block=6)
    y64 = y.astype(np.float64)
    ss_tot = float(((y64 - y64.mean(axis=0)) ** 2).sum())
    for k in (2, 5, 8):
        pred = y64[idx[:, :k]].mean(axis=1)
        expected = 1.0 - float(((y64 - pred) ** 2).sum()) / ss_tot
        assert abs(got_knn[str(k)] - expected) < 1e-12
    assert "[memory-bounded] pair-stats kth=3" in caplog.text
    assert "[memory-bounded] knn-r2 k=8" in caplog.text


def test_batched_bootstraps_are_batch_size_invariant() -> None:
    """Draw batching preserves the exact seeded bootstrap sample stream and estimates."""
    rng = np.random.default_rng(SEED + 4)
    values = rng.normal(size=91)
    assert np.allclose(
        VD.bootstrap_ci(values, n_boot=73, seed=17, batch_size=1),
        VD.bootstrap_ci(values, n_boot=73, seed=17, batch_size=19),
    )

    d2 = rng.uniform(0.1, 4.0, size=127)
    stat = 2.0 + 0.7 * d2 + rng.normal(scale=0.2, size=d2.size)
    serial_batches = VD.binned_intercept(
        d2, stat, frac=0.75, n_bins=8, n_boot=71, seed=23, bootstrap_batch=1
    )
    grouped_batches = VD.binned_intercept(
        d2, stat, frac=0.75, n_bins=8, n_boot=71, seed=23, bootstrap_batch=17
    )
    assert np.allclose(serial_batches["ci95"], grouped_batches["ci95"])
