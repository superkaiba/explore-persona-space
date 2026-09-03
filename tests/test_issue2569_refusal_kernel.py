"""Unit tests for scripts/issue2569_refusal_kernel.py pure helpers.

Synthetic shapes use d >= 16 and n != d throughout so a transposed GEMM or a
row/column mix-up cannot silently pass.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "scripts"))

import issue2569_refusal_kernel as RK  # noqa: E402

D_TEST = 24
N_PAIRS = 17  # != D_TEST


def _orthonormal(d: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    q, _ = np.linalg.qr(rng.standard_normal((d, d)))
    return q


def test_unit_rows_normalizes_and_rejects_zero():
    rng = np.random.default_rng(1)
    x = rng.standard_normal((N_PAIRS, D_TEST)) * 7.0
    u = RK.unit_rows(x)
    assert u.shape == (N_PAIRS, D_TEST)
    np.testing.assert_allclose(np.linalg.norm(u, axis=1), 1.0, atol=1e-12)
    x[3] = 0.0
    with pytest.raises(ValueError):
        RK.unit_rows(x)


def test_bootstrap_median_ci_brackets_median():
    rng = np.random.default_rng(2)
    v = rng.normal(0.7, 0.05, size=61)
    rec = RK.bootstrap_median_ci(v, n_boot=500, seed=3)
    assert rec["n"] == 61
    assert rec["ci95"][0] <= rec["median"] <= rec["ci95"][1]
    assert 0.6 < rec["median"] < 0.8


def test_sample_index_pairs_never_self_pairs():
    rng = np.random.default_rng(4)
    prs = RK.sample_index_pairs(9, 500, rng)
    assert prs.shape == (500, 2)
    assert np.all(prs[:, 0] != prs[:, 1])
    assert prs.min() >= 0 and prs.max() < 9


def test_distance_matched_pairs_respects_bins():
    rng = np.random.default_rng(5)
    real = np.linspace(1.0, 2.0, 40)
    cand = np.concatenate([np.linspace(0.9, 2.1, 3000), np.full(1000, 50.0)])
    idx, cov = RK.distance_matched_pairs(real, cand, n_bins=4, per_bin=50, rng=rng)
    assert idx.size == 200
    assert np.all(cand[idx] <= 2.0 + 1e-9)
    assert cov["n_matched"] == 200
    # a far-away real profile finds no support
    idx2, cov2 = RK.distance_matched_pairs(
        np.linspace(100.0, 101.0, 10), cand[:3000], n_bins=2, per_bin=5, rng=rng
    )
    assert idx2.size == 0 and cov2["n_matched"] == 0


def test_kernel_share_projector_exact_on_synthetic():
    """A direction built inside the masked subspace has share 1, outside 0."""
    U = _orthonormal(D_TEST, seed=6)
    mask = np.zeros(D_TEST, dtype=bool)
    mask[-10:] = True  # 10 kernel directions
    inside = U[:, -3] * 2.5 + U[:, -1] * 1.5
    outside = U[:, 0] * 4.0
    mixed = U[:, 0] + U[:, -1]  # half in, half out
    import issue2569_kernel_interpretation as KI

    sh = KI.shares_at_masks(U, np.stack([inside, outside, mixed]), {0.99: mask})[0.99]
    np.testing.assert_allclose(sh, [1.0, 0.0, 0.5], atol=1e-12)


def test_rowwise_cos_and_transport_r2_recover_exact_map():
    rng = np.random.default_rng(7)
    A = rng.standard_normal((D_TEST, D_TEST)) * 0.2
    dc = rng.standard_normal((N_PAIRS, D_TEST))
    obs = dc @ A  # observed exactly equals predicted
    pred = dc @ A
    cos = RK.rowwise_cos(pred, obs)
    np.testing.assert_allclose(cos, 1.0, atol=1e-12)
    r2 = RK.transport_r2(pred, obs)
    assert r2["raw"] == pytest.approx(1.0, abs=1e-12)
    assert r2["gain"] == pytest.approx(1.0, abs=1e-12)
    # shrunken prediction: raw < gain-calibrated == 1
    r2s = RK.transport_r2(0.5 * pred, obs)
    assert r2s["gain"] == pytest.approx(2.0, abs=1e-9)
    assert r2s["gain_calibrated"] == pytest.approx(1.0, abs=1e-9)
    assert r2s["raw"] < r2s["gain_calibrated"]


def test_loo_axis_excludes_own_pair():
    rng = np.random.default_rng(8)
    base = rng.standard_normal(D_TEST)
    deltas = np.tile(base, (N_PAIRS, 1)) + 0.01 * rng.standard_normal((N_PAIRS, D_TEST))
    member = np.zeros(N_PAIRS, dtype=bool)
    member[:5] = True
    # make member 0 a huge outlier along an orthogonal direction
    ortho = np.zeros(D_TEST)
    ortho[0] = 1.0
    deltas[0] = 100.0 * ortho
    proj = RK.loo_axis_projections(deltas, member)
    # member 0's own axis excludes itself, so its projection reflects base, not ortho
    ax_wo_0 = deltas[1:5].mean(axis=0)
    ax_wo_0 /= np.linalg.norm(ax_wo_0)
    assert proj[0] == pytest.approx(float(deltas[0] @ ax_wo_0), rel=1e-9)
    # non-members use the full member mean
    full = deltas[:5].mean(axis=0)
    full /= np.linalg.norm(full)
    assert proj[7] == pytest.approx(float(deltas[7] @ full), rel=1e-9)


def test_project_pairs_on_axis_matches_loo_convention():
    rng = np.random.default_rng(9)
    deltas = rng.standard_normal((N_PAIRS, D_TEST))
    pred = rng.standard_normal((N_PAIRS, D_TEST))
    member = np.zeros(N_PAIRS, dtype=bool)
    member[2:8] = True
    out = RK.project_pairs_on_axis(pred, deltas, member)
    obs_loo = RK.loo_axis_projections(deltas, member)
    np.testing.assert_allclose(out["obs"], obs_loo, atol=1e-12)
    # a member's predicted projection uses the axis without that member
    m_rows = np.flatnonzero(member)
    r = m_rows[0]
    ax = deltas[m_rows[1:]].mean(axis=0)
    ax /= np.linalg.norm(ax)
    assert out["pred"][r] == pytest.approx(float(pred[r] @ ax), rel=1e-9)


def test_spear_monotone_and_rejects_nan():
    x = np.arange(20.0)
    y = x**3 + 1
    rec = RK.spear(x, y)
    assert rec["rho"] == pytest.approx(1.0)
    with pytest.raises(ValueError):
        RK.spear(np.array([1.0, np.nan]), np.array([1.0, 2.0]))


def test_strip_arrays_drops_private_and_converts_numpy():
    doc = {
        "a": np.float64(1.5),
        "_arrays": {"big": np.zeros(5)},
        "nested": {"v": np.arange(3), "_tmp": 1},
        "lst": [np.int64(2)],
    }
    out = RK._strip_arrays(doc)
    assert "_arrays" not in out and "_tmp" not in out["nested"]
    assert out["a"] == 1.5 and out["nested"]["v"] == [0, 1, 2] and out["lst"] == [2]
