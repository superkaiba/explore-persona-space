"""Issue #1895 driver pins: the registered rotation-invariance unit test (plan
assumption 13), the fingerprint-resume predicate, the shell-partition +
within-shell rotation null construction, the H3 plug-in reference arithmetic,
and the sparse->dense code scatter. CPU-only; no network, no model loads."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "src"))

import issue779_ffc_n1m_fits as N1M  # noqa: E402
import issue779_percontext_recon as PR  # noqa: E402
import issue1895_subspaces as D  # noqa: E402


def _synth(n=96, d=12, dt=8, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, d)).astype(np.float32)
    W = rng.standard_normal((d, dt)).astype(np.float32)
    Y = (X @ W + 0.2 * rng.standard_normal((n, dt))).astype(np.float32)
    idx = rng.permutation(n)
    return X, Y, idx[:64], idx[64:80], idx[80:]


def _haar(dt, seed=1):
    g = np.random.default_rng(seed).standard_normal((dt, dt))
    q, r = np.linalg.qr(g)
    return (q * np.sign(np.diag(r))[None, :]).astype(np.float64)


def test_pooled_r2_rotation_invariant_at_fixed_lambda():
    """Plan assumption 13 (registered unit test): pooled ridge R2 at a FIXED
    lambda is invariant to any orthogonal rotation of the TARGET — which is why
    a rotation null is vacuous for pooled R2 and the H3 plug-in is used instead."""
    X, Y, tr, _va, te = _synth()
    rot = _haar(Y.shape[1])
    Yr = (Y.astype(np.float64) @ rot).astype(np.float32)
    lam = 10.0
    fac = N1M._ridge_factorize(X, Y, tr, "cpu", 32)
    fac_r = N1M._ridge_factorize(X, Yr, tr, "cpu", 32)
    r2 = PR._pooled_r2(N1M._ridge_predict_one(X, te, fac, lam, "cpu", 32), Y[te])
    r2_r = PR._pooled_r2(N1M._ridge_predict_one(X, te, fac_r, lam, "cpu", 32), Yr[te])
    assert np.isclose(r2, r2_r, rtol=0, atol=1e-6), (r2, r2_r)


def test_fingerprint_resume_predicate(tmp_path, monkeypatch):
    """Stale/missing fingerprint => recompute; matching fingerprint => skip;
    never bare file existence (the #952 gate-5 manifest shape)."""
    monkeypatch.setattr(
        D,
        "committed_split_shas",
        lambda: {
            "train_full_sha256": "a",
            "holdout_sha256": "b",
            "sae_fit_sha256": "c",
            "sae_val_sha256": "d",
        },
    )
    monkeypatch.setattr(D, "_code_sha", lambda: "deadbeef")
    args = SimpleNamespace(
        out_root=tmp_path,
        sae_k=64,
        k_grid=(16, 32),
        n_shells=32,
        shell_grid=(16, 32, 64),
        angle_draws=25,
        boot_draws=200,
        seed=1895,
        smoke=True,
        max_shards=4,
        holdout_cap=500,
        fit_cap=2000,
        val_cap=200,
        pilot_n=8,
        tiny_model=True,
        force_path="A",
    )
    fp = D.fingerprint(args)
    assert not D.resume_ok(args, "recon", fp)  # missing sentinel
    D.write_done(args, "recon", fp)
    assert D.resume_ok(args, "recon", fp)  # matching fingerprint
    args2 = SimpleNamespace(**{**vars(args), "boot_draws": 500})
    fp2 = D.fingerprint(args2)
    assert not D.resume_ok(args2, "recon", fp2)  # regime change => stale
    # per-unit shard predicate: bare existence never suffices
    shard = tmp_path / "u.npz"
    np.savez(shard, x=np.zeros(2), fingerprint=np.array(D._fp_str(fp)))
    assert D.shard_resume_ok(shard, fp)
    assert not D.shard_resume_ok(shard, fp2)
    np.savez(shard, x=np.zeros(2))  # no fingerprint key at all
    assert not D.shard_resume_ok(shard, fp)


def test_shell_partition_covers_and_is_disjoint():
    eigvals = np.geomspace(200.0, 0.01, 97)  # desc spectrum
    shells = D.shell_partition(eigvals, 8)
    allidx = np.concatenate(shells)
    assert len(allidx) == 97 and len(np.unique(allidx)) == 97
    # geometric shells partition by VALUE — each shell's dims are contiguous in rank
    for rows in shells:
        assert np.array_equal(np.sort(rows), np.arange(rows.min(), rows.max() + 1))


def test_within_shell_rotation_null_construction():
    """Small synthetic: the null preserves within-shell energy structure; a basis
    aligned with the coordinate subspace scores O=1 observed, while shell
    rotations push null O below 1; identity-subspace overlap arithmetic is exact."""
    rng = np.random.default_rng(3)
    n_dim, k = 24, 4
    eigvals = np.geomspace(50.0, 0.1, n_dim)
    shells = D.shell_partition(eigvals, 4)
    s_pred = np.arange(k)
    B = np.zeros((n_dim, k), dtype=np.float32)
    B[:k, :k] = np.eye(k)  # B == the coordinate subspace itself
    cs, obs = D.overlap_observed(s_pred, B)
    assert np.allclose(cs, 1.0) and np.isclose(obs, 1.0)
    draws = D.overlap_null_draws(s_pred, B, shells, n_draws=64, seed=7, device="cpu")
    assert draws.shape == (64,)
    assert np.all(draws <= 1.0 + 1e-6) and np.all(draws >= 0.0)
    assert draws.mean() < 0.999  # rotations genuinely move the basis
    # a basis fully OUTSIDE the s_pred shell(s) stays orthogonal under the null
    # whenever no shell straddles the boundary
    B2 = np.zeros((n_dim, k), dtype=np.float32)
    B2[-k:, :] = np.eye(k)
    shell_of = np.empty(n_dim, int)
    for si, rows in enumerate(shells):
        shell_of[rows] = si
    if set(shell_of[:k]).isdisjoint(set(shell_of[-k:])):
        _, obs2 = D.overlap_observed(s_pred, B2)
        draws2 = D.overlap_null_draws(s_pred, B2, shells, n_draws=16, seed=9, device="cpu")
        assert np.isclose(obs2, 0.0, atol=1e-6)
        assert np.allclose(draws2, 0.0, atol=1e-5)
    # rotated row-frames are orthonormal: rank-preserving => svdvals <= 1 already checked
    del rng


def test_h3_plugin_reference_arithmetic():
    """R2_H3(T) = sum_u energy_u(T) g(u) / sum_u energy_u(T): hand-checked values,
    plus the point-mass identity (all energy in one direction => plug-in == g(u))."""
    g = np.array([0.9, 0.5, 0.1])
    energy = np.array([2.0, 1.0, 1.0])
    expected = (2 * 0.9 + 1 * 0.5 + 1 * 0.1) / 4.0
    got = float((energy * g).sum() / energy.sum())
    assert np.isclose(got, expected)
    point = np.array([0.0, 3.0, 0.0])
    assert np.isclose(float((point * g).sum() / point.sum()), 0.5)


def test_h3_bootstrap_matches_direct_reduction(tmp_path):
    """The 3-GEMM bootstrap reduction reproduces a direct per-draw recomputation
    on a tiny synthetic (counts-weighted pooled R2 + plug-in)."""
    rng = np.random.default_rng(11)
    n, dd = 20, 5
    V = rng.standard_normal((n, dd)).astype(np.float32)
    R1 = 0.3 * rng.standard_normal((n, dd)).astype(np.float32)
    E = rng.standard_normal((n, dd)).astype(np.float32)
    R3 = 0.5 * rng.standard_normal((n, dd)).astype(np.float32)
    W = rng.multinomial(n, np.full(n, 1 / n), size=3).astype(np.float32)
    # batched (the driver's shape)
    mean_v = (W @ V) / n
    sstot_v = W @ (V * V) - n * mean_v**2
    rs1 = W @ (R1 * R1)
    g_u = 1.0 - rs1 / np.maximum(sstot_v, 1e-9)
    mean_e = (W @ E) / n
    sstot_e = W @ (E * E) - n * mean_e**2
    rs3 = W @ (R3 * R3)
    r2_e = 1.0 - rs3.sum(1) / np.maximum(sstot_e.sum(1), 1e-9)
    plug_e = (sstot_e * g_u).sum(1) / np.maximum(sstot_e.sum(1), 1e-9)
    # direct per-draw expansion
    for d_i in range(3):
        idx = np.repeat(np.arange(n), W[d_i].astype(int))
        ve, ee, r1e, r3e = V[idx], E[idx], R1[idx], R3[idx]
        sstot_v_d = ((ve - ve.mean(0)) ** 2).sum(0)
        g_d = 1.0 - (r1e**2).sum(0) / np.maximum(sstot_v_d, 1e-9)
        sstot_e_d = ((ee - ee.mean(0)) ** 2).sum(0)
        r2_d = 1.0 - (r3e**2).sum() / max(sstot_e_d.sum(), 1e-9)
        plug_d = (sstot_e_d * g_d).sum() / max(sstot_e_d.sum(), 1e-9)
        assert np.allclose(g_u[d_i], g_d, atol=1e-3)
        assert np.isclose(r2_e[d_i], r2_d, atol=1e-3)
        assert np.isclose(plug_e[d_i], plug_d, atol=1e-3)


def test_dense_codes_scatter_matches_loop_reference():
    """Union-index sparse (idx_off row COUNTS + concatenated ans_idx/ans_mean)
    -> dense scatter matches a per-row python reference."""
    part = {
        "idx_off": np.array([2, 0, 3], dtype=np.int64),
        "ans_idx": np.array([1, 4, 0, 2, 5], dtype=np.int32),
        "ans_mean": np.array([0.5, -1.0, 2.0, 0.25, 1.5], dtype=np.float16),
    }
    dense = D._dense_codes(part, "cpu", dict_size=6).numpy()
    ref = np.zeros((3, 6), dtype=np.float32)
    ref[0, 1], ref[0, 4] = 0.5, -1.0
    ref[2, 0], ref[2, 2], ref[2, 5] = 2.0, 0.25, 1.5
    assert np.allclose(dense, ref, atol=1e-3)


def test_partial_spearman_zero_when_relation_is_pure_covariate():
    """When x and y are both monotone functions of the covariate alone, the
    variance-partialled correlation collapses toward 0 (the H3/P_var0 read)."""
    rng = np.random.default_rng(5)
    c = np.arange(300, dtype=np.float64)
    x = -c + 0.5 * rng.standard_normal(300)
    y = -c + 0.5 * rng.standard_normal(300)
    raw = float(np.corrcoef(D._midrank_1d(x), D._midrank_1d(y))[0, 1])
    part = D._partial_spearman_obs(x, y, [c])
    assert raw > 0.9
    assert abs(part) < 0.2


def test_bh_fdr_basic():
    p = [0.001, 0.011, 0.02, 0.8, np.nan]
    passed = D._bh_fdr(p, q=0.05)
    assert passed[0] and not passed[3] and not passed[4]


def test_smoke_estimator_validity_exemption_documented():
    """The smoke branch deliberately runs n_train < d (#1701 regularization-limit
    exemption) — pin that the production branch asserts n_train >= d."""
    import inspect

    src = inspect.getsource(D._build_design)
    assert "1701" in src and "assert len(tr) >= X_sub.shape[1]" in src


@pytest.mark.parametrize("k,kb", [(4, 4), (6, 3)])
def test_overlap_observed_bounds(k, kb):
    rng = np.random.default_rng(2)
    B, _ = np.linalg.qr(rng.standard_normal((20, kb)))
    cs, o = D.overlap_observed(np.arange(k), B.astype(np.float32))
    assert len(cs) == min(k, kb)
    assert 0.0 <= o <= 1.0 + 1e-9


def test_haar_row_frame_is_orthonormal():
    """The Stiefel row-frame trick in overlap_null_draws: QR of a (d, m) gaussian
    gives orthonormal columns; transposed rows are an orthonormal m-frame."""
    gen = torch.Generator().manual_seed(0)
    G = torch.randn((5, 9, 3), generator=gen)
    Qc, R = torch.linalg.qr(G)
    sign = torch.sign(torch.diagonal(R, dim1=-2, dim2=-1))
    sign = torch.where(sign == 0, torch.ones_like(sign), sign)
    Qc = Qc * sign.unsqueeze(-2)
    eye = torch.eye(3).expand(5, 3, 3)
    assert torch.allclose(Qc.transpose(-2, -1) @ Qc, eye, atol=1e-5)
