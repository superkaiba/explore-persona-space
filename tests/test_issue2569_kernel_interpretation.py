"""Unit tests for scripts/issue2569_kernel_interpretation.py pure helpers.

Tiny-synthetic only (d <= 48), mirroring the test_issue2569_weights.py posture:
operators are built with DISTINCT singular values so the SVD never mixes
degenerate subspaces, and every expected value is computed independently from
the constructed factors (never from the code under test).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "scripts"))

import issue2569_kernel_interpretation as KI  # noqa: E402


def _synthetic(d: int = 32, seed: int = 7):
    """Row-action operator with known factors: A = U diag(s) Vh."""
    rng = np.random.default_rng(seed)
    U, _ = np.linalg.qr(rng.standard_normal((d, d)))
    V, _ = np.linalg.qr(rng.standard_normal((d, d)))
    s = np.sort(rng.uniform(0.05, 4.0, size=d))[::-1]
    s *= 1.0 + 1e-3 * np.arange(d)[::-1]  # enforce strictly distinct values
    A = U @ np.diag(s) @ V.T
    return A, U, s, V.T


def test_svd_row_action_recovers_factors():
    A, U0, s0, Vh0 = _synthetic()
    U, s, Vh = KI.svd_row_action(A)
    assert np.allclose(s, s0, rtol=1e-10)
    # row action identity: u_i @ A = s_i v_i for every triplet
    for i in range(len(s)):
        assert np.allclose(U[:, i] @ A, s[i] * Vh[i], atol=1e-10)
    # recovered read directions match the constructed ones up to sign
    for i in range(len(s)):
        assert abs(abs(U[:, i] @ U0[:, i]) - 1.0) < 1e-8


def test_mass_partitions_matches_manual_cumsum():
    _A, _U, s, _Vh = _synthetic(d=24, seed=1)
    parts = KI.mass_partitions(s, masses=(0.99, 0.90))
    for m in (0.99, 0.90):
        cum = np.cumsum(s**2) / np.sum(s**2)
        rank = int(np.searchsorted(cum, m) + 1)
        assert parts[m]["rank"] == rank
        assert parts[m]["tau"] == s[rank - 1]
        # strict <: the boundary sigma stays OUT of the kernel
        assert parts[m]["mask"].sum() == int((s < s[rank - 1]).sum())
        assert not parts[m]["mask"][rank - 1]


def test_kernel_share_exact_on_constructed_directions():
    A, _U0, s, _Vh0 = _synthetic(d=32, seed=3)
    U, s2, _ = KI.svd_row_action(A)
    parts = KI.mass_partitions(s2, masses=(0.90,))
    mask = parts[0.90]["mask"]
    k_idx = np.flatnonzero(mask)
    r_idx = np.flatnonzero(~mask)
    assert k_idx.size >= 2 and r_idx.size >= 2
    d_ker = U[:, k_idx[0]]
    d_rng = U[:, r_idx[0]]
    d_mix = (U[:, k_idx[0]] + U[:, r_idx[0]]) / np.sqrt(2.0)
    dirs = np.stack([d_ker, d_rng, d_mix])
    shares = KI.shares_at_masks(U, dirs, {0.90: mask}, block=2)[0.90]
    assert np.allclose(shares, [1.0, 0.0, 0.5], atol=1e-10)


def test_kernel_share_handles_unnormalized_and_zero_rows():
    A, *_ = _synthetic(d=16, seed=5)
    U, s, _ = KI.svd_row_action(A)
    mask = KI.mass_partitions(s, masses=(0.90,))[0.90]["mask"]
    k0 = int(np.flatnonzero(mask)[0])
    dirs = np.stack([3.7 * U[:, k0], np.zeros(16)])
    shares = KI.shares_at_masks(U, dirs, {0.90: mask})[0.90]
    assert abs(shares[0] - 1.0) < 1e-10  # scale-invariant
    assert np.isnan(shares[1])  # zero row -> NaN, never a silent 0


def test_projected_cov_trace_fraction_and_modes_match_dense():
    d = 20
    A, *_ = _synthetic(d=d, seed=11)
    U, s, _ = KI.svd_row_action(A)
    mask = KI.mass_partitions(s, masses=(0.90,))[0.90]["mask"]
    rng = np.random.default_rng(2)
    M = rng.standard_normal((d, d))
    sigma = M @ M.T / d
    # trace fraction vs dense projector
    P = U[:, mask] @ U[:, mask].T
    want = np.trace(P @ sigma) / np.trace(sigma)
    got = KI.projected_cov_trace_fraction(U, mask, sigma)
    assert abs(got - want) < 1e-12
    # modes vs dense eigh of P sigma P
    vals, modes = KI.projected_cov_modes(U, mask, sigma, top_k=3)
    w_dense, q_dense = np.linalg.eigh(P @ sigma @ P)
    order = np.argsort(w_dense)[::-1][:3]
    assert np.allclose(vals, w_dense[order], atol=1e-10)
    for r in range(3):
        # eigenvector match up to sign; modes are unit rows inside the subspace
        assert abs(abs(modes[r] @ q_dense[:, order[r]]) - 1.0) < 1e-8
        assert abs(np.linalg.norm(modes[r]) - 1.0) < 1e-12
        assert np.linalg.norm(P @ modes[r] - modes[r]) < 1e-10


def test_kernel_plus_range_shares_sum_to_one():
    A, *_ = _synthetic(d=24, seed=13)
    U, s, _ = KI.svd_row_action(A)
    mask = KI.mass_partitions(s, masses=(0.99,))[0.99]["mask"]
    rng = np.random.default_rng(4)
    dirs = rng.standard_normal((8, 24))
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
    ker = KI.shares_at_masks(U, dirs, {0.99: mask})[0.99]
    rng_share = KI.shares_at_masks(U, dirs, {0.99: ~mask})[0.99]
    assert np.allclose(ker + rng_share, 1.0, atol=1e-10)


def test_redact_scrubs_key_shaped_strings():
    # tokens built by concatenation so no key-shaped literal sits in the source
    fake_sk = "sk-" + "abcDEF1234567890" + "abcdef"
    fake_bearer = "Bearer " + "AbCdEfGh12345678" + "XyZ"
    s = KI.redact(f"use {fake_sk} and {fake_bearer} now")
    assert "sk-" not in s and "Bearer A" not in s and "[REDACTED]" in s
    # generic long mixed-alphanumeric token
    s2 = KI.redact("token is a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4 ok")
    assert "a1b2c3d4e5f6" not in s2
    # ordinary prose survives
    s3 = KI.redact("please write a short story about a dragon")
    assert s3 == "please write a short story about a dragon"


def test_tail_quote_takes_tail_and_flattens():
    text = "first line\nsecond   line " + "x" * 50 + " END"
    q = KI.tail_quote(text, 20)
    assert len(q) <= 20 and q.endswith("END") and "\n" not in q
