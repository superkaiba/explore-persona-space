"""Unit tests for scripts/issue2569_eigen_dashboards_v2.py (plane-cosine + whitening).

Tiny-synthetic only (d <= 48), per the repo convention: the dense d=3584
factorizations are production territory; these tests exercise the pure helpers
— conjugate-pair collapse into invariant 2-planes, plane cosines, the Haar
2-plane null, and the shrunk-covariance whitening pipeline — on operators
small enough to verify against hand-computed values.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "scripts"))

import issue2569_eigen_dashboards_v2 as V2  # noqa: E402


def _rot_scale_block(r: float, theta: float) -> np.ndarray:
    """2x2 rotation-scaling block with eigenvalues r * exp(+-i theta)."""
    c, s = np.cos(theta), np.sin(theta)
    return r * np.array([[c, -s], [s, c]])


def _synthetic_operator(d: int = 48, n_blocks: int = 8, seed: int = 0) -> np.ndarray:
    """Block-diagonal real operator: n_blocks rotation-scale blocks (complex
    conjugate pairs) + distinct real eigenvalues on the remaining diagonal."""
    rng = np.random.default_rng(seed)
    A = np.zeros((d, d))
    radii = np.linspace(2.0, 0.6, n_blocks)
    for i, r in enumerate(radii):
        th = 0.3 + 0.15 * i
        A[2 * i : 2 * i + 2, 2 * i : 2 * i + 2] = _rot_scale_block(r, th)
    n_real = d - 2 * n_blocks
    A[np.arange(2 * n_blocks, d), np.arange(2 * n_blocks, d)] = np.linspace(
        0.5, 0.05, n_real
    ) * np.sign(rng.standard_normal(n_real) + 3.0)
    return A


# ── plane basis ────────────────────────────────────────────────────────────────────


def test_plane_basis_orthonormal_and_spans_re_im():
    rng = np.random.default_rng(1)
    d = 16
    z = rng.standard_normal(d) + 1j * rng.standard_normal(d)
    Q, diag = V2.plane_basis_from_complex(z)
    assert Q.shape == (2, d) and not diag["degenerate"]
    np.testing.assert_allclose(Q @ Q.T, np.eye(2), atol=1e-12)
    # Re z and Im z both lie in span(Q)
    for v in (z.real, z.imag):
        proj = Q.T @ (Q @ v)
        np.testing.assert_allclose(proj, v, atol=1e-10)
    # axis 1 is exactly the normalized real part (the v1 read)
    np.testing.assert_allclose(Q[0], z.real / np.linalg.norm(z.real), atol=1e-14)
    assert diag["im_frac"] == pytest.approx(np.linalg.norm(z.imag) / np.linalg.norm(z))


def test_plane_basis_degenerate_cases():
    d = 8
    re = np.zeros(d)
    re[0] = 1.0
    # purely real vector -> degenerate 1-D
    Q, diag = V2.plane_basis_from_complex(re.astype(np.complex128))
    assert Q.shape == (1, d) and diag["degenerate"] and diag["im_frac"] == 0.0
    # purely imaginary vector -> degenerate 1-D along Im
    Q, diag = V2.plane_basis_from_complex(1j * re)
    assert Q.shape == (1, d) and diag["degenerate"]
    np.testing.assert_allclose(np.abs(Q[0]), re, atol=1e-14)


# ── conjugate-pair collapse on a synthetic operator ────────────────────────────────


def test_collapse_eigen_directions_synthetic_operator():
    import scipy.linalg as sla

    d, n_blocks = 48, 8
    A = _synthetic_operator(d, n_blocks)
    lam, V = sla.eig(A)
    U = np.linalg.inv(V)
    n_want = 12
    entries = V2.collapse_eigen_directions(lam, V, U, n_want=n_want)
    assert len(entries) == n_want
    # |lambda| non-increasing across entries
    mags = [e["abs_lambda"] for e in entries]
    assert all(mags[i] >= mags[i + 1] - 1e-12 for i in range(len(mags) - 1))
    # the 8 largest-|lambda| entries are the rotation blocks -> planes with 2 ranks
    for e in entries[:n_blocks]:
        assert e["kind"] == "plane" and len(e["ranks"]) == 2 and e["conjugate_partner_found"]
        # block-diagonal: the invariant read plane is a coordinate 2-plane; the
        # basis must reproduce it (both coordinate axes lie in span(Q))
        Q = e["read_basis"]
        nz = np.flatnonzero(np.abs(Q).sum(axis=0) > 1e-8)
        assert nz.size == 2
        for ax in nz:
            v = np.zeros(d)
            v[ax] = 1.0
            assert np.linalg.norm(Q.T @ (Q @ v) - v) < 1e-8
    # the remaining entries are real eigenvalues -> lines
    for e in entries[n_blocks:]:
        assert e["kind"] == "line" and len(e["ranks"]) == 1
        assert e["im_frac_read"] < 1e-10 and e["im_frac_write"] < 1e-10
    # read/write biorthogonality of a plane pair: write plane of entry i is
    # orthogonal to read planes of other blocks (block-diagonal structure)
    Qw = entries[0]["write_basis"]
    Qr_other = entries[1]["read_basis"]
    assert np.abs(Qw @ Qr_other.T).max() < 1e-8


# ── plane cosine values ────────────────────────────────────────────────────────────


def test_entry_cosine_stats_known_values():
    d = 12
    e = np.eye(d)
    Q = e[:2]  # plane = span(e1, e2)
    line = e[3][None, :]
    # dictionary columns: e1 (in-plane), (e1+e3)/sqrt2 (45deg), e4 (orthogonal), e2 (imag axis)
    D = np.stack([e[0], (e[0] + e[3]) / np.sqrt(2), e[4], e[1]], axis=1).astype(np.float32)
    stats = V2.entry_cosine_stats([Q, line], D, top_m=4)
    plane, ln = stats
    assert plane["max"] == pytest.approx(1.0, abs=1e-6)
    vals = dict(zip(plane["top_ids"], plane["top_vals"]))
    assert vals[0] == pytest.approx(1.0, abs=1e-6)  # e1 fully in plane
    assert vals[3] == pytest.approx(1.0, abs=1e-6)  # e2 fully in plane (imag axis)
    assert vals[1] == pytest.approx(1 / np.sqrt(2), abs=1e-6)
    assert vals[2] == pytest.approx(0.0, abs=1e-6)
    # imag-axis share: feature e2 carries everything on axis 2; e1 nothing
    share = dict(zip(plane["top_ids"], plane["axis2_share_top"]))
    assert share[0] == pytest.approx(0.0, abs=1e-9)
    assert share[3] == pytest.approx(1.0, abs=1e-9)
    # axis1_max = max |cos| against the real axis alone = 1.0 (feature e1)
    assert plane["axis1_max"] == pytest.approx(1.0, abs=1e-6)
    # 1-D line along e4: max is the 45deg column
    assert ln["max"] == pytest.approx(1 / np.sqrt(2), abs=1e-6)
    assert ln["top_ids"][0] == 1


# ── whitening ──────────────────────────────────────────────────────────────────────


def test_whitening_identity_covariance_is_noop():
    rng = np.random.default_rng(2)
    d, n_feat = 24, 40
    D = rng.standard_normal((d, n_feat)).astype(np.float32)
    Dn = D / np.linalg.norm(D, axis=0, keepdims=True)
    Ssh, tau = V2.shrunk_covariance(np.eye(d), 1e-2)
    assert tau == pytest.approx(1.0)
    Wm = V2.inv_sqrt_psd(Ssh)
    # Wm is a positive multiple of I -> cosines are unchanged
    z = rng.standard_normal(d) + 1j * rng.standard_normal(d)
    Q, _ = V2.plane_basis_from_complex(z)
    line = V2.unit(rng.standard_normal(d))[None, :]
    raw = V2.entry_cosine_stats([Q, line], Dn, top_m=5)
    wbases = V2.whiten_bases([Q, line], Wm)
    Dw = (Wm.astype(np.float32) @ D)
    Dwn = Dw / np.linalg.norm(Dw, axis=0, keepdims=True)
    wht = V2.entry_cosine_stats(wbases, Dwn, top_m=5)
    for r, w in zip(raw, wht):
        assert w["max"] == pytest.approx(r["max"], abs=1e-5)
        assert w["top_ids"] == r["top_ids"]


def test_whitened_cosine_matches_direct_formula_anisotropic():
    rng = np.random.default_rng(3)
    d, n_feat = 16, 25
    diag = np.linspace(5.0, 0.2, d)
    Sigma = np.diag(diag)
    Ssh, _ = V2.shrunk_covariance(Sigma, 1e-3)
    Wm = V2.inv_sqrt_psd(Ssh)
    # direct check of the inverse square root
    np.testing.assert_allclose(Wm @ Ssh @ Wm, np.eye(d), atol=1e-10)
    D = rng.standard_normal((d, n_feat)).astype(np.float32)
    x = rng.standard_normal(d)
    # pipeline value for a 1-D direction
    Dw = Wm.astype(np.float32) @ D
    Dwn = Dw / np.linalg.norm(Dw, axis=0, keepdims=True)
    (stat,) = V2.entry_cosine_stats(V2.whiten_bases([x[None, :]], Wm), Dwn, top_m=1)
    # direct formula: cos(Wm x, Wm d_j), max over j
    wx = Wm @ x
    direct = np.abs(
        (wx @ (Wm @ D.astype(np.float64))) / (np.linalg.norm(wx) * np.linalg.norm(Wm @ D.astype(np.float64), axis=0))
    )
    assert stat["max"] == pytest.approx(float(direct.max()), abs=1e-5)
    assert stat["top_ids"][0] == int(direct.argmax())


def test_whiten_bases_reorthonormalizes_planes():
    rng = np.random.default_rng(4)
    d = 20
    Sigma = np.cov(rng.standard_normal((200, d)), rowvar=False)
    Ssh, _ = V2.shrunk_covariance(Sigma, 1e-2)
    Wm = V2.inv_sqrt_psd(Ssh)
    z = rng.standard_normal(d) + 1j * rng.standard_normal(d)
    Q, _ = V2.plane_basis_from_complex(z)
    (Qw,) = V2.whiten_bases([Q], Wm)
    np.testing.assert_allclose(Qw @ Qw.T, np.eye(2), atol=1e-10)
    # row 0 stays parallel to the whitened real axis
    w0 = Q[0] @ Wm
    cos0 = abs(w0 @ Qw[0]) / np.linalg.norm(w0)
    assert cos0 == pytest.approx(1.0, abs=1e-10)


# ── null floors ────────────────────────────────────────────────────────────────────


def test_analytic_plane_floor_sane():
    f1 = V2.analytic_max_plane_cos_floor(1000, 48)
    f2 = V2.analytic_max_plane_cos_floor(10_000, 48)
    f3 = V2.analytic_max_plane_cos_floor(1000, 24)
    assert 0.0 < f1 < 1.0
    assert f2 > f1  # more features -> higher floor
    assert f3 > f1  # smaller d -> higher floor
    # NOTE: no cross-kind comparison against analytic_max_cos_floor here — the
    # 1-D convention sqrt(2 ln N / d) is a Gaussian-tail approximation that
    # overshoots the exact Beta value at small d, so it can sit above the exact
    # plane floor even though a plane dominates a line under the same law. The
    # plane-dominates-line fact is asserted on the EMPIRICAL nulls below.


def test_empirical_nulls_whitener_identity_matches_raw():
    rng = np.random.default_rng(5)
    d, n_feat, n_draws = 32, 300, 64
    D = rng.standard_normal((d, n_feat)).astype(np.float32)
    Dn = D / np.linalg.norm(D, axis=0, keepdims=True)
    eye32 = np.eye(d, dtype=np.float32)
    a = V2.empirical_max_cos_null_lines(Dn, n_draws, seed=7)
    b = V2.empirical_max_cos_null_lines(Dn, n_draws, seed=7, whitener32=eye32)
    assert a["p95"] == pytest.approx(b["p95"], abs=1e-6)
    ap = V2.empirical_max_plane_cos_null(Dn, n_draws, seed=7)
    bp = V2.empirical_max_plane_cos_null(Dn, n_draws, seed=7, whitener32=eye32)
    assert ap["p95"] == pytest.approx(bp["p95"], abs=1e-6)
    # planes dominate lines on the same dictionary
    assert ap["p50"] > a["p50"]


def test_empirical_plane_null_tracks_analytic_scale():
    rng = np.random.default_rng(6)
    d, n_feat, n_draws = 32, 2000, 128
    D = rng.standard_normal((d, n_feat)).astype(np.float32)
    Dn = D / np.linalg.norm(D, axis=0, keepdims=True)
    emp = V2.empirical_max_plane_cos_null(Dn, n_draws, seed=11)
    ana = V2.analytic_max_plane_cos_floor(n_feat, d)
    assert 0.5 * ana < emp["p50"] < 1.5 * ana
