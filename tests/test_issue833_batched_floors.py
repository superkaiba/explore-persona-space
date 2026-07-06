#!/usr/bin/env python3
# ruff: noqa: RUF003
# Intentional Unicode (λ, Δ) in docstrings/comments.
"""Issue #833 round-5 — batched refit-floor equivalence + joined-cache regressions.

Pins the round-5 throughput fix's permanent invariants:

1. ``make_refit_pair_batched`` consumes the EXACT ``np.random.default_rng(seed)``
   stream ``make_refit_pair`` does (idx_a, idx_b, two child-seed draws per pair)
   — a drift here silently changes every resample.
2. The batched estimator matches the serial one on synthetic well-conditioned
   data (per-pair floor statistics ≤ 1e-6 relative; identical λ picks; zero
   serial fallbacks) — the CPU-cheap twin of the real-cell ``--floors-selftest``.
3. The tiny-n guard routes EVERY resample through the bit-faithful serial
   fallback (n < target_dim + 2) and still matches ``make_refit_pair``.
4. The joined-design cache round-trips exactly and MISSES (never serves stale
   data) on any regime-key drift or a corrupt file (#722 r3 resume-key lesson).

Pure CPU, no HF, seconds-scale.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"
_SRC = Path(__file__).resolve().parent.parent / "src"
for p in (str(_SCRIPTS), str(_SRC)):
    if p not in sys.path:
        sys.path.insert(0, p)

import issue722_fit_M as fitM  # noqa: E402
import issue833_batched_floors as bf  # noqa: E402
from issue722_bootstrap import make_refit_pair  # noqa: E402


def _synthetic(n=48, d=40, g=20, n_fam=4, seed=7):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, d))
    A = rng.normal(size=(d, d)) / np.sqrt(d)
    Y = X @ A + 0.3 * rng.normal(size=(n, d))
    grid = rng.normal(size=(g, d))
    r = rng.normal(size=d)
    r_hat = r / np.linalg.norm(r)
    families = [f"f{i % n_fam}" for i in range(n)]
    return X, Y, grid, r_hat, families


def test_rng_stream_parity():
    """Batched index draws replicate make_refit_pair's stream bit-exactly.

    X's single column IS the row index, so a recording fit_fn recovers the exact
    idx arrays (order included) the serial loop drew — compared against
    ``_draw_pair_indices`` output, child-seed draws and clustered branch included.
    """
    n, n_pairs = 30, 5
    X = np.arange(n, dtype=np.float64)[:, None]
    Y = np.zeros((n, 3))
    grid = np.zeros((4, 1))
    r_hat = np.ones(3) / np.sqrt(3)
    families = [f"f{i % 3}" for i in range(n)]
    captured: list[np.ndarray] = []

    def fit_fn(Xb, Yb, _rng):
        captured.append(Xb[:, 0].astype(int).copy())
        return np.zeros((grid.shape[0], Y.shape[1]))

    make_refit_pair(X, Y, fit_fn, grid, r_hat, families, n_pairs=n_pairs, seed=0)
    expected = bf._draw_pair_indices(n, families, n_pairs, seed=0)
    assert len(captured) == len(expected) == 2 * n_pairs
    for got, want in zip(captured, expected, strict=True):
        assert np.array_equal(got, want), (got, want)


def test_batched_matches_serial_synthetic(monkeypatch):
    """Per-pair floor stats ≤ 1e-6 relative vs serial; identical λ; 0 fallbacks."""
    X, Y, grid, r_hat, families = _synthetic()
    target_dim, n_pairs = 8, 6
    monkeypatch.setattr(fitM, "TARGET_DIM", target_dim)
    sc_s: dict = {}
    stats_s = make_refit_pair(
        X, Y, fitM._refit_ridge_fn(grid), grid, r_hat, families, n_pairs=n_pairs, skip_counter=sc_s
    )
    sc_b: dict = {}
    stats_b, det = bf.make_refit_pair_batched(
        X,
        Y,
        grid,
        r_hat,
        families,
        n_pairs=n_pairs,
        seed=0,
        target_dim=target_dim,
        skip_counter=sc_b,
        chunk_size=5,  # deliberately not divides 2*n_pairs — exercises the ragged tail
        return_details=True,
    )
    assert sc_s == sc_b == {"n_attempted": n_pairs, "n_skipped": 0}
    assert det["n_fallback_serial"] == 0, det["fallback_serial"]
    assert stats_b.shape == stats_s.shape
    # A pair whose two resamples drew the SAME row multiset is float-noise-zero in
    # BOTH engines (batched exactly 0 — weights are order-free; serial ~1e-15 row-
    # order noise): assert those are ≈0 at floor scale, and pure-relative on the rest.
    scale = float(np.abs(stats_s).max())
    degenerate = np.abs(stats_s) <= 1e-9 * scale
    assert np.all(np.abs(stats_b[degenerate]) <= 1e-9 * scale)
    rel = np.abs(stats_b[~degenerate] - stats_s[~degenerate]) / np.abs(stats_s[~degenerate])
    assert rel.max() <= 1e-6, rel
    assert all(li is not None for li in det["lam_idx"])


def test_small_n_all_serial_fallback(monkeypatch):
    """n < target_dim + 2 routes ALL resamples through the serial fallback, exactly."""
    X, Y, grid, r_hat, families = _synthetic(n=12, d=16, g=6, n_fam=3, seed=11)
    target_dim = 16  # > n − 2 → uncertifiable batched boundary
    monkeypatch.setattr(fitM, "TARGET_DIM", target_dim)
    stats_s = make_refit_pair(
        X, Y, fitM._refit_ridge_fn(grid), grid, r_hat, families, n_pairs=4, seed=0
    )
    stats_b, det = bf.make_refit_pair_batched(
        X,
        Y,
        grid,
        r_hat,
        families,
        n_pairs=4,
        seed=0,
        target_dim=target_dim,
        return_details=True,
    )
    assert det["n_fallback_serial"] == 8
    scale = float(np.abs(stats_s).max())
    degenerate = np.abs(stats_s) <= 1e-12 * scale
    assert np.all(np.abs(stats_b[degenerate]) <= 1e-12 * scale)
    rel = np.abs(stats_b[~degenerate] - stats_s[~degenerate]) / np.abs(stats_s[~degenerate])
    assert rel.max() <= 1e-9, rel  # fallback IS the serial computation (assoc-only noise)


def test_rank_deficient_target_all_serial_fallback(monkeypatch):
    """A structurally rank-deficient Y (< target_dim) routes wholesale to serial.

    Mirrors the real m0/shift floors (#833: V0 centered rank ≈ 29 < 64) where the
    registered estimator is algorithm-coupled (gesdd null-basis is part of the
    floor) — the batched path must detect it via the full-data pre-check and
    reproduce serial exactly through the fallback.
    """
    rng = np.random.default_rng(3)
    n, d, rank, target_dim, n_pairs = 40, 32, 5, 8, 3
    B = rng.normal(size=(rank, d))
    Y = rng.normal(size=(n, rank)) @ B  # rank 5 < target_dim 8
    X = rng.normal(size=(n, d))
    grid = rng.normal(size=(6, d))
    r = rng.normal(size=d)
    r_hat = r / np.linalg.norm(r)
    families = [f"f{i % 4}" for i in range(n)]
    assert not bf._full_data_certifiable(Y, target_dim)
    monkeypatch.setattr(fitM, "TARGET_DIM", target_dim)
    stats_s = make_refit_pair(
        X, Y, fitM._refit_ridge_fn(grid), grid, r_hat, families, n_pairs=n_pairs, seed=0
    )
    stats_b, det = bf.make_refit_pair_batched(
        X,
        Y,
        grid,
        r_hat,
        families,
        n_pairs=n_pairs,
        seed=0,
        target_dim=target_dim,
        return_details=True,
    )
    assert det["n_fallback_serial"] == 2 * n_pairs
    scale = float(np.abs(stats_s).max())
    degenerate = np.abs(stats_s) <= 1e-12 * scale
    assert np.all(np.abs(stats_b[degenerate]) <= 1e-12 * scale)
    rel = np.abs(stats_b[~degenerate] - stats_s[~degenerate]) / np.abs(stats_s[~degenerate])
    assert rel.max() <= 1e-9, rel


# ── joined-design cache (issue833_fit_onpolicy) ────────────────────────────────


def _tiny_joined(n=3, d=8, seed=5):
    import issue833_fit_onpolicy as fit833

    rng = np.random.default_rng(seed)
    joined = {k: rng.normal(size=(n, d)) for k in fit833._JOINED_STACK_KEYS}
    joined["families"] = [f"fam{i}" for i in range(n)]
    joined["source_cids"] = [f"s{i}" for i in range(n)]
    joined["target_cids"] = [f"t{i}" for i in range(n)]
    joined["cell_keys"] = [f"em/s{i}__t{i}" for i in range(n)]
    joined["legs"] = [
        fit833.OnPolicyLeg(
            behavior="em",
            source_cid=f"s{i}",
            target_cid=f"t{i}",
            layer=7,
            v_plus_on=joined["Von"][i],
            v0_on=joined["V0on"][i],
            resp_sha256=[f"sha{i}a", f"sha{i}b"],
            resp_sha256_base=[f"base{i}a", f"base{i}b"],
            resp_texts=[f"text {i} a", f"text {i} b"],
            probe_idx=[0, 3],
        )
        for i in range(n)
    ]
    return fit833, joined


def _regime():
    return {"issue": 833, "behavior": "em", "layer": 7, "data_repo_main_sha": "abc123"}


def test_joined_cache_roundtrip(tmp_path):
    fit833, joined = _tiny_joined()
    path = tmp_path / "em_L7.npz"
    fit833.store_joined_cache(path, _regime(), joined)
    out = fit833.load_joined_cache(path, _regime())
    assert out is not None
    for k in fit833._JOINED_STACK_KEYS:
        assert np.array_equal(out[k], joined[k]), k
    for k in fit833._JOINED_STR_KEYS:
        assert out[k] == joined[k], k
    for lo, li in zip(out["legs"], joined["legs"], strict=True):
        assert (lo.behavior, lo.source_cid, lo.target_cid, lo.layer) == (
            li.behavior,
            li.source_cid,
            li.target_cid,
            li.layer,
        )
        assert lo.resp_sha256 == li.resp_sha256
        assert lo.resp_sha256_base == li.resp_sha256_base
        assert lo.resp_texts == li.resp_texts
        assert lo.probe_idx == li.probe_idx
        assert np.array_equal(lo.v_plus_on, li.v_plus_on)
        assert np.array_equal(lo.v0_on, li.v0_on)


def test_joined_cache_regime_mismatch_misses(tmp_path):
    """ANY regime-key drift (incl. the repo main sha) is a MISS — never stale data."""
    fit833, joined = _tiny_joined()
    path = tmp_path / "em_L7.npz"
    fit833.store_joined_cache(path, _regime(), joined)
    drifted = _regime() | {"data_repo_main_sha": "def456"}
    assert fit833.load_joined_cache(path, drifted) is None
    extra_key = _regime() | {"legs_mode": "reextracted"}
    assert fit833.load_joined_cache(path, extra_key) is None


def test_joined_cache_corrupt_and_missing_miss(tmp_path):
    fit833, _ = _tiny_joined()
    missing = tmp_path / "nope.npz"
    assert fit833.load_joined_cache(missing, _regime()) is None
    corrupt = tmp_path / "bad.npz"
    corrupt.write_bytes(b"not an npz")
    assert fit833.load_joined_cache(corrupt, _regime()) is None
