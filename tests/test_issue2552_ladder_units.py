"""Unit pins for scripts/issue2552_ladder.py (#2552 P3 driver).

Covers the smoke-caught `_orth` rank regression (a block residualized to numerical
dust must yield rank 0, never spurious full rank), the Wilson helper, the registered
5-cell verdict lattice, and the within-quintile permutation invariant. No network,
no committed-artifact reads (repo-root safe in sparse worktrees)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
for _p in (str(REPO_ROOT / "scripts"), str(REPO_ROOT / "scripts" / "vendored_2476")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import issue2552_ladder as L  # noqa: E402


def test_orth_spanned_block_is_rank_zero():
    """Regression (#2552 unit-3 smoke): a candidate block fully inside the design span,
    residualized to dust, must NOT come back as full-rank noise."""
    rng = np.random.default_rng(0)
    n = 400
    onehot = np.zeros((n, 5))
    onehot[np.arange(n), rng.integers(0, 5, n)] = 1.0
    q = L._orth(np.column_stack([np.ones(n), onehot]))
    dust = onehot - q @ (q.T @ onehot)  # numerically ~1e-16
    qc = L._orth(dust, scale=float(np.linalg.norm(onehot)))
    assert qc.shape[1] == 0, qc.shape
    # and WITHOUT residualization the same block has rank 4 given an intercept
    q0 = L._orth(np.ones((n, 1)))
    resid = onehot - q0 @ (q0.T @ onehot)
    qc4 = L._orth(resid, scale=float(np.linalg.norm(onehot)))
    assert qc4.shape[1] == 4, qc4.shape


def test_wilson_known_value():
    lo, hi = L._wilson(8, 10)
    assert 0.49 < lo < 0.50 and 0.94 < hi < 0.95, (lo, hi)  # canonical 8/10 Wilson
    lo0, hi0 = L._wilson(0, 0)
    assert np.isnan(lo0) and np.isnan(hi0)


def test_lattice_cells_exhaustive():
    assert L._lattice_cell((0.1, 0.3), (0.05, 0.2)) == "Reproduced"
    assert L._lattice_cell((-0.3, -0.1), (-0.2, -0.05)) == "Reversed"
    assert L._lattice_cell((0.1, 0.3), (-0.2, -0.05)) == "Not reproduced - pt_max dominance"
    assert L._lattice_cell((-0.3, -0.1), (0.05, 0.2)) == "Not reproduced - rep_ta dominance"
    assert L._lattice_cell((-0.1, 0.3), (0.05, 0.2)) == "Inconclusive"
    assert L._lattice_cell((0.1, 0.3), (-0.05, 0.2)) == "Inconclusive"


def test_perm_within_preserves_quintile_multisets():
    rng = np.random.default_rng(1)
    e0 = rng.normal(size=97)
    quint = rng.integers(0, 5, size=97)
    out = L._perm_within(rng, e0, quint, 7)
    assert out.shape == (97, 7)
    for q in range(5):
        idx = quint == q
        ref = np.sort(e0[idx])
        for b in range(7):
            assert np.allclose(np.sort(out[idx, b]), ref)  # a permutation, never a mix


def test_rank01_range_and_monotone():
    x = np.array([3.0, 1.0, 2.0, 10.0])
    r = L._rank01(x)
    assert (r > 0).all() and (r < 1).all()
    assert r[3] == r.max() and r[1] == r.min()
