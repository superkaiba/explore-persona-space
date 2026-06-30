"""Unit tests for the issue #761 paired Δrho estimator (plan §4.4-3 / §6.4 / §8 risk (g)).

The headline Δrho CI is the PAIRED context-resample bootstrap, NOT the disjoint-arm
independent estimator. These tests pin the three properties the plan's risk-(g)
mitigation requires:

  (1) On perfectly-correlated synthetic arms (arm_corr=1) the paired CI is TIGHTER
      than the independent estimator on the SAME single-arm draws.
  (2) The SAME context-index set drives BOTH arms per draw (the mechanistic
      paired-resample check).
  (3) On disjoint-arm synthetic data (arm_corr=0) paired ≈ independent.

These exercise the estimator directly on synthetic ``(N, n_layers, H)`` arms — no
GPU, no HF, no reuse data.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"
sys.path.insert(0, str(SCRIPTS))

from issue761_paired_bootstrap import (  # noqa: E402
    _arm_rho_on_subset,
    _independent_delta_rho_ci,
    _paired_delta_rho_ci,
)


def _ci_width(ci: list[float]) -> float:
    return ci[1] - ci[0]


def _make_arm(n: int, n_layers: int, h: int, y: np.ndarray, signal: float, rng) -> np.ndarray:
    """A synthetic v0 cube (N, n_layers, H) whose layer 0 carries ``signal``*y + noise."""
    X = rng.standard_normal((n, n_layers, h)).astype(np.float64)
    # plant a y-correlated signal in every layer's first few PCs so ridge can read it
    X[:, :, 0] += signal * y[:, None]
    X[:, :, 1] += 0.5 * signal * y[:, None]
    return X


def _single_arm_draws(X, y, *, n_boot, seed):
    """The single-arm cluster-bootstrap draws the independent estimator consumes."""
    n = X.shape[0]
    rng = np.random.default_rng(seed)
    draws = []
    attempts = 0
    while len(draws) < n_boot and attempts < 50 * n_boot:
        attempts += 1
        idx = rng.integers(0, n, size=n)
        rho = _arm_rho_on_subset(X, y, idx)
        if rho is not None:
            draws.append(rho)
    return draws


def test_paired_ci_tighter_on_correlated_arms():
    """(1) arm_corr≈1 → paired CI strictly tighter than the independent CI."""
    rng = np.random.default_rng(0)
    n, n_layers, h = 50, 4, 30
    y = rng.standard_normal(n)
    # two arms reading the SAME signal → strongly positively-correlated per-draw rho
    matched = _make_arm(n, n_layers, h, y, signal=2.0, rng=np.random.default_rng(1))
    mismatched = matched + 0.05 * rng.standard_normal(matched.shape)

    paired = _paired_delta_rho_ci(matched, mismatched, y, n_boot=120, seed=761)
    # independent estimator on the SAME single-arm draws
    m_draws = _single_arm_draws(matched, y, n_boot=120, seed=11)
    mm_draws = _single_arm_draws(mismatched, y, n_boot=120, seed=12)
    indep = _independent_delta_rho_ci(m_draws, mm_draws, seed=99)

    assert _ci_width(paired["ci95"]) < _ci_width(indep["ci95"]), (
        f"paired CI width {_ci_width(paired['ci95']):.4f} should be < independent "
        f"{_ci_width(indep['ci95']):.4f} on correlated arms"
    )


def test_same_index_set_drives_both_arms(monkeypatch):
    """(2) mechanistic — both arms refit on the SAME resampled context indices per draw."""
    rng = np.random.default_rng(2)
    n, n_layers, h = 40, 3, 20
    y = rng.standard_normal(n)
    matched = _make_arm(n, n_layers, h, y, signal=1.5, rng=np.random.default_rng(3))
    mismatched = _make_arm(n, n_layers, h, y, signal=1.5, rng=np.random.default_rng(4))

    import issue761_paired_bootstrap as mod

    seen: list[np.ndarray] = []
    real = mod._arm_rho_on_subset

    def _spy(X, yy, idx):
        seen.append(np.asarray(idx).copy())
        return real(X, yy, idx)

    monkeypatch.setattr(mod, "_arm_rho_on_subset", _spy)
    _paired_delta_rho_ci(matched, mismatched, y, n_boot=20, seed=761)

    # _arm_rho_on_subset is called twice per draw (matched then mismatched) with the
    # SAME idx; consecutive pairs must be identical.
    assert len(seen) >= 40, len(seen)
    for k in range(0, len(seen) - 1, 2):
        np.testing.assert_array_equal(
            seen[k], seen[k + 1], err_msg=f"draw {k // 2}: arms saw DIFFERENT index sets"
        )


def test_paired_approx_independent_on_disjoint_arms():
    """(3) arm_corr≈0 → paired CI ≈ independent CI (independence is the degenerate case)."""
    rng = np.random.default_rng(5)
    n, n_layers, h = 50, 4, 30
    y = rng.standard_normal(n)
    # two arms reading INDEPENDENT signals → near-zero per-draw rho correlation
    matched = _make_arm(n, n_layers, h, y, signal=2.0, rng=np.random.default_rng(6))
    mismatched = _make_arm(n, n_layers, h, y, signal=2.0, rng=np.random.default_rng(7))
    # decorrelate: replace the planted signal in mismatched with an independent one
    y2 = rng.standard_normal(n)
    mismatched[:, :, 0] = rng.standard_normal((n, n_layers)) + 2.0 * y2[:, None]

    paired = _paired_delta_rho_ci(matched, mismatched, y, n_boot=150, seed=761)
    m_draws = _single_arm_draws(matched, y, n_boot=150, seed=21)
    mm_draws = _single_arm_draws(mismatched, y, n_boot=150, seed=22)
    indep = _independent_delta_rho_ci(m_draws, mm_draws, seed=99)

    wp = _ci_width(paired["ci95"])
    wi = _ci_width(indep["ci95"])
    # on disjoint arms the two CIs should be within ~40% of each other (independence
    # is the regime where the paired estimator reduces to the independent one)
    assert wp == pytest.approx(wi, rel=0.40), (
        f"paired width {wp:.4f} vs independent {wi:.4f} should be close on disjoint arms"
    )


def test_vectorized_ridge_matches_inherited_serial():
    """The batched LOCO ridge is bit-equal to the inherited serial _ridge_predict_loco.

    The paired bootstrap replaces the ~1s/call serial helper with a batched one for
    tractability (plan §9 compute-deviation); this pins that the swap did NOT change
    the numbers. Trips if the batched standardization / nested-lambda / dual-solve
    drifts from the oracle.
    """
    import issue761_common as common

    for seed in (0, 1, 5, 13):
        max_abs = common._assert_vectorized_ridge_exactness(seed=seed)
        assert max_abs <= 1e-6, (seed, max_abs)


def test_all_layers_batched_matches_per_layer():
    """`_all_layers_loco_preds` (layer axis batched) == per-layer `_vectorized_ridge_loco_preds`."""
    import issue761_common as common

    rng = np.random.default_rng(3)
    n, n_layers, h = 40, 5, 25
    y = rng.standard_normal(n)
    X = rng.standard_normal((n, n_layers, h))
    X[:, :, 0] += 1.5 * y[:, None]
    batched = common._all_layers_loco_preds(X, y, 8)  # (n_layers, n)
    for li in range(n_layers):
        single = common._vectorized_ridge_loco_preds(
            common._pca_reduce(X[:, li, :], 8), y, list(common.RIDGE_LAMBDAS)
        )
        np.testing.assert_allclose(batched[li], single, atol=1e-9, err_msg=f"layer {li}")


def test_paired_point_delta_and_overlap_keys():
    """The estimator returns the contract keys (ci95, point_delta, draws, null_overlap)."""
    rng = np.random.default_rng(8)
    n, n_layers, h = 30, 2, 15
    y = rng.standard_normal(n)
    matched = _make_arm(n, n_layers, h, y, signal=2.0, rng=np.random.default_rng(9))
    mismatched = _make_arm(n, n_layers, h, y, signal=0.3, rng=np.random.default_rng(10))
    out = _paired_delta_rho_ci(matched, mismatched, y, n_boot=50, seed=761)
    assert set(out) >= {"ci95", "point_delta", "draws", "n_boot", "null_overlap"}
    assert len(out["draws"]) == 50
    assert out["ci95"][0] <= out["ci95"][1]
