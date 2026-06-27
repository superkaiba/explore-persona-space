# ruff: noqa: RUF001, RUF002, RUF003
# Intentional Unicode (ρ, λ, ×, ≤, Δ) in scientific docstrings + assert messages.
"""Exactness regression for the #658 predictor-fit GPU/batched performance rewrite.

The recovery-mode rewrite (2026-06-27) replaced the A3.4 ridge nested-CV LOCO
fit — previously a per-(inner fold × λ) primal refit, ``np.linalg.solve(XᵀX+λI,
XᵀY)``, the O(D³) path that ran ~40h on CPU with no output — with the EXACT
closed-form dual/PRESS leave-one-out identity (one eigendecomposition of the N×N
Gram, vectorized over the λ grid). "Exact" is the gate: the reported held-out
LOCO Spearman ρ for A3.4 / A3.5 / the chain ρ MUST NOT MOVE.

These tests pin that invariant so a future refactor of the fast path can never
silently drift the DV away from the primal-refit oracle and stay green:

- the fast ``_ridge_predict_loco`` reproduces the primal-refit
  ``_ridge_predict_loco_refit`` to <= 1e-6 in both predictions AND ρ;
- the in-script ``_assert_ridge_exactness`` gate (run at every startup) passes
  and reports a delta within tolerance.

CPU-only; runs in ~1s. No GPU, no store, no network.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
from scipy.stats import spearmanr

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))
sys.path.insert(0, str(REPO_ROOT / "src"))

fit = pytest.importorskip("issue658_fit_predictors")


def _synthetic(seed: int, n: int = 16, d: int = 50, p: int = 3):
    """Low-rank-signal + noise (X, Y) so ridge has real structure to fit."""
    rng = np.random.default_rng(seed)
    z = rng.standard_normal((n, 4))
    W = rng.standard_normal((4, d))
    X = z @ W + 0.1 * rng.standard_normal((n, d))
    B = rng.standard_normal((d, p))
    Y = X @ B * 0.05 + 0.1 * rng.standard_normal((n, p))
    return X, Y


@pytest.mark.parametrize("seed", [0, 1, 7])
def test_dual_press_loco_matches_primal_refit_predictions(seed):
    """The fast dual/PRESS LOCO ridge == the primal-refit oracle, <= 1e-6 on preds.

    This is the core exactness claim: the closed-form leave-one-out identity is
    mathematically the same fit as refitting ridge on each (N-1)-row subset, so
    every held-out prediction must agree to numerical precision.
    """
    fit.DEVICE = "cpu"
    X, Y = _synthetic(seed)
    lambdas = [1e-1, 1.0, 10.0, 100.0]
    fast = fit._ridge_predict_loco(X, Y, lambdas)
    ref = fit._ridge_predict_loco_refit(X, Y, lambdas)
    max_abs = float(np.max(np.abs(fast - ref)))
    assert max_abs <= 1e-6, f"dual PRESS LOCO drifted from primal refit: max|Δpred|={max_abs:.3e}"


@pytest.mark.parametrize("seed", [0, 1, 7])
def test_dual_press_loco_matches_primal_refit_rho(seed):
    """The REPORTED statistic (per-output held-out Spearman ρ) is unchanged."""
    fit.DEVICE = "cpu"
    X, Y = _synthetic(seed)
    lambdas = [1e-1, 1.0, 10.0, 100.0]
    fast = fit._ridge_predict_loco(X, Y, lambdas)
    ref = fit._ridge_predict_loco_refit(X, Y, lambdas)
    for k in range(Y.shape[1]):
        rf = spearmanr(fast[:, k], Y[:, k]).correlation
        rr = spearmanr(ref[:, k], Y[:, k]).correlation
        if np.isnan(rf) and np.isnan(rr):
            continue
        assert abs(float(rf - rr)) <= 1e-6, f"output {k}: ρ drifted (fast {rf} vs refit {rr})"


def test_assert_ridge_exactness_gate_passes():
    """The in-script startup gate ``_assert_ridge_exactness`` passes within tol.

    main() runs this at every startup; a failure aborts the run loud. Pin it here
    so the gate itself can never be quietly weakened (e.g. tolerance loosened, or
    the oracle swapped for the fast path so it trivially compares to itself)."""
    fit.DEVICE = "cpu"
    res = fit._assert_ridge_exactness()
    assert res["tol"] == 1e-6
    assert res["max_abs_pred_delta"] <= res["tol"]
    assert res["max_rho_delta"] <= res["tol"]


def test_refit_oracle_is_distinct_from_fast_path():
    """Guard against the gate degenerating: the oracle must NOT call the fast path.

    ``_assert_ridge_exactness`` is only meaningful if the reference really is the
    independent primal-refit implementation. A direct smoke that the oracle uses
    the primal ``np.linalg.solve`` solve (not the dual one) — the two functions
    are different objects with different source.
    """
    import inspect

    ref_src = inspect.getsource(fit._ridge_predict_loco_refit)
    assert "_ridge_solve" in ref_src, "the exactness oracle must use the primal _ridge_solve refit"
    fast_src = inspect.getsource(fit._ridge_predict_loco)
    assert "_press_loo_mse_per_lambda" in fast_src, "the fast path must use the PRESS closed form"
    assert "_ridge_dual_weights" in fast_src, "the fast path must use the dual/Woodbury solve"
