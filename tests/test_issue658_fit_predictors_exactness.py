# ruff: noqa: RUF001, RUF002, RUF003
# Intentional Unicode (ρ, λ, ×, ≤, Δ) in scientific docstrings + assert messages.
"""Exactness regression for the #658 predictor-fit GPU/batched performance rewrite.

The recovery-mode rewrite (2026-06-27) replaced the A3.4 ridge nested-CV LOCO
fit — previously a per-(inner fold × λ) primal refit, ``np.linalg.solve(XᵀX+λI,
XᵀY)``, the O(D³) path that ran ~40h on CPU with no output — with the EXACT
closed-form dual/PRESS leave-one-out identity (one eigendecomposition of the N×N
Gram, vectorized over the λ grid). "Exact" is the gate: the reported held-out
LOCO Spearman ρ for A3.4 / A3.5 / the chain ρ MUST NOT MOVE.

The MLP halves (A3.2 single-output, A3.5 multi-output gap) were ALSO re-batched
onto a vmapped ensemble on ``--device``. The batched MLP is not exact to machine
precision (batched GEMM vs per-net GEMV reduction order), but it must reproduce
the OLD serial loop to <= 1e-6 — AND the multi-output gap path must reproduce the
serial RESEED-PER-DIM init stream (the round-2 finding: an all-at-once seed drifts
dims 1+ by ~0.38; the tiled init fixes it).

These tests pin both invariants so a future refactor of either fast path can never
silently drift the DV away from the serial oracle and stay green:

- the fast ``_ridge_predict_loco`` reproduces the primal-refit
  ``_ridge_predict_loco_refit`` to <= 1e-6 in both predictions AND ρ;
- the batched ``_fit_mlp_loco`` (A3.2) + ``_fit_mlp_ensemble_loco`` gap path
  (A3.5) reproduce ``_fit_mlp_loco_serial_reference`` to <= 1e-6;
- the in-script ``_assert_ridge_exactness`` + ``_assert_mlp_exactness`` startup
  gates pass and report deltas within tolerance;
- the per-cell param-hash invalidates stale checkpoint cells on a hyperparameter
  change (the resume-into-reused-out-dir stale-serve fix).

CPU-only; runs in a few seconds. No GPU, no store, no network.
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


# ── MLP exactness (A3.2 single-output + A3.5 multi-output gap) ──────────────────


def _mlp_synthetic(seed: int, n: int = 14, d: int = 30, p: int = 6):
    """Low-rank-signal + noise (X, Y) fp32 for the MLP equivalence checks."""
    rng = np.random.default_rng(seed)
    z = rng.standard_normal((n, 4))
    W = rng.standard_normal((4, d))
    X = (z @ W + 0.1 * rng.standard_normal((n, d))).astype(np.float32)
    B = rng.standard_normal((d, p))
    Y = (X @ B * 0.05 + 0.1 * rng.standard_normal((n, p))).astype(np.float32)
    return X, Y


@pytest.mark.parametrize("seed", [0, 3])
def test_batched_single_output_mlp_matches_serial(seed, monkeypatch):
    """The A3.2 path (``_fit_mlp_loco``) reproduces the serial reference <= 1e-6.

    The batched-vmap single-output LOCO MLP must match the OLD per-fold serial loop
    (``_fit_mlp_loco_serial_reference``) — same arch, AdamW, epochs, per-fold
    standardization, and per-fold init stream. (~3.6e-7 on CPU, reduction-order.)
    """
    monkeypatch.setattr(fit, "DEVICE", "cpu")
    monkeypatch.setattr(fit, "MLP_MAX_EPOCHS", 25)
    X, Y = _mlp_synthetic(seed)
    ser = fit._fit_mlp_loco_serial_reference(X, Y[:, 0])
    bat = fit._fit_mlp_loco(X, Y[:, 0])
    max_abs = float(np.max(np.abs(ser - bat)))
    assert max_abs <= 1e-6, f"batched single-output MLP drifted from serial: max|Δ|={max_abs:.3e}"


@pytest.mark.parametrize("seed", [0, 3])
def test_batched_gap_mlp_matches_serial_reseed_per_dim(seed, monkeypatch):
    """The A3.5 gap path reproduces the serial RESEED-PER-DIM reference <= 1e-6.

    The OLD gap MLP called ``_fit_mlp_loco(Xc, Yv[:, k])`` once per output dim, and
    each call re-seeds ``torch.manual_seed(658)`` — so every dim reuses the SAME n
    per-fold inits. The batched ensemble must reproduce that by TILING the n
    per-fold inits across the gap_dim blocks (see ``_fit_mlp_ensemble_loco``).
    Without the tile, dims 1+ diverge ~0.38 (the round-2 finding this guards).
    """
    monkeypatch.setattr(fit, "DEVICE", "cpu")
    monkeypatch.setattr(fit, "MLP_MAX_EPOCHS", 25)
    X, Y = _mlp_synthetic(seed)
    gap = 4
    ser = np.stack([fit._fit_mlp_loco_serial_reference(X, Y[:, k]) for k in range(gap)], axis=1)
    bat = fit._fit_mlp_ensemble_loco(X, Y, target_idx=list(range(gap)), seed=658)
    max_abs = float(np.max(np.abs(ser - bat)))
    assert max_abs <= 1e-6, (
        f"batched gap MLP drifted from serial reseed-per-dim: max|Δ|={max_abs:.3e}. "
        "A >1e-2 delta means the per-dim init tile in _fit_mlp_ensemble_loco regressed "
        "(dims 1+ drawing from a shifted RNG stream)."
    )


def test_assert_mlp_exactness_gate_passes():
    """The in-script startup gate ``_assert_mlp_exactness`` passes within tol.

    main() runs this at every startup alongside the ridge gate; a failure aborts
    the run loud. Pin it so the gate cannot be quietly weakened. Both the single-
    output and the tiled multi-output gap deltas must be within tolerance.
    """
    res = fit._assert_mlp_exactness()
    assert res["tol"] == 1e-6
    assert res["single_delta"] <= res["tol"], res
    assert res["multi_delta"] <= res["tol"], res


def test_mlp_oracle_is_distinct_from_batched_path():
    """The MLP exactness oracle must be the serial loop, not the batched path.

    ``_assert_mlp_exactness`` is only meaningful if the reference is the independent
    per-fold serial implementation. Smoke that the oracle uses a per-fold Python
    loop with a fresh per-fold optimizer (the OLD shape), and the batched path uses
    the vmapped ensemble.
    """
    import inspect

    oracle_src = inspect.getsource(fit._fit_mlp_loco_serial_reference)
    assert "for i in range(n)" in oracle_src, "the MLP oracle must be the per-fold serial loop"
    assert "torch.optim.AdamW(net.parameters()" in oracle_src, (
        "the MLP oracle must build a fresh per-fold optimizer (the old serial shape)"
    )
    batched_src = inspect.getsource(fit._fit_mlp_ensemble_loco)
    assert "vmap" in batched_src, "the batched MLP path must use torch.func.vmap"


def test_param_hash_invalidates_stale_cells(tmp_path, monkeypatch):
    """A checkpoint cell written under one set of hyperparams is STALE under another.

    Guards the resume-into-reused-out-dir stale-serve fix: the per-cell param hash
    must change when a load-bearing constant (λ grid / MLP epochs / feat_dim /
    A35_MLP_TARGET_DIM / bootstrap) changes, and ``_load_cell`` must return None
    (recompute) on a hash mismatch while serving a matching cell.
    """
    monkeypatch.setattr(fit, "A35_MLP_TARGET_DIM", 64)
    ph1 = fit._param_hash("a34a35", feat_dim=0)
    fit._save_cell(tmp_path, "a34a35", "meanprompt__L0", {"per_layer": {"x": 1}}, param_hash=ph1)
    # same params -> served
    assert fit._load_cell(tmp_path, "a34a35", "meanprompt__L0", param_hash=ph1) is not None
    # change A35_MLP_TARGET_DIM -> different hash -> stale -> None
    monkeypatch.setattr(fit, "A35_MLP_TARGET_DIM", 16)
    ph2 = fit._param_hash("a34a35", feat_dim=0)
    assert ph2 != ph1
    assert fit._load_cell(tmp_path, "a34a35", "meanprompt__L0", param_hash=ph2) is None
    # a32 hash is independent of A35_MLP_TARGET_DIM (phase-scoped)
    monkeypatch.setattr(fit, "A35_MLP_TARGET_DIM", 64)
    a = fit._param_hash("a32", feat_dim=0)
    monkeypatch.setattr(fit, "A35_MLP_TARGET_DIM", 16)
    assert fit._param_hash("a32", feat_dim=0) == a, "a32 hash must NOT depend on A35_MLP_TARGET_DIM"
