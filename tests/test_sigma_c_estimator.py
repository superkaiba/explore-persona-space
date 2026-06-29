# ruff: noqa: RUF002, RUF003
# Intentional Unicode (Σ, λ, κ, ×, ≤, ⁻¹, ᵀ) in scientific docstrings + asserts.
"""Σ_c broad-corpus estimator for issue-666 (plan §4c, §11, Must-Fix 1).

Theory: `docs/leakage_theory_paper.tex` A7 (L234) ``Σc := E[ccᵀ]`` + the
whitened metric ``W = (Σc + λI)⁻¹`` (L1016-1018, L1352-1356). Plan §4c +
§11 + assumption 14: Σc is estimated off a BROAD background corpus (≥2-5k
contexts), NEVER the n=50 battery (degenerate whitening — rank ≤ 49 at
d=3584). Ridge λ is CV-chosen over the registered grid
``logspace(-6, 2, 17)`` (17 points, λ ∈ [1e-6, 1e2]).

These tests pin: (1) the second-moment estimator exactly matches
(1/N) Σ c_i c_iᵀ; (2) CV-λ selects a value from the registered grid and reports
the conditioning of (Σc + λI); (3) a rank-deficient n=50 input (the forbidden
battery-only case) is detected (rank ≤ 49) AND the regularized inverse is still
finite; (4) the broad-corpus contract — a better-conditioned N≥2000 Σc selects
a SMALLER λ than the rank-deficient N=50 case.

CPU-only; synthetic context vectors; no store, no network, no GPU.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))


class _LazyModule:
    """Proxy that imports the target on first attribute access (TDD).

    The net-new module does NOT exist this round, so the first ``sig.<fn>``
    access inside each test raises ImportError → the test FAILS (not skips).
    A module-level ``importorskip`` was rejected because it skips COLLECTION,
    so the proposed-test count could not be verified by approve-tests.
    """

    def __init__(self, dotted: str):
        self._dotted = dotted

    def __getattr__(self, name):
        import importlib

        return getattr(importlib.import_module(self._dotted), name)


sig = _LazyModule("explore_persona_space.analysis.leakage_predictor")

RTOL = 1e-6
# The registered CV-λ grid (plan §11 / §4c).
LAMBDA_GRID = np.logspace(-6, 2, 17)


def test_lambda_grid_is_the_registered_grid():
    """The module must expose the pre-registered grid (plan §11 reproducibility)."""
    grid = np.asarray(sig.SIGMA_C_LAMBDA_GRID, dtype=float)
    assert grid.shape == (17,)
    assert grid[0] == pytest.approx(1e-6, rel=RTOL)
    assert grid[-1] == pytest.approx(1e2, rel=RTOL)
    assert np.allclose(grid, LAMBDA_GRID, rtol=RTOL)


# ---------------------------------------------------------------------------
# Σc = E[ccᵀ] exactly.
# ---------------------------------------------------------------------------
def test_sigma_c_is_uncentered_second_moment():
    rng = np.random.default_rng(40)
    N, d = 500, 24
    C = rng.standard_normal((N, d))  # (N, d) corpus of context vectors
    Sigma = sig.Sigma_c(C)
    expected = (C.T @ C) / N
    assert Sigma.shape == (d, d)
    assert np.allclose(Sigma, expected, rtol=RTOL, atol=RTOL)
    # Uncentered (NOT covariance): a constant-offset corpus has nonzero Σc.
    Cshift = C + 5.0
    Sigma_shift = sig.Sigma_c(Cshift)
    assert not np.allclose(Sigma_shift, Sigma, rtol=1e-2), (
        "Σc must be the UNCENTERED second moment E[ccᵀ], not the covariance"
    )


# ---------------------------------------------------------------------------
# CV-λ selection on the registered grid + conditioning report.
# ---------------------------------------------------------------------------
def test_cv_lambda_selects_from_registered_grid_and_reports_conditioning():
    rng = np.random.default_rng(41)
    N, d = 2000, 16
    C = rng.standard_normal((N, d))
    result = sig.estimate_sigma_inv(C, lambda_grid=LAMBDA_GRID, seed=0)
    # The chosen λ must be one of the grid points.
    assert any(np.isclose(result.lam, LAMBDA_GRID, rtol=1e-9)), (
        f"chosen λ={result.lam} not on the registered grid"
    )
    # A regularized inverse is returned, finite, symmetric.
    assert result.Sigma_inv.shape == (d, d)
    assert np.all(np.isfinite(result.Sigma_inv))
    assert np.allclose(result.Sigma_inv, result.Sigma_inv.T, rtol=1e-5)
    # Conditioning number κ(Σc + λI) is reported and finite/positive.
    assert np.isfinite(result.cond_number) and result.cond_number > 0


def test_cv_lambda_keeps_ill_conditioned_sigma_under_threshold():
    """On a deliberately ill-conditioned Σc the picked λ keeps κ(Σc+λI) bounded."""
    rng = np.random.default_rng(42)
    d = 20
    # Strongly anisotropic: a few large directions + many tiny ones.
    eig = np.concatenate([np.full(3, 1e3), np.full(d - 3, 1e-6)])
    Q, _ = np.linalg.qr(rng.standard_normal((d, d)))
    Sigma = Q @ np.diag(eig) @ Q.T
    # Many samples drawn from this Σc.
    L = np.linalg.cholesky(Sigma + 1e-12 * np.eye(d))
    C = rng.standard_normal((4000, d)) @ L.T
    result = sig.estimate_sigma_inv(C, lambda_grid=LAMBDA_GRID, seed=1)
    cond_reg = np.linalg.cond(result.Sigma_c + result.lam * np.eye(d))
    assert cond_reg < 1e8, f"regularized conditioning {cond_reg:.2e} not bounded"


# ---------------------------------------------------------------------------
# Rank-deficient n=50 input — the FORBIDDEN battery-only case (d=3584).
# ---------------------------------------------------------------------------
def test_rank_deficient_n50_detected_and_inverse_well_defined():
    """n=50 contexts at d=3584 → raw Σc rank ≤ 49 (the transductive-whitening risk)."""
    rng = np.random.default_rng(43)
    d = 3584
    N = 50  # the battery — far too small for a d×d second moment
    C = rng.standard_normal((N, d)).astype(np.float64)
    Sigma = sig.Sigma_c(C)
    assert Sigma.shape == (d, d)
    rank = np.linalg.matrix_rank(Sigma)
    assert rank <= N - 1, f"n=50 Σc rank {rank} should be ≤ 49 (rank-deficient)"
    # The regularized inverse is still finite (ridge rescues the singular Σc).
    result = sig.estimate_sigma_inv(C, lambda_grid=LAMBDA_GRID, seed=2)
    assert np.all(np.isfinite(result.Sigma_inv)), "ridge inverse must be finite"
    # And the estimator should FLAG this as rank-deficient (n ≤ d).
    assert result.rank_deficient is True, (
        "estimator must flag n=50/d=3584 as rank-deficient (the battery-only risk)"
    )


# ---------------------------------------------------------------------------
# Broad-corpus contract: N≥2000 selects a smaller λ than the rank-deficient n=50.
# ---------------------------------------------------------------------------
def test_broad_corpus_selects_smaller_lambda_than_n50():
    """Better-conditioned broad Σc needs LESS regularization → smaller CV-λ.

    This is the regression test that asserts the broad corpus is the right input
    (plan Must-Fix 1): a well-estimated d×d second moment over many contexts is
    far better-conditioned than the n=50 battery, so the held-out CV picks a
    smaller ridge λ.
    """
    rng = np.random.default_rng(44)
    d = 40
    # Ground-truth anisotropic Σc with a heavy tail (so λ matters).
    eig = np.concatenate([np.linspace(5.0, 1.0, 5), np.linspace(0.05, 1e-3, d - 5)])
    Q, _ = np.linalg.qr(rng.standard_normal((d, d)))
    Sigma_true = Q @ np.diag(eig) @ Q.T
    Lchol = np.linalg.cholesky(Sigma_true + 1e-12 * np.eye(d))

    def draw(n, seed):
        r = np.random.default_rng(seed)
        return r.standard_normal((n, d)) @ Lchol.T

    C_small = draw(40, 100)  # n < d → rank-deficient
    C_broad = draw(4000, 101)  # n >> d → well-estimated

    res_small = sig.estimate_sigma_inv(C_small, lambda_grid=LAMBDA_GRID, seed=7)
    res_broad = sig.estimate_sigma_inv(C_broad, lambda_grid=LAMBDA_GRID, seed=7)

    assert res_broad.lam < res_small.lam, (
        f"broad-corpus λ ({res_broad.lam:.2e}) should be < n<d λ ({res_small.lam:.2e}); "
        "a better-conditioned Σc needs less regularization"
    )
    # And the broad-corpus Σc is markedly better-conditioned.
    cond_small = np.linalg.cond(res_small.Sigma_c + res_small.lam * np.eye(d))
    cond_broad = np.linalg.cond(res_broad.Sigma_c + res_broad.lam * np.eye(d))
    assert cond_broad < cond_small


def test_battery_sigma_is_diagnostic_only_flag():
    """The estimator marks an n=50-battery Σc as a diagnostic, never headline.

    Plan §4c / §11: the n=50-battery Σc is a smoke/diagnostic FALLBACK only; the
    module must carry a structural flag so a caller cannot silently feed the
    battery Σc into a headline number (the H2-fatal in-sample-d.o.f. confound).
    """
    rng = np.random.default_rng(45)
    d = 3584
    C = rng.standard_normal((50, d))
    res = sig.estimate_sigma_inv(C, lambda_grid=LAMBDA_GRID, seed=3, corpus_kind="battery")
    assert res.headline_eligible is False, (
        "an n=50 battery Σc must be flagged headline-INELIGIBLE (diagnostic only)"
    )
    # A broad corpus is headline-eligible.
    C_broad = rng.standard_normal((3000, 40))
    res_b = sig.estimate_sigma_inv(C_broad, lambda_grid=LAMBDA_GRID, seed=3, corpus_kind="broad")
    assert res_b.headline_eligible is True
