"""Unit tests for scripts/issue722_skill_over_mean.py (issue #722 amendment).

CPU-only, no HF download, no model load — every test fabricates tiny in-memory
designs. Covers the three load-bearing invariants from plan §10:

1. Zero-skill design → ridge R² ≈ 0 (train-mean intercept gives the mean for free).
2. PRESS λ monotonicity → λ-selection picks the smallest λ when SS_res is
   monotone-decreasing in λ.
3. gesvd fallback exercisable → forcing np.linalg.svd to raise LinAlgError makes
   the torch gesvd fallback run; a truly-singular layer (both backends fail)
   reports NaN + mlp_n_folds_skipped == n with no crash.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS = REPO_ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS))


def _load_module():
    """Import scripts/issue722_skill_over_mean.py by path (not a package)."""
    spec = importlib.util.spec_from_file_location(
        "issue722_skill_over_mean", SCRIPTS / "issue722_skill_over_mean.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def m():
    mod = _load_module()
    mod.i658.DEVICE = "cpu"
    torch.manual_seed(42)
    np.random.seed(42)
    return mod


# ── (a) zero-skill design → ridge R² ≈ 0 ──────────────────────────────────────


def test_zero_skill_independent_target_gives_ridge_skill_near_zero(m):
    """Random c_C → target with INDEPENDENT real variance ⇒ train-mean intercept ⇒ skill ≈ 0.

    The c_C input carries NO information about the target (independent isotropic
    noise), so a correctly-built skill-over-mean ridge cannot beat the
    predict-the-mean baseline. The nested-CV λ pick selects the LARGEST λ (maximal
    shrinkage toward the centered-target zero), the held-out prediction collapses to
    ≈ v̄0_train, and skill ≈ 0 — NOT ≪0 as the intercept-free #658 ridge would give
    (it shrinks toward the raw-space zero, off by the whole anisotropic mean), NOT 1.

    Uses n > d so the design is well-posed and the ridge can actually pick a large
    λ; the d≫n over-parameterized regime is the REAL substrate's regime (n=50,
    d=3584), where the map nonetheless wins because c_C genuinely predicts v0 — that
    is the signal case, not the zero-skill null this test pins.
    """
    rng = np.random.default_rng(0)
    n, d, h = 60, 20, 8
    Xc = rng.standard_normal((n, d))
    Yv = rng.standard_normal((n, h))  # target fully independent of X, real isotropic variance

    skill = m._ridge_skill(Xc, Yv)["skill"]
    assert abs(skill) < 0.05, f"zero-skill design should give ridge skill ≈ 0, got {skill:+.4f}"


def test_skill_over_mean_excludes_skipped_folds(m):
    """A None prediction is excluded from BOTH SS sums (parity), not scored as 0."""
    n, h = 6, 4
    V = np.random.default_rng(1).standard_normal((n, h))
    train_idx = [[j for j in range(n) if j != i] for i in range(n)]
    preds = [None] * n
    # all folds skipped → ss_tot == 0 path → NaN skill, n_folds_used == 0
    res = m._skill_over_mean(preds, V, train_idx)
    assert res["n_folds_used"] == 0
    assert np.isnan(res["skill"])


# ── (b) PRESS λ monotonicity ──────────────────────────────────────────────────


def test_press_lambda_selection_picks_smallest_when_ssres_monotone_decreasing(m):
    """Contrived clean design where larger λ only shrinks a real signal ⇒ smallest λ.

    A low-noise full-rank-in-the-dual design where the smallest λ in the grid gives
    the lowest leave-one-out PRESS error (less shrinkage = better fit when there is
    real linear signal and little noise). The ridge skill path's λ-selection
    (argmin over RIDGE_LAMBDAS of the #658 PRESS MSE) must pick that smallest λ.
    """
    rng = np.random.default_rng(2)
    n, d, h = 25, 10, 3
    Xc = rng.standard_normal((n, d))
    W_true = rng.standard_normal((d, h))
    Yv = Xc @ W_true + 0.001 * rng.standard_normal((n, h))  # strong signal, tiny noise

    # standardize like the ridge path and run the #658 PRESS directly.
    Xt = torch.from_numpy(Xc).double()
    Yt = torch.from_numpy(Yv).double()
    Xn = (Xt - Xt.mean(0)) / (Xt.std(0, correction=0) + 1e-9)
    Yc = Yt - Yt.mean(0)
    mse = m._press_loo_mse_per_lambda(Xn, Yc, m.RIDGE_LAMBDAS)
    # MSE must be monotone non-decreasing in λ for this clean design...
    mse_np = mse.numpy()
    assert np.all(np.diff(mse_np) >= -1e-9), f"PRESS MSE not monotone in λ: {mse_np}"
    # ...so the chosen λ is the smallest grid value.
    chosen = m.RIDGE_LAMBDAS[int(np.argmin(mse_np))]
    assert chosen == min(m.RIDGE_LAMBDAS), f"expected smallest λ, got {chosen}"

    # and the full ridge_skill reports lambda_chosen == that smallest λ.
    lam = m._ridge_skill(Xc, Yv)["lambda_chosen"]
    assert lam == min(m.RIDGE_LAMBDAS), f"ridge_skill lambda_chosen={lam} != {min(m.RIDGE_LAMBDAS)}"


# ── (c) gesvd fallback exercisable + truly-singular → NaN, no crash ───────────


def test_gesvd_fallback_runs_when_numpy_svd_raises(m, monkeypatch):
    """Monkeypatch np.linalg.svd to raise LinAlgError ⇒ torch gesvd fallback returns a basis."""
    rng = np.random.default_rng(3)
    Y = rng.standard_normal((20, 16))

    def _raise(*_a, **_k):
        raise np.linalg.LinAlgError("forced gesdd failure")

    monkeypatch.setattr(np.linalg, "svd", _raise)
    mu, comps, fallback = m._robust_pca_basis(Y, k=8)
    assert fallback is True, "fallback flag should be set when gesdd raised"
    assert comps.shape == (8, 16), f"expected (8, 16) basis from gesvd fallback, got {comps.shape}"
    assert mu.shape == (16,)
    # basis rows are (near-)orthonormal right singular vectors
    gram = comps @ comps.T
    assert np.allclose(gram, np.eye(8), atol=1e-6), "gesvd fallback basis not orthonormal"


def test_truly_singular_layer_reports_nan_and_skipped_no_crash(m, monkeypatch):
    """Both SVD drivers fail ⇒ MLP read is NaN + mlp_n_folds_skipped == n, no exception.

    Forcing BOTH np.linalg.svd AND torch.linalg.svd to raise LinAlgError simulates a
    layer the robust PCA cannot factor at all. _mlp_skill must catch it, report the
    layer as NaN, and set n_folds_skipped == n rather than crashing — the #722
    near-singular crash mode the fallback + skip-on-failure guards against.
    """
    rng = np.random.default_rng(4)
    n, d, h = 12, 40, 16
    Xc = rng.standard_normal((n, d))
    Yv = rng.standard_normal((n, h))

    def _np_raise(*_a, **_k):
        raise np.linalg.LinAlgError("forced gesdd failure")

    def _torch_raise(*_a, **_k):
        raise torch.linalg.LinAlgError("forced gesvd failure")

    monkeypatch.setattr(np.linalg, "svd", _np_raise)
    monkeypatch.setattr(torch.linalg, "svd", _torch_raise)

    res = m._mlp_skill(Xc, Yv)  # must not raise
    assert np.isnan(res["skill"]), (
        f"truly-singular layer should report NaN skill, got {res['skill']}"
    )
    assert res["n_folds_skipped"] == n, (
        f"expected n_folds_skipped == {n}, got {res['n_folds_skipped']}"
    )
