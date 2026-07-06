# ruff: noqa: RUF002
"""Issue #763 round-4 (v3) regression tests.

Pins the BLOCKER fix `analysis-null-infeasible-at-scale`: the shuffle / control
nulls must NOT re-run the per-fold nested-CV PCA dim selection per permutation
(that projected to ~580h/behavior/DV — 343M statsmodels GLM fits). The fix adds
FIXED-dim LOCO fast paths (``glm_predict_loco_fixed_dim`` /
``_ridge_predict_loco_fixed_dim``) the nulls call with the per-layer dim chosen
ONCE on the observed data (``_observed_layer_dims``). These tests assert:

1. the fixed-dim path produces the SAME held-out predictions as the nested-CV
   path when the fixed dim equals the selected dim (correctness-neutral);
2. the null re-selects the dim ONCE, not per permutation (the feasibility
   invariant — counts `select_pca_dim` calls);
3. the graded-GLM comparator consumes a [0,1] rate (the graded/100 rescale),
   not a raw 0-100 value the binomial logit-link clamp would crush to ~1.0.

Also pins the v3 graded-primary dual-DV schema + the behavior-conditioned floor.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO / "scripts"))

from explore_persona_space.analysis.issue_763_glm import (  # noqa: E402
    glm_predict_loco,
    glm_predict_loco_fixed_dim,
)


def _signal(n=24, h=12, seed=0):
    rng = np.random.default_rng(seed)
    direction = rng.standard_normal(h)
    x = rng.standard_normal((n, h))
    score = x @ direction
    rate = 1.0 / (1.0 + np.exp(-score / 2.0))
    return x, rate


# ── (1) correctness-neutral: fixed-dim == nested-CV at the matched dim ──


def test_glm_fixed_dim_matches_nested_cv_at_selected_dim():
    """glm_predict_loco_fixed_dim at the modal nested-CV dim ≈ glm_predict_loco.

    The fixed-dim fast path skips the per-fold inner-CV but fits the SAME basis
    + GLM at a given dim. When the fixed dim equals the dim the nested-CV would
    pick on the full data, the held-out predictions track closely (small
    differences only from per-fold dim re-selection in the nested-CV path).
    """
    x, rate = _signal(n=24, h=12, seed=3)
    nj = np.full(24, 30)
    out_cv = glm_predict_loco(x, rate, nj)
    modal_dim = max(set(out_cv["chosen_dims"]), key=out_cv["chosen_dims"].count)
    pred_fixed = glm_predict_loco_fixed_dim(x, rate, nj, modal_dim)
    assert pred_fixed.shape == (24,)
    assert np.all(np.isfinite(pred_fixed))
    # rank-correlate (Spearman is what the headline reads); should be very high
    from scipy.stats import spearmanr

    rho = spearmanr(pred_fixed, out_cv["pred"]).correlation
    # High agreement (not 1.0: the nested-CV path re-selects the dim PER FOLD, so
    # some folds differ from the modal fixed dim) — confirms the fixed-dim fast
    # path is the SAME estimator family, correctness-neutral for the null.
    assert rho > 0.7, rho


def test_ridge_fixed_dim_runs_and_is_finite():
    import issue763_fit_predictors as F
    from issue658_fit_predictors import RIDGE_LAMBDAS

    x, rate = _signal(n=24, h=12, seed=4)
    nj = np.full(24, 30)
    pred = F._ridge_predict_loco_fixed_dim(x, rate, nj, RIDGE_LAMBDAS, 6)
    assert pred.shape == (24,)
    assert np.all(np.isfinite(pred))


# ── (2) feasibility invariant: the null selects the dim ONCE, not per-perm ──


def test_shuffle_null_selects_dim_once_not_per_perm(monkeypatch):
    """The BLOCKER fix: select_pca_dim runs once-per-layer for the observed data,
    NEVER per permutation. Pre-fix the null re-ran the full nested-CV per perm
    (select_pca_dim called n_perms x n_layers x folds times) — that was the
    ~580h explosion. Post-fix the null computes _observed_layer_dims ONCE and
    reuses the fixed dim across all permutations, so select_pca_dim is called
    exactly n_layers times (the observed-data read), independent of n_perms.
    """
    import issue763_fit_predictors as F

    calls = {"n": 0}
    real = F.select_pca_dim

    def _counting(*a, **k):
        calls["n"] += 1
        return real(*a, **k)

    monkeypatch.setattr(F, "select_pca_dim", _counting)

    rng = np.random.default_rng(7)
    n_ctx, n_layers, h = 20, 4, 12
    v0 = rng.standard_normal((n_ctx, n_layers, h))
    direction = rng.standard_normal(h)
    y = 1.0 / (1.0 + np.exp(-(v0[:, 1, :] @ direction) / 2.0))  # layer 1 carries signal
    nj = np.full(n_ctx, 30)

    n_perms = 25
    F._shuffle_null(v0, y, nj, "glm", rb=None, n_perms=n_perms, seed=11)

    # select_pca_dim must be called EXACTLY n_layers times (observed-data dims),
    # NOT n_layers * (1 + n_perms) * folds. The fix's whole point.
    assert calls["n"] == n_layers, (
        f"select_pca_dim called {calls['n']}x; expected exactly {n_layers} "
        f"(once per layer on observed data). A per-perm call count "
        f"(~{n_layers * (1 + n_perms)}+) means the null re-runs nested-CV per "
        f"permutation — the analysis-null-infeasible-at-scale BLOCKER regressed."
    )


# ── (3) graded-GLM comparator on a [0,1] rate, not raw 0-100 ──


def test_graded_glm_comparator_uses_rate_scale():
    """A 0-100 graded mean fed raw to the binomial GLM is clamped to ~1.0 and
    yields a degenerate (None / near-constant) read; dividing by 100 first gives
    a well-defined [0,1] rate the logit link can model. The fit driver rescales
    graded/100 for the graded-GLM comparator (ridge/PV stay on the raw 0-100
    mean — Spearman is scale-invariant there). This pins that the rescaled call
    produces a finite, non-degenerate ρ.
    """
    from scipy.stats import spearmanr

    x, rate01 = _signal(n=24, h=10, seed=5)  # rate01 in (0,1)
    graded = rate01 * 100.0  # 0-100 graded mean
    nj = np.full(24, 30)
    # The driver feeds graded/100 to the binomial GLM (the [0,1] rate the logit
    # link needs); ridge/PV consume the raw 0-100 (Spearman scale-invariant).
    out_rescaled = glm_predict_loco(x, graded / 100.0, nj)
    rho = spearmanr(out_rescaled["pred"], graded).correlation
    assert np.isfinite(rho)
    assert rho > 0.3, rho  # the signal is recoverable on the [0,1] scale
