# ruff: noqa: RUF001, RUF002  -- math glyphs match the #841 script + plan notation
"""Regression pins for the issue #841 affine-ridge BLOCKER fix (round-1 code review).

The Stage-0/Stage-1 ridge Δ-map (`fit_ridge_split`) is AFFINE with an intercept
(plan §4.3 `Δ̂ = A_ℓ h_ℓ + c_ℓ`): Δ carries a large nonzero mean, so a no-intercept
ridge forced through the origin systematically underfits it. These tests trip if a
future edit strips the bias back to the origin-forced form:

- `test_affine_ridge_recovers_nonzero_mean_target` — on a mean-dominated target the
  affine fit's stored `bias` recovers the injected offset and the identity-relative
  R² stays high; a no-intercept fit (weights are zero-mean on train, so it cannot
  reach the offset) collapses R² toward the offset's share of the total energy. The
  test asserts BOTH that the affine fit passes AND that the no-bias counterfactual
  fails, so it fails pre-fix and passes post-fix.
- `test_predict_zero_identity_relative_r2_is_exactly_zero` — the predict-zero null
  the atlas identity class relies on scores exactly 0.
"""

import numpy as np
import torch

from explore_persona_space.experiments.issue_841 import maps


def _mean_dominated_split(seed: int = 12345):
    """Synthetic (X, Δ) where Δ = small linear signal + a large constant offset.

    The offset dominates Σ‖Δ‖², so recovering it (via the intercept) is what carries
    the identity-relative R². Returns (x_train, y_train, x_eval, y_eval, offset).
    """
    rng = np.random.default_rng(seed)
    d_in, p_out, n_train, n_eval = 16, 8, 200, 50
    A = rng.standard_normal((p_out, d_in)) * 0.05  # small linear signal
    offset = np.full(p_out, 5.0, dtype=np.float64)  # large nonzero Δ mean

    def gen(n):
        x = rng.standard_normal((n, d_in))
        y = x @ A.T + offset + rng.standard_normal((n, p_out)) * 0.01
        return x.astype(np.float64), y.astype(np.float64)

    x_train, y_train = gen(n_train)
    x_eval, y_eval = gen(n_eval)
    return x_train, y_train, x_eval, y_eval, offset


def test_affine_ridge_recovers_nonzero_mean_target():
    x_train, y_train, x_eval, y_eval, offset = _mean_dominated_split()

    eval_pred, rmap = maps.fit_ridge_split(x_train, y_train, x_eval, sigma=1.0, device="cpu")

    # (1) the stored intercept recovers the injected offset (= train mean of Δ)
    assert np.allclose(rmap.bias.numpy(), y_train.mean(0), atol=1e-4), (
        "RidgeMap.bias must equal the train-mean of Δ (the affine intercept c_ℓ)"
    )
    assert np.allclose(rmap.bias.numpy(), offset, atol=0.05), (
        f"bias {rmap.bias.numpy().mean():.3f} should recover offset {offset.mean():.3f}"
    )

    # (2) eval predictions recover the offset-dominated target: high identity-rel R²
    r2_affine = maps.identity_relative_r2(eval_pred, y_eval)
    assert r2_affine > 0.95, f"affine ridge R²_id {r2_affine:.4f} should be near 1"

    # (2b) rmap.apply reproduces the eval prediction (the transport-side path)
    applied = rmap.apply(torch.from_numpy(x_eval).float()).numpy()
    assert np.allclose(applied, eval_pred, atol=1e-3), "apply() must match eval_pred"

    # (3) the no-intercept counterfactual FAILS on the same data — this is what the
    #     BLOCKER fix corrects. Weights are zero-mean on train ⇒ can't reach the
    #     offset ⇒ R²_id collapses toward the offset's share of Σ‖Δ‖².
    r2_nobias = maps.identity_relative_r2(eval_pred - rmap.bias.numpy(), y_eval)
    assert r2_nobias < 0.5, (
        f"origin-forced (no-bias) R²_id {r2_nobias:.4f} must collapse on a "
        "mean-dominated target — proves the bias is load-bearing"
    )


def test_predict_zero_identity_relative_r2_is_exactly_zero():
    _, _, _, y_eval, _ = _mean_dominated_split()
    r2_zero = maps.identity_relative_r2(np.zeros_like(y_eval), y_eval)
    assert r2_zero == 0.0, f"predict-zero identity-relative R² must be exactly 0, got {r2_zero}"
