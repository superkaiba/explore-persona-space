"""Test 2 (plan v7 §13 item 2 + §4 Stage-1 step 2 + §12 row 5) — full-sample
LEACE erasure is EXACT on the fitted data (closed-form guarantee, Belrose 2023,
arXiv:2306.03819), and it changes directions orthogonal to the erased concept
as LITTLE as possible (the minimal-change property).

This is the Option-A FULL-SAMPLE eraser (one eraser fit on all 50 contexts per
behavior x genre, plan §4 Stage 1) — NOT a fold-local one.

Planted ground-truth (see conftest.make_leace_exact_dataset):
  E0 = v0 @ w + small_noise, with w a known unit basis direction. After LEACE:
    * cov(E0, P @ v0) ~= 0 along EVERY direction (no linear classifier recovers E0)
    * the component of v0 ORTHOGONAL to w is ~unchanged (minimal change)

Seed: 7421 (LEACE synthetic, plan §10 reproducibility card).
"""

from __future__ import annotations

import numpy as np
import pytest

from .conftest import impl, impl_has, make_leace_exact_dataset

ERASE_TOL = 1e-6  # closed-form exactness tolerance (plan §13 item 2)


@pytest.mark.skipif(
    not (impl_has("fit_leace") or impl_has("leace_residual")),
    reason="implementation pending round 2",
)
def test_full_sample_leace_zeroes_covariance_with_E0():
    # v0 (n=80, d=12) with E0 linearly decodable along basis axis 3.
    v0, E0, _w = make_leace_exact_dataset(n=80, d=12, signal_dim=3, seed=7421)

    # full-sample residual after LEACE fit on ALL points
    if impl_has("leace_residual"):
        residual = impl.leace_residual(v0, E0)
    else:
        eraser = impl.fit_leace(v0, E0)
        residual = eraser.transform(v0)

    # cov(E0, residual) must be ~0 along EVERY coordinate (closed-form guarantee)
    E0c = E0 - E0.mean()
    resid_c = residual - residual.mean(axis=0, keepdims=True)
    cov_per_dim = (resid_c * E0c[:, None]).mean(axis=0)  # (d,)
    max_abs_cov = float(np.abs(cov_per_dim).max())
    assert max_abs_cov <= ERASE_TOL, (
        f"post-LEACE max |cov(E0, residual)| = {max_abs_cov:.2e} exceeds {ERASE_TOL:.0e} "
        "(LEACE must zero the covariance with E0 along every direction)"
    )

    # a best-fit linear probe of E0 on the residual must have ~0 correlation
    beta, *_ = np.linalg.lstsq(np.column_stack([np.ones(len(E0)), residual]), E0, rcond=None)
    E0_hat = np.column_stack([np.ones(len(E0)), residual]) @ beta
    if np.std(E0_hat) > 1e-9:
        rho = np.corrcoef(E0_hat, E0)[0, 1]
        assert abs(rho) <= 1e-3, (
            f"a linear probe still recovers E0 from the LEACE residual (rho={rho:.2e})"
        )


@pytest.mark.skipif(
    not (impl_has("fit_leace") or impl_has("leace_residual")),
    reason="implementation pending round 2",
)
def test_leace_preserves_directions_orthogonal_to_concept():
    # minimal-change property: the projection of v0 onto a direction ORTHOGONAL
    # to the E0-decodable subspace must be ~unchanged after erasure.
    v0, E0, _w = make_leace_exact_dataset(n=80, d=12, signal_dim=3, seed=7421)

    if impl_has("leace_residual"):
        residual = impl.leace_residual(v0, E0)
    else:
        residual = impl.fit_leace(v0, E0).transform(v0)

    # an orthogonal probe direction (basis axis 7, != signal axis 3)
    u = np.zeros(v0.shape[1])
    u[7] = 1.0
    proj_before = v0 @ u
    proj_after = residual @ u

    # the orthogonal projection is preserved up to LEACE's minimal-change slack
    rel_change = np.linalg.norm(proj_after - proj_before) / (np.linalg.norm(proj_before) + 1e-12)
    assert rel_change <= 0.05, (
        f"LEACE perturbed an E0-orthogonal direction by rel-change {rel_change:.3f} "
        "(>5%): the minimal-change property is violated"
    )
