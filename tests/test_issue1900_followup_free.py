"""Boundary pins for the #1900 free-analysis follow-up (scripts/issue1900_followup_free.py).

1. OLS residualization removes the base term exactly (noise-free y = a + b*x
   -> resid ~ 0) and is Pearson-orthogonal to the base term in general.
2. The coupling null centers where the closed form says on synthetic
   bivariate-Gaussian data (Spearman via the exact normal-pair conversion
   (6/pi)*asin(rho/2) of the Pearson closed form).
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "scripts"))


def _fu():
    import issue1900_followup_free as fu

    return fu


def test_residualization_removes_base_term_exactly():
    fu = _fu()
    rng = np.random.default_rng(0)
    x = rng.normal(size=800)
    # noise-free: y is an affine function of the base term -> residual == 0
    y = 2.5 + 3.0 * x
    resid, fit = fu.residualize_ols(y, x)
    assert np.max(np.abs(resid)) < 1e-8
    assert abs(fit["alpha"] - 2.5) < 1e-8 and abs(fit["beta"] - 3.0) < 1e-8
    assert fit["r2_1d_fit"] > 1.0 - 1e-12
    # noisy: residual is Pearson-orthogonal to the base term (intercept + slope)
    e = rng.normal(size=800)
    y2 = 1.0 - 2.0 * x + e
    resid2, fit2 = fu.residualize_ols(y2, x)
    assert abs(np.corrcoef(resid2, x)[0, 1]) < 1e-10
    assert abs(resid2.mean()) < 1e-10
    assert 0.0 < fit2["r2_1d_fit"] < 1.0


def test_coupling_null_centers_at_closed_form_on_gaussian():
    fu = _fu()
    rng = np.random.default_rng(1)
    n, sigma_b, sigma_t = 1500, 2.0, 1.0
    base = rng.normal(scale=sigma_b, size=n)
    trained = rng.normal(scale=sigma_t, size=n)  # independent of base
    # p7 candidate IS the base term (the content-arm case)
    null = fu.coupling_null_battery(base[:, None], base, trained, n_draws=400, seed=7)
    assert null.shape == (400, 1)
    cf = fu.closed_form_pearson_null(base, base, trained)
    assert abs(cf["r_p7_baseterm"] - 1.0) < 1e-12  # reduces to -sigma_B/sqrt(...)
    assert cf["value"] < 0
    # exact normal-pair Pearson->Spearman conversion for the expected center
    expected_spearman = (6.0 / math.pi) * math.asin(cf["value"] / 2.0)
    assert abs(float(null.mean()) - expected_spearman) < 0.02, (
        float(null.mean()),
        expected_spearman,
    )
    # a base-independent candidate centers at 0 under the same null
    indep = rng.normal(size=n)
    null2 = fu.coupling_null_battery(indep[:, None], base, trained, n_draws=400, seed=7)
    assert abs(float(null2.mean())) < 0.01
