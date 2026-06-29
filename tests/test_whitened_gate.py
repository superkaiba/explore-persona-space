"""B3 reduction unit test for the whitened key-query gate (issue #665 Phase 3).

Pins two invariants the plan §5 / §7 make load-bearing BEFORE any A3.9/A3.10
number may be reported:

1. **Reduction to cosine.** In the Σc = I (M ∝ I) + equal-norm + δ ∥ r_B limit,
   `whitened_gate(c_C, c_C', M=I) == cos(c_C, c_C')` to within 1e-6 on synthetic
   data. (With M = I and ‖c_C‖ = ‖c_C'‖, g = c_Cᵀc_C' / ‖c_C‖² = cos.)

2. **Finite / non-NaN at the smallest swept λ = 1e-3.** n = 3000 < d = 3584 makes
   the raw Σc genuinely SINGULAR, so the λ floor is what makes `(Σc+λI)⁻¹` exist
   at all. A too-small λ that lets the singular Σc leak through would produce a
   non-finite gate — this test catches that before any A3.9/A3.10 number.
"""

from __future__ import annotations

import numpy as np

from explore_persona_space.analysis.whitened_gate import (
    metric_ablation,
    raw_cosine_gate,
    sigma_c_inv,
    whitened_gate,
)


def test_whitened_gate_reduces_to_cosine_in_identity_equal_norm_limit():
    """Σc = I + equal-norm keys/queries → whitened_gate == cos within 1e-6."""
    rng = np.random.default_rng(0)
    d = 64
    tol = 1e-6
    for _ in range(50):
        c_C = rng.standard_normal(d)
        c_Cp = rng.standard_normal(d)
        # equal-norm limit: rescale c_C' to ‖c_C'‖ == ‖c_C‖
        c_Cp = c_Cp * (np.linalg.norm(c_C) / np.linalg.norm(c_Cp))
        # M ∝ I: identity scalar metric (the metric_ablation "I" cell)
        g = whitened_gate(c_C, c_Cp, M=1.0)
        cos = raw_cosine_gate(c_C, c_Cp)
        assert abs(g - cos) < tol, f"gate {g} vs cos {cos} (diff {abs(g - cos):.2e})"


def test_whitened_gate_source_self_gate_is_unity():
    """g_C(C=C') == 1 by construction (normalizer == numerator at the source)."""
    rng = np.random.default_rng(1)
    d = 64
    c_C = rng.standard_normal(d)
    # arbitrary PSD metric M = AᵀA + I
    A = rng.standard_normal((d, d))
    M = A.T @ A + np.eye(d)
    g_self = whitened_gate(c_C, c_C, M=M)
    assert abs(g_self - 1.0) < 1e-9, f"source self-gate {g_self} != 1"


def test_sigma_c_inv_finite_and_nonnan_at_smallest_lambda_singular_sigma():
    """The λ floor is load-bearing: a genuinely SINGULAR Σc (n<d, rank-deficient)
    still yields a finite, non-NaN inverse + gate at the smallest swept λ=1e-3."""
    rng = np.random.default_rng(2)
    d = 128
    n = 30  # n < d -> rank-deficient empirical covariance (the n=3000<d=3584 regime)
    X = rng.standard_normal((n, d))
    sigma_c = (X.T @ X) / n  # rank <= n < d => SINGULAR
    # confirm the raw Sigma_c is genuinely singular (the precondition the lambda guards)
    rank = np.linalg.matrix_rank(sigma_c)
    assert rank <= n < d, f"expected rank-deficient Sigma_c (rank {rank}, n {n}, d {d})"

    smallest_lambda = 1e-3
    M = sigma_c_inv(sigma_c, smallest_lambda)
    assert np.all(np.isfinite(M)), "inverse metric has non-finite entries at lambda=1e-3"

    c_C = rng.standard_normal(d)
    c_Cp = rng.standard_normal(d)
    g = whitened_gate(c_C, c_Cp, M=M)
    assert np.isfinite(g), f"whitened gate non-finite at lambda=1e-3 on singular Sigma_c: {g}"


def test_whitened_gate_finite_across_full_lambda_sweep():
    """All three swept λ {1e-3, 1e-2, 1e-1} give finite gates on a singular Σc."""
    rng = np.random.default_rng(3)
    d = 96
    n = 20
    X = rng.standard_normal((n, d))
    sigma_c = (X.T @ X) / n
    c_C = rng.standard_normal(d)
    c_Cp = rng.standard_normal(d)
    for lam in (1e-3, 1e-2, 1e-1):
        M = sigma_c_inv(sigma_c, lam)
        g = whitened_gate(c_C, c_Cp, M=M)
        assert np.isfinite(g), f"gate non-finite at lambda={lam}: {g}"


def test_metric_ablation_three_cells_present_and_shaped():
    """metric_ablation returns the three A3.9 metric cells with correct shapes."""
    rng = np.random.default_rng(4)
    d = 48
    n = 15
    X = rng.standard_normal((n, d))
    sigma_c = (X.T @ X) / n
    cells = metric_ablation(sigma_c, lam=1e-2)
    assert set(cells.keys()) == {"I", "diag_Sigma_inv", "Sigma_inv"}
    assert np.ndim(cells["I"]) == 0
    assert np.asarray(cells["diag_Sigma_inv"]).shape == (d,)
    assert np.asarray(cells["Sigma_inv"]).shape == (d, d)
    # every metric cell yields a finite gate
    c_C = rng.standard_normal(d)
    c_Cp = rng.standard_normal(d)
    for key, M in cells.items():
        g = whitened_gate(c_C, c_Cp, M=M)
        assert np.isfinite(g), f"metric cell {key} gate non-finite: {g}"
