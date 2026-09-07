"""Numerical and selection checks for the cached-state CoT rank reproduction."""

import numpy as np
import pytest
import torch
from scripts.issue2546_qwen3_rank_reproduction import (
    combine,
    fit,
    moments,
    participation_ratio,
    predict,
    select_rank,
    sse_curve,
    validate_production_class,
)


def test_moment_ridge_matches_direct_and_projection_curve():
    """Shared Gram reduction matches direct ridge; every rank matches explicit SSE."""
    torch.manual_seed(12)
    x = torch.randn(90, 12, dtype=torch.float64) + 3
    y = x @ torch.randn(12, 9, dtype=torch.float64) + torch.randn(90, 9, dtype=torch.float64)
    b = combine([moments(x[:40], y[:40]), moments(x[40:70], y[40:70])])
    model = fit(b, 1.7)
    xm, sd, ym = x[:70].mean(0), x[:70].std(0) + 1e-9, y[:70].mean(0)
    xn = (x[:70] - xm) / sd
    beta = torch.linalg.solve(xn.T @ xn + 1.7 * torch.eye(12), xn.T @ (y[:70] - ym))
    p = predict(model, x[70:])
    torch.testing.assert_close(p, (x[70:] - xm) / sd @ beta + ym, rtol=1e-8, atol=1e-8)
    curve = sse_curve(p, y[70:], model["ymu"], model["vectors"])
    for rank in range(10):
        v = model["vectors"][:, :rank]
        low = (p - ym) @ v @ v.T + ym
        np.testing.assert_allclose(curve[rank], float((low - y[70:]).square().sum()), rtol=1e-10)


def test_pr_is_raw_centered_covariance_and_invariant_to_scale():
    """Participation ratio is not the standardized-input correlation dimension."""
    torch.manual_seed(3)
    x = torch.randn(100, 8, dtype=torch.float64) * torch.arange(1, 9)
    cov = torch.cov(x.T)
    expected = float(cov.trace().square() / cov.square().sum())
    assert participation_ratio(moments(x, x)) == pytest.approx(expected)
    assert participation_ratio(moments(x * 4 + 9, x)) == pytest.approx(expected)


def test_threshold_is_extra_sse_and_allows_nonmonotone_curves():
    """Select minimum passing validation rank, never a percentage of R-squared."""
    curve = np.array([150.0, 115.0, 109.0, 112.0, 100.0])
    assert select_rank(curve, 0.1) == 2
    assert select_rank(curve, 0.05) == 4
    with pytest.raises(ValueError):
        combine([])


def test_matches_extracted_production_class():
    """Exercise the exact original Ridge class without its module-level side effects."""
    from pathlib import Path

    torch.manual_seed(7)
    x, y = torch.randn(45, 8).double(), torch.randn(45, 8).double()
    source = Path(__file__).resolve().parents[1] / "scripts/issue2546_allfit_necessity.py"
    validate_production_class(source, x, y, moments(x, y), 10.0)
