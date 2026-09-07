"""Algebra/orientation and reference tests for the Qwen3 existing-data analysis."""

import importlib.util
from pathlib import Path

import numpy as np
import torch

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/issue2546_qwen3_map_geometry.py"
SPEC = importlib.util.spec_from_file_location("qwen3_geometry", SCRIPT)
MOD = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MOD)


def test_production_raw_operator_and_orientation():
    """Check exact standardization reversal with unequal input/output dimensions."""
    ridge = MOD.production_ridge_class(SCRIPT.with_name("issue2546_allfit_necessity.py"))
    rng = torch.Generator().manual_seed(0)
    x = torch.randn(80, 6, generator=rng, dtype=torch.float64)
    x = x * torch.arange(1, 7) + torch.arange(6) * 10
    y = torch.randn(80, 4, generator=rng, dtype=torch.float64)
    model = ridge(x, y, 3.0)
    operator = MOD.raw_operator(model)
    assert operator.shape == (4, 6)
    torch.testing.assert_close(model.xsd, x.std(0, correction=1) + 1e-9)
    intercept = model.ymu - operator @ model.xmu
    torch.testing.assert_close(x @ operator.T + intercept, model.predict(x))
    left, values, right_t = torch.linalg.svd(operator, full_matrices=False)
    torch.testing.assert_close(operator @ right_t.T, left * values)
    assert left.shape == (4, 4) and right_t.T.shape == (6, 4)


def test_subspace_invariance_and_reference():
    """Basis rotations/signs leave overlap invariant; orthogonal subspaces vanish."""
    first, other = (
        torch.eye(16, dtype=torch.float64)[:, :4],
        torch.eye(16, dtype=torch.float64)[:, 4:8],
    )
    rotated = first[:, [2, 0, 3, 1]] * torch.tensor([1, -1, 1, -1])
    assert np.isclose(MOD.overlap(first, rotated)["mean_principal_cos"], 1)
    assert MOD.overlap(first, other)["sq_projection_share"] == 0
    reference = MOD.random_reference(64, 4, 5, 42)
    assert len(reference["principal_cosines_per_draw"]) == 5
    assert reference["analytic_expected_sq_projection_share"] == 4 / 64
    assert reference == MOD.random_reference(64, 4, 5, 42)


def test_atomic_tensor_and_json_roundtrip(tmp_path):
    """Check actual persisted file layout and payloads."""
    path = tmp_path / "bundle.npz"
    MOD.write_npz(path, operator=np.eye(3))
    with np.load(path, allow_pickle=False) as archive:
        np.testing.assert_array_equal(archive["operator"], np.eye(3))
    MOD.write_json(tmp_path / "summary.json", {"complete": True})
    assert (tmp_path / "summary.json").read_text().endswith("\n")
