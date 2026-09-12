"""Sparse benchmark parity cannot overlook dropped rows or changed diagnostics."""

import runpy
from pathlib import Path

import pytest
import torch

from explore_persona_space.analysis.workspace_lenses import nonnegative_gradient_pursuit


@pytest.fixture
def benchmark_parity(monkeypatch):
    from explore_persona_space.orchestrate import env

    monkeypatch.setattr(env, "load_dotenv", lambda: None)
    script = Path(__file__).resolve().parents[1] / "scripts/workspace_jr_sparse_benchmark.py"
    return runpy.run_path(str(script))["parity"]


@pytest.fixture
def reference():
    """Two identical rows make accidental broadcasting observable."""
    dictionary = torch.eye(32)
    x = torch.zeros(2, 32)
    x[:, :3] = torch.tensor([1.0, 0.5, -0.25])
    results = {}
    for k in (5, 10, 25):
        result = nonnegative_gradient_pursuit(x, dictionary, k=k)
        results[k] = {name: getattr(result, name) for name in result.__dataclass_fields__}
    return results


def clone(results):
    return {
        k: {name: value.clone() for name, value in fields.items()} for k, fields in results.items()
    }


def test_exact_reference_passes_every_k_and_field(benchmark_parity, reference):
    report = benchmark_parity(clone(reference), reference)
    assert report["passed"] is True
    assert set(report["fields"]) == {"5", "10", "25"}
    for k, fields in reference.items():
        assert set(report["fields"][str(k)]) == set(fields)


@pytest.mark.parametrize("field", ["component", "indices", "active_atoms", "squared_error"])
def test_broadcasting_cannot_hide_a_missing_token_row(benchmark_parity, reference, field):
    actual = clone(reference)
    actual[10][field] = actual[10][field][:1]
    with pytest.raises(ValueError):
        benchmark_parity(actual, reference)


@pytest.mark.parametrize("mutation", ["missing_k", "extra_k", "missing_field", "extra_field"])
def test_exact_checkpoint_and_field_sets_are_required(benchmark_parity, reference, mutation):
    actual = clone(reference)
    if mutation == "missing_k":
        actual.pop(5)
    elif mutation == "extra_k":
        actual[20] = actual[25]
    elif mutation == "missing_field":
        del actual[10]["zero_update_steps"]
    else:
        actual[10]["unexpected"] = torch.ones(2)
    with pytest.raises(ValueError):
        benchmark_parity(actual, reference)


@pytest.mark.parametrize("field", ["component", "indices"])
def test_parity_requires_exact_tensor_dtypes(benchmark_parity, reference, field):
    actual = clone(reference)
    actual[10][field] = actual[10][field].to(
        torch.float64 if field == "component" else torch.float32
    )
    with pytest.raises(ValueError):
        benchmark_parity(actual, reference)


@pytest.mark.parametrize("value", [float("nan"), float("inf")])
@pytest.mark.parametrize("side", ["actual", "reference"])
def test_nonfinite_computation_fails_before_nonfinite_report_serialization(
    benchmark_parity, reference, value, side
):
    actual = clone(reference)
    (actual if side == "actual" else reference)[10]["component"][0, 0] = value
    with pytest.raises(ValueError):
        benchmark_parity(actual, reference)


@pytest.mark.parametrize(
    "field", ["component", "coefficients", "squared_error", "input_squared_norm"]
)
def test_each_floating_field_obeys_the_declared_numerical_gate(benchmark_parity, reference, field):
    actual = clone(reference)
    actual[25][field].view(-1)[0] += 0.01
    assert benchmark_parity(actual, reference)["passed"] is False


@pytest.mark.parametrize(
    "field", ["indices", "active_atoms", "zero_update_steps", "increasing_error_steps"]
)
def test_each_integer_diagnostic_must_match_exactly(benchmark_parity, reference, field):
    actual = clone(reference)
    actual[5][field].view(-1)[0] += 1
    assert benchmark_parity(actual, reference)["passed"] is False


def test_declared_float_tolerance_is_preserved(benchmark_parity, reference):
    actual = clone(reference)
    actual[5]["component"][0, 0] += 5e-6
    actual[10]["coefficients"][0, -1] += 5e-7
    assert benchmark_parity(actual, reference)["passed"] is True
