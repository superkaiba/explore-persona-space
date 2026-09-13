"""Readout validation uses the producer's precision without relaxing its checks."""

from pathlib import Path

import numpy as np
import pytest


@pytest.fixture
def projection_case(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    from workspace_jr_supplement import verify_readout_projection

    rng = np.random.default_rng(27)
    basis = rng.normal(size=(5120, 144)).astype(np.float32)
    basis /= np.linalg.norm(basis.astype(np.float64), axis=0)
    assert not np.allclose(np.linalg.norm(basis, axis=0), 1, rtol=1e-6, atol=1e-8)
    target, predicted = rng.normal(size=(2, 5120)), rng.normal(size=(2, 5120))
    arrays = {}
    for arm in ("J", "R", "random", "pca"):
        arrays[f"direction__{arm}"] = basis.copy()
        arrays[f"target__{arm}"] = target @ basis.astype(np.float64)
        arrays[f"prediction__ridge__{arm}"] = predicted @ basis.astype(np.float64)
    args = (
        {"test_context_ids": [1, 2]},
        arrays,
        {"full": target},
        {"ridge": {"full": predicted}},
        [1, 2],
    )
    return verify_readout_projection, args


def test_normalized_fp32_columns_use_fp64_accumulation(projection_case, capsys):
    verify, args = projection_case
    before = {name: value.copy() for name, value in args[1].items()}
    verify(*args)
    assert "norm_accumulation=float64 arms=J,R,random,pca" in capsys.readouterr().out
    for name, value in before.items():
        np.testing.assert_array_equal(args[1][name], value)


@pytest.mark.parametrize("invalid", [1.0001, 0.0, np.nan, np.inf])
def test_invalid_directions_still_fail(projection_case, invalid):
    verify, args = projection_case
    args[1]["direction__J"][:, 0] *= invalid
    with pytest.raises(ValueError, match="not normalized"):
        verify(*args)


@pytest.mark.parametrize("field", ["target__J", "prediction__ridge__J"])
def test_changed_saved_projection_still_fails(projection_case, field):
    verify, args = projection_case
    args[1][field][0, 0] += 0.001
    with pytest.raises(ValueError, match="differ"):
        verify(*args)
