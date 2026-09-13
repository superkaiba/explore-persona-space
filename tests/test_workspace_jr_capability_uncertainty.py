"""Check question-cluster weighting, cross-model pairing and immutable outputs."""

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

_PATH = Path(__file__).resolve().parents[1] / "scripts/workspace_jr_capability_uncertainty.py"
_SPEC = importlib.util.spec_from_file_location("capability_uncertainty", _PATH)
reader = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(reader)


def test_unequal_cluster_counts_and_paired_difference_against_scalar_oracle():
    """Changing question multiplicities must preserve both models' shared resample."""
    correct = np.array([[1, 0, 4], [0, 3, 2]])
    counts = np.array([[1, 2, 5], [2, 3, 4]])
    indices = np.array([[0, 0, 1], [2, 1, 2], [1, 1, 1], [0, 2, 1]])
    pooled, equal = reader.cluster_bootstrap(correct, counts, indices)
    for model in range(2):
        for b, draw in enumerate(indices):
            assert pooled[model, b] == sum(correct[model, i] for i in draw) / sum(
                counts[model, i] for i in draw
            )
            oracle = sum(correct[model, i] / counts[model, i] for i in draw) / len(draw)
            assert np.isclose(equal[model, b], oracle, rtol=0, atol=1e-15)
    assert not np.array_equal(pooled, equal)
    result = reader.estimates(correct.sum(1) / counts.sum(1), pooled)
    interval = result["primary_minus_comparison"]["ci95"]
    np.testing.assert_array_equal(interval, np.quantile(pooled[0] - pooled[1], [0.025, 0.975]))
    unpaired = pooled[0] - pooled[1, ::-1]
    assert not np.array_equal(interval, np.quantile(unpaired, [0.025, 0.975]))


def test_completed_artifacts_are_immutable(tmp_path):
    """A repeated invocation must stop before touching published scientific outputs."""
    plan = tmp_path / "plan.json"
    plan.write_text(json.dumps({"models": ["q35_27b", "q35_4b"], "seeds": list(range(42, 47))}))
    out = tmp_path / "output"
    out.mkdir()
    sentinel = b'{"status":"complete"}\n'
    (out / "complete.json").write_bytes(sentinel)
    with pytest.raises(ValueError, match="must not be overwritten"):
        reader.main(plan, out)
    assert list(out.iterdir()) == [out / "complete.json"]
    assert (out / "complete.json").read_bytes() == sentinel
