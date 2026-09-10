"""Recovery must surface historical differences and halt structural failures."""

import numpy as np
import pytest

from scripts import issue2054_k3_recover as recovery


def test_historical_exceedance_survives_as_warning_with_controls(monkeypatch):
    reference = np.array([[1.0, 1.0], [2.0, 2.0]])
    current = reference * np.array([[1.03], [1.01]])
    monkeypatch.setattr(
        recovery, "parent_vectors", lambda *a, **k: (current, reference * 1.3, reference * 1.5, 0.0)
    )
    rel, audit = recovery.parity_audit(
        None,
        None,
        [],
        current,
        reference,
        {"cutoff": 0.025, "adjudication_concern": "numerical drift"},
    )
    np.testing.assert_allclose(rel, [0.03, 0.01], atol=1e-6)
    assert audit["severity"] == "WARN"
    assert audit["warning_row_indices"] == [0]
    assert audit["historical_relative_error"] == rel.tolist()


@pytest.mark.parametrize("fault", ["hook", "wrong_layer", "nonfinite"])
def test_structural_capture_faults_always_halt(monkeypatch, fault):
    reference = np.ones((2, 3))
    current = reference.copy()
    if fault == "wrong_layer":
        current *= 1.4
    elif fault == "nonfinite":
        current[0, 0] = np.nan
    monkeypatch.setattr(
        recovery,
        "parent_vectors",
        lambda *a, **k: (
            reference,
            reference * 1.3,
            reference * 1.5,
            0.125 if fault == "hook" else 0.0,
        ),
    )
    with pytest.raises(RuntimeError):
        recovery.parity_audit(
            None,
            None,
            [],
            current,
            reference,
            {"cutoff": 0.025, "adjudication_concern": "numerical drift"},
        )
