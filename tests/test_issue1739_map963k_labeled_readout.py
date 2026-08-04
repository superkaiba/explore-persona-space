"""Equivalence pins for the 963k-ridge -> MapFit shim (#1739 labeled readouts).

The shim (``build_963k_mapfit``) presents #779's per-layer n1m ridge payloads as
one ``MapFit(kind='linear')`` so ``arms.run_cell_multi``'s map-consuming arms work
unchanged. These tests execute BOTH real application paths — ``fits.apply_map``
(numpy fp64) and ``issue779_ffc_n1m_fits.apply_map`` (torch fp64, the canonical
#779 predict path) — on synthetic payloads and pin their agreement, plus the
gate's refusal behavior and the committed-convention shuffled-control weights.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import issue1739_map963k_labeled_readout as mod  # noqa: E402
from issue1739_map963k_readout import shuffle_rows  # noqa: E402

from explore_persona_space.experiments.issue_1739 import fits  # noqa: E402


def _ridge_payload(rng: np.random.Generator, d: int, layer: int) -> dict:
    """A synthetic #779 n1m ridge payload (fp32 tensors, the persisted shape)."""
    return {
        "kind": "ridge",
        "layer": layer,
        "fitter": "ridge",
        "W": torch.as_tensor(rng.standard_normal((d, d)).astype(np.float32)),
        "xmu": torch.as_tensor(rng.standard_normal(d).astype(np.float32)),
        "xsd": torch.as_tensor((0.5 + rng.random(d)).astype(np.float32)),
        "ymu": torch.as_tensor(rng.standard_normal(d).astype(np.float32)),
    }


def test_shim_matches_n1m_apply_map() -> None:
    """fits.apply_map(x, shim) == issue779 apply_map(payload, x) per layer (fp64)."""
    rng = np.random.default_rng(0)
    d, n = 16, 7
    payloads = [_ridge_payload(rng, d, layer) for layer in (14, 19, 26)]
    shim = mod.build_963k_mapfit(payloads)
    assert shim.kind == "linear" and shim.w.shape == (3, d, d)

    x = rng.standard_normal((3, n, d))
    a = fits.apply_map(x, shim)
    b = np.stack([mod.apply_963k(payloads[li], x[li], "cpu") for li in range(3)])
    assert a.shape == b.shape == (3, n, d)
    # Same fp64 algebra; tiny-d GEMMs agree to reduction-order noise.
    np.testing.assert_allclose(a, b, rtol=0, atol=1e-12)


def test_equivalence_gate_passes_and_refuses() -> None:
    """The gate PASSes on the honest shim and RAISES on a corrupted standardizer."""
    rng = np.random.default_rng(1)
    d = 12
    payloads = [_ridge_payload(rng, d, layer) for layer in (14, 19, 26)]
    shim = mod.build_963k_mapfit(payloads)
    x = rng.standard_normal((3, 9, d))
    report = mod.equivalence_gate(x, shim, payloads)
    assert report["rel_max_abs"] <= report["rel_tol"]

    bad = mod.build_963k_mapfit(payloads)
    bad.x_mu = bad.x_mu + 1.0  # a real mis-normalization is O(1), not reduction noise
    with pytest.raises(RuntimeError, match="EQUIVALENCE GATE FAILED"):
        mod.equivalence_gate(x, bad, payloads)


def test_shim_refuses_non_ridge_payload() -> None:
    """The linear shim is exact for ridge only — mlp/krr payloads must refuse."""
    rng = np.random.default_rng(2)
    payloads = [_ridge_payload(rng, 8, 14)]
    payloads[0]["kind"] = "mlp"
    with pytest.raises(ValueError, match="RIDGE payload only"):
        mod.build_963k_mapfit(payloads)


def test_committed_shuffled_weights_convention() -> None:
    """Per-layer committed shuffle: matches shuffle_rows(seed=0), norm-preserving."""
    rng = np.random.default_rng(3)
    w = rng.standard_normal((3, 10, 10))
    shuf = mod.committed_shuffled_weights(w)
    for li in range(3):
        np.testing.assert_array_equal(shuf[li], shuffle_rows(w[li], 0))
        assert np.isclose(np.linalg.norm(shuf[li]), np.linalg.norm(w[li]), rtol=1e-12)
        assert not np.array_equal(shuf[li], w[li])
