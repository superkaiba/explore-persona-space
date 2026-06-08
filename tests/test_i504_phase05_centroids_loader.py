# em-dash + Qwen marker intentional in i504 module docstrings
"""Regression test for the Phase 0.5 centroids loader (task #504).

Pins both schema branches of ``_load_centroids_layer`` in
``scripts/i504_phase_phase05.py``:

* PRIMARY — the structured #472 schema actually written by
  ``scripts/i472_phase_centroids.py`` /
  ``contrastive_neg_geometry_472.centroids.build_centroids``.

* FALLBACK — the legacy flat ``{persona_name: tensor}`` layout the
  CPU smoke used to write before #504 round-3.

The round-2 launch crashed at Phase 0.5 with ``ValueError: could not
convert string to float: 'librarian'`` because the loader iterated a
structured-dict payload (centroids/persona_names/cos_matrix/...) as
if it were the flat layout, hitting the ``persona_names`` LIST and
trying to coerce a name string to float32.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from scripts.i504_phase_phase05 import _load_centroids_layer


def _write_structured(path: Path, names: list[str], dim: int = 16) -> np.ndarray:
    """Write a #472-shape structured centroids bundle and return the matrix."""
    rng = np.random.default_rng(0)
    mat = rng.standard_normal((len(names), dim)).astype(np.float32)
    cos = mat @ mat.T  # NOT row-normalised; smoke-only stand-in.
    torch.save(
        {
            "centroids": torch.from_numpy(mat),
            "persona_names": names,
            "cos_matrix": torch.from_numpy(cos.astype(np.float32)),
            "layer": 10,
            "base_model": "synthetic-test",
            "questions": ["q_0", "q_1"],
        },
        str(path),
    )
    return mat


def _write_flat(path: Path, names: list[str], dim: int = 16) -> dict[str, np.ndarray]:
    """Write a legacy flat ``{name: tensor}`` centroids file and return arrays."""
    rng = np.random.default_rng(1)
    out: dict[str, np.ndarray] = {}
    flat: dict[str, torch.Tensor] = {}
    for n in names:
        v = rng.standard_normal(dim).astype(np.float32)
        out[n] = v
        flat[n] = torch.from_numpy(v)
    torch.save(flat, str(path))
    return out


def test_load_structured_schema_returns_name_keyed_arrays(tmp_path: Path) -> None:
    """Structured #472 payload should unpack to ``{name: float32 ndarray}``."""
    names = ["villain", "qwen_default", "near_persona", "far_persona", "librarian"]
    mat = _write_structured(tmp_path / "centroids_L10.pt", names)

    out = _load_centroids_layer(tmp_path, layer=10)

    assert set(out.keys()) == set(names), out.keys()
    for i, n in enumerate(names):
        v = out[n]
        assert isinstance(v, np.ndarray), (n, type(v))
        assert v.dtype == np.float32, (n, v.dtype)
        np.testing.assert_array_equal(v, mat[i])


def test_load_legacy_flat_schema_returns_name_keyed_arrays(tmp_path: Path) -> None:
    """Smoke-shaped flat ``{name: tensor}`` payload should still unpack cleanly."""
    names = ["villain", "qwen_default", "probe_persona_001"]
    expected = _write_flat(tmp_path / "centroids_L10.pt", names)

    out = _load_centroids_layer(tmp_path, layer=10)

    assert set(out.keys()) == set(names), out.keys()
    for n in names:
        v = out[n]
        assert isinstance(v, np.ndarray), (n, type(v))
        assert v.dtype == np.float32, (n, v.dtype)
        np.testing.assert_allclose(v, expected[n], rtol=0.0, atol=0.0)


def test_load_missing_file_raises(tmp_path: Path) -> None:
    """Missing centroids file should fail loud, not silently default."""
    with pytest.raises(FileNotFoundError):
        _load_centroids_layer(tmp_path, layer=99)


def test_load_structured_schema_length_mismatch_raises(tmp_path: Path) -> None:
    """centroids row-count vs persona_names length mismatch should raise."""
    path = tmp_path / "centroids_L10.pt"
    mat = np.random.default_rng(0).standard_normal((3, 8)).astype(np.float32)
    torch.save(
        {
            "centroids": torch.from_numpy(mat),
            "persona_names": ["a", "b"],  # length 2, centroids has 3 rows
            "cos_matrix": torch.from_numpy(mat @ mat.T),
            "layer": 10,
            "base_model": "synthetic-test",
            "questions": [],
        },
        str(path),
    )
    with pytest.raises(ValueError, match="length mismatch"):
        _load_centroids_layer(tmp_path, layer=10)


def test_load_round2_crash_regression(tmp_path: Path) -> None:
    """Reproduces the exact #504 round-2 crash on the structured schema.

    Round-2 logged ``could not convert string to float: 'librarian'`` because
    ``persona_names`` (a list[str]) was iterated as ``{name: tensor}``. The
    fix is verified by including a persona name that previously triggered
    the coercion path AND asserting we get the correct vector back.
    """
    names = ["librarian", "villain", "doctor", "qwen_default"]
    mat = _write_structured(tmp_path / "centroids_L15.pt", names, dim=12)

    out = _load_centroids_layer(tmp_path, layer=15)

    # Specifically check the name that crashed round-2.
    assert "librarian" in out
    assert out["librarian"].shape == (12,)
    np.testing.assert_array_equal(out["librarian"], mat[0])
