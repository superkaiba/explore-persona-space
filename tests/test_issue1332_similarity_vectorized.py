"""Serial-vs-batched equivalence pin for the #1332 similarity pair-scoring vectorization.

The mid-run vectorize fix replaced ``similarity_at_layer``'s per-(i, j) serial
``r2()`` double loop with pair-axis-batched fp64 reductions
(``_score_similarity_fold_batched``). This test pins the batched PRODUCTION
path to the retained serial oracle (``_similarity_at_layer_serial_reference``)
on tiny synthetic families through the REAL ``solver="fast"`` ridge path (the
production ``parity_gate`` outcome). The production-shape gate is the script's
``--verify-sim-layer`` mode against the persisted serial L5 JSON.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

fits = pytest.importorskip("issue1332_fits")

_MATRIX_KEYS = (
    "S_trans",
    "S_sym",
    "S_asym",
    "S_agree",
    "S_dmap_one_minus",
    "S_mean_target",
    "S_excess",
)


class _StubCache:
    """Duck-typed ShardCache over in-memory fp32 arrays (real methods, no mocks)."""

    def __init__(self, data: dict[str, tuple[np.ndarray, np.ndarray]]):
        self._data = data

    def bank_indices(self, fam: str) -> list[int]:
        return list(range(self._data[fam][0].shape[0]))

    def arrays(self, fam: str, key: str, layers: list[int]) -> np.ndarray:
        X, Y = self._data[fam]
        assert layers == [0], layers
        return (X if key == "cx_last" else Y)[:, None, :]


def _tiny_store(seed: int = 0, n_fams: int = 3, n_rows: int = 12, hdim: int = 7):
    rng = np.random.default_rng(seed)
    data = {}
    for k in range(n_fams):
        X = rng.normal(size=(n_rows, hdim)).astype(np.float32)
        W = rng.normal(size=(hdim, hdim)).astype(np.float32)
        Y = (X @ W + 0.1 * rng.normal(size=(n_rows, hdim))).astype(np.float32)
        data[f"fam{k}"] = (X, Y)
    return data


def test_batched_similarity_matches_serial_reference():
    """Every similarity matrix agrees with the serial oracle (matrix-scale rel diff)."""
    data = _tiny_store()
    cache = _StubCache(data)
    fams = sorted(data)
    folds = [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11]]
    batched = fits.similarity_at_layer(cache, fams, 0, folds, solver="fast")
    serial = fits._similarity_at_layer_serial_reference(cache, fams, 0, folds, solver="fast")
    assert batched["families"] == serial["families"] == fams
    assert batched["n_folds"] == serial["n_folds"] == len(folds)
    for key in _MATRIX_KEYS:
        a = np.asarray(batched[key], dtype=np.float64)
        b = np.asarray(serial[key], dtype=np.float64)
        rel = float(np.abs(a - b).max()) / (float(np.abs(b).max()) + 1e-12)
        # 1e-5 headroom over the expected ~1e-7: d_map's serial num/den are fp32
        # (the batched path computes them in fp64); everything else ~1e-12.
        assert rel <= 1e-5, (key, rel)


def test_batched_similarity_diagonal_conventions():
    """Serial exact-diagonal identities survive the Gram-trick rewrite."""
    data = _tiny_store(seed=1)
    cache = _StubCache(data)
    fams = sorted(data)
    folds = [[0, 1, 2, 3, 4, 5], [6, 7, 8, 9, 10, 11]]
    out = fits.similarity_at_layer(cache, fams, 0, folds, solver="fast")
    s_agree = np.asarray(out["S_agree"])
    dmap_one_minus = np.asarray(out["S_dmap_one_minus"])
    # serial: r2(p_i, p_i) has ss_res exactly 0 -> a_ii = 1; d_map_ii = 0
    assert np.allclose(np.diag(s_agree), 1.0, atol=1e-12)
    assert np.allclose(np.diag(dmap_one_minus), 1.0, atol=1e-12)
