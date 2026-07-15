"""#1332 r5 regression pins — i545 parity-gate / pilot fold derivation.

Pre-fix, the i545 arm reused the marker arm's shared 400-query-bank folds:
i545 units carry their OWN row indexing (~8..400 rows per unit), so a bank
fold could miss a unit's index range entirely and ``parity_gate`` crashed
with ``ValueError: zero-size array to reduction operation maximum``
(``np.abs(ref).max()`` on an empty eval split). The fix derives per-unit
folds from each unit's own row count (``folds=None``) and fail-louds on any
empty split — the gate is never skipped silently.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import issue1332_fits as F


class _FakeCache:
    """Minimal ShardCache stand-in: per-fam row counts, deterministic arrays.

    Mirrors the real interface parity_gate/run_pilot dereference:
    ``bank_indices`` (row->bank ids; i545 shards fall back to range(n_rows))
    and ``arrays`` returning (n_rows, len(layers), H) fp32.
    """

    def __init__(self, n_by_fam: dict[str, int], hidden: int = 16):
        self.n_by_fam = n_by_fam
        self.hidden = hidden

    def bank_indices(self, fam: str) -> list[int]:
        return list(range(self.n_by_fam[fam]))

    def arrays(self, fam: str, key: str, layers: list[int]) -> np.ndarray:
        rng = np.random.default_rng(abs(hash((fam, key))) % (2**32))
        return rng.normal(size=(self.n_by_fam[fam], len(layers), self.hidden)).astype(np.float32)


def test_parity_gate_i545_derives_folds_from_unit_row_count():
    """folds=None (i545): per-unit folds from each unit's own n -> gate runs.

    Pre-fix this call shape did not exist; the equivalent production call
    (shared 400-bank folds against 8-row units) crashed on an empty eval
    split (the r5 incident).
    """
    cache = _FakeCache({"col__tiny": 8, "row__mid": 40, "row__big": 120})
    gate = F.parity_gate(cache, ["col__tiny", "row__mid", "row__big"], [0], None)
    assert set(gate) == {"max_rel_diff", "tolerance", "pass", "solver"}
    assert np.isfinite(gate["max_rel_diff"])


def test_parity_gate_empty_split_raises_never_silent():
    """A fold that misses a unit's row-index range raises loud (never skipped).

    Pre-fix the same condition surfaced as an opaque numpy ValueError
    (zero-size reduction); the guard names the cell and the split sizes.
    """
    cache = _FakeCache({"col__tiny": 8})
    bad_folds = [list(range(200, 240))]  # marker-bank-style indices; no overlap with range(8)
    with pytest.raises(RuntimeError, match="empty split"):
        F.parity_gate(cache, ["col__tiny"], [0], bad_folds)


def test_run_pilot_i545_pilots_largest_unit():
    """folds=None (i545): the pilot fam is the LARGEST unit (conservative basis)."""
    cache = _FakeCache({"col__tiny": 8, "row__big": 60})
    pilot = F.run_pilot(cache, ["col__tiny", "row__big"], n_layers=2, folds=None, solver="fast")
    # KFold(5) val fold on n=60 has 12 rows -> train 48; an 8-row pilot would show n_train <= 7
    assert pilot["n_train"] + len(F.C.query_folds(60)[0]) == 60
    assert pilot["n_train"] > 8
