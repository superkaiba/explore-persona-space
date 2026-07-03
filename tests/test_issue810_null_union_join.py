#!/usr/bin/env python3
# ruff: noqa: RUF002, RUF003
# Intentional Unicode (×, →, —) in scientific docstrings + comments (same waiver
# as tests/test_issue810_batched_null.py).
"""Issue #810 `_he` round — collision-safe multi-matrix null-union invariants.

Pins the plan v15 Must-Fix union contract in ``issue810_fit_reconstruction.py``
(persisted concern ``he-union-invariant-test``):

1. The union is COMPOSITE-keyed ``(matrix_id, summary, layer)`` — two matrices
   sharing the same bare ``(summary, layer)`` name (the production case: the 9
   fresh empty rows reuse committed row names im_end/turn_nl) BOTH survive; a
   bare-name-keyed join would silently overwrite and SHRINK the 55-row family.
2. A genuine duplicate composite key fails loud (RuntimeError).
3. The production count guard: exactly ``UNION_EXPECTED_ROWS == 55`` composite
   rows AND ``55 × len(capture_layers)`` cells (55 × 28 == 1540 in production,
   where capture_layers has 28 entries) BEFORE any band is emitted — a wrong
   row count or a wrong cell count raises.

Pure-Python / CPU, tiny synthetic matrices; the loaders, parity gates, and the
band re-reduction are monkeypatched so no HF / disk / GPU is touched — the REAL
``_union_cells_from`` + ``_null_join_and_bands_multi`` guard code runs.
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

_SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"
_SRC = Path(__file__).resolve().parent.parent / "src"
for p in (str(_SCRIPTS), str(_SRC)):
    if p not in sys.path:
        sys.path.insert(0, p)

import issue810_fit_reconstruction as recon  # noqa: E402

N_PERMS = 3
LAYERS = [18, 21]  # 2 synthetic capture layers (production: 28)


def _matrix(rows: list[str], layers: list[int], n_perms: int = N_PERMS, base: float = 0.0) -> dict:
    """A tiny per-draw matrix: {summary: {layer_str: [n_perms draws]}}."""
    return {s: {str(la): [base + i for i in range(n_perms)] for la in layers} for s in rows}


# ─────────────────────────────────────────────────────────────────────────────
# 1. Collision safety: same bare (summary, layer) under DIFFERENT matrix_ids
# ─────────────────────────────────────────────────────────────────────────────
def test_same_bare_name_under_different_matrix_ids_both_survive():
    """A bare-name-keyed join would overwrite; the composite key keeps BOTH cells."""
    cells: dict = {}
    recon._union_cells_from(
        cells, "null_matrix_reconstruction", {"im_end": {"18": [0.1, 0.2, 0.3]}}, N_PERMS
    )
    recon._union_cells_from(
        cells, recon.FRESH_MATRIX_ID, {"im_end": {"18": [1.1, 1.2, 1.3]}}, N_PERMS
    )
    assert set(cells) == {
        ("null_matrix_reconstruction", "im_end", "18"),
        (recon.FRESH_MATRIX_ID, "im_end", "18"),
    }
    np.testing.assert_allclose(
        cells[("null_matrix_reconstruction", "im_end", "18")], [0.1, 0.2, 0.3]
    )
    np.testing.assert_allclose(cells[(recon.FRESH_MATRIX_ID, "im_end", "18")], [1.1, 1.2, 1.3])


# ─────────────────────────────────────────────────────────────────────────────
# 2. Genuine duplicate composite key (matrix_id, summary, layer) fails loud
# ─────────────────────────────────────────────────────────────────────────────
def test_duplicate_composite_key_raises():
    cells: dict = {}
    m = {"uh_mean3": {"21": [0.0, 1.0, 2.0]}}
    recon._union_cells_from(cells, "m", m, N_PERMS)
    with pytest.raises(RuntimeError, match=r"duplicate composite key"):
        recon._union_cells_from(cells, "m", m, N_PERMS)


def test_incomplete_draw_cells_are_skipped_not_stored():
    """A cell whose draws list != n_perms is skipped (feeds the cell-count guard)."""
    cells: dict = {}
    recon._union_cells_from(
        cells, "m", {"mean": {"18": [0.0, 1.0], "21": [0.0, 1.0, 2.0]}}, N_PERMS
    )
    assert set(cells) == {("m", "mean", "21")}


# ─────────────────────────────────────────────────────────────────────────────
# 3. Production count guard via the REAL _null_join_and_bands_multi code path
# ─────────────────────────────────────────────────────────────────────────────
def _run_multi_join(monkeypatch, committed_by_id: dict, fresh: dict, captured: dict | None = None):
    """Drive _null_join_and_bands_multi with loaders/gates/band-reduce mocked.

    The union-building + count-guard code (the invariant under test) runs REAL;
    only IO (committed-matrix load, uh pack) + parity gates + the downstream band
    re-reduction are faked. Returns the function's result dict (or raises).
    """

    def _fake_load(args, path):
        mid = Path(path).stem
        return mid, committed_by_id[mid]

    monkeypatch.setattr(recon, "_load_committed_matrix", _fake_load)
    monkeypatch.setattr(recon, "_load_uh_pack_matrix", lambda *a, **k: {})
    monkeypatch.setattr(recon, "_parity_gate", lambda *a, **k: {"pass": True})
    monkeypatch.setattr(recon, "_parity_gate_uh", lambda *a, **k: {"pass": True})

    def _fake_bands(args, cells, committed, null_matrix_new, results, diag):
        if captured is not None:
            captured["cells"] = dict(cells)
        return {}

    monkeypatch.setattr(recon, "_band_rows_from_union_multi", _fake_bands)
    args = SimpleNamespace(
        null_join=[f"/fake/{mid}.json" for mid in committed_by_id],
        uh_summaries="/fake/uh_pack",
        n_perms=N_PERMS,
        smoke=False,
    )
    return recon._null_join_and_bands_multi(args, fresh, {}, {}, [], LAYERS, {})


def test_multi_join_production_row_shape_passes_and_keeps_name_collisions(monkeypatch):
    """55 composite rows × len(LAYERS) cells passes; colliding fresh names survive."""
    assert recon.UNION_EXPECTED_ROWS == 55  # 37 round-1 + 9 round-3 + 9 fresh (× 28 L = 1540)
    committed_rows = ["mean", "im_end", "turn_nl"] + [f"row_{i}" for i in range(43)]  # 46
    fresh_rows = ["im_end", "turn_nl"] + [f"fresh_{i}" for i in range(7)]  # 9, 2 name-colliding
    captured: dict = {}
    out = _run_multi_join(
        monkeypatch,
        {"cm1": _matrix(committed_rows, LAYERS)},
        _matrix(fresh_rows, LAYERS, base=5.0),
        captured,
    )
    assert out["mode"] == "union_join_multi"
    assert out["parity_gate"]["pass"] is True
    cells = captured["cells"]
    assert len(cells) == recon.UNION_EXPECTED_ROWS * len(LAYERS)  # 55 × 2 == 110
    # the production collision: committed AND fresh copies of the same bare name coexist
    assert ("cm1", "im_end", "18") in cells
    assert (recon.FRESH_MATRIX_ID, "im_end", "18") in cells


def test_multi_join_wrong_row_count_raises(monkeypatch):
    """54 != 55 composite rows fails loud BEFORE any band is emitted."""
    committed_rows = ["mean"] + [f"row_{i}" for i in range(44)]  # 45
    fresh_rows = [f"fresh_{i}" for i in range(9)]  # 45 + 9 == 54 rows
    with pytest.raises(RuntimeError, match=r"union row count 54 != 55"):
        _run_multi_join(
            monkeypatch, {"cm1": _matrix(committed_rows, LAYERS)}, _matrix(fresh_rows, LAYERS)
        )


def test_multi_join_wrong_cell_count_raises(monkeypatch):
    """55 rows but a missing (row × layer) cell → n_union_cells != 55 × n_layers raises."""
    committed_rows = ["mean"] + [f"row_{i}" for i in range(45)]  # 46
    fresh = _matrix([f"fresh_{i}" for i in range(9)], LAYERS, base=5.0)  # 46 + 9 == 55 rows
    fresh["fresh_8"]["21"] = [0.0, 1.0]  # 2 != n_perms draws → skipped → 109 != 110 cells
    with pytest.raises(RuntimeError, match=r"n_union_cells 109"):
        _run_multi_join(monkeypatch, {"cm1": _matrix(committed_rows, LAYERS)}, fresh)


def test_multi_join_duplicate_matrix_id_raises(monkeypatch):
    """Two --null-join paths resolving to the same matrix id fail loud."""

    def _fake_load(args, path):
        return "cm1", _matrix(["mean"], LAYERS)

    monkeypatch.setattr(recon, "_load_committed_matrix", _fake_load)
    monkeypatch.setattr(recon, "_load_uh_pack_matrix", lambda *a, **k: {})
    args = SimpleNamespace(
        null_join=["/a/cm1.json", "/b/cm1.json"], uh_summaries="x", n_perms=N_PERMS, smoke=False
    )
    with pytest.raises(RuntimeError, match=r"duplicate matrix id"):
        recon._null_join_and_bands_multi(args, {}, {}, {}, [], LAYERS, {})


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
