"""#1320 parity gate: batched ``run_mlp_secondary`` vs the frozen serial oracle.

Exercises the LIVE dispatched entrypoint (``issue825_fit_cells.run_mlp_secondary``,
the #779 hollow-gate rule) against the byte-identical pre-#1320 serial loop
(``_run_mlp_secondary_serial_reference``) on seeded tiny data:

  - REQUIRED case with target dim D=80 > pca_k=64 so the PCA branch is ACTIVE
    on both paths (the device-routed torch-SVD prep is exactly where #931's r1
    recipe-drift fixes lived; #931's own test uses h=80 for the same reason).
  - real-signal floor: serial-reference pooled r2_obs > 0.5 on >=1 layer BEFORE
    the delta comparison (two ~0-R^2 noise fits would satisfy |delta| <= 0.02
    vacuously — the #931 test precedent).
  - tolerances: pooled per-layer |delta r2_obs| <= 0.02, per-draw
    |delta r2_null| <= 0.02 (Source: #931 G1b bar); per-fold r2_obs_folds at
    0.05 (informational — smaller per-fold n, higher variance).
  - tombstone: FutureWarning at call + RuntimeError under
    EPM_FORBID_SERIAL_FITS=1 (raised before any compute).

CPU-only, tiny shapes, no GPU, no network; writes only to tmp_path.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import issue825_fit_cells as fc

POOLED_TOL = 0.02  # #931 G1b bar — not a free knob (plan #1320 kill-criterion 1)
FOLD_TOL = 0.05  # informational per-fold bound (smaller n per fold)


def _make_res(n: int, n_layers: int, d_in: int, d_out: int, seed: int = 0) -> dict:
    """Seeded near-linear low-noise (N, L, D) fixture with real per-layer signal.

    Shape follows tests/test_issue931_mlp_parity.py::_parity_data (which
    achieves r2 > 0.5 under the same recipe); conv_ids are unique per row so
    _cv_folds splits by rows. Target SCALE is deliberately ~5x that fixture
    (signal 0.5, noise 0.1): the batched-vs-serial residual is the init draw
    (key-seeded vs global manual_seed), and on permuted NULL targets the
    init's fixed-magnitude output field enters r2 as (init/target-scale)^2 —
    at the #931 scale the per-draw null deltas ran ~0.02-0.03 (over the G1b
    bar with no real drift; measured 2026-07-15), at 5x they sit <=~0.007.
    """
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, n_layers, d_in)).astype(np.float32)
    B = rng.standard_normal((d_in, d_out)).astype(np.float32)
    Y = np.empty((n, n_layers, d_out), dtype=np.float32)
    for li in range(n_layers):
        Y[:, li, :] = X[:, li, :] @ B * 0.5 + 0.1 * rng.standard_normal((n, d_out)).astype(
            np.float32
        )
    conv_ids = np.array([f"c{i:03d}" for i in range(n)])
    return {"xy": {"X": X, "Y": Y, "conv_ids": conv_ids}}


def _run_both(res: dict, tmp_path: Path, *, n_folds: int, n_null: int) -> tuple[dict, dict]:
    """Run the serial oracle + the live batched entrypoint; return both mlp blocks."""
    serial_dir = tmp_path / "serial"
    batched_dir = tmp_path / "batched"
    with pytest.warns(FutureWarning, match="superseded by the batched run_mlp_secondary"):
        fc._run_mlp_secondary_serial_reference(
            res, serial_dir, cell_id="parity", n_folds=n_folds, seed=0, n_null=n_null
        )
    fc.run_mlp_secondary(res, batched_dir, cell_id="parity", n_folds=n_folds, seed=0, n_null=n_null)
    serial = json.loads((serial_dir / "cells_parity.json").read_text())
    batched = json.loads((batched_dir / "cells_parity.json").read_text())
    assert serial["mlp_budget_exhausted"] is False
    assert batched["mlp_budget_exhausted"] is False
    return serial["mlp"], batched["mlp"]


def _assert_parity(serial_mlp: dict, batched_mlp: dict, *, n_null: int) -> None:
    assert set(serial_mlp) == set(batched_mlp) and serial_mlp, (
        set(serial_mlp),
        set(batched_mlp),
    )
    # Real-signal floor FIRST (the #931 precedent): the serial reference must
    # actually learn on >=1 layer, else the delta comparison is vacuous.
    serial_obs = {li: blk["r2_obs"] for li, blk in serial_mlp.items()}
    assert max(serial_obs.values()) > 0.5, serial_obs
    for li, sblk in serial_mlp.items():
        bblk = batched_mlp[li]
        assert abs(bblk["r2_obs"] - sblk["r2_obs"]) <= POOLED_TOL, (li, sblk, bblk)
        assert len(bblk["r2_null"]) == len(sblk["r2_null"]) == n_null, (li, sblk, bblk)
        for d, (sn, bn) in enumerate(zip(sblk["r2_null"], bblk["r2_null"], strict=True)):
            assert abs(bn - sn) <= POOLED_TOL, (li, d, sn, bn)
        assert len(bblk["r2_obs_folds"]) == len(sblk["r2_obs_folds"]), (li, sblk, bblk)
        for k, (sf, bf) in enumerate(zip(sblk["r2_obs_folds"], bblk["r2_obs_folds"], strict=True)):
            assert abs(bf - sf) <= FOLD_TOL, (li, k, sf, bf)
        assert sblk["budget_hit_folds"] == [] and bblk["budget_hit_folds"] == []


def test_batched_matches_serial_reference_pca_active(tmp_path):
    """REQUIRED case: D=80 > pca_k=64 — the PCA branch is ACTIVE on both paths.

    L=20 layers keeps frozen layers {14, 18, 19} in range (26 excluded), so
    >=2 layers x 3 folds x 2 null draws are compared.
    """
    res = _make_res(n=150, n_layers=20, d_in=12, d_out=80, seed=0)
    layers = [v for v in fc.FROZEN_LAYERS if v < 20]
    assert len(layers) >= 2, layers  # the >=2-layer requirement is structural
    serial_mlp, batched_mlp = _run_both(res, tmp_path, n_folds=3, n_null=2)
    assert set(serial_mlp) == {str(li) for li in layers}
    _assert_parity(serial_mlp, batched_mlp, n_null=2)


def test_batched_matches_serial_reference_pca_skipped(tmp_path):
    """Companion case: D=16 <= pca_k — the PCA-skip (comps=None) branch."""
    res = _make_res(n=48, n_layers=15, d_in=12, d_out=16, seed=1)
    serial_mlp, batched_mlp = _run_both(res, tmp_path, n_folds=3, n_null=1)
    assert set(serial_mlp) == {"14"}
    _assert_parity(serial_mlp, batched_mlp, n_null=1)


def test_serial_reference_tombstone_env_raises_before_compute(tmp_path, monkeypatch):
    """EPM_FORBID_SERIAL_FITS=1 raises at entry — res={} proves no compute ran."""
    monkeypatch.setenv("EPM_FORBID_SERIAL_FITS", "1")
    with pytest.warns(FutureWarning), pytest.raises(RuntimeError, match="EPM_FORBID_SERIAL_FITS"):
        # An empty res would KeyError at xy access — the RuntimeError firing
        # instead proves the guard precedes any compute.
        fc._run_mlp_secondary_serial_reference(
            {}, tmp_path, cell_id="x", n_folds=3, seed=0, n_null=1
        )
