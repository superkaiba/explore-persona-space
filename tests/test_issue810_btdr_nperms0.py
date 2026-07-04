#!/usr/bin/env python3
"""Issue #810 btdr round — the ``--n-perms 0`` no-nulls guard (plan v18 §4.6 item 3).

Pins the guard in ``issue810_fit_reconstruction._fit_grid``
(``if do_loco and args.n_perms > 0:``): with ``n_perms=0`` the permutation
battery is skipped entirely (an EMPTY per-summary null dict) while the LOCO
point fits still run. Pre-fix (the parent's bare ``if do_loco:``), ``n_perms=0``
crashed inside ``_fit_null_draws`` via ``make_perm_matrix(n, 0, rng)`` ->
``np.stack([])`` ``ValueError: need at least one array to stack`` — pinned here
directly against the unguarded callee so the guard cannot be silently dropped.

Behavior preservation: ``n_perms>=1`` still produces the non-empty per-draw null
path, and the LOCO point skill is identical with and without the null battery.

Pure CPU, tiny synthetic input — no store, no HF, no GPU.
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

import issue810_fit_reconstruction as fr  # noqa: E402

_N_CTX = 8
_H = 16  # summary activation width
_H_C = 12  # c_C predictor width
_CAPTURE_LAYER = 13
_SUMMARY = "turn_nl"  # a position recipe (reads pos_summaries + coverage)


def _tiny_grid_inputs():
    """Synthetic (ctx_ids, pos_summaries, coverage, cc) for one 8-ctx cell."""
    rng = np.random.default_rng(0)
    ctx_ids = [f"c{i}" for i in range(_N_CTX)]
    pos_summaries = {c: {_SUMMARY: [rng.normal(size=_H).astype(np.float64)]} for c in ctx_ids}
    coverage = {c: {_SUMMARY: 1} for c in ctx_ids}
    cc = {c: [rng.normal(size=_H_C).astype(np.float64)] for c in ctx_ids}
    return ctx_ids, pos_summaries, coverage, cc


def _run_fit_grid(n_perms: int):
    """Call the REAL ``_fit_grid`` (the dispatched fit path) on the tiny cell."""
    ctx_ids, pos_summaries, coverage, cc = _tiny_grid_inputs()
    args = SimpleNamespace(n_perms=n_perms, no_mlp=True, device="cpu")
    return fr._fit_grid(
        args,
        [_SUMMARY],
        [0],  # layer indices
        ctx_ids,
        [_CAPTURE_LAYER],
        {},  # free_summaries (unused for position recipes)
        pos_summaries,
        coverage,
        cc,
        None,  # fam_map (LOFO off)
        True,  # do_loco
        False,  # do_lofo
    )


def test_nperms0_skips_null_battery_keeps_loco_skills():
    """n_perms=0: no crash, EMPTY null dict, LOCO point skill still present."""
    results, null_matrix, mlp_jobs, _cell_ref = _run_fit_grid(n_perms=0)
    assert null_matrix == {_SUMMARY: {}}, null_matrix
    cell = results[_SUMMARY][0]
    assert cell["layer"] == _CAPTURE_LAYER
    assert cell["n"] == _N_CTX
    assert isinstance(cell["ridge_skill"], float), cell
    assert mlp_jobs == []  # --no-mlp


def test_nperms1_still_produces_nonempty_null_path():
    """Behavior preservation: n_perms>=1 keeps the per-draw null (same LOCO skill)."""
    results0, _null0, _jobs0, _ = _run_fit_grid(n_perms=0)
    results1, null1, _jobs1, _ = _run_fit_grid(n_perms=1)
    draws = null1[_SUMMARY][str(_CAPTURE_LAYER)]
    assert len(draws) == 1 and isinstance(draws[0], float), null1
    # The point fit is untouched by the null battery (the guard only gates nulls).
    assert results1[_SUMMARY][0]["ridge_skill"] == results0[_SUMMARY][0]["ridge_skill"]


def test_unguarded_fit_null_draws_crashes_at_nperms0():
    """Why the guard exists: the unguarded callee crashes on n_perms=0 (pre-fix path)."""
    ctx_ids, pos_summaries, _coverage, cc = _tiny_grid_inputs()
    Yv = np.stack([pos_summaries[c][_SUMMARY][0] for c in ctx_ids])
    Xc = np.stack([cc[c][0] for c in ctx_ids])
    with pytest.raises(ValueError, match="at least one array"):
        fr._fit_null_draws(Xc, Yv, 4, 0, fr.SHUFFLE_NULL_SEED, device="cpu")
