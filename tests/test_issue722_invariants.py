#!/usr/bin/env python3
# ruff: noqa: RUF002, RUF003
# Intentional Unicode (Δ, ρ, ×) in scientific docstrings + comments.
"""Issue #722 round-2 regression tests — the four substantive BLOCKER fixes.

Each test trips a permanent invariant added in round 2 and would FAIL against the
round-1 code (row-i.i.d. floors / a `_h_call` that ignored the excluded CI / a
missing paired ρ-shift CI / a kill criterion that counted clean H_input toward
the straddle). Pure-Python, no GPU / no HF — exercises the helpers directly.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

_SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"
_SRC = Path(__file__).resolve().parent.parent / "src"
for p in (str(_SCRIPTS), str(_SRC)):
    if p not in sys.path:
        sys.path.insert(0, p)

import issue722_analyze as analyze  # noqa: E402
import issue722_bootstrap as boot  # noqa: E402


# ─────────────────────────────────────────────────────────────────────────────
# MF#6 — make_refit_pair is family-clustered (resamples whole families), not row-iid
# ─────────────────────────────────────────────────────────────────────────────
def test_make_refit_pair_requires_families_arg():
    """The families argument is mandatory + positional (round-1 had no such arg)."""
    import inspect

    params = inspect.signature(boot.make_refit_pair).parameters
    assert "families" in params, "make_refit_pair must take a `families` arg (MF#6)"
    # positional-or-keyword, before the keyword-only block
    assert params["families"].kind is inspect.Parameter.POSITIONAL_OR_KEYWORD


def test_make_refit_pair_resamples_whole_families():
    """A family-clustered resample only ever selects WHOLE families' rows together.

    Construct 3 families with disjoint, easily-identified rows. A fit_fn that
    records which row indices it received lets us assert every recorded resample
    is a union of complete family blocks — impossible under a row-iid bootstrap
    (which would mix partial families).
    """
    # 3 families, sizes 4 / 3 / 5; X rows carry their family id in column 0 so the
    # fit_fn can recover them from the resampled X it is handed.
    fam_blocks = {"A": [0, 1, 2, 3], "B": [4, 5, 6], "C": [7, 8, 9, 10, 11]}
    n = 12
    families = [""] * n
    fam_id = np.zeros(n)
    for fid, (fam, rows) in enumerate(fam_blocks.items()):
        for r in rows:
            families[r] = fam
            fam_id[r] = fid
    X = np.zeros((n, 2))
    X[:, 0] = fam_id  # column 0 = family id, recoverable after resample
    Y = np.random.default_rng(0).normal(size=(n, 3))
    grid = np.zeros((2, 3))  # tiny grid; fit_fn ignores it (returns zeros)
    r_hat = np.zeros(3)

    seen_family_id_multisets = []

    def _fit_fn(Xb, _Yb, _rng):
        # Record the family ids of the rows this refit received.
        seen_family_id_multisets.append(np.sort(Xb[:, 0]).tolist())
        return np.zeros((grid.shape[0], 3))

    boot.make_refit_pair(X, Y, _fit_fn, grid, r_hat, families, n_pairs=20)

    # Every recorded resample must be a concatenation of WHOLE family blocks: the
    # count of each family id present is a multiple of that family's block size.
    block_size = {float(fid): len(rows) for fid, rows in enumerate(fam_blocks.values())}
    assert seen_family_id_multisets, "fit_fn never called"
    for sample in seen_family_id_multisets:
        arr = np.asarray(sample)
        for fid, size in block_size.items():
            cnt = int((arr == fid).sum())
            assert cnt % size == 0, (
                f"family {fid} appeared {cnt} times (not a multiple of its block "
                f"size {size}) — resample split a family (row-iid, not clustered)"
            )


def test_make_refit_pair_single_family_falls_back_to_row_bootstrap():
    """<2 families → i.i.d. row bootstrap (clustering is a no-op); still runs."""
    n = 8
    X = np.random.default_rng(1).normal(size=(n, 2))
    Y = np.random.default_rng(2).normal(size=(n, 3))
    grid = np.zeros((2, 2))
    r_hat = np.zeros(3)
    out = boot.make_refit_pair(
        X, Y, lambda Xb, Yb, rng: np.zeros((grid.shape[0], 3)), grid, r_hat, ["A"] * n, n_pairs=5
    )
    assert out.shape == (5,)


# ─────────────────────────────────────────────────────────────────────────────
# MF#5 — _h_call consults the large-shift-excluded CI and flips H_function
# ─────────────────────────────────────────────────────────────────────────────
def _cell(full_ci, excl_ci, floor):
    return {
        "Delta_med_ci": full_ci,
        "Delta_med_excl_large_shift_ci": excl_ci,
        "floor_combined": floor,
    }


def test_h_call_downgrades_when_excluded_ci_flips_h_function():
    """Full CI above floor (H_function) but excluded CI straddles → downgrade + flip flag."""
    cell = _cell(
        full_ci={"ci_lo": 1.5, "ci_hi": 2.5},  # entirely above floor → H_function
        excl_ci={"ci_lo": 0.5, "ci_hi": 2.5},  # straddles floor → H_mixed
        floor=1.0,
    )
    call, flip = analyze._h_call(cell)
    assert call == "H_mixed", call
    assert flip is True


def test_h_call_keeps_h_function_when_excluded_agrees():
    """Both CIs above floor → H_function stands, no flip."""
    cell = _cell({"ci_lo": 1.5, "ci_hi": 2.5}, {"ci_lo": 1.2, "ci_hi": 2.4}, floor=1.0)
    call, flip = analyze._h_call(cell)
    assert call == "H_function"
    assert flip is False


def test_h_call_no_excl_ci_does_not_flip():
    cell = {"Delta_med_ci": {"ci_lo": 1.5, "ci_hi": 2.5}, "floor_combined": 1.0}
    call, flip = analyze._h_call(cell)
    assert call == "H_function"
    assert flip is False


# ─────────────────────────────────────────────────────────────────────────────
# MF#4 — paired ρ-shift CI exists and behaves
# ─────────────────────────────────────────────────────────────────────────────
def test_paired_rho_diff_ci_present_and_brackets_point():
    """The paired CI helper returns a finite point + bracketing CI on a real shift."""
    import issue722_fit_M as fit_M  # imports torch; only needed for this test

    rng = np.random.default_rng(7)
    n = 24
    families = [f"fam{i % 6}" for i in range(n)]  # 6 families
    E = rng.normal(size=n)
    # M0 chain weakly correlated with E; M+ chain strongly correlated → positive shift.
    chain_m0 = 0.2 * E + rng.normal(size=n)
    chain_mplus = 0.9 * E + 0.1 * rng.normal(size=n)
    ci = fit_M._clustered_paired_rho_diff_ci(chain_m0, chain_mplus, E, families, n_resamples=300)
    assert ci["point"] is not None
    assert ci["ci_lo"] <= ci["point"] <= ci["ci_hi"]
    assert ci["point"] > 0  # M+ correlates more strongly → positive ρ-shift
    assert ci["n_families"] == 6


def test_paired_rho_diff_ci_degenerate_input_returns_none_point():
    """Zero-variance chains → _rho is None → point-None CI, no crash."""
    import issue722_fit_M as fit_M

    n = 8
    families = ["A"] * 4 + ["B"] * 4
    E = np.arange(n, dtype=float)
    const = np.ones(n)  # zero variance → _rho returns None
    ci = fit_M._clustered_paired_rho_diff_ci(const, const, E, families)
    assert ci["point"] is None


# ─────────────────────────────────────────────────────────────────────────────
# MF#4 (Claude Major#2) — kill criterion: clean 3×H_input is NOT inconclusive
# ─────────────────────────────────────────────────────────────────────────────
def _h_input_cell(behavior, *, shuffle_pass=True):
    """A clean below-floor H_input cell (full + excluded CI both below floor)."""
    below = {"ci_lo": -0.5, "ci_hi": 0.5}  # entirely <= floor=1.0 → H_input
    fc = {
        "behavior": behavior,
        "layer": analyze.PRIMARY_LAYER,
        "n_cells": 16,
        "Delta_med": 0.2,
        "Delta_med_ci": below,
        "Delta_med_excl_large_shift_ci": below,
        "floor_M0_refit": 0.5,
        "floor_Mplus_refit": 0.5,
        "floor_shifted": 0.5,
        "floor_combined": 1.0,
    }
    # MLP beats shuffle → passing shuffle check (or fails it if shuffle_pass=False)
    cb = {"rho_M0_mlp": 0.6 if shuffle_pass else 0.1, "rho_M0_shuffle": 0.2}
    return fc, cb


def _h_function_cell(behavior):
    above = {"ci_lo": 1.5, "ci_hi": 2.5}  # entirely > floor=1.0 → H_function
    fc = {
        "behavior": behavior,
        "layer": analyze.PRIMARY_LAYER,
        "n_cells": 16,
        "Delta_med": 2.0,
        "Delta_med_ci": above,
        "Delta_med_excl_large_shift_ci": above,
        "floor_M0_refit": 0.5,
        "floor_Mplus_refit": 0.5,
        "floor_shifted": 0.5,
        "floor_combined": 1.0,
    }
    cb = {"rho_M0_mlp": 0.6, "rho_M0_shuffle": 0.2}
    return fc, cb


def _assemble_from_cells(tmp_path, cell_specs):
    """Write per-cell fit JSONs (fc + chain block) and run assemble; return summary."""
    import json

    cells_dir = tmp_path / "cells"
    cells_dir.mkdir()
    for fc, cb in cell_specs:
        # The fit-cell JSON carries the function-change fields at top level + a
        # `chain_rho` block; assemble() reads both.
        rec = dict(fc)
        rec["chain_rho"] = cb
        rec["cross_transfer"] = {}
        (cells_dir / f"{fc['behavior']}_L{fc['layer']}.json").write_text(json.dumps(rec))
    out_dir = tmp_path / "out"
    summary = analyze.assemble(cells_dir, out_dir)
    fc_json = json.loads((out_dir / "function_change.json").read_text())
    return summary, fc_json


def test_kill_criterion_clean_three_h_input_is_not_inconclusive(tmp_path):
    """3×clean H_input with passing shuffle → outcome=H_input_clean, NOT inconclusive (MF#4)."""
    specs = [_h_input_cell(b) for b in analyze.HEADLINE_BEHAVIORS]
    _summary, fc = _assemble_from_cells(tmp_path, specs)
    kill = fc["kill_criterion"]
    assert kill["inconclusive"] is False, kill
    assert kill["outcome"] == "H_input_clean", kill
    assert kill["n_H_input_clean"] == 3


def test_kill_criterion_two_straddle_is_inconclusive(tmp_path):
    """2 H_mixed (large-shift-flipped) + 1 H_function → inconclusive (plan §3 >=2 straddle)."""

    # Build a cell whose full CI is H_function but excluded CI straddles → flips to H_mixed.
    def _flip_cell(behavior):
        fc = {
            "behavior": behavior,
            "layer": analyze.PRIMARY_LAYER,
            "n_cells": 16,
            "Delta_med": 2.0,
            "Delta_med_ci": {"ci_lo": 1.5, "ci_hi": 2.5},  # H_function
            "Delta_med_excl_large_shift_ci": {"ci_lo": 0.5, "ci_hi": 2.5},  # straddle
            "floor_M0_refit": 0.5,
            "floor_Mplus_refit": 0.5,
            "floor_shifted": 0.5,
            "floor_combined": 1.0,
        }
        return fc, {"rho_M0_mlp": 0.6, "rho_M0_shuffle": 0.2}

    specs = [
        _flip_cell("em"),
        _flip_cell("sycophancy"),
        _h_function_cell("fact"),
    ]
    _summary, fc = _assemble_from_cells(tmp_path, specs)
    kill = fc["kill_criterion"]
    assert kill["inconclusive"] is True, kill
    assert kill["outcome"] == "inconclusive"
    assert kill["n_straddle"] == 2


def test_kill_criterion_below_floor_failing_shuffle_counts_as_straddle(tmp_path):
    """A below-floor H_input that FAILS the shuffle check is not a clean H_input."""
    specs = [
        _h_input_cell("em", shuffle_pass=False),
        _h_input_cell("sycophancy", shuffle_pass=False),
        _h_input_cell("fact", shuffle_pass=True),
    ]
    _summary, fc = _assemble_from_cells(tmp_path, specs)
    kill = fc["kill_criterion"]
    # 2 shuffle-failed H_input → straddle >= 2 → inconclusive
    assert kill["inconclusive"] is True, kill
    assert kill["n_straddle"] == 2


def test_kill_criterion_all_h_function_is_clean(tmp_path):
    specs = [_h_function_cell(b) for b in analyze.HEADLINE_BEHAVIORS]
    _summary, fc = _assemble_from_cells(tmp_path, specs)
    kill = fc["kill_criterion"]
    assert kill["inconclusive"] is False
    assert kill["outcome"] == "H_function_clean"


# ─────────────────────────────────────────────────────────────────────────────
# Round-3 — _pca_basis_v0 survives a gesdd SVD non-convergence on a near-singular
# bootstrap resample (the workload-time crash at sycophancy L7; em fit cleanly).
# ─────────────────────────────────────────────────────────────────────────────
def test_pca_basis_v0_recovers_from_svd_nonconvergence(monkeypatch):
    """A LinAlgError from np.linalg.svd (gesdd) is caught + recovered via scipy gesvd.

    The family-clustered bootstrap RESAMPLES (Yb in the floor refit) draw whole
    families with replacement, so a heavily-duplicated draw is mean-centered to a
    rank-deficient matrix that LAPACK gesdd can fail to converge on. Round-2's
    `_pca_basis_v0` called `np.linalg.svd` unguarded and propagated the
    LinAlgError (crashing the whole production run); round-3 falls back to
    `scipy.linalg.svd(lapack_driver='gesvd')`. This test forces the gesdd choke
    via monkeypatch (the trigger is LAPACK-version-fragile, so we simulate the
    raise rather than hunt a specific choking matrix) and asserts the fix
    recovers with the correct shape AND a basis numerically equal to the gesvd
    reference. It FAILS against round-2 (the LinAlgError propagates).
    """
    import issue722_fit_M as fit_M
    import numpy.linalg as nla

    # A finite, heavily rank-deficient resample: 24 rows but only 3 distinct
    # (exact duplicates — what a family-clustered resample-with-replacement
    # produces when 3 families are drawn repeatedly).
    rng = np.random.default_rng(0)
    base = rng.standard_normal((3, 64))
    V = base[rng.integers(0, 3, size=24)]

    real_svd = nla.svd
    calls = {"n": 0}

    def _flaky_svd(a, *args, **kw):
        calls["n"] += 1
        raise np.linalg.LinAlgError("SVD did not converge")  # simulate gesdd choke

    monkeypatch.setattr(nla, "svd", _flaky_svd)
    basis = fit_M._pca_basis_v0(V, 4)

    assert calls["n"] == 1, "np.linalg.svd should be attempted exactly once"
    assert basis.shape == (4, 64), basis.shape  # signature/return shape preserved

    # The fallback computes the SAME SVD of the SAME centered matrix → directions
    # match the gesvd reference up to sign.
    from scipy.linalg import svd as _scipy_svd

    Vc = V - V.mean(axis=0, keepdims=True)
    _, _, Vt_ref = _scipy_svd(Vc, full_matrices=False)
    ref = Vt_ref[:4]
    cos = np.abs(np.sum(basis * ref, axis=1)) / (
        np.linalg.norm(basis, axis=1) * np.linalg.norm(ref, axis=1) + 1e-30
    )
    assert cos.min() >= 0.999, f"recovered basis diverges from gesvd reference: {cos}"


def test_pca_basis_v0_clean_input_does_not_use_fallback(monkeypatch):
    """On a well-conditioned input the gesdd path succeeds — the fallback is never
    reached, so the common (clean) path is unchanged (the em JSONs stay valid)."""
    import issue722_fit_M as fit_M
    import scipy.linalg as sla

    sp_calls = {"n": 0}
    real_sp = sla.svd

    def _count_sp(a, *args, **kw):
        sp_calls["n"] += 1
        return real_sp(a, *args, **kw)

    monkeypatch.setattr(sla, "svd", _count_sp)
    V = np.random.default_rng(3).standard_normal((24, 64))  # full-rank, well-conditioned
    basis = fit_M._pca_basis_v0(V, 4)
    assert basis.shape == (4, 64)
    assert sp_calls["n"] == 0, "scipy gesvd fallback fired on a clean input (should not)"


# ─────────────────────────────────────────────────────────────────────────────
# MF#3 — the co-primary deliverable is written under the plan §6.5 name
# ─────────────────────────────────────────────────────────────────────────────
def test_chain_rho_deliverable_filename(tmp_path):
    specs = [_h_function_cell(b) for b in analyze.HEADLINE_BEHAVIORS]
    _summary, _fc = _assemble_from_cells(tmp_path, specs)
    out_dir = tmp_path / "out"
    assert (out_dir / "chain_rho_M0_Mplus.json").exists(), "MF#3: plan §6.5 deliverable name"
    assert not (out_dir / "chain_rho.json").exists(), "old filename must not be written"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
