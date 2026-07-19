"""Issue #1482 pdshrink round pins (plan v12): per-direction lambda selection math on
synthetic data, the fp16 holdout-scoring convention, the Statistics MF-1 tuned-arm
gather/store consistency (incl. the parent-lambda-slice sketch bug the assert must
catch), the K1 HALT plumbing (gates.json written FIRST -> SystemExit(23)), the
lambda-grid dedup semantics, and the batched-bootstrap identities. CPU-only; no
network, no model loads; real numpy/torch bodies throughout (no mocks)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "src"))

import issue779_ffc_n1m_fits as N1M  # noqa: E402
import issue1482_error_analysis as D  # noqa: E402

ARANGE = np.arange(4)


def _synthetic_tables(scale: float = 50.0, n: int = 40, n_dir: int = 4, n_lam: int = 5):
    """Synthetic per-lambda holdout predictions with a PLANTED per-direction optimum:
    direction d's prediction error is smallest at lambda index (d % n_lam). Full-dim
    D_full=8 predictions projected through an orthonormal 8x4 basis, scored through
    the REAL _score_holdout_projection (fp16 convention)."""
    rng = np.random.default_rng(1482)
    d_full = 8
    top = np.eye(d_full)[:, :n_dir]  # orthonormal columns
    yh_rot = scale * rng.standard_normal((n, n_dir))
    yh_full = yh_rot @ top.T
    planted = np.asarray([d % n_lam for d in range(n_dir)], np.int64)
    hold_tab = np.full((n_lam, n_dir), np.nan)
    proj_tabs = np.empty((n_lam, n, n_dir), dtype=np.float16)
    ph_by_lam = []
    for i in range(n_lam):
        # error scale per direction: small (0.05*scale) at the planted index, large away
        err_scale = np.where(planted == i, 0.05, 1.0) * scale
        ph = yh_full + (rng.standard_normal((n, n_dir)) * err_scale) @ top.T
        ph_by_lam.append(ph)
        proj_tabs[i], hold_tab[i] = D._score_holdout_projection(ph, top, yh_rot)
    # val table planted directly: argmax over it must recover the planted indices
    val_tab = -np.abs(rng.standard_normal((n_lam, n_dir)))
    val_tab[planted, ARANGE[:n_dir]] = 1.0
    return {
        "top": top,
        "yh_rot": yh_rot,
        "val_tab": val_tab,
        "hold_tab": hold_tab,
        "proj_tabs": proj_tabs,
        "planted": planted,
        "ph_by_lam": ph_by_lam,
    }


# ── lambda grid ──────────────────────────────────────────────────────────────────


def test_lambda_grid_dedup_sorted_and_membership():
    grid = D._pdshrink_lambda_grid()
    assert np.all(np.diff(grid) > 0), "grid must be strictly increasing (deduped)"
    union = (
        {float(v) for v in np.logspace(-5, -2, 16)} | {float(v) for v in N1M.LAMBDAS_N1M} | {1e-9}
    )
    # dedup semantics (no literal-count hard-assert — plan §4): len == |union set|
    assert len(grid) == len(union)
    assert set(grid.tolist()) == union
    assert 1e-9 in set(grid.tolist())
    assert 24 <= len(grid) <= 40  # sanity band around the expected ~38


# ── selection math + MF-1 gather/store consistency ───────────────────────────────


def test_selection_recovers_planted_optimum():
    s = _synthetic_tables()
    val_sel = np.where(np.isfinite(s["val_tab"]), s["val_tab"], -np.inf)
    sel = val_sel.argmax(0)
    assert np.array_equal(sel, s["planted"])
    r2_tuned = s["hold_tab"][sel, ARANGE]
    # the planted-optimum lambda also has the best HOLDOUT R2 per direction here
    assert np.array_equal(s["hold_tab"].argmax(0), s["planted"])
    assert np.all(r2_tuned >= s["hold_tab"].min(0))


def test_mf1_gather_store_consistency_exact():
    s = _synthetic_tables()
    sel = s["planted"]
    tuned = D._gather_tuned(s["proj_tabs"], sel)
    r2_check = D._per_feature_metrics(tuned.astype(np.float64), s["yh_rot"])["r2"]
    diff = np.max(np.abs(r2_check - s["hold_tab"][sel, ARANGE]))
    assert diff <= 1e-6, diff  # the MF-1 runtime assert, mirrored


def test_mf1_assert_catches_parent_lambda_slice_bug():
    """The plan-sketch slip (persisting the PARENT-lambda slice instead of the
    per-direction gather) must FAIL the MF-1 consistency check."""
    s = _synthetic_tables()
    sel = s["planted"]
    i_parent = 0  # planted sel differs from 0 for directions 1..3
    assert np.any(sel != i_parent)
    wrong_store = s["proj_tabs"][i_parent]  # the bug: a single-lambda slice
    r2_wrong = D._per_feature_metrics(wrong_store.astype(np.float64), s["yh_rot"])["r2"]
    diff = np.max(np.abs(r2_wrong - s["hold_tab"][sel, ARANGE]))
    assert diff > 1e-6, "parent-lambda slice must trip the MF-1 consistency assert"


# ── fp16 holdout-scoring convention ──────────────────────────────────────────────


def test_fp16_convention_scoring_side_only():
    """Predictions cast through fp16, projections stored fp16, R2 computed FROM the
    stored fp16 values (exactly reproducible from the persisted store); the pure-fp64
    projection R2 differs (nonzero fp16 effect) but only at ~1e-6 scale."""
    rng = np.random.default_rng(7)
    n, d_full, n_dir = 60, 8, 4
    # non-trivial orthonormal basis: the projection MIXES columns, so fp16-storing
    # the projection is a real quantization (an identity basis would hide it)
    q, _ = np.linalg.qr(rng.standard_normal((d_full, d_full)))
    top = q[:, :n_dir]
    yh_rot = 100.0 * rng.standard_normal((n, n_dir))
    ph = yh_rot @ top.T + 5.0 * rng.standard_normal((n, d_full))
    proj16, r2 = D._score_holdout_projection(ph, top, yh_rot)
    assert proj16.dtype == np.float16
    r2_from_store = D._per_feature_metrics(proj16.astype(np.float64), yh_rot)["r2"]
    assert np.array_equal(r2_from_store, r2)  # store reproduces the table EXACTLY
    r2_fp64_proj = D._per_feature_metrics(ph.astype(np.float16).astype(np.float64) @ top, yh_rot)[
        "r2"
    ]
    diff = np.max(np.abs(r2_fp64_proj - r2))
    assert 0.0 < diff < 1e-3, diff  # fp16 projection quantization: visible, tiny


# ── G1 verdict lattice + K1 HALT plumbing ────────────────────────────────────────

_PASSING_GATES = {
    "split_sha_match": True,
    "eigval_med_rel": 5e-4,
    "ridge_med_abs_delta": 5e-4,
    "ridge_spearman": 0.9999,
    "mlp_med_abs_delta": 5e-4,
}


def test_gates_pass_lattice():
    assert D._pdshrink_gates_pass(_PASSING_GATES)
    for key, bad in (
        ("split_sha_match", False),
        ("eigval_med_rel", 2e-3),
        ("ridge_med_abs_delta", 2e-3),
        ("ridge_spearman", 0.99),
        ("mlp_med_abs_delta", 2e-3),
        ("eigval_med_rel", float("nan")),  # non-finite gate field FAILS
    ):
        g = dict(_PASSING_GATES)
        g[key] = bad
        assert not D._pdshrink_gates_pass(g), key


def test_gate_halt_writes_gates_json_first_then_exits_23(tmp_path):
    failing = dict(_PASSING_GATES, ridge_med_abs_delta=0.5)
    with pytest.raises(SystemExit) as exc:
        D._pdshrink_gate_halt(failing, tmp_path, smoke=False)
    assert exc.value.code == D.RC_G1_PDSHRINK == 23
    doc = json.loads((tmp_path / "gates.json").read_text())  # written BEFORE the raise
    assert doc["verdict"] == "HALT"
    assert doc["smoke_demoted"] is False


def test_gate_halt_smoke_demotes_instead_of_exiting(tmp_path):
    failing = dict(_PASSING_GATES, split_sha_match=False)
    doc = D._pdshrink_gate_halt(failing, tmp_path, smoke=True)
    assert doc["verdict"] == "SMOKE_DEMOTED_FAIL"
    assert doc["smoke_demoted"] is True
    assert (tmp_path / "gates.json").exists()


def test_gate_halt_pass_writes_and_returns(tmp_path):
    doc = D._pdshrink_gate_halt(dict(_PASSING_GATES), tmp_path, smoke=False)
    assert doc["verdict"] == "PASS"
    assert json.loads((tmp_path / "gates.json").read_text())["verdict"] == "PASS"


# ── batched paired bootstrap ─────────────────────────────────────────────────────


def test_bootstrap_identical_arms_give_zero_delta_and_is_deterministic():
    import torch

    rng = np.random.default_rng(3)
    n = 64
    yh = rng.standard_normal((n, 256))
    E = (0.3 * rng.standard_normal((n, 256))) ** 2
    T = (yh - yh.mean(0)) ** 2
    dev = torch.device("cpu")
    b1 = D._pdshrink_bootstrap({"mlp": E, "tuned": E, "oracle": E}, T, 50, 148202, dev, chunk=16)
    assert b1["delta_tuned"].shape == (50,)
    assert np.allclose(b1["delta_tuned"], 0.0)  # identical arms => gap 0 every draw
    assert np.allclose(b1["delta_oracle"], 0.0)
    E2 = (0.5 * rng.standard_normal((n, 256))) ** 2
    b2 = D._pdshrink_bootstrap({"mlp": E, "tuned": E2, "oracle": E2}, T, 50, 148202, dev, chunk=16)
    b3 = D._pdshrink_bootstrap({"mlp": E, "tuned": E2, "oracle": E2}, T, 50, 148202, dev, chunk=16)
    assert np.array_equal(b2["delta_tuned"], b3["delta_tuned"])  # seed-deterministic
    assert set(b2["tuned_band_gap_draws"]) == set(D.PDSHRINK_BANDS)


# ── resume regime guard (data-dependent gate probe) ──────────────────────────────


def test_resume_regime_mismatch_raises(tmp_path, monkeypatch):
    """phase_pdshrink refuses an out dir holding a run under a DIFFERENT regime
    (the #722 resume-key rule); a MATCHING regime skips cleanly."""
    import argparse

    out = tmp_path / "eval"
    (out / D.PDSHRINK_SUBDIR).mkdir(parents=True)
    args = argparse.Namespace(
        out_eval=out,
        smoke=True,
        max_chunks=1,
        holdout_n=10,
        fit_n=2000,
        n_boot=200,
        bootstrap_seed=148202,
        device="cpu",
    )
    grid_len = len(D._pdshrink_lambda_grid())
    regime = {
        "smoke": True,
        "max_chunks": 1,
        "holdout_n": 10,
        "fit_n": 2000,
        "n_boot": 200,
        "bootstrap_seed": 148202,
        "device": "cpu",
        "n_lambda_realized": grid_len,
    }
    spath = out / D.PDSHRINK_SUBDIR / "pdshrink_summary.json"
    spath.write_text(json.dumps({"regime": regime}))
    D.phase_pdshrink(args)  # matching regime -> resume skip, no further attrs touched
    spath.write_text(json.dumps({"regime": dict(regime, n_boot=999)}))
    with pytest.raises(RuntimeError, match="DIFFERENT regime"):
        D.phase_pdshrink(args)
