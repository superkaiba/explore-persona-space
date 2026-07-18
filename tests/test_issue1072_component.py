"""Issue #1072 unit pins: component-cell math, gate trips, alignment mapping.

All CPU, tiny shapes. Covers (a) the fp64 additivity identity + the batched-vs
-serial-oracle parity on real (tiny) data, (b) degenerate-input probes for the
data-dependent gates (g4 compare, additivity raise, K4 pilot abort, regime
quarantine, ratio-bank parity), (c) the next-token slot alignment convention.
"""

from __future__ import annotations

import json
import pathlib
import sys

import numpy as np
import pytest

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from explore_persona_space.experiments.issue_952.run_952 import (  # noqa: E402
    SLOT_NAMES,
    _slot_positions_and_validity,
)
from explore_persona_space.experiments.issue_1072 import component_ridge as cr  # noqa: E402
from explore_persona_space.experiments.issue_1072 import run_1072 as r72  # noqa: E402
from scripts.issue1072_stats import RatioBank, _safe_ratio, serial_ratio_parity  # noqa: E402


def _tiny_pair_problem(n_tr=40, n_te=12, h=10, g=3, seed=0):
    """Random X + per-row û decomposition targets (par exactly y·û û)."""
    rng = np.random.default_rng(seed)
    x_tr = rng.standard_normal((n_tr, h))
    x_te = rng.standard_normal((n_te, h))
    pairs = {}
    for gi in range(g):
        for split, n in (("train", n_tr), ("test", n_te)):
            y_full = rng.standard_normal((n, h))
            u = rng.standard_normal((n, h))
            u /= np.linalg.norm(u, axis=1, keepdims=True)
            alpha = np.einsum("ij,ij->i", y_full, u)
            pairs[(split, gi)] = (alpha[:, None] * u, y_full)
    return x_tr, x_te, (lambda split, gi: pairs[(split, gi)])


def test_component_cell_additivity_and_parity():
    x_tr, x_te, pair_fn = _tiny_pair_problem()
    names = ["g0", "g1", "g2"]
    lams = np.array([0.1, 1.0, 10.0])
    res = cr.run_component_cell(x_tr, {"test": x_te}, pair_fn, names, lams, device="cpu")
    # fp64 additivity identity (plan success criterion 3).
    assert res.additivity_max_dev < cr.ADDITIVITY_TOL
    for gi in range(3):
        p = {k: res.pooled["test"][k][gi] for k in res.pooled["test"]}
        assert abs(p["r2_full"] - (p["C_par"] + p["C_perp"] + p["C_cross"])) < 1e-9
    # Batched vs independent serial oracle (kernel-form ridge), 3 cells.
    rec = cr.component_parity_gate(
        x_tr, {"test": x_te}, pair_fn, names, lams, res, [(0, "test"), (1, "test"), (2, "test")]
    )
    assert rec["max_rel_diff"] < cr.PARITY_TOL


def test_component_cell_cross_term_nonzero():
    """Per-sample û varies, so the fitted residual cross term is NOT identically
    0 even for exactly-decomposed targets (fact-check correction, plan v2)."""
    x_tr, x_te, pair_fn = _tiny_pair_problem(seed=3)
    res = cr.run_component_cell(x_tr, {"test": x_te}, pair_fn, ["g"], np.array([1.0]), device="cpu")
    cross = res.channels["test"][:, 0, cr.CHANNELS.index("cross_res")]
    assert np.abs(cross).max() > 0.0


def test_additivity_gate_trips(monkeypatch):
    """Degenerate-input probe: the additivity raise branch executes (designed
    loud halt), forced via a monkeypatched tolerance."""
    x_tr, x_te, pair_fn = _tiny_pair_problem(seed=1)
    monkeypatch.setattr(cr, "ADDITIVITY_TOL", -1.0)
    with pytest.raises(RuntimeError, match="additivity identity"):
        cr.run_component_cell(x_tr, {"test": x_te}, pair_fn, ["g"], np.array([1.0]), device="cpu")


def test_parity_gate_trips_on_corrupted_channels():
    x_tr, x_te, pair_fn = _tiny_pair_problem(seed=2)
    names = ["g0", "g1", "g2"]
    lams = np.array([1.0, 1.0, 1.0])
    res = cr.run_component_cell(x_tr, {"test": x_te}, pair_fn, names, lams, device="cpu")
    res.channels["test"][:, 0, 0] += 1.0  # corrupt one channel
    with pytest.raises(RuntimeError, match="parity gate FAIL"):
        cr.component_parity_gate(
            x_tr,
            {"test": x_te},
            pair_fn,
            names,
            lams,
            res,
            [(0, "test"), (1, "test"), (2, "test")],
        )


def test_sensitivity_lambda_grid_matches_frozen():
    x_tr, x_te, pair_fn = _tiny_pair_problem(seed=4, g=1)
    grid = np.array([0.1, 1.0, 10.0])
    res = cr.run_component_cell(
        x_tr,
        {"test": x_te},
        pair_fn,
        ["g"],
        np.array([1.0]),
        device="cpu",
        sensitivity_lambdas=grid,
    )
    assert res.sens_pooled is not None
    # The λ=1.0 sensitivity row equals the frozen-λ pooled values.
    row = res.sens_pooled["test"][1, 0, :]
    for i, k in enumerate(("C_par", "C_perp", "C_cross", "r2_full", "w_par")):
        assert abs(row[i] - res.pooled["test"][k][0]) < 1e-12


def test_slot_next_token_alignment():
    """Plan §4.2 convention: f16_t{t} -> full_ids[rs+t]; l16_m1 excluded;
    l16_m2 -> the final real token; c_last -> the first answer token."""
    rs, span = 7, 40
    ee = rs + span
    full_ids = list(range(100, 100 + ee))
    pos, valid = _slot_positions_and_validity(rs, ee, span)
    next_ext = np.asarray(full_ids[rs:ee], dtype=np.int64)
    nids = r72.slot_next_token_ids(pos, valid, rs, span, next_ext)
    idx = {n: i for i, n in enumerate(SLOT_NAMES[:46])}
    assert nids[idx["c_last"]] == full_ids[rs]  # first answer token
    for t in (1, 3, 16):
        assert nids[idx[f"f16_t{t}"]] == full_ids[rs + t]
    assert nids[idx["l16_m1"]] == -1  # trailing \n — no realized next token
    assert nids[idx["l16_m2"]] == full_ids[ee - 1]  # next = trailing \n token
    # d10 deciles: next = full_ids[pos+1] wherever pos+1 < ee.
    for k, _slot in enumerate(SLOT_NAMES[:46]):
        if nids[k] >= 0:
            assert nids[k] == full_ids[pos[k] + 1]


def test_g4_compare_gate():
    ref = {
        "lambda_table": {"f16_t1": 10.0},
        "by_layer": {
            "own": {"f16_t1": {"test_pooled_r2": 0.5, "lambda": 10.0, "n_valid_test": 100}}
        },
    }
    ok: list[str] = []
    r72._g4_compare("t", json.loads(json.dumps(ref)), ref, ok)
    assert ok == []
    # R² beyond 1e-6 trips; λ requires exact; count exact; missing key trips.
    bad = json.loads(json.dumps(ref))
    bad["by_layer"]["own"]["f16_t1"]["test_pooled_r2"] = 0.500002
    ms: list[str] = []
    r72._g4_compare("t", bad, ref, ms)
    assert len(ms) == 1 and "test_pooled_r2" in ms[0]
    bad = json.loads(json.dumps(ref))
    bad["lambda_table"]["f16_t1"] = 10.000001
    ms = []
    r72._g4_compare("t", bad, ref, ms)
    assert len(ms) == 1 and "lambda" in ms[0]
    bad = json.loads(json.dumps(ref))
    del bad["by_layer"]["own"]
    ms = []
    r72._g4_compare("t", bad, ref, ms)
    assert ms and "missing" in ms[0]


def test_pilot_abort_rc7(tmp_path):
    """K4 designed abort: report JSON + distinct rc (production); smoke demotes."""
    with pytest.raises(SystemExit) as exc:
        r72._pilot_check(
            "capture",
            measured_wall_s=3600.0,
            units_done=1,
            units_total=100,
            booked_h=r72.CAPTURE_BOOKED_H,
            base_dir=tmp_path,
            smoke=False,
            execution_shape="test",
        )
    assert exc.value.code == r72.PILOT_ABORT_RC
    rec = json.loads(
        (tmp_path / "eval_results" / "issue_1072" / "pilot_gate_capture.json").read_text()
    )
    assert rec["verdict"] == "ABORT"
    # Smoke: same computation, verdict demoted to a log line (no exit).
    r72._pilot_check(
        "capture",
        3600.0,
        1,
        100,
        r72.CAPTURE_BOOKED_H,
        tmp_path,
        smoke=True,
        execution_shape="test",
    )


def test_battery_ckpt_regime_quarantine(tmp_path):
    d = r72._battery_ckpt_dir(tmp_path, {"a": 1})
    (d / "fold0_L0.npz").write_bytes(b"x")
    d2 = r72._battery_ckpt_dir(tmp_path, {"a": 2})
    assert d2.exists() and not (d2 / "fold0_L0.npz").exists()
    stale = [p for p in d2.parent.iterdir() if "stale" in p.name]
    assert stale and (stale[0] / "fold0_L0.npz").exists()  # quarantined, not deleted


def test_ratio_bank_observed_draws_and_parity():
    rng = np.random.default_rng(0)
    pool = list(range(20))
    bank = RatioBank(pool)
    ids = np.asarray(pool[:15])
    for c in range(3):
        bank.add(f"c{c}", ids, rng.standard_normal(15), np.abs(rng.standard_normal(15)) + 0.5)
    obs = bank.observed()
    num, den = bank.stacks()
    assert np.allclose(obs, num.sum(0) / den.sum(0))
    w = rng.multinomial(20, np.full(20, 1 / 20), size=50).astype(np.float64)
    draws = bank.draws(w)
    rec = serial_ratio_parity(bank, w, draws)
    assert rec["max_abs_diff"] <= 1e-8
    # Degenerate-denominator gate: NaN, never a division blowup.
    assert np.isnan(_safe_ratio(np.array([1.0]), np.array([0.0])))[0]


def test_smoke_fold_floors():
    """Smoke sizing vs downstream min-N floors: 10 ids / 5 folds -> 6/2/2 per
    fold — above min_train=4 and the >=2 val/test survivor floors."""
    from explore_persona_space.experiments.issue_952.run_952 import make_kfold_splits

    folds = make_kfold_splits(list(range(10)), 5)
    for f in folds:
        assert len(f["train"]) == 6 >= 4
        assert len(f["val"]) == 2 >= 2
        assert len(f["test"]) == 2 >= 2
