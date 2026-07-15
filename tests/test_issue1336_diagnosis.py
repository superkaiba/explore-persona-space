"""#1336 D1 diagnosis-round invariants (plan v7 amendment).

Pins, per the round's report contract:
  1. The `lambdas=None` source-module kwarg on `_ridge_predict_cached` /
     `heldout_r2_sweep` is DEFAULT-PRESERVING (byte-equivalent to the
     pre-change behavior; #931 return_lam pattern).
  2. DG0: the battery's v0 committed-convention rerun FAILS LOUD (exit 3)
     when the realized best R^2 misses the target by more than +/-0.02.
  3. DG1: verdict assembly FAILS LOUD when qwen_cal's bar_std was computed
     AFTER the battery started (calibration-ordering invariant), and the
     canonical step order puts qwen_cal before battery before verdict.
  4. The R1-R5 routing reads the registered lattice (S', D CI, accounting
     set, capture-defect + calibration gates).

All fits run on tiny synthetic stores THROUGH THE REAL DRIVER functions
(no seam stubs; the only fakes are the tiny tensors themselves).
"""

from __future__ import annotations

import json
import sys
from argparse import Namespace
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
for p in (str(REPO / "scripts"), str(REPO / "src")):
    if p not in sys.path:
        sys.path.insert(0, p)

import issue825_fit_cells as fc  # noqa: E402
import issue1336_diagnose_g1 as diag  # noqa: E402


def _tiny_xy(n=24, d=6, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, d)).astype(np.float32)
    W = rng.normal(size=(d, d)).astype(np.float32) / np.sqrt(d)
    Y = (X @ W + 0.5 * rng.normal(size=(n, d))).astype(np.float32)
    conv = np.asarray([f"c{i}" for i in range(n)])
    return X, Y, conv


# ---------------------------------------------------------------------------
# 1. lambdas kwarg — default-preserving at both source functions
# ---------------------------------------------------------------------------
def test_ridge_predict_lambdas_default_byte_equivalent():
    X, Y, _ = _tiny_xy()
    cache = fc._prep_fold(X[:16], X[16:])
    pred_default = fc._ridge_predict_cached(cache, Y[:16])
    pred_explicit = fc._ridge_predict_cached(cache, Y[:16], lambdas=fc.LAMBDAS)
    assert np.array_equal(pred_default, pred_explicit)


def test_ridge_predict_lambdas_grid_membership():
    X, Y, _ = _tiny_xy(seed=1)
    cache = fc._prep_fold(X[:16], X[16:])
    _, lam_default = fc._ridge_predict_cached(cache, Y[:16], return_lam=True)
    assert lam_default in set(float(v) for v in fc.LAMBDAS)
    _, lam_wide = fc._ridge_predict_cached(
        cache, Y[:16], return_lam=True, lambdas=diag.LAMBDAS_WIDE
    )
    assert lam_wide in set(float(v) for v in diag.LAMBDAS_WIDE)


def test_heldout_sweep_lambdas_default_byte_equivalent():
    X, Y, conv = _tiny_xy(seed=2)
    kw = dict(n_folds=3, seed=0, null_draws=2, frozen_layers=(0,))
    a = fc.heldout_r2_sweep(X[:, None, :], Y[:, None, :], conv, **kw)
    b = fc.heldout_r2_sweep(X[:, None, :], Y[:, None, :], conv, lambdas=fc.LAMBDAS, **kw)
    assert np.array_equal(a["r2_obs"], b["r2_obs"])
    assert np.array_equal(a["r2_null"], b["r2_null"])


# ---------------------------------------------------------------------------
# Shared driver fixture (REAL fixture-builder + REAL driver steps, tmp dirs)
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def diag_env(tmp_path_factory):
    from issue1336_smoke_fixtures import build_diag_fixture

    root = tmp_path_factory.mktemp("diag_fixture")
    build_diag_fixture(root, cells=("rlvr_chat_lmsys5k",), n=14, layers=2, dim=8, seed=0)
    return root


def _args(root: Path, out: Path, **over) -> Namespace:
    base = dict(
        steps="",
        cells="rlvr_chat_lmsys5k",
        cell_ids=["rlvr_chat_lmsys5k"],
        stage_root=root,
        out_dir=out,
        turnstore_dir=None,
        preds_dir=root / "preds",
        gen_dir=root / "gen",
        qwen_reduced=root / "qwen_reduced" / "qwen_s1_reduced.pt",
        tokenizer_dir=None,
        spotcheck_n=5,
        null_draws=2,
        n_boot=25,
        folds=3,
        seed=0,
        dg0_targets_json=None,
        committed_eval_dir=root / "nonexistent_committed",
        no_pilot_abort=True,
        wall_budget_h=1.0,
        expect_n=14,
    )
    base.update(over)
    ns = Namespace(**base)
    ns.turnstore_dir = None  # per-cell dirs under stage_root
    return ns


def _oracle_best(root: Path) -> float:
    import issue1336_fit_cells as f36

    bundle = fc._load_bundle_any(root / "turnstore_rlvr_chat_lmsys5k", "rlvr", "chat", "lmsys5k")
    xy = f36._cell_xy_1336(bundle, 2)
    sweep = fc.heldout_r2_sweep(
        xy["X"], xy["Y"], xy["conv_ids"], n_folds=3, seed=0, null_draws=0, frozen_layers=(1,)
    )
    return float(np.nanmax(sweep["r2_obs"]))


# ---------------------------------------------------------------------------
# 2. DG0 — exit 3 on a committed-convention reproduction miss
# ---------------------------------------------------------------------------
def test_dg0_gate_passes_on_oracle_target(diag_env, tmp_path):
    best = _oracle_best(diag_env)
    args = _args(
        diag_env,
        tmp_path / "out",
        dg0_targets_json=json.dumps({"rlvr_chat_lmsys5k": best}),
    )
    diag.step_battery(args)  # must not raise
    v0 = json.loads((tmp_path / "out" / "refit_v0_rlvr_chat_lmsys5k.json").read_text())
    assert v0["dg0"]["pass"] is True
    # v0 through the driver == the production heldout_r2_sweep oracle exactly.
    assert v0["best_r2"] == pytest.approx(best, abs=0.0)


def test_dg0_gate_fails_loud_on_target_miss(diag_env, tmp_path):
    best = _oracle_best(diag_env)
    args = _args(
        diag_env,
        tmp_path / "out",
        dg0_targets_json=json.dumps({"rlvr_chat_lmsys5k": best + 10 * diag.DG0_TOL}),
    )
    with pytest.raises(SystemExit) as exc:
        diag.step_battery(args)
    assert exc.value.code == 3
    # Ordering pin (r3 concern dg0-checkpoint-written-before-gate): a DG0-FAIL
    # run leaves NO pass-1 resume checkpoint, so a same-fingerprint rerun
    # re-runs the gate instead of resuming past it.
    ck = tmp_path / "out" / "checkpoints" / "battery_rlvr_chat_lmsys5k_pass1.json"
    assert not ck.exists(), "pass-1 checkpoint written despite DG0 FAIL"
    with pytest.raises(SystemExit) as exc2:
        diag.step_battery(args)  # rerun must NOT resume past the gate
    assert exc2.value.code == 3


def test_verdict_requires_spotcheck(diag_env, tmp_path):
    """r3 Minor 2: a direct battery->verdict invocation without the D1.3
    spot-check output fails loud instead of routing with capture_defect=False."""
    out = tmp_path / "out"
    best = _oracle_best(diag_env)
    args = _args(diag_env, out, dg0_targets_json=json.dumps({"rlvr_chat_lmsys5k": best}))
    diag.step_qwen_cal(args)
    diag.step_battery(args)
    diag.step_audit(args)
    with pytest.raises(AssertionError, match="spotcheck"):
        diag.step_verdict(args)


def test_verdict_asserts_dg0_pass(diag_env, tmp_path):
    """cr-v4 Minor 1: a stale refit_v0 whose DG0 gate FAILED cannot carry the
    verdict — the battery's exit-3 chain blocks the realistic path; this belt
    closes the contrived stale-v0 + changed-targets rerun."""
    out = tmp_path / "out"
    best = _oracle_best(diag_env)
    args = _args(diag_env, out, dg0_targets_json=json.dumps({"rlvr_chat_lmsys5k": best}))
    diag.step_qwen_cal(args)
    diag.step_battery(args)
    diag.step_spotcheck(args)
    diag.step_audit(args)
    v0_path = out / "refit_v0_rlvr_chat_lmsys5k.json"
    v0 = json.loads(v0_path.read_text())
    v0["dg0"]["pass"] = False  # simulate the stale / changed-targets rerun
    v0_path.write_text(json.dumps(v0))
    with pytest.raises(AssertionError, match="DG0"):
        diag.step_verdict(args)


# ---------------------------------------------------------------------------
# 3. DG1 — canonical step order + verdict ordering assert
# ---------------------------------------------------------------------------
def test_step_order_normalization_puts_qwen_cal_first():
    assert diag.normalize_steps("battery,verdict,qwen_cal") == ["qwen_cal", "battery", "verdict"]
    assert diag.normalize_steps("verdict, stage") == ["stage", "verdict"]
    with pytest.raises(AssertionError):
        diag.normalize_steps("battery,not_a_step")


def test_dg1_ordering_verdict(diag_env, tmp_path):
    out = tmp_path / "out"
    best = _oracle_best(diag_env)
    args = _args(diag_env, out, dg0_targets_json=json.dumps({"rlvr_chat_lmsys5k": best}))
    # Canonical order: qwen_cal BEFORE battery -> verdict passes DG1.
    diag.step_qwen_cal(args)
    diag.step_battery(args)
    diag.step_spotcheck(args)
    diag.step_audit(args)
    diag.step_verdict(args)
    verdict = json.loads((out / "diagnosis_verdict.json").read_text())
    assert verdict["gates"]["dg1"]["ordering_ok"] is True
    assert verdict["routed_decision"] in {
        "R1_resume",
        "R2_d2_required",
        "D2_required",
        "R3_scope_finding_absence",
        "R4_scope_finding_weak",
        "R5_replan",
    }
    # Tamper: qwen_cal computed AFTER the battery started -> DG1 assert fires.
    qc_path = out / "refit_qwen_cal.json"
    qc = json.loads(qc_path.read_text())
    v2 = json.loads((out / "refit_v2_rlvr_chat_lmsys5k.json").read_text())
    qc["computed_ts_unix"] = float(v2["started_ts_unix"]) + 999.0
    qc_path.write_text(json.dumps(qc))
    with pytest.raises(AssertionError, match="DG1 ORDERING VIOLATION"):
        diag.step_verdict(args)


# ---------------------------------------------------------------------------
# 4. Spotcheck catches a planted slot/span defect (H-B gate input)
# ---------------------------------------------------------------------------
def test_spotcheck_flags_planted_defect(tmp_path):
    from issue1336_smoke_fixtures import build_diag_fixture

    root = tmp_path / "fixture"
    build_diag_fixture(
        root,
        cells=("rlvr_chat_lmsys5k",),
        n=8,
        layers=2,
        dim=8,
        seed=1,
        corrupt_one_span=True,
    )
    args = _args(root, tmp_path / "out", spotcheck_n=8)
    diag.step_spotcheck(args)
    spot = json.loads((tmp_path / "out" / "spotcheck.json").read_text())
    cell = spot["cells"]["rlvr_chat_lmsys5k"]
    assert cell["mismatches"] >= 1
    assert cell["defect_gate_fired"] is True


# ---------------------------------------------------------------------------
# 5. D2 convention flags (r3 concern d2-convention-flag-deferred)
# ---------------------------------------------------------------------------
def _valid_rendered():
    from explore_persona_space.experiments.issue_1336.common import Rendered

    return Rendered(
        input_ids=list(range(60)),
        slot_idx={"prefix": 2, "a1": 20},
        spans={"u1": (4, 18), "a1": (22, 50)},
        format="chat",
        conv_id="s0",
    )


def test_d2_resolve_convention_contract(tmp_path):
    import issue1336_extract_turnstore as ext

    # committed: default-preserving; no override may ride along.
    assert ext.resolve_convention("committed", None, None) is None
    with pytest.raises(AssertionError, match="only consumed"):
        ext.resolve_convention("committed", tmp_path / "o.json", None)
    # corrected: requires the D1.3-emitted override JSON AND an explicit out dir.
    ov = tmp_path / "o.json"
    ov.write_text(json.dumps({"slot_offsets": {"a1": -1}}))
    with pytest.raises(AssertionError, match="requires --offset-override"):
        ext.resolve_convention("corrected", None, tmp_path)
    with pytest.raises(AssertionError, match="explicit --out-dir"):
        ext.resolve_convention("corrected", ov, None)
    got = ext.resolve_convention("corrected", ov, tmp_path)
    assert got == {"slot_offsets": {"a1": -1}, "span_offsets": {}}
    # an override indicting NO offset is meaningless — fail loud.
    empty = tmp_path / "empty.json"
    empty.write_text("{}")
    with pytest.raises(AssertionError, match="names no slot_offsets"):
        ext.resolve_convention("corrected", empty, tmp_path)


def test_d2_offset_override_applies_and_revalidates():
    import issue1336_extract_turnstore as ext

    r = _valid_rendered()
    corrected = ext.apply_offset_override(
        r, {"slot_offsets": {"a1": -1}, "span_offsets": {"a1": (1, 0)}}
    )
    assert corrected.slot_idx == {"prefix": 2, "a1": 19}
    assert corrected.spans["a1"] == (23, 50)
    assert r.slot_idx["a1"] == 20 and r.spans["a1"] == (22, 50)  # original untouched
    # an override that degenerates a span fails the consumer-exact asserts.
    with pytest.raises(AssertionError, match="corrected render invalid"):
        ext.apply_offset_override(r, {"slot_offsets": {}, "span_offsets": {"a1": (28, -20)}})
    # unknown slot/span names fail loud.
    with pytest.raises(AssertionError, match="unknown slot"):
        ext.apply_offset_override(r, {"slot_offsets": {"nope": 1}, "span_offsets": {}})


def test_d2_row_allowlist_filters_and_fails_loud(tmp_path):
    import issue1336_extract_turnstore as ext

    kept = [{"prompt_idx": i} for i in range(6)]
    p = tmp_path / "allow.json"
    p.write_text(json.dumps(["s0", 2, "s5"]))
    picked = ext.filter_row_allowlist(kept, p)
    assert [r["prompt_idx"] for r in picked] == [0, 2, 5]
    p.write_text(json.dumps(["s0", "s99"]))
    with pytest.raises(AssertionError, match="not found"):
        ext.filter_row_allowlist(kept, p)
