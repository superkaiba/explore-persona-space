"""#1336 E1 recalibration-round invariants (plan v9 amendment).

Pins, per the round's report contract:
  1. Cross-fitting correctness — no row's recalibration is fit on itself
     (fold-k (a, b) invariant under fold-k truth perturbation; an
     independent-noise fixture where the in-sample recal overfits while the
     cross-fitted read stays honest).
  2. The batched suff-stats path reproduces the direct reference (observed,
     one permutation draw, one bootstrap resample) within float tolerance —
     the vectorize-rule item-6 equivalence gate for the draw batteries.
  3. The per-resample LAYER-MAX bootstrap convention (selection-inheriting,
     pinned in plan v9 §3).
  4. bar_r fallback recording: qwen_recal_cal.json absent => bar_r_fallback
     true + V-gate UNDEFINED + terminal-eligible-only routing (never a
     silent proceed).
  5. route_verdict: all four terminal routes + both E2-trigger arms +
     the e2_fired re-read (no second E2).
  6. Fixture e2e THROUGH THE REAL DRIVER steps: the battery producer
     (issue1336_diagnose_g1 --steps battery) writes the battery_v0 npz the
     recal consumer reads (cross-phase data contract), DG-E0 passes on the
     oracle and fails loud (exit 3) on a target miss, DG-E1 ordering is
     enforced, and the E2 leg runs both variants on the fixture.

All fits run on tiny synthetic stores THROUGH THE REAL DRIVER functions
(no seam stubs; the only fakes are the tiny tensors themselves).
"""

from __future__ import annotations

import json
import shutil
import sys
from argparse import Namespace
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
for p in (str(REPO / "scripts"), str(REPO / "src")):
    if p not in sys.path:
        sys.path.insert(0, p)

import issue1336_diagnose_g1 as diag  # noqa: E402
import issue1336_recal_verdict as rv  # noqa: E402

CHAT = "rlvr_chat_lmsys5k"
NAT = "rlvr_naturalistic_lmsys5k"


# ---------------------------------------------------------------------------
# Math-core helpers (synthetic, no fixture)
# ---------------------------------------------------------------------------
def _toy(n=30, d=6, seed=0, signal=True):
    rng = np.random.default_rng(seed)
    Y = rng.normal(size=(n, d))
    # signal=True: deliberately mis-scaled + offset preds (recal has work to do).
    P = 0.4 * Y + 0.3 * rng.normal(size=(n, d)) + 0.2 if signal else rng.normal(size=(n, d))
    folds = np.arange(n) % 3
    return P, Y, folds


# ---------------------------------------------------------------------------
# 1. Cross-fitting correctness
# ---------------------------------------------------------------------------
def test_crossfit_fold_params_never_fit_on_own_fold():
    P, Y, folds = _toy(seed=1)
    ref = rv._crossfit_recal_direct(P, Y, folds)
    Y2 = Y.copy()
    Y2[folds == 0] += 5.0  # perturb ONLY fold 0's truth rows
    per = rv._crossfit_recal_direct(P, Y2, folds)
    # Fold 0's (a, b) are fit on folds 1+2 only — invariant under the change.
    assert np.array_equal(ref["a"][0], per["a"][0])
    assert np.array_equal(ref["b"][0], per["b"][0])
    # Folds 1 and 2 train on fold 0 — their params MUST move (leakage witness).
    assert not np.allclose(ref["b"][1], per["b"][1])
    assert not np.allclose(ref["b"][2], per["b"][2])


def test_crossfit_honest_on_independent_noise():
    # Few rows x many dims: the 2-params-per-dim in-sample fit manufactures
    # positive R^2 on pure noise; the cross-fitted read must not inherit it.
    rng = np.random.default_rng(2)
    n, d = 12, 300
    P = rng.normal(size=(n, d))
    Y = rng.normal(size=(n, d))
    folds = np.arange(n) % 3
    insample = rv._insample_recal_r2(P, Y)
    heldout = rv._crossfit_recal_direct(P, Y, folds)["r2"]
    assert insample > 0.08, f"noise fixture not overfitting in-sample ({insample})"
    assert heldout < 0.05, f"cross-fitted read inherited in-sample optimism ({heldout})"
    assert insample - heldout > 0.05


# ---------------------------------------------------------------------------
# 2. Batched suff-stats path == direct reference
# ---------------------------------------------------------------------------
def test_stats_path_matches_direct_observed():
    P, Y, folds = _toy(seed=3)
    st = rv._suff_stats_observed(P, Y, folds)
    r2_stats = float(
        rv._recal_r2_from_stats(st["s_p"], st["s_y"], st["s_pp"], st["s_yy"], st["s_py"], st["n"])
    )
    r2_direct = rv._crossfit_recal_direct(P, Y, folds)["r2"]
    assert r2_stats == pytest.approx(r2_direct, rel=1e-9)


def test_null_battery_matches_direct_recompute():
    P0, Y0, folds = _toy(seed=4)
    P1, Y1, _ = _toy(seed=5)
    n_draws, seed = 3, 17
    mat, layers = rv._null_battery_matrix({0: P0, 1: P1}, {0: Y0, 1: Y1}, folds, n_draws, seed)
    assert layers == [0, 1] and mat.shape == (n_draws, 2)
    perms = rv._within_fold_perms(folds, n_draws, seed)  # same deterministic stream
    for t in range(n_draws):
        assert (folds[perms[t]] == folds).all()  # within-fold by construction
        for lix, (P, Y) in enumerate(((P0, Y0), (P1, Y1))):
            direct = rv._crossfit_recal_direct(P, Y[perms[t]], folds)["r2"]
            assert mat[t, lix] == pytest.approx(direct, rel=1e-9)


def test_bootstrap_matches_direct_recompute():
    P0, Y0, folds = _toy(seed=6)
    P1, Y1, _ = _toy(seed=7)
    weights = rv._bootstrap_weights(len(folds), 4, 23)
    mat, layers = rv._bootstrap_matrix({0: P0, 1: P1}, {0: Y0, 1: Y1}, folds, weights, chunk=2)
    assert layers == [0, 1] and mat.shape == (4, 2)
    for t in range(4):
        idx = np.repeat(np.arange(len(folds)), weights[t].astype(int))
        for lix, (P, Y) in enumerate(((P0, Y0), (P1, Y1))):
            direct = rv._crossfit_recal_direct(P[idx], Y[idx], folds[idx])["r2"]
            assert mat[t, lix] == pytest.approx(direct, rel=1e-9)


def test_repartition_norms_match_direct_blocks():
    rng = np.random.default_rng(8)
    M = rng.normal(size=(17, 5))
    out = rv._repartition_norms(M, 3, 2, seed=9)
    assert out.shape == (2, 3)
    rng2 = np.random.default_rng(9)
    sizes = [len(b) for b in np.array_split(np.arange(17), 3)]
    mu = M.astype(np.float64).mean(0)
    for t in range(2):
        perm = rng2.permutation(17)
        start = 0
        for ki, s in enumerate(sizes):
            block = M[perm[start : start + s]].astype(np.float64)
            assert out[t, ki] == pytest.approx(float(np.linalg.norm(block.mean(0) - mu)), rel=1e-9)
            start += s


# ---------------------------------------------------------------------------
# 5. route_verdict — all four routes + trigger arms
# ---------------------------------------------------------------------------
def _route(**over):
    base = dict(
        s_prime_r=0.1,
        d_ci=(0.05, 0.4),
        v_gate="pass",
        a_r=0.95,
        trigger1=False,
        trigger2=False,
        e2_fired=False,
    )
    base.update(over)
    return rv.route_verdict(**base)


def test_route_verdict_all_routes_and_trigger_arms():
    assert _route() == ("resume_on_recalibrated_dv", "lattice_branch_1_accounted")
    assert _route(s_prime_r=-0.05) == ("weak_transfer_scope", "lattice_branch_2")
    assert _route(d_ci=(-0.1, 0.2), a_r=0.95) == (
        "absence_with_account",
        "lattice_branch_3_accounted",
    )
    assert _route(d_ci=(-0.1, 0.2), a_r=0.3) == ("terminal_diagnosis_only", "no_account_no_trigger")
    assert _route(a_r=0.3) == ("terminal_diagnosis_only", "usable_strength_unaccounted_no_trigger")
    assert _route(v_gate="fail", trigger1=True) == ("terminal_diagnosis_only", "v_gate_failed")
    assert _route(v_gate="undefined") == (
        "terminal_diagnosis_only",
        "bar_r_fallback_v_gate_undefined",
    )
    assert _route(trigger1=True) == ("e2_refit_required", "trigger1_fold_indictment")
    assert _route(trigger2=True) == ("e2_refit_required", "trigger2_boundary_straddle")
    assert _route(trigger1=True, trigger2=True) == ("e2_refit_required", "trigger1_fold_indictment")
    # e2_fired: the four routes apply verbatim to v5 quantities — no second E2.
    assert _route(trigger1=True, e2_fired=True) == (
        "resume_on_recalibrated_dv",
        "lattice_branch_1_accounted",
    )


# ---------------------------------------------------------------------------
# Shared fixture env (REAL battery producer -> REAL recal consumer)
# ---------------------------------------------------------------------------
def _d1_args(root: Path, out: Path, **over) -> Namespace:
    base = dict(
        steps="",
        cells=f"{CHAT},{NAT}",
        cell_ids=[CHAT, NAT],
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
    return Namespace(**base)


def _recal_args(env: dict, out_dir: Path | None = None, **over) -> Namespace:
    root = env["root"]
    base = dict(
        steps="",
        cells=f"{CHAT},{NAT}",
        cell_ids=[CHAT, NAT],
        stage_root=root,
        out_dir=out_dir if out_dir is not None else env["recal_out"],
        turnstore_dir=None,
        preds_dir=root / "preds",
        battery_preds_dir=env["diag_out"] / "tensors",
        gen_dir=root / "gen",
        qwen_reduced=root / "qwen_reduced" / "qwen_s1_reduced.pt",
        committed_eval_dir=env["out"],
        folds=3,
        seed=0,
        fold_rand_seed=1,
        recal_null_draws=8,
        n_boot=25,
        n_repart=25,
        expect_n=14,
        dge0_targets=dict(env["dge0_targets"]),
        r2_v0_l29=env["dge0_targets"]["l29"],
        use_e2=False,
        e2_variant="auto",
        inner_folds=3,
        no_pilot_abort=True,
        wall_budget_h=1.0,
    )
    base.update(over)
    ns = Namespace(**base)
    ns.turnstore_dir = None  # per-cell dirs under stage_root
    return ns


@pytest.fixture(scope="module")
def recal_env(tmp_path_factory):
    from issue1336_smoke_fixtures import build_diag_fixture

    root = tmp_path_factory.mktemp("recal_fixture")
    build_diag_fixture(root, n=14, layers=2, dim=8, seed=0)
    out = tmp_path_factory.mktemp("recal_pipeline")
    diag_out = out / "diagnosis"
    # DG0 oracle for the d1 battery producer (same shape as the dispatch smoke).
    import issue825_fit_cells as fc
    import issue1336_fit_cells as f36

    targets = {}
    for cell in (CHAT, NAT):
        bundle = fc._load_bundle_any(root / f"turnstore_{cell}", *cell.split("_", 2))
        xy = f36._cell_xy_1336(bundle, 2)
        sweep = fc.heldout_r2_sweep(
            xy["X"], xy["Y"], xy["conv_ids"], n_folds=3, seed=0, null_draws=0, frozen_layers=(1,)
        )
        targets[cell] = float(np.nanmax(sweep["r2_obs"]))
    d1a = _d1_args(root, diag_out, dg0_targets_json=json.dumps(targets))
    diag.step_battery(d1a)  # REAL producer: battery_v0_preds_*.npz + refit_v0 JSONs
    v0 = json.loads((diag_out / f"refit_v0_{CHAT}.json").read_text())
    l1 = float(v0["r2_per_layer_obs"][1])
    env = {
        "root": root,
        "out": out,
        "diag_out": diag_out,
        "recal_out": out / "diagnosis" / "recal",
        "dge0_targets": {"l29": l1, "l30": l1},
    }
    # Full E1 pipeline once, module-scoped (steps in canonical order).
    args = _recal_args(env)
    rv.step_stage(args)
    rv.step_qwen_recal(args)
    rv.step_recal(args)
    rv.step_fold_exch(args)
    rv.step_verdict(args)
    return env


# ---------------------------------------------------------------------------
# 6. Fixture e2e through the real driver
# ---------------------------------------------------------------------------
def test_pipeline_outputs_exist_and_dge0_passes(recal_env):
    out = recal_env["recal_out"]
    for f in (
        f"heldout_recal_{CHAT}.json",
        f"heldout_recal_{NAT}.json",
        f"fold_exch_{CHAT}.json",
        f"fold_exch_{NAT}.json",
        "qwen_recal_cal.json",
        "recal_verdict.json",
    ):
        assert (out / f).exists(), f
    hr = json.loads((out / f"heldout_recal_{CHAT}.json").read_text())
    for label in ("l29", "l30"):
        assert hr["dg_e0"][label]["pass"] is True
    # Consumer recompute == the producer sweep's own r2 at the verdict layer.
    assert hr["per_layer"]["1"]["raw_r2"] == pytest.approx(
        recal_env["dge0_targets"]["l29"], abs=1e-6
    )


def test_bootstrap_layer_max_convention_in_pipeline(recal_env):
    out = recal_env["recal_out"]
    hr = json.loads((out / f"heldout_recal_{CHAT}.json").read_text())
    npz = np.load(out / "tensors" / f"recal_draws_{CHAT}.npz")
    boot = npz["boot_r2_matrix"]
    # Per-resample LAYER-MAX (selection-inheriting; plan v9 §3 pinned).
    assert np.allclose(np.asarray(hr["bootstrap"]["s_r_per_draw"]), np.nanmax(boot, axis=1))
    assert np.allclose(npz["null_layer_max"], np.nanmax(npz["null_r2_matrix"], axis=1))
    assert hr["s_r"] == pytest.approx(max(v["heldout_recal_r2"] for v in hr["per_layer"].values()))


def test_verdict_routing_self_consistent(recal_env):
    out = recal_env["recal_out"]
    v = json.loads((out / "recal_verdict.json").read_text())
    li = v["lattice_inputs"]
    expected = rv.route_verdict(
        s_prime_r=li["s_prime_r"],
        d_ci=tuple(li["d_r_ci95"]),
        v_gate=v["v_gate"]["outcome"],
        a_r=v["mechanism_account"]["a_r"],
        trigger1=v["e2_trigger"]["trigger1_fired"],
        trigger2=v["e2_trigger"]["trigger2_fired"],
        e2_fired=False,
    )
    assert (v["routed_decision"], v["route_reason"]) == expected
    assert v["dg_e1"]["ordering_ok"] is True
    assert v["dg_e1"]["bar_r_fallback"] is False


def test_dup_audit_digest_only_counts(recal_env):
    out = recal_env["recal_out"]
    fe = json.loads((out / f"fold_exch_{CHAT}.json").read_text())
    dup = fe["dup_audit"]
    assert dup["n_rows"] == 14 and dup["join_rate"] == 1.0
    # Fixture prompts carry a per-row "(variant i)" suffix — no duplicates.
    for tier in ("exact", "normalized"):
        assert dup["tiers"][tier]["n_unique"] == 14
        assert dup["tiers"][tier]["total_dup_pairs"] == 0
    # No prompt text anywhere in the persisted audit (digest-only discipline).
    assert "prompt" not in json.dumps(dup).lower()


def test_dge0_fails_loud_on_target_miss(recal_env, tmp_path):
    bad = dict(recal_env["dge0_targets"])
    bad["l29"] = bad["l29"] + 0.5
    args = _recal_args(recal_env, out_dir=tmp_path / "out", dge0_targets=bad)
    with pytest.raises(SystemExit) as ei:
        rv.step_recal(args)
    assert ei.value.code == 3


def test_bar_r_fallback_records_undefined_v_gate(recal_env, tmp_path):
    out2 = tmp_path / "out"
    out2.mkdir(parents=True)
    src = recal_env["recal_out"]
    for f in (f"heldout_recal_{CHAT}.json", f"fold_exch_{CHAT}.json"):
        shutil.copy(src / f, out2 / f)  # NO qwen_recal_cal.json -> fallback
    args = _recal_args(recal_env, out_dir=out2)
    rv.step_verdict(args)
    v = json.loads((out2 / "recal_verdict.json").read_text())
    assert v["dg_e1"]["bar_r_fallback"] is True
    assert v["lattice_inputs"]["bar_r"] == pytest.approx(0.20)
    assert v["v_gate"]["outcome"] == "undefined"
    assert v["routed_decision"] == "terminal_diagnosis_only"
    assert v["route_reason"] == "bar_r_fallback_v_gate_undefined"


def test_dg_e1_ordering_violation_fails_loud(recal_env, tmp_path):
    out3 = tmp_path / "out"
    out3.mkdir(parents=True)
    src = recal_env["recal_out"]
    for f in (f"heldout_recal_{CHAT}.json", f"fold_exch_{CHAT}.json", "qwen_recal_cal.json"):
        shutil.copy(src / f, out3 / f)
    qc = json.loads((out3 / "qwen_recal_cal.json").read_text())
    qc["computed_ts_unix"] = qc["computed_ts_unix"] + 1e6  # AFTER the Llama read
    (out3 / "qwen_recal_cal.json").write_text(json.dumps(qc))
    args = _recal_args(recal_env, out_dir=out3)
    with pytest.raises(AssertionError, match="DG-E1 ORDERING VIOLATION"):
        rv.step_verdict(args)


def test_e2_refuses_when_not_triggered_and_auto(recal_env, tmp_path):
    out4 = tmp_path / "out"
    out4.mkdir(parents=True)
    v = json.loads((recal_env["recal_out"] / "recal_verdict.json").read_text())
    v["e2_trigger"] = {"trigger1_fired": False, "trigger2_fired": False, "fired": False}
    (out4 / "recal_verdict.json").write_text(json.dumps(v))
    args = _recal_args(recal_env, out_dir=out4)
    with pytest.raises(AssertionError, match="E2 not triggered"):
        rv._resolve_e2_variant(args)


def test_e2_both_variants_and_use_e2_verdict(recal_env):
    args = _recal_args(recal_env, e2_variant="fold")
    rv.step_e2(args)
    out = recal_env["recal_out"]
    v5 = json.loads((out / f"refit_v5_{CHAT}.json").read_text())
    assert v5["variant"] == "v5_fold" and v5["fold_seeds"] == [0, 1, 2]
    assert v5["full_chat_curve"] is not None
    npz = np.load(out / "tensors" / f"refit_v5_draws_{CHAT}.npz")
    assert np.allclose(np.asarray(v5["bootstrap"]["s_r_per_draw"]), npz["boot_s_r"])
    # E2-fired lattice re-read on the v5 outputs (same registered quantities).
    args_v = _recal_args(recal_env, use_e2=True)
    rv.step_verdict(args_v)
    v = json.loads((out / "recal_verdict.json").read_text())
    assert v["e2_fired"] is True and v["read_source"] == "v5"
    assert v["lattice_inputs"]["s_r"] == pytest.approx(v5["s_r"])
    assert v["routed_decision"] != "e2_refit_required"  # no second E2
    # v5-cal variant (nested-CV lambda selection) also runs on the fixture.
    args_c = _recal_args(recal_env, e2_variant="cal", cells=CHAT, cell_ids=[CHAT])
    rv.step_e2(args_c)
    v5c = json.loads((out / f"refit_v5_{CHAT}.json").read_text())
    assert v5c["variant"] == "v5_cal"
    assert v5c["lambda_by_layer_fold"]["1"], "per-(layer, fold) selected lambdas missing"
    # Restore the fold-variant verdict state for any later reader.
    rv.step_verdict(_recal_args(recal_env))
