"""Gate tests for scripts/issue823_ladder_fits.py (#823 P-Fit driver).

Pins the six binding-reconciliation requirements with REAL production bodies
(no mocks anywhere — every test executes the registered function): R1 the
5000-context split-call refusal, R2 drop accounting + the d-boundary
UNREALIZABLE labeling, R3 the PRE-DROP banked-split assert, R4 the verdict
lattice with BOTH interval endpoints (incl. the registered boundary fixture
0.02 / [-0.02, 0.06] -> Intermediate), R5 the single registered primary
estimator, R6 conditioning of EVERY lattice outcome on the joint distinctness
predicate. Plus the registered re-reads (Sigma ss_res ladder + fixed-reference
R2), the G1 reduction-convention pin against the REAL banked artifact
(`eval_results/issue_823/ridge_r2_by_arm.json` — registered in
tests/sparse_cones.txt), the paired bootstrap (shared per-draw context
resample), arm/mask assembly on a tiny synthetic store, checkpoint fingerprint
gating, and the designed-abort rc contract.

Offline: no HF fetch; the only repo fixture reads are the two banked
issue-823 JSONs named above.
"""

from __future__ import annotations

import inspect
import json
import pathlib

import numpy as np
import pytest
import torch
from sklearn.model_selection import KFold

from scripts import issue779_fitter_fair_comparison as FFC
from scripts import issue823_ladder_fits as LF

# ── R4: verdict lattice ──────────────────────────────────────────────────────


def test_lattice_boundary_fixture_is_intermediate():
    """REGISTERED boundary fixture: point 0.02 with CI [-0.02, 0.06] -> Intermediate.

    The point lies inside the Flat band but the interval reaches past +0.03,
    so the interval conjuncts (ci_high <= 0.03) must demote Flat -> Intermediate.
    """
    assert LF.lattice_verdict(0.02, -0.02, 0.06) == "Intermediate"


def test_lattice_representative_labels():
    assert LF.lattice_verdict(0.08, 0.02, 0.14) == "Degrades"
    assert LF.lattice_verdict(0.0, -0.02, 0.02) == "Flat"
    # point above 0.05 but CI touching zero -> not Degrades, not Flat
    assert LF.lattice_verdict(0.08, -0.01, 0.14) == "Intermediate"
    # reversal (k=16 better) -> Intermediate, never Flat
    assert LF.lattice_verdict(-0.06, -0.10, -0.02) == "Intermediate"
    with pytest.raises(ValueError):
        LF.lattice_verdict(float("nan"), -0.02, 0.02)


# ── R6: conditioning on the joint distinctness predicate ─────────────────────


def test_every_lattice_outcome_conditioned_on_distinctness():
    for verdict in ("Degrades", "Flat", "Intermediate"):
        for m1, m2 in ((False, True), (True, False), (False, False)):
            out = LF.conditioned_interpretation(verdict, m1, m2)
            assert out["interpretation"] == "manipulation failure"
            assert out["lattice_label_numeric"] == verdict  # numeric retained
            assert out["distinct"] is False
        ok = LF.conditioned_interpretation(verdict, True, True)
        assert ok["interpretation"] == verdict
        assert ok["distinct"] is True


# ── R5: one registered primary estimator ─────────────────────────────────────


def test_primary_estimator_constant():
    assert LF.PRIMARY_ESTIMATOR == "gcv-pure-parent-parity"


# ── R1: split-call refusal + nominal counts ──────────────────────────────────


def test_checked_fixed_split_refuses_masked_n_ctx():
    with pytest.raises(ValueError, match="4998"):
        LF.checked_fixed_split(4998)


def test_checked_fixed_split_nominal_counts():
    tr, va, te = LF.checked_fixed_split(5000)
    assert (len(tr), len(va), len(te)) == (3600, 400, 1000)
    union = np.concatenate([tr, va, te])
    assert len(np.unique(union)) == 5000


# ── R3: pre-drop banked-split assert ─────────────────────────────────────────


def _parent_set_matching_banked() -> tuple[tuple, set]:
    """5000-id space minus 1 train id + 1 val id == the banked 3599/399/1000 record."""
    pre = LF.checked_fixed_split(5000)
    tr, va, _te = pre
    parent = set(range(5000)) - {int(tr[0]), int(va[0])}
    return pre, parent


def test_predrop_banked_split_check_passes_on_banked_convention():
    pre, parent = _parent_set_matching_banked()
    rec = LF.predrop_banked_split_check(pre, np.array(sorted(parent)))
    assert rec["check"].startswith("PASS")
    assert set(rec["predrop_sha256"]) == {"train", "val", "test"}


def test_predrop_banked_split_check_fails_on_wrong_subset_counts():
    pre = LF.checked_fixed_split(5000)
    _tr, _va, te = pre
    # two drops landing in TEST contradict the banked 3599/399/1000 record
    parent_bad = set(range(5000)) - {int(te[0]), int(te[1])}
    with pytest.raises(RuntimeError, match="arms_masked"):
        LF.predrop_banked_split_check(pre, np.array(sorted(parent_bad)))


def test_banked_split_json_matches_embedded_constants():
    d = json.loads(LF.BANKED_SPLIT_JSON.read_text())
    assert d["split_realized"]["parent"] == LF.BANKED_SPLIT_PARENT
    got = d["split_realized"]["arms_masked"]
    assert {k: got[k] for k in LF.BANKED_SPLIT_ARMS_MASKED} == LF.BANKED_SPLIT_ARMS_MASKED


# ── R2: drop accounting + d-boundary ─────────────────────────────────────────


def test_realized_split_with_drops_accounting():
    pre, parent = _parent_set_matching_banked()
    tr, va, te = pre
    # ~10 NEW drops: 4 train + 3 val + 3 test (from ids still in the parent set)
    new_dropped = (
        [int(i) for i in tr if i in parent][1:5]
        + [int(i) for i in va if i in parent][1:4]
        + [int(i) for i in te if i in parent][1:4]
    )
    new = parent - set(new_dropped)
    subsets, drops = LF.realized_split_with_drops(
        pre, np.array(sorted(parent)), np.array(sorted(new))
    )
    tr_rec = drops["train"]
    assert (tr_rec["pre_drop_n"], tr_rec["realized_n"]) == (3600, 3595)
    assert (tr_rec["parent_drops"], tr_rec["new_drops"]) == (1, 4)
    assert set(tr_rec["new_dropped_ids"]) == set(new_dropped[:4])
    assert drops["val"]["realized_n"] == 396 and drops["val"]["new_drops"] == 3
    assert drops["test"]["realized_n"] == 997 and drops["test"]["new_drops"] == 3
    assert drops["total_drops"] == 12  # the parent's 2 + ~10 new, NOT <= 2
    assert len(subsets["train"]) == 3595
    assert set(subsets["test"].tolist()) <= set(te.tolist())


def test_d_boundary_disposition_and_degeneracy():
    ok = LF.d_boundary_disposition(3595)
    assert ok["pass"] is True and ok["d_rung_status"] == "realizable"
    bad = LF.d_boundary_disposition(3583)
    assert bad["pass"] is False and bad["d_rung_status"] == "UNREALIZABLE"
    assert LF.estimator_degenerate(3584) is True  # n_train == d is still degenerate
    assert LF.estimator_degenerate(3585) is False


def test_p2_rung_table_labels_unrealizable_rungs():
    rungs = LF.p2_rung_table(3595)
    assert rungs[0] == {"n_train": 3595, "status": "top-rung (realized_train)"}
    assert all(r["status"] == "realizable" for r in rungs[1:])
    small = {r["n_train"]: r["status"] for r in LF.p2_rung_table(3000)}
    assert small[3584] == "UNREALIZABLE"
    assert small[2400] == "realizable"


# ── G1 reduction-convention pin ──────────────────────────────────────────────


def test_fold_mean_vs_pooled_differ_and_precheck_raises_on_gap():
    comps = [(1.0, 10.0), (5.0, 5.5)]  # wildly different fold ss_tot
    fm = LF.fold_mean_r2(comps)
    pooled = LF.pooled_r2_from_components(comps)
    assert abs(fm - pooled) > 0.1  # the two reductions genuinely differ
    with pytest.raises(RuntimeError, match="convention"):
        LF.reduction_convention_precheck(comps)
    # homogeneous folds: gap ~0 -> pre-check passes
    good = [(1.0, 10.0)] * 5
    rec = LF.reduction_convention_precheck(good)
    assert rec["pass"] is True and rec["gap"] < LF.G1_TOL / 2


def test_banked_g1_constants_match_committed_artifact():
    """Embedded fold-mean constants re-derived from the banked JSON (drift guard)."""
    got = LF.load_banked_fold_means()
    assert set(got) == set(LF.G1_BANKED_FOLD_MEAN)
    for key, want in LF.G1_BANKED_FOLD_MEAN.items():
        assert got[key] == pytest.approx(want, abs=1e-12), key


# ── Bootstrap: paired resample + both CI endpoints ───────────────────────────


def test_bootstrap_paired_identical_cells_give_zero_delta_ci():
    rng = np.random.default_rng(0)
    n = 40
    base_res = rng.uniform(0.5, 1.5, size=n)
    base_tot = rng.uniform(2.0, 3.0, size=n)
    layers = (0,)
    ss_res = {("k1", 0): base_res, ("k16", 0): base_res.copy(), ("own", 0): base_res * 0.5}
    ss_tot = {("k1", 0): base_tot, ("k16", 0): base_tot.copy(), ("own", 0): base_tot.copy()}
    out = LF.bootstrap_paired(ss_res, ss_tot, n_draws=200, seed=0, delta_layers=layers)
    # identical k1/k16 arrays under a SHARED per-draw resample -> delta exactly 0
    assert out["ci_low_delta_mean"] == 0.0 and out["ci_high_delta_mean"] == 0.0
    assert set(out["per_cell_ci"]) == {"k1:L0", "k16:L0", "own:L0"}
    for ci in out["per_cell_ci"].values():
        assert ci["ci_low"] <= ci["ci_high"]


def test_bootstrap_paired_detects_real_gap():
    rng = np.random.default_rng(1)
    n = 60
    tot = rng.uniform(2.0, 3.0, size=n)
    res_good = rng.uniform(0.2, 0.4, size=n)  # k1: high R2
    res_bad = rng.uniform(1.2, 1.6, size=n)  # k16: low R2
    ss_res = {("k1", 0): res_good, ("k16", 0): res_bad}
    ss_tot = {("k1", 0): tot, ("k16", 0): tot.copy()}
    out = LF.bootstrap_paired(ss_res, ss_tot, n_draws=300, seed=0, delta_layers=(0,))
    assert out["ci_low_delta_mean"] > 0  # k1 - k16 gap resolved away from zero
    assert out["ci_high_delta_mean"] > out["ci_low_delta_mean"]


# ── Mixture-floor re-reads ───────────────────────────────────────────────────


def test_fixed_reference_r2_arithmetic():
    ss_res_k = np.array([1.0, 2.0])  # sum 3
    ss_tot_k1 = np.array([5.0, 5.0])  # sum 10
    assert LF.fixed_reference_r2(ss_res_k, ss_tot_k1) == pytest.approx(0.7, abs=1e-9)


# ── Solver seams ─────────────────────────────────────────────────────────────


def test_svd_gcv_lambda_matches_gram_gcv_selection():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(24, 6))
    y = x @ rng.normal(size=(6, 4)) + 0.1 * rng.normal(size=(24, 4))
    fact = FFC._factorize(x, torch.device("cpu"))
    lam_gram, _vty, _ymu = FFC._gcv_solve(fact, y)
    lam_svd = LF.svd_gcv_lambda(x, y, FFC.LAMBDAS)
    assert lam_gram == lam_svd  # same GCV criterion, two decompositions


def test_val_select_lambda_returns_grid_member_and_finite_r2():
    rng = np.random.default_rng(2)
    x = rng.normal(size=(30, 5))
    y = x @ rng.normal(size=(5, 3)) + 0.05 * rng.normal(size=(30, 3))
    xv = rng.normal(size=(10, 5))
    yv = xv @ np.linalg.lstsq(x, y, rcond=None)[0]
    fact = FFC._factorize(x, torch.device("cpu"))
    vty, ymu = FFC._vty_ymu(fact, y)
    kval = FFC._cross_kernel(fact, xv)
    grid = np.logspace(-2, 8, 21)
    lam, val_r2 = LF.val_select_lambda(fact, vty, ymu, kval, yv, grid)
    assert lam in grid and np.isfinite(val_r2)


def test_gcv_dof_capped_excludes_low_lambdas():
    rng = np.random.default_rng(3)
    x = rng.normal(size=(6, 4))
    y = rng.normal(size=(6, 2))
    fact = FFC._factorize(x, torch.device("cpu"))
    lam_pure, _, _ = FFC._gcv_solve(fact, y)
    lam_cap, _vty, _ymu, dof = LF.gcv_solve_dof_capped(fact, y, cap_frac=0.5)
    assert dof <= 0.5 * 6 + 1e-9
    assert lam_cap >= lam_pure  # sensitivity re-selection can only regularize harder here
    with pytest.raises(RuntimeError, match="cap"):
        LF.gcv_solve_dof_capped(fact, y, cap_frac=1e-6)


# ── Arm/mask assembly on a tiny synthetic store ──────────────────────────────


def _tiny_inputs(n_ctx=8, n_layers=28, hidden=4, invalid_pair=None, shift=None, invalid_pairs=()):
    """Synthetic LadderInputs; invalid_pair(s)=(ctx, persona) zero those spans."""
    rng = np.random.default_rng(0)
    cx = torch.tensor(rng.normal(size=(n_ctx, n_layers, hidden)), dtype=torch.float32)
    parent_arm = {
        a: rng.normal(size=(n_ctx, n_layers, hidden)).astype(np.float32) for a in ("own", "plain")
    }
    store_v, store_ctx, store_span = {}, {}, {}
    for p in range(LF.N_PERSONAS):
        ctxs = sorted({i for i in range(n_ctx) if any(i % k == p for k in LF.LADDER_KS)})
        v = rng.normal(size=(len(ctxs), n_layers, hidden)).astype(np.float32)
        store_ctx[p] = np.array(ctxs, dtype=np.int64)
        store_span[p] = np.full(len(ctxs), 7, dtype=np.int64)
        store_v[p] = v
    if shift is not None:
        # persona p>=1 rows = persona-0 rows (same ctx) + constant shift (M2 fixture)
        row0 = {int(c): j for j, c in enumerate(store_ctx[0])}
        for p in range(1, LF.N_PERSONAS):
            for j, c in enumerate(store_ctx[p]):
                store_v[p][j] = store_v[0][row0[int(c)]] + shift
    pairs = list(invalid_pairs)
    if invalid_pair is not None:
        pairs.append(invalid_pair)
    for ctx, p in pairs:
        j = {int(c): j for j, c in enumerate(store_ctx[p])}[ctx]
        store_span[p][j] = 0
    return LF.LadderInputs(
        cx_last=cx,
        layers_map=list(range(n_layers)),
        parent_arm=parent_arm,
        parent_valid_ids=np.arange(n_ctx),
        store_v=store_v,
        store_ctx=store_ctx,
        store_span=store_span,
        n_contexts=n_ctx,
        n_layers=n_layers,
        hidden=hidden,
    )


def _tiny_by_persona(n_ctx=8, overrides=None, drop_pairs=(), strip_validity=()):
    """Minimal P-Gen-style pair records (v11 `validity` labels), one per served pair.

    overrides: {(ctx, persona): validity_label}; drop_pairs: pairs with NO record
    (missing_record class); strip_validity: pairs whose record lacks the label key
    (the v11 schema-break shape).
    """
    ov = dict(overrides or {})
    out = {}
    for p in range(LF.N_PERSONAS):
        ctxs = sorted({i for i in range(n_ctx) if any(i % k == p for k in LF.LADDER_KS)})
        rows = []
        for c in ctxs:
            if (c, p) in drop_pairs:
                continue
            r = {"context_id": c, "validity": ov.get((c, p), "ok")}
            if (c, p) in strip_validity:
                r.pop("validity")
            rows.append(r)
        out[p] = rows
    return out


def test_build_mask_drops_invalid_pair_and_counts_per_arm():
    inputs = _tiny_inputs(invalid_pair=(3, 3))  # ctx 3, persona 3 (= 3 % 4)
    mask, _gathers, drops = LF.build_mask_and_gathers(inputs, _tiny_by_persona())
    assert 3 not in set(mask.tolist())
    assert len(mask) == inputs.n_contexts - 1
    # persona 3 serves ctx 3 under k4/k8/k16 (3%4 == 3%8 == 3%16 == 3), not k1/k2
    assert drops["new_drops_per_arm"] == {"k1": 0, "k2": 0, "k4": 1, "k8": 1, "k16": 1}
    assert drops["new_dropped_ids_union"] == [3]
    # gen-side label is "ok" => store-side drop classifies capture_zero_span (integrity)
    assert drops["new_drops_per_arm_by_class"]["k4"] == {"refusal": 0, "integrity": 1}
    assert drops["new_drop_subclasses_per_arm"]["k4"] == {"capture_zero_span": 1}
    assert drops["abort_class"] == "integrity"
    assert drops["mask_gate_schema_id"] == LF.MASK_GATE_SCHEMA_ID


def test_mask_gate_refusal_drops_never_trip_integrity_verdict():
    """v11 kill-1 semantics (plan v13 L619-623/L1175): >50 refusal-labeled drops in
    one arm leave the integrity verdict at 0 (the superseded v10 gate aborted on
    the undifferentiated total — the 2026-08-19 rc=5 false abort), while the SAME
    drop pattern labeled integrity-class ("empty") trips the unchanged threshold."""
    n = 128
    pairs = [(i, 0) for i in range(60)]  # arm k1 (persona 0) loses 60 contexts
    inputs = _tiny_inputs(n_ctx=n, invalid_pairs=pairs)
    refusal = _tiny_by_persona(n_ctx=n, overrides={(i, 0): "refusal" for i in range(60)})
    _, _, drops = LF.build_mask_and_gathers(inputs, refusal)
    assert drops["new_drops_per_arm"]["k1"] == 60 > LF.NEW_DROPS_ABORT_PER_ARM
    assert drops["new_drops_per_arm_by_class"]["k1"] == {"refusal": 60, "integrity": 0}
    assert LF.mask_integrity_verdict(drops)[1] == 0  # no abort
    empty = _tiny_by_persona(n_ctx=n, overrides={(i, 0): "empty" for i in range(60)})
    _, _, drops2 = LF.build_mask_and_gathers(inputs, empty)
    worst = LF.mask_integrity_verdict(drops2)
    assert worst == ("k1", 60) and worst[1] > LF.NEW_DROPS_ABORT_PER_ARM  # aborts
    assert drops2["new_drop_subclasses_per_arm"]["k1"] == {"empty": 60}
    # mask construction itself is label-independent (plan v13 L587 unchanged)
    assert drops["mask_n"] == drops2["mask_n"]


def test_mask_gate_missing_record_is_integrity_and_missing_label_raises():
    inputs = _tiny_inputs(invalid_pair=(3, 3))
    byp = _tiny_by_persona(drop_pairs={(3, 3)})  # no P-Gen record for the dropped pair
    _, _, drops = LF.build_mask_and_gathers(inputs, byp)
    assert drops["new_drop_subclasses_per_arm"]["k4"] == {"missing_record": 1}
    assert drops["new_drops_per_arm_by_class"]["k4"]["integrity"] == 1
    # a PRESENT record without the v11 `validity` key is a schema break — fail
    # loud even when that pair never dropped (global scan, no quiet class vote)
    bad = _tiny_by_persona(strip_validity={(2, 0)})
    with pytest.raises(RuntimeError, match="validity"):
        LF.build_mask_and_gathers(_tiny_inputs(), bad)


def test_assert_mask_gate_schema_pins_v11_id():
    ok = {
        "generation_config_fingerprint": {"fields": {"mask_gate_schema_id": LF.MASK_GATE_SCHEMA_ID}}
    }
    LF.assert_mask_gate_schema(ok)  # no raise
    for bad in (
        {},
        {"generation_config_fingerprint": {}},
        {
            "generation_config_fingerprint": {
                "fields": {"mask_gate_schema_id": "issue823_mask_gate_v10_span_proxy"}
            }
        },
    ):
        with pytest.raises(RuntimeError, match="mask_gate_schema_id"):
            LF.assert_mask_gate_schema(bad)


def test_arm_target_gathers_correct_store_rows():
    inputs = _tiny_inputs()
    mask, gathers, _ = LF.build_mask_and_gathers(inputs, _tiny_by_persona())
    layer = 1
    y = LF.arm_target(inputs, gathers, "k2", layer, mask, mask)
    for out_row, ctx in enumerate(mask.tolist()):
        p = ctx % 2
        store_row = {int(c): j for j, c in enumerate(inputs.store_ctx[p])}[ctx]
        expected = inputs.store_v[p][store_row, layer, :].astype(np.float64)
        np.testing.assert_allclose(y[out_row], expected)
    # own/plain anchors index the parent tensors directly
    y_own = LF.arm_target(inputs, gathers, "own", layer, mask, mask)
    np.testing.assert_allclose(y_own, inputs.parent_arm["own"][mask, layer, :].astype(np.float64))


# ── M2 paired separation ─────────────────────────────────────────────────────


def test_m2_paired_separation_passes_on_strong_shift_fails_on_none():
    strong = _tiny_inputs(n_ctx=32, shift=10.0)
    mask, _, _ = LF.build_mask_and_gathers(strong, _tiny_by_persona(n_ctx=32))
    m2 = LF.m2_paired_separation(strong, mask)
    assert m2["m2_pass"] is True
    assert m2["n_personas_passing"] == LF.N_PERSONAS - 1
    none = _tiny_inputs(n_ctx=32, shift=0.0)  # identical rows -> zero shift, zero floor
    m2_none = LF.m2_paired_separation(none, mask)
    assert m2_none["m2_pass"] is False


# ── Checkpoint fingerprint gating ────────────────────────────────────────────


def test_chunk_done_fingerprint_gating(tmp_path):
    fp = LF.checkpoint_fingerprint(np.arange(5), {"chunk": "p1", "n_contexts": 5})
    assert LF.chunk_done(tmp_path, "p1_L00", fp) is False
    LF.save_chunk(tmp_path, "p1_L00", {"z": np.arange(3)}, fp)
    assert LF.chunk_done(tmp_path, "p1_L00", fp) is True
    fp2 = LF.checkpoint_fingerprint(np.arange(5), {"chunk": "p1", "n_contexts": 6})
    with pytest.raises(RuntimeError, match="fingerprint"):
        LF.chunk_done(tmp_path, "p1_L00", fp2)
    (tmp_path / "p1_L00.npz").unlink()
    with pytest.raises(RuntimeError, match="partial"):
        LF.chunk_done(tmp_path, "p1_L00", fp)


def test_checkpoint_fingerprint_keys_on_generating_parameters():
    fp = LF.checkpoint_fingerprint(np.arange(4), {"chunk": "p1"})
    assert fp["p1_grid"] == ["logspace", -2, 4, 13]  # parameters, never float bytes
    assert fp["estimator"] == LF.PRIMARY_ESTIMATOR
    # round-trips through JSON unchanged (sidecar comparison is post-round-trip)
    assert json.loads(json.dumps(fp)) == fp


# ── Designed aborts ──────────────────────────────────────────────────────────


def test_designed_abort_rcs_distinct_and_routed(tmp_path):
    assert len({LF.RC_MASK_ABORT, LF.RC_FITS_WALL_ABORT, LF.RC_SOLVER_PARITY_ABORT}) == 3
    assert {LF.RC_MASK_ABORT, LF.RC_FITS_WALL_ABORT, LF.RC_SOLVER_PARITY_ABORT} & {0, 1, 3, 4} == (
        set()
    )
    # smoke: informational, report written, NO exit
    LF.designed_abort(tmp_path, "mask_integrity", LF.RC_MASK_ABORT, {"x": 1}, smoke=True)
    rep = json.loads((tmp_path / "fits_abort_report.json").read_text())
    assert rep["abort_kind"] == "mask_integrity" and rep["rc"] == LF.RC_MASK_ABORT
    # production: SystemExit with the DISTINCT rc (never a bare rc=1)
    with pytest.raises(SystemExit) as exc:
        LF.designed_abort(tmp_path, "mask_integrity", LF.RC_MASK_ABORT, {"x": 1}, smoke=False)
    assert exc.value.code == LF.RC_MASK_ABORT


# ═══ Bounded fix round (binding reconciliation) — tests written BEFORE the fixes ═══
# Items keyed to the reconciliation fix list: F1 membership shas + early split
# integrity, F2 consumed-input fingerprint binding, F3 producer completion
# predicate + row counts, F4 per-persona per-context resume, F5 shared-eigh
# sensitivity, F6 canonical-upload gate, F7 smoke/production aliasing, F8
# retrieval on every fitted layer, F10 contingency contract, F13 smoke honesty.


# ── F1a: pre-drop MEMBERSHIP pins (counts alone pass a membership swap) ──────


def test_predrop_membership_sha_constants_match_realized_split():
    tr, va, te = LF.checked_fixed_split(5000)
    got = {"train": LF._ids_sha(tr), "val": LF._ids_sha(va), "test": LF._ids_sha(te)}
    assert got == LF.BANKED_SPLIT_PREDROP_SHA256


def test_predrop_check_catches_count_preserving_membership_swap():
    pre, parent = _parent_set_matching_banked()
    tr, va, te = (np.array(x, copy=True) for x in pre)
    # swap one TRAIN member with one VAL member — both in the parent set, so every
    # count (pre-drop AND parent-masked) is preserved; only MEMBERSHIP drifts.
    swap_tr = next(int(i) for i in tr if int(i) in parent)
    swap_va = next(int(i) for i in va if int(i) in parent)
    tr[np.where(tr == swap_tr)[0][0]] = swap_va
    va[np.where(va == swap_va)[0][0]] = swap_tr
    with pytest.raises(RuntimeError, match=r"[Mm]embership"):
        LF.predrop_banked_split_check((tr, va, te), np.array(sorted(parent)))


# ── F1b: split integrity runs BEFORE the gates and the P1 fits ───────────────


def test_split_integrity_precedes_gates_and_p1_in_main():
    src = inspect.getsource(LF.main)
    i_split = src.index("pfit_split_integrity")
    assert i_split < src.index("pfit_g2")
    assert i_split < src.index("pfit_g1")
    assert i_split < src.index("pfit_p1")


# ── F2: consumed-input identity bound into the P1 fingerprint ────────────────


def test_p1_fingerprint_binds_store_identity_and_post_gate_solver():
    src = inspect.getsource(LF.main)
    i_gate = src.index('g2_record["contingency_engaged"]')
    i_recompute = src.index(
        'solver_mode = "canonical-contingency" if contingency else "gram-fast-path"', i_gate
    )
    i_fp = src.index("fp_p1 = checkpoint_fingerprint", i_recompute)
    assert i_gate < i_recompute < i_fp  # EFFECTIVE post-gate mode, never the arg-time one
    assert "store_identity" in src[i_fp : i_fp + 400]


# ── F3: producer completion predicate + per-persona row counts ───────────────


def _fake_local_store(tmp_path, n_contexts, sidecars=True, sidecar_n_contexts=None, digest=True):
    tensors = tmp_path / "analysis_tensors"
    tensors.mkdir(parents=True, exist_ok=True)
    for p in range(LF.N_PERSONAS):
        if LF.expected_store_rows(p, n_contexts) == 0:
            continue
        (tensors / f"v_pairs_p{p:02d}.pt").write_bytes(b"x")
        if sidecars:
            (tensors / f"v_pairs_p{p:02d}.done.json").write_text(
                json.dumps(
                    {
                        "fingerprint": {
                            "n_contexts": sidecar_n_contexts or n_contexts,
                            "n_layers": 28,
                            "hidden": 3584,
                        },
                        "n_rows": 1,
                    }
                )
            )
    if digest:
        (tensors / "capture_digest.json").write_text("{}")
    return tmp_path


def test_stage_pair_store_local_requires_sidecars(tmp_path):
    root = _fake_local_store(tmp_path, 5000, sidecars=False)
    with pytest.raises(RuntimeError, match="sidecar"):
        LF.stage_pair_store(root, "prefix", None, 5000)


def test_stage_pair_store_local_refuses_stale_regime_sidecar(tmp_path):
    root = _fake_local_store(tmp_path, 5000, sidecar_n_contexts=10)
    with pytest.raises(RuntimeError, match="stale"):
        LF.stage_pair_store(root, "prefix", None, 5000)


def test_stage_pair_store_local_requires_capture_digest(tmp_path):
    root = _fake_local_store(tmp_path, 5000, digest=False)
    with pytest.raises(RuntimeError, match="capture_digest"):
        LF.stage_pair_store(root, "prefix", None, 5000)


def test_stage_pair_store_local_happy_path_returns_identity_and_digest(tmp_path):
    root = _fake_local_store(tmp_path, 5000)
    paths, identity, digest = LF.stage_pair_store(root, "prefix", None, 5000)
    assert set(paths) == set(range(LF.N_PERSONAS))
    assert identity["source"] == "local-sidecars"
    assert len(identity["sidecar_fingerprint_sha256"]) == 64
    assert digest.name == "capture_digest.json" and digest.exists()


def test_expected_store_rows_matches_registered_pair_arithmetic():
    from scripts.issue823_ladder_gen import registered_pair_total

    assert LF.expected_store_rows(0, 5000) == 5000
    assert LF.expected_store_rows(1, 5000) == 2500
    assert LF.expected_store_rows(2, 5000) == 1250
    assert LF.expected_store_rows(4, 5000) == 625
    assert LF.expected_store_rows(8, 5000) == 312
    assert LF.expected_store_rows(15, 5000) == 312
    total = sum(LF.expected_store_rows(p, 5000) for p in range(LF.N_PERSONAS))
    assert total == registered_pair_total(5000) == 14996
    assert LF.expected_store_rows(15, 10) == 0  # smoke slice: personas >= 10 have no rows


def test_load_inputs_asserts_per_persona_row_count_source_pin():
    src = inspect.getsource(LF.load_inputs)
    assert "expected_store_rows" in src  # row-count assert wired at the load seam


# ── F4: per-persona per-context arrays survive resume ────────────────────────


def test_restore_pp_arrays_roundtrip_and_set_check(tmp_path):
    fp = LF.checkpoint_fingerprint(np.arange(4), {"chunk": "p2", "n_contexts": 4})
    pp = {}
    for layer in LF.P2_LAYERS:
        for suff in ("a_sres", "a_stot", "c_sres", "c_stot"):
            pp[f"pp_k2_p0_L{layer}_{suff}"] = np.arange(3, dtype=float)
    LF.save_chunk(
        tmp_path,
        "p2_pp_k2",
        {
            "cells": np.array(json.dumps({"p0": {"L14": {}}})),
            "gmix": np.array(json.dumps([])),
            **pp,
        },
        fp,
    )
    z = np.load(tmp_path / "p2_pp_k2.npz", allow_pickle=True)
    p2_pc: dict = {}
    n = LF.restore_pp_arrays(z, p2_pc)
    assert n == len(pp) and set(p2_pc) == set(pp)
    per_persona = {"k2": {"p0": {f"L{layer}": {} for layer in LF.P2_LAYERS}}}
    LF.assert_p2_percontext_complete(per_persona, p2_pc)  # complete -> no raise
    del p2_pc["pp_k2_p0_L14_a_sres"]
    with pytest.raises(RuntimeError, match="missing"):
        LF.assert_p2_percontext_complete(per_persona, p2_pc)
    # skipped small cells impose no per-context keys
    LF.assert_p2_percontext_complete({"k2": {"p1": {"status": "skipped_small_cell (smoke)"}}}, {})


def test_p2_resume_and_set_check_wired_in_main():
    src = inspect.getsource(LF.main)
    i_p2 = src.index("pfit_p2")
    assert "restore_pp_arrays" in src[i_p2:]
    assert "assert_p2_percontext_complete" in src[i_p2:]


# ── F5: capped sensitivity shares ONE factorization per (layer, fold) ────────


def test_dof_cap_sensitivity_one_factorization_per_layer_fold(tmp_path, monkeypatch):
    inputs = _tiny_inputs(n_ctx=20)
    mask, gathers, _ = LF.build_mask_and_gathers(inputs, _tiny_by_persona(n_ctx=20))
    folds = list(KFold(n_splits=5, shuffle=True, random_state=0).split(np.zeros(len(mask))))
    calls = {"n": 0}
    real = LF.factorize_robust

    def counting(x, dev_):
        calls["n"] += 1
        return real(x, dev_)

    monkeypatch.setattr(LF, "factorize_robust", counting)
    fp = LF.checkpoint_fingerprint(mask, {"chunk": "p1_sens", "n_contexts": 20})
    layers = (1, 2)
    cells = LF.dof_cap_sensitivity(
        inputs, gathers, mask, folds, torch.device("cpu"), tmp_path, fp, layers=layers
    )
    # ONE eigh per (layer, fold), shared across all 7 arms — 105 -> 15 at production shape
    assert calls["n"] == len(layers) * 5
    cell = cells["k1:L1"]
    assert cell["cap"] == LF.DOF_CAP and len(cell["folds"]) == 5
    assert set(cell["folds"][0]) >= {"fold", "lambda", "lambda_edge", "dof", "n_train"}
    assert set(cells) == {f"{arm}:L{layer}" for arm in LF.ARM_NAMES for layer in layers}
    # per-layer checkpoints resume without recomputing any factorization
    calls["n"] = 0
    cells2 = LF.dof_cap_sensitivity(
        inputs, gathers, mask, folds, torch.device("cpu"), tmp_path, fp, layers=layers
    )
    assert calls["n"] == 0 and set(cells2) == set(cells)


# ── F6: canonical-upload gate (the SHARED gen gate, not a second copy) ───────


def test_canonical_upload_gate_is_the_shared_gen_gate():
    from scripts.issue823_ladder_gen import _require_canonical_upload as gate

    assert LF._require_canonical_upload is gate
    src = inspect.getsource(LF.main)
    assert "_require_canonical_upload(url" in src


# ── F7: smoke and production outputs must never alias ────────────────────────


def test_smoke_root_aliasing_predicate_and_sentinel_names():
    assert LF.smoke_root_aliases_production(LF.PROD_POD_OUT_ROOT) is True
    assert LF.smoke_root_aliases_production(LF.PROD_POD_OUT_ROOT / "fits_smoke") is True
    assert LF.smoke_root_aliases_production(pathlib.Path("/tmp/issue-823-smoke/lf")) is False
    assert LF.sentinel_filename(False) == "issue-823-ladder-fits-done.json"
    assert LF.sentinel_filename(True) == "issue-823-ladder-fits-smoke-done.json"


def test_main_refuses_smoke_out_root_under_production():
    with pytest.raises(SystemExit) as exc:
        LF.main(["--smoke", "--out-root", str(LF.PROD_POD_OUT_ROOT / "fits_smoke")])
    assert exc.value.code == 2  # argparse error BEFORE any staging / mkdir


# ── F8: retrieval baseline covers every fitted P1 layer ──────────────────────


def test_p1_baselines_cover_every_fitted_layer():
    src = inspect.getsource(LF.main)
    assert "if layer in READ_OUT_LAYERS or layer in P2_LAYERS:" not in src
    assert "layer_base[key] = cell_baselines" in src


# ── F10: contingency contract ────────────────────────────────────────────────


def test_g1_dispatches_effective_solver_under_contingency():
    src = inspect.getsource(LF.main)
    g1_block = src[src.index("pfit_g1") : src.index("pfit_p1")]
    assert "if contingency:" in g1_block
    assert "ridge_fit_predict(x_tr, y_tr, x_te)" in g1_block  # the EFFECTIVE headline solver


def test_p2_withheld_result_shape_and_wiring():
    rec = LF.p2_withheld_result({"m": 1}, {"s": 2})
    assert rec["status"].startswith("WITHHELD")
    assert rec["metadata"] == {"m": 1} and rec["split"] == {"s": 2}
    assert "unverified" in rec["reason"]
    assert rec["full_arms"] == {} and rec["n_ladder"] == {} and rec["per_persona"] == {}
    src = inspect.getsource(LF.main)
    assert "p2_withheld_result" in src[src.index("pfit_p2") :]


def test_completion_sentinel_annotates_gates_and_p2_status():
    src = inspect.getsource(LF.main)
    tail = src[src.index("write_sentinel") :]
    assert '"p2_status"' in tail and '"gates"' in tail
    assert "sentinel_filename(args.smoke)" in src


# ── F13: smoke honesty — degenerate val selection fail-loud / labeled ────────


def test_val_select_lambda_degenerate_raises_in_production_and_labels_in_smoke():
    rng = np.random.default_rng(5)
    x = rng.normal(size=(8, 3))
    y = rng.normal(size=(8, 2))
    xv = rng.normal(size=(1, 3))
    yv = rng.normal(size=(1, 2))  # ONE val row -> ss_tot degenerate -> every score NaN
    fact = FFC._factorize(x, torch.device("cpu"))
    vty, ymu = FFC._vty_ymu(fact, y)
    kval = FFC._cross_kernel(fact, xv)
    grid = np.logspace(-2, 8, 5)
    with pytest.raises(RuntimeError, match="degenerate"):
        LF.val_select_lambda(fact, vty, ymu, kval, yv, grid)
    lam, r2 = LF.val_select_lambda(fact, vty, ymu, kval, yv, grid, degenerate_ok=True)
    assert lam == grid[0] and not np.isfinite(r2)


def test_module_docstring_enumerates_predrop_skip_and_val_degeneracy():
    blind = LF.__doc__[LF.__doc__.index("Smoke blind-spot enumeration") :]
    assert "pre-drop banked-split" in blind  # the smoke SKIP of the check is ENUMERATED
    assert "val-selection" in blind and "degenerate" in blind
