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

import json

import numpy as np
import pytest
import torch

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


def _tiny_inputs(n_ctx=8, n_layers=28, hidden=4, invalid_pair=None, shift=None):
    """Synthetic LadderInputs; invalid_pair=(ctx, persona) zeroes one span."""
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
    if invalid_pair is not None:
        ctx, p = invalid_pair
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


def test_build_mask_drops_invalid_pair_and_counts_per_arm():
    inputs = _tiny_inputs(invalid_pair=(3, 3))  # ctx 3, persona 3 (= 3 % 4)
    mask, _gathers, drops = LF.build_mask_and_gathers(inputs)
    assert 3 not in set(mask.tolist())
    assert len(mask) == inputs.n_contexts - 1
    # persona 3 serves ctx 3 under k4/k8/k16 (3%4 == 3%8 == 3%16 == 3), not k1/k2
    assert drops["new_drops_per_arm"] == {"k1": 0, "k2": 0, "k4": 1, "k8": 1, "k16": 1}
    assert drops["new_dropped_ids_union"] == [3]


def test_arm_target_gathers_correct_store_rows():
    inputs = _tiny_inputs()
    mask, gathers, _ = LF.build_mask_and_gathers(inputs)
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
    mask, _, _ = LF.build_mask_and_gathers(strong)
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
