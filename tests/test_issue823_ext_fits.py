"""CPU tests for scripts/issue823_ladder_ext_fits.py (#823 origin-ladder-more-contexts, unit 4).

Covers: the designed-halt rc table (19-22, disjoint from the sibling drivers'),
dual/primal/canonical three-way solver agreement on toy fixtures (n > d AND
n < d), parity of `solve_capped` against the parent's named reference
`LF.gcv_solve_dof_capped`, the dof-cap refusal path, Gate C projection
arithmetic + rc-19 halt, Gate D slice records + the rc-20 contingency
terminal, Gate E compare functions + rc-21, companion-manifest construction
+ set-checks + rc-22, row-coverage fail-loud, the fingerprinted completion
sentinel (roundtrip + mutation), fit_rung on a synthetic source (dof-cap
persisted per fit, chunk resume, fingerprint-mutation refusal), the rung-dir
writer's paired-script consumption contract (including an end-to-end
subprocess run of the UNMODIFIED `issue823_shared_persona_paired.py` through
the pod-safe shim), P2 grid clamping, and the smoke mask cap.

Flat synthetic fixtures only — no network, no GPU, no real-corpus text. Each
designed-halt test asserts the DISTINCT rc from the driver's docstring table
AND that no downstream completion sentinel exists after the halt.
"""

from __future__ import annotations

import json
import math
import pathlib

import numpy as np
import pytest
import torch

from scripts import issue823_ladder_ext_capture as EXTCAP
from scripts import issue823_ladder_ext_fits as FITS
from scripts import issue823_ladder_fits as LF
from scripts import issue823_shared_persona_paired as SPP

CPU = torch.device("cpu")

# ── Fixtures ─────────────────────────────────────────────────────────────────


def _toy_xy(n: int, d: int = 8, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, d))
    w = rng.normal(size=(d, d))
    y = x @ w + 0.1 * rng.normal(size=(n, d))
    return x, y


class FakeFitSources:
    """Duck-typed fit source: layer-independent (n_total, d) design + arm targets."""

    def __init__(self, n_total: int, d: int = 8, seed: int = 0, bad_ids: tuple[int, ...] = ()):
        rng = np.random.default_rng(seed)
        self.x = rng.normal(size=(n_total, d))
        w = rng.normal(size=(d, d))
        self.y = {
            "k1": self.x @ w + 0.1 * rng.normal(size=(n_total, d)),
            "k16": self.x @ (0.5 * w) + 0.1 * rng.normal(size=(n_total, d)),
        }
        self.bad = set(bad_ids)

    def cx_col(self, layer: int, ids) -> np.ndarray:
        return self.x[np.asarray(ids, dtype=np.int64)]

    def arm_col(self, arm: str, layer: int, ids) -> np.ndarray:
        return self.y[arm][np.asarray(ids, dtype=np.int64)]

    def pair_ok(self, ctx_id: int, persona: int) -> bool:
        return int(ctx_id) not in self.bad

    def has_cx(self, ctx_id: int) -> bool:
        return True


class FakePairSources:
    """pair_col-only source for write_rung_dir: v_p(i) = 0.01 * p * ones(HIDDEN)."""

    def pair_col(self, persona: int, layer: int, ids) -> np.ndarray:
        n = len(np.asarray(ids))
        return np.full((n, FITS.HIDDEN), 0.01 * int(persona) + 1e-4 * int(layer))


def _no_sentinels(root: pathlib.Path) -> bool:
    names = ("_fits_complete.json", "_capture_complete.json", "_owngen_complete.json")
    return not any(list(root.rglob(name)) for name in names)


def _make_rungfit(tag: str, label: str, train_ids, eval_ids, seed: int = 3) -> FITS.RungFit:
    rng = np.random.default_rng(seed)
    n_eval = len(eval_ids)
    shape = (len(FITS.ARM_NAMES), FITS.EXPECTED_LAYERS, n_eval)
    return FITS.RungFit(
        tag=tag,
        label=label,
        train_ids=np.asarray(train_ids, dtype=np.int64),
        eval_ids=np.asarray(eval_ids, dtype=np.int64),
        sres=np.abs(rng.normal(size=shape)) + 0.1,
        stot=np.abs(rng.normal(size=shape)) + 1.0,
        id_sres=np.abs(rng.normal(size=shape)) + 0.1,
        id_stot=np.abs(rng.normal(size=shape)) + 1.0,
        cells={},
        fit_records=[],
        g2_slices=[],
        knn={},
        sens_pure={},
        solver="dual",
        fold_ns=[int(len(train_ids) * 4 // 5)] * FITS.N_FOLDS,
    )


# ── rc table ─────────────────────────────────────────────────────────────────


def test_rc_table_distinct_and_disjoint_from_siblings():
    assert set(FITS.RC_TABLE) == {19, 20, 21, 22}
    assert (
        len(
            {
                FITS.RC_FITS_WALL,
                FITS.RC_SOLVER_PARITY,
                FITS.RC_BANKED_CONTINUITY,
                FITS.RC_RAND_MANIFEST,
            }
        )
        == 4
    )
    sibling_rcs = {int(k) for k in EXTCAP.RC_TABLE}
    assert not (set(FITS.RC_TABLE) & sibling_rcs)
    # parent driver reserves 5/6/7; unit 2 reserves 3, 6-11 — all below 19.
    assert min(FITS.RC_TABLE) > 18


# ── solver core ──────────────────────────────────────────────────────────────


def test_threeway_dual_primal_canonical_agreement_n_gt_d():
    x, y = _toy_xy(40)
    x_ev = _toy_xy(10, seed=1)[0]
    fact_d = LF.factorize_robust(x, CPU)
    fact_d["kind"] = "dual"
    lam_d, proj_d, ymu_d, dof_d = FITS.solve_capped(fact_d, y)
    pred_d = FITS.apply_fit(fact_d, lam_d, proj_d, ymu_d, FITS.eval_kernel(fact_d, x_ev))
    fact_p = FITS._factorize_primal(x, CPU)
    lam_p, proj_p, ymu_p, dof_p = FITS.solve_capped(fact_p, y)
    pred_p = FITS.apply_fit(fact_p, lam_p, proj_p, ymu_p, FITS.eval_kernel(fact_p, x_ev))
    pred_c, lam_c, dof_c = FITS.canonical_capped_fit(x, y, x_ev)
    assert lam_d == lam_p == lam_c
    assert abs(dof_d - dof_c) < 1e-6 and abs(dof_p - dof_c) < 1e-6
    scale = np.abs(pred_c).max()
    assert np.abs(pred_d - pred_c).max() / scale < 1e-8
    assert np.abs(pred_p - pred_c).max() / scale < 1e-8


def test_threeway_agreement_n_lt_d():
    x, y = _toy_xy(6, d=8, seed=2)
    x_ev = _toy_xy(4, d=8, seed=5)[0]
    fact_d = LF.factorize_robust(x, CPU)
    fact_d["kind"] = "dual"
    lam_d, proj_d, ymu_d, _ = FITS.solve_capped(fact_d, y)
    pred_d = FITS.apply_fit(fact_d, lam_d, proj_d, ymu_d, FITS.eval_kernel(fact_d, x_ev))
    fact_p = FITS._factorize_primal(x, CPU)
    lam_p, proj_p, ymu_p, _ = FITS.solve_capped(fact_p, y)
    pred_p = FITS.apply_fit(fact_p, lam_p, proj_p, ymu_p, FITS.eval_kernel(fact_p, x_ev))
    pred_c, lam_c, _ = FITS.canonical_capped_fit(x, y, x_ev)
    assert lam_d == lam_p == lam_c
    scale = np.abs(pred_c).max()
    assert np.abs(pred_d - pred_c).max() / scale < 1e-8
    assert np.abs(pred_p - pred_c).max() / scale < 1e-8


def test_solve_capped_matches_parent_named_reference():
    """Estimator-diff duty: solve_capped == LF.gcv_solve_dof_capped on the dual path."""
    x, y = _toy_xy(30, seed=7)
    fact = LF.factorize_robust(x, CPU)
    lam_ref, vty_ref, ymu_ref, dof_ref = LF.gcv_solve_dof_capped(fact, y, cap_frac=FITS.DOF_CAP)
    fact["kind"] = "dual"
    lam, proj, ymu, dof = FITS.solve_capped(fact, y, FITS.LAMBDAS, FITS.DOF_CAP)
    assert lam == lam_ref
    assert abs(dof - dof_ref) < 1e-9
    assert torch.allclose(proj, vty_ref) and torch.allclose(ymu, ymu_ref)


def test_dof_cap_refusal_raises():
    x, y = _toy_xy(12, seed=9)
    fact = FITS.rung_factorize(x, CPU)
    with pytest.raises(RuntimeError, match="excludes EVERY lambda"):
        FITS.solve_capped(fact, y, FITS.LAMBDAS, 1e-9)
    with pytest.raises(RuntimeError, match="excludes every grid lambda"):
        FITS.canonical_capped_fit(x, y, x, FITS.LAMBDAS, 1e-9)


def test_dof_cap_binds_and_pure_gcv_exceeds():
    """Capped selection honors dof <= 0.9 n_train; pure GCV (cap=inf) may exceed it."""
    x, y = _toy_xy(10, d=32, seed=11)
    fact = FITS.rung_factorize(x, CPU)
    _, _, _, dof_capped = FITS.solve_capped(fact, y, FITS.LAMBDAS, FITS.DOF_CAP)
    assert dof_capped <= FITS.DOF_CAP * 10 + 1e-9
    _, _, _, dof_pure = FITS.solve_capped(fact, y, FITS.LAMBDAS, math.inf)
    assert dof_pure >= dof_capped - 1e-9


def test_rung_factorize_routes_by_n(monkeypatch):
    monkeypatch.setattr(FITS, "DUAL_N_MAX", 30)
    x, _ = _toy_xy(40, seed=13)
    assert FITS.rung_factorize(x, CPU)["kind"] == "primal"
    x_small, _ = _toy_xy(20, seed=13)
    assert FITS.rung_factorize(x_small, CPU)["kind"] == "dual"


# ── Gate C (rc 19) ───────────────────────────────────────────────────────────


def test_gate_c_projection_and_rc19(tmp_path):
    wall = FITS.project_battery_wall(2.0, 10.0, 100, 50, boot_allowance_s=600.0)
    assert wall == pytest.approx((2.0 * 100 + 10.0 * 50 + 600.0) / 3600.0)
    ok = FITS.gate_c_record(0.1, 0.1, 10, 10, planned_wall_h=3.0)
    assert ok["pass"]
    FITS.enforce_gate_c(ok, tmp_path, smoke=False)  # no raise
    bad = FITS.gate_c_record(100.0, 400.0, 500, 500, planned_wall_h=3.0)
    assert not bad["pass"]
    FITS.enforce_gate_c(bad, tmp_path, smoke=True)  # smoke: informational only
    with pytest.raises(SystemExit) as ei:
        FITS.enforce_gate_c(bad, tmp_path, smoke=False)
    assert ei.value.code == FITS.RC_FITS_WALL
    report = json.loads((tmp_path / "ext_fits_wall_report.json").read_text())
    assert report["gate_c"]["projected_wall_h"] > 2.0 * 3.0
    assert _no_sentinels(tmp_path)


# ── Gate D (rc 20) ───────────────────────────────────────────────────────────


def test_g2_slice_record_pass_and_fail():
    pred = np.random.default_rng(0).normal(size=(6, 4))
    ok = FITS.g2_slice_record(pred, pred.copy(), 1.0, 1.0, 0.5, 0.5, 14, 0, "k1")
    assert ok["pass"] and ok["max_rel"] == 0.0
    bad_pred = FITS.g2_slice_record(pred * 1.01, pred, 1.0, 1.0, 0.5, 0.5, 14, 0, "k1")
    assert not bad_pred["pass"]
    bad_lam = FITS.g2_slice_record(pred, pred.copy(), 10.0, 1.0, 0.5, 0.5, 14, 0, "k1")
    assert not bad_lam["pass"] and not bad_lam["lambda_agree"]


def test_contingency_parity_check_passes_on_toy():
    x, y = _toy_xy(25, seed=17)
    x_ev = _toy_xy(5, seed=18)[0]
    rec = FITS.contingency_parity_check(x, y, x_ev, CPU)
    assert rec["pass"] and rec["max_rel"] < 1e-8


def test_contingency_parity_fail_halts_rc20(tmp_path):
    records = [{"max_rel": 1.0, "lambda_canonical": 1.0, "lambda_primal": 10.0, "pass": False}]
    with pytest.raises(SystemExit) as ei:
        FITS.enforce_contingency_parity(records, tmp_path, "primary/5000")
    assert ei.value.code == FITS.RC_SOLVER_PARITY
    assert (tmp_path / "ext_solver_parity_report.json").exists()
    assert _no_sentinels(tmp_path)


# ── Gate E (rc 21) ───────────────────────────────────────────────────────────


def _banked_like(val: float) -> dict:
    return {
        "arms": {
            "k16": {
                "per_layer": {
                    "L14": {"offset_bias_control": {"ratio_measured_over_full_energy": val}}
                }
            }
        }
    }


def test_bridge_loader_compare_and_rc21(tmp_path):
    ok = FITS.bridge_loader_compare(_banked_like(0.771), _banked_like(0.771))
    assert ok["pass"]
    bad = FITS.bridge_loader_compare(_banked_like(0.7711), _banked_like(0.771))
    assert not bad["pass"]
    with pytest.raises(SystemExit) as ei:
        FITS.enforce_gate_e(bad, tmp_path, "loader-level rerun (E(i))")
    assert ei.value.code == FITS.RC_BANKED_CONTINUITY
    assert (tmp_path / "ext_banked_continuity_report.json").exists()
    assert _no_sentinels(tmp_path)


def test_bridge_refit_compare_abs_tolerance():
    assert FITS.bridge_refit_compare({"L14": 0.80}, {"L14": 0.771})["pass"]  # |d| = 0.029
    assert not FITS.bridge_refit_compare({"L14": 0.90}, {"L14": 0.771})["pass"]


# ── Companion ladder (rc 22) ─────────────────────────────────────────────────


def _companion_inputs():
    # 320 banked-era ids (0..319) + 480 ext-era ids (1000..1479); n_prefix=1000.
    top = np.concatenate([np.arange(320), np.arange(1000, 1480)]).astype(np.int64)
    rung_sizes = {"100": 100, "300": 300, "800": len(top)}
    return top, rung_sizes, 1000


def test_companion_sets_nesting_strata_and_manifest():
    top, rung_sizes, n_prefix = _companion_inputs()
    e_eval, subsets, manifest = FITS.build_companion_sets(top, rung_sizes, n_prefix)
    assert FITS.check_companion_manifest(e_eval, subsets, manifest, top) == []
    e_set = set(e_eval.tolist())
    assert all(i % FITS.POOLED_K == 0 for i in e_eval)
    labels = sorted(subsets, key=lambda k: rung_sizes[k])
    prev: set[int] = set()
    for label in labels:
        t = set(subsets[label].tolist())
        assert prev <= t and not (t & e_set)
        prev = t
    # top companion rung == whole pool (top mask minus E_eval)
    assert len(subsets[labels[-1]]) == len(top) - len(e_eval)
    # era stratification within ~1% of pool proportions
    pool_frac = manifest["pool"]["banked_fraction"]
    for label in labels[:-1]:
        t = subsets[label]
        frac = float((t < n_prefix).mean())
        assert abs(frac - pool_frac) < 0.05


def test_companion_manifest_violation_halts_rc22(tmp_path):
    top, rung_sizes, n_prefix = _companion_inputs()
    e_eval, subsets, manifest = FITS.build_companion_sets(top, rung_sizes, n_prefix)
    # corrupt: leak an E_eval id into the smallest subset
    small = sorted(subsets, key=lambda k: rung_sizes[k])[0]
    subsets[small] = np.concatenate([subsets[small], e_eval[:1]])
    violations = FITS.check_companion_manifest(e_eval, subsets, manifest, top)
    assert violations
    with pytest.raises(SystemExit) as ei:
        FITS.enforce_companion_manifest(violations, tmp_path, manifest)
    assert ei.value.code == FITS.RC_RAND_MANIFEST
    assert (tmp_path / "ext_rand_manifest_report.json").exists()
    assert _no_sentinels(tmp_path)


# ── Row coverage + masks ─────────────────────────────────────────────────────


def test_row_coverage_raises_on_missing_rows():
    rf = _make_rungfit("primary", "100", np.arange(20), np.arange(20))
    FITS.row_coverage_check(rf, (14, 26, 17))  # finite fixture passes
    rf.sres[0, 14, 3] = np.nan
    with pytest.raises(RuntimeError, match="row-coverage"):
        FITS.row_coverage_check(rf, (14, 26, 17))


def test_realize_rung_masks_drop_accounting_and_smoke_cap():
    src = FakeFitSources(64, bad_ids=(3, 17))
    mask_obj = {"rungs": {"32": {"ids": list(range(32))}, "64": {"ids": list(range(64))}}}
    masks, drops = FITS.realize_rung_masks(mask_obj, src, smoke=False)
    assert drops["32"]["n_capture_dropped"] == 2
    assert set(masks["32"]) == set(range(32)) - {3, 17}
    assert drops["64"]["n_realized"] == 62
    capped, _ = FITS.realize_rung_masks(mask_obj, src, smoke=True, cap=10)
    assert len(capped["64"]) == 10


def test_smoke_cap_mask_keeps_shared_persona_rows_first():
    ids = np.arange(100)
    capped = FITS.smoke_cap_mask(ids, 10)
    assert len(capped) == 10
    shared = [i for i in capped if i % FITS.POOLED_K == 0]
    assert len(shared) == 7  # all of {0,16,32,48,64,80,96} kept before filler ids


# ── Fingerprint sentinel ─────────────────────────────────────────────────────


def test_fits_sentinel_roundtrip_and_mutation(tmp_path):
    fp = {"rung_mask_shas": {"100": "a" * 64}, "estimator": {"grid": [1, 2]}}
    assert not FITS.fits_done(tmp_path, fp)
    FITS.write_fits_sentinel(tmp_path, fp, {"smoke": True})
    assert FITS.fits_done(tmp_path, fp)
    mutated = {"rung_mask_shas": {"100": "b" * 64}, "estimator": {"grid": [1, 2]}}
    assert not FITS.fits_done(tmp_path, mutated)


# ── fit_rung on a synthetic source ───────────────────────────────────────────


def test_fit_rung_primary_dof_cap_g2_resume_and_mutation(tmp_path, monkeypatch):
    src = FakeFitSources(30, seed=21)
    ids = np.arange(25, dtype=np.int64)
    ckpt = tmp_path / "ckpt"
    rf = FITS.fit_rung(
        "primary",
        "25",
        ids,
        src,
        CPU,
        ckpt,
        layers=(14,),
        g2_slices=((14, 0),),
        sens_pure=True,
    )
    assert np.isfinite(rf.sres[:, 14, :]).all() and np.isfinite(rf.stot[:, 14, :]).all()
    assert rf.fit_records and all(
        r["dof"] <= FITS.DOF_CAP * r["n_train"] + 1e-9 for r in rf.fit_records
    )
    assert rf.g2_slices and all(s["pass"] for s in rf.g2_slices)
    assert rf.sens_pure  # rung-1 pure-GCV sensitivity populated at read-out layer
    assert any(k.startswith("k1:L14") for k in rf.knn)
    # resume: a second call must load the chunk without refitting
    monkeypatch.setattr(FITS, "rung_factorize", _raise_if_called)
    rf2 = FITS.fit_rung(
        "primary",
        "25",
        ids,
        src,
        CPU,
        ckpt,
        layers=(14,),
        g2_slices=((14, 0),),
        sens_pure=True,
    )
    assert np.allclose(rf.sres[:, 14, :], rf2.sres[:, 14, :])
    monkeypatch.undo()
    # fingerprint mutation (different mask, same chunk name) => fail-loud
    with pytest.raises(RuntimeError, match="DIFFERENT fingerprint"):
        FITS.fit_rung("primary", "25", np.arange(26), src, CPU, ckpt, layers=(14,))


def _raise_if_called(*a, **k):
    raise AssertionError("resume path must not refit")


def test_fit_rung_companion_mean_aggregation_and_knn(tmp_path):
    src = FakeFitSources(60, seed=23)
    train = np.arange(40, dtype=np.int64)
    e_eval = np.arange(40, 56, dtype=np.int64)
    rf = FITS.fit_rung(
        "companion",
        "40",
        train,
        src,
        CPU,
        tmp_path / "ckpt",
        layers=(14,),
        eval_ids=e_eval,
        g2_slices=((14, 0),),
    )
    assert rf.sres.shape[2] == len(e_eval)
    assert np.isfinite(rf.sres[:, 14, :]).all()
    # ss_tot centered on the eval population's own mean, fold-independent
    y = src.arm_col("k1", 14, e_eval)
    expect_tot = ((y - y.mean(0)) ** 2).sum(axis=1)
    assert np.allclose(rf.stot[0, 14, :], expect_tot)
    assert any(k == "k1:L14" for k in rf.knn)  # companion fold-mean retrieval read
    assert rf.g2_slices and all(s["pass"] for s in rf.g2_slices)


# ── Rung-dir writer + paired-script consumption contract ────────────────────


def _write_toy_rung_dir(tmp_path: pathlib.Path):
    n_total = 320
    train_ids = np.arange(n_total, dtype=np.int64)
    rf = _make_rungfit("primary", "320", train_ids, train_ids)
    rung_dir = tmp_path / "rung_320"
    implied = FITS.write_rung_dir(
        rung_dir, rf, FakePairSources(), n_total, {"phase": "test"}, diff_train_ids=train_ids
    )
    return rung_dir, rf, implied


def test_write_rung_dir_matches_load_mixture_diffs_and_energy(tmp_path):
    rung_dir, _rf, implied = _write_toy_rung_dir(tmp_path)
    md = SPP.load_mixture_diffs(
        rung_dir / "mixture_diffs.npz", (FITS.POOLED_K,), tuple(FITS.READ_OUT_LAYERS)
    )
    groups = md.groups(FITS.POOLED_K, 14)
    assert [p for p, _ in groups] == list(range(1, 16))  # ascending personas, no p0
    n0 = md.n_persona0(FITS.POOLED_K, -1)
    assert n0 == 20  # 320/16 shared-persona rows in the denominator population
    from scripts.issue823_ladder_common import mixture_energy_from_group_diffs

    # md.groups yields (PERSONA id, D_p); the energy helper takes (COUNT, D_p).
    groups_nd = [(d.shape[0], d) for _p, d in groups]
    e_direct = mixture_energy_from_group_diffs(iter(groups_nd), n0)
    e_summary = implied[f"k{FITS.POOLED_K}:L14"]["between_persona_mean_shift_energy"]
    assert e_direct == pytest.approx(e_summary, rel=1e-12)
    assert e_direct > 0
    z = np.load(rung_dir / "percontext_ladder.npz")
    assert list(z["arm_names"]) == ["k1", "k16"]
    assert z["p1_ss_res"].shape == (2, FITS.EXPECTED_LAYERS, 320)
    assign = json.loads((rung_dir / "assignment.json").read_text())
    assert assign["arms"]["16"][17] == 1 and len(assign["arms"]["1"]) == 320


@pytest.mark.slow
def test_paired_script_subprocess_end_to_end(tmp_path):
    """The UNMODIFIED paired script consumes a unit-4 rung dir via the pod-safe shim."""
    rung_dir, _, _ = _write_toy_rung_dir(tmp_path)
    out = tmp_path / "paired.json"
    FITS.run_paired_script(
        FITS._REPO_ROOT, out, rung_dir, arms=str(FITS.POOLED_K), n_boot=50, full_ratio_ci=True
    )
    d = json.loads(out.read_text())
    cell = d["arms"][f"k{FITS.POOLED_K}"]["per_layer"]["L14"]
    obc = cell["offset_bias_control"]
    assert "ratio_measured_over_full_energy" in obc
    # --full-ratio-ci fields attach at the per-layer cell top level (SPP main()).
    assert "rho_ci95" in cell and "n_negligible_E_draws" in cell


# ── P2 grid ──────────────────────────────────────────────────────────────────


def test_p2_rung_grid_clamps_and_appends_realized_max():
    grid = FITS.p2_rung_grid(5_000)
    assert grid[-1] == 5_000 and all(n <= 5_000 for n in grid)
    assert 3_584 in grid and 28_672 not in grid
    full = FITS.p2_rung_grid(40_000)
    assert full[-1] == 40_000 and 28_672 in full


def test_p2_boundary_ladder_smoke_toy():
    src = FakeFitSources(60, seed=29)
    out = FITS.p2_boundary_ladder(src, np.arange(60), CPU, smoke=True)
    assert out["holdout_n"] == 20 and out["pool_n"] == 40
    assert out["cells"]
    for cell in out["cells"].values():
        assert cell["dof"] <= FITS.DOF_CAP * cell["n_train"] + 1e-9
        assert "identity_bias_r2" in cell and "knn" in cell
