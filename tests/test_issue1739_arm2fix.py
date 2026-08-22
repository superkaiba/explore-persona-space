"""#1739 arm2fix (leg 2) unit tests — plan §4 "Leg 2" D0/D1/D2 code surface.

Covers, in plan order:
1. D0-P1 (wiring probe, committed regardless of outcome): a PLANTED-direction
   synthetic through the REAL ``run_transfer_cell`` dispatch with arm2 in the
   roster — eval-block scores must equal the hand-computed projection of the
   direction fit on the fold-1 (readout-pool) rows ONLY, the fit must recover
   the plant, and the wrong-wiring (train+eval pooled split) expectation must
   NOT match. P1 PASSES on the current dispatch ⇒ repair R-A (wiring fix) is
   a no-op; the ladder proceeds on P2/P3/P5 evidence.
2. arm2q (R-C quantile fallback): dispatch semantics against the SHARED
   ``arms.arm2q_thresholds`` helper + the scorer-side parity subset
   (``_quantile_fit_rows``) matching the arm-internal split rows exactly.
3. The repair-ladder pass plan (``_arm2fix_passes``) incl. the matched-budget
   parity pass (row-matched arm7 refit).
4. Parse-time guards: --skip-map-fit refusal on a map-consuming roster,
   --arms-only-extra, adapter/parity/shufpair incompatibilities.
5. The per-seed resume predicate's arm2fix keys (a legacy output can never
   satisfy an arm2fix invocation).
6. The matched-regime sanity read (real ``run_cell`` path, synthetic pool).
7. The arm2fix fold's sanity-mask + lattice branch logic on synthetic
   fixtures (single-behavior sanity fail -> exclusion + restated denominator;
   MAP-ADVANTAGE-NOT-SHOWN; MAP-BEATS parity requirement; INCONCLUSIVE-ADAPTER).
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
import sys  # noqa: E402

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import scripts.issue1739_r2v2_score as sc  # noqa: E402
from explore_persona_space.experiments.issue_1739 import arms, fits  # noqa: E402


@pytest.fixture
def roster_guard():
    """Save/restore the scorer's module-global roster around parse_args probes
    (the test_extra_arms_guard convention — never a module reload)."""
    saved = (sc.ROSTER, sc.LABEL_CONSUMING)
    yield sc
    sc.ROSTER, sc.LABEL_CONSUMING = saved


def _planted(ly=2, d=6, n_tr=40, n_ev=12, seed=1739):
    """Planted-direction synthetic: hi/lo dv bands displaced along a unit axis."""
    rng = np.random.default_rng(seed)
    u = rng.normal(size=d)
    u /= np.linalg.norm(u)
    dv_tr = np.concatenate([rng.uniform(60, 100, n_tr // 2), rng.uniform(0, 40, n_tr // 2)])
    signal = (dv_tr - dv_tr.mean()) / dv_tr.std()
    z_tr = signal[None, :, None] * u[None, None, :] + 0.05 * rng.normal(size=(ly, n_tr, d))
    z_ev = rng.normal(size=(ly, n_ev, d))
    # adversarial eval dv: if eval rows leaked into the fit split, the midpoint
    # would shift by orders of magnitude and the exact-equality assert breaks.
    dv_ev = rng.uniform(0, 1, n_ev) * 10_000.0
    return u, z_tr, dv_tr, z_ev, dv_ev


def _transfer_scores(z_tr, dv_tr, z_ev, dv_ev, roster):
    ly, n_tr = z_tr.shape[0], z_tr.shape[1]
    rng = np.random.default_rng(7)
    data = arms.CellData(
        z_ctx=z_tr, dv=dv_tr, rb=rng.normal(size=(ly, z_tr.shape[2])), layers=tuple(range(ly))
    )
    cell = fits.BudgetCell(
        row_idx=np.arange(n_tr),
        fold_ids=np.zeros(n_tr, dtype=np.int64),
        n_folds=1,
        budget_l=n_tr,
        draw=0,
        seed=0,
        fold_scheme="p1",
    )
    return arms.run_transfer_cell(
        data, cell, z_ev, dv_ev, arms=list(roster), device="cpu", ridge_folds=(0,)
    )


# ---------------------------------------------------------------------------
# 1. D0-P1: planted-direction wiring probe (pre-registered; commit either way)
# ---------------------------------------------------------------------------


def test_p1_planted_direction_transfer_wiring():
    u, z_tr, dv_tr, z_ev, dv_ev = _planted()
    scores, skipped = _transfer_scores(z_tr, dv_tr, z_ev, dv_ev, ["arm2_ctx_native"])
    assert not skipped, f"unexpected skips: {skipped}"

    # hand-computed OOF semantics: midpoint split on the TRAIN (fold-1) dv
    # ONLY; hi/lo diff-of-means direction on TRAIN z; projected on eval rows.
    mid = 0.5 * (dv_tr.max() + dv_tr.min())
    hi, lo = dv_tr >= mid, dv_tr < mid
    assert hi.any() and lo.any()
    direction = z_tr[:, hi].mean(axis=1) - z_tr[:, lo].mean(axis=1)  # (Ly, d)
    expected = np.einsum("ld,lnd->ln", direction, z_ev)
    np.testing.assert_allclose(scores["arm2_ctx_native"], expected, rtol=1e-10, atol=1e-12)

    # plant recovery: the fitted direction IS (up to noise) the planted axis.
    for li in range(z_tr.shape[0]):
        cos = float(direction[li] @ u / np.linalg.norm(direction[li]))
        assert cos > 0.99, f"layer {li}: planted-direction cosine {cos:.4f}"

    # wiring falsifier (test sensitivity): the WRONG expectation — a split
    # computed over train+eval pooled dv (the leak the probe hunts) — must
    # NOT reproduce the scores, so a pass is not vacuous.
    dv_all = np.concatenate([dv_tr, dv_ev])
    z_all = np.concatenate([z_tr, z_ev], axis=1)
    mid_bad = 0.5 * (dv_all.max() + dv_all.min())
    hi_b, lo_b = dv_all >= mid_bad, dv_all < mid_bad
    dir_bad = z_all[:, hi_b].mean(axis=1) - z_all[:, lo_b].mean(axis=1)
    exp_bad = np.einsum("ld,lnd->ln", dir_bad, z_ev)
    assert not np.allclose(scores["arm2_ctx_native"], exp_bad, atol=1e-6)


# ---------------------------------------------------------------------------
# 2. arm2q dispatch semantics + parity-subset identity (shared thresholds)
# ---------------------------------------------------------------------------


def test_arm2q_quantile_split_matches_shared_thresholds():
    _u, z_tr, dv_tr, z_ev, dv_ev = _planted()
    scores, skipped = _transfer_scores(
        z_tr, dv_tr, z_ev, dv_ev, ["arm2_ctx_native", "arm2q_ctx_native"]
    )
    assert not skipped, f"unexpected skips: {skipped}"
    q_lo, q_hi = arms.arm2q_thresholds(dv_tr)
    hi, lo = dv_tr >= q_hi, dv_tr <= q_lo
    assert hi.any() and lo.any() and not (hi & lo).any()
    direction = z_tr[:, hi].mean(axis=1) - z_tr[:, lo].mean(axis=1)
    expected = np.einsum("ld,lnd->ln", direction, z_ev)
    np.testing.assert_allclose(scores["arm2q_ctx_native"], expected, rtol=1e-10, atol=1e-12)
    # a NEW slug BESIDE the unrepaired arm2, never a relabel: both present,
    # numerically distinct (quartile sides != midpoint sides on this dv).
    assert not np.allclose(scores["arm2q_ctx_native"], scores["arm2_ctx_native"])

    # matched-budget parity currency: the scorer-side subset equals the
    # arm-internal split rows EXACTLY (one shared threshold helper).
    rows = sc._quantile_fit_rows(np.arange(len(dv_tr)), dv_tr)
    assert set(rows.tolist()) == set(np.flatnonzero(hi | lo).tolist())


def test_arm2q_registry_and_map_consuming_set():
    assert "arm2q_ctx_native" in arms.ARM_REGISTRY
    assert arms.ARM_REGISTRY["arm2q_ctx_native"]["rb_dep"] is False
    assert arms.ARM_REGISTRY["arm2q_ctx_native"]["family"] == "context"
    # the --skip-map-fit guard's set: map-free arm2 family, all slugs real.
    assert not set(sc.A2_FAMILY) & arms.MAP_CONSUMING_ARMS
    assert set(arms.ARM_REGISTRY) > arms.MAP_CONSUMING_ARMS
    assert {"arm6_map_proj_e1", "arm7_map_ridge_pred", "arm20_shuffled_map_ridge"} <= (
        arms.MAP_CONSUMING_ARMS
    )


# ---------------------------------------------------------------------------
# 3. repair-ladder pass plan (pure)
# ---------------------------------------------------------------------------


def test_arm2fix_pass_plans_all_adapters():
    pool = [np.array([0, 1, 2, 3]), np.array([4, 5, 6])]
    wc = np.array([7, 8, 9])
    dv_z = np.linspace(0.0, 1.0, 10)

    p = sc._arm2fix_passes("v1", ("arm2_ctx_native",), pool, wc, dv_z)
    assert [x.label for x in p] == ["std"]
    assert list(p[0].readout) == list(range(10))
    assert p[0].arm_meta["arm2_ctx_native"]["adapter"] == "v1"

    # R-B: WildChat excluded from the arm2 fit; other roster arms keep the
    # standard shared-pool pass.
    p = sc._arm2fix_passes(
        "v2-component-restricted", ("arm2_ctx_native", "arm4_ridge_ctx"), pool, wc, dv_z
    )
    assert [x.label for x in p] == ["a2r", "std"]
    assert list(p[0].readout) == list(range(7)) and p[0].roster == ("arm2_ctx_native",)
    assert p[0].preds_tag == "a2r"
    assert p[0].arm_meta["arm2_ctx_native"]["adapter"] == "v2-component-restricted"
    assert p[1].roster == ("arm4_ridge_ctx",) and list(p[1].readout) == list(range(10))

    # R-C: one shared-pool pass; arm2q's fit rows are the shared-threshold
    # quantile subset; arm2 stays the unrepaired v1.
    roster_q = ("arm2_ctx_native", "arm2q_ctx_native")
    p = sc._arm2fix_passes("v2-quantile", roster_q, pool, wc, dv_z)
    assert [x.label for x in p] == ["std"] and p[0].roster == roster_q
    q_rows = p[0].arm_meta["arm2q_ctx_native"]["fit_rows"]
    assert set(q_rows.tolist()) == set(sc._quantile_fit_rows(np.arange(10), dv_z).tolist())
    assert p[0].arm_meta["arm2_ctx_native"]["adapter"] == "v1"

    # parity: row-matched arm7 refit on the REPAIRED arm's fit rows.
    p = sc._arm2fix_passes("v2-quantile", roster_q, pool, wc, dv_z, parity_refit_arm7=True)
    assert p[-1].label == "parity" and p[-1].roster == ("arm7_map_ridge_pred",)
    assert set(p[-1].readout.tolist()) == set(q_rows.tolist())
    assert p[-1].arm_meta["arm7_map_ridge_pred"]["adapter"] == "parity-row-matched"

    # R-B x R-C composition: arm2q eliciting-only; arm2 stays standard.
    p = sc._arm2fix_passes(
        "v2-quantile-restricted", roster_q, pool, wc, dv_z, parity_refit_arm7=True
    )
    assert [x.label for x in p] == ["a2qr", "std", "parity"]
    assert list(p[0].readout) == list(range(7))
    assert set(p[2].readout.tolist()) == set(p[0].arm_meta["arm2q_ctx_native"]["fit_rows"].tolist())

    with pytest.raises(ValueError, match="unknown arm2 adapter"):
        sc._arm2fix_passes("v99", ("arm2_ctx_native",), pool, wc, dv_z)
    with pytest.raises(ValueError, match="needs arm2_ctx_native"):
        sc._arm2fix_passes("v1", ("arm4_ridge_ctx",), pool, wc, dv_z)


def test_row_ids_sha256_is_order_independent():
    a = sc._row_ids_sha256(["c2", "c1", "c3"])
    b = sc._row_ids_sha256(["c1", "c3", "c2"])
    c = sc._row_ids_sha256(["c1", "c3"])
    assert a == b and a != c and len(a) == 64


# ---------------------------------------------------------------------------
# 4. parse-time guards (--skip-map-fit refusal etc.)
# ---------------------------------------------------------------------------

D1_ARGV = [
    "--protocols",
    "B",
    "--seeds",
    "0",
    "--extra-arms",
    "arm2_ctx_native",
    "--arms-only-extra",
    "--skip-map-fit",
    "--map-variants",
    "true",
    "--arm2-adapter",
    "v1",
]


def test_d1_invocation_shape_parses_and_restricts_roster(roster_guard):
    sc.parse_args(D1_ARGV)
    assert sc.ROSTER == ("arm2_ctx_native",)
    assert sc.LABEL_CONSUMING == ("arm2_ctx_native",)


def test_skip_map_fit_guard_refuses_map_consuming_roster(roster_guard):
    # default roster carries arm6/arm7 (map-consuming) -> loud refusal
    with pytest.raises(SystemExit, match="map-consuming"):
        sc.parse_args(["--protocols", "B", "--seeds", "0", "--skip-map-fit"])


def test_skip_map_fit_guard_refuses_shufpair_and_protocol_c(roster_guard):
    base = ["--seeds", "0", "--extra-arms", "arm2_ctx_native", "--arms-only-extra"]
    with pytest.raises(SystemExit, match="shufpair"):
        sc.parse_args(
            ["--protocols", "B", *base, "--skip-map-fit", "--map-variants", "true", "shufpair"]
        )
    with pytest.raises(SystemExit, match="protocol C"):
        sc.parse_args(["--protocols", "ABC", *base, "--skip-map-fit"])


def test_parity_guards(roster_guard):
    base = [
        "--protocols",
        "B",
        "--seeds",
        "0",
        "--extra-arms",
        "arm2_ctx_native",
        "--arms-only-extra",
    ]
    with pytest.raises(SystemExit, match="needs --arm2-adapter"):
        sc.parse_args([*base, "--parity-refit-arm7"])
    with pytest.raises(SystemExit, match="TRUE map"):
        sc.parse_args(
            [
                *base,
                "--arm2-adapter",
                "v2-component-restricted",
                "--parity-refit-arm7",
                "--skip-map-fit",
            ]
        )
    # parity WITHOUT --skip-map-fit is the sanctioned matched-budget shape
    sc.parse_args([*base, "--arm2-adapter", "v2-component-restricted", "--parity-refit-arm7"])
    assert sc.ROSTER == ("arm2_ctx_native",)


def test_quantile_adapter_appends_arm2q_beside_arm2(roster_guard):
    sc.parse_args(
        [
            "--protocols",
            "B",
            "--seeds",
            "0",
            "--extra-arms",
            "arm2_ctx_native",
            "--arms-only-extra",
            "--skip-map-fit",
            "--arm2-adapter",
            "v2-quantile",
        ]
    )
    assert sc.ROSTER == ("arm2_ctx_native", "arm2q_ctx_native")
    assert "arm2q_ctx_native" in sc.LABEL_CONSUMING


def test_arms_only_extra_needs_extra_arms(roster_guard):
    with pytest.raises(SystemExit, match="needs --extra-arms"):
        sc.parse_args(["--protocols", "B", "--seeds", "0", "--arms-only-extra"])


# ---------------------------------------------------------------------------
# 5. per-seed resume predicate: arm2fix keys
# ---------------------------------------------------------------------------


def test_seed_resume_predicate_keys_on_adapter(tmp_path):
    for name in ("map_diagnostics.json", "readout_pools.json"):
        (tmp_path / name).write_text("{}")
    meta = {
        "git_commit": "abc",
        "out_schema_version": sc.SEED_OUT_SCHEMA_VERSION,
        "seed": 0,
        "map_variants": ["true"],
    }
    (tmp_path / "all_arms_spearman.json").write_text(json.dumps({"meta": meta}))
    ok, _ = sc._seed_output_resume_ok(tmp_path, commit="abc", seed=0, map_variants=["true"])
    assert ok
    # a LEGACY output can never satisfy an arm2fix invocation
    ok, why = sc._seed_output_resume_ok(
        tmp_path, commit="abc", seed=0, map_variants=["true"], arm2_adapter="v1"
    )
    assert not ok and "arm2_adapter" in why

    meta2 = {**meta, "arm2_adapter": "v2-component-restricted", "skip_map_fit": True}
    (tmp_path / "all_arms_spearman.json").write_text(json.dumps({"meta": meta2}))
    ok, _ = sc._seed_output_resume_ok(
        tmp_path,
        commit="abc",
        seed=0,
        map_variants=["true"],
        arm2_adapter="v2-component-restricted",
        skip_map_fit=True,
    )
    assert ok
    ok, why = sc._seed_output_resume_ok(
        tmp_path,
        commit="abc",
        seed=0,
        map_variants=["true"],
        arm2_adapter="v2-quantile",
        skip_map_fit=True,
    )
    assert not ok and "arm2_adapter" in why


# ---------------------------------------------------------------------------
# 6. matched-regime sanity read (real run_cell path on a synthetic pool)
# ---------------------------------------------------------------------------


def test_matched_regime_sanity_rows_synthetic(roster_guard):
    rng = np.random.default_rng(11)
    ly, d = 2, 5
    n_a, n_b = 24, 16
    z_ctx = rng.normal(size=(ly, n_a + n_b, d))
    dv_raw = rng.uniform(0, 100, n_a + n_b)
    rb_w = rng.normal(size=(ly, d))
    datasets = [
        SimpleNamespace(
            name="dsA",
            rows=np.arange(n_a),
            groups=np.asarray([f"gA{i // 3}" for i in range(n_a)]),
        ),
        SimpleNamespace(
            name="dsB",
            rows=np.arange(n_a, n_a + n_b),
            groups=np.asarray([f"gB{i // 2}" for i in range(n_b)]),
        ),
    ]
    args = SimpleNamespace(
        seed=0,
        draw=0,
        train_frac=0.8,
        regime="e1",
        device="cpu",
        a2_sanity_folds=3,
        arm2_adapter="v2-component-restricted",
        map_variants=["true"],
    )
    sc.ROSTER = ("arm2_ctx_native",)
    rows = sc._matched_regime_sanity(
        args,
        "evil",
        list(range(ly)),
        datasets,
        z_ctx,
        dv_raw,
        rb_w,
        {"arm2_ctx_native": 1},
        "context_end",
        99,
    )
    assert len(rows) == 1
    r = rows[0]
    assert r["rung_kind"] == "sanity_matched_regime"
    assert r["arm"] == "arm2_ctx_native"
    assert r["adapter"] == "v2-component-restricted"
    assert r["n_folds"] >= 2
    assert np.isfinite(r["rho_frozen"])
    assert len(r["rho_per_layer"]) == ly
    assert r["map_variant"] == "true"
    # the sanity pool is the ELICITING train slices only (no WildChat block
    # exists in `datasets` by construction — the roster IS the eliciting set)
    assert r["n_rows"] <= n_a + n_b
