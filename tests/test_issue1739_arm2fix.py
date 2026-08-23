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

    # r1 CONCERN arm2fix-resume-key-completeness: EVERY output-affecting lane
    # key is in the resume identity — arms_only_extra, a2_sanity_folds (on the
    # adapter lane), and the transfer_preds sidecar existence companion.
    ok, why = sc._seed_output_resume_ok(
        tmp_path,
        commit="abc",
        seed=0,
        map_variants=["true"],
        arm2_adapter="v2-component-restricted",
        skip_map_fit=True,
        arms_only_extra=True,
    )
    assert not ok and "arms_only_extra" in why
    ok, why = sc._seed_output_resume_ok(
        tmp_path,
        commit="abc",
        seed=0,
        map_variants=["true"],
        arm2_adapter="v2-component-restricted",
        skip_map_fit=True,
        a2_sanity_folds=5,
    )
    assert not ok and "a2_sanity_folds" in why
    meta3 = {**meta2, "arms_only_extra": True, "a2_sanity_folds": 5}
    (tmp_path / "all_arms_spearman.json").write_text(json.dumps({"meta": meta3}))
    kw = dict(
        commit="abc",
        seed=0,
        map_variants=["true"],
        arm2_adapter="v2-component-restricted",
        skip_map_fit=True,
        arms_only_extra=True,
        a2_sanity_folds=5,
    )
    ok, _ = sc._seed_output_resume_ok(tmp_path, **kw)
    assert ok
    # preds-writing invocations require >=1 transfer_preds sidecar present
    # (pre-manifest fallback floor for outputs whose meta lacks the manifest)
    ok, why = sc._seed_output_resume_ok(tmp_path, transfer_preds=True, **kw)
    assert not ok and "transfer_preds" in why
    (tmp_path / "transfer_preds").mkdir()
    (tmp_path / "transfer_preds" / "P-B-holdout-x.jsonl").write_text("")
    ok, _ = sc._seed_output_resume_ok(tmp_path, transfer_preds=True, **kw)
    assert ok
    # MANIFEST verification (codex r2 minor): a meta-recorded sidecar manifest
    # is checked file-by-file — one present-but-partial sidecar set can no
    # longer read as resume-complete; an empty manifest is not a preds run.
    meta4 = {**meta3, "transfer_preds_files": ["P-B-holdout-x.jsonl", "P-B-holdout-y.jsonl"]}
    (tmp_path / "all_arms_spearman.json").write_text(json.dumps({"meta": meta4}))
    ok, why = sc._seed_output_resume_ok(tmp_path, transfer_preds=True, **kw)
    assert not ok and "manifest files absent" in why and "P-B-holdout-y.jsonl" in why
    (tmp_path / "transfer_preds" / "P-B-holdout-y.jsonl").write_text("")
    ok, _ = sc._seed_output_resume_ok(tmp_path, transfer_preds=True, **kw)
    assert ok
    meta5 = {**meta3, "transfer_preds_files": []}
    (tmp_path / "all_arms_spearman.json").write_text(json.dumps({"meta": meta5}))
    ok, why = sc._seed_output_resume_ok(tmp_path, transfer_preds=True, **kw)
    assert not ok and "manifest empty" in why
    # legacy flagless resume is unchanged (no adapter on either side)
    meta_legacy = {
        "git_commit": "abc",
        "out_schema_version": sc.SEED_OUT_SCHEMA_VERSION,
        "seed": 0,
        "map_variants": ["true"],
    }
    (tmp_path / "all_arms_spearman.json").write_text(json.dumps({"meta": meta_legacy}))
    ok, _ = sc._seed_output_resume_ok(tmp_path, commit="abc", seed=0, map_variants=["true"])
    assert ok


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


# ---------------------------------------------------------------------------
# fold arm2fix mode (plan §4 D2 + §3 amended lattice) — synthetic fixtures
# ---------------------------------------------------------------------------

import scripts.issue1739_claim4_fold as fold  # noqa: E402

REG2 = {
    "evil": frozenset({"evil_pair", "toxicchat"}),
    "sycophancy": frozenset({"sycomwe"}),
}


def _committed_root(tmp_path, bands):
    for b, (lo, hi) in bands.items():
        p = tmp_path / b / "arm_results"
        p.mkdir(parents=True, exist_ok=True)
        rows = [
            {
                "arm": fold.ARM_CTXDIR,
                "variant": "context_end",
                "regime": "e1",
                "u_rung_label": "full",
                "rho_frozen": v,
            }
            for v in (lo, hi)
        ]
        (p / "all_arms_spearman.json").write_text(json.dumps({"arm_rows": rows}))
    return tmp_path


def _sanity_row(b, seed, rho, arm="arm2_ctx_native", adapter="v1"):
    return {
        "behavior": b,
        "seed": seed,
        "arm": arm,
        "rho_frozen": rho,
        "rung_kind": "sanity_matched_regime",
        "protocol": "P-B",
        "fit": "sanity-elic-pool-cv",
        "eval_rung": "sanity_elic_pool_cv",
        "adapter": adapter,
    }


def _prow(b, rung, seed, arm, rho, mv="true", adapter=None, sha=None, n_rows=None):
    r = {
        "behavior": b,
        "eval_rung": rung,
        "seed": seed,
        "arm": arm,
        "rho_frozen": rho,
        "protocol": "P-B",
        "fit": f"P-B-holdout-{rung}",
        "map_variant": mv,
    }
    if adapter is not None:
        r["adapter"] = adapter
    if sha is not None:
        r["train_row_ids_sha256"] = sha
        r["train_rows_n"] = n_rows
    return r


def _res(arm="arm2_ctx_native", *, restricted=False, parity=False, adapter=None):
    if adapter is None:
        if not restricted:
            adapter = "v1"
        else:
            adapter = "v2-quantile" if arm == "arm2q_ctx_native" else "v2-component-restricted"
    return {
        "repaired_arm": arm,
        # ONE canonical registered adapter tag per behavior (codex r2
        # arm2fix-mixed-adapter-provenance)
        "adapter": adapter,
        "rows_restricted": restricted,
        "parity_present": parity,
        # HARD form (r2 blocker arm2fix-parity-partial-coverage): the §4
        # parity duty is mandatory wherever the repair restricted rows.
        "parity_required": bool(restricted),
    }


def test_a2fix_sanity_records_pass_fail_and_coverage_exits(tmp_path):
    """Sanity PASS/FAIL is reserved for the MEASURED band read; every
    incomplete/missing-band state is a LOUD infra coverage exit — never
    pass=False evidence against the adapter (r1 sustained blocker
    arm2fix-sanity-coverage-fail-open). Fails pre-fix: the pre-r2 code
    converted a 3-of-5-seed record into pass=False."""
    root = _committed_root(tmp_path, {"evil": (0.40, 0.70), "sycophancy": (0.30, 0.53)})
    res = {"evil": _res(), "sycophancy": _res()}
    rows = [_sanity_row("evil", s, 0.55) for s in range(5)]
    rows += [_sanity_row("sycophancy", s, 0.60) for s in range(5)]  # above band
    recs = fold.a2fix_sanity_records(rows, ["evil", "sycophancy"], range(5), res, root)
    assert recs["evil"]["pass"] and recs["evil"]["miss_side"] is None
    assert recs["evil"]["n_seeds"] == 5 and len(recs["evil"]["per_seed"]) == 5
    assert not recs["sycophancy"]["pass"] and recs["sycophancy"]["miss_side"] == "above"
    # incomplete record (3 of 5 seeds) is an INFRA coverage exit, never a verdict
    rows3 = [_sanity_row("evil", s, 0.55) for s in range(3)]
    with pytest.raises(SystemExit, match=r"COVERAGE ERROR.*incomplete"):
        fold.a2fix_sanity_records(rows3, ["evil"], range(5), {"evil": _res()}, root)
    # missing committed band is an INFRA coverage exit too
    with pytest.raises(SystemExit, match=r"COVERAGE ERROR.*band"):
        fold.a2fix_sanity_records(
            [_sanity_row("hallucination", s, 0.5) for s in range(5)],
            ["hallucination"],
            range(5),
            {"hallucination": _res()},
            root,
        )
    # duplicate sanity row fails loud
    with pytest.raises(SystemExit, match="duplicate sanity row"):
        fold.a2fix_sanity_records(
            [*rows, _sanity_row("evil", 0, 0.5)], ["evil"], range(5), {"evil": _res()}, root
        )
    # per-behavior arm filter: the mixed R-C shape (evil repaired arm2q, syco
    # arm2) reads each behavior's OWN sanity rows — complete on both sides
    mixed_rows = [
        _sanity_row("evil", s, 0.55, arm="arm2q_ctx_native", adapter="v2-quantile")
        for s in range(5)
    ]
    mixed_rows += [_sanity_row("evil", s, 0.10) for s in range(5)]  # unrepaired arm2 beside
    mixed_rows += [_sanity_row("sycophancy", s, 0.40) for s in range(5)]
    res_mixed = {"evil": _res("arm2q_ctx_native", restricted=True), "sycophancy": _res()}
    recs_m = fold.a2fix_sanity_records(
        mixed_rows, ["evil", "sycophancy"], range(5), res_mixed, root
    )
    assert recs_m["evil"]["arm"] == "arm2q_ctx_native" and recs_m["evil"]["pass"]
    assert recs_m["sycophancy"]["arm"] == "arm2_ctx_native" and recs_m["sycophancy"]["pass"]


def test_a2fix_sanity_provenance_and_nonfinite_are_loud(tmp_path):
    """codex r2 blockers: (a) a sanity row whose adapter tag differs from the
    behavior's resolved canonical tag is foreign provenance — loud exit, never
    mask input; (b) a NaN sanity value or committed-band value is a NON-FINITE
    validity exit BEFORE band containment — never a measured band miss
    (pre-fix: NaN mean failed containment as pass=False -> false
    INDETERMINATE-ADAPTER)."""
    root = _committed_root(tmp_path, {"evil": (0.40, 0.70)})
    ok_rows = [_sanity_row("evil", s, 0.55) for s in range(5)]
    # (a) adapter-tag mismatch on a consumed sanity row
    bad_tag = list(ok_rows)
    bad_tag[2] = _sanity_row("evil", 2, 0.55, adapter="v2-component-restricted")
    with pytest.raises(SystemExit, match=r"PROVENANCE ERROR.*sanity row"):
        fold.a2fix_sanity_records(bad_tag, ["evil"], range(5), {"evil": _res()}, root)
    # (b1) NaN sanity value -> validity exit, not a band verdict
    bad_val = list(ok_rows)
    bad_val[1] = _sanity_row("evil", 1, float("nan"))
    with pytest.raises(SystemExit, match=r"NON-FINITE STATISTIC.*sanity row"):
        fold.a2fix_sanity_records(bad_val, ["evil"], range(5), {"evil": _res()}, root)
    # (b2) NaN committed-band cell -> validity exit, not a containment input
    root_nan = _committed_root(tmp_path / "nanband", {"evil": (0.40, float("nan"))})
    with pytest.raises(SystemExit, match=r"NON-FINITE STATISTIC.*committed"):
        fold.a2fix_sanity_records(ok_rows, ["evil"], range(5), {"evil": _res()}, root_nan)


def test_a2fix_resolver_rejects_mixed_or_foreign_adapter_tags():
    """codex r2 blocker arm2fix-mixed-adapter-provenance regression: seed 0
    tagged v1 + seed 1 tagged v2-component-restricted within ONE behavior is
    a loud provenance exit — never collapsed into one rows_restricted bool;
    a missing/unregistered tag is equally loud. Fails pre-fix (the set was
    silently reduced to configuration)."""
    mixed = [
        _prow("evil", "evil_pair", 0, "arm2_ctx_native", 0.3, adapter="v1"),
        _prow("evil", "evil_pair", 1, "arm2_ctx_native", 0.3, adapter="v2-component-restricted"),
    ]
    with pytest.raises(SystemExit, match=r"PROVENANCE ERROR.*distinct adapter tags"):
        fold.a2fix_resolve_repairs(mixed, ["evil"])
    # missing tag (adapter key absent -> "None") is unregistered provenance
    missing = [_prow("evil", "evil_pair", 0, "arm2_ctx_native", 0.3)]
    with pytest.raises(SystemExit, match=r"PROVENANCE ERROR.*unregistered adapter tag"):
        fold.a2fix_resolve_repairs(missing, ["evil"])
    # a foreign tag on the repaired arm is unregistered too
    foreign = [_prow("evil", "evil_pair", 0, "arm2q_ctx_native", 0.3, adapter="v1")]
    with pytest.raises(SystemExit, match=r"PROVENANCE ERROR.*unregistered adapter tag"):
        fold.a2fix_resolve_repairs(foreign, ["evil"])
    # the canonical single-tag form resolves and records the tag
    ok = [_prow("evil", "evil_pair", s, "arm2_ctx_native", 0.3, adapter="v1") for s in range(2)]
    res = fold.a2fix_resolve_repairs(ok, ["evil"])
    assert res["evil"]["adapter"] == "v1" and not res["evil"]["rows_restricted"]


def test_a2fix_resolve_repairs_mixed_topology_and_overrides():
    """PER-BEHAVIOR resolution (r1 sustained blocker arm2fix-mixed-adapter-fold):
    a behavior that ran the quantile repair resolves to arm2q while its
    siblings stay on arm2 — one fold run, mixed topology. Fails pre-fix: the
    pre-r2 fold resolved ONE global repaired_arm."""
    new = [
        _prow("evil", "evil_pair", 0, "arm2q_ctx_native", 0.5, adapter="v2-quantile"),
        _prow("evil", "evil_pair", 0, "arm2_ctx_native", 0.1, adapter="v1"),
        _prow("sycophancy", "sycomwe", 0, "arm2_ctx_native", 0.4, adapter="v1"),
        _prow(
            "evil",
            "evil_pair",
            0,
            fold.ARM_MAP,
            0.6,
            adapter="parity-row-matched",
            sha="h",
            n_rows=9,
        ),
    ]
    res = fold.a2fix_resolve_repairs(new, ["evil", "sycophancy"])
    assert res["evil"]["repaired_arm"] == "arm2q_ctx_native"
    assert res["evil"]["rows_restricted"] and res["evil"]["parity_present"]
    assert res["evil"]["parity_required"]
    assert res["sycophancy"]["repaired_arm"] == "arm2_ctx_native"
    assert not res["sycophancy"]["rows_restricted"] and not res["sycophancy"]["parity_required"]
    # HARD form: a restricted behavior with NO emitted parity rows still
    # REQUIRES them (the join fails loud downstream) — never a quiet degrade
    new_noparity = [r for r in new if r.get("adapter") != "parity-row-matched"]
    res_np = fold.a2fix_resolve_repairs(new_noparity, ["evil"])
    assert res_np["evil"]["rows_restricted"] and not res_np["evil"]["parity_present"]
    assert res_np["evil"]["parity_required"] is True
    # per-behavior override honored; an override naming an absent arm exits loud
    res_ov = fold.a2fix_resolve_repairs(new, ["evil"], {"evil": "arm2_ctx_native"})
    assert res_ov["evil"]["repaired_arm"] == "arm2_ctx_native"
    assert not res_ov["evil"]["rows_restricted"]  # v1 rows only under the override
    with pytest.raises(SystemExit, match="no primary"):
        fold.a2fix_resolve_repairs(new, ["sycophancy"], {"sycophancy": "arm2q_ctx_native"})
    # a behavior with NO arm2-family primary rows is a loud coverage error
    with pytest.raises(SystemExit, match="COVERAGE ERROR"):
        fold.a2fix_resolve_repairs(new, ["hallucination"])


def test_parse_repaired_arm_tokens():
    behaviors = ["evil", "sycophancy"]
    assert fold.parse_repaired_arm_tokens(["auto"], behaviors) == {}
    assert fold.parse_repaired_arm_tokens(["arm2q_ctx_native"], behaviors) == {
        "evil": "arm2q_ctx_native",
        "sycophancy": "arm2q_ctx_native",
    }
    assert fold.parse_repaired_arm_tokens(["evil=arm2q_ctx_native"], behaviors) == {
        "evil": "arm2q_ctx_native"
    }
    with pytest.raises(SystemExit, match="mixes"):
        fold.parse_repaired_arm_tokens(["arm2_ctx_native", "evil=arm2q_ctx_native"], behaviors)
    with pytest.raises(SystemExit, match="unknown behavior"):
        fold.parse_repaired_arm_tokens(["nope=arm2_ctx_native"], behaviors)
    with pytest.raises(SystemExit, match="unknown arm"):
        fold.parse_repaired_arm_tokens(["evil=arm9_zzz"], behaviors)
    with pytest.raises(SystemExit, match="duplicates"):
        fold.parse_repaired_arm_tokens(["evil=arm2_ctx_native", "evil=arm2q_ctx_native"], behaviors)
    with pytest.raises(SystemExit, match="exactly one"):
        fold.parse_repaired_arm_tokens(["arm2_ctx_native", "arm2q_ctx_native"], behaviors)


def _series_rows(behaviors_rungs, seeds, d=0.05, base=0.30):
    """new_rows + banked_rows with arm7 = arm2 + d on every join key."""
    new, banked = [], []
    for b, rung in behaviors_rungs:
        for s in seeds:
            a2 = base + 0.01 * s
            new.append(_prow(b, rung, s, "arm2_ctx_native", a2, adapter="v1"))
            banked.append(_prow(b, rung, s, fold.ARM_MAP, a2 + d, mv="true"))
            banked.append(_prow(b, rung, s, fold.ARM_MAP, a2 - 0.10, mv="shufpair"))
            banked.append(_prow(b, rung, s, fold.ARM_CTX, a2 + 0.02, mv="true"))
    return new, banked


def test_a2fix_join_restated_denominator_and_gap_exit():
    seeds = [0, 1]
    pairs = [("evil", "evil_pair"), ("evil", "toxicchat"), ("sycophancy", "sycomwe")]
    new, banked = _series_rows(pairs, seeds)
    res = {"evil": _res(), "sycophancy": _res()}
    cells = fold.a2fix_index_cells(new, banked, res)
    # passing set excludes sycophancy -> denominator RESTATED from 6 to 4
    join = fold.a2fix_join_assert(cells, ["evil"], seeds, REG2, resolution=res)
    assert join["expected_pairs"] == 4 and join["restated_from"] == 6
    assert join["realized_pairs"] == 4 and "4/4" in join["assert"]
    # a missing banked series cell on the passing set is a loud exit
    banked_gap = [
        r
        for r in banked
        if not (r["eval_rung"] == "evil_pair" and r["seed"] == 1 and r["arm"] == fold.ARM_MAP)
    ]
    cells_gap = fold.a2fix_index_cells(new, banked_gap, res)
    with pytest.raises(SystemExit, match="join INCOMPLETE"):
        fold.a2fix_join_assert(cells_gap, ["evil"], seeds, REG2, resolution=res)
    # duplicate join key fails loud at indexing time
    with pytest.raises(SystemExit, match="duplicate join key"):
        fold.a2fix_index_cells([*new, new[0]], banked, res)


def test_a2fix_join_mixed_parity_demand_is_per_behavior():
    """The R-B/R-C mixed regression (r1 sustained blocker): parity series are
    demanded ONLY for the behavior whose repair restricted rows — the pre-r2
    global parity_required=True SystemExited on sycophancy's plan-conforming
    ABSENT parity rows (the plan forbids refitting unrestricted behaviors)."""
    seeds = [0, 1]
    pairs = [("evil", "evil_pair"), ("evil", "toxicchat"), ("sycophancy", "sycomwe")]
    new, banked = _series_rows(pairs, seeds)
    # evil ran restricted + parity: add parity rows for evil ONLY
    for rung in ("evil_pair", "toxicchat"):
        for s in seeds:
            new.append(
                _prow(
                    "evil",
                    rung,
                    s,
                    fold.ARM_MAP,
                    0.4,
                    adapter="parity-row-matched",
                    sha="h" * 8,
                    n_rows=50,
                )
            )
    res = {
        "evil": _res(restricted=True, parity=True),
        "sycophancy": _res(),
    }
    cells = fold.a2fix_index_cells(new, banked, res)
    join = fold.a2fix_join_assert(cells, ["evil", "sycophancy"], seeds, REG2, resolution=res)
    assert join["expected_pairs"] == 6
    assert join["series_required_by_behavior"]["evil"][-1] == "arm7_parity"
    assert "arm7_parity" not in join["series_required_by_behavior"]["sycophancy"]
    # the OLD global demand (parity required for every behavior) would gap out
    res_global = {
        "evil": _res(restricted=True, parity=True),
        "sycophancy": _res(restricted=True, parity=True),
    }
    with pytest.raises(SystemExit, match="join INCOMPLETE"):
        fold.a2fix_join_assert(cells, ["evil", "sycophancy"], seeds, REG2, resolution=res_global)


def test_a2fix_parity_partial_coverage_is_loud_never_map_beats():
    """r2 BLOCKER arm2fix-parity-partial-coverage regression: with >=2
    rows-restricted passing behaviors and ONE missing ALL its parity rows,
    (a) the HARD join demand fails LOUD (parity_required = rows_restricted),
    and (b) even on a direct lattice call the parity coverage condition
    refuses MAP-BEATS — never a silent n_rungs undercount. Fails pre-fix:
    the r2 code minted MAP-BEATS with parity_read.n_rungs=1 of 2."""
    seeds = [0, 1]
    pairs = [("evil", "evil_pair"), ("sycophancy", "sycomwe")]
    new, banked = _series_rows(pairs, seeds)
    # BOTH behaviors restricted; parity rows emitted for evil ONLY
    for s in seeds:
        new.append(
            _prow(
                "evil",
                "evil_pair",
                s,
                fold.ARM_MAP,
                0.4,
                adapter="parity-row-matched",
                sha="h" * 8,
                n_rows=50,
            )
        )
    res = {
        "evil": _res("arm2q_ctx_native", restricted=True, parity=True),
        "sycophancy": _res("arm2_ctx_native", restricted=True, parity=False),
    }
    reg = {"evil": frozenset({"evil_pair"}), "sycophancy": frozenset({"sycomwe"})}
    cells = fold.a2fix_index_cells(new, banked, res)
    with pytest.raises(SystemExit, match="join INCOMPLETE"):
        fold.a2fix_join_assert(cells, ["evil", "sycophancy"], seeds, reg, resolution=res)
    # defense-in-depth: a DIRECT lattice call on the same topology (both CIs
    # clear, positive parity median on the emitted half) must NOT mint
    # MAP-BEATS — coverage_complete is False and the reason names the gap
    sanity = {"evil": {"pass": True}, "sycophancy": {"pass": True}}
    rows = [
        _entry("evil", "evil_pair", 0.06, [0.02, 0.10], [0.01, 0.11], d_parity=0.05),
        _entry("sycophancy", "sycomwe", 0.04, [0.01, 0.07], [0.005, 0.08]),  # no parity
    ]
    v = fold.a2fix_lattice(rows, sanity, resolution=res, registered=reg)
    assert v["verdict"] == "WEAK-MIXED"
    assert v["parity_read"]["coverage_complete"] is False
    assert v["parity_read"]["n_rungs"] == 1 and v["parity_read"]["n_rungs_expected"] == 2
    assert v["parity_read"]["uncovered_rungs"] == [["sycophancy", "sycomwe"]]
    assert v["parity_read"]["positive"] is False
    assert "INCOMPLETE" in v["reason"]
    # complete coverage on the same topology DOES mint MAP-BEATS
    rows_full = [
        _entry("evil", "evil_pair", 0.06, [0.02, 0.10], [0.01, 0.11], d_parity=0.05),
        _entry("sycophancy", "sycomwe", 0.04, [0.01, 0.07], [0.005, 0.08], d_parity=0.03),
    ]
    v2 = fold.a2fix_lattice(rows_full, sanity, resolution=res, registered=reg)
    assert v2["verdict"] == "MAP-BEATS-CONTEXT-DIRECTION"
    assert v2["parity_read"]["coverage_complete"] is True


def test_a2fix_parity_registered_universe_covers_omitted_rungs():
    """codex r3 BLOCKER arm2fix-parity-universe-undercoverage regression: the
    coverage universe is every restricted passing behavior's REGISTERED rungs
    — a rung record ENTIRELY absent from per_rung (or present without a D
    read) is UNCOVERED, never silently outside its own denominator. Fails
    pre-fix: positive parity on the surviving strict subset + a clear
    flagship minted MAP-BEATS with n_rungs_expected derived from realized
    rows."""
    sanity = {"evil": {"pass": True}, "sycophancy": {"pass": True}}
    res = {
        "evil": _res("arm2q_ctx_native", restricted=True, parity=True),
        "sycophancy": _res("arm2_ctx_native", restricted=True, parity=True),
    }
    reg = {"evil": frozenset({"evil_pair"}), "sycophancy": frozenset({"sycomwe"})}
    # sycophancy's registered rung record is ENTIRELY ABSENT from per_rung;
    # evil's flagship is both-CIs-clear with positive parity
    rows_omitted = [
        _entry("evil", "evil_pair", 0.06, [0.02, 0.10], [0.01, 0.11], d_parity=0.05),
    ]
    v = fold.a2fix_lattice(rows_omitted, sanity, resolution=res, registered=reg)
    assert v["verdict"] != "MAP-BEATS-CONTEXT-DIRECTION"
    assert v["parity_read"]["coverage_complete"] is False
    assert v["parity_read"]["n_rungs_expected"] == 2  # from the REGISTERED universe
    assert ["sycophancy", "sycomwe"] in v["parity_read"]["uncovered_rungs"]
    assert v["parity_read"]["positive"] is False
    # present-but-D-less variant (the row survives per_rung but carries no D
    # read, so the :1336 filter drops it) is the SAME undercoverage — caught
    e_nod = {
        "behavior": "sycophancy",
        "eval_rung": "sycomwe",
        "flagship": True,
        "excluded_by_sanity": False,
        "seeds_used": [],
        "complete": False,
        "D": {"mean": None, "tci": None},
        "D_ctx_ci": None,
    }
    v2 = fold.a2fix_lattice([*rows_omitted, e_nod], sanity, resolution=res, registered=reg)
    assert v2["verdict"] != "MAP-BEATS-CONTEXT-DIRECTION"
    assert v2["parity_read"]["coverage_complete"] is False
    assert ["sycophancy", "sycomwe"] in v2["parity_read"]["uncovered_rungs"]


def test_a2fix_per_rung_d_read_and_parity_hash():
    seeds = [0, 1, 2]
    pairs = [("evil", "evil_pair")]
    new, banked = _series_rows(pairs, seeds, d=0.05)
    # parity rows: identical hashes -> assert passes and D_parity computed
    for s in seeds:
        new.append(
            _prow(
                "evil",
                "evil_pair",
                s,
                fold.ARM_MAP,
                0.36,
                adapter="parity-row-matched",
                sha="h" * 8,
                n_rows=100,
            )
        )
    for r in new:
        if r["arm"] == "arm2_ctx_native":
            r["train_row_ids_sha256"], r["train_rows_n"] = "h" * 8, 100
    res = {"evil": _res(restricted=True, parity=True)}
    cells = fold.a2fix_index_cells(new, banked, res)
    n = fold.a2fix_parity_hash_assert(cells, ["evil"], seeds, REG2, res)
    assert n == 3
    sanity = {"evil": {"pass": True}}
    table = fold.a2fix_per_rung(cells, ["evil"], seeds, sanity)
    assert len(table) == 1
    e = table[0]
    assert e["complete"] and e["flagship"] and not e["excluded_by_sanity"]
    assert abs(e["D"]["mean"] - 0.05) < 1e-12  # arm7_true - arm2 per seed
    assert e["D_parity"]["mean"] is not None
    assert e["adapter"] == ["v1"]
    # a mismatched parity hash is a loud exit
    bad = dict(cells)
    key = ("evil", "evil_pair", 0, "arm7_parity")
    bad[key] = {**bad[key], "train_row_ids_sha256": "different"}
    with pytest.raises(SystemExit, match="PARITY VIOLATION"):
        fold.a2fix_parity_hash_assert(bad, ["evil"], seeds, REG2, res)
    # a None-for-None hash match is NOT parity evidence (r1 CONCERN
    # arm2fix-parity-currency-fail-open) — missing currency exits loud
    bad2 = dict(cells)
    bad2[key] = {k: v for k, v in bad2[key].items() if k != "train_row_ids_sha256"}
    k2 = ("evil", "evil_pair", 0, "arm2_new")
    bad2[k2] = {k: v for k, v in bad2[k2].items() if k != "train_row_ids_sha256"}
    with pytest.raises(SystemExit, match="PARITY CURRENCY MISSING"):
        fold.a2fix_parity_hash_assert(bad2, ["evil"], seeds, REG2, res)


def _entry(b, rung, d_mean, tci, ctx_ci=None, d_parity=None, excl=False):
    e = {
        "behavior": b,
        "eval_rung": rung,
        "flagship": (b, rung) in set(fold.FLAGSHIPS),
        "excluded_by_sanity": excl,
        "seeds_used": [0, 1, 2, 3, 4],
        "complete": True,
        "D": {"mean": d_mean, "tci": tci},
        "D_ctx_ci": ctx_ci,
    }
    if d_parity is not None:
        e["D_parity"] = {"mean": d_parity}
    return e


def test_a2fix_lattice_inconclusive_adapter_first():
    sanity = {
        "evil": {"pass": False},
        "sycophancy": {"pass": False},
        "hallucination": {"pass": True},
    }
    res = {"evil": _res(), "sycophancy": _res(), "hallucination": _res()}
    v = fold.a2fix_lattice(
        [_entry("hallucination", "nqopen", 0.1, [0.05, 0.15], [0.02, 0.2])],
        sanity,
        resolution=res,
    )
    assert v["verdict"] == "INCONCLUSIVE-ADAPTER"
    assert v["per_behavior_adapter_verdicts"] == {
        "evil": "INDETERMINATE-ADAPTER",
        "sycophancy": "INDETERMINATE-ADAPTER",
    }


def test_a2fix_lattice_map_advantage_not_shown_and_exclusion():
    """Median D <= 0 over the passing set -> MAP-ADVANTAGE-NOT-SHOWN with the
    mandated failure-to-demonstrate framing; the sanity-excluded behavior's
    rungs stay OUT of the median."""
    sanity = {
        "evil": {"pass": True},
        "sycophancy": {"pass": True},
        "hallucination": {"pass": False},
    }
    rows = [
        _entry("evil", "evil_pair", -0.02, [-0.05, 0.01], [-0.04, 0.0]),
        _entry("sycophancy", "sycomwe", -0.01, [-0.03, 0.01], [-0.03, 0.01]),
        # excluded behavior carries a large positive D that must NOT rescue the median
        _entry("hallucination", "nqopen", 0.5, [0.4, 0.6], [0.4, 0.6], excl=True),
    ]
    res = {"evil": _res(), "sycophancy": _res(), "hallucination": _res()}
    v = fold.a2fix_lattice(rows, sanity, resolution=res)
    assert v["verdict"] == "MAP-ADVANTAGE-NOT-SHOWN"
    assert "failure to demonstrate" in v["reason"]
    assert v["n_rungs_in_median"] == 2 and v["median_D_passing_set"] < 0
    assert set(v["per_behavior_median_D"]) == {"evil", "sycophancy"}


def test_a2fix_lattice_map_beats_requires_parity_when_restricted():
    sanity = {"evil": {"pass": True}, "sycophancy": {"pass": True}}
    rows = [
        _entry("evil", "evil_pair", 0.06, [0.02, 0.10], [0.01, 0.11], d_parity=0.05),
        _entry("sycophancy", "sycomwe", 0.04, [0.01, 0.07], [0.005, 0.08], d_parity=0.03),
    ]
    res_all = {
        "evil": _res(restricted=True, parity=True),
        "sycophancy": _res(restricted=True, parity=True),
    }
    reg = {"evil": frozenset({"evil_pair"}), "sycophancy": frozenset({"sycomwe"})}
    # restricted + positive parity read -> MAP-BEATS
    v = fold.a2fix_lattice(rows, sanity, resolution=res_all, registered=reg)
    assert v["verdict"] == "MAP-BEATS-CONTEXT-DIRECTION"
    assert v["parity_read"]["positive"] is True
    # restricted + NEGATIVE parity read -> WEAK-MIXED naming the parity duty
    rows_neg = [
        _entry("evil", "evil_pair", 0.06, [0.02, 0.10], [0.01, 0.11], d_parity=-0.02),
        _entry("sycophancy", "sycomwe", 0.04, [0.01, 0.07], [0.005, 0.08], d_parity=-0.01),
    ]
    v2 = fold.a2fix_lattice(rows_neg, sanity, resolution=res_all, registered=reg)
    assert v2["verdict"] == "WEAK-MIXED" and "parity" in v2["reason"]
    # unrestricted (v1 repair was a no-op) needs no parity read
    res_v1 = {"evil": _res(), "sycophancy": _res()}
    v3 = fold.a2fix_lattice(rows, sanity, resolution=res_v1, registered=reg)
    assert v3["verdict"] == "MAP-BEATS-CONTEXT-DIRECTION"
    assert "parity_read" not in v3
    # positive median but NO flagship with both CIs clear -> WEAK-MIXED
    rows_noflag = [
        _entry("evil", "evil_pair", 0.06, [-0.01, 0.13], [0.01, 0.11]),
        _entry("sycophancy", "sycomwe", 0.04, [0.01, 0.07], [-0.005, 0.08]),
    ]
    v4 = fold.a2fix_lattice(rows_noflag, sanity, resolution=res_v1, registered=reg)
    assert v4["verdict"] == "WEAK-MIXED" and "flagship" in v4["reason"]
    # the REGISTERED universe binds under the default too: with the real
    # REGISTERED_PRIMARY_RUNGS, evil_pair+sycomwe alone cover 2 of 11
    # registered restricted rungs -> coverage incomplete -> not MAP-BEATS
    v5 = fold.a2fix_lattice(rows, sanity, resolution=res_all)
    assert v5["verdict"] == "WEAK-MIXED"
    assert v5["parity_read"]["coverage_complete"] is False
    assert v5["parity_read"]["n_rungs_expected"] == 11


def test_a2fix_lattice_mixed_parity_scoped_to_restricted_behaviors():
    """Mixed topology (r1 sustained blocker): the parity read is computed over
    exactly the ROWS-RESTRICTED passing behaviors' rungs — sycophancy's
    (correctly) absent D_parity neither blocks MAP-BEATS nor dilutes the
    parity median."""
    sanity = {"evil": {"pass": True}, "sycophancy": {"pass": True}}
    res = {"evil": _res("arm2q_ctx_native", restricted=True, parity=True), "sycophancy": _res()}
    reg = {"evil": frozenset({"evil_pair"}), "sycophancy": frozenset({"sycomwe"})}
    rows = [
        _entry("evil", "evil_pair", 0.06, [0.02, 0.10], [0.01, 0.11], d_parity=0.05),
        _entry("sycophancy", "sycomwe", 0.04, [0.01, 0.07], [0.005, 0.08]),  # no parity
    ]
    v = fold.a2fix_lattice(rows, sanity, resolution=res, registered=reg)
    assert v["verdict"] == "MAP-BEATS-CONTEXT-DIRECTION"
    assert v["rows_restricted_behaviors"] == ["evil"]
    assert v["parity_read"]["behaviors"] == ["evil"] and v["parity_read"]["n_rungs"] == 1
    # evil's parity read negative -> WEAK-MIXED even though syco carries no parity
    rows_neg = [
        _entry("evil", "evil_pair", 0.06, [0.02, 0.10], [0.01, 0.11], d_parity=-0.02),
        _entry("sycophancy", "sycomwe", 0.04, [0.01, 0.07], [0.005, 0.08]),
    ]
    v2 = fold.a2fix_lattice(rows_neg, sanity, resolution=res, registered=reg)
    assert v2["verdict"] == "WEAK-MIXED" and "parity" in v2["reason"]


def test_a2fix_finite_and_ctx_gap_asserts():
    """Hardening gates (r1 CONCERNs arm2fix-nonfinite-verdict +
    arm2fix-context-bootstrap-gap): non-finite statistics and passing-set
    ctx-CI gaps are loud exits BEFORE the lattice, never notes under a
    rendered verdict."""
    ok_rows = [_entry("evil", "evil_pair", 0.06, [0.02, 0.10], [0.01, 0.11])]
    assert fold.a2fix_assert_finite(ok_rows) > 0
    bad = [_entry("evil", "evil_pair", float("nan"), [0.02, 0.10], [0.01, 0.11])]
    with pytest.raises(SystemExit, match="NON-FINITE"):
        fold.a2fix_assert_finite(bad)
    bad_ctx = [_entry("evil", "evil_pair", 0.06, [0.02, 0.10], [float("inf"), 0.11])]
    with pytest.raises(SystemExit, match="NON-FINITE"):
        fold.a2fix_assert_finite(bad_ctx)
    # excluded rows are skipped (they never reach the median)
    assert fold.a2fix_assert_finite([_entry("e", "r", float("nan"), None, excl=True)]) == 0
    # ctx-CI gap on a complete passing-set rung is a loud exit ...
    gap = [_entry("evil", "evil_pair", 0.06, [0.02, 0.10], None)]
    gap[0]["ctx_bootstrap_note"] = "missing preds file: X"
    with pytest.raises(SystemExit, match="CTX-BOOTSTRAP GAP"):
        fold.a2fix_ctx_gap_assert(gap, skipped=False)
    # ... unless the bootstrap was deliberately skipped or the row is excluded
    fold.a2fix_ctx_gap_assert(gap, skipped=True)
    fold.a2fix_ctx_gap_assert([_entry("e", "r", 0.1, None, None, excl=True)], skipped=False)


def test_p4_directions_pure_core_recovers_plant():
    """The four P4 direction variants + the folded train-mode direction all
    recover a planted dv-aligned axis (cos > 0.99) on a synthetic table."""
    from scripts.issue1739_arm2fix_d0 import _cos, _p4_directions

    rng = np.random.default_rng(1739)
    n, dch = 240, 8
    u = np.zeros(dch)
    u[0] = 1.0
    dv_raw = np.concatenate([rng.uniform(0.5, 3.0, 200), rng.uniform(0.5, 3.0, 40)])
    z1 = np.outer(dv_raw - dv_raw.mean(), u) + 0.01 * rng.standard_normal((n, dch))
    d_a = SimpleNamespace(
        name="poolA", rows=np.arange(0, 100), groups=[f"gA{i // 2}" for i in range(100)]
    )
    d_b = SimpleNamespace(
        name="poolB", rows=np.arange(100, 200), groups=[f"gB{i // 2}" for i in range(100)]
    )
    wc_rows = np.arange(200, 240)
    elic = SimpleNamespace(row_idx=np.arange(0, 100), fold_ids=np.arange(100) % 5, n_folds=5)
    out = _p4_directions(z1, dv_raw, [d_a, d_b], wc_rows, elic, seed=0, train_frac=0.8)
    assert set(out["dirs"]) == {"v1", "restricted", "quantile", "quantile-restricted"}
    for name, vec in out["dirs"].items():
        assert _cos(vec, u) > 0.99, name
    assert _cos(out["folded_dir"], u) > 0.99
    assert all(c > 0.9 for c in out["folded_per_fold_cos"])
    # restricted variants exclude the wc block from the fit budget
    assert out["counts"]["restricted"]["n_fit_rows"] < out["counts"]["v1"]["n_fit_rows"]
    q = out["counts"]["quantile"]
    assert q["n_hi"] + q["n_lo"] < q["n_fit_rows"]  # quantile split consumes a strict subset
