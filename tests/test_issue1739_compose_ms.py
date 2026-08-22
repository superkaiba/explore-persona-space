"""Compose-multiseed (cms) leg-1 unit tests (#1739 plan v26 §4 leg 1).

Covers: the replicate spec enumeration against the plan-§4 cell table EXACTLY
(13 specs per variant per seed; the designed-skip feasibility mirror yields
11 realized evil cells / 13 syco / 13 hall), one-spec-per-(draw, seed)
emission, the generating-parameter map-fit cache key identity (f_u=0 shares
across L anchors; f_u=0.5/f_l=0 does not), the whitening-rng spec-seed
threading (spec seed 0 reproduces the banked first-CLI-seed path), the
_map_key draws regression, and the fold's pure verdict/audit functions on
synthetic fixtures. No network, no GPU; all fixtures neutral synthetic.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_script(name: str):
    spec = importlib.util.spec_from_file_location(name, REPO_ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


fits_cli = _load_script("issue1739_fits")
fold = _load_script("issue1739_compose_ms_fold")

SEEDS = (0, 1, 2, 3, 4)
CORE_GRID = dict(f_u_grid=(0.0, 0.5), f_l_grid=(0.0, 1.0))
ABLATION_GRID = dict(f_u_grid=(0.1, 0.25, 0.75, 1.0), f_l_grid=(0.0,))
N_CTX = {"evil": 6_468, "sycophancy": 16_000, "hallucination": 16_000}
U_SIZE = 5_000


def _cms_specs(*, budgets, seeds=SEEDS, **grid):
    return fits_cli.compose_run_specs(
        variants=("context_end", "prefix_end"),
        regimes=("e1",),
        u_sizes=(None,),
        budgets=budgets,
        draws=(0,),
        seeds=seeds,
        compose=True,
        compose_only=True,
        compose_replicates=True,
        compose_u_size=U_SIZE,
        **grid,
    )


def _feasible(spec, n_ctx: int) -> bool:
    """Designed-skip arithmetic mirror (plan §4): at f_l=0 the eliciting pool
    is the anchor cell's complement (cell caps at n_ctx); quota = round(f_u x
    5000) must fit. f_l>=1 uses the whole table; f_u=0 needs no eliciting."""
    if spec.f_u == 0.0:
        return True
    avail = n_ctx if (spec.f_l or 0.0) >= 1.0 else n_ctx - min(spec.budgets[0], n_ctx)
    return round(spec.f_u * U_SIZE) <= avail


# ---------------------------------------------------------------------------
# enumeration — the §4 cell table EXACTLY
# ---------------------------------------------------------------------------


def test_cms_cell_table_counts_match_plan_exactly():
    core = _cms_specs(budgets=(250, 2500, 8000), **CORE_GRID)
    ablation = _cms_specs(budgets=(2500,), **ABLATION_GRID)
    specs = core + ablation
    # one spec per (draw, seed): every compose spec carries singleton axes
    assert all(len(s.draws) == 1 and len(s.seeds) == 1 and len(s.budgets) == 1 for s in specs)
    # per variant per seed: core 9 (3 dedup keys x 3 anchors) + ablation 4 = 13
    for variant in ("context_end", "prefix_end"):
        for seed in SEEDS:
            sub = [s for s in specs if s.variant == variant and s.seeds == (seed,)]
            assert len(sub) == 13, (variant, seed, len(sub))
    # totals + realized-after-designed-skips per behavior (feasibility mirror)
    assert len(specs) == 13 * 2 * len(SEEDS)
    realized = {b: sum(_feasible(s, n) for s in specs) for b, n in N_CTX.items()}
    assert realized["sycophancy"] == 13 * 2 * len(SEEDS) == 130
    assert realized["hallucination"] == 130
    assert realized["evil"] == 11 * 2 * len(SEEDS) == 110
    # the evil designed-skip set is EXACTLY {(0.5,0,8000), (1.0,0,2500)}
    skipped = {(s.f_u, s.f_l, s.budgets[0]) for s in specs if not _feasible(s, N_CTX["evil"])}
    assert skipped == {(0.5, 0.0, 8000), (1.0, 0.0, 2500)}
    n_skipped_evil = sum(not _feasible(s, N_CTX["evil"]) for s in specs)
    assert n_skipped_evil == 2 * 2 * len(SEEDS) == 20
    # evil f_u=0.75 is the CONDITIONALLY-designed class: feasible-but-thin
    thin = [s for s in ablation if s.f_u == 0.75]
    assert thin and all(_feasible(s, N_CTX["evil"]) for s in thin)
    assert round(0.75 * U_SIZE) <= N_CTX["evil"] - 2500  # 3750 <= 3968


def test_cms_replicates_default_off_keeps_legacy_enumeration():
    legacy = fits_cli.compose_run_specs(
        variants=("context_end", "prefix_end"),
        regimes=("e1", "e2p"),
        u_sizes=(None, 5000),
        budgets=(250, 2500, 8000),
        draws=(0,),
        seeds=(0,),
        compose=True,
        f_u_grid=(0.0, 0.5),
        f_l_grid=(0.0, 1.0),
    )
    compose = [s for s in legacy if s.f_u is not None]
    plain = [s for s in legacy if s.f_u is None]
    assert len(plain) == 2 * 2 * 2  # variants x u_sizes x regimes
    assert len(compose) == 2 * 3 * 3  # variants x dedup keys x anchors (single replicate)
    assert all(s.draws == (0,) and s.seeds == (0,) for s in compose)


def test_cms_compose_only_requires_compose():
    with pytest.raises(ValueError, match="compose_only/compose_replicates"):
        fits_cli.compose_run_specs(
            variants=("context_end",),
            regimes=("e1",),
            u_sizes=(None,),
            budgets=(250,),
            draws=(0,),
            seeds=(0,),
            compose=False,
            compose_only=True,
        )


def test_cms_cache_sharers_are_consecutive():
    specs = _cms_specs(budgets=(250, 2500, 8000), **CORE_GRID)
    keys = [fits_cli.compose_fit_key(s) for s in specs]
    spans: dict[tuple, tuple[int, int]] = {}
    for i, k in enumerate(keys):
        a, b = spans.get(k, (i, i))
        spans[k] = (min(a, i), max(b, i))
    for k, (a, b) in spans.items():
        assert b - a + 1 == keys.count(k), f"non-consecutive sharers for {k}"


# ---------------------------------------------------------------------------
# map-fit cache key identity (plan §4 item 4)
# ---------------------------------------------------------------------------


def test_compose_fit_key_shares_anchor_independent_pools_only():
    specs = _cms_specs(budgets=(250, 2500, 8000), **CORE_GRID)

    def keys(f_u, f_l, seed):
        return [
            fits_cli.compose_fit_key(s)
            for s in specs
            if s.variant == "context_end" and s.f_u == f_u and s.f_l == f_l and s.seeds == (seed,)
        ]

    # f_u=0: no eliciting rows -> pool anchor-independent -> ONE key over 3 anchors
    assert len(set(keys(0.0, 0.0, 0))) == 1 and len(keys(0.0, 0.0, 0)) == 3
    # f_l=1: eliciting pool is the whole table -> anchor-independent too
    assert len(set(keys(0.5, 1.0, 0))) == 1 and len(keys(0.5, 1.0, 0)) == 3
    # f_u=0.5/f_l=0: anchor-cell exclusion -> anchor-SPECIFIC keys
    assert len(set(keys(0.5, 0.0, 0))) == 3
    # keys never share across seeds or variants
    assert not set(keys(0.0, 0.0, 0)) & set(keys(0.0, 0.0, 1))
    pfx = [
        fits_cli.compose_fit_key(s)
        for s in specs
        if s.variant == "prefix_end" and s.f_u == 0.0 and s.seeds == (0,)
    ]
    assert not set(keys(0.0, 0.0, 0)) & set(pfx)
    # unique-fit count per variant-x-seed slice: 3 keys' worth = 1 + 1 + 3 = 5
    ctx_s0 = {
        fits_cli.compose_fit_key(s) for s in specs if s.variant == "context_end" and s.seeds == (0,)
    }
    assert len(ctx_s0) == 5


def test_map_key_separates_draw_replicates():
    a = fits_cli.RunSpec(
        variant="context_end",
        regime="e1",
        u_size=5000,
        f_u=0.5,
        f_l=0.0,
        budgets=(250,),
        draws=(0,),
        seeds=(1,),
    )
    b = fits_cli.RunSpec(
        variant="context_end",
        regime="e1",
        u_size=5000,
        f_u=0.5,
        f_l=0.0,
        budgets=(250,),
        draws=(1,),
        seeds=(1,),
    )
    assert fits_cli._map_key(a) != fits_cli._map_key(b)
    assert fits_cli.compose_fit_key(a) != fits_cli.compose_fit_key(b)


# ---------------------------------------------------------------------------
# whitening-rng spec-seed threading (plan §4 item 3)
# ---------------------------------------------------------------------------


def test_map_seed_for_spec_threads_spec_seed_and_preserves_plain_pin():
    compose = fits_cli.RunSpec(
        variant="context_end",
        regime="e1",
        u_size=5000,
        f_u=0.5,
        f_l=0.0,
        budgets=(250,),
        draws=(0,),
        seeds=(3,),
    )
    assert fits_cli.map_seed_for_spec(compose, (0, 1, 2, 3, 4)) == 3
    plain = fits_cli.RunSpec(
        variant="context_end",
        regime="e1",
        u_size=None,
        budgets=(250, 2500),
        draws=(0,),
        seeds=(0, 1, 2),
    )
    # plain rung: spec.seeds IS the CLI tuple -> the committed first-seed pin
    assert fits_cli.map_seed_for_spec(plain, (0, 1, 2)) == 0
    assert fits_cli.map_seed_for_spec(plain, (0, 1, 2)) == int(plain.seeds[0])


def test_whitening_seed0_reproduces_banked_path_and_seed1_diverges():
    from explore_persona_space.experiments.issue_1739 import fits

    rng = np.random.default_rng(7)
    x = rng.normal(size=(2, 60, 8))
    # banked round-1 path: map_seed = int(args.seeds[0]) with --seeds 0
    banked = fits.fit_whitening(x, seed=0)
    threaded = fits.fit_whitening(x, seed=0)  # spec seed 0 via map_seed_for_spec
    np.testing.assert_array_equal(banked.w, threaded.w)
    other = fits.fit_whitening(x, seed=1)
    assert not np.allclose(banked.w, other.w)


def test_compose_label_matches_banked_format():
    spec = fits_cli.RunSpec(
        variant="context_end",
        regime="e1",
        u_size=5000,
        f_u=0.5,
        f_l=0.0,
        budgets=(250,),
        draws=(0,),
        seeds=(2,),
    )
    assert fits_cli.compose_label(spec) == "compose5000_fu0.5_fl0.0_L250"


# ---------------------------------------------------------------------------
# fold pure functions
# ---------------------------------------------------------------------------


def _cell_row(variant, f_u, f_l, budget_l, seed, delta, *, draw=0, arm_rhos=None):
    arm_rhos = arm_rhos or {}
    base = dict(variant=variant, f_u=f_u, f_l=f_l, budget_l=budget_l, draw=draw, seed=seed)
    arms = [
        {**base, "arm": slug, "rho_frozen": arm_rhos.get(slug, 0.1)}
        for slug in ("arm6_map_proj_e1", "arm2_ctx_native", "arm13_shuffled_map")
    ]
    return {
        "arms": arms,
        "headline": {
            "pair": list(fold.HEADLINE_PAIR),
            "delta_rho_frozen": delta,
            "ci_delta_frozen": [delta - 0.05, delta + 0.05],
        },
        "unit_key": f"{variant}|{f_u}|{f_l}|{budget_l}|{draw}|{seed}",
    }


def _full_grid_cells(delta_fn):
    rows = []
    for variant in fold.VARIANTS:
        for anchor in fold.ANCHOR_BUDGETS:
            for seed in SEEDS:
                for f_u in (0.0, 0.5):
                    rows.append(_cell_row(variant, f_u, 0.0, anchor, seed, delta_fn(f_u, seed)))
    return rows


def test_fold_uniqueness_assert_fires_on_duplicate():
    rows = _full_grid_cells(lambda f_u, seed: 0.1)
    fold.assert_unique_cells(rows)  # clean union passes
    with pytest.raises(ValueError, match="duplicate compose cell"):
        fold.assert_unique_cells([*rows, rows[0]])


def test_flip_contrast_pair_coverage_and_verdict_confirmed():
    rows = _full_grid_cells(lambda f_u, seed: (0.30 if f_u == 0.5 else -0.05) + 0.01 * seed)
    by_key = fold.assert_unique_cells(rows)
    contrast = fold.flip_contrast(by_key, seeds=SEEDS)
    assert contrast["pairs"] == 20 and not contrast["missing_pairs"]
    c_tci = fold.t_ci([v for v in contrast["per_seed_C"].values()])
    assert c_tci["lo"] > 0
    assert fold.lattice_verdict(c_tci, contrast["anchor_seedmean_delta05"]) == "FLIP-CONFIRMED"


def test_lattice_verdict_falsified_requires_ci_below_zero():
    rows = _full_grid_cells(lambda f_u, seed: (-0.30 if f_u == 0.5 else 0.05) + 0.01 * seed)
    by_key = fold.assert_unique_cells(rows)
    contrast = fold.flip_contrast(by_key, seeds=SEEDS)
    c_tci = fold.t_ci(list(contrast["per_seed_C"].values()))
    assert c_tci["hi"] < 0
    assert fold.lattice_verdict(c_tci, contrast["anchor_seedmean_delta05"]) == "FLIP-FALSIFIED"


def test_lattice_verdict_indeterminate_on_straddling_ci_or_anchor_floor():
    # CI straddles 0 -> INDETERMINATE
    noisy = fold.t_ci([0.3, -0.2, 0.1, -0.15, 0.05])
    assert fold.lattice_verdict(noisy, {"a": 1.0, "b": 1.0}) == "INDETERMINATE"
    # CI above 0 but <2 anchors with positive seed-mean Delta(0.5,0) -> INDETERMINATE
    tight = fold.t_ci([0.20, 0.21, 0.19, 0.22, 0.20])
    assert tight["lo"] > 0
    anchors = {"ctx|L250": 0.5, "ctx|L2500": -0.1, "pfx|L250": -0.2, "pfx|L2500": -0.3}
    assert fold.lattice_verdict(tight, anchors) == "INDETERMINATE"


def test_round_verdict_grammar():
    assert fold.round_verdict(["FLIP-CONFIRMED", "FLIP-CONFIRMED", "INDETERMINATE"]) == "CONFIRMED"
    assert fold.round_verdict(["FLIP-FALSIFIED", "FLIP-FALSIFIED", "FLIP-CONFIRMED"]) == "FALSIFIED"
    assert fold.round_verdict(["FLIP-CONFIRMED", "FLIP-FALSIFIED", "INDETERMINATE"]) == "MIXED"
    assert (
        fold.round_verdict(["HALTED-SKIP-AUDIT", "FLIP-CONFIRMED", "FLIP-CONFIRMED"]) == "PENDING"
    )


def test_t_ci_uses_df4_multiplier():
    vals = [1.0, 2.0, 3.0, 4.0, 5.0]
    ci = fold.t_ci(vals)
    sd = float(np.std(vals, ddof=1))
    half = 2.7764451 * sd / np.sqrt(5)
    assert ci["mean"] == pytest.approx(3.0)
    assert ci["hi"] - ci["mean"] == pytest.approx(half, rel=1e-5)


def _skip_rows(behavior_combos, seeds=SEEDS):
    return [
        dict(variant=v, f_u=fu, f_l=fl, budget_l=ll, draw=0, seed=s, reason="quota")
        for (fu, fl, ll) in behavior_combos
        for v in fold.VARIANTS
        for s in seeds
    ]


def test_designed_skip_audit_exact_set_passes():
    rows = _skip_rows(((0.5, 0.0, 8000), (1.0, 0.0, 2500)))
    audit = fold.designed_skip_audit(rows, "evil", seeds=SEEDS)
    assert audit["ok"] and audit["n_recorded"] == 20 and audit["n_designed"] == 20
    assert not audit["extra_skips"] and not audit["missing_designed_skips"]


def test_designed_skip_audit_halts_on_extra_and_missing():
    rows = _skip_rows(((0.5, 0.0, 8000),))  # missing the (1.0, 0, 2500) class
    audit = fold.designed_skip_audit(rows, "evil", seeds=SEEDS)
    assert not audit["ok"] and len(audit["missing_designed_skips"]) == 10
    extra = _skip_rows(((0.5, 0.0, 8000), (1.0, 0.0, 2500), (0.5, 0.0, 250)))
    audit = fold.designed_skip_audit(extra, "evil", seeds=SEEDS)
    assert not audit["ok"] and len(audit["extra_skips"]) == 10
    # syco/hall design NO skips: any recorded skip is extra
    audit = fold.designed_skip_audit(_skip_rows(((0.5, 0.0, 8000),)), "sycophancy", seeds=SEEDS)
    assert not audit["ok"] and len(audit["extra_skips"]) == 10


def test_designed_skip_audit_conditional_class_never_halts():
    rows = _skip_rows(((0.5, 0.0, 8000), (1.0, 0.0, 2500), (0.75, 0.0, 2500)))
    audit = fold.designed_skip_audit(rows, "evil", seeds=SEEDS)
    assert audit["ok"]
    assert len(audit["conditional_skips_recorded"]) == 10


def test_banked_repro_join_and_tolerance():
    fresh_rows = [
        _cell_row("context_end", 0.5, 0.0, 250, 0, 0.3496),
        _cell_row("context_end", 0.0, 0.0, 250, 0, -0.0820),
        _cell_row("context_end", 0.5, 0.0, 250, 1, 0.9),  # seed 1: never joined
    ]
    by_key = fold.assert_unique_cells(fresh_rows)
    banked = [
        _cell_row("context_end", 0.5, 0.0, 250, 0, 0.3496),
        _cell_row("context_end", 0.0, 0.0, 250, 0, -0.0820),
        _cell_row("context_end", 0.5, 1.0, 8000, 0, 0.1),  # not in fresh -> ignored
    ]
    repro = fold.banked_repro(by_key, banked, tol=5e-3)
    assert repro["ok"] and repro["n_compared"] == 2
    banked_div = [_cell_row("context_end", 0.5, 0.0, 250, 0, 0.30)]
    repro = fold.banked_repro(by_key, banked_div, tol=5e-3)
    assert not repro["ok"] and len(repro["material_divergences"]) == 1


def test_dose_curve_levels_and_seed_grain():
    rows = []
    for variant in fold.VARIANTS:
        for seed in SEEDS:
            for f_u in (0.0, 0.1, 0.25, 0.5, 0.75, 1.0):
                rows.append(_cell_row(variant, f_u, 0.0, 2500, seed, f_u * 0.4))
    by_key = fold.assert_unique_cells(rows)
    curve = fold.dose_curve(by_key, seeds=SEEDS)
    assert set(curve) == {"fu0.0", "fu0.1", "fu0.25", "fu0.5", "fu0.75", "fu1.0"}
    assert curve["fu0.5"]["mean"] == pytest.approx(0.2)
    assert curve["fu0.5"]["n"] == 5


def test_pairwise_pool_overlap_jaccard():
    rows = [
        dict(
            variant="context_end",
            f_u=0.5,
            f_l=0.0,
            budget_l=250,
            draw=0,
            seed=0,
            elic_ctx_ids=["a", "b", "c"],
        ),
        dict(
            variant="context_end",
            f_u=0.5,
            f_l=0.0,
            budget_l=250,
            draw=0,
            seed=1,
            elic_ctx_ids=["b", "c", "d"],
        ),
    ]
    out = fold.pairwise_pool_overlap(rows)
    key = "context_end|fu0.5_fl0.0_L250"
    assert out[key]["n_pairs"] == 1
    assert out[key]["jaccard_mean"] == pytest.approx(2 / 4)


def test_sensitivity_recompute_pending_without_local_preds(tmp_path):
    cell = _cell_row("context_end", 0.5, 0.0, 250, 0, 0.3)
    cell["preds_npz"] = "deadbeef.npz"
    cell["_source_root"] = str(tmp_path / "absent")
    by_key = fold.assert_unique_cells([cell])
    meta = [
        dict(
            variant="context_end",
            f_u=0.5,
            f_l=0.0,
            budget_l=250,
            draw=0,
            seed=0,
            overlap_group_count=2,
            overlap_groups=["g1", "g2"],
            elic_ctx_ids=[],
        )
    ]
    sens = fold.sensitivity_recompute(by_key, meta, {})
    assert sens["status"] == "pending" and sens["n_pending"] == 1


def test_sensitivity_recompute_excludes_overlap_group_rows(tmp_path):
    rng = np.random.default_rng(3)
    n = 24
    dv = rng.normal(size=n).astype(np.float32)
    pred6 = (dv + 0.2 * rng.normal(size=n)).astype(np.float32)
    pred2 = rng.normal(size=n).astype(np.float32)
    ctx = np.asarray([f"c{i}" for i in range(n)])
    preds_dir = tmp_path / "half" / "arm_results" / "percell" / "preds"
    preds_dir.mkdir(parents=True)
    np.savez(
        preds_dir / "abc.npz",
        row_idx=np.arange(n),
        context_ids=ctx,
        dv=dv,
        unit_key=np.asarray("k"),
        pred__arm6_map_proj_e1=pred6,
        pred__arm2_ctx_native=pred2,
    )
    cell = _cell_row("context_end", 0.5, 0.0, 250, 0, 0.3)
    cell["preds_npz"] = "abc.npz"
    cell["_source_root"] = str(tmp_path / "half")
    by_key = fold.assert_unique_cells([cell])
    groups = {f"c{i}": ("gA" if i < 4 else f"g{i}") for i in range(n)}
    meta = [
        dict(
            variant="context_end",
            f_u=0.5,
            f_l=0.0,
            budget_l=250,
            draw=0,
            seed=0,
            overlap_group_count=1,
            overlap_groups=["gA"],
            elic_ctx_ids=[],
        )
    ]
    sens = fold.sensitivity_recompute(by_key, meta, groups)
    assert sens["status"] == "ok" and sens["n_recomputed"] == 1
    assert sens["rows"][0]["n_kept"] == n - 4
    assert sens["rows"][0]["delta_excl_overlap"] is not None


def test_map_diag_presence_missing_level_reported():
    diag = {
        "context_end|compose5000_fu0.0_fl0.0_L250|draw0_seed0": {},
        "prefix_end|compose5000_fu0.0_fl0.0_L250|draw0_seed0": {},
    }
    out = fold.map_diag_presence(diag, levels=((0.0, 0.0), (0.5, 0.0)))
    assert not out["ok"]
    assert "context_end|fu0.5_fl0.0|seed0" in out["missing"]
    ok = fold.map_diag_presence(diag, levels=((0.0, 0.0),))
    assert ok["ok"]


def test_cell_class_table_seed_grain_and_arm_columns():
    rows = [
        _cell_row(
            "context_end",
            0.5,
            0.0,
            250,
            s,
            0.2 + 0.01 * s,
            arm_rhos={"arm6_map_proj_e1": 0.5, "arm2_ctx_native": 0.3, "arm13_shuffled_map": 0.1},
        )
        for s in SEEDS
    ]
    by_key = fold.assert_unique_cells(rows)
    table = fold.cell_class_table(by_key, {}, seeds=SEEDS)
    assert len(table) == 1
    row = table[0]
    assert row["n_seeds"] == 5 and row["seeds"] == list(SEEDS)
    assert row["delta_mean"] == pytest.approx(0.22)
    assert row["delta_tci_lo"] is not None and row["delta_tci_lo"] < row["delta_mean"]
    assert row["arm6_minus_arm13_margin"] == pytest.approx(0.4)
    assert len(row["per_seed_boot_ci"]) == 5


def test_cms_tiny_real_e2e_replicates_cache_sidecars_and_merge(tmp_path, capsys):
    """Tiny-real CPU e2e of the NEW path: --compose-only --compose-replicates
    through the production _run_real — replicate cells land per (draw, seed),
    the single-entry fit cache HITs on anchor-independent pools, the
    compose_pool_meta/compose_skips sidecars append, map_diagnostics keys
    carry the replicate suffix, a second (ablation-style) invocation into the
    SAME out-root MERGES instead of clobbering, and a re-run resumes."""
    from tests.test_issue1739_fits import _write_tiny_real_inputs

    dv_json, _feats = _write_tiny_real_inputs(tmp_path)
    out_root = tmp_path / "out"
    argv = [
        "--behavior",
        "evil",
        "--labeled-store",
        str(tmp_path / "labeled"),
        "--dv-json",
        str(dv_json),
        "--u-store",
        str(tmp_path / "ustore"),
        "--e1-store",
        str(tmp_path / "e1"),
        "--out-root",
        str(out_root),
        "--tensors-root",
        str(tmp_path / "tensors"),
        "--device",
        "cpu",
        "--config",
        "config_a",
        "--regimes",
        "e1",
        "--budgets",
        "6",
        "10",
        "--draws",
        "0",
        "--seeds",
        "0",
        "1",
        "--layers",
        "0",
        "1",
        "2",
        "--n-boot",
        "20",
        "--n-perm",
        "20",
        "--arms",
        "arm2_ctx_native",
        "arm4_ridge_ctx",
        "arm6_map_proj_e1",
        "arm7_map_ridge_pred",
        "arm13_shuffled_map",
        "--compose",
        "--compose-only",
        "--compose-replicates",
        "--compose-u-size",
        "8",
    ]
    assert fits_cli.main(argv) == 0
    out1 = capsys.readouterr().out
    # dedup keys {(0,0),(0.5,0),(0.5,1)} x 2 anchors x 2 variants x 2 seeds = 24
    percell = out_root / "arm_results" / "percell"
    cells = [json.loads(ln) for ln in (percell / "cells.jsonl").read_text().splitlines() if ln]
    assert len(cells) == 24
    ids = {fold.cell_id(c) for c in cells}
    assert len(ids) == 24
    assert {k[5] for k in ids} == {0, 1}  # both seeds realized
    # anchor-independent pools ((0,0) + (0.5,1)) HIT the cache on anchor 2 of 2:
    # 2 keys x 2 variants x 2 seeds = 8 hits
    assert out1.count("whitening+map cache HIT") == 8
    # pool meta: the anchor-DEPENDENT class only — (0.5, 0) x 2 anchors x 2
    # variants x 2 seeds = 8 rows, each with realized eliciting ids
    metas = [
        json.loads(ln)
        for ln in (percell / "compose_pool_meta.jsonl").read_text().splitlines()
        if ln
    ]
    assert len(metas) == 8
    assert all(m["f_u"] == 0.5 and m["f_l"] == 0.0 and m["elic_ctx_ids"] for m in metas)
    # no quota failure at this tiny shape -> no skip sidecar rows
    assert not (percell / "compose_skips.jsonl").exists()
    diag = json.loads((out_root / "map_diagnostics.json").read_text())
    assert all("|draw0_seed" in k for k in diag)
    assert any(k.endswith("seed0") for k in diag) and any(k.endswith("seed1") for k in diag)
    assert any(v.get("map_source") == "cache" for v in diag.values())
    # ablation-style second invocation into the SAME out-root: sidecars append,
    # diagnostics MERGE (first invocation's keys survive)
    argv2 = [a for a in argv]
    b_i = argv2.index("--budgets")
    argv2[b_i : b_i + 3] = ["--budgets", "6"]
    assert fits_cli.main([*argv2, "--f-u-grid", "0.25", "--f-l-grid", "0.0"]) == 0
    cells2 = [json.loads(ln) for ln in (percell / "cells.jsonl").read_text().splitlines() if ln]
    assert len(cells2) == 24 + 2 * 2  # + fu0.25 x 2 variants x 2 seeds at one anchor
    diag2 = json.loads((out_root / "map_diagnostics.json").read_text())
    assert set(diag).issubset(set(diag2)) and any("fu0.25" in k for k in diag2)
    # idempotent resume of the first invocation: no new rows
    assert fits_cli.main(argv) == 0
    cells3 = [json.loads(ln) for ln in (percell / "cells.jsonl").read_text().splitlines() if ln]
    assert len(cells3) == len(cells2)
    assert "SKIP (resume)" in capsys.readouterr().out


def test_fold_load_behavior_union_reads_both_halves(tmp_path):
    root = tmp_path / "compose_multiseed"
    for half, seeds in (("s02", (0, 1, 2)), ("s34", (3, 4))):
        percell = root / "evil" / half / "arm_results" / "percell"
        percell.mkdir(parents=True)
        with (percell / "cells.jsonl").open("w") as fh:
            for s in seeds:
                fh.write(json.dumps(_cell_row("context_end", 0.0, 0.0, 250, s, 0.1)) + "\n")
        (percell / "compose_skips.jsonl").write_text(
            json.dumps(
                dict(
                    variant="context_end",
                    f_u=1.0,
                    f_l=0.0,
                    budget_l=2500,
                    draw=0,
                    seed=seeds[0],
                    u_size=5000,
                    reason="quota",
                )
            )
            + "\n"
        )
        (root / "evil" / half / "map_diagnostics.json").write_text(
            json.dumps({f"context_end|compose5000_fu0.0_fl0.0_L250|draw0_seed{seeds[0]}": {}})
        )
    data = fold.load_behavior(root, "evil", ("s02", "s34"))
    assert len(data["cells"]) == 5
    assert not data["missing_halves"]
    assert len(data["skips"]) == 2  # distinct seeds -> both kept
    assert len(data["diag"]) == 2
    by_key = fold.assert_unique_cells(data["cells"])
    assert len(by_key) == 5
