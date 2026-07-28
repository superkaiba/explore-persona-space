"""Synthetic-only tests for the issue #1739 Phase-3/4 engine (round C1).

Covers the round-C1 brief list: fold-sharing identity, matched-budget
accounting, shuffled-W norm preservation, einsum-projection correctness vs a
naive loop, batched-vs-serial bootstrap parity, kNN chance on a constant
predictor, W-consumption (return_weights) round-trip, is_eval_only exclusion,
and a figures smoke off a synthetic results payload. Everything is tiny +
CPU; no network, no staged data, no GPU.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

from explore_persona_space.experiments.issue_1739 import arms, dv_build, fits, gates, store_io

RNG = np.random.default_rng(1739)


def _synthetic_cell_data(n=30, n_layers=3, d=8, n_groups=6, seed=0):
    rng = np.random.default_rng(seed)
    rb_raw = rng.normal(size=(n_layers, d))
    x_ctx = rng.normal(size=(n_layers, n, d))
    y_ans = 0.7 * x_ctx + 0.3 * rng.normal(size=(n_layers, n, d))
    dv = np.clip(
        50
        + 14 * np.einsum("lnd,ld->n", y_ans, rb_raw) / (n_layers * np.sqrt(d))
        + rng.normal(scale=3, size=n),
        0,
        100,
    )
    groups = [f"g{i % n_groups}" for i in range(n)]
    x_u = rng.normal(size=(n_layers, 64, d))
    y_u = 0.7 * x_u + 0.3 * rng.normal(size=x_u.shape)
    wh = fits.fit_whitening(x_u, seed=seed)
    mapfit = fits.fit_linear_map(fits.apply_whitening(x_u, wh), fits.apply_whitening(y_u, wh))
    data = arms.CellData(
        z_ctx=fits.apply_whitening(x_ctx, wh),
        z_ans=fits.apply_whitening(y_ans, wh),
        dv=dv,
        rb=np.einsum("ld,lde->le", rb_raw, wh.w),
        mapfit=mapfit,
        text_emb=rng.normal(size=(n, 5)),
        text_features=rng.normal(size=(n, 3)),
        layers=(0, 1, 2),
    )
    return data, groups


# ---------------------------------------------------------------------------
# matched-budget protocol
# ---------------------------------------------------------------------------


def test_budget_cell_fold_sharing_identity():
    groups = [f"g{i % 7}" for i in range(40)]
    a = fits.realize_budget_cell(groups, budget_l=20, draw=2, seed=1)
    b = fits.realize_budget_cell(groups, budget_l=20, draw=2, seed=1)
    # The SAME (L, draw, seed) realizes IDENTICAL rows + folds — every arm
    # consuming this cell shares them by construction.
    assert np.array_equal(a.row_idx, b.row_idx)
    assert np.array_equal(a.fold_ids, b.fold_ids)
    c = fits.realize_budget_cell(groups, budget_l=20, draw=3, seed=1)
    assert not np.array_equal(a.row_idx, c.row_idx)  # a different draw re-samples


def test_budget_cell_accounting_and_group_folds():
    groups = [f"g{i % 9}" for i in range(50)]
    cell = fits.realize_budget_cell(groups, budget_l=23, draw=0, seed=0)
    assert len(cell.row_idx) == 23  # exact budget accounting
    assert len(np.unique(cell.row_idx)) == 23
    keys = np.asarray(groups)[cell.row_idx]
    for g in np.unique(keys):  # a group's rows never straddle folds
        assert len(set(cell.fold_ids[keys == g])) == 1
    assert (np.bincount(cell.fold_ids, minlength=cell.n_folds) > 0).all()
    over = fits.realize_budget_cell(groups, budget_l=500, draw=0, seed=0)
    assert len(over.row_idx) == 50  # capped at the table size


# ---------------------------------------------------------------------------
# fits: whitening / map / directions / controls
# ---------------------------------------------------------------------------


def test_shuffled_map_weights_preserve_frobenius():
    w = RNG.normal(size=(3, 8, 8))
    w_shuf = fits.shuffled_map_weights(w, seed=7)
    for li in range(3):
        assert np.isclose(np.linalg.norm(w[li]), np.linalg.norm(w_shuf[li]))
        assert not np.allclose(w[li], w_shuf[li])


def test_ridge_return_weights_roundtrip():
    # W-consumption path: preds reconstruct EXACTLY from (W, x_mu, x_sd, y_mu).
    from explore_persona_space.experiments.issue_779.fit_h import (
        ridge_fit_predict_fast_layer_batched,
    )

    rng = np.random.default_rng(3)
    x_tr = rng.normal(size=(2, 20, 6))
    y_tr = rng.normal(size=(2, 20, 6))
    x_ev = rng.normal(size=(2, 5, 6))
    preds, w = ridge_fit_predict_fast_layer_batched(x_tr, y_tr, x_ev, return_weights=True)
    x_mu = x_tr.mean(axis=1, keepdims=True)
    x_sd = x_tr.std(axis=1, keepdims=True) + 1e-9
    y_mu = y_tr.mean(axis=1, keepdims=True)
    manual = ((x_ev - x_mu) / x_sd) @ w + y_mu
    assert np.allclose(preds, manual, atol=1e-8)


def test_apply_map_matches_full_pool_refit():
    # Round-2 contract: diagnostics come from the 80/20 split fit, but the
    # FROZEN weights are refit on the FULL U pool (review minor: effective U
    # must equal the nominal rung). apply_map must reproduce a direct helper
    # fit on the full pool EXACTLY.
    rng = np.random.default_rng(4)
    x = rng.normal(size=(2, 40, 6))
    y = 0.5 * x + 0.1 * rng.normal(size=x.shape)
    m = fits.fit_linear_map(x, y, seed=5)
    assert m.diagnostics["w_refit_on_full_u"] is True
    assert m.diagnostics["w_fit_rows"] == 40
    x_new = rng.normal(size=(2, 7, 6))
    _preds, w_full = fits.ridge_layer_batched_auto(x, y, x_new, return_weights=True)
    manual = ((x_new - x.mean(axis=1, keepdims=True)) / (x.std(axis=1, keepdims=True) + 1e-9)) @ (
        w_full
    ) + y.mean(axis=1, keepdims=True)
    assert np.allclose(fits.apply_map(x_new, m), manual, atol=1e-8)


def test_primal_dual_ridge_parity():
    # M6: the primal (d x d Gram) twin must reproduce the parent dual helper
    # bit-close — preds, weights, AND the per-slice GCV lambda selection —
    # at a shape where both branches run (n_tr > d so primal is the router
    # choice; the dual helper still accepts it).
    from explore_persona_space.experiments.issue_779.fit_h import (
        ridge_fit_predict_fast_layer_batched,
    )

    rng = np.random.default_rng(11)
    x_tr = rng.normal(size=(3, 60, 8))
    y_tr = 0.4 * x_tr + 0.2 * rng.normal(size=(3, 60, 8))
    x_ev = rng.normal(size=(3, 9, 8))
    lam = np.asarray(fits.RIDGE_LAMBDAS, dtype=np.float64)
    p_dual, w_dual = ridge_fit_predict_fast_layer_batched(
        x_tr, y_tr, x_ev, lambdas=lam, return_weights=True
    )
    p_prim, w_prim = fits.ridge_fit_predict_primal_layer_batched(
        x_tr, y_tr, x_ev, lambdas=lam, return_weights=True, layer_chunk=2
    )
    assert np.allclose(p_prim, p_dual, atol=1e-8), np.max(np.abs(p_prim - p_dual))
    assert np.allclose(w_prim, w_dual, atol=1e-8), np.max(np.abs(w_prim - w_dual))
    # Router: n_tr > d -> primal; n_tr <= d -> dual (delegation returns the
    # dual helper's own output verbatim).
    p_auto = fits.ridge_layer_batched_auto(x_tr, y_tr, x_ev, lambdas=lam)
    assert np.allclose(p_auto, p_prim, atol=1e-12)
    x_small = x_tr[:, :6]
    y_small = y_tr[:, :6]
    p_auto_small = fits.ridge_layer_batched_auto(x_small, y_small, x_ev, lambdas=lam)
    p_dual_small = ridge_fit_predict_fast_layer_batched(x_small, y_small, x_ev, lambdas=lam)
    assert np.allclose(p_auto_small, p_dual_small, atol=1e-12)


def test_matched_pair_split_weights_reproduce_extract():
    # The row-weight refactor (E2/E2p over flat per-rollout store rows) must
    # reproduce extract_rb_matched exactly on random data, incl. NaN drops.
    rng = np.random.default_rng(12)
    acts = rng.normal(size=(9, 5, 2, 4))
    scores = rng.uniform(0, 100, size=(9, 5))
    scores[rng.random(size=(9, 5)) < 0.2] = np.nan
    scores[0] = 50.0  # one non-qualifying (flat) context
    for pooled in (False, True):
        rb_ref, n_ref = fits.extract_rb_matched(acts, scores, spread_min=15.0, pooled=pooled)
        w_hi, w_lo, n_w = fits.matched_pair_split_weights(scores, spread_min=15.0, pooled=pooled)
        rb_w = np.einsum("ck,ckld->ld", w_hi - w_lo, acts)
        assert n_w == n_ref
        assert np.allclose(rb_w, rb_ref, atol=1e-12)


def test_extract_rb_e1_and_matched():
    rng = np.random.default_rng(6)
    direction = rng.normal(size=(2, 5))
    pos = rng.normal(size=(10, 2, 5)) + direction
    neg = rng.normal(size=(12, 2, 5)) - direction
    rb = fits.extract_rb_e1(pos, neg)
    assert rb.shape == (2, 5)
    cos = (rb * 2 * direction).sum() / (np.linalg.norm(rb) * np.linalg.norm(2 * direction))
    assert cos > 0.5  # recovers the planted direction

    acts = rng.normal(size=(8, 4, 2, 5))
    scores = np.tile(np.array([10.0, 20.0, 80.0, 90.0]), (8, 1))
    acts += (scores >= 50)[:, :, None, None] * direction
    rb2, n_qual = fits.extract_rb_matched(acts, scores, spread_min=15.0)
    assert rb2.shape == (2, 5) and n_qual == 8
    rb2p, _ = fits.extract_rb_matched(acts, scores, spread_min=15.0, pooled=True)
    assert rb2p.shape == (2, 5)
    with pytest.raises(ValueError):
        fits.extract_rb_matched(acts, np.full((8, 4), 50.0), spread_min=15.0)


def test_fit_pool_mask_is_eval_only_exclusion():
    meta = [{"is_eval_only": False} for _ in range(6)]
    meta += [{"is_eval_only": True}, {"stratum": "battery"}, {}]
    mask = store_io.fit_pool_mask(meta)
    assert mask.tolist() == [True] * 6 + [False, False, True]


def test_compose_u_pool_fractions():
    gen, elic = fits.compose_u_pool(100, 50, f_u=0.5, size=40, seed=0)
    assert len(gen) == 20 and len(elic) == 20
    gen0, elic0 = fits.compose_u_pool(100, 0, f_u=0.0, size=30, seed=0)
    assert len(gen0) == 30 and len(elic0) == 0
    with pytest.raises(ValueError):
        fits.compose_u_pool(10, 2, f_u=0.5, size=40, seed=0)


# ---------------------------------------------------------------------------
# arms: projections, metrics, batched-vs-serial parity
# ---------------------------------------------------------------------------


def test_projection_einsum_matches_naive_loop():
    z = RNG.normal(size=(3, 10, 8))
    rb = RNG.normal(size=(3, 8))
    fast = arms._proj(z, rb)
    naive = np.empty((3, 10))
    for li in range(3):
        for i in range(10):
            naive[li, i] = float(np.dot(z[li, i], rb[li]))
    assert np.allclose(fast, naive, atol=1e-12)


def test_spearman_and_bootstrap_batched_vs_serial():
    from scipy.stats import spearmanr

    rng = np.random.default_rng(8)
    scores = rng.normal(size=(2, 20))
    dv = rng.normal(size=20)
    batched = arms.spearman_rows(scores, dv)
    for s in range(2):
        assert np.isclose(batched[s], spearmanr(scores[s], dv).statistic, atol=1e-12)

    idx = arms.make_bootstrap_idx(20, n_boot=8, seed=0)
    draws = arms.bootstrap_rhos(scores, dv, idx, chunk_draws=3)
    assert draws.shape == (2, 8)
    for b in range(8):  # serial reference: same shared draws, scipy spearman
        for s in range(2):
            ref = spearmanr(scores[s][idx[b]], dv[idx[b]]).statistic
            assert np.isclose(draws[s, b], ref, atol=1e-12)


def test_auroc_rank_formula():
    from sklearn.metrics import roc_auc_score

    rng = np.random.default_rng(9)
    scores = rng.normal(size=(3, 40))
    labels = rng.random(40) > 0.5
    ours = arms.auroc_rows(scores, labels)
    for s in range(3):
        assert np.isclose(ours[s], roc_auc_score(labels, scores[s]), atol=1e-12)


def test_knn_chance_on_constant_predictor():
    from explore_persona_space.analysis.mapping_baselines import knn_retrieval

    rng = np.random.default_rng(10)
    true = rng.normal(size=(25, 6))
    pred = np.tile(true.mean(axis=0), (25, 1))  # degenerate constant predictor
    for metric in ("euclidean", "cosine"):
        res = knn_retrieval(pred, true, ks=(1, 5), metric=metric)
        for k in (1, 5):
            assert np.isclose(res["acc_at_k"][k], res["chance_at_k"][k], atol=1e-12)


def test_permutation_null_selection_symmetric():
    rng = np.random.default_rng(11)
    n = 60
    dv = rng.normal(size=n)
    null_scores = rng.normal(size=(4, n))
    res = arms.permutation_null_max(null_scores, dv, n_perm=200, seed=0)
    assert 0.0 < res["p_max_over_arms"] <= 1.0
    signal = np.vstack([null_scores, dv + 0.05 * rng.normal(size=n)])
    res2 = arms.permutation_null_max(signal, dv, n_perm=200, seed=0)
    assert res2["p_max_over_arms"] < 0.05  # a real max survives its own selection null


def test_split_half_ceiling_item_aligned():
    rng = np.random.default_rng(12)
    latent = rng.normal(size=40)
    per_rollout = latent[:, None] + 0.3 * rng.normal(size=(40, 4))
    res = arms.split_half_ceiling(per_rollout)
    assert res["n"] == 40 and res["r_half"] > 0.5 and res["ceiling_sb"] >= res["r_half"]
    assert res["scheme"].startswith("item-aligned")


def test_arm9_l0_gate_flips_on_perturbation(monkeypatch):
    # M1: the gate runs the REAL arm-9 alpha + residual-ridge path, so a
    # residual-assembly bug (simulated by corrupting the ridge slice pool the
    # arm-9 path consumes) MUST flip it — the round-1 tautology could not.
    data, _groups = _synthetic_cell_data()
    arms.verify_arm9_l0_degeneracy(data)  # real path: must not raise

    real_solve = arms._solve_ridge_slices

    def corrupted(slices, **kw):
        return {k: v + 0.5 for k, v in real_solve(slices, **kw).items()}

    monkeypatch.setattr(arms, "_solve_ridge_slices", corrupted)
    with pytest.raises(AssertionError, match="arm9 L2-SP"):
        arms.verify_arm9_l0_degeneracy(data)


def test_map_identity_reduces_arm6_to_arm3():
    # The plan's OWN arm-9 sanity (§5 arm_09_map_identity): under M = I with
    # y_mu = the learned bias b, arm 6 (map-then-project) must equal arm 3
    # (identity+learned-bias). Exact when (z_ans - z_ctx) is CONSTANT across
    # rows, so every fold's train-mean bias equals the global bias.
    rng = np.random.default_rng(21)
    n_layers, n, d = 2, 24, 6
    z = rng.normal(size=(n_layers, n, d))
    bias = rng.normal(size=(n_layers, 1, d))
    za = z + bias  # constant difference => per-fold b == global b
    rb = rng.normal(size=(n_layers, d))
    dv = rng.uniform(0, 100, size=n)
    ident = np.stack([np.eye(d)] * n_layers)
    mapfit = fits.MapFit(
        w=ident,
        x_mu=np.zeros((n_layers, 1, d)),
        x_sd=np.ones((n_layers, 1, d)),
        y_mu=bias,
        diagnostics={},
    )
    data = arms.CellData(z_ctx=z, dv=dv, rb=rb, z_ans=za, mapfit=mapfit, layers=(0, 1))
    cell = fits.realize_budget_cell([f"g{i}" for i in range(n)], budget_l=n, draw=0, seed=0)
    scores, skipped = arms.run_cell(data, cell, arms=["arm3_identity_bias", "arm6_map_proj_e1"])
    assert not skipped
    assert np.allclose(scores["arm6_map_proj_e1"], scores["arm3_identity_bias"], atol=1e-8)


# ---------------------------------------------------------------------------
# run_cell / run_grid end-to-end (synthetic)
# ---------------------------------------------------------------------------


def test_run_cell_all_arms_shapes_and_fold_sharing():
    data, groups = _synthetic_cell_data()
    cell = fits.realize_budget_cell(groups, budget_l=24, draw=0, seed=0)
    scores, skipped = arms.run_cell(data, cell, mlp_kwargs={"max_epochs": 2, "hidden": 4})
    assert not skipped, skipped
    assert set(scores) == set(arms.ARM_REGISTRY)
    n_l = len(cell.row_idx)
    for slug, sc in scores.items():
        expect_ly = 3 if arms.ARM_REGISTRY[slug]["layered"] else 1
        assert sc.shape == (expect_ly, n_l), (slug, sc.shape)
        assert np.isfinite(sc).all(), f"{slug} has non-finite OOF scores"
    # Oracle projection recovers the planted signal (dv is built FROM y_ans
    # projected on rb) and beats the shuffled-map control at these fixed seeds.
    rho = {
        s: float(np.nanmax(arms.spearman_rows(sc, data.dv[cell.row_idx])))
        for s, sc in scores.items()
    }
    assert rho["arm11_oracle_proj"] > 0.4
    assert rho["arm11_oracle_proj"] > rho["arm13_shuffled_map"]


def test_run_cell_arm5_fold_floor_skip():
    # Degenerate-input probe for the data-dependent MLP fold floor (the
    # callee's `n - max_fold >= 2` ddof-1 assert): below it, arm 5 records a
    # SKIP reason (never a grid-killing crash); a 1-fold cell fails LOUD.
    data, _groups = _synthetic_cell_data()
    tight = fits.realize_budget_cell(["gA", "gA", "gA", "gB"], budget_l=4, draw=0, seed=0)
    scores, skipped = arms.run_cell(
        data, tight, arms=["arm1_ctx_e1", "arm5_mlp_ctx"], mlp_kwargs={"max_epochs": 2}
    )
    assert "arm5_mlp_ctx" in skipped and "fold floor" in skipped["arm5_mlp_ctx"]
    assert "arm1_ctx_e1" in scores  # the rest of the matched-budget cell still runs
    one_fold = fits.realize_budget_cell(["gA", "gA", "gA"], budget_l=3, draw=0, seed=0)
    with pytest.raises(RuntimeError, match=">=2 group folds"):
        arms.run_cell(data, one_fold, arms=["arm1_ctx_e1"])


def test_run_cell_skips_without_optional_inputs():
    data, groups = _synthetic_cell_data()
    bare = arms.CellData(z_ctx=data.z_ctx, dv=data.dv, rb=data.rb, layers=data.layers)
    cell = fits.realize_budget_cell(groups, budget_l=20, draw=0, seed=0)
    scores, skipped = arms.run_cell(bare, cell, mlp_kwargs={"max_epochs": 2, "hidden": 4})
    assert "arm6_map_proj_e1" in skipped and "arm11_oracle_proj" in skipped
    assert "arm15_text_only" in skipped and "arm16_surface_feat" in skipped
    assert "arm1_ctx_e1" in scores and "arm4_ridge_ctx" in scores


def test_run_grid_checkpoint_resume_and_summary(tmp_path):
    data, groups = _synthetic_cell_data()
    prov = {
        "behavior": "synthetic",
        "variant": "context_end",
        "regime": "e1",
        "u_rung": 64,
        "eval_rung": "train",
        "config": "synthetic",
    }
    kw = dict(
        budgets=[16],
        draws=[0, 1],
        seeds=[0],
        provenance=prov,
        out_dir=tmp_path,
        arms=["arm1_ctx_e1", "arm2_ctx_native", "arm6_map_proj_e1"],
        n_boot=25,
        n_perm=25,
    )
    recs = arms.run_grid(data, groups, **kw)
    assert len(recs) == 2
    percell = tmp_path / "percell" / "cells.jsonl"
    assert percell.exists() and len(percell.read_text().strip().splitlines()) == 2
    # M3b: a resumed run SKIPS the compute but still returns the stored
    # records, so a crash+resume summary aggregates every cell.
    recs2 = arms.run_grid(data, groups, **kw)
    assert len(recs2) == 2
    assert [r["unit_key"] for r in recs2] == [r["unit_key"] for r in recs]
    assert len(percell.read_text().strip().splitlines()) == 2  # no re-run, no dup rows
    # M3a: an output-affecting flag change (n_boot here) changes the unit key,
    # so the resume predicate can never serve the other regime's cached rows.
    recs3 = arms.run_grid(data, groups, **{**kw, "n_boot": 30})
    assert len(recs3) == 2
    assert len(percell.read_text().strip().splitlines()) == 4
    out = arms.write_summary(recs2, tmp_path / "all_arms_spearman.json", meta={"mode": "test"})
    payload = json.loads(out.read_text())
    assert payload["n_cells"] == 2 and payload["n_arm_rows"] == 6
    row = payload["arm_rows"][0]
    for key in (
        "arm",
        "variant",
        "behavior",
        "regime",
        "u_rung",
        "budget_l",
        "draw",
        "seed",
        "fold_scheme",
    ):
        assert key in row, key
    assert payload["headlines"] and payload["nulls"]


# ---------------------------------------------------------------------------
# gate-2 fabrication-fraction rows (round-C1 design-note resolution)
# ---------------------------------------------------------------------------


def test_gate2_accepts_fabrication_fraction_rows():
    three_way = {
        f"ctx{i:02d}~k{k:02d}": ("fabricated" if (i + k) % 3 == 0 else "correct")
        for i in range(12)
        for k in range(3)
    }
    rows = [
        dict(r, dv=(None if r["dv"] is None else 100.0 * r["dv"]))
        for r in dv_build.build_three_way_dv(three_way)
    ]
    report = gates.gate2_spread_floor(rows, behavior="hallucination")
    assert report["verdict"] in ("PASS", "FAIL")
    assert all(r["dv"] is None or 0.0 <= r["dv"] <= 100.0 for r in rows)


# ---------------------------------------------------------------------------
# figures smoke (synthetic results payload; tmp_path out-root ONLY)
# ---------------------------------------------------------------------------


def _load_fits_cli():
    path = Path(__file__).resolve().parents[1] / "scripts" / "issue1739_fits.py"
    spec = importlib.util.spec_from_file_location("issue1739_fits_cli", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def test_synthetic_cli_and_figures_smoke(tmp_path):
    from explore_persona_space.experiments.issue_1739 import figures

    cli = _load_fits_cli()
    rc = cli.main(
        [
            "--synthetic",
            "24",
            "--budgets",
            "12",
            "--draws",
            "0",
            "1",
            "--seeds",
            "0",
            "--n-boot",
            "25",
            "--n-perm",
            "25",
            "--arms",
            "arm1_ctx_e1",
            "arm4_ridge_ctx",
            "arm6_map_proj_e1",
            "arm13_shuffled_map",
            "--out-root",
            str(tmp_path / "results"),
        ]
    )
    assert rc == 0
    summary_path = tmp_path / "results" / "arm_results" / "all_arms_spearman.json"
    summary = json.loads(summary_path.read_text())
    assert summary["n_arm_rows"] == 8

    fig_dir = tmp_path / "figs"
    paths = figures.render_summary_figures(summary, fig_dir)
    pngs = [p for p in paths if str(p).endswith(".png")]
    assert pngs, paths
    for png in pngs:
        assert png.exists() and png.stat().st_size > 5_000  # non-trivial render
    import matplotlib.image as mpimg

    img = mpimg.imread(pngs[0])
    assert img.shape[0] > 100 and img.shape[1] > 100  # non-empty canvas
    meta = json.loads((fig_dir / "hero_spearman_by_arm.meta.json").read_text())
    assert meta.get("points"), "hero figure sidecar has no plotted points (empty axes?)"

    scatter = figures.fig_percell_scatter(
        np.arange(10.0), np.linspace(0, 100, 10), [f"ctx{i}" for i in range(10)], fig_dir
    )
    assert Path(scatter["png"]).exists()
