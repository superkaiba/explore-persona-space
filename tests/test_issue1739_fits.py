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

    real_solve = arms._solve_ridge_groups

    def corrupted(jobs, **kw):
        return {
            key: {name: pred + 0.5 for name, pred in evs.items()}
            for key, evs in real_solve(jobs, **kw).items()
        }

    monkeypatch.setattr(arms, "_solve_ridge_groups", corrupted)
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
        f"ctx{i:02d}_k{k:02d}": ("fabricated" if (i + k) % 3 == 0 else "correct")
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


# ---------------------------------------------------------------------------
# round-3: distribution-shift transfer leg (M-A)
# ---------------------------------------------------------------------------


def _transfer_fixture(n_tr=24, n_ev=12, n_layers=2, d=6, seed=3):
    rng = np.random.default_rng(seed)
    rb = rng.normal(size=(n_layers, d))
    z_tr = rng.normal(size=(n_layers, n_tr, d))
    z_ev = rng.normal(size=(n_layers, n_ev, d))
    dv_tr = np.clip(50 + 20 * np.einsum("lnd,ld->n", z_tr, rb) / n_layers, 0, 100)
    dv_ev = np.clip(50 + 20 * np.einsum("lnd,ld->n", z_ev, rb) / n_layers, 0, 100)
    groups = [f"g{i % 6}" for i in range(n_tr)]
    data = arms.CellData(z_ctx=z_tr, dv=dv_tr, rb=rb, layers=(0, 1))
    cell = fits.realize_budget_cell(groups, budget_l=n_tr, draw=0, seed=0)
    return data, cell, z_ev, dv_ev, rng


def test_transfer_cell_fits_on_train_never_on_eval_dv():
    """run_transfer_cell: parameter-free arms score eval rows identically to a
    direct projection; the fitted arm-4 predictor changes with TRAIN dv but is
    BIT-INVARIANT to eval dv (the frozen-predictor / no-eval-leakage pin)."""
    data, cell, z_ev, dv_ev, rng = _transfer_fixture()
    want = ["arm1_ctx_e1", "arm4_ridge_ctx"]
    s1, sk1 = arms.run_transfer_cell(data, cell, z_ev, dv_ev, arms=want)
    assert not sk1, sk1
    assert s1["arm1_ctx_e1"].shape == (2, z_ev.shape[1])
    np.testing.assert_allclose(s1["arm1_ctx_e1"], np.einsum("lnd,ld->ln", z_ev, data.rb))

    # eval-DV perturbation: transfer scores must be IDENTICAL (never fit on eval DV)
    s2, _ = arms.run_transfer_cell(
        data, cell, z_ev, dv_ev + rng.normal(scale=10, size=dv_ev.shape), arms=want
    )
    np.testing.assert_array_equal(s1["arm4_ridge_ctx"], s2["arm4_ridge_ctx"])

    # train-DV perturbation: the fitted predictor MUST move (fit-on-train evidence)
    data_p = arms.CellData(
        z_ctx=data.z_ctx,
        dv=np.clip(data.dv + rng.normal(scale=25, size=data.dv.shape), 0, 100),
        rb=data.rb,
        layers=data.layers,
    )
    s3, _ = arms.run_transfer_cell(data_p, cell, z_ev, dv_ev, arms=want)
    assert not np.allclose(s1["arm4_ridge_ctx"], s3["arm4_ridge_ctx"])


def test_evaluate_transfer_frozen_reuse_and_min_n_gate():
    """evaluate_transfer scores at the TRAIN-frozen layer and records (never
    silently drops) a rung below the min_n floor — the data-dependent-gate
    degenerate probe."""
    n_ev = 10
    dv_ev = np.linspace(0, 100, n_ev)
    rungs = np.asarray(["big"] * 8 + ["tiny"] * 2)
    sc = np.stack([np.random.default_rng(0).normal(size=n_ev), dv_ev.copy()])  # layer 1 == dv
    cell = fits.realize_budget_cell([f"g{i}" for i in range(12)], budget_l=12, draw=0, seed=0)
    rows, skips = arms.evaluate_transfer(
        {"arm1_ctx_e1": sc},
        dv_ev,
        rungs,
        {"arm1_ctx_e1": 1},
        provenance={"behavior": "syn", "eval_rung": "joined", "config": "config_a"},
        cell=cell,
        layers=(0, 1),
        n_boot=25,
        min_n=3,
    )
    assert len(rows) == 1 and rows[0]["eval_rung"] == "big"
    assert rows[0]["rho_frozen"] == pytest.approx(1.0)  # frozen layer 1 == dv exactly
    assert rows[0]["rung_kind"] == "eval_transfer" and rows[0]["n_eval"] == 8
    assert len(rows[0]["ci_frozen"]) == 2
    assert skips and skips[0]["eval_rung"] == "tiny" and skips[0]["reason"] == "min_n"
    # missing frozen entry -> arm-level skip, never a KeyError
    rows2, skips2 = arms.evaluate_transfer(
        {"armX": sc},
        dv_ev,
        rungs,
        {},
        provenance={},
        cell=cell,
        layers=(0, 1),
        n_boot=25,
        min_n=3,
    )
    assert not rows2 and "no train frozen layer" in skips2[0]["reason"]


def test_write_summary_extra_merges_transfer_rows(tmp_path):
    out = arms.write_summary(
        [],
        tmp_path / "s.json",
        meta={"mode": "syn"},
        extra={"transfer_rows": [{"arm": "a"}], "transfer_skips": []},
    )
    payload = json.loads(out.read_text())
    assert payload["transfer_rows"] == [{"arm": "a"}] and payload["transfer_skips"] == []


def test_render_summary_figures_transfer_ladder_and_composition(tmp_path):
    """The §6.5 ladder figure renders from transfer_rows (train rung ordered
    first) and the §4b composition figure renders from f_u-bearing arm rows."""
    from explore_persona_space.experiments.issue_1739 import figures

    def _arm_row(**kw):
        base = {
            "arm": "arm6_map_proj_e1",
            "family": "map",
            "rho_frozen": 0.5,
            "ci_frozen": [0.3, 0.7],
            "budget_l": 250,
            "draw": 0,
            "seed": 0,
            "eval_rung": "wildchat",
            "f_u": None,
            "f_l": None,
        }
        base.update(kw)
        return base

    summary = {
        "arm_rows": [
            _arm_row(),
            _arm_row(arm="arm1_ctx_e1", family="context"),
            _arm_row(f_u=0.0, f_l=0.0, rho_frozen=0.2),
            _arm_row(f_u=0.5, f_l=1.0, rho_frozen=0.4),
        ],
        "transfer_rows": [
            _arm_row(rung_kind="train_in_split"),
            _arm_row(eval_rung="hhrt", rung_kind="eval_transfer", rho_frozen=0.35),
            _arm_row(eval_rung="toxicchat", rung_kind="eval_transfer", rho_frozen=0.2),
        ],
    }
    paths = figures.render_summary_figures(summary, tmp_path)
    names = {Path(p).name for p in paths}
    assert "distribution_shift_ladder.png" in names, names
    assert "composition_factor.png" in names, names


def test_save_map_fp16_roundtrip_idempotent(tmp_path):
    cli = _load_fits_cli()
    rng = np.random.default_rng(5)
    n_layers, n, d = 2, 40, 6
    x = rng.normal(size=(n_layers, n, d))
    y = 0.6 * x + 0.1 * rng.normal(size=x.shape)
    mapfit = fits.fit_linear_map(x, y)
    out = cli._save_map(tmp_path, "context_end", "full", mapfit, [0, 1])
    assert out == tmp_path / "maps" / "context_end__ufull.npz"
    mt0 = out.stat().st_mtime_ns
    with np.load(out, allow_pickle=False) as z:
        assert z["w"].dtype == np.float16
        meta = json.loads(str(z["meta"]))
        assert meta["u_label"] == "full" and meta["variant"] == "context_end"
        assert "git_commit" in meta and "ts" in meta
        # fp16 W reproduces the frozen map's predictions within fp16 tolerance
        m2 = fits.MapFit(
            w=z["w"].astype(np.float64),
            x_mu=z["x_mu"].astype(np.float64),
            x_sd=z["x_sd"].astype(np.float64),
            y_mu=z["y_mu"].astype(np.float64),
            diagnostics={},
        )
    p_ref = fits.apply_map(x[:, :5], mapfit)
    p_np = fits.apply_map(x[:, :5], m2)
    scale = float(np.abs(p_ref).max()) or 1.0
    assert float(np.abs(p_ref - p_np).max()) / scale < 2e-3
    # idempotent: the second call SKIPS (no rewrite — behavior-shared file)
    assert cli._save_map(tmp_path, "context_end", "full", mapfit, [0, 1]) == out
    assert out.stat().st_mtime_ns == mt0


def test_compose_pilot_report_fence_and_abort():
    cli = _load_fits_cli()
    kw = dict(
        n_map_fits=6,
        map_fit_s=60.0,
        # one measured regime-shared group wall per budget (round-8 basis)
        unit_group_walls={250: 2.0, 2500: 4.0, 8000: 10.0},
        n_plain_groups={250: 90, 2500: 90, 8000: 90},
        n_compose_units={8000: 6, 250: 6},
        transfer_s=30.0,
        n_pilot_transfer_units=9,
        n_transfer_units=810,
        abort_mult=3.0,
        n_units=828,
    )
    # projected = (6*60 + 90*(2+4+10) + (6*10 + 6*2) + 810*(30/9))/3600
    #           = (360 + 1440 + 72 + 2700)/3600 = 1.27 h
    rep = cli.compose_pilot_report(plan_wall_h=0.67, **kw)
    assert rep["projected_wall_h"] == pytest.approx(4572.0 / 3600.0)
    assert rep["fence_wall_h"] == pytest.approx(2 * 4572.0 / 3600.0)
    assert rep["abort"] is False  # 1.27 < 3 x 0.67 = 2.01
    rep2 = cli.compose_pilot_report(plan_wall_h=0.4, **kw)
    assert rep2["abort"] is True  # 1.27 > 3 x 0.4 = 1.2
    assert rep["transfer_unit_s"] == pytest.approx(30.0 / 9.0)
    # a budget missing from the measured walls falls back to the max wall
    rep3 = cli.compose_pilot_report(plan_wall_h=0.67, **{**kw, "unit_group_walls": {8000: 10.0}})
    assert rep3["projected_wall_h"] > rep["projected_wall_h"]
    assert cli.PILOT_ABORT_RC == 7


def test_record_compose_skip_reraises_on_plain_rung():
    """Minor 1: the compose-skip except is scoped — a plain-rung load failure
    RE-RAISES (run failure), only a composition quota records a skip."""
    cli = _load_fits_cli()
    plain = cli.RunSpec(variant="context_end", regime="e1", u_size=None)
    comp = cli.RunSpec(variant="context_end", regime="e1", u_size=16, f_u=0.5, f_l=0.0)
    skips: list = []
    with pytest.raises(ValueError, match="boom"):
        cli._record_compose_skip(plain, ValueError("boom"), skips)
    assert skips == []
    cli._record_compose_skip(comp, ValueError("quota"), skips)
    assert len(skips) == 1 and skips[0]["reason"] == "quota"


# ---------------------------------------------------------------------------
# round-8 batched-engine equivalence gates (serial oracle vs grouped solver)
# ---------------------------------------------------------------------------

_GATE_LAMBDAS = (0.1, 1.0, 10.0)


def _serial_reference_ridge_scores(data, cell, *, lambdas=_GATE_LAMBDAS):
    """Pre-round-8 serial oracle: ONE single-target ridge call per
    (arm, layer, fold) — the exact primitive the retired per-slice pool
    dispatched (`ridge_layer_batched_auto`, unchanged) — plus the old arm-9/14
    alpha + residual assembly verbatim. Seeded + deterministic; the batched
    grouped solver must reproduce it within float tolerance
    (vectorize-many-cell-fits.md item 6)."""
    lam = np.asarray(lambdas, dtype=np.float64)
    idx, folds = cell.row_idx, cell.fold_ids
    n_l, n_folds = len(idx), cell.n_folds
    z = np.asarray(data.z_ctx[:, idx], dtype=np.float64)
    dv = np.asarray(data.dv[idx], dtype=np.float64)
    rb = np.asarray(data.rb, dtype=np.float64)
    za = np.asarray(data.z_ans[:, idx], dtype=np.float64)
    mp = fits.apply_map(z, data.mapfit)
    n_layers = z.shape[0]
    tr_masks, ev_masks = arms._fold_masks(folds, n_folds)
    tr_w = tr_masks.astype(np.float64)
    tr_w /= np.maximum(tr_w.sum(axis=1, keepdims=True), 1.0)
    ev_rows = [np.flatnonzero(ev_masks[f]) for f in range(n_folds)]
    tr_rows = [np.flatnonzero(tr_masks[f]) for f in range(n_folds)]

    def solve(xt, yt, xe):
        preds = fits.ridge_layer_batched_auto(xt[None], yt[None, :, None], xe[None], lambdas=lam)
        return preds[0][:, 0]

    def scatter(x_tr_src, x_ev_src, target_per_fold_layer=None):
        arr = np.full((x_tr_src.shape[0], n_l), np.nan)
        for f in range(n_folds):
            for li in range(x_tr_src.shape[0]):
                y = (
                    dv[tr_rows[f]]
                    if target_per_fold_layer is None
                    else target_per_fold_layer[f][li][tr_rows[f]]
                )
                arr[li, ev_rows[f]] = solve(x_tr_src[li][tr_rows[f]], y, x_ev_src[li][ev_rows[f]])
        return arr

    out = {
        "arm4_ridge_ctx": scatter(z, z),
        "arm7_map_ridge_pred": scatter(mp, mp),
        "arm8_map_ridge_true": scatter(za, mp),
        "arm12_oracle_reg": scatter(za, za),
        "arm15_text_only": scatter(
            np.asarray(data.text_emb[idx], dtype=np.float64)[None],
            np.asarray(data.text_emb[idx], dtype=np.float64)[None],
        ),
        "arm16_surface_feat": scatter(
            np.asarray(data.text_features[idx], dtype=np.float64)[None],
            np.asarray(data.text_features[idx], dtype=np.float64)[None],
        ),
    }
    # arm 9 / 14: the old closed-form L2-SP assembly, verbatim
    rng = np.random.default_rng([1739, 6, cell.seed])
    rb_shuf = np.stack([r[rng.permutation(r.shape[0])] for r in rb])
    for slug, rb_v in (("arm9_pretrain_ft", rb), ("arm14_shuffled_pt", rb_shuf)):
        s_dir = arms._proj(mp, rb_v)
        s_mu = np.einsum("fn,ln->lf", tr_w, s_dir)
        d_mu = tr_w @ dv
        cov = np.einsum(
            "fn,lfn->lf",
            tr_w,
            (s_dir[:, None, :] - s_mu[:, :, None]) * (dv[None, None, :] - d_mu[None, :, None]),
        )
        var = np.einsum("fn,lfn->lf", tr_w, (s_dir[:, None, :] - s_mu[:, :, None]) ** 2)
        alpha = np.where(var > 1e-30, cov / np.maximum(var, 1e-30), 0.0)
        resid_targets = [
            [dv - alpha[li, f] * s_dir[li] for li in range(n_layers)] for f in range(n_folds)
        ]
        resid = scatter(mp, mp, target_per_fold_layer=resid_targets)
        out[slug] = alpha[:, folds] * s_dir + resid
    return out


def test_run_cell_ridge_arms_match_serial_oracle():
    """Round-8 equivalence gate: the grouped shared-factorization solver
    reproduces the serial per-(arm, layer, fold) oracle on 2 small seeded
    synthetic cells, every ridge-family arm, within float tolerance."""
    data, groups = _synthetic_cell_data()
    ridge_arms = [
        "arm4_ridge_ctx",
        "arm7_map_ridge_pred",
        "arm8_map_ridge_true",
        "arm9_pretrain_ft",
        "arm12_oracle_reg",
        "arm14_shuffled_pt",
        "arm15_text_only",
        "arm16_surface_feat",
    ]
    for budget_l, draw, seed in ((24, 0, 0), (18, 1, 2)):
        cell = fits.realize_budget_cell(groups, budget_l=budget_l, draw=draw, seed=seed)
        scores, skipped = arms.run_cell(data, cell, arms=ridge_arms, lambdas=_GATE_LAMBDAS)
        assert not skipped, skipped
        ref = _serial_reference_ridge_scores(data, cell)
        for slug in ridge_arms:
            assert np.allclose(scores[slug], ref[slug], atol=1e-8, equal_nan=True), (
                slug,
                budget_l,
                float(np.nanmax(np.abs(scores[slug] - ref[slug]))),
            )


def test_run_cell_multi_matches_single_regime_runs():
    """Regime batching is output-identical: run_cell_multi over 2 rb variants
    (shared z/za/map by identity) equals the independent per-regime
    run_cell calls, arm by arm (MLP arm included — same seeds)."""
    import dataclasses as _dc

    data, groups = _synthetic_cell_data()
    rng = np.random.default_rng(77)
    data2 = _dc.replace(data, rb=np.asarray(data.rb) + 0.3 * rng.normal(size=data.rb.shape))
    cell = fits.realize_budget_cell(groups, budget_l=24, draw=0, seed=0)
    mlp_kw = {"max_epochs": 2, "hidden": 4}
    multi = arms.run_cell_multi([data, data2], cell, mlp_kwargs=mlp_kw)
    for d, (sc_multi, sk_multi) in zip((data, data2), multi, strict=True):
        sc_single, sk_single = arms.run_cell(d, cell, mlp_kwargs=mlp_kw)
        assert sk_multi == sk_single
        assert set(sc_multi) == set(sc_single)
        for slug in sc_single:
            assert np.allclose(sc_multi[slug], sc_single[slug], atol=1e-9, equal_nan=True), slug


def test_bootstrap_counting_ranks_bitexact_and_headline_reuse():
    """The counting-sort bootstrap ranks are BIT-IDENTICAL to ranking the
    drawn values directly (tie-heavy fixture), and evaluate_cell's
    draws-reuse headline equals the old explicit re-bootstrap."""
    rng = np.random.default_rng(3)
    n = 40
    scores = np.round(rng.normal(size=(3, n)) * 3)  # tie-heavy
    dv = rng.integers(0, 6, size=n).astype(float)
    idx = arms.make_bootstrap_idx(n, n_boot=25, seed=1)
    new = arms.bootstrap_rhos(scores, dv, idx, chunk_draws=7)
    ref = np.empty((3, 25))
    for lo in range(0, 25, 7):
        sl = idx[lo : lo + 7]
        ref[:, lo : lo + sl.shape[0]] = arms._pearson_rows(
            arms.rank_rows(scores[:, sl]), arms.rank_rows(dv[sl])[None]
        )
    assert np.array_equal(new, ref)
    # headline draws-reuse: frozen-row delta draws == re-bootstrapped pair rows
    da = arms.bootstrap_rhos(scores, dv, idx)
    fa = int(np.nanargmax(arms.spearman_rows(scores, dv)))
    pair = arms.bootstrap_rhos(np.stack([scores[fa], scores[0]]), dv, idx)
    assert np.array_equal(da[fa], pair[0]) and np.array_equal(da[0], pair[1])


def test_ridge_gcv_per_target_matches_auto():
    """Per-target GCV over one shared factorization == looping the unchanged
    single-target auto helper, BOTH branches (primal ntr>d, dual ntr<=d),
    multiple eval matrices."""
    rng = np.random.default_rng(5)
    for ntr, d in ((12, 5), (6, 9)):
        x = rng.normal(size=(2, ntr, d))
        y = rng.normal(size=(2, ntr, 3))
        e1 = rng.normal(size=(2, 4, d))
        e2 = rng.normal(size=(2, 3, d))
        preds = fits.ridge_gcv_predict_per_target(x, y, [e1, e2], lambdas=_GATE_LAMBDAS)
        for t in range(3):
            for ei, ev in enumerate((e1, e2)):
                ref = fits.ridge_layer_batched_auto(
                    x, y[:, :, t : t + 1], ev, lambdas=np.asarray(_GATE_LAMBDAS)
                )
                assert np.allclose(preds[ei][:, :, t : t + 1], ref, atol=1e-9), (ntr, d, t, ei)


def test_transfer_ridge_folds_skip_is_output_identical():
    """ridge_folds=(0,) skips ONLY the discarded reverse-fold fit: the
    returned eval-block transfer scores are identical for every arm."""
    data, groups = _synthetic_cell_data()
    rng = np.random.default_rng(9)
    z_ev = rng.normal(size=(3, 7, 8))
    za_ev = rng.normal(size=(3, 7, 8))
    dv_ev = rng.uniform(0, 100, size=7)
    cell = fits.realize_budget_cell(groups, budget_l=20, draw=0, seed=1)
    full, sk_full = arms.run_transfer_cell(data, cell, z_ev, dv_ev, za_ev=za_ev)
    skip, sk_skip = arms.run_transfer_cell(data, cell, z_ev, dv_ev, za_ev=za_ev, ridge_folds=(0,))
    assert sk_full == sk_skip and set(full) == set(skip)
    for slug in full:
        assert np.allclose(full[slug], skip[slug], atol=0, equal_nan=True), slug


def test_run_grid_multi_partial_resume(tmp_path):
    """A regime slice already on disk resumes (records loaded, not recomputed)
    while the pending sibling regime computes fresh — per-regime unit keys."""
    import dataclasses as _dc

    data, groups = _synthetic_cell_data()
    data2 = _dc.replace(data, rb=np.asarray(data.rb) * 0.5)
    kw = dict(
        budgets=[18],
        draws=[0],
        seeds=[0, 1],
        out_dir=tmp_path,
        arms=["arm1_ctx_e1", "arm4_ridge_ctx", "arm6_map_proj_e1", "arm2_ctx_native"],
        n_boot=20,
        n_perm=20,
    )
    first = arms.run_grid(data, groups, provenance={"regime": "e1"}, **kw)
    both = arms.run_grid_multi([data, data2], [{"regime": "e1"}, {"regime": "e2"}], groups, **kw)
    assert [r["unit_key"] for r in both[0]] == [r["unit_key"] for r in first]
    assert json.dumps(both[0], sort_keys=True) == json.dumps(first, sort_keys=True)
    assert len(both[1]) == 2 and all("e2" in r["unit_key"] for r in both[1])


# ---------------------------------------------------------------------------
# round-8 tiny-real CLI e2e: pilot -> full real grid -> transfer -> resume
# ---------------------------------------------------------------------------


def _write_tiny_real_inputs(tmp_path, *, d=16, layers=(0, 1, 2), k_rollouts=3, seed=1739):
    """Complete tiny-real input set for the REAL-mode fits CLI: labeled store
    (per-rollout rows, both splits), U store (fit-pool mask), E1 store
    (pos/neg sides), DV labeling.json (splits + groups + rungs + per-rollout
    scores), and the arms-15/16 features npz."""
    rng = np.random.default_rng(seed)
    train_ctx = [f"tr{i:02d}" for i in range(14)]
    eval_ctx = [f"ev{i:02d}" for i in range(6)]

    def write_store(root, meta_rows):
        root.mkdir(parents=True, exist_ok=True)
        n = len(meta_rows)
        for kind in ("prefix_end", "context_end", "t1"):
            for ly in layers:
                np.save(root / f"{kind}_L{ly:02d}.npy", rng.normal(size=(n, d)).astype(np.float16))
        with (root / "row_index.jsonl").open("w", encoding="utf-8") as fh:
            for r in meta_rows:
                fh.write(json.dumps(r) + "\n")

    labeled_rows = [
        {"context_id": c, "rollout_k": k}
        for c in (*train_ctx, *eval_ctx)
        for k in range(k_rollouts)
    ]
    write_store(tmp_path / "labeled", labeled_rows)
    write_store(
        tmp_path / "ustore",
        [{"context_id": f"u{i:02d}", "is_eval_only": i >= 30} for i in range(40)],
    )
    write_store(
        tmp_path / "e1",
        [{"context_id": f"p{i:02d}", "side": "pos" if i < 6 else "neg"} for i in range(12)],
    )
    dv_rows = []
    for i, c in enumerate(train_ctx):
        dv_rows.append(
            {
                "context_id": c,
                "dv": float(rng.uniform(5, 95)),
                "split": "train",
                "group_key": f"g{i % 7}",
                "rung": "wildchat",
                "per_rollout_scores": {"k0": 20.0 + i, "k1": 78.0 - i, "k2": 50.0},
            }
        )
    for i, c in enumerate(eval_ctx):
        dv_rows.append(
            {
                "context_id": c,
                "dv": float(rng.uniform(5, 95)),
                "split": "eval",
                "group_key": f"ge{i % 3}",
                "rung": "lmsys" if i % 2 == 0 else "prism",
                "per_rollout_scores": {"k0": 30.0 + i, "k1": 70.0 - i, "k2": 45.0},
            }
        )
    dv_json = tmp_path / "labeling.json"
    dv_json.write_text(json.dumps({"rows": dv_rows}))
    all_ctx = [*train_ctx, *eval_ctx]
    feats = tmp_path / "features.npz"
    np.savez(
        feats,
        context_ids=np.asarray(all_ctx),
        emb=rng.normal(size=(len(all_ctx), 5)),
        features=rng.normal(size=(len(all_ctx), 3)),
    )
    return dv_json, feats


def test_real_mode_pilot_full_and_resume_e2e(tmp_path, capsys):
    """Tiny-real e2e of the restructured CLI: --pilot (per-budget unit-groups,
    all regimes) -> full real grid (regime groups + composition + transfer)
    -> idempotent resume. Exercises _run_real / _run_pilot /
    _run_transfer_for_group + the ans_rows memory-scoping free."""
    cli = _load_fits_cli()
    dv_json, feats = _write_tiny_real_inputs(tmp_path)
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
        "--text-emb",
        str(feats),
        "--text-features",
        str(feats),
        "--device",
        "cpu",
        "--config",
        "config_a",
        "--transfer",
        "--transfer-min-n",
        "2",
        "--regimes",
        "e1",
        "e2",
        "e2p",
        "--u-sizes",
        "8",
        "full",
        "--budgets",
        "6",
        "10",
        "--draws",
        "0",
        "1",
        "--seeds",
        "0",
        "--layers",
        "0",
        "1",
        "2",
        "--mlp-epochs",
        "2",
        "--n-boot",
        "20",
        "--n-perm",
        "20",
        "--compose",
        "--compose-u-size",
        "8",
    ]
    assert cli.main([*argv, "--pilot", "--plan-wall-h", "100"]) == 0
    report = json.loads((out_root / "pilot_report.json").read_text())
    assert report["abort"] is False
    assert set(report["unit_group_walls_s"]) == {"6", "10"}
    assert report["n_plain_groups"] == {"6": 8, "10": 8}
    out1 = capsys.readouterr().out
    assert "freed per-rollout answer rows" in out1
    assert "[fits] batched slice:" in out1

    assert cli.main(argv) == 0
    summary = json.loads((out_root / "arm_results" / "all_arms_spearman.json").read_text())
    # plain: 2 variants x 2 U x 3 regimes x (2 budgets x 2 draws x 1 seed) = 48
    # compose: dedup{(0,0),(0.5,0),(0.5,1)} x 2 anchors x 2 variants = 12
    assert summary["n_cells"] == 60
    assert summary["transfer_rows"], "transfer ladder rows must land in the summary"
    cells = (out_root / "arm_results" / "percell" / "cells.jsonl").read_text()
    n_lines = sum(1 for ln in cells.split("\n") if ln.strip())
    assert n_lines == 60
    tlines = (out_root / "arm_results" / "percell" / "transfer.jsonl").read_text()
    # plain (unit x regime) transfer evaluations: 2 U-keys... transfer runs per
    # plain group: 2 variants x 2 U x (2 budgets x 2 draws) x 3 regimes = 48
    assert sum(1 for ln in tlines.split("\n") if ln.strip()) == 48
    # frozen maps persisted per (variant, plain U label); regime dirs per regime
    maps = sorted(p.name for p in (tmp_path / "tensors" / "maps").glob("*.npz"))
    assert len(maps) == 4, maps
    for regime in ("e1", "e2", "e2p"):
        assert (tmp_path / "tensors" / f"r_b_{regime}" / "evil.npz").exists()

    # idempotent resume: nothing recomputed, no new rows
    assert cli.main(argv) == 0
    cells2 = (out_root / "arm_results" / "percell" / "cells.jsonl").read_text()
    assert sum(1 for ln in cells2.split("\n") if ln.strip()) == 60
    out3 = capsys.readouterr().out
    assert "SKIP (resume)" in out3


def test_ridge_device_fanout_threaded_branch_matches_serial(monkeypatch):
    """Bare 'cuda' fans ridge jobs across all GPUs; pinned devices don't.
    The THREADED pool branch (exercised here with two 'cpu' workers) must
    reproduce the serial branch exactly."""
    assert arms._ridge_devices("cpu") == ["cpu"]
    assert arms._ridge_devices("cuda:1") == ["cuda:1"]
    data, groups = _synthetic_cell_data()
    cell = fits.realize_budget_cell(groups, budget_l=20, draw=0, seed=0)
    ridge_arms = ["arm4_ridge_ctx", "arm7_map_ridge_pred", "arm12_oracle_reg"]
    serial, _ = arms.run_cell(data, cell, arms=ridge_arms, lambdas=_GATE_LAMBDAS)
    monkeypatch.setattr(arms, "_ridge_devices", lambda device: ["cpu", "cpu"])
    threaded, _ = arms.run_cell(data, cell, arms=ridge_arms, lambdas=_GATE_LAMBDAS)
    for slug in ridge_arms:
        assert np.allclose(serial[slug], threaded[slug], atol=1e-12, equal_nan=True), slug


# ---------------------------------------------------------------------------
# r10 crash fix: arm-10 batched normal-eq solve — degenerate-slice fallback
# ---------------------------------------------------------------------------


def _singular_normal_eq_stack():
    """(4, 3, 3) normal-eq stack whose slice 2 is EXACTLY singular.

    Slice 2 is the Gram of a constant-column design (col1 = 2 * col0 — the
    production degeneracy shape: an exactly-collinear arm-10 feature),
    integer-exact so LAPACK gesv hits an exact zero pivot deterministically.
    """
    rng = np.random.default_rng(1739)
    a = rng.normal(size=(4, 3, 3))
    ata = a @ a.transpose(0, 2, 1) + 3.0 * np.stack([np.eye(3)] * 4)  # healthy SPD slices
    design = np.array([[1.0, 2.0, 0.0], [1.0, 2.0, 1.0], [1.0, 2.0, 3.0], [1.0, 2.0, -1.0]])
    ata[2] = design.T @ design  # rank 2 — exactly singular
    atb = rng.normal(size=(4, 3, 1))
    return ata, atb


def test_solve_stacked_normal_eqs_pinv_fallback_flags_and_matches():
    """One singular slice kills the whole batched np.linalg.solve (the #1739
    sycophancy-lane crash — asserted below as the pre-fix arm-10 behavior);
    the robust solver completes, flags EXACTLY the singular slice, matches
    the per-slice solve on healthy slices and pinv on the flagged one."""
    ata, atb = _singular_normal_eq_stack()
    with pytest.raises(np.linalg.LinAlgError):  # pre-fix arm-10 call == the crash
        np.linalg.solve(ata, atb)
    beta, degenerate = arms._solve_stacked_normal_eqs(ata, atb)
    assert degenerate == [2]
    for li in (0, 1, 3):
        assert np.allclose(beta[li], np.linalg.solve(ata[li], atb[li]), atol=1e-12), li
    assert np.allclose(beta[2], np.linalg.pinv(ata[2]) @ atb[2], atol=1e-10)
    # healthy stack: identical to the batched solve, nothing flagged
    healthy = ata.copy()
    healthy[2] = np.eye(3) * 2.0
    beta_h, deg_h = arms._solve_stacked_normal_eqs(healthy, atb)
    assert deg_h == [] and np.array_equal(beta_h, np.linalg.solve(healthy, atb))


def test_run_grid_multi_flags_degenerate_ols_cell(tmp_path, monkeypatch):
    """A degenerate arm-10 cell no longer kills the grid: the REAL
    _solve_stacked_normal_eqs body runs on a genuinely singular stack
    (injected into the ata INPUT at the real arm-10 call site — the solver
    itself is never stubbed), the unit completes, and the per-cell record +
    cells.jsonl row carry degenerate_ols=True with the fold->layer detail."""
    data, groups = _synthetic_cell_data()
    calls: list[int] = []
    real = arms._solve_stacked_normal_eqs
    design = np.array([[1.0, 2.0, 0.0], [1.0, 2.0, 1.0], [1.0, 2.0, 3.0], [1.0, 2.0, -1.0]])

    def corrupt_then_solve(ata, atb):
        if not calls:  # first fold only: layer-1 slice made exactly singular
            ata = ata.copy()
            ata[1] = design.T @ design
        calls.append(1)
        return real(ata, atb)

    monkeypatch.setattr(arms, "_solve_stacked_normal_eqs", corrupt_then_solve)
    kw = dict(
        budgets=[18],
        draws=[0],
        seeds=[0],
        out_dir=tmp_path,
        arms=["arm4_ridge_ctx", "arm6_map_proj_e1", "arm10_stacked"],
        n_boot=20,
        n_perm=20,
    )
    recs = arms.run_grid(data, groups, provenance={"regime": "e1"}, **kw)
    assert calls, "arm10 never dispatched through the robust solver"
    (rec,) = recs
    assert rec["degenerate_ols"] is True
    detail = rec["degenerate_ols_detail"]["arm10_stacked"]
    assert detail["degenerate_ols"] is True and detail["fold_layers"] == {"0": [1]}
    row = json.loads((tmp_path / "percell" / "cells.jsonl").read_text().strip())
    assert row["degenerate_ols"] is True
    # a clean sibling run records the flag as False, with no detail key
    recs2 = arms.run_grid(
        data, groups, provenance={"regime": "e2"}, **{**kw, "out_dir": tmp_path / "clean"}
    )
    assert recs2[0]["degenerate_ols"] is False and "degenerate_ols_detail" not in recs2[0]


# ---------------------------------------------------------------------------
# new-arm-round: arm-18 KRR helper parity + fc (--rb-point) direction builder
# ---------------------------------------------------------------------------


def test_krr_scalar_fold_predict_matches_exact_dual_krr():
    """At m_centers >= n_inner, Nystrom features over ALL inner rows span the
    kernel space, so the feature ridge IS exact dual KRR:
    pred = K_ev,inner (K_inner + lam I)^-1 (y - ymu) + ymu. The batched helper
    must reproduce that closed form (single (gamma, lambda) pair, so the
    inner-val selection is trivially that pair)."""
    rng = np.random.default_rng(0)
    ly, ntr, nev, d = 2, 40, 7, 6
    x = rng.normal(size=(ly, ntr, d))
    xe = rng.normal(size=(ly, nev, d))
    y = np.sin(x[0, :, 0]) + 0.1 * rng.normal(size=ntr)
    lam = 0.1
    diag: dict = {}
    pred = fits.krr_scalar_fold_predict(
        x,
        y,
        xe,
        seed=3,
        device="cpu",
        m_centers=10_000,  # >= n_inner -> exact-KRR regime
        gamma_mult=(1.0,),
        lambdas=(lam,),
        diag_out=diag,
    )
    assert pred.shape == (ly, nev)
    # Replicate the helper's own inner/val split ([1739, 5, seed] key family).
    p = np.random.default_rng([1739, 5, 3]).permutation(ntr)
    inner = p[max(2, round(0.1 * ntr)) :]
    ymu = y[inner].mean()
    yc = y[inner] - ymu
    for li in range(ly):
        sel = diag["per_layer"][li]
        assert sel["lambda"] == lam and sel["gamma"] > 0
        d_ii = ((x[li, inner][:, None] - x[li, inner][None]) ** 2).sum(-1)
        d_ei = ((xe[li][:, None] - x[li, inner][None]) ** 2).sum(-1)
        k_ii = np.exp(-sel["gamma"] * d_ii)
        alpha = np.linalg.solve(k_ii + lam * np.eye(len(inner)), yc)
        ref = np.exp(-sel["gamma"] * d_ei) @ alpha + ymu
        np.testing.assert_allclose(pred[li], ref, rtol=1e-5, atol=1e-6)


def test_krr_scalar_fold_predict_selects_on_inner_val():
    """The (gamma, lambda) grid is selected on the INNER-val MSE (never the
    eval rows): with an absurd ridge in the grid the helper must pick the
    sane one, and the recorded selection must come from the grid."""
    rng = np.random.default_rng(1)
    ly, ntr, nev, d = 1, 60, 5, 4
    x = rng.normal(size=(ly, ntr, d))
    xe = rng.normal(size=(ly, nev, d))
    y = x[0, :, 0] ** 2 + 0.05 * rng.normal(size=ntr)
    diag: dict = {}
    fits.krr_scalar_fold_predict(x, y, xe, seed=0, device="cpu", lambdas=(1e-1, 1e8), diag_out=diag)
    assert diag["per_layer"][0]["lambda"] == pytest.approx(1e-1)
    assert np.isfinite(diag["per_layer"][0]["val_mse"])


def test_krr_degenerate_gamma_layer_is_recorded_not_fatal():
    """One duplicate-dominated layer (median sq distance 0) yields NaN preds +
    a diag flag for THAT layer only — never a grid-killing assert (code-review
    r1 Minor 5); ALL layers degenerate fails loud."""
    rng = np.random.default_rng(2)
    ly, ntr, nev, d = 2, 30, 4, 3
    x = rng.normal(size=(ly, ntr, d))
    x[1] = 1.0  # layer 1: all rows identical -> median sq distance exactly 0
    xe = rng.normal(size=(ly, nev, d))
    y = x[0, :, 0] + 0.1 * rng.normal(size=ntr)
    diag: dict = {}
    pred = fits.krr_scalar_fold_predict(x, y, xe, seed=0, device="cpu", diag_out=diag)
    assert np.isfinite(pred[0]).all(), "healthy layer must stay finite"
    assert np.isnan(pred[1]).all(), "degenerate layer must be NaN (recorded skip)"
    assert diag["per_layer"][1].get("degenerate_gamma") is True
    assert "degenerate_gamma" not in diag["per_layer"][0]
    assert diag["n_degenerate_gamma_layers"] == 1
    x_all = np.ones((ly, ntr, d))
    with pytest.raises(ValueError, match="ALL"):
        fits.krr_scalar_fold_predict(x_all, y, xe, seed=0, device="cpu")


def _fc_labeled_table(cli, n_ctx=3, k=2, ly=2, d=4, seed=0):
    """Tiny LabeledTable whose ans_rows carry the CONTEXT_END per-rollout rows
    (what _load_labeled loads under rollout_rows_kind='context_end')."""
    rng = np.random.default_rng(seed)
    per_rollout = np.array([[90.0, 10.0], [80.0, 20.0], [70.0, 30.0]])[:n_ctx, :k]
    n_rows = n_ctx * k
    ans_rows = {li: rng.normal(size=(n_rows, d)) for li in range(ly)}
    return cli.LabeledTable(
        z_by_variant={},
        z_ans=rng.normal(size=(ly, n_ctx, d)),
        dv=per_rollout.mean(axis=1),
        groups=[f"g{i}" for i in range(n_ctx)],
        per_rollout=per_rollout,
        ctx_order=[f"c{i}" for i in range(n_ctx)],
        rungs=["train"],
        ans_rows=ans_rows,
        ans_row_ctx=np.repeat(np.arange(n_ctx), k),
        ans_row_k=np.tile(np.arange(k), n_ctx),
    )


def test_extract_rb_fc_applies_split_weights_to_the_loaded_rows():
    """e2p_fc = the SAME pooled matched_pair_split_weights row weights applied
    to the rows _load_labeled loaded (context_end under --rb-point
    context_end) — position is the ONLY change (plan v9 item 1; matched-e2_fc
    is structurally dropped, see the refusal tests below)."""
    import argparse

    from explore_persona_space.experiments.issue_1739.constants import E2_SPREAD_MIN

    cli = _load_fits_cli()
    tbl = _fc_labeled_table(cli)
    args = argparse.Namespace(behavior="toy", e1_store=None, rb_point="context_end")
    rb = cli._extract_rb("e2p_fc", args, tbl, [0, 1], 4)
    w_hi, w_lo, _n = fits.matched_pair_split_weights(
        tbl.per_rollout, spread_min=E2_SPREAD_MIN, pooled=True
    )
    w_row = (w_hi - w_lo)[tbl.ans_row_ctx, tbl.ans_row_k]
    for li in (0, 1):
        np.testing.assert_allclose(rb[li], w_row @ tbl.ans_rows[li])


def test_extract_rb_refuses_matched_e2_under_fc():
    """Plan v9 structural restriction: matched-e2_fc is REFUSED structurally
    (the within-context hi/lo weights cancel exactly on context-level rows;
    K2's norm check is blind to the float residue)."""
    import argparse

    cli = _load_fits_cli()
    tbl = _fc_labeled_table(cli)
    args = argparse.Namespace(behavior="toy", e1_store=None, rb_point="context_end")
    with pytest.raises(SystemExit, match="structurally undefined"):
        cli._extract_rb("e2_fc", args, tbl, [0, 1], 4)


def test_parse_args_refuses_e2_regime_at_context_end():
    """Flag-level enforcement of the plan-v9 structural restriction: the CLI
    refuses --regimes e2 under --rb-point context_end (argparse error, rc 2);
    e1/e2p under context_end and e2 under t1 both parse."""
    cli = _load_fits_cli()
    with pytest.raises(SystemExit) as exc:
        cli._parse_args(["--rb-point", "context_end", "--regimes", "e1", "e2", "e2p"])
    assert exc.value.code == 2
    ok_fc = cli._parse_args(["--rb-point", "context_end", "--regimes", "e1", "e2p"])
    assert ok_fc.regimes == ["e1", "e2p"]
    ok_t1 = cli._parse_args(["--regimes", "e1", "e2", "e2p"])
    assert ok_t1.regimes == ["e1", "e2", "e2p"]


def test_extract_rb_fc_k2_halts_on_a_degenerate_direction():
    """K2 (plan v8 par.7): a zero-norm fc direction HALTS with a named report,
    never a fabricated direction. The committed t1 path is untouched."""
    import argparse

    cli = _load_fits_cli()
    tbl = _fc_labeled_table(cli)
    for li in tbl.ans_rows:
        tbl.ans_rows[li] = np.zeros_like(tbl.ans_rows[li])
    args = argparse.Namespace(behavior="toy", e1_store=None, rb_point="context_end")
    with pytest.raises(SystemExit, match="K2 HALT"):
        cli._extract_rb("e2p_fc", args, tbl, [0, 1], 4)


def test_rb_point_and_fixed_coordinate_cli_defaults():
    """Defaults keep committed behavior byte-identical: rb_point=t1,
    fixed_coordinate absent."""
    cli = _load_fits_cli()
    args = cli._parse_args(["--behavior", "toy"])
    assert args.rb_point == "t1"
    assert args.fixed_coordinate is None
    args_fc = cli._parse_args(["--rb-point", "context_end", "--fixed-coordinate", "u=full"])
    assert args_fc.rb_point == "context_end"
    assert args_fc.fixed_coordinate == "u=full"


# ---------------------------------------------------------------------------
# crash-fix r3: chunked/aliased memory paths are BIT-IDENTICAL + the RSS guard
# ---------------------------------------------------------------------------


def _fit_whitening_dense_reference(x_u, *, seed=0, device="cpu", layer_chunk=8):
    """The PRE-r3 fit_whitening body (upfront whole-array fp64 cast) — the
    parity oracle for the chunked-cast rewrite."""
    import torch

    from explore_persona_space.experiments.issue_1739.constants import (
        WHITEN_HOLDOUT_FRAC,
        WHITEN_SHRINKAGE_GRID,
    )

    gammas = WHITEN_SHRINKAGE_GRID
    x = np.asarray(x_u, dtype=np.float64)
    n_layers, n, d = x.shape
    rng = np.random.default_rng([1739, 3, int(seed)])
    perm = rng.permutation(n)
    n_hold = round(WHITEN_HOLDOUT_FRAC * n) if n >= 5 else 0
    hold, tr = perm[:n_hold], perm[n_hold:]
    dev = torch.device(device)
    mu = np.empty((n_layers, d))
    w_out = np.empty((n_layers, d, d))
    gamma_out = np.empty(n_layers)
    for lo in range(0, n_layers, layer_chunk):
        sl = slice(lo, min(lo + layer_chunk, n_layers))
        xt = torch.as_tensor(x[sl][:, tr], device=dev)
        m = xt.mean(dim=1, keepdim=True)
        xc = xt - m
        cov = xc.transpose(1, 2) @ xc / max(len(tr), 1)
        evals, evecs = fits._eigh_robust(cov)
        evals = torch.clamp(evals, min=0.0)
        tr_mean = evals.mean(dim=1, keepdim=True)
        if n_hold:
            xh = torch.as_tensor(x[sl][:, hold], device=dev) - m
            diag_hold = ((xh @ evecs) ** 2).mean(dim=1)
        else:
            diag_hold = evals
        nlls = []
        for g in gammas:
            lam = (1.0 - g) * evals + g * tr_mean
            nlls.append(torch.log(lam).sum(dim=1) + (diag_hold / lam).sum(dim=1))
        gi = torch.stack(nlls, dim=1).argmin(dim=1)
        g_best = torch.as_tensor([float(gammas[int(i)]) for i in gi], device=dev)
        lam_best = (1.0 - g_best[:, None]) * evals + g_best[:, None] * tr_mean
        inv_sqrt = evecs @ (lam_best.clamp(min=1e-12).rsqrt()[:, :, None] * evecs.transpose(1, 2))
        mu[sl] = m.squeeze(1).cpu().numpy()
        w_out[sl] = inv_sqrt.cpu().numpy()
        gamma_out[sl] = g_best.cpu().numpy()
    return fits.Whitening(mu=mu, w=w_out, gamma=gamma_out)


@pytest.mark.parametrize("in_dtype", [np.float16, np.float32, np.float64])
def test_fit_whitening_chunked_matches_dense_reference(in_dtype):
    """r3 per-chunk fp64 cast == the old upfront whole-array cast, bitwise."""
    x_u = RNG.normal(size=(3, 40, 6)).astype(in_dtype)
    got = fits.fit_whitening(x_u, seed=3, layer_chunk=2)
    ref = _fit_whitening_dense_reference(x_u, seed=3, layer_chunk=2)
    assert np.array_equal(got.mu, ref.mu)
    assert np.array_equal(got.w, ref.w)
    assert np.array_equal(got.gamma, ref.gamma)


@pytest.mark.parametrize("in_dtype", [np.float16, np.float32, np.float64])
def test_apply_whitening_chunked_matches_dense(in_dtype):
    """r3 per-layer apply == the old batched `(x - mu) @ w` expression, bitwise."""
    x_u = RNG.normal(size=(3, 25, 6)).astype(np.float64)
    wh = fits.fit_whitening(x_u, seed=1)
    x = RNG.normal(size=(3, 11, 6)).astype(in_dtype)
    got = fits.apply_whitening(x, wh)
    ref = (np.asarray(x, dtype=np.float64) - wh.mu[:, None, :]) @ wh.w
    assert got.dtype == np.float64
    assert np.array_equal(got, ref)


def test_apply_map_chunked_matches_dense():
    """r3 per-layer apply_map == the old batched standardize-matmul, bitwise."""
    rng = np.random.default_rng(7)
    x_u = rng.normal(size=(3, 40, 6))
    y_u = 0.5 * x_u + rng.normal(size=x_u.shape)
    m = fits.fit_linear_map(x_u, y_u)
    x = rng.normal(size=(3, 9, 6))
    got = fits.apply_map(x, m)
    ref = ((np.asarray(x, dtype=np.float64) - m.x_mu) / m.x_sd) @ m.w + m.y_mu
    assert np.array_equal(got, ref)
    # the shuffled-weight override path chunks identically
    w_shuf = fits.shuffled_map_weights(m.w, seed=0)
    got_s = fits.apply_map(x, m, w=w_shuf)
    ref_s = ((np.asarray(x, dtype=np.float64) - m.x_mu) / m.x_sd) @ w_shuf + m.y_mu
    assert np.array_equal(got_s, ref_s)


def test_take_rows_view_and_copy():
    """_take_rows: contiguous arange -> VIEW; anything else -> the fancy copy."""
    arr = RNG.normal(size=(2, 10, 3))
    rows = np.arange(3, 8)
    view = arms._take_rows(arr, rows)
    assert view.base is arr and np.array_equal(view, arr[:, rows])
    scattered = np.array([1, 4, 5])
    copy = arms._take_rows(arr, scattered)
    assert copy.base is not arr and np.array_equal(copy, arr[:, scattered])
    single = arms._take_rows(arr, np.array([4]))
    assert np.array_equal(single, arr[:, [4]])


def test_concat_train_eval_matches_concatenate():
    full = RNG.normal(size=(2, 12, 4))
    ev = RNG.normal(size=(2, 5, 4))
    idx = np.array([7, 2, 9, 0])
    got = arms._concat_train_eval(full, idx, ev)
    assert np.array_equal(got, np.concatenate([full[:, idx], ev], axis=1))


def test_run_cell_multi_alias_identity_matches_copy_and_never_mutates():
    """The transfer leg's identity row set: aliased z/za give IDENTICAL scores
    to an explicit-copy comb, and the shared arrays are never mutated."""
    data, _groups = _synthetic_cell_data(n=24, seed=5)
    n = data.z_ctx.shape[1]
    cell = arms.BudgetCell(
        row_idx=np.arange(n),
        fold_ids=np.concatenate([np.ones(16, dtype=np.int64), np.zeros(8, dtype=np.int64)]),
        n_folds=2,
        budget_l=16,
        draw=0,
        seed=0,
        fold_scheme="transfer-train-vs-eval",
    )
    want = ["arm1_ctx_e1", "arm4_ridge_ctx", "arm6_map_proj_e1", "arm11_oracle_proj"]
    z_pristine, za_pristine = data.z_ctx.copy(), data.z_ans.copy()
    scores_alias, _ = arms.run_cell(data, cell, arms=want, ridge_folds=(0,))
    assert np.array_equal(data.z_ctx, z_pristine), "alias path mutated z_ctx"
    assert np.array_equal(data.z_ans, za_pristine), "alias path mutated z_ans"
    data_copy = arms.CellData(
        z_ctx=data.z_ctx.copy(),
        z_ans=data.z_ans.copy(),
        dv=data.dv,
        rb=data.rb,
        mapfit=data.mapfit,
        text_emb=data.text_emb,
        text_features=data.text_features,
        layers=data.layers,
    )
    scores_copy, _ = arms.run_cell(data_copy, cell, arms=want, ridge_folds=(0,))
    for slug in want:
        assert np.array_equal(scores_alias[slug], scores_copy[slug], equal_nan=True) or np.allclose(
            scores_alias[slug], scores_copy[slug], equal_nan=True, atol=0
        ), slug


def test_run_cell_multi_lazy_mp_skips_apply_map(monkeypatch):
    """mp is built ONLY when a map-consuming arm is requested (r3 memory fix);
    a non-map roster never calls apply_map, and skip semantics are unchanged."""
    data, groups = _synthetic_cell_data(n=24, seed=6)
    cell = fits.realize_budget_cell(groups, budget_l=18, draw=0, seed=0)
    calls = {"n": 0}
    real_apply = arms.apply_map

    def _counting_apply(*a, **kw):
        calls["n"] += 1
        return real_apply(*a, **kw)

    monkeypatch.setattr(arms, "apply_map", _counting_apply)
    scores, _skipped = arms.run_cell(
        data, cell, arms=["arm1_ctx_e1", "arm11_oracle_proj", "arm4_ridge_ctx"]
    )
    assert calls["n"] == 0, "non-map roster must not materialize mp"
    assert "arm1_ctx_e1" in scores and "arm4_ridge_ctx" in scores
    scores6, _ = arms.run_cell(data, cell, arms=["arm6_map_proj_e1"])
    assert calls["n"] == 1 and "arm6_map_proj_e1" in scores6
    # mapfit=None + map arm requested still records the "no mapfit" skip
    data_nomap = arms.CellData(
        z_ctx=data.z_ctx, z_ans=data.z_ans, dv=data.dv, rb=data.rb, layers=data.layers
    )
    _, skipped_nomap = arms.run_cell(data_nomap, cell, arms=["arm6_map_proj_e1"])
    assert skipped_nomap["arm6_map_proj_e1"] == "no mapfit"


def test_mem_guard_components_and_refusal(tmp_path, monkeypatch, capsys):
    from explore_persona_space.experiments.issue_1739 import mem_guard

    # component arithmetic: the documented full-U shape projects ~75 GiB extra
    comp = mem_guard.whitening_map_components(28, 18793, 3584, n_ctx=8000, n_ev=2666)
    total_gib = sum(comp.values()) / 2**30
    assert 60 < total_gib < 110, total_gib
    tc = mem_guard.transfer_components(
        28, 16000, 7188, 3584, list(arms.TRANSFER_ARMS_WIDE), has_map=True
    )
    assert 50 < sum(tc.values()) / 2**30 < 100
    # no-map / projection-only roster: no mp, no mlp, no ridge terms
    tc_min = mem_guard.transfer_components(28, 250, 100, 3584, ["arm1_ctx_e1"], has_map=True)
    assert set(tc_min) == {"comb_z_za"}
    # ok verdict on a tiny projection
    rec = mem_guard.check_phase("unit_ok", {"tiny": 1024}, out_root=tmp_path)
    assert rec["verdict"] == "ok"
    # forced refusal: projection larger than any real box
    with pytest.raises(mem_guard.MemGuardRefusal):
        mem_guard.check_phase("unit_refuse", {"huge": 10**15}, out_root=tmp_path)
    report = json.loads((tmp_path / "rss_guard_report.json").read_text())
    assert report["checks"][-1]["phase"] == "unit_refuse"
    assert report["checks"][-1]["verdict"] == "REFUSE"
    out = capsys.readouterr().out
    assert "[fits][rss-guard] phase=unit_refuse" in out and "verdict=REFUSE" in out
    # log-only kill switch
    monkeypatch.setenv("EPM_I1739_RSS_GUARD", "0")
    rec2 = mem_guard.check_phase("unit_logonly", {"huge": 10**15}, out_root=tmp_path)
    assert rec2["verdict"] == "over-log-only"
