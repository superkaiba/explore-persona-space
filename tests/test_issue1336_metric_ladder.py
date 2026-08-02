"""#1336 Unit C: Phase LAD metric-ladder battery pins.

Covers the plan v13 §4 binding requirements on synthetic fixtures (no
network, CPU-only): tier nesting (T1 rescues a pure context offset that T0
cannot), Procrustes orthogonality + direction-awareness (T5 rescues a
rotated-target pair that the scalar T4 cannot), T8 == the parent
comp_samefn_b2i chain (vs issue825_map_alignment's _ridge_prep/_ridge_predict
at a FORCED single-lambda grid), the per-pair intersection machinery
(la._align_rows), per-tier null application, sufficientTier (incl. the
"none <= T8" case), scale consistency (raw + recal blocks carry the same
gap algebra on BOTH arms), and the §3 lattice-verdict predicates.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

REPO = Path(__file__).resolve().parents[1]
if str(REPO / "scripts") not in sys.path:
    sys.path.insert(0, str(REPO / "scripts"))

import issue825_fit_cells as fc  # noqa: E402
import issue825_map_alignment as ma  # noqa: E402
import issue1336_decision_v2 as dv  # noqa: E402
import issue1336_ladder_alignment as la  # noqa: E402
import issue1336_metric_ladder as ml  # noqa: E402

torch.set_num_threads(2)

GRID = np.logspace(-2, 4, 7)


def _linked_pair(
    n: int = 90,
    d: int = 6,
    seed: int = 0,
    *,
    ctx_offset: float = 3.0,
    noise: float = 0.05,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """(Xs, Ys, Xt, Yt, ids) with target = source under a pure CONTEXT offset:
    the tier-1 correction (x_t - dx) exactly restores the source geometry."""
    rng = np.random.default_rng(seed)
    xs = rng.normal(size=(n, d))
    w_true = rng.normal(size=(d, d)) / np.sqrt(d)
    b_true = rng.normal(size=d)
    ys = xs @ w_true + b_true + noise * rng.normal(size=(n, d))
    dx = ctx_offset * rng.normal(size=d)
    xt = xs + dx
    yt = (xt - dx) @ w_true + b_true + noise * rng.normal(size=(n, d))
    ids = np.asarray([f"s{i}" for i in range(n)])
    return xs, ys, xt, yt, ids


def _as_layered(a: np.ndarray) -> np.ndarray:
    return a[:, None, :]


@pytest.fixture(scope="module")
def offset_battery() -> dict:
    """ONE battery run on the ctx-offset fixture serving several asserts."""
    xs, ys, xt, yt, ids = _linked_pair()
    payload, preds = ml.run_battery_arrays(
        _as_layered(xs),
        _as_layered(ys),
        _as_layered(xt),
        _as_layered(yt),
        ids,
        frozen_layers=(0,),
        null_draws=3,
        n_boot=32,
        boot_seed=123,
        grid=GRID,
        band=0.02,
        full_tier_layers=(0,),
    )
    return {"payload": payload, "preds": preds}


def test_tier_nesting_t1_rescues_ctx_offset(offset_battery) -> None:
    blk = offset_battery["payload"]["per_layer"]["0"]["raw"]
    assert blk["within_r2"] > 0.9, blk["within_r2"]
    assert blk["tiers"]["t0"]["r2"] < 0.5, blk["tiers"]["t0"]["r2"]
    assert blk["tiers"]["t1"]["r2"] > 0.9, blk["tiers"]["t1"]["r2"]
    # nesting: the richer tiers never fall apart once T1 rescued the offset
    assert blk["tiers"]["t8"]["r2"] > 0.8, blk["tiers"]["t8"]["r2"]


def test_per_tier_nulls_applied_and_below_observed(offset_battery) -> None:
    pl = offset_battery["payload"]["per_layer"]["0"]
    nulls = pl["nulls"]
    mat = np.asarray(nulls["r2_matrix"], dtype=float)
    assert mat.shape == (3, len(ml.DRAWS_ORDER)), mat.shape
    assert nulls["order"] == list(ml.DRAWS_ORDER)
    # shuffled-pairing nulls: within/t1 nulls sit far below the observed reads
    within_null = np.nanmax(mat[:, 0])
    assert within_null < 0.2, within_null
    assert pl["raw"]["within_r2"] - within_null > 0.5


def test_scale_consistency_recal_gap_on_both_arms(offset_battery) -> None:
    pl = offset_battery["payload"]["per_layer"]["0"]
    for scale in ("raw", "recal"):
        blk = pl[scale]
        assert set(blk["tiers"]) == set(ml.TIER_NAMES)
        for nm in ml.TIER_NAMES:
            t = blk["tiers"][nm]
            # gap algebra holds INSIDE the scale: both arms went through the
            # same transform class (within_r2 and tier r2 of the SAME block)
            assert abs(t["gap"] - (blk["within_r2"] - t["r2"])) < 1e-9
        d8 = blk["delta_tier8"]
        assert abs(d8["point"] - (blk["within_r2"] - blk["tiers"]["t8"]["r2"])) < 1e-9


def test_sufficient_tier_block_and_preds_store(offset_battery) -> None:
    blk = offset_battery["payload"]["per_layer"]["0"]["raw"]
    st = blk["sufficient_tier"]
    assert st["tier"] != "none"  # T1 rescues under band=0.02 on this fixture
    assert int(st["tier"]) <= 4
    hist = st["per_draw_hist"]
    assert sum(hist.values()) == 32  # one entry per bootstrap draw
    preds = offset_battery["preds"]
    for key in ("within_l0", "t8_l0", "within_recal_l0", "t8_recal_l0", "y_l0", "t0_l0"):
        assert key in preds, sorted(preds)
    assert preds["tier_r2_draws_l0"].shape == (len(ml.DRAWS_ORDER), 32)


def test_sufficient_tier_unit_incl_none() -> None:
    assert ml.sufficient_tier(0.9, [0.1] * 8 + [0.85], 0.06) == 8
    assert ml.sufficient_tier(0.9, [0.86] + [0.1] * 8, 0.06) == 0
    assert ml.sufficient_tier(0.9, [0.1] * 9, 0.06) is None


def _mk_preps(xs_t, xt_t, ys_t, tr) -> dict:
    return {
        "s": ml._v2_prep(xs_t[tr], inner_seed=7, n_inner=2),
        "t": ml._v2_prep(xt_t[tr], inner_seed=7, n_inner=2),
        "ys": ml._v2_prep(ys_t[tr], inner_seed=7, n_inner=2),
    }


def test_procrustes_orthogonal_direction_aware() -> None:
    """Rotated-target fixture: T5 (orthogonal Procrustes) rescues what the
    scalar T4 cannot; the fitted R is orthogonal and recovers the true R0."""
    rng = np.random.default_rng(3)
    n, d = 96, 6
    xs = rng.normal(size=(n, d))
    w_true = rng.normal(size=(d, d)) / np.sqrt(d)
    ys = xs @ w_true + 0.02 * rng.normal(size=(n, d))
    r0, _ = np.linalg.qr(rng.normal(size=(d, d)))
    xt = xs.copy()
    yt = ys @ r0  # target answers = rotated source answers
    xs_t = torch.as_tensor(xs, dtype=torch.float64)
    ys_t = torch.as_tensor(ys, dtype=torch.float64)
    xt_t = torch.as_tensor(xt, dtype=torch.float64)
    yt_t = torch.as_tensor(yt, dtype=torch.float64)
    tr = torch.zeros(n, dtype=torch.bool)
    tr[: n - 24] = True
    te = ~tr
    preps = _mk_preps(xs_t, xt_t, ys_t, tr)
    te_preds, aux = ml._fold_observed(xs_t, ys_t, xt_t, yt_t, tr, te, GRID, preps)

    def _r2(pred: torch.Tensor) -> float:
        res = float(((yt_t[te] - pred) ** 2).sum())
        tot = float(((yt_t[te] - yt_t[te].mean(0)) ** 2).sum())
        return 1.0 - res / tot

    assert _r2(te_preds["t4"]) < 0.5, _r2(te_preds["t4"])  # scalar can't rotate
    assert _r2(te_preds["t5"]) > 0.9, _r2(te_preds["t5"])  # Procrustes can
    r_fit = np.asarray(aux["orth_R"], dtype=float)
    assert np.allclose(r_fit.T @ r_fit, np.eye(d), atol=1e-8)  # orthogonality
    # direction-awareness: the recovered rotation matches the true R0 (the
    # fit is on W_s outputs ~ ys, targets yt = ys @ r0)
    assert np.abs(r_fit - r0).max() < 0.15, np.abs(r_fit - r0).max()


def test_t8_equals_parent_comp_samefn_single_lambda() -> None:
    """T8 reproduces the parent comp_samefn_b2i chain (A_ans o W_s o
    A_ctx_rev) computed through issue825_map_alignment's own
    _ridge_prep/_ridge_predict at a FORCED single-lambda grid."""
    xs, ys, xt, yt, _ids = _linked_pair(n=120, d=8, seed=11)
    tr_np = np.zeros(120, dtype=bool)
    tr_np[:90] = True
    te_np = ~tr_np
    lam = 1.0
    single = np.asarray([lam], dtype=np.float64)
    old_fc, old_ma = fc.LAMBDAS, ma.LAMBDAS
    try:
        fc.LAMBDAS = single
        ma.LAMBDAS = single
        xs_t = torch.as_tensor(xs, dtype=torch.float64)
        ys_t = torch.as_tensor(ys, dtype=torch.float64)
        xt_t = torch.as_tensor(xt, dtype=torch.float64)
        yt_t = torch.as_tensor(yt, dtype=torch.float64)
        tr = torch.as_tensor(tr_np)
        te = torch.as_tensor(te_np)
        preps = _mk_preps(xs_t, xt_t, ys_t, tr)
        te_preds, _aux = ml._fold_observed(xs_t, ys_t, xt_t, yt_t, tr, te, single, preps)

        # parent chain: same (tr, te), same single-lambda grid. Mapping
        # b->s, i->t (comp_samefn_b2i evaluated on target answers).
        prep_xi = ma._ridge_prep(xt_t[tr])  # "Xi" = target contexts
        prep_xb = ma._ridge_prep(xs_t[tr])  # "Xb" = source contexts
        prep_yb = ma._ridge_prep(ys_t[tr])  # "Yb" = source answers
        xbhat = ma._ridge_predict(prep_xi, xs_t[tr], xt_t[te])  # A_ctx_rev
        ybhat = ma._ridge_predict(prep_xb, ys_t[tr], xbhat)  # W_s (M_base)
        parent_t8 = ma._ridge_predict(prep_yb, yt_t[tr], ybhat)  # A_ans
    finally:
        fc.LAMBDAS = old_fc
        ma.LAMBDAS = old_ma
    diff = float((te_preds["t8"] - parent_t8).abs().max())
    assert diff < 1e-6, f"T8 vs parent comp_samefn max |diff| = {diff}"


def test_align_rows_intersection_recompute() -> None:
    """The per-pair intersection: rows are re-keyed by prompt id, garbage
    rows outside the intersection are excluded from BOTH sides."""
    ids0 = np.asarray([f"s{i}" for i in range(60)] + [f"g{i}" for i in range(10)])
    ids1 = np.asarray([f"s{i}" for i in range(10, 70)])
    common, i0, i1 = la._align_rows(ids0, ids1)
    assert set(common) == {f"s{i}" for i in range(10, 60)}
    assert np.all(ids0[i0] == common) and np.all(ids1[i1] == common)
    x0 = np.arange(len(ids0), dtype=float)[:, None]
    x1 = np.arange(len(ids1), dtype=float)[:, None]
    # row content keyed by id survives the alignment on both sides
    for c, a, b in zip(common, x0[i0, 0], x1[i1, 0], strict=True):
        assert ids0[int(a)] == c and ids1[int(b)] == c


def test_prep_cache_no_cross_pair_target_collision() -> None:
    """Two pairs sharing a source (base->A, base->B) through ONE retaining
    PrepCache must NOT reuse pair A's TARGET prep for pair B — the "t" key
    carries m1 (fails pre-fix: an m0-only tag collided the "t" entries)."""
    xs, ys, xt_a, yt_a, ids = _linked_pair(n=90, d=6, seed=5)
    # pair B: a WILDLY different target (rescaled + shuffled dims) — a
    # collision that served pair A's Xt prep would wreck B's within read.
    rng = np.random.default_rng(6)
    xt_b = 50.0 * xs[:, ::-1].copy() + rng.normal(size=xs.shape)
    w_b = rng.normal(size=(6, 6)) / np.sqrt(6)
    yt_b = xt_b @ w_b + 0.05 * rng.normal(size=xs.shape)
    cache = ml.PrepCache(capacity=1000)  # RETAINING cache — the collision regime

    def _run(xt, yt, m1: str) -> dict:
        payload, _ = ml.run_battery_arrays(
            _as_layered(xs),
            _as_layered(ys),
            _as_layered(xt),
            _as_layered(yt),
            ids,
            frozen_layers=(0,),
            null_draws=0,
            n_boot=8,
            boot_seed=1,
            grid=GRID,
            band=0.02,
            prep_cache=cache,
            cache_tags={
                "s": ("c", "f", "base"),
                "t": ("c", "f", m1),
                "ys": ("c", "f", "base"),
            },
        )
        return payload["per_layer"]["0"]["raw"]

    blk_a = _run(xt_a, yt_a, "dpo")
    blk_b = _run(xt_b, yt_b, "rlvr")
    # shared-source preps ("s"/"ys") HIT on pair B; target preps do not
    assert cache.hits > 0
    # pair B's within read stays healthy — a served pair-A Xt prep would
    # standardize/rotate B's contexts with A's statistics and destroy it
    assert blk_b["within_r2"] > 0.9, blk_b["within_r2"]
    assert blk_a["within_r2"] > 0.9, blk_a["within_r2"]


def test_lattice_verdict_all_branches() -> None:
    assert (
        dv.lattice_verdict(
            {"point": 1.0, "ci_lo": 0.5, "ci_hi": 1.5}, {"point": 2.0, "ci_lo": 1.0, "ci_hi": 3.0}
        )
        == "rlvr_teaching"
    )
    assert (
        dv.lattice_verdict(
            {"point": -1.0, "ci_lo": -1.5, "ci_hi": -0.5},
            {"point": -2.0, "ci_lo": -3.0, "ci_hi": -1.0},
        )
        == "rlvr_unlearning"
    )
    assert (
        dv.lattice_verdict(
            {"point": -1.0, "ci_lo": -1.5, "ci_hi": -0.5},
            {"point": 2.0, "ci_lo": 1.0, "ci_hi": 3.0},
        )
        == "elicitation_consistent"
    )
    assert (
        dv.lattice_verdict(
            {"point": 0.1, "ci_lo": -0.5, "ci_hi": 0.5}, {"point": 0.1, "ci_lo": -0.5, "ci_hi": 0.5}
        )
        == "inconclusive"
    )
