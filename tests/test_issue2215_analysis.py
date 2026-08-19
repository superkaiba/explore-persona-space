"""Issue #2215 Phase C analysis — CPU pins over the REAL statistical core.

Synthetic small-grain fixtures (2 cells x 3 carriers x 1 value-pair, H=6,
L=2) exercise the PRODUCTION functions of ``scripts/issue2215_analysis.py``
— pooling/loader math, Δ geometry, sign-flip + derangement nulls, 2AFC,
identity+bias LOTO folds, bootstrap semantics — plus the Phase C/D gate
failure modes in ``scripts/issue2215_run.py``. No network, no GPU, no repo
``eval_results/`` reads (sparse-worktree safe); fakes exist only as DATA
(synthetic tensors / payload files), never as substitutes for the functions
under test.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from scipy.stats import spearmanr

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "scripts"))

import issue2215_analysis as A  # noqa: E402
import issue2215_run as R  # noqa: E402

CELLS = ("cell_a", "cell_b")
# 6 carriers so the sign-flip null's identity atom (mass 2/2^m) sits BELOW
# the 5% tail — at m=3 the 95th percentile is the identity atom itself and
# a perfectly-consistent fixture can never clear the band (the production
# grid has m=12).
CARRIERS = ("car1", "car2", "car3", "car4", "car5", "car6")
H = 12  # disjoint basis slots: cells 0-1, carriers 2-7, Δ dirs 8-10, jitter 11
L = 2
K_DRAWS = 4


# ── synthetic bank + tensors ──────────────────────────────────────────


def make_bank(degenerate: tuple[str, ...] = ("cell_b",)) -> dict:
    contexts: dict[str, dict] = {}
    pairs: list[dict] = []
    for cell in CELLS:
        for car in CARRIERS:
            for val in ("v1", "v2"):
                contexts[f"{cell}|{car}|{val}"] = {
                    "cell": cell,
                    "value_id": val,
                    "carrier": car,
                }
            pairs.append(
                {
                    "pair_id": f"{cell}|{car}|v1v2",
                    "cell": cell,
                    "carrier": car,
                    "value_a": "v1",
                    "value_b": "v2",
                    "a": f"{cell}|{car}|v1",
                    "b": f"{cell}|{car}|v2",
                }
            )
    return {
        "cells": {c: {"base_type": c} for c in CELLS},
        "contexts": contexts,
        "pairs": pairs,
        "degenerate_at_pe_cells": list(degenerate),
    }


def _basis(i: int) -> np.ndarray:
    e = np.zeros(H)
    e[i % H] = 1.0
    return e


def make_vc(pt: A.PairTable, *, degenerate_pe: tuple[str, ...] = ("cell_b",)) -> dict:
    """v_ce: per-context distinct base + shared per-cell Δ direction (v2 =
    v1 + 3*u_cell -> consistency 1.0, norm 3.0). v_pe: degenerate cells get
    CARRIER-only vectors (pair Δ exactly 0, pair cos 1.0)."""
    ce = torch.zeros(len(pt.ids), L, H)
    pe = torch.zeros(len(pt.ids), L, H)
    for row, cid in enumerate(pt.ids):
        cell, car, val = cid.split("|")
        ci = CELLS.index(cell)
        base = 5.0 * _basis(ci) + 1.5 * _basis(2 + CARRIERS.index(car))
        u_cell = _basis(8 + ci)
        v = base + (3.0 * u_cell if val == "v2" else 0.0)
        for layer in range(L):
            ce[row, layer] = torch.tensor(v * (1.0 + 0.1 * layer))
            if cell in degenerate_pe:
                pe[row, layer] = torch.tensor(base)  # value-independent
            else:
                pe[row, layer] = torch.tensor(v + 0.5 * _basis(1))
    return {"layers": list(range(L)), "hidden": H, "ce": ce, "pe": pe}


def tail_vec(cid: str, draw: int) -> np.ndarray:
    """Per-draw answer state: context-keyed base + small draw jitter (so
    split halves differ and the split-half floor is > 0)."""
    cell, car, val = cid.split("|")
    base = (
        4.0 * _basis(CELLS.index(cell))
        + 1.0 * _basis(2 + CARRIERS.index(car))
        + (2.0 * _basis(10) if val == "v2" else 0.0)
    )
    return base + 0.125 * draw * _basis(11)


def write_va_shards(
    va_dir: Path,
    ids: list[str],
    *,
    empty: set[tuple[str, int]] = frozenset(),
    span_offset: float = 0.0,
    k: int = K_DRAWS,
) -> None:
    """Two va2215-format shards splitting the contexts (real payload keys)."""
    va_dir.mkdir(parents=True, exist_ok=True)
    halves = [ids[: len(ids) // 2], ids[len(ids) // 2 :]]
    for si, part in enumerate(halves):
        index, rows_tail, rows_span, empties = [], [], [], []
        for cid in part:
            for draw in range(k):
                if (cid, draw) in empty:
                    empties.append(len(index))
                index.append({"context_id": cid, "draw": draw})
                v = np.stack([tail_vec(cid, draw) * (1.0 + 0.1 * layer) for layer in range(L)])
                rows_tail.append(torch.tensor(v, dtype=torch.float16))
                rows_span.append(torch.tensor(v + span_offset, dtype=torch.float16))
        torch.save(
            {
                "layers": list(range(L)),
                "index": index,
                "va_tail_incl": torch.stack(rows_tail),
                "va_span_excl": torch.stack(rows_span),
                "empty_rows": sorted(empties),
            },
            va_dir / f"va2215_part_w{si}.pt",
        )


def make_ridge_payload(path: Path, *, layer: int, w_scale: float = 1.0) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "kind": "ridge",
            "layer": layer,
            "xmu": torch.zeros(H, dtype=torch.float32),
            "xsd": torch.ones(H, dtype=torch.float32),
            "ymu": torch.zeros(H, dtype=torch.float32),
            "W": torch.eye(H, dtype=torch.float32) * w_scale,
        },
        path,
    )


@pytest.fixture()
def bank():
    return make_bank()


@pytest.fixture()
def pt(bank):
    return A.PairTable.from_bank(bank, None)


@pytest.fixture()
def views(bank, pt):
    return A.build_cell_views(bank, pt)


# ── pair table + cell views ───────────────────────────────────────────


def test_pair_table_and_cell_views_complete_grid(bank, pt, views):
    n_car = len(CARRIERS)
    assert len(pt.ids) == 4 * n_car and len(pt.pair_ids) == 2 * n_car
    assert pt.cells == list(CELLS)
    for cell in CELLS:
        cv = views[cell]
        assert cv.pair_at.shape == (n_car, 1) and (cv.pair_at >= 0).all()
        assert len(cv.ctx_rows) == 2 * n_car and cv.values == ["v1", "v2"]
    scoped = A.PairTable.from_bank(bank, ("cell_a",))
    assert scoped.cells == ["cell_a"] and len(scoped.ids) == 2 * n_car


def test_pair_table_incomplete_grid_fails_loud():
    # A cell with TWO vps (v1-v2, v1-v3): dropping one (carrier, vp) pair
    # leaves the carrier present via the other vp -> a genuinely incomplete
    # (carrier x vp) grid, which must fail loud.
    bank2 = make_bank(degenerate=())
    for car in CARRIERS:
        bank2["contexts"][f"cell_a|{car}|v3"] = {
            "cell": "cell_a",
            "value_id": "v3",
            "carrier": car,
        }
        bank2["pairs"].append(
            {
                "pair_id": f"cell_a|{car}|v1v3",
                "cell": "cell_a",
                "carrier": car,
                "value_a": "v1",
                "value_b": "v3",
                "a": f"cell_a|{car}|v1",
                "b": f"cell_a|{car}|v3",
            }
        )
    bank2["pairs"] = [p for p in bank2["pairs"] if p["pair_id"] != "cell_a|car1|v1v3"]
    pt2 = A.PairTable.from_bank(bank2, None)
    with pytest.raises(AssertionError, match="incomplete"):
        A.build_cell_views(bank2, pt2)


def make_bank_with_rev_cell() -> dict:
    """Borrowed-membership shape (the frozen #2162 ``conflict_*_rev`` cells):
    ``cell_a_rev`` owns ZERO contexts — its pairs re-pair ``cell_a``'s
    contexts with the value roles swapped."""
    bank = make_bank(degenerate=())
    for car in CARRIERS:
        bank["pairs"].append(
            {
                "pair_id": f"cell_a_rev|{car}|v2v1",
                "cell": "cell_a_rev",
                "carrier": car,
                "value_a": "v2",
                "value_b": "v1",
                "a": f"cell_a|{car}|v2",
                "b": f"cell_a|{car}|v1",
            }
        )
    bank["cells"]["cell_a_rev"] = {"base_type": "cell_a_rev"}
    return bank


def test_build_cell_views_borrowed_membership_cell(caplog):
    # Regression for the production Phase C crash (KeyError: 24): a cell
    # whose pairs reference contexts ATTRIBUTED to another cell must derive
    # ctx_rows from its own pairs, not from the cell_of grouping (which is
    # empty for such a cell and crashed at the first a_loc lookup).
    bank = make_bank_with_rev_cell()
    pt = A.PairTable.from_bank(bank, None)
    with caplog.at_level("INFO", logger="issue2215.analysis"):
        views = A.build_cell_views(bank, pt)
    assert set(views) == {"cell_a", "cell_a_rev", "cell_b"}
    cv = views["cell_a_rev"]
    # Membership borrowed from cell_a: identical global row set.
    assert np.array_equal(cv.ctx_rows, views["cell_a"].ctx_rows)
    assert len(cv.ctx_rows) == 2 * len(CARRIERS)
    assert len(cv.pair_idx) == len(CARRIERS)
    # a/b sides resolve to the exact context ids each pair names.
    by_id = {p["pair_id"]: p for p in bank["pairs"]}
    for cell in views:
        v = views[cell]
        for j, k in enumerate(v.pair_idx):
            pair = by_id[pt.pair_ids[int(k)]]
            assert pt.ids[int(v.ctx_rows[v.a_loc[j]])] == pair["a"]
            assert pt.ids[int(v.ctx_rows[v.b_loc[j]])] == pair["b"]
    # Complete (carrier x vp) grid for the rev cell.
    assert cv.pair_at.shape == (len(CARRIERS), 1) and (cv.pair_at >= 0).all()
    # Fix-engaged signal: the build-time line names the borrowed cell + count.
    assert "borrowed-membership" in caplog.text
    assert f"cell_a_rev({2 * len(CARRIERS)})" in caplog.text


def test_build_cell_views_healthy_bank_membership_matches_cell_of(bank, pt, caplog):
    # Invariance on the healthy path: when every cell's pairs reference only
    # its own contexts, pair-derived membership == the cell_of grouping and
    # no borrowed-membership cell is reported.
    with caplog.at_level("INFO", logger="issue2215.analysis"):
        views = A.build_cell_views(bank, pt)
    for cell in pt.cells:
        attributed = {r for r, c in enumerate(pt.cell_of) if c == cell}
        assert {int(r) for r in views[cell].ctx_rows} == attributed
    assert "borrowed-membership: none" in caplog.text


# ── small math helpers vs naive references ────────────────────────────


def test_mean_pairwise_cosine_matches_naive():
    rng = np.random.default_rng(0)
    d = rng.normal(size=(4, H))
    dn = d / np.linalg.norm(d, axis=1, keepdims=True)
    g = dn @ dn.T
    naive = np.mean([g[i, j] for i in range(4) for j in range(4) if i != j])
    assert np.isclose(A.mean_pairwise_cosine_from_gram(g), naive)


def test_signflip_null_matches_naive_enumeration():
    rng = np.random.default_rng(1)
    d = rng.normal(size=(3, H))
    dn = d / np.linalg.norm(d, axis=1, keepdims=True)
    g = dn @ dn.T
    signs = np.array([[1, 1, 1], [1, -1, 1], [-1, -1, 1], [-1, 1, -1]])
    got = A.signflip_null_consistency(g, signs)
    for k, s in enumerate(signs):
        gs = g * np.outer(s, s)
        naive = (gs.sum() - np.trace(gs)) / (3 * 2)
        assert np.isclose(got[k], naive)


def test_bootstrap_pairwise_cosine_excludes_duplicate_draws():
    rng = np.random.default_rng(2)
    d = rng.normal(size=(3, H))
    dn = d / np.linalg.norm(d, axis=1, keepdims=True)
    g = dn @ dn.T
    idx = np.array([[0, 0, 1], [0, 1, 2]])
    got = A.bootstrap_pairwise_cosine(g, idx)
    # row 0: distinct pairs are only (0,1)/(1,0) style combos across the two
    # DISTINCT carriers 0 and 1 -> mean must be g[0,1] exactly (self/dup
    # pairs with cos=1 excluded).
    assert np.isclose(got[0], g[0, 1])
    assert np.isclose(got[1], A.mean_pairwise_cosine_from_gram(g))


def test_deranged_perms_valid_and_deterministic():
    p1 = A.deranged_perms(4, 100, np.random.default_rng([9, 9]))
    p2 = A.deranged_perms(4, 100, np.random.default_rng([9, 9]))
    assert (p1 == p2).all() and p1.shape == (100, 4)
    assert (np.sort(p1, axis=1) == np.arange(4)).all()  # permutations
    assert (p1 != np.arange(4)).all()  # no fixed point


def test_bootstrap_spearman_matches_scipy_loop():
    rng = np.random.default_rng(3)
    x, y = rng.normal(size=12), rng.normal(size=12)
    out = A.bootstrap_spearman(x, y, 40, [77])
    assert np.isclose(out["obs"], spearmanr(x, y).statistic)
    idx = np.random.default_rng([77]).integers(0, 12, size=(40, 12))
    naive = np.array([spearmanr(x[i], y[i]).statistic for i in idx])
    assert np.allclose(out["draws"], naive, equal_nan=True)


# ── DV1 geometry ──────────────────────────────────────────────────────


def test_dv1_magnitude_consistency_and_flags(bank, pt, views):
    vc = make_vc(pt)
    nulls: dict[str, np.ndarray] = {}
    dv1 = A.compute_dv1(
        vc,
        pt,
        views,
        bank["cells"],
        set(bank["degenerate_at_pe_cells"]),
        null_b=200,
        boot_b=200,
        nulls_out=nulls,
    )
    rec = dv1["per_cell"]["cell_a"]["ce"]
    # Every v2 = v1 + 3*u_cell at layer 0 -> median ||Δ|| = 3.0 exactly and
    # perfectly aligned directions -> consistency 1.0 above any sign-flip band.
    assert np.isclose(rec["median_norm"][0], 3.0)
    assert np.isclose(rec["primary"]["consistency"], 1.0)
    assert rec["primary"]["band95"] < 1.0
    assert rec["primary"]["consistency_ci_excludes_band"]
    # Yardstick: same-value cross-carrier distance = |1.5*(e_i - e_j)| scaled.
    assert rec["yardstick"][0] > 0
    assert np.isclose(rec["primary"]["ratio"], 3.0 / rec["yardstick"][0])
    # Degenerate-at-pe cell: sanity PASS + excluded from pe aggregates.
    pe_rec = dv1["per_cell"]["cell_b"]["pe"]
    assert pe_rec["degenerate_at_pe"] and "excluded" in pe_rec
    assert np.isclose(pe_rec["median_norm"][0], 0.0)
    assert dv1["aggregates"]["pe"]["n_cells"] == 1  # cell_b excluded
    assert dv1["aggregates"]["ce"]["n_cells"] == 2
    # Null matrices persisted per (slot, cell, layer).
    assert any(k.startswith("dv1|ce|cell_a|") for k in nulls)
    # Per-pair rows carry both slots for every pair.
    assert len(dv1["per_pair_rows"]) == 2 * len(pt.pair_ids)


def test_dv1_degenerate_pe_violation_fails_loud(bank, pt, views):
    vc = make_vc(pt, degenerate_pe=())  # pe now value-dependent everywhere
    with pytest.raises(AssertionError, match="degenerate-at-pe"):
        A.compute_dv1(
            vc,
            pt,
            views,
            bank["cells"],
            {"cell_b"},  # declared degenerate but the tensors violate it
            null_b=50,
            boot_b=50,
            nulls_out={},
        )


# ── DV2 loader + geometry ─────────────────────────────────────────────


def test_load_answer_means_accumulation_halves_and_exclusions(tmp_path, pt):
    dead = pt.ids[0]  # all draws empty -> n_valid = 0
    flagged = pt.ids[1]  # one empty draw -> n_valid = 3 (< 4, split-flagged)
    empty = {(dead, d) for d in range(K_DRAWS)} | {(flagged, 0)}
    write_va_shards(tmp_path / "va", pt.ids, empty=empty)
    ans = A.load_answer_means(tmp_path / "va", pt.ids, pt.row_of, banked_dir=None, k_draws=K_DRAWS)
    r_ok = pt.row_of[pt.ids[2]]
    expected = np.stack([np.mean([tail_vec(pt.ids[2], d) for d in range(K_DRAWS)], axis=0) * 1.0])[
        0
    ]
    got = ans.mean["tail"][r_ok, 0].numpy()
    assert np.allclose(got, expected, atol=1e-2)  # fp16 storage
    h1 = np.mean([tail_vec(pt.ids[2], d) for d in (0, 1)], axis=0)
    assert np.allclose(ans.half1["tail"][r_ok, 0].numpy(), h1, atol=1e-2)
    assert ans.n_valid[pt.row_of[dead]] == 0
    assert ans.n_valid[pt.row_of[flagged]] == 3
    assert (ans.mean["tail"][pt.row_of[dead]] == 0).all()
    assert "tiny substitution" in ans.span_source


def test_load_answer_means_duplicate_row_fails_loud(tmp_path, pt):
    write_va_shards(tmp_path / "va", pt.ids)
    # Duplicate one (cid, draw) in a third shard.
    dup = tmp_path / "va" / "va2215_dup_w9.pt"
    src = torch.load(sorted((tmp_path / "va").glob("va2215_*.pt"))[0], weights_only=False)
    torch.save(
        {
            "layers": src["layers"],
            "index": src["index"][:1],
            "va_tail_incl": src["va_tail_incl"][:1],
            "va_span_excl": src["va_span_excl"][:1],
            "empty_rows": [],
        },
        dup,
    )
    with pytest.raises(AssertionError, match="duplicate"):
        A.load_answer_means(tmp_path / "va", pt.ids, pt.row_of, banked_dir=None, k_draws=K_DRAWS)


def test_compute_dv2_noise_normalization_and_exclusions(tmp_path, bank, pt, views):
    dead = pt.ids[0]
    empty = {(dead, d) for d in range(K_DRAWS)}
    write_va_shards(tmp_path / "va", pt.ids, empty=empty)
    ans = A.load_answer_means(tmp_path / "va", pt.ids, pt.row_of, banked_dir=None, k_draws=K_DRAWS)
    included = (ans.n_valid[pt.a_row] > 0) & (ans.n_valid[pt.b_row] > 0)
    assert included.sum() == len(pt.pair_ids) - 1  # dead context kills its pair
    nulls: dict[str, np.ndarray] = {}
    dv2 = A.compute_dv2(ans, pt, views, included, null_b=100, boot_b=100, nulls_out=nulls)
    dead_cell = bank["contexts"][dead]["cell"]
    rec = dv2["per_cell"][dead_cell]["tail"]
    n_car = len(CARRIERS)
    assert rec["n_included_pairs"] == n_car - 1 and rec["n_pairs"] == n_car
    # v2 - v1 = 2*e5 in tail space at layer 0 -> median ||Δ|| = 2; the
    # split-half floor is the draw-jitter distance (> 0), so the
    # noise-normalized shift is finite and > 1 for this construction.
    other = dv2["per_cell"]["cell_a" if dead_cell != "cell_a" else "cell_b"]["tail"]
    assert np.isclose(other["median_norm"][0], 2.0, atol=0.05)
    assert other["noise_normalized_primary"] > 1.0
    assert other["split_half"]["cell_cross_half_consistency"] is not None
    rows = dv2["per_pair_rows"]
    assert sum(not r["included"] for r in rows) == 1
    assert dv2["n_valid_zero_contexts"] == 1


# ── DV3 pieces ────────────────────────────────────────────────────────


def test_observed_2afc_identity_map_is_perfect():
    rng = np.random.default_rng(5)
    t = rng.normal(size=(6, H))
    s = A.sim_blocks(t, t)
    a = np.array([0, 2, 4])
    b = np.array([1, 3, 5])
    for metric in A.METRICS:
        m_a, m_b = A.observed_2afc(s[metric], a, b)
        assert (m_a > 0).all() and (m_b > 0).all()


def test_null_2afc_cell_mean_half_and_derangement(bank, pt, views):
    rng = np.random.default_rng(6)
    cv = views["cell_a"]
    n_pairs = len(cv.pair_idx)
    t = rng.normal(size=(len(cv.ctx_rows), H))
    s = A.sim_blocks(t, t)["cosine"]
    b_draws = 2000
    sigma = A.deranged_perms(len(cv.carriers), b_draws, np.random.default_rng([1]))
    q = cv.pair_at[sigma[:, cv.carrier_loc], cv.vp_loc]
    assert (q != np.arange(n_pairs)[None, :]).all()  # never own pair
    side = np.random.default_rng([2]).integers(0, 2, size=(b_draws, n_pairs)).astype(bool)
    side2 = np.random.default_rng([3]).integers(0, 2, size=(b_draws, n_pairs)).astype(bool)
    valid = np.ones(n_pairs, dtype=bool)
    nc, nt = A.null_2afc_cell(s, cv, sigma, side, side2, valid)
    assert (nt == 2 * n_pairs).all()
    assert abs((nc / nt).mean() - 0.5) < 0.05  # side randomization centers at 1/2
    # Excluded pair drops both its own comparisons and its target-duo draws.
    valid2 = valid.copy()
    valid2[-1] = False
    _, nt2 = A.null_2afc_cell(s, cv, sigma, side, side2, valid2)
    assert (nt2 <= 2 * (n_pairs - 1)).all()


def test_idbias_loto_predict_leaves_own_type_out():
    cells = ["cell_a"] * 3 + ["cell_b"] * 3
    x = np.zeros((6, H))
    t = np.stack([_basis(0) if c == "cell_a" else _basis(1) for c in cells])
    valid = np.ones(6, dtype=bool)
    pred = A.idbias_loto_predict(x, t, cells, valid)
    # cell_a's b is fit ONLY on cell_b rows (t - x = e1) and vice versa.
    assert np.allclose(pred[:3], _basis(1)[None, :])
    assert np.allclose(pred[3:], _basis(0)[None, :])


def test_pooled_r2_cos_and_verdict_lattice():
    rng = np.random.default_rng(7)
    t = rng.normal(size=(8, H))
    perfect = A.pooled_r2_cos(t, t)
    assert np.isclose(perfect["r2_pooled"], 1.0) and np.isclose(perfect["mean_cosine"], 1.0)
    mean_pred = np.repeat(t.mean(axis=0, keepdims=True), 8, axis=0)
    assert abs(A.pooled_r2_cos(mean_pred, t)["r2_pooled"]) < 1e-9
    assert A.discrimination_verdict(0.8, [0.6, 0.9]) == "discriminates"
    assert A.discrimination_verdict(0.4, [0.2, 0.45]) == "fails-to-discriminate"
    assert A.discrimination_verdict(0.55, [0.4, 0.6]) == "inconclusive"
    assert A.discrimination_verdict(0.9, None) == "inconclusive"


def test_compute_dv3_identity_arm_discriminates_and_pe_excluded(tmp_path, bank, pt, views):
    vc = make_vc(pt)
    write_va_shards(tmp_path / "va", pt.ids)
    ans = A.load_answer_means(tmp_path / "va", pt.ids, pt.row_of, banked_dir=None, k_draws=K_DRAWS)
    # Targets ARE the ce states at each layer -> the identity ridge map is a
    # perfect predictor; overwrite ans.mean with vc-derived targets.
    for pool in A.POOLINGS:
        ans.mean[pool] = vc["ce"].double().clone()
    ridge = tmp_path / "ridge_L0.pt"
    make_ridge_payload(ridge, layer=0)
    ridge1 = tmp_path / "ridge_L1.pt"
    make_ridge_payload(ridge1, layer=1)
    arm_specs = [
        {"arm": "779ce", "slot": "ce", "paths": {0: ridge, 1: ridge1}},
        {"arm": "1738pe", "slot": "pe", "paths": {0: ridge, 1: ridge1}},
    ]
    nulls: dict[str, np.ndarray] = {}
    dv3 = A.compute_dv3(
        vc,
        ans,
        pt,
        views,
        arm_specs,
        set(bank["degenerate_at_pe_cells"]),
        np.ones(len(pt.pair_ids), dtype=bool),
        null_b=200,
        boot_b=200,
        nulls_out=nulls,
    )
    reg = dv3["registered"]
    assert reg["config"].startswith("L0")
    assert np.isclose(reg["pooled"]["779ce"]["acc"], 1.0)
    assert reg["pooled"]["779ce"]["verdict"] == "discriminates"
    rec = dv3["per_config"]["779ce|L0|tail"]
    assert np.isclose(rec["r2_pooled"], 1.0, atol=1e-9)
    assert rec["knn"]["cosine"]["acc_at_k"][1] == pytest.approx(1.0)
    band = rec["pooled"]["cosine"]["null_band"]
    assert band[0] < 0.9  # shuffled-pair null sits far below the identity acc
    # pe-input arms are N/A on the degenerate cell (incl. the idbias_pe twin).
    for arm in ("1738pe", "idbias_pe"):
        assert dv3["per_config"][f"{arm}|L0|tail"]["per_type"]["cell_b"] == {
            "na": "N/A — degenerate at pe"
        }
    assert any(k.endswith("|__pooled__|null") for k in nulls)
    assert "dv3|cluster_acc_values" in nulls
    assert any("-minus-idbias_ce|" in k for k in dv3["diff_vs_idbias"])
    # Per-pair rows exist for the primary config only, for included pairs.
    arms_in_rows = {r["arm"] for r in dv3["per_pair_rows"]}
    assert "779ce" in arms_in_rows
    assert all(r["layer"] == 0 and r["pooling"] == "tail" for r in dv3["per_pair_rows"])
    # Carrier transfer recorded at the primary config for ce arms.
    assert "carrier_transfer" in dv3["per_config"]["779ce|L0|tail"]


def test_compute_coupling_persists_per_cell_xy():
    """H2 record carries per_cell_xy (unit 3: the H2 figure reads these
    values verbatim off coupling.json — no recomputation at render time)."""
    dv1 = {"per_pair_rows": [], "cell_primary": {}}
    dv2 = {
        "per_pair_rows": [],
        "cell_primary": {
            "c1": {"noise_normalized": 1.0},
            "c2": {"noise_normalized": 2.0},
            "c3": {"noise_normalized": 3.0},
        },
    }
    sep = {"c1": 0.1, "c2": 0.5, "c3": 0.9}
    nulls: dict = {}
    out = A.compute_coupling(dv1, dv2, None, sep, boot_b=25, nulls_out=nulls)
    assert out["h2"]["per_cell_xy"] == {
        "c1": {"x": 1.0, "y": 0.1},
        "c2": {"x": 2.0, "y": 0.5},
        "c3": {"x": 3.0, "y": 0.9},
    }
    assert out["h2"]["obs"] == pytest.approx(1.0)  # monotone fixture
    assert "h2|spearman_boot" in nulls


# ── run_analysis end-to-end on synthetic data (real functions, no stubs) ──


def test_run_analysis_writes_all_outputs(tmp_path, bank, pt):
    vc = make_vc(pt)
    vc_path = tmp_path / "vc_bank.pt"
    per_context = {}
    for row, cid in enumerate(pt.ids):
        per_context[cid] = {"v_ce": vc["ce"][row].clone(), "v_pe": vc["pe"][row].clone()}
    torch.save({"layers": vc["layers"], "per_context": per_context}, vc_path)
    write_va_shards(tmp_path / "va", pt.ids)
    ridge = tmp_path / "ridge_L0.pt"
    make_ridge_payload(ridge, layer=0)
    anchors = tmp_path / "anchors.jsonl"
    with anchors.open("w") as fh:
        for p in bank["pairs"]:
            fh.write(json.dumps({"cell": p["cell"], "separation": 0.4}) + "\n")
    results = tmp_path / "results"
    nulldir = tmp_path / "nulls"
    inp = A.AnalysisInputs(
        bank=bank,
        vc_bank_path=vc_path,
        va_dir=tmp_path / "va",
        banked_anchor_dir=None,
        arm_specs=[{"arm": "779ce", "slot": "ce", "paths": {0: ridge}}],
        results_dir=results,
        null_dir=nulldir,
        anchors_jsonl=anchors,
        cells=None,
        null_b=60,
        boot_b=60,
        k_draws=K_DRAWS,
        repro={"test": True},
    )
    digest = A.run_analysis(inp)
    for name in (
        "dv1_context_shift.json",
        "dv2_answer_shift.json",
        "dv3_map_discrimination.json",
        "coupling.json",
        "null_bands.json",
    ):
        payload = json.loads((results / name).read_text())
        assert payload["repro"] == {"test": True}, name
    for name in ("dv1_pairs.jsonl", "dv2_pairs.jsonl", "dv3_pairs.jsonl"):
        assert (results / "perpair" / name).exists(), name
    npz = np.load(nulldir / "null_matrices.npz")
    idx = json.loads((nulldir / "null_matrices_index.json").read_text())
    assert sorted(npz.files) == idx["keys"] and idx["n_keys"] == len(npz.files)
    assert digest["n_cells"] == 2 and digest["n_excluded_pairs"] == 0
    # H2 degrades to a recorded skip below 3 overlapping cells (smoke grain).
    coupling = json.loads((results / "coupling.json").read_text())
    assert "skipped" in coupling["h2"] and "3" in coupling["h2"]["skipped"]
    assert digest["dv3_registered"]["pooled"]["779ce"]["acc"] is not None


# ── Phase C/D driver gates ────────────────────────────────────────────


def _cfg(tmp_path: Path, **over) -> R.RunConfig2215:
    args = R.parse_args().parse_args(
        [
            "--phase",
            over.pop("phase", "c"),
            "--staged-root",
            str(tmp_path / "staged"),
            "--out-root",
            str(tmp_path / "out"),
            "--tiny",
            "--null-b",
            "50",
            "--boot-b",
            "50",
            *over.pop("extra", []),
        ]
    )
    cfg = R.build_config(args)
    for k, v in over.items():
        setattr(cfg, k, v)
    cfg.out_root.mkdir(parents=True, exist_ok=True)
    return cfg


def test_phase_analysis_gate_refuses_without_upload_sentinel(tmp_path):
    cfg = _cfg(tmp_path)
    with pytest.raises(AssertionError, match="unit2-phase-c-sentinel-gate"):
        R.phase_analysis(cfg)


def test_phase_analysis_gate_refuses_on_regime_fp_mismatch(tmp_path):
    cfg = _cfg(tmp_path)
    R._write_json_atomic(cfg.out_root / "va2215_uploaded.json", {"regime_fp": "not-this-regime"})
    with pytest.raises(AssertionError, match="DIFFERENT capture regime"):
        R.phase_analysis(cfg)


def _seed_phase_c_gates(cfg) -> None:
    fp = R.regime_fingerprint(cfg)
    R._write_json_atomic(cfg.out_root / "va2215_uploaded.json", {"regime_fp": fp})
    R._write_json_atomic(R.stage_done_path(cfg), {"regime_fp": fp})


def test_phase_analysis_skips_on_matching_analysis_done(tmp_path, monkeypatch):
    cfg = _cfg(tmp_path)
    _seed_phase_c_gates(cfg)
    R._write_json_atomic(
        cfg.out_root / "analysis_done.json",
        {"analysis_fp": R.analysis_fingerprint(cfg), "digest": {}},
    )
    monkeypatch.setattr(
        A, "run_analysis", lambda inp: pytest.fail("analysis re-ran despite matching fp")
    )
    assert R.phase_analysis(cfg) == R.RC_OK


def test_phase_analysis_refuses_on_analysis_fp_mismatch(tmp_path):
    cfg = _cfg(tmp_path)
    _seed_phase_c_gates(cfg)
    R._write_json_atomic(cfg.out_root / "analysis_done.json", {"analysis_fp": "stale-knobs"})
    with pytest.raises(AssertionError, match="--force"):
        R.phase_analysis(cfg)


def test_analysis_fingerprint_tracks_knobs_not_capture(tmp_path):
    cfg = _cfg(tmp_path)
    base = R.analysis_fingerprint(cfg)
    cfg.boot_b = 999
    assert R.analysis_fingerprint(cfg) != base
    cfg.boot_b = 50
    assert R.analysis_fingerprint(cfg) == base
    assert R.regime_fingerprint(cfg) == R.regime_fingerprint(cfg)  # capture key unaffected


def _seed_tiny_driver_inputs(cfg, bank, pt, tmp_path: Path) -> None:
    """Stage the synthetic bank/vc/va inputs where the driver seam expects
    them (shared by the Phase C e2e and the Phase D figure-render e2e)."""
    vcdir = cfg.staged_root / R.VC_BANK_PREFIX
    vcdir.mkdir(parents=True, exist_ok=True)
    (vcdir / "bank.json").write_text(json.dumps(bank))
    vc = make_vc(pt)
    per_context = {
        cid: {"v_ce": vc["ce"][row].clone(), "v_pe": vc["pe"][row].clone()}
        for row, cid in enumerate(pt.ids)
    }
    torch.save({"layers": vc["layers"], "per_context": per_context}, vcdir / "vc_bank.pt")
    write_va_shards(cfg.va_dir, pt.ids, k=R.K_DRAWS)  # driver threads K_DRAWS=10
    anchors = tmp_path / "anchors.jsonl"
    with anchors.open("w") as fh:
        for k, p in enumerate(bank["pairs"]):
            fh.write(json.dumps({"cell": p["cell"], "separation": 0.3 + 0.01 * k}) + "\n")
    cfg.anchors_jsonl = anchors


def test_phase_analysis_runs_end_to_end_through_the_driver_seam(tmp_path, bank, pt):
    """The driver->analysis seam executed for real (AnalysisInputs kwargs,
    staged-path arithmetic, K_DRAWS threading) under --tiny: DV3 records its
    declared skip, outputs land in the out-root results twin, and an
    immediate re-run takes the idempotent analysis_done skip."""
    cfg = _cfg(tmp_path)
    _seed_phase_c_gates(cfg)
    _seed_tiny_driver_inputs(cfg, bank, pt, tmp_path)
    assert R.phase_analysis(cfg) == R.RC_OK
    adone = json.loads((cfg.out_root / "analysis_done.json").read_text())
    assert adone["analysis_fp"] == R.analysis_fingerprint(cfg)
    assert adone["digest"]["n_cells"] == 2
    dv3 = json.loads((cfg.results_dir / "dv3_map_discrimination.json").read_text())
    assert "skipped" in dv3 and "tiny" in dv3["skipped"]  # declared, never silent
    assert (cfg.null_dir / "null_matrices.npz").exists()
    assert str(cfg.out_root) in str(cfg.results_dir)  # tiny twin, not repo eval_results
    assert R.phase_analysis(cfg) == R.RC_OK  # idempotent skip on matching fp


def test_phase_finalize_requires_analysis_done_and_writes_outputs(tmp_path, monkeypatch):
    """Gate + sentinel semantics of phase_finalize (render_figures stubbed
    here — the REAL figure-render body runs in
    test_phase_finalize_renders_figures_end_to_end below)."""
    cfg = _cfg(tmp_path, phase="d", upload_mode="none")
    with pytest.raises(AssertionError, match="run --phase c first"):
        R.phase_finalize(cfg)
    R._write_json_atomic(
        cfg.out_root / "analysis_done.json",
        {"analysis_fp": R.analysis_fingerprint(cfg), "digest": {"n_cells": 1}},
    )
    monkeypatch.setattr(
        R, "render_figures", lambda cfg: {"out_dir": "stub", "n_written": 0, "skipped": {}}
    )
    assert R.phase_finalize(cfg) == R.RC_OK
    up = json.loads((cfg.out_root / "upload_done.json").read_text())
    assert up["null_matrices"] == {"uploaded": False, "reason": "--upload none"}
    assert up["figures"]["out_dir"] == "stub"  # threaded into upload_done
    assert up["results_git"]["committed"] is False  # tiny -> never touches git
    sent = json.loads(Path(up["sentinel"]).read_text())
    assert sent["kind"] == "epm:smoke-result"  # tiny/smoke never posts epm:results
    assert set(sent) == {"sentinel_schema_version", "kind", "version", "note"}
    assert sent["sentinel_schema_version"] == 1 and sent["version"] == 1


def test_phase_finalize_renders_figures_end_to_end(tmp_path, bank, pt):
    """Phase C -> Phase D through the driver seam with the REAL
    render_figures body (unit 3): figures land in the out-root twin under
    tiny (never repo figures/), DV3-dependent figures record their skips in
    the manifest, and upload_done carries the figure record."""
    cfg = _cfg(tmp_path, upload_mode="none")
    _seed_phase_c_gates(cfg)
    _seed_tiny_driver_inputs(cfg, bank, pt, tmp_path)
    assert R.phase_analysis(cfg) == R.RC_OK
    assert R.phase_finalize(cfg) == R.RC_OK
    up = json.loads((cfg.out_root / "upload_done.json").read_text())
    figs = up["figures"]
    assert figs["n_written"] >= 2, figs
    assert str(cfg.out_root) in figs["out_dir"]  # tiny twin, never repo figures/
    twin = Path(figs["out_dir"])
    assert (twin / "hero2_shift_ratio_per_type.png").stat().st_size > 0
    assert (twin / "hero2_shift_ratio_per_type.meta.json").exists()
    # tiny run skipped DV3 upstream -> DV3-dependent figures record skips
    assert "hero1_per_type_2afc" in figs["skipped"]
    assert "tiny" in figs["skipped"]["hero1_per_type_2afc"]
    assert up["results_git"]["committed"] is False


def test_write_results_sentinel_production_kind(tmp_path):
    cfg = _cfg(tmp_path)
    cfg.tiny = False
    cfg.smoke = False
    cfg.cells = None
    cfg.sentinel_dir_arg = tmp_path / "logs"
    cfg.results_dir_arg = tmp_path / "results"
    path = R.write_results_sentinel(cfg, {"n_cells": 39})
    payload = json.loads(path.read_text())
    assert payload["kind"] == "epm:results"
    assert "39" in payload["note"]


def test_commit_results_git_skips_smoke_and_tiny(tmp_path):
    cfg = _cfg(tmp_path)
    rec = R.commit_results_git(cfg)
    assert rec["committed"] is False and "smoke/tiny" in rec["reason"]


def test_commit_results_git_requires_figures_dir(tmp_path, monkeypatch):
    """Unit-3 invariant: the production commit fails loud when the Phase-D
    figure render did not populate figures/issue_2215 (the unit-2 explicit
    logged skip is retired — an absent dir now means broken wiring). The
    assert fires BEFORE any git call, so no real repo is touched."""
    repo = tmp_path / "repo"
    (repo / "eval_results" / "issue_2215").mkdir(parents=True)
    cfg = _cfg(tmp_path, results_dir_arg=repo / "eval_results" / "issue_2215")
    cfg.tiny = False
    cfg.smoke = False
    cfg.cells = None
    monkeypatch.setattr(R, "_repo_root", lambda: repo)
    with pytest.raises(AssertionError, match="renders figures"):
        R.commit_results_git(cfg)
