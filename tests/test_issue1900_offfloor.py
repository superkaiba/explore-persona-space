"""Boundary pins for the #1900 off-floor round (scripts/issue1900_offfloor.py).

Plan v11 registered protections:
1. Subset construction is DETERMINISTIC at the fixed seeds (syc slice 19001,
   syc top-up 1900) and follows the frozen selection rule (score_mean > 0 on
   the SELECTION record; cas parent-contained; imp all-nonzero; syc parent
   nonzero + seeded top-up to the target with shortfall-take-all).
2. F2 fresh-draw isolation: the estimation judge work root is DISJOINT from
   the selection root and the est_/sel_ tag namespaces cannot collide on the
   per-(tag, max_tokens) #1019 checkpoint dirs — a selection draw can never
   be served back to the estimation pass (llm-judging rule 24(ii)).
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "scripts"))


def _off():
    import issue1900_offfloor as off

    return off


def _row(sha: str, draws: list[float] | None) -> dict:
    if draws is None:
        return {"sha": sha, "score_mean": None, "kept_draw_scores": []}
    return {
        "sha": sha,
        "score_mean": sum(draws) / len(draws),
        "kept_draw_scores": draws,
    }


def _records(off) -> tuple[dict, set[str]]:
    """Synthetic selection records: parent = p00..p09; new = n00..n09."""
    parent = {f"p{i:02d}" for i in range(10)}
    cas_rows = [_row(f"p{i:02d}", [10.0, 20.0, 30.0]) for i in range(8)]
    cas_rows += [_row("p08", [0.0, 0.0, 0.0]), _row("p09", None)]
    # imp: 3 parent nonzero (one single-draw, one mean<=5) + 4 new nonzero + zeros
    imp_rows = [
        _row("p00", [0.0, 0.0, 9.0]),  # single-draw nonzero, mean 3.0 <= 5
        _row("p01", [20.0, 30.0, 10.0]),
        _row("p02", [5.0, 5.0, 5.0]),  # mean 5.0 <= 5
        _row("p03", [0.0, 0.0, 0.0]),
        _row("n00", [40.0, 10.0, 10.0]),
        _row("n01", [0.0, 0.0, 3.0]),  # single-draw nonzero, mean 1.0 <= 5
        _row("n02", [15.0, 15.0, 15.0]),
        _row("n03", [8.0, 8.0, 8.0]),
        _row("n04", [0.0, 0.0, 0.0]),
    ]
    # syc: 3 parent nonzero + 6 new nonzero + one zero
    syc_rows = [_row(f"p{i:02d}", [12.0, 12.0, 12.0]) for i in range(3)]
    syc_rows += [_row(f"n{i:02d}", [7.0 + i, 7.0, 7.0]) for i in range(6)]
    syc_rows += [_row("n09", [0.0, 0.0, 0.0])]
    judge = {
        "judge_model": "claude-sonnet-4-5-20250929",
        "n_draws": 3,
        "max_tokens": 400,
        "rubric_sha256": "x",
    }
    recs = {
        "cas": {"rows": cas_rows, "judge": judge, "beh_key": "cas"},
        "imp": {"rows": imp_rows, "judge": judge, "beh_key": "imp"},
        "syc": {"rows": syc_rows, "judge": judge, "beh_key": "syc"},
    }
    return recs, parent


def test_build_subsets_deterministic_and_rule_conformant():
    off = _off()
    recs, parent = _records(off)
    subsets1, report1 = off.build_subsets(recs, parent, syc_target=6, topup_seed=1900)
    subsets2, report2 = off.build_subsets(recs, parent, syc_target=6, topup_seed=1900)
    assert subsets1 == subsets2, "subset construction not deterministic at fixed seeds"
    assert report1 == report2
    # cas: nonzero AND parent-contained (p08 zero + p09 unscored excluded)
    assert subsets1["cas"] == [f"p{i:02d}" for i in range(8)]
    # imp: ALL nonzero (parent + new), zeros excluded
    assert subsets1["imp"] == ["n00", "n01", "n02", "n03", "p00", "p01", "p02"]
    # syc: 3 parent nonzero + seeded top-up of exactly 3 of the 6 new nonzero
    syc = subsets1["syc"]
    assert len(syc) == 6 and {"p00", "p01", "p02"} <= set(syc)
    assert all(s.startswith(("p", "n")) for s in syc) and "n09" not in syc
    # a different top-up seed changes the drawn top-up (same parent core)
    alt, _ = off.build_subsets(recs, parent, syc_target=6, topup_seed=7)
    assert {"p00", "p01", "p02"} <= set(alt["syc"])
    # composition report: registered per-subset counts (plan section 4 (v))
    imp_rep = report1["per_family"]["imp"]
    assert imp_rep["single_draw_nonzero"] == 2  # p00 + n01
    assert imp_rep["mean_le5"] == 3  # p00 (3.0), p02 (5.0), n01 (1.0)
    assert imp_rep["n_parent_rows"] == 3 and imp_rep["n_new_rows"] == 4
    assert report1["low_power_imp"] is True  # 7 < 500 floor -> label, never a kill


def test_syc_topup_shortfall_takes_all_and_reports():
    off = _off()
    recs, parent = _records(off)
    subsets, report = off.build_subsets(recs, parent, syc_target=50, topup_seed=1900)
    assert len(subsets["syc"]) == 9  # 3 parent + ALL 6 new nonzero (shortfall)
    assert report["per_family"]["syc"]["n_new_rows"] == 6


def test_draw_syc_slice_deterministic_without_replacement():
    off = _off()
    cand = [f"c{i:04d}" for i in range(500)]
    s1 = off.draw_syc_slice(cand, n=100, seed=19001)
    s2 = off.draw_syc_slice(list(reversed(cand)), n=100, seed=19001)
    assert s1 == s2, "slice must be order-invariant (sorted) + seed-deterministic"
    assert len(s1) == 100 == len(set(s1)) and set(s1) <= set(cand)
    assert off.draw_syc_slice(cand[:50], n=100) == sorted(cand[:50])  # <= n: all


def test_estimation_work_root_disjoint_from_selection():
    """rule 24(ii): no path under which an estimation judge run can resolve a
    selection-pass #1019 checkpoint (fresh draws by construction)."""
    off = _off()
    for smoke in (False, True):
        roots = off.make_roots(smoke)
        est, sel = roots.est_work_dir.resolve(), roots.sel_work_dir.resolve()
        assert est != sel
        assert not est.is_relative_to(sel) and not sel.is_relative_to(est)
        # judge_unit keys its checkpoint dir on (work_dir, tag, max_tokens):
        # est_/sel_ tag namespaces stay disjoint even under a shared root.
        for fam in off.CONTENT_FAMILIES:
            est_dir = est / f"est_base_{fam}_mt{off.JUDGE_MAX_TOKENS}"
            sel_dir = sel / f"sel_base_{fam}_mt{off.JUDGE_MAX_TOKENS}"
            assert est_dir != sel_dir and est_dir.name != sel_dir.name


def test_rewrite_arm_parquet_survives_dup_sha_frames(tmp_path):
    """r1 Critical 1 regression: the #1768 stores carry duplicate-sha rows
    (16,400 rows / 16,318 unique in the parent parquet; `load_corpus_cell`
    keeps them), so BOTH the parent parquet AND the mapcols parquet can be
    dup-sha. Pre-fix, `tab["sha"].map(mc[col])` on the non-unique mapcols
    index raised pandas InvalidIndexError in production F3 (smoke fixtures
    were sha-unique by construction, so the smoke could not catch it)."""
    import pandas as pd

    off = _off()
    roots = off.Roots(data=tmp_path / "data", evalr=tmp_path / "eval", smoke=True)
    arm = "imp-pers-con-lr3e5-s42"
    # parent parquet: shas a, b, b, c — b duplicated (never a subset member)
    parent = pd.DataFrame(
        {
            "sha": ["a", "b", "b", "c"],
            "in_judge_subset": [False, False, False, False],
            **{col: [10.0, 20.0, 21.0, 30.0] for col in off.MAP_COLS},
        }
    )
    ppath = roots.p1_root / "parent_tables" / f"{arm}_L{off.LAYER}.parquet"
    ppath.parent.mkdir(parents=True)
    parent.to_parquet(ppath, index=False)
    # mapcols parquet: dup-sha too (the pre-dedup producer shape)
    mc = pd.DataFrame(
        {"sha": ["a", "b", "b", "c"], **{col: [1.0, 2.0, 3.0, 4.0] for col in off.MAP_COLS}}
    )
    mpath = roots.columns_dir / f"{arm}_L{off.LAYER}_mapcols.parquet"
    mpath.parent.mkdir(parents=True)
    mc.to_parquet(mpath, index=False)

    out = off.rewrite_arm_parquet(roots, {"arm_id": arm}, {"a", "c"})  # pre-fix: raises
    tab = pd.read_parquet(out)
    assert len(tab) == 4, "dup rows must be preserved in the rewritten parquet"
    assert list(tab["in_judge_subset"]) == [True, False, False, True]
    for col in off.MAP_COLS:
        assert tab.loc[0, col] == 1.0 and tab.loc[3, col] == 4.0  # refit values on subset rows
        assert tab.loc[0, f"{col}_parentmap"] == 10.0  # parent values preserved (record-only)
        assert tab.loc[1, col] == tab.loc[2, col] == 2.0  # keep="first"; non-raced dup rows


def test_planted_selection_checkpoint_never_visible_from_estimation_root(tmp_path):
    """A selection-draw checkpoint planted under the selection root is not
    resolvable through the estimation root's namespace (path-level isolation)."""
    off = _off()
    roots = off.make_roots(True)
    rel_sel = roots.sel_work_dir.relative_to(roots.data)
    rel_est = roots.est_work_dir.relative_to(roots.data)
    sel = tmp_path / rel_sel / f"sel_base_imp_mt{off.JUDGE_MAX_TOKENS}"
    sel.mkdir(parents=True)
    (sel / "judge_raw.json").write_text("{}")
    est = tmp_path / rel_est / f"est_base_imp_mt{off.JUDGE_MAX_TOKENS}"
    assert not est.exists()
    assert (
        not list((tmp_path / rel_est).glob("**/judge_raw.json"))
        if (tmp_path / rel_est).exists()
        else True
    )
