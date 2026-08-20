"""#2388 pinned tests for the readout/fits driver (scripts/issue2388_fits.py).

Pins (plan v4):
- the section-4 pre-fit joint-feasibility assertion's five clauses (violations
  RAISE — arithmetic + manifest modes, per-key resolved |U| identity, and the
  QA disjoint variant's GROUP-grain disjointness);
- MF-A map-key resolution (code's OWN generic-only key vs the shared key);
- the section-3 REGISTERED H3 gap definition (mapped = better of
  {arm6_map_proj_e1, arm7_map_ridge_pred}; direct = arm4_ridge_ctx ALONE;
  MEAN over (seed, draw) cells, never a max; registered-cell row filter) —
  including the two sign-flip fixtures that prove the pin binds;
- group-grain properties of the label draws and the permutation exchange;
- the QA effective-split derivation over the banked labeling shape
  (split={train,eval}, rung={train,nqopen,simpleqa} — schema probed from the
  real artifact 2026-08-20).

CPU-only, synthetic fixtures, seconds-scale.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def drv():
    spec = importlib.util.spec_from_file_location(
        "issue2388_fits", REPO_ROOT / "scripts" / "issue2388_fits.py"
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules.setdefault("issue2388_fits", mod)
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------------------
# MF-A map-key resolution
# ---------------------------------------------------------------------------


def test_resolve_map_key_mfa(drv):
    # code's f_U=0 is code's OWN generic-only key; qa/math/mcq share ONE key
    assert drv.resolve_map_key("code", "linear", 0.0) == "linear__code__fu0"
    assert drv.resolve_map_key("code", "mlp", 0.0) == "mlp__code__fu0"
    for s in ("qa", "math", "mcq"):
        assert drv.resolve_map_key(s, "linear", 0.0) == "linear__shared__fu0"
    assert drv.resolve_map_key("math", "mlp", 0.5) == "mlp__math__fu05"
    assert drv.resolve_map_key("code", "linear", 1.0) == "linear__code__fu1"
    assert drv.resolve_map_key("qa", "linear", 0.5, additive=True) == "linear__qa__additive"
    with pytest.raises(AssertionError):
        drv.resolve_map_key("math", "linear", 0.5, additive=True)


# ---------------------------------------------------------------------------
# joint-feasibility assertion — five clauses
# ---------------------------------------------------------------------------


def _manifest_for(drv, surface, u, *, code_u=None):
    out = {}
    for fam in ("linear", "mlp"):
        for fu in drv.FU_CELLS:
            key = drv.resolve_map_key(surface, fam, fu)
            out[key] = {"realized_u": code_u if key.endswith("code__fu0") else u}
    return out


def test_clause_ii_pool_size_is_registered_min_formula(drv):
    # |U|_s = min(8,000, realized train) (plan section 4 part 2); code = realized
    assert drv._pool_size("math", 8750) == 8000
    assert drv._pool_size("math", 42) == 42
    assert drv._pool_size("code", 4218) == 4218
    rep = drv.assert_joint_feasibility({"math": {"train": 100, "dev": 10, "test": 20}})
    assert rep["surfaces"]["math"]["pool"] == 100


def test_clause_v_qa_disjoint_arithmetic_raises(drv):
    # |train| < |U| + L at the registered anchor 2,000
    with pytest.raises(RuntimeError, match="disjoint variant infeasible"):
        drv.assert_joint_feasibility({"qa": {"train": 9000, "dev": 100, "test": 200}})


def test_clause_i_manifest_u_identity_raises(drv):
    man = _manifest_for(drv, "math", 8000)
    man["linear__shared__fu0"]["realized_u"] = 7000  # drifted shared key
    with pytest.raises(RuntimeError, match=r"realized \|U\| differs"):
        drv.assert_joint_feasibility(
            {"math": {"train": 9000, "dev": 100, "test": 200}}, key_manifest=man
        )


def test_clause_i_unresolved_key_raises(drv):
    man = _manifest_for(drv, "math", 8000)
    del man["mlp__math__fu1"]
    with pytest.raises(RuntimeError, match="unresolved map key"):
        drv.assert_joint_feasibility(
            {"math": {"train": 9000, "dev": 100, "test": 200}}, key_manifest=man
        )


def test_feasibility_pass_reports_budgets(drv):
    man = _manifest_for(drv, "qa", 8000)
    rep = drv.assert_joint_feasibility(
        {"qa": {"train": 16000, "dev": 1600, "test": 3200}}, key_manifest=man
    )
    assert rep["mode"] == "manifest"
    assert rep["surfaces"]["qa"]["budgets"][-1] == "full"
    assert 8000 in rep["surfaces"]["qa"]["budgets"]


def test_clause_ii_iii_membership_assert(drv):
    n = 12
    table = drv.SurfaceTable(
        surface="math",
        ctx_ids=[f"c{i}" for i in range(n)],
        dv=np.linspace(0, 1, n),
        split=np.array(["train"] * 8 + ["dev"] * 2 + ["test"] * 2),
        group=np.array([f"g{i}" for i in range(n)]),
        boot_group=np.array([f"b{i}" for i in range(n)]),
        benchmark=np.array(["math_full"] * n),
        level=np.full(n, 1.0),
        category=np.array(["x"] * n),
        z_ctx=np.zeros((1, n, 4), dtype=np.float16),
        z_t1=np.zeros((1, n, 4), dtype=np.float16),
        z_tlast=None,
    )
    drv.assert_partition_membership(table, np.arange(8))  # train-only: OK
    with pytest.raises(RuntimeError, match=r"feasibility\(ii/iii\)"):
        drv.assert_partition_membership(table, np.array([0, 9]))  # dev row in pool


def test_clause_v_group_grain_disjoint_draw(drv):
    # 10 groups x 3 rows; pool = groups g0..g2's rows; draw must avoid them
    groups = np.array([f"g{i // 3}" for i in range(30)])
    train_idx = np.arange(30)
    pool_idx = np.arange(9)  # g0, g1, g2
    draw = drv.qa_disjoint_draw(train_idx, groups, pool_idx, 6, [1, 2, 3])
    assert len(draw) == 6
    assert not (set(groups[draw]) & set(groups[pool_idx]))
    with pytest.raises(RuntimeError, match=r"feasibility\(v\)"):
        drv.qa_disjoint_draw(train_idx, groups, pool_idx, 25, [1, 2, 3])


# ---------------------------------------------------------------------------
# group-grain draw + permutation properties
# ---------------------------------------------------------------------------


def test_group_respecting_draw_properties(drv):
    groups = np.array([f"g{i // 4}" for i in range(40)])  # 10 groups of 4
    train_idx = np.arange(40)
    draw = drv.group_respecting_draw(train_idx, groups, 10, [7, 8, 9])
    assert len(draw) == 10
    assert set(draw) <= set(train_idx)
    # whole groups except at most ONE truncated group
    sizes = [int((groups[draw] == g).sum()) for g in sorted(set(groups[draw]))]
    assert sum(1 for s in sizes if s not in (0, 4)) <= 1
    # full passthrough + infeasible raise
    assert drv.group_respecting_draw(train_idx, groups, "full", [1]).tolist() == list(range(40))
    with pytest.raises(RuntimeError, match="could not reach"):
        drv.group_respecting_draw(train_idx, groups, 41, [1])


def test_group_permuted_targets_exchanges_whole_groups(drv):
    rng = np.random.default_rng(0)
    n = 24
    groups = np.array([f"g{i // 4}" for i in range(n)])  # 6 equal groups
    dv = rng.uniform(size=n)
    rows = np.arange(n)
    out = drv.group_permuted_targets(dv, groups, rows, 8, [1, 2])
    assert out.shape == (n, 8)
    orig_group_sets = {g: frozenset(np.round(dv[groups == g], 12)) for g in set(groups)}
    all_sets = set(orig_group_sets.values())
    for d in range(8):
        col = out[:, d]
        # each group's permuted values are exactly SOME original group's values
        for g in set(groups):
            got = frozenset(np.round(col[groups == g], 12))
            assert got in all_sets
        # the multiset of group-value-sets is a permutation of the original
        assert {frozenset(np.round(col[groups == g], 12)) for g in set(groups)} == all_sets


# ---------------------------------------------------------------------------
# H3 pinned gap definition (plan section 3)
# ---------------------------------------------------------------------------


def _h3_row(arm, rho, seed=0, draw=0, budget=2500, **over):
    row = {
        "arm": arm,
        "budget_l": budget,
        "rho_frozen": rho,
        "u_rung_label": "full",
        "variant": "context_end",
        "config": "config_a",
        "eval_rung": "train",
        "f_u": None,
        "seed": seed,
        "draw": draw,
    }
    row.update(over)
    return row


def _write_all_arms(tmp_path, rows, name="all_arms_spearman.json"):
    p = tmp_path / name
    p.write_text(json.dumps({"arm_rows": rows}))
    return p


def test_h3_gap_mean_not_max_sign_flip(drv, tmp_path):
    """Mean-vs-max sign-flip fixture: under MAX the gap would read +0.15;
    under the pinned MEAN it reads -0.05 — the aggregation pin flips the
    ANSWER on this file, so a max regression cannot pass."""
    rows = [
        _h3_row("arm6_map_proj_e1", 0.5, seed=0),
        _h3_row("arm6_map_proj_e1", 0.1, seed=1),
        _h3_row("arm7_map_ridge_pred", 0.0, seed=0),
        _h3_row("arm7_map_ridge_pred", 0.0, seed=1),
        _h3_row("arm4_ridge_ctx", 0.35, seed=0),
        _h3_row("arm4_ridge_ctx", 0.35, seed=1),
    ]
    out = drv.h3_gap_from_all_arms(_write_all_arms(tmp_path, rows), budget=2500, n_boot=50)
    assert out["headline_mapped_arm"] == "arm6_map_proj_e1"
    assert out["headline_gap"] == pytest.approx(-0.05)
    assert out["aggregation"].startswith("mean-over")
    assert out["n_paired_cells"] == 2


def test_h3_gap_arm5_is_not_direct_sign_flip(drv, tmp_path):
    """arm5_mlp_ctx classification fixture: arm5's rho (0.9) exceeds every
    mapped arm — if arm5 counted as direct the gap would be NEGATIVE; the pin
    (direct = arm4_ridge_ctx ALONE) makes it +0.1."""
    rows = []
    for seed in (0, 1):
        rows += [
            _h3_row("arm6_map_proj_e1", 0.5, seed=seed),
            _h3_row("arm7_map_ridge_pred", 0.45, seed=seed),
            _h3_row("arm4_ridge_ctx", 0.4, seed=seed),
            _h3_row("arm5_mlp_ctx", 0.9, seed=seed),
        ]
    out = drv.h3_gap_from_all_arms(_write_all_arms(tmp_path, rows), budget=2500, n_boot=50)
    assert out["direct_arm"] == "arm4_ridge_ctx"
    assert out["headline_gap"] == pytest.approx(0.1)
    assert out["headline_gap"] > 0  # would be -0.4 under an arm5-as-direct read


def test_h3_gap_registered_cell_filter_binds(drv, tmp_path):
    """Rows off the registered cell (prefix_end variant, composition f_u,
    other u_rung, other budget) carry huge rho and must NOT move the gap."""
    rows = [
        _h3_row("arm6_map_proj_e1", 0.5),
        _h3_row("arm7_map_ridge_pred", 0.4),
        _h3_row("arm4_ridge_ctx", 0.45),
        _h3_row("arm6_map_proj_e1", 5.0, variant="prefix_end"),
        _h3_row("arm6_map_proj_e1", 5.0, f_u=0.5),
        _h3_row("arm6_map_proj_e1", 5.0, u_rung_label="250"),
        _h3_row("arm6_map_proj_e1", 5.0, budget=16000),
        _h3_row("arm4_ridge_ctx", -5.0, eval_rung="eval"),
    ]
    out = drv.h3_gap_from_all_arms(_write_all_arms(tmp_path, rows), budget=2500, n_boot=50)
    assert out["headline_gap"] == pytest.approx(0.05)
    assert out["cell_filter"]["variant"] == "context_end"
    assert out["cell_filter"]["u_rung_label"] == "full"


def test_h3_gap_missing_arm_raises(drv, tmp_path):
    rows = [_h3_row("arm6_map_proj_e1", 0.5), _h3_row("arm4_ridge_ctx", 0.4)]
    with pytest.raises(RuntimeError, match="arms missing"):
        drv.h3_gap_from_all_arms(_write_all_arms(tmp_path, rows), budget=2500, n_boot=50)


def test_h3_gap_selection_inside_bootstrap_draw(drv, tmp_path):
    """The 2-element mapped selection is applied INSIDE each bootstrap draw:
    with arm6/arm7 alternating which is better per (seed, draw) cell, the
    per-draw max exceeds either arm's own resampled mean whenever the draw
    mixes cells — so the boot gap distribution must dominate the per-arm one."""
    rows = []
    for seed, (a6, a7) in enumerate([(0.8, 0.2), (0.2, 0.8)] * 3):
        rows += [
            _h3_row("arm6_map_proj_e1", a6, seed=seed),
            _h3_row("arm7_map_ridge_pred", a7, seed=seed),
            _h3_row("arm4_ridge_ctx", 0.4, seed=seed),
        ]
    out = drv.h3_gap_from_all_arms(_write_all_arms(tmp_path, rows), budget=2500, n_boot=400)
    # point: means are 0.5/0.5 -> headline gap +0.1; the selection-inherited
    # CI must sit at/above the naive per-arm gap (max inside draw >= mean arm)
    assert out["headline_gap"] == pytest.approx(0.1)
    assert out["headline_gap_ci95"][0] >= 0.1 - 1e-9
    assert out["ci_wholly_positive"] is True


def test_load_arm_rows_schema_guard(drv, tmp_path):
    p = tmp_path / "bad.json"
    p.write_text(json.dumps({"headlines": {}, "meta": {}}))
    with pytest.raises(RuntimeError, match="no arm_rows/rows key"):
        drv._load_arm_rows(p)


# ---------------------------------------------------------------------------
# QA effective-split derivation (banked labeling shape)
# ---------------------------------------------------------------------------


def _banked_qa_rows():
    rows = []
    for i in range(30):  # 10 entity groups x 3 questions
        rows.append(
            {
                "context_id": f"hallucination-train-train-{i:06d}",
                "dv": 0.5,
                "split": "train",
                "rung": "train",
                "group_key": f"ent{i // 3}",
            }
        )
    for i in range(5):
        rows.append(
            {
                "context_id": f"hallucination-eval-nqopen-{i:06d}",
                "dv": 0.5,
                "split": "eval",
                "rung": "nqopen",
                "group_key": f"nq{i}",
            }
        )
    for i in range(3):
        rows.append(
            {
                "context_id": f"hallucination-eval-simpleqa-{i:06d}",
                "dv": 0.5,
                "split": "eval",
                "rung": "simpleqa",
                "group_key": f"sq{i}",
            }
        )
    return rows


def test_effective_rows_qa_partition(drv):
    out = drv._effective_rows("qa", _banked_qa_rows())
    by_split: dict[str, list[dict]] = {}
    for r in out:
        by_split.setdefault(r["eff_split"], []).append(r)
    # simpleqa DROPPED; nqopen -> rung1
    assert sum(len(v) for v in by_split.values()) == 35
    assert len(by_split["rung1"]) == 5
    assert all(r["rung"] == "nqopen" for r in by_split["rung1"])
    # 70/10/20 at GROUP grain over the 10 entity groups: 7/1/2 groups
    for name, n_groups in (("train", 7), ("dev", 1), ("test", 2)):
        gs = {r["group_key"] for r in by_split[name]}
        assert len(gs) == n_groups, (name, gs)
        assert len(by_split[name]) == n_groups * 3
    # no group straddles two splits
    seen: dict[str, str] = {}
    for r in out:
        if r["eff_split"] == "rung1":
            continue
        assert seen.setdefault(r["group_key"], r["eff_split"]) == r["eff_split"]
    # deterministic (seeded)
    out2 = drv._effective_rows("qa", _banked_qa_rows())
    assert [r["eff_split"] for r in out2] == [r["eff_split"] for r in out]


def test_effective_rows_new_surface_passthrough_and_guard(drv):
    rows = [
        {"context_id": "a", "dv": 1.0, "split": "train", "group_key": "g"},
        {"context_id": "b", "dv": 0.0, "split": "dev", "group_key": "h"},
    ]
    out = drv._effective_rows("math", rows)
    assert [r["eff_split"] for r in out] == ["train", "dev"]
    with pytest.raises(RuntimeError, match="unexpected split"):
        drv._effective_rows("math", [{"context_id": "c", "dv": 1.0, "split": "eval"}])


# ---------------------------------------------------------------------------
# selector telemetry extraction
# ---------------------------------------------------------------------------


def test_selected_lambda_dof_chunk_mapping(drv):
    telem = [
        {
            "slice_offset": 0,
            "lambda_selected": np.array([[1.0, 9.0], [2.0, 9.0]]),
            "dof_selected": np.array([[10.0, 0.0], [20.0, 0.0]]),
        },
        {
            "slice_offset": 2,
            "lambda_selected": np.array([[3.0, 9.0]]),
            "dof_selected": np.array([[30.0, 0.0]]),
        },
    ]
    assert drv._selected_lambda_dof(telem, 0) == (1.0, 10.0)
    assert drv._selected_lambda_dof(telem, 1) == (2.0, 20.0)
    assert drv._selected_lambda_dof(telem, 2) == (3.0, 30.0)
    assert drv._selected_lambda_dof(telem, 5) == (None, None)


def test_stable_seed_is_process_independent(drv):
    # sha256-derived, never str.hash (PYTHONHASHSEED-salted)
    assert drv._stable_seed("math") == 1487328896
    assert drv._stable_seed("qa") != drv._stable_seed("code")


def test_surface_features_tfidf_train_fold_only(drv):
    """bl_feats: TF-IDF vocabulary comes from TRAIN rows only (no eval-text
    leakage into the fitted vectorizer)."""
    n = 8
    table = drv.SurfaceTable(
        surface="math",
        ctx_ids=[f"c{i}" for i in range(n)],
        dv=np.linspace(0, 1, n),
        split=np.array(["train"] * 5 + ["dev", "test", "test"]),
        group=np.array([f"g{i}" for i in range(n)]),
        boot_group=np.array([f"b{i}" for i in range(n)]),
        benchmark=np.array(["math_full"] * n),
        level=np.array([1.0, 2, 3, 4, 5, 1, 2, 3]),
        category=np.array(["x"] * n),
        z_ctx=np.zeros((1, n, 4), dtype=np.float16),
        z_t1=np.zeros((1, n, 4), dtype=np.float16),
        z_tlast=None,
    )
    table.meta["questions"] = [f"train word{i} shared" for i in range(5)] + [
        "evalonlytoken shared",
        "evalonlytoken shared",
        "evalonlytoken shared",
    ]
    feats = drv._surface_features(table, np.arange(5))
    assert feats.shape[0] == 1 and feats.shape[1] == n
    # the eval-only token is OUT of the trained vocabulary: its rows' TF-IDF
    # block differs from train rows' ONLY via train-vocab terms ("shared")
    # -> columns are len(2) + level onehot(5) + tfidf(train vocab)
    # eval rows must have zero mass on any train-specific word column
    tf = feats[0, :, 7:]  # tfidf block
    # rows 5..7 contain no train-vocab word except "shared": nonzero count == 1
    assert all(int((tf[i] > 0).sum()) == 1 for i in (5, 6, 7))


# ---------------------------------------------------------------------------
# tiny-real CPU end-to-end: maps -> sweep -> select -> bootstrap (+ verdict)
# ---------------------------------------------------------------------------


def _write_store(
    root: Path, ctx_ids, k_roll, d, layers, rng, kinds=("context_end", "t1", "t_last")
):
    root.mkdir(parents=True, exist_ok=True)
    n_rows = len(ctx_ids) * k_roll
    for kind in kinds:
        for ly in range(layers):
            arr = rng.normal(size=(n_rows, d)).astype(np.float16)
            np.save(root / f"{kind}_L{ly:02d}.npy", arr)
    with (root / "row_index.jsonl").open("w") as fh:
        for cid in ctx_ids:
            for k in range(k_roll):
                fh.write(json.dumps({"context_id": cid, "rollout_k": k}) + "\n")


@pytest.fixture(scope="module")
def smoke_env(drv, tmp_path_factory):
    """Synthetic math surface (60 ctx x K=2, d=32, 3 layers) + tiny U store."""
    base = tmp_path_factory.mktemp("i2388smoke")
    rng = np.random.default_rng(0)
    d, layers, k_roll = 32, 3, 2
    ctx_ids = [f"mathfull-algebra-train-{i:05d}" for i in range(60)]
    _write_store(base / "store_2388" / "math_full", ctx_ids, k_roll, d, layers, rng)
    u_ids = [f"u{i:04d}" for i in range(80)]
    _write_store(
        base / "u_store" / "cell_inst_own", u_ids, 1, d, layers, rng, kinds=("context_end", "t1")
    )
    # labeling.json (dv_build shape); group-level 70/10/20 across 20 groups
    rows = []
    splits = ["train"] * 42 + ["dev"] * 6 + ["test"] * 12
    for i, cid in enumerate(ctx_ids):
        verd = [bool(rng.integers(0, 2)) for _ in range(k_roll)]
        dv = sum(verd) / k_roll
        rows.append(
            {
                "context_id": cid,
                "benchmark": "math_full",
                "n_rollouts": k_roll,
                "dv": dv,
                "fractions": {"correct": dv},
                "per_rollout_scores": {f"k{k}": float(v) for k, v in enumerate(verd)},
                "agree_frac": float(rng.uniform(0.5, 1.0)),  # bl_agree input (r2)
                "group_key": cid,
                "split": splits[i],
                "level": int(1 + (i % 5)),
                "subject": "algebra",
                "category": None,
                "rung": "math_full",
            }
        )
    dv_dir = base / "dv" / "math"
    dv_dir.mkdir(parents=True)
    (dv_dir / "labeling.json").write_text(json.dumps({"rows": rows}))
    return {"base": base, "d": d, "layers": layers}


def _argv(env, extra):
    base = env["base"]
    return [
        "--dv-root",
        str(base / "dv"),
        "--fits-root",
        str(base / "fits"),
        "--maps-out",
        str(base / "maps"),
        "--store-root",
        str(base / "store_2388"),
        "--u-store-dir",
        str(base / "u_store"),
        "--layers",
        str(env["layers"]),
        "--hidden-dim",
        str(env["d"]),
        "--smoke",
        *extra,
    ]


def test_e2e_maps_sweep_select_bootstrap(drv, smoke_env):
    """Tiny-real CPU e2e through main(): map fits (linear + MLP + fu05 mix
    incl. the U-pool path), the sweep at one budget (dual n<d route) + full
    (primal route), select freeze, paired group bootstrap."""
    env = smoke_env
    base = env["base"]
    # feasibility FIRST (P1 pre-step shape: no manifest yet -> arithmetic mode)
    rc = drv.main(_argv(env, ["--phase", "feasibility", "--surfaces", "math"]))
    assert rc == 0
    rep = json.loads((base / "fits" / "feasibility_report.json").read_text())
    assert rep["mode"] == "arithmetic"
    assert rep["surfaces"]["math"]["pool"] == 42
    assert "git_commit" in rep["metadata"]
    rc = drv.main(
        _argv(
            env,
            [
                "--phase",
                "maps",
                "--keys",
                "linear__math__fu1",
                "mlp__math__fu1",
                "linear__math__fu05",
            ],
        )
    )
    assert rc == 0
    manifest = json.loads((base / "maps" / "key_manifest.json").read_text())
    assert manifest["linear__math__fu1"]["realized_u"] == 42
    assert manifest["linear__math__fu05"]["composition"]["generic"] == 21
    assert (base / "maps" / "mlp__math__fu1.pt").exists()
    # sweep: one small budget (dual route) + full (primal route), 1 draw
    arms = [
        "arm_ctx",
        "arm_maplin",
        "arm_mapmlp",
        "arm_oracle",
        "arm_oracle_tlast",
        "bl_shufmap",
        "bl_shufmap_mlp",
        "bl_identity",
        "arm_dir_ctx",
        "arm_dir_map",
        "bl_agree",
        "bl_const",
    ]
    rc = drv.main(
        _argv(
            env,
            [
                "--phase",
                "sweep",
                "--surface",
                "math",
                "--budgets",
                "16",
                "full",
                "--n-draws",
                "1",
                "--arms",
                *arms,
            ],
        )
    )
    assert rc == 0
    cells = sorted((base / "fits" / "math" / "cells").glob("*.json"))
    assert len(cells) == 2 * len(arms) + 2  # + arm_ctx_pca companion per cell
    row = json.loads((base / "fits" / "math" / "cells" / "arm_ctx__L16_draw0.json").read_text())
    assert row["dof_cap"] == 0.9
    assert row["selector"]["mode"] == "gcv-dof-capped"
    assert row["selector"]["lambda_selected"] is not None
    assert row["selector"]["dof_selected"] is not None
    assert row["per_eval"]["rung0"]["rho"] == row["per_eval"]["rung0"]["rho"]  # finite/NaN-legal
    assert row["n_train_vs_d"] == [16, env["d"]]
    # rung-1 TRANSFER read (r2): under-floor smoke draw discloses the fallback;
    # the FULL-budget draw (42 train rows, ~25 in levels 1-3) actually REFITS.
    assert row["rung1_fit"] == {"refit": False, "smoke_fallback": True}
    row_full = json.loads(
        (base / "fits" / "math" / "cells" / "arm_ctx__Lfull_draw0.json").read_text()
    )
    assert row_full["rung1_fit"]["refit"] is True
    assert row_full["rung1_fit"]["restriction"] == "levels 1-3 only"
    assert row_full["rung1_fit"]["n_fit_rows"] < row_full["n_draw_rows"]
    # new r2 arms landed cells: shuffled-MLP control + agreement reference row
    row_shufmlp = json.loads(
        (base / "fits" / "math" / "cells" / "bl_shufmap_mlp__L16_draw0.json").read_text()
    )
    assert row_shufmlp["pooling"] == "t1-mapped"
    row_agree = json.loads(
        (base / "fits" / "math" / "cells" / "bl_agree__L16_draw0.json").read_text()
    )
    assert row_agree["selector"]["mode"] == "reference-row (no fit)"
    assert isinstance(row_agree["per_eval"]["dev"]["rho"], float)
    # preds JSONL: real context ids + real y_true for EVERY arm incl. direction
    for arm in ("arm_ctx", "arm_dir_ctx", "arm_ctx_pca"):
        pf = base / "fits" / "math" / "preds" / f"preds_{arm}_L16_draw0.jsonl"
        lines = [json.loads(x) for x in pf.read_text().split("\n") if x.strip()]
        assert lines and all(line["context_id"].startswith("mathfull-") for line in lines)
        assert all(np.isfinite(line["y_true"]) for line in lines)
    # nulls npz per basis
    assert (base / "fits" / "math" / "nulls" / "arm_ctx__L16_draw0__ambient.npz").exists()
    assert (base / "fits" / "math" / "nulls" / "arm_ctx__L16_draw0__pca16.npz").exists()
    # select + bootstrap from the persisted artifacts
    rc = drv.main(_argv(env, ["--phase", "select", "--surface", "math"]))
    assert rc == 0
    sel = json.loads((base / "fits" / "math" / "selection.json").read_text())
    assert sel["cells"]["arm_ctx__L16_draw0"]["selector"]["mode"] == "gcv-dof-capped"
    rc = drv.main(_argv(env, ["--phase", "bootstrap", "--surface", "math"]))
    assert rc == 0
    boot = json.loads((base / "fits" / "math" / "bootstrap_summary.json").read_text())
    assert boot["cells"], "bootstrap produced no cells"
    c0 = boot["cells"][0]
    assert set(c0["arms"]) >= {"arm_ctx", "arm_maplin"}
    assert len(c0["rho_ci"]) == len(c0["arms"])
    assert c0["n_groups"] > 1


def test_e2e_h3_verdict_ordering_gate(drv, smoke_env, tmp_path):
    """phase_h3 verdict + the G4(d) stage-2 refusal: verdict writes
    stage1_verdict.json from fixture recompute roots; stage2 REFUSES before
    the verdict exists."""
    env = smoke_env
    fits_root = tmp_path / "fits"
    h3_root = tmp_path / "h3_recompute"
    for behavior in ("sycophancy", "evil", "hallucination"):
        d = h3_root / behavior / "arm_results"
        d.mkdir(parents=True)
        rows = []
        for seed in (0, 1):
            rows += [
                _h3_row("arm6_map_proj_e1", 0.6, seed=seed),
                _h3_row("arm7_map_ridge_pred", 0.5, seed=seed),
                _h3_row("arm4_ridge_ctx", 0.4, seed=seed),
            ]
        (d / "all_arms_spearman.json").write_text(json.dumps({"arm_rows": rows}))
    argv = [
        "--phase",
        "h3",
        "--h3-step",
        "stage2",
        "--fits-root",
        str(fits_root),
        "--dv-root",
        str(env["base"] / "dv"),
    ]
    with pytest.raises(RuntimeError, match=r"G4\(d\)"):
        drv.main(argv)
    rc = drv.main(
        [
            "--phase",
            "h3",
            "--h3-step",
            "verdict",
            "--fits-root",
            str(fits_root),
        ]
    )
    assert rc == 0
    verdict = json.loads((h3_root / "stage1_verdict.json").read_text())
    assert verdict["recorded_before_correctness_read"] is True
    assert verdict["kill_branch_enabled"] is True
    assert verdict["behaviors"]["sycophancy"]["headline_gap"] == pytest.approx(0.2)
