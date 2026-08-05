"""CPU-only unit tests for the issue #2094 P7 analysis + P8 stage-2 drivers.

No model, no GPU, no network: f-table assembly (coherence gating / marking /
companions), the cell-coverage set-check, per-TRAIN-FOLD PC bases, the
held-out context-family fold, batched-vs-naive bootstrap equivalence, the
stage-2 selection restriction, donor stratification annotation, the stage-2
row contract (unit-D walker fields), the additivity combo builder, and the
generate_batch history-render seam pin (the conv-prefix render fix).
"""

from __future__ import annotations

import ast
import inspect
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "scripts"))

import issue2094_analysis as A  # noqa: E402
import issue2094_stage2 as S2  # noqa: E402

from explore_persona_space.experiments.issue1415 import steering  # noqa: E402
from explore_persona_space.experiments.issue2094 import bank as BANK  # noqa: E402

H = 8  # tiny hidden dim for synthetic V_a tensors (28 real layers kept)
K_ANCHOR = 4


@pytest.fixture(scope="module")
def pairs() -> list[BANK.Pair]:
    return BANK.build_pairs()


@pytest.fixture(scope="module")
def pairs_by_id(pairs):
    return {p.pair_id: p for p in pairs}


# ── read-layer rule + donor annotation ─────────────────────────────────


def test_read_layer_rule():
    assert A.read_layer_for((14,)) == (26, False)
    assert A.read_layer_for(tuple(range(14, 21))) == (26, False)  # joint_mid
    assert A.read_layer_for((26,)) == (27, True)
    assert A.read_layer_for((27,)) == (27, True)
    assert A.read_layer_for(tuple(range(28))) == (27, True)  # joint_all


def test_annotate_donor_stratification():
    base = {"arm": "null", "context_a": "persona__q1", "context_b": "conv__q1"}
    b_row = A.annotate_donor({**base, "vec_type": "B"})
    assert b_row == {"donor_kind": "typeB-centroid-swap", "donor_antiparallel": True}
    b2 = A.annotate_donor(
        {"arm": "null", "vec_type": "B", "context_a": "bare__q1", "context_b": "persona__q1"}
    )
    assert b2["donor_antiparallel"] is False
    a_row = A.annotate_donor({**base, "vec_type": "A"})
    assert a_row == {"donor_kind": "typeA-derangement", "donor_antiparallel": False}
    steered = A.annotate_donor({**base, "arm": "steered", "vec_type": "B"})
    assert steered == {"donor_kind": None, "donor_antiparallel": None}


# ── f-table assembly fixtures ──────────────────────────────────────────


def _mk_anchor_va(contexts: list[str], seed: int = 0) -> dict:
    g = torch.Generator().manual_seed(seed)
    out = {}
    for cid in contexts:
        out[cid] = {
            "span": torch.randn(K_ANCHOR, 28, H, generator=g),
            "tail": torch.randn(K_ANCHOR, 28, H, generator=g),
            "draws": list(range(K_ANCHOR)),
        }
    return out


def _mk_grid_row(pair: BANK.Pair, *, arm="steered", cap_hit=False, **over) -> dict:
    row = {
        "block_key": f"ce|L14|a1|A|{arm}",
        "slot": "ce",
        "layer_variant": "L14",
        "layers": [14],
        "dose": "a1",
        "alpha": 1.0,
        "vec_type": "A",
        "arm": arm,
        "pair_id": pair.pair_id,
        "setting": pair.setting,
        "context_a": pair.a,
        "context_b": pair.b,
        "donor_pair_id": None,
        "cap_hit": cap_hit,
    }
    row.update(over)
    return row


def _mk_shard_va(n: int, seed: int = 1) -> dict:
    g = torch.Generator().manual_seed(seed)
    return {"index": None, "va_span": torch.randn(n, 28, H, generator=g), "empty_rows": []}


def _mk_lookups(rows, coh_scores, beh_scores) -> A.JudgeLookups:
    """``beh_scores``: {(row_index, kind, side): score}."""
    lk = A.JudgeLookups()
    for row, coh in zip(rows, coh_scores, strict=True):
        lk.grid_coh[(row["block_key"], row["pair_id"])] = coh
    for (idx, kind, side), score in beh_scores.items():
        row = rows[idx]
        lk.grid_beh[(row["block_key"], row["pair_id"], kind, side)] = score
    return lk


def _pair_stats_for(pair: BANK.Pair, kind: str, floor=0.1, ceiling=0.9) -> dict:
    return {
        (pair.pair_id, kind): {
            "pair_id": pair.pair_id,
            "setting": pair.setting,
            "kind": kind,
            "context_a": pair.a,
            "context_b": pair.b,
            "floor": {"mean": floor, "n": K_ANCHOR, "n_incoherent": 0, "n_judge_missing": 0},
            "ceiling": {"mean": ceiling, "n": K_ANCHOR, "n_incoherent": 0, "n_judge_missing": 0},
            "separation": ceiling - floor,
        }
    }


def _assemble(rows, lk, pair_stats, pairs_by_id, seed=1):
    shard_va = _mk_shard_va(len(rows), seed=seed)
    shard_va["index"] = [{"pair_id": r["pair_id"], "context_a": r["context_a"]} for r in rows]
    ctxs = sorted({r["context_a"] for r in rows} | {r["context_b"] for r in rows})
    anchor_va = _mk_anchor_va(ctxs)
    return A.assemble_shard_rows(
        rows, shard_va, lk, pair_stats, anchor_va, pairs_by_id, profiles=False
    )


def test_ftable_coherent_gating_marks_never_suppresses(pairs, pairs_by_id):
    mp = [p for p in pairs if p.setting == "matched_prefix"][:2]
    rows = [_mk_grid_row(mp[0]), _mk_grid_row(mp[1])]
    beh = {}
    stats = {}
    for i, p in enumerate(mp):
        beh[(i, "query", "a")] = 20.0
        beh[(i, "query", "b")] = 80.0
        stats.update(_pair_stats_for(p, "query"))
    lk = _mk_lookups(rows, [90.0, 10.0], beh)  # row 2 is INCOHERENT (10 <= 60)
    out = _assemble(rows, lk, stats, pairs_by_id)
    assert len(out) == 2  # marked, never dropped
    ok, bad = out
    assert ok["coherent"] is True and ok["excluded_incoherent"] is False
    assert ok["n_coherent"] == 1 and ok["n_total"] == 1
    assert bad["coherent"] is False and bad["excluded_incoherent"] is True
    assert bad["n_coherent"] == 0
    # gated values null; raw companions retained for the record.
    assert bad["f_act"] is None and bad["f_act_raw"] is not None
    assert bad["f_beh"]["query"]["f_beh"] is None
    assert "excluded_incoherent_raw" in bad["f_beh"]["query"]


def test_ftable_fbeh_matches_hand_computed(pairs, pairs_by_id):
    p = next(p for p in pairs if p.setting == "matched_prefix")
    rows = [_mk_grid_row(p)]
    lk = _mk_lookups(rows, [95.0], {(0, "query", "a"): 20.0, (0, "query", "b"): 80.0})
    out = _assemble(rows, lk, _pair_stats_for(p, "query"), pairs_by_id)
    rec = out[0]["f_beh"]["query"]
    # delta_patched = (80-20)/100 = 0.6; f_beh = (0.6-0.1)/(0.9-0.1) = 0.625
    assert rec["delta_patched"] == pytest.approx(0.6)
    assert rec["f_beh"] == pytest.approx(0.625, abs=1e-6)
    assert rec["contrast"] == pytest.approx(0.5, abs=1e-6)
    assert rec["denominator"] == pytest.approx(0.8, abs=1e-6)
    assert out[0]["primary_kind"] == "query"


def test_ftable_degenerate_denominator_flagged_not_coerced(pairs, pairs_by_id):
    p = next(p for p in pairs if p.setting == "matched_prefix")
    rows = [_mk_grid_row(p)]
    lk = _mk_lookups(rows, [95.0], {(0, "query", "a"): 20.0, (0, "query", "b"): 80.0})
    out = _assemble(rows, lk, _pair_stats_for(p, "query", floor=0.5, ceiling=0.5), pairs_by_id)
    rec = out[0]["f_beh"]["query"]
    assert rec["f_beh"] is None  # NaN -> null, never a coerced number
    assert rec["degenerate_denominator"] is True
    assert rec["contrast"] == pytest.approx(0.1, abs=1e-6)  # companion stays valid


def test_ftable_cap_hit_next_to_incoherence(pairs, pairs_by_id):
    p = next(p for p in pairs if p.setting == "matched_prefix")
    rows = [_mk_grid_row(p, cap_hit=True)]
    lk = _mk_lookups(rows, [10.0], {})
    out = _assemble(rows, lk, {}, pairs_by_id)
    row = out[0]
    assert row["cap_hit"] is True  # counted NEXT TO, never blended with
    assert row["excluded_incoherent"] is True


def test_ftable_judge_missing_tagged(pairs, pairs_by_id):
    p = next(p for p in pairs if p.setting == "matched_prefix")
    rows = [_mk_grid_row(p)]
    lk = _mk_lookups(rows, [95.0], {(0, "query", "a"): 20.0})  # side b dropped
    out = _assemble(rows, lk, _pair_stats_for(p, "query"), pairs_by_id)
    assert out[0]["f_beh"]["query"] == {"f_beh": None, "missing": "judge_dropped"}


def test_ftable_cross_setting_carries_both_kinds(pairs, pairs_by_id):
    p = next(p for p in pairs if p.setting == "cross")
    rows = [_mk_grid_row(p)]
    beh = {
        (0, "query", "a"): 10.0,
        (0, "query", "b"): 90.0,
        (0, "prefix", "a"): 30.0,
        (0, "prefix", "b"): 70.0,
    }
    stats = {**_pair_stats_for(p, "query"), **_pair_stats_for(p, "prefix")}
    lk = _mk_lookups(rows, [95.0], beh)
    out = _assemble(rows, lk, stats, pairs_by_id)
    assert set(out[0]["f_beh"]) == {"query", "prefix"}
    assert out[0]["primary_kind"] is None  # cross reports BOTH (Result 2c)


def test_read_layer_marking_in_rows(pairs, pairs_by_id):
    p = next(p for p in pairs if p.setting == "matched_prefix")
    rows = [
        _mk_grid_row(p),
        _mk_grid_row(p, block_key="ce|L27|a1|A|steered", layer_variant="L27", layers=[27]),
    ]
    lk = _mk_lookups(rows, [95.0, 95.0], {})
    out = _assemble(rows, lk, {}, pairs_by_id)
    assert (out[0]["read_layer"], out[0]["read_layer_marked"]) == (26, False)
    assert (out[1]["read_layer"], out[1]["read_layer_marked"]) == (27, True)


# ── coverage gate ──────────────────────────────────────────────────────


def test_coverage_check_refuses_on_mismatch():
    expected = {("b1", "p1"), ("b1", "p2"), ("b2", "p1")}
    ok = A.coverage_check(set(expected), expected)
    assert ok["passed"] is True
    missing = A.coverage_check({("b1", "p1")}, expected)
    assert missing["passed"] is False and missing["n_missing"] == 2
    extra = A.coverage_check(expected | {("b9", "p9")}, expected)
    assert extra["passed"] is False and extra["n_extra"] == 1
    assert A.RC_COVERAGE_GATE != A.RC_OK  # distinct rc for the gate (plan §7)


# ── folds + PC basis ───────────────────────────────────────────────────


def test_group_kfold_pairs_partition(pairs):
    ids = sorted({p.pair_id for p in pairs})
    folds = A.group_kfold_pairs(ids, A.N_PAIR_FOLDS, A.FOLD_SEED)
    assert len(folds) == A.N_PAIR_FOLDS
    flat = [i for f in folds for i in f]
    assert sorted(flat) == ids  # disjoint cover
    assert folds == A.group_kfold_pairs(ids, A.N_PAIR_FOLDS, A.FOLD_SEED)  # deterministic


def test_family_folds_exclude_touching_pairs(pairs):
    folds = A.family_folds(pairs)
    assert len(folds) == 15
    for fold in folds:
        c = fold["context"]
        for pid in fold["test"]:
            p = next(q for q in pairs if q.pair_id == pid)
            assert c in (p.a, p.b)
        for pid in fold["train"]:
            p = next(q for q in pairs if q.pair_id == pid)
            assert p.a != c and p.b != c  # train touches NEITHER endpoint


def test_pc_basis_uses_train_rows_only(monkeypatch):
    """A huge outlier direction living ONLY in the test fold must not enter the
    per-fold PC basis (the statistics critic's fold-leakage fix)."""
    rng = np.random.default_rng(0)
    n_pairs, d = 8, 12
    pair_ids = [f"p{i}" for i in range(n_pairs)]
    x = np.repeat(rng.standard_normal((n_pairs, d)), 2, axis=0)
    row_pairs = [pid for pid in pair_ids for _ in range(2)]
    outlier = np.zeros(d)
    outlier[-1] = 1.0
    test_fold = ["p0", "p1"]
    for i, pid in enumerate(row_pairs):
        if pid in test_fold:
            x[i] = 1e6 * outlier
    y = x @ rng.standard_normal((d, d)) * 1e-3

    seen: list[np.ndarray] = []
    orig = A.pc_basis

    def spy(x_train, k):
        seen.append(x_train.copy())
        return orig(x_train, k)

    monkeypatch.setattr(A, "pc_basis", spy)
    A.fit_family_folds(x, y, row_pairs, [test_fold], pc_dim=4)
    assert seen, "pc_basis never called"
    for x_train in seen:
        assert np.abs(x_train[:, -1]).max() < 1e5  # no test-fold outlier row in the basis input
    _mu, basis = orig(seen[0], 4)
    assert np.abs(basis.T @ outlier).max() < 0.5  # outlier direction not in the basis


def test_fit_family_recovers_linear_map(pairs_by_id):
    """Synthetic exactly-linear y = xW: PC-ridge OOF R2 ~ 1; identity+bias worse."""
    rng = np.random.default_rng(1)
    n_pairs, d, rank = 12, 16, 5
    deltas = rng.standard_normal((n_pairs, rank)) @ rng.standard_normal((rank, d))
    alphas = (0.5, 1.0, 2.0, 4.0)
    pair_ids = [f"pp{i}" for i in range(n_pairs)]
    x = np.stack([a * deltas[i] for i in range(n_pairs) for a in alphas])
    row_pairs = [pair_ids[i] for i in range(n_pairs) for _ in alphas]
    w_true = rng.standard_normal((d, d))
    y = x @ w_true + 0.5
    folds = A.group_kfold_pairs(pair_ids, 4, A.FOLD_SEED)
    res = A.fit_family_folds(x, y, row_pairs, folds, pc_dim=8)
    assert res["pooled_r2"] > 0.95
    assert res["pooled_r2_identity_bias"] < res["pooled_r2"]
    assert len(res["selected_lambdas"]) == 4


# ── bootstrap ──────────────────────────────────────────────────────────


def test_bootstrap_batched_equals_naive():
    rng = np.random.default_rng(2)
    values = rng.standard_normal((7, 3))
    values[0, 1] = np.nan
    values[3:5, 2] = np.nan
    b = A.bootstrap_family_means_batched(values, 50, seed=9, block=2000)
    n = A._bootstrap_family_means_naive(values, 50, seed=9)
    assert np.allclose(b, n, equal_nan=True)


def test_bootstrap_nan_handling():
    values = np.array([[1.0, np.nan], [3.0, np.nan], [5.0, 2.0]])
    boots = A.bootstrap_family_means_batched(values, 200, seed=3)
    col0 = boots[:, 0]
    assert np.isfinite(col0).all()
    assert col0.min() >= 1.0 and col0.max() <= 5.0
    col1 = boots[:, 1]  # only pair 2 carries family 1: mean is 2.0 or NaN (pair undrawn)
    finite = col1[np.isfinite(col1)]
    assert finite.size > 0 and np.allclose(finite, 2.0)


# ── stage-2 selection ──────────────────────────────────────────────────


def _sel_row(setting, slot, variant, f_act, pair_id, dose="a1", vec="A"):
    return {
        "arm": "steered",
        "setting": setting,
        "slot": slot,
        "layer_variant": variant,
        "dose": dose,
        "vec_type": vec,
        "pair_id": pair_id,
        "f_act": f_act,
        "f_beh": {"query": {"f_beh": f_act / 2 if f_act is not None else None}},
    }


def test_select_stage2_never_picks_off_restriction_layer():
    rows = []
    for i in range(5):
        # L20 (OFF the ce banked set) carries the BEST raw values...
        rows.append(_sel_row("matched_prefix", "ce", "L20", 5.0 + i, f"p{i}"))
        # ...L14 (in-set) is lower but must win under the restriction.
        rows.append(_sel_row("matched_prefix", "ce", "L14", 1.0 + 0.1 * i, f"p{i}"))
    sel = A.select_best_cells(rows)
    for cell in sel["cells"]:
        assert int(cell["layer_variant"][1:]) in A.STAGE2_LAYER_RESTRICTION[cell["slot"]]
    picked = sel["selections"]["matched_prefix|activation"]
    assert picked["layer_variant"] == "L14"
    assert sel["post_selection"] is True


def test_select_stage2_cap_and_dedupe():
    rows = []
    for setting in ("matched_prefix", "matched_query", "cross"):
        for i in range(5):
            rows.append(_sel_row(setting, "ce", "L14", 2.0, f"{setting}-p{i}"))
    sel = A.select_best_cells(rows)
    assert len(sel["cells"]) <= A.STAGE2_MAX_CELLS
    # activation + behavior pick the SAME cell per setting here -> deduped with
    # both levels recorded on one cell.
    for cell in sel["cells"]:
        assert len(cell["selected_for"]) == 2


def test_select_stage2_min_pairs_floor():
    rows = [_sel_row("matched_prefix", "ce", "L19", 99.0, "only-one")]  # 1 pair < floor
    rows += [_sel_row("matched_prefix", "ce", "L14", 1.0, f"p{i}") for i in range(4)]
    sel = A.select_best_cells(rows, min_pairs=3)
    assert sel["selections"]["matched_prefix|activation"]["layer_variant"] == "L14"


# ── stage-2 rows + additivity ──────────────────────────────────────────


def test_stage2_rows_satisfy_unit_d_walker_contract(pairs):
    cell = {
        "setting": "matched_prefix",
        "slot": "ce",
        "layer_variant": "L14",
        "dose": "a1",
        "vec_type": "A",
    }
    mp = [p for p in pairs if p.setting == "matched_prefix"][:2]
    pair_cells = [
        {
            "pair_id": p.pair_id,
            "setting": p.setting,
            "context_a": p.a,
            "context_b": p.b,
            "positions": [10],
        }
        for p in mp
    ]
    draws = 3
    texts = [[f"t{b}{d}" for d in range(draws)] for b in range(len(mp))]
    n_tok = [5] * (len(mp) * draws)
    rows = S2.stage2_rows_for_cell(
        cell, S2.cell_key(cell), pair_cells, texts, n_tok, 1024, 1.0, "add", "add", (14,), 42
    )
    assert len(rows) == len(mp) * draws
    for r in rows:
        for req in ("pair_id", "setting", "text", "draw"):  # unit-D _STAGE2_REQUIRED
            assert req in r, req
        assert r["cell"] == S2.cell_key(cell)
        assert r["post_selection"] is True
        assert r["temperature"] == 1.0
        assert r["seed"] == 42 + r["draw"]
    # flattening parity with the capture pass: pair-major, draw-minor.
    assert [r["text"] for r in rows] == [t for ts in texts for t in ts]


def test_stage2_cell_restriction_assert():
    with pytest.raises(AssertionError):
        S2._check_cell_restriction(
            {"slot": "ce", "layer_variant": "L20", "dose": "a1", "vec_type": "A"}, tiny=False
        )
    S2._check_cell_restriction(
        {"slot": "pe", "layer_variant": "L26", "dose": "a1", "vec_type": "A"}, tiny=False
    )


def test_additivity_combos_share_recipient_context(pairs):
    combos = S2.additivity_combos(pairs)
    assert len(combos) == 6
    by_id = {p.pair_id: p for p in pairs}
    prefixes = set()
    for c in combos:
        p1, p2 = by_id[c["pair_1"]], by_id[c["pair_2"]]
        assert p1.setting == p2.setting == "matched_prefix"
        assert p1.a == p2.a == c["context_a"]  # both edits apply to ONE recipient
        assert p1.pair_id != p2.pair_id
        prefixes.add(c["context_a"].split("__")[0])
    assert prefixes == {"bare", "persona", "conv"}  # round-robin coverage
    assert len({c["combo_id"] for c in combos}) == 6


# ── the generate_batch history-render seam (unit-C conv-prefix fix) ────


def test_steering_render_drops_history_but_2094_render_keeps_it():
    ctx = BANK.build_contexts()["conv__q1"]
    parent = steering.context_messages(ctx)
    ours = BANK.context_messages_2094(ctx)
    assert len(parent) == 1  # steering silently DROPS the history turns
    assert len(ours) == 3  # user+assistant history + final user turn
    assert ours[-1]["content"] == ctx["user"]


def test_generate_batch_accepts_render_seam():
    sig = inspect.signature(steering.generate_batch)
    assert "render_fn" in sig.parameters and "ids_fn" in sig.parameters
    assert sig.parameters["render_fn"].default is None  # default behavior unchanged


@pytest.mark.parametrize("script", ["issue2094_run.py", "issue2094_stage2.py"])
def test_every_generate_batch_call_threads_2094_render(script):
    """Regression pin (fails pre-fix on issue2094_run.py): EVERY generate_batch
    call in the 2094 drivers passes render_fn + ids_fn — steering's default
    render drops the conv `history`, mis-rendering conv contexts and breaking
    the hook arm() length invariant for conv-prefixed context_a rows."""
    tree = ast.parse((REPO_ROOT / "scripts" / script).read_text(encoding="utf-8"))
    calls = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        fn = node.func
        name = fn.id if isinstance(fn, ast.Name) else getattr(fn, "attr", None)
        if name == "generate_batch":
            calls.append(node)
    assert calls, f"no generate_batch calls found in {script}"
    for call in calls:
        kwargs = {kw.arg for kw in call.keywords}
        assert {"render_fn", "ids_fn"} <= kwargs, (
            f"{script}:{call.lineno}: generate_batch call missing the 2094 render seam "
            f"(render_fn/ids_fn) — conv-history contexts would render WITHOUT their prefix"
        )


# ── fmetrics wiring sanity used by the assembly ────────────────────────


def test_loglog_slope_unity_on_linear_shifts():
    alphas = torch.tensor([0.5, 1.0, 2.0, 4.0], dtype=torch.float64)
    base = torch.randn(H, dtype=torch.float64)
    norms = torch.stack([(a * base).norm() for a in alphas]).unsqueeze(0)
    slope, _ = A.FM.log_log_magnitude_fit(alphas, norms)
    assert float(slope[0]) == pytest.approx(1.0, abs=1e-6)
