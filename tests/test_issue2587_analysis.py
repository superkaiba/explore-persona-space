"""CPU pins for scripts/issue2587_analysis.py (issue #2587 unit 5b).

No network, no GPU: every fixture is synthetic; the PRODUCTION bodies
(compute_side, crossmodel_contrasts, compute_h1, load_fire, the numeric
helpers) execute for real (code-style.md § one production-body test per
seam-stubbed function — nothing here is seam-stubbed; the boundary is the
synthetic Stores/bank fixtures).

Pinned cross-unit constraints (plan v3 §4.4 constraint list):
  1. install orientation handled PER PAIR CLASS (pilot a=value,b=bare vs
     parent a=bare,b=value) — test_constraint1_install_orientation_per_class
  2. merged pilot pairs group on ONE key (cell==axis) —
     test_constraint2_axis_key
  3. embed engine-version parity ASSERTED — test_constraint3_engine_parity
  4. store layer columns resolved via the store's own ``layers`` list
     (captured[L] == hidden_states[L+1]) — test_constraint4_store_col
  5. no cross-model contrast at a twin layer —
     test_constraint5_frozen_layer_pair
Plus: the frozen-L* read (never re-argmaxed), the unique primary_h2_7b_arm,
the H1 three-branch lattice (disjoint + exhaustive), the exact-DP Spearman
permutation p vs brute force, and the carrier-clustered bootstrap machinery
vs a brute-force reference.
"""

from __future__ import annotations

import itertools
import json
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "scripts"))

import issue2587_analysis as AN  # noqa: E402

# ── tiny synthetic world ───────────────────────────────────────────────

CARRIERS = ["c01", "c02", "c03"]
VALS = ["v1", "v2", "v3"]
LANGS = ["de", "fr", "zh"]
OW_SLUGS = ["owa", "owb", "owc", "owd"]


def _parent_pairs_and_contexts() -> tuple[dict, list[dict]]:
    contexts: dict = {}
    pairs: list[dict] = []

    def ctx(cid: str, cell: str, carrier: str) -> None:
        contexts[cid] = {"id": cid, "cell": cell, "carrier": carrier}

    for c in CARRIERS:
        ctx(f"tx::E::{c}", "toneX", c)
        ctx(f"q::E::{c}", "query", c)
        ctx(f"q::Ep::{c}", "query", c)
        for v in VALS:
            ctx(f"tx::{v}::{c}", "toneX", c)
            ctx(f"tx::{v}p::{c}", "toneX", c)

    pid = [0]

    def add(pc: str, cell: str, va: str, vb: str, a: str, b: str, carrier: str, **extra) -> None:
        p = {
            "pair_id": f"P::{pc}::{va}-{vb}::{carrier}",
            "pair_class": pc,
            "cell": cell,
            "axis": cell if cell != "query" else pc,
            "value_a": va,
            "value_b": vb,
            "a": a,
            "b": b,
            "carrier": carrier,
            "changed_tokens": 1 + (pid[0] % 3),
        }
        p.update(extra)
        pairs.append(p)
        pid[0] += 1

    for c in CARRIERS:
        for vi, vj in itertools.combinations(VALS, 2):
            add("swap", "toneX", vi, vj, f"tx::{vi}::{c}", f"tx::{vj}::{c}", c)
            add("famswap", "toneX", f"{vi}p", f"{vj}p", f"tx::{vi}p::{c}", f"tx::{vj}p::{c}", c)
        for v in VALS:
            add("instruction_paraphrase", "toneX", v, f"{v}p", f"tx::{v}::{c}", f"tx::{v}p::{c}", c)
            # PARENT install orientation: a = bare-E context, b = value context
            add("install", "toneX", "E", v, f"tx::E::{c}", f"tx::{v}::{c}", c)
        add("query_paraphrase", "query", "E", "Ep", f"q::E::{c}", f"q::Ep::{c}", c)
    for ca, cb in itertools.combinations(CARRIERS, 2):
        add(
            "query_content",
            "query",
            ca,
            cb,
            f"q::E::{ca}",
            f"q::E::{cb}",
            f"{ca}-{cb}",
            carrier_a=ca,
            carrier_b=cb,
        )
    return contexts, pairs


def _pilot_pairs_and_contexts() -> tuple[dict, list[dict]]:
    contexts: dict = {}
    pairs: list[dict] = []

    def ctx(cid: str, cell: str, carrier: str) -> None:
        contexts[cid] = {"id": cid, "cell": cell, "carrier": carrier}

    def add(pc: str, cell: str, va: str, vb: str, a: str, b: str, carrier: str) -> None:
        pairs.append(
            {
                "pair_id": f"P::{pc}::{va}-{vb}::{carrier}",
                "pair_class": pc,
                "cell": cell,
                "axis": cell,
                "value_a": va,
                "value_b": vb,
                "a": a,
                "b": b,
                "carrier": carrier,
                "changed_tokens": 2,
            }
        )

    for c in CARRIERS:
        ctx(f"al::bare::{c}", "answer_language", c)
        for lang in LANGS:
            ctx(f"al::{lang}::{c}", "answer_language", c)
        for vi, vj in itertools.combinations(LANGS, 2):
            add("swap", "answer_language", vi, vj, f"al::{vi}::{c}", f"al::{vj}::{c}", c)
        for lang in LANGS:
            # PILOT install orientation (constraint 1): a = VALUE, b = bare —
            # the OPPOSITE of the parent convention above.
            add("install", "answer_language", lang, "bare", f"al::{lang}::{c}", f"al::bare::{c}", c)
    for k, s in enumerate(OW_SLUGS):
        c = CARRIERS[k % len(CARRIERS)]
        ctx(f"ow::{s}a::{c}", "query_content_oneword", c)
        ctx(f"ow::{s}b::{c}", "query_content_oneword", c)
        add(
            "query_content_oneword",
            "query_content_oneword",
            f"{s}a",
            f"{s}b",
            f"ow::{s}a::{c}",
            f"ow::{s}b::{c}",
            c,
        )
    return contexts, pairs


def _banks() -> tuple[dict, dict]:
    pctx, ppairs = _parent_pairs_and_contexts()
    ictx, ipairs = _pilot_pairs_and_contexts()
    bank7 = {"contexts": pctx, "pairs": ppairs, "n_contexts": len(pctx), "n_pairs": len(ppairs)}
    ctx9 = {**pctx, **ictx}
    pairs9 = ppairs + ipairs
    bank9 = {"contexts": ctx9, "pairs": pairs9, "n_contexts": len(ctx9), "n_pairs": len(pairs9)}
    return bank9, bank7


def _stores(bank: dict, layers: tuple[int, ...], d: int, primary: int, engine, seed: int):
    rng = np.random.default_rng(seed)
    ctx_ids = sorted(bank["contexts"])
    n = len(ctx_ids)
    k = 4
    tail_draws = rng.standard_normal((n, k, d)).astype(np.float32)
    va_tail = {}
    va_span = {}
    vc = {}
    for layer in layers:
        base = rng.standard_normal((n, d))
        va_tail[layer] = base if layer != primary else tail_draws.mean(axis=1).astype(np.float64)
        va_span[layer] = base + 0.1 * rng.standard_normal((n, d))
        vc[layer] = rng.standard_normal((n, d))
    return AN.Stores(
        ctx_ids=ctx_ids,
        row_of={cid: i for i, cid in enumerate(ctx_ids)},
        cells=sorted({c["cell"] for c in bank["contexts"].values()}),
        carriers=sorted({c["carrier"] for c in bank["contexts"].values()}),
        va_tail_mean=va_tail,
        va_span_mean=va_span,
        tail_draws=tail_draws,
        draw_valid=np.ones((n, k), dtype=bool),
        n_valid=np.full(n, k, dtype=np.int64),
        ans_len_mean=rng.integers(5, 40, size=n).astype(np.float64),
        vc=vc,
        emb_mean=rng.standard_normal((n, 8)),
        emb_engine=engine,
        d=d,
    )


def _spec9(d: int = 16, lstar: int = 2) -> AN.SideSpec:
    base = AN.make_spec_9b(lstar, ("toneX",))
    return replace(
        base,
        d=d,
        twin_layers=(3,),
        store_layers=(lstar, 3),
        expected_contexts=None,
        expected_pairs=None,
    )


def _spec7(d: int = 16) -> AN.SideSpec:
    base = AN.make_spec_7b(("toneX",))
    return replace(base, d=d, store_layers=(AN.L19,), expected_contexts=None, expected_pairs=None)


def _fire_doc(pilots: bool, not_fired: str) -> dict:
    def vrow(axis: str, vid: str, fired: bool) -> dict:
        verdict = "fired" if fired else "not_fired"
        return {
            "axis": axis,
            "value_id": vid,
            "verdict": verdict,
            "sensitivity": {"50": "fired", "90": verdict},
        }

    value_rows = [vrow("toneX", v, v != not_fired) for v in VALS]
    axis_rows = [{"axis": "toneX", "floor_met": True}]
    if pilots:
        value_rows += [vrow("answer_language", lang, lang != not_fired) for lang in LANGS]
        axis_rows += [
            {"axis": "answer_language", "floor_met": True},
            {"axis": "query_content_oneword", "verdict": "no_manipulation_check_query_class"},
        ]
    return {"value_rows": value_rows, "axis_rows": axis_rows, "meta": {}}


def _cfg(tmp_path: Path, b: int = 40) -> AN.CfgX:
    return AN.CfgX(
        in_root_9b=None,
        in_root_7b=None,
        stage_dir=tmp_path / "stage",
        out_dir=tmp_path / "out",
        bank_9b=tmp_path / "bank9.json",
        bank_7b=None,
        manip_9b=tmp_path / "manip9.json",
        manip_7b=tmp_path / "manip7.json",
        sweep_json=tmp_path / "sweep.json",
        ridge_9b=None,
        preds_9b=None,
        preds7b_dir=None,
        ref7b_parent=tmp_path / "ref.json",
        ref7b_parent_commit="testcommit",
        embed_parity_report=None,
        smoke=True,
        b_boot=b,
        b_null=b,
        n_splits=4,
        prefix_2587=AN.PREFIX_2587,
        prefix_2564=AN.PREFIX_2564,
        prefix_fits=AN.PREFIX_FITS,
        prefix_preds7b=AN.PREFIX_PREDS7B,
    )


def _fire(tmp_path: Path, name: str, pilots: bool, not_fired: str) -> dict:
    p = tmp_path / name
    p.write_text(json.dumps(_fire_doc(pilots, not_fired)))
    return AN.load_fire(p)


@pytest.fixture(scope="module")
def world(tmp_path_factory) -> dict:
    tmp_path = tmp_path_factory.mktemp("an2587")
    bank9, bank7 = _banks()
    spec9 = _spec9()
    spec7 = _spec7()
    st9 = _stores(bank9, spec9.store_layers, spec9.d, spec9.primary_layer, "0.11.0", seed=7)
    st7 = _stores(bank7, spec7.store_layers, spec7.d, spec7.primary_layer, None, seed=11)
    cfg = _cfg(tmp_path)
    fire9 = _fire(tmp_path, "manip9.json", pilots=True, not_fired="v3")
    fire7 = _fire(tmp_path, "manip7.json", pilots=False, not_fired="v2")
    rng = np.random.default_rng([AN.BOOT_SEED])
    n_car = len(st9.carriers)
    idx_draws = rng.integers(0, n_car, size=(cfg.b_boot, n_car))
    mult = AN.carrier_multiplicities(idx_draws, n_car)
    rng9 = np.random.default_rng(1)
    mapped9 = {
        AN.ARM_FRESH9B: rng9.standard_normal((len(st9.ctx_ids), spec9.d)),
        AN.ARM_IDD9B: st9.vc[spec9.primary_layer],
    }
    mapped7 = {
        AN.ARM_7B_MATCHED: rng9.standard_normal((len(st7.ctx_ids), spec7.d)),
        AN.ARM_IDD7B: st7.vc[AN.L19],
    }
    run9 = AN.compute_side(cfg, spec9, bank9, st9, fire9, mapped9, mult, idx_draws)
    run7 = AN.compute_side(cfg, spec7, bank7, st7, fire7, mapped7, mult, idx_draws)
    cm_doc, cm_perdraw = AN.crossmodel_contrasts(
        run9, run7, spec9.primary_layer, mult, {"axes": {}}, "testcommit", cfg
    )
    return {
        "bank9": bank9,
        "bank7": bank7,
        "st9": st9,
        "st7": st7,
        "cfg": cfg,
        "run9": run9,
        "run7": run7,
        "cm_doc": cm_doc,
        "cm_perdraw": cm_perdraw,
        "mult": mult,
    }


# ── bootstrap machinery vs brute force ─────────────────────────────────


def test_boot_pair_sums_matches_bruteforce() -> None:
    rng = np.random.default_rng(0)
    n_pairs, n_car, b = 12, 3, 5
    vals = rng.standard_normal(n_pairs)
    ca = rng.integers(0, n_car, n_pairs)
    cb = rng.integers(0, n_car, n_pairs)
    dyad = rng.random(n_pairs) < 0.4
    cb = np.where(dyad, cb, ca)
    mult = AN.carrier_multiplicities(rng.integers(0, n_car, (b, n_car)), n_car)
    got = AN.boot_pair_sums(vals, ca, cb, dyad, mult)
    want = np.zeros(b)
    for bi in range(b):
        for i in range(n_pairs):
            w = mult[bi, ca[i]] * mult[bi, cb[i]] if dyad[i] else mult[bi, ca[i]]
            want[bi] += w * vals[i]
    np.testing.assert_allclose(got, want, rtol=1e-12)


def test_dyad_pair_weights_product_convention() -> None:
    mult = np.array([[2.0, 0.0, 1.0]])
    ca = np.array([0, 1])
    cb = np.array([2, 2])
    np.testing.assert_allclose(AN.dyad_pair_weights(mult, ca, cb), [[2.0, 0.0]])


def test_loco_multiplicities() -> None:
    m = AN.loco_multiplicities(4)
    assert m.shape == (4, 4)
    assert (np.diag(m) == 0).all() and m.sum() == 12


# ── Spearman machinery ─────────────────────────────────────────────────


def test_exact_spearman_dp_matches_bruteforce_n5() -> None:
    rng = np.random.default_rng(3)
    for _ in range(3):
        x = rng.standard_normal(5)
        y = rng.standard_normal(5)
        rho = AN.spearman_rho(x, y)
        p_dp = AN.exact_spearman_perm_pvalue(rho, 5)
        rx = AN._rankdata(x)
        ry = AN._rankdata(y)
        hits = 0
        perms = list(itertools.permutations(range(5)))
        for perm in perms:
            rp = rx[list(perm)]
            r = np.corrcoef(rp, ry)[0, 1]
            if abs(r) >= abs(rho) - 1e-12:
                hits += 1
        assert abs(p_dp - hits / len(perms)) < 1e-9, (p_dp, hits / len(perms))


def test_exact_spearman_dp_n6_extremes() -> None:
    # a perfect monotone pair: p = P(|rho|=1) = 2/6!
    x = np.arange(6.0)
    assert abs(AN.exact_spearman_perm_pvalue(1.0, 6) - 2 / 720) < 1e-12
    assert AN.exact_spearman_perm_pvalue(AN.spearman_rho(x, x), 6) == pytest.approx(2 / 720)


def test_spearman_block_ties_fall_back_to_mc() -> None:
    rng = np.random.default_rng(5)
    x = np.array([1.0, 1.0, 2.0, 3.0, 4.0, 5.0])
    y = rng.standard_normal(6)
    blk = AN.spearman_block(x, y, rng, mc_b=500)
    assert blk["method"].startswith("monte-carlo")
    assert 0.0 < blk["p"] <= 1.0


def test_partial_spearman_removes_shared_covariate() -> None:
    rng = np.random.default_rng(9)
    n = 200
    z = rng.standard_normal(n)
    x = z + rng.standard_normal(n)
    y = -z + rng.standard_normal(n)
    raw = abs(AN.spearman_rho(x, y))
    part = abs(AN.partial_spearman(x, y, z, z))
    assert raw > 0.3  # shared-covariate correlation is present
    assert part < 0.2 and part < raw  # ...and mostly removed by the partial
    # a correlation NOT running through the covariate survives the partial
    u = rng.standard_normal(n)
    z2 = rng.standard_normal(n)
    assert abs(AN.partial_spearman(u, u + 0.1 * rng.standard_normal(n), z2, z2)) > 0.8


# ── H1 lattice + paired row bootstrap ──────────────────────────────────


def test_h1_verdict_lattice_disjoint_exhaustive() -> None:
    assert AN.h1_verdict(-0.2, -0.05) == "h1_consistent"
    assert AN.h1_verdict(0.05, 0.2) == "h1_contradicted"
    assert AN.h1_verdict(-0.05, 0.05) == "h1_inconclusive"
    assert AN.h1_verdict(0.0, 0.1) == "h1_inconclusive"  # lo == 0 does NOT exclude 0
    assert AN.h1_verdict(-0.1, 0.0) == "h1_inconclusive"
    rng = np.random.default_rng(2)
    for _ in range(200):
        a, b = sorted(rng.standard_normal(2))
        v = AN.h1_verdict(a, b)
        matches = [b < 0, a > 0, a <= 0 <= b]
        assert sum(matches) == 1 and v in ("h1_consistent", "h1_contradicted", "h1_inconclusive")


def _mk_preds(rng: np.random.Generator, n: int, d: int, layer: int, quality: float) -> dict:
    t = rng.standard_normal((n, d))
    p = quality * t + (1 - quality) * rng.standard_normal((n, d))
    return {
        "layer": layer,
        "ci_te": [f"r{i:03d}" for i in range(n)],
        "pred_te": p,
        "target_te": t,
    }


def test_compute_h1_contradicted_and_perdraw() -> None:
    rng = np.random.default_rng(4)
    p9 = _mk_preds(rng, 50, 6, 2, quality=1.0)  # perfect: R2 = 1
    p7 = _mk_preds(rng, 50, 6, 19, quality=0.0)  # noise: R2 ~ << 1
    doc, draws = AN.compute_h1(p9, p7, lstar=2, b_boot=200, rng=np.random.default_rng(0))
    assert doc["verdict"] == "h1_contradicted"
    assert doc["r2_9b_lstar"] == pytest.approx(1.0)
    assert len(draws["delta_draws"]) == 200
    assert doc["delta_ci95"][0] > 0


def test_compute_h1_ordered_id_mismatch_raises() -> None:
    rng = np.random.default_rng(4)
    p9 = _mk_preds(rng, 20, 4, 2, 1.0)
    p7 = _mk_preds(rng, 20, 4, 19, 1.0)
    p7["ci_te"] = list(reversed(p7["ci_te"]))
    with pytest.raises(AssertionError, match="ordered"):
        AN.compute_h1(p9, p7, lstar=2, b_boot=10, rng=np.random.default_rng(0))
    p7["ci_te"] = list(reversed(p7["ci_te"]))
    with pytest.raises(AssertionError):
        AN.compute_h1(p9, p7, lstar=5, b_boot=10, rng=np.random.default_rng(0))  # wrong layer


def test_pooled_r2_draws_matches_bruteforce() -> None:
    rng = np.random.default_rng(6)
    n, d, b = 15, 3, 6
    t = rng.standard_normal((n, d))
    p = t + 0.3 * rng.standard_normal((n, d))
    idx = rng.integers(0, n, (b, n))
    counts = np.zeros((b, n))
    np.add.at(counts, (np.repeat(np.arange(b), n), idx.ravel()), 1.0)
    got = AN._pooled_r2_draws(p, t, counts)
    for bi in range(b):
        want = AN._pooled_r2(p[idx[bi]], t[idx[bi]])
        assert got[bi] == pytest.approx(want, rel=1e-10)


# ── frozen L* + primary arm + constraint pins ──────────────────────────


def test_lstar_read_never_reargmaxed(tmp_path: Path) -> None:
    doc = {
        "lstar": {
            "lstar": 5,
            "frozen": True,
            "criterion": "argmax over layers of ridge val_r2_at_selected",
            "val_r2_by_layer": {"0": 0.1, "5": 0.5, "7": 0.9},  # argmax says 7 — must NOT win
        }
    }
    p = tmp_path / "sweep.json"
    p.write_text(json.dumps(doc))
    assert AN.load_lstar(p)["lstar"] == 5
    doc["lstar"]["frozen"] = False
    p.write_text(json.dumps(doc))
    with pytest.raises(RuntimeError, match="frozen"):
        AN.load_lstar(p)


def test_primary_h2_arm_unique_and_pinned() -> None:
    assert AN.resolve_primary_h2_arm([AN.ARM_7B_MATCHED]) == AN.PRIMARY_H2_7B_ARM
    with pytest.raises(RuntimeError):
        AN.resolve_primary_h2_arm(["arm_779ce"])
    with pytest.raises(RuntimeError):
        AN.resolve_primary_h2_arm([AN.ARM_7B_MATCHED, AN.ARM_7B_MATCHED])
    with pytest.raises(RuntimeError, match="sensitivity"):
        AN.resolve_primary_h2_arm([AN.ARM_7B_MATCHED, AN.REF_7B_PARENT])


def test_constraint1_install_orientation_per_class(world: dict) -> None:
    assert "install (parent instruction axes)" in AN.ORIENTATION_CONVENTIONS
    assert "install (answer_language pilot)" in AN.ORIENTATION_CONVENTIONS
    pa9 = world["run9"].pa
    pilot = [
        i for i in range(pa9.n) if pa9.cls[i] == "install" and pa9.axis[i] == "answer_language"
    ]
    parent = [i for i in range(pa9.n) if pa9.cls[i] == "install" and pa9.axis[i] == "toneX"]
    assert pilot and parent
    # per-pair orientation strings preserve each class's own a->b order —
    # never a global reorientation (cross-unit constraint 1)
    assert all(pa9.orientation[i].endswith("->bare") for i in pilot)
    assert all(pa9.orientation[i].startswith("E->") for i in parent)
    assert all(pa9.value_b[i] == "bare" for i in pilot)
    assert all(pa9.value_a[i] == "E" for i in parent)


def test_constraint2_axis_key(world: dict) -> None:
    pa9 = world["run9"].pa
    for i in range(pa9.n):
        if pa9.cls[i] in ("query_content", "query_paraphrase"):
            assert pa9.axis[i] == pa9.cls[i]  # cell == "query" -> class is the key
        if pa9.axis[i] in ("answer_language", "query_content_oneword"):
            # merged pilot pairs: cell (== axis field in the manifest) is the ONE key
            assert pa9.axis[i] in AN.PILOT_AXES


def test_constraint3_engine_parity(tmp_path: Path) -> None:
    assert AN.assert_engine_parity("qwen35_9b", "0.11.0", None)["mode"] == "repo-pin"
    with pytest.raises(RuntimeError, match="vllm_version"):
        AN.assert_engine_parity("qwen35_9b", None, None)
    with pytest.raises(RuntimeError, match="parity"):
        AN.assert_engine_parity("qwen35_9b", "0.12.0", None)
    rep = tmp_path / "rep.json"
    rep.write_text(
        json.dumps({"parity_pass": True, "engine": "0.12.0", "reference_engine": "0.11.0"})
    )
    assert AN.assert_engine_parity("qwen35_9b", "0.12.0", rep)["mode"] == "parity-report"
    rep.write_text(
        json.dumps({"parity_pass": False, "engine": "0.12.0", "reference_engine": "0.11.0"})
    )
    with pytest.raises(AssertionError):
        AN.assert_engine_parity("qwen35_9b", "0.12.0", rep)
    # the 7B reference side may legitimately lack the key (by-pin provenance)
    assert AN.assert_engine_parity("qwen25_7b", None, None)["mode"] == "reference-by-pin"


def test_constraint4_store_col_resolves_by_layer_value() -> None:
    store = {"layers": [5, 2, 9], "layer_convention": "x captured[L] == hidden_states[L+1] y"}
    assert AN._store_col(store, 2) == 1
    assert AN._store_col(store, 9) == 2
    with pytest.raises(AssertionError):
        AN._store_col(store, 3)  # absent layer fails loud
    bad = {"layers": [5, 2, 9], "layer_convention": "captured[L] == hidden_states[L]"}
    with pytest.raises(AssertionError):
        AN._store_col(bad, 2)  # a DIFFERENT recorded convention fails loud


def test_constraint5_frozen_layer_pair() -> None:
    AN.assert_frozen_layer_pair(22, 19, lstar=22)  # ok even when L* IS a twin id
    with pytest.raises(RuntimeError, match="L19"):
        AN.assert_frozen_layer_pair(22, 14, lstar=22)
    with pytest.raises(RuntimeError, match="twins"):
        AN.assert_frozen_layer_pair(16, 19, lstar=22)


def test_spec_pins_match_units() -> None:
    s9 = AN.make_spec_9b(22, ("a",))
    assert s9.d == 4096 and s9.primary_layer == 22
    assert 22 not in s9.twin_layers and set(s9.twin_layers) == {16, 30}
    s7 = AN.make_spec_7b(("a",))
    assert s7.d == 3584 and s7.primary_layer == 19 and s7.twin_layers == ()
    assert s7.map_arm == AN.PRIMARY_H2_7B_ARM


# ── per-side battery e2e (production body) ─────────────────────────────


def test_compute_side_9b_axes_and_pilots(world: dict) -> None:
    run9 = world["run9"]
    axes = run9.axes_out
    assert set(axes) == {"toneX", "answer_language", "query_content", "query_content_oneword"}
    for axis in AN.PILOT_AXES:
        assert axes[axis]["pilot_axis"] is True
        assert axes[axis]["cross_model_status"] == AN.PILOT_LABEL
    assert axes["toneX"]["pilot_axis"] is False
    # null schemes: grid derangement for toneX + answer_language; pair
    # derangement for the dyads AND the single-carrier oneword pilots
    assert run9.views["toneX"].null_kind == "grid"
    assert run9.views["answer_language"].null_kind == "grid"
    assert run9.views["query_content"].null_kind == "pair_derangement"
    assert run9.views["query_content_oneword"].null_kind == "pair_derangement"
    # oneword: no carrier-replicated vp -> identity undefined
    assert "n/a" in axes["query_content_oneword"]["identity"]
    # answer_language: install control present, NO paraphrase family
    d = axes["answer_language"]["direction"][AN.ARM_FRESH9B]
    assert "install" in d["controls"] and "instruction_paraphrase" not in d["controls"]
    assert axes["answer_language"]["text_space"]["no_para_family"] is True
    # fire gating: v3 not fired -> toneX headline pairs < primary pairs
    ftx = axes["toneX"]["fire"]
    assert 0 < ftx["n_headline_pairs_fired70"] < ftx["n_primary_pairs"]
    # iddelta twin layer present on the 9B side only
    assert "3" in axes["toneX"]["layer_twins"]
    assert "n/a" in world["run7"].axes_out["toneX"]["layer_twins"]
    # perpair rows carry the model tag on every row
    assert all(r["model_tag"] == "qwen35_9b" for r in run9.perpair)
    assert all(r["model_tag"] == "qwen25_7b" for r in world["run7"].perpair)


def test_compute_side_identity_baseline_and_retrieval(world: dict) -> None:
    run9 = world["run9"]
    assert run9.id_check["max_abs_err"] <= run9.id_check["tol"]  # real identity_bias_predict
    assert run9.engine_parity["mode"] == "repo-pin"
    assert world["run7"].engine_parity["mode"] == "reference-by-pin"
    for arm in (AN.ARM_FRESH9B, AN.ARM_IDD9B):
        assert arm in run9.retrieval["global"]
        assert "cosine" in run9.retrieval["global"][arm]
    assert run9.retrieval["chance"]["rule"] == "chance = k / n_pool"


def test_pilot_placement_common_statistic(world: dict) -> None:
    blk = AN.pilot_placement_block(world["run9"])
    assert set(blk["pilots"]) == set(AN.PILOT_AXES)
    for axis in AN.PILOT_AXES:
        q = blk["pilots"][axis]["quartile"]
        assert q in (1, 2, 3, 4)
    assert len(blk["snr_by_axis"]) == len(world["run9"].views)


# ── cross-model contrasts + H2 (production body) ───────────────────────


def test_crossmodel_stats_tables_and_perdraw(world: dict) -> None:
    doc, perdraw = world["cm_doc"], world["cm_perdraw"]
    assert doc["layer_pair"] == {"qwen35_9b": 2, "qwen25_7b": 19}
    assert doc["primary_h2_7b_arm"] == AN.PRIMARY_H2_7B_ARM
    assert doc["ref_7b_parent"]["label"] == AN.REF_7B_PARENT
    expected_stats = {
        "direction_cos",
        "calibration_ratio_to_global",
        "crossfam_cos_observed",
        "crossfam_cos_maparm",
        "obs_separation_snr",
        "axis_identity_cos",
    }
    assert set(doc["stats"]) == expected_stats
    dc = doc["stats"]["direction_cos"]
    assert {r["axis"] for r in dc["axes"]} == {"toneX", "query_content"}
    for r in dc["axes"]:
        assert np.isfinite(r["s_9b"]) and np.isfinite(r["s_7b"])
        assert np.isfinite(r["delta_9b_minus_7b"])
        assert len(r["delta_ci95"]) == 2 and len(r["delta_t11_ci95"]) == 2
        lo, hi = r["delta_loco_jackknife_range"]
        assert lo <= hi
        assert np.isnan(r["s_7b_ref_parent"])  # empty ref doc -> NaN sensitivity read
    b = world["mult"].shape[0]
    pd = perdraw["direction_cos"]
    assert pd["draws_9b"].shape == (2, b) and pd["delta_draws"].shape == (2, b)
    assert pd["loco_delta"].shape == (2, 3)
    np.testing.assert_allclose(
        pd["delta_draws"], pd["draws_9b"] - pd["draws_7b"], rtol=0, atol=0
    )  # ONE shared carrier resample: the delta IS the paired per-draw difference
    # grid-only stats restrict to the instruction axis
    assert {r["axis"] for r in doc["stats"]["axis_identity_cos"]["axes"]} == {"toneX"}


def test_crossmodel_symmetric_fire_drops(world: dict) -> None:
    # 9B fire drops v3, 7B fire drops v2 -> BOTH one-sided drop counts > 0
    rows = {r["axis"]: r for r in world["cm_doc"]["stats"]["direction_cos"]["axes"]}
    f = rows["toneX"]["fire"]
    assert f["n_dropped_9b_only"] > 0 and f["n_dropped_7b_only"] > 0
    assert f["n_symmetric_fired"] < f["n_shared_primary"]


def test_crossmodel_h2_block_fields(world: dict) -> None:
    h2 = world["cm_doc"]["h2"]
    assert h2["primary_h2_7b_arm"] == AN.PRIMARY_H2_7B_ARM
    assert h2["read_a_obs_separation"]["verdict"] in (
        "h2_shared",
        "h2_falsified",
        "h2_inconclusive",
        "h2_undetermined",
    )
    assert h2["combined_verdict"] in ("h2_shared", "h2_falsified", "h2_inconclusive")
    assert isinstance(h2["sign_disagreement_axes"], list)


def test_crossmodel_refuses_non_frozen_layer_pair(world: dict) -> None:
    with pytest.raises(RuntimeError, match="frozen L"):
        AN.crossmodel_contrasts(
            world["run9"], world["run7"], 3, world["mult"], {"axes": {}}, "c", world["cfg"]
        )


# ── misc ───────────────────────────────────────────────────────────────


def test_split_half_stats_deterministic(world: dict) -> None:
    st9, run9 = world["st9"], world["run9"]
    rel_a = AN.split_half_stats(st9, run9.pa, 4)
    rel_b = AN.split_half_stats(st9, run9.pa, 4)
    np.testing.assert_allclose(rel_a["r_half"], rel_b["r_half"], equal_nan=True)
    assert rel_a["n_pairs_insufficient_draws"] == 0


def test_json_sanitize_nan_to_none() -> None:
    out = AN._json_sanitize({"a": float("nan"), "b": [1.0, float("inf")], "c": np.float64("nan")})
    assert out == {"a": None, "b": [1.0, None], "c": None}


def test_ref7b_stat_extraction_shapes() -> None:
    ref_axes = {
        "toneX": {
            "direction": {"arm_779ce": {"mean_cos_headline": 0.4}},
            "calibration": {"arm_779ce": {"ratio_to_global": 1.2}},
            "cross_family": {"observed": {"median": 0.3}, "arm_779ce": {"median": 0.5}},
            "surface": {"observed": {"flip_norm_mean": 2.0}},
            "reliability": {"noise_norm_mean": 0.5},
            "identity": {"arm_779ce": {"median": 0.7}},
        }
    }
    assert AN._ref7b_stat(ref_axes, "toneX", "direction_cos") == pytest.approx(0.4)
    assert AN._ref7b_stat(ref_axes, "toneX", "obs_separation_snr") == pytest.approx(4.0)
    assert np.isnan(AN._ref7b_stat(ref_axes, "missing", "direction_cos"))
    assert np.isnan(AN._ref7b_stat({}, "toneX", "axis_identity_cos"))
