"""Unit pins for scripts/issue2552_ladder.py (#2552 P3 driver).

Covers the smoke-caught `_orth` rank regression (a block residualized to numerical
dust must yield rank 0, never spurious full rank), the Wilson helper, the registered
5-cell verdict lattice (all 9 CI-sign combinations, r2 g6-M4), the within-quintile
permutation invariant, the producer<->consumer perfeature key parity (r2 g6-C1),
the semantic-none taxonomy (r2 g6-M2), the fingerprinted draw-matrix store (r2
ladder-restartability), and the Der-reference embedding literals (r2 swap fix).
No network, no committed-artifact reads (repo-root safe in sparse worktrees);
torch-free (the turnsae producer is inspected via ast, never imported)."""

from __future__ import annotations

import ast
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
for _p in (str(REPO_ROOT / "scripts"), str(REPO_ROOT / "scripts" / "vendored_2476")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import issue2552_ladder as L  # noqa: E402

TURNSAE_SRC = (REPO_ROOT / "scripts" / "issue2552_turnsae_der.py").read_text()


def _producer_perfeature_keys() -> set[str]:
    """The turnsae phase_perfeature_r2 savez_atomic LITERAL key set, via ast
    (never an import — the producer is torch-bearing)."""
    tree = ast.parse(TURNSAE_SRC)
    fn = next(
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.FunctionDef) and n.name == "phase_perfeature_r2"
    )
    for call in ast.walk(fn):
        if (
            isinstance(call, ast.Call)
            and isinstance(call.func, ast.Name)
            and call.func.id == "savez_atomic"
        ):
            keys = {kw.arg for kw in call.keywords if kw.arg is not None}
            if "r2_map" in keys:
                return keys
    raise AssertionError("phase_perfeature_r2 savez_atomic call not found")


def test_orth_spanned_block_is_rank_zero():
    """Regression (#2552 unit-3 smoke): a candidate block fully inside the design span,
    residualized to dust, must NOT come back as full-rank noise."""
    rng = np.random.default_rng(0)
    n = 400
    onehot = np.zeros((n, 5))
    onehot[np.arange(n), rng.integers(0, 5, n)] = 1.0
    q = L._orth(np.column_stack([np.ones(n), onehot]))
    dust = onehot - q @ (q.T @ onehot)  # numerically ~1e-16
    qc = L._orth(dust, scale=float(np.linalg.norm(onehot)))
    assert qc.shape[1] == 0, qc.shape
    # and WITHOUT residualization the same block has rank 4 given an intercept
    q0 = L._orth(np.ones((n, 1)))
    resid = onehot - q0 @ (q0.T @ onehot)
    qc4 = L._orth(resid, scale=float(np.linalg.norm(onehot)))
    assert qc4.shape[1] == 4, qc4.shape


def test_wilson_known_value():
    lo, hi = L._wilson(8, 10)
    assert 0.49 < lo < 0.50 and 0.94 < hi < 0.95, (lo, hi)  # canonical 8/10 Wilson
    lo0, hi0 = L._wilson(0, 0)
    assert np.isnan(lo0) and np.isnan(hi0)


def test_lattice_cells_exhaustive():
    assert L._lattice_cell((0.1, 0.3), (0.05, 0.2)) == "Reproduced"
    assert L._lattice_cell((-0.3, -0.1), (-0.2, -0.05)) == "Reversed"
    assert L._lattice_cell((0.1, 0.3), (-0.2, -0.05)) == "Not reproduced - pt_max dominance"
    assert L._lattice_cell((-0.3, -0.1), (0.05, 0.2)) == "Not reproduced - rep_ta dominance"
    assert L._lattice_cell((-0.1, 0.3), (0.05, 0.2)) == "Inconclusive"
    assert L._lattice_cell((0.1, 0.3), (-0.05, 0.2)) == "Inconclusive"


def test_perm_within_preserves_quintile_multisets():
    rng = np.random.default_rng(1)
    e0 = rng.normal(size=97)
    quint = rng.integers(0, 5, size=97)
    out = L._perm_within(rng, e0, quint, 7)
    assert out.shape == (97, 7)
    for q in range(5):
        idx = quint == q
        ref = np.sort(e0[idx])
        for b in range(7):
            assert np.allclose(np.sort(out[idx, b]), ref)  # a permutation, never a mix


def test_rank01_range_and_monotone():
    x = np.array([3.0, 1.0, 2.0, 10.0])
    r = L._rank01(x)
    assert (r > 0).all() and (r < 1).all()
    assert r[3] == r.max() and r[1] == r.min()


# ── r2 round: producer<->consumer perfeature contract (g6-C1) ─────────────────────


def test_perfeature_producer_consumer_key_parity():
    """The ladder `_load_dv` rep branch + the smoke fixture must both key on the
    turnsae producer's LITERAL savez set (schema-from-artifact, #2379 class)."""
    prod = _producer_perfeature_keys()
    # the consumer's hard-required subset
    assert {"feat_ids", "r2_map", "counts", "alive_f240", "alive_f1200"} <= prod, prod
    # keys the r1 consumer wrongly expected must NOT be in the producer set
    assert not ({"tier", "r2", "activity"} & prod), prod
    # the ladder smoke fixture writes EXACTLY the producer key set
    import inspect

    src = inspect.getsource(L._synth_inputs)
    tree = ast.parse("def _w():\n" + "\n".join("    " + ln for ln in src.splitlines()[1:]))
    fix_keys: set[str] = set()
    for call in ast.walk(tree):
        if (
            isinstance(call, ast.Call)
            and isinstance(call.func, ast.Attribute)
            and call.func.attr == "savez"
            and any(kw.arg == "r2_map" for kw in call.keywords if kw.arg)
        ):
            ks = {kw.arg for kw in call.keywords if kw.arg is not None}
            if "n_holdout" in ks:  # the perfeature fixture (covariates lack it)
                fix_keys = ks
    assert fix_keys == prod, (sorted(fix_keys ^ prod), "fixture drifted from producer")


def _write_dv_fixture(eval_in: Path, n: int = 40) -> None:
    """Producer-literal perfeature npz + minimal covariates + measured regime."""
    rng = np.random.default_rng(0)
    keys = _producer_perfeature_keys()
    counts = np.concatenate(
        [rng.integers(240, 1000, size=n - 10), rng.integers(1200, 9000, size=10)]
    ).astype(np.int64)
    arrs: dict[str, np.ndarray] = {}
    for k in keys:
        if k == "feat_ids":
            arrs[k] = np.arange(n, dtype=np.int64)
        elif k == "counts":
            arrs[k] = counts
        elif k.startswith("alive_f"):
            arrs[k] = counts >= int(k.removeprefix("alive_f"))
        elif k == "shuffle_seeds":
            arrs[k] = np.asarray([0, 1, 2], np.int64)
        elif k in ("n_fit_rows", "n_holdout"):
            arrs[k] = np.int64(123)
        else:
            arrs[k] = rng.random(n)
    np.savez(eval_in / "perfeature_rep.npz", **arrs)
    (eval_in / "regime_measured.json").write_text(
        json.dumps({"trainer2_fve_evalpass": 0.7, "rep_sae_val_var_fve": 0.6})
    )
    for fam in L.DICTS:
        np.savez(eval_in / f"covariates_{fam}.npz", counts=np.ones(n))


def test_load_dv_executes_on_producer_key_fixture(tmp_path):
    """PRODUCTION-BODY: `_load_dv` runs end-to-end on a producer-literal npz."""
    _write_dv_fixture(tmp_path)
    args = SimpleNamespace(smoke=True, p1_eval_dir=None, hf_prefix="x")
    io = SimpleNamespace(eval_in=tmp_path, union_paths={})
    dv = L._load_dv(args, io, "rep_ta", 240)
    assert dv.tier is None and dv.panel.dtype == np.bool_
    assert dv.panel.sum() >= 1 and dv.feat_ids.dtype == np.int64
    dv1200 = L._load_dv(args, io, "rep_ta", 1200)
    assert dv1200.panel.sum() <= dv.panel.sum()


# ── r2 round: 9-combination lattice (g6-M4) ──────────────────────────────────────


def test_lattice_cells_all_nine_sign_combos():
    """Every (disc CI, cov CI) sign combination: pos/neg/spanning x pos/neg/spanning."""
    pos, neg, span = (0.1, 0.3), (-0.3, -0.1), (-0.1, 0.2)
    expected = {
        (pos, pos): "Reproduced",
        (neg, neg): "Reversed",
        (pos, neg): "Not reproduced - pt_max dominance",
        (neg, pos): "Not reproduced - rep_ta dominance",
        (pos, span): "Inconclusive",
        (neg, span): "Inconclusive",
        (span, pos): "Inconclusive",
        (span, neg): "Inconclusive",
        (span, span): "Inconclusive",
    }
    for (a, b), want in expected.items():
        assert L._lattice_cell(a, b) == want, (a, b, want)


# ── r2 round: semantic-none taxonomy (g6-M2) ─────────────────────────────────────


def test_labels_status_semantic_none(tmp_path):
    """'none' derives from the per-item VALUE == ('none','none') — never from
    absence-in-assignments (the producer EXCLUDES none features there)."""
    import issue2552_judge_waves as JW

    agg = tmp_path / "agg"
    raw = tmp_path / "raw_w3"
    for d in (agg, raw):
        d.mkdir(parents=True)
    field_valid = next(iter(JW.FIELD_TO_CATEGORY))
    cat_valid = JW.FIELD_TO_CATEGORY[field_valid]
    (agg / "w3_categories_rep_ta.json").write_text(
        json.dumps({"assignments": {"1": {"category": cat_valid, "field": field_valid}}})
    )
    all_scores = {
        "w3-rep_ta-f1__00000__00": {"field": field_valid, "stop_reason": "end_turn"},
        "w3-rep_ta-f2__00000__00": {"field": "none", "stop_reason": "end_turn"},
        "w3-rep_ta-f3__00000__00": {"junk": 1, "stop_reason": "end_turn"},
    }
    (raw / "judge_raw_w3.json").write_text(json.dumps({"all_scores": all_scores}))
    args = SimpleNamespace(smoke=True, hf_prefix="x")
    io = SimpleNamespace(agg_in=agg, raw_w3=raw, work=tmp_path / "work")
    cats, stats = L._labels_status(args, io, "rep_ta", np.asarray([1, 2, 3, 4], np.int64))
    assert list(stats) == ["valid", "none", "malformed", "unjudged"], list(stats)
    assert cats[0] == cat_valid and list(cats[1:]) == ["", "", ""], list(cats)


# ── r2 round: fingerprinted draw-matrix store (ladder-restartability) ─────────────


def test_draw_store_probe_sink_roundtrip(tmp_path):
    fid = np.arange(7, dtype=np.int64)
    probe, sink = L._draw_store(tmp_path, "rep_ta", "primary", fid, (2552, 0, 0), 16, 3)
    names = ["a", "b"]
    assert probe(1, names) is None  # nothing persisted yet
    mat = np.random.default_rng(0).random((16, 2)).astype(np.float32)
    sink(1, names, mat, {"a": 0.1, "b": 0.2})
    got = probe(1, names)
    assert got is not None and np.array_equal(got, mat)
    # regime changes invalidate the fingerprint -> recompute (None), never reuse
    probe2, _ = L._draw_store(tmp_path, "rep_ta", "primary", fid, (2552, 0, 1), 16, 3)
    assert probe2(1, names) is None  # different seed_key
    probe3, _ = L._draw_store(tmp_path, "rep_ta", "primary", fid[:5], (2552, 0, 0), 16, 3)
    assert probe3(1, names) is None  # different panel ids
    assert probe(1, ["a", "c"]) is None  # different candidate names
    assert probe(2, names) is None  # different step


def test_run_ladder_resumes_from_draw_store(tmp_path):
    """Two `_run_ladder` passes over the same store: pass 2 marks resumed_drawmat
    and reproduces pass 1's null bands byte-for-byte."""
    rng = np.random.default_rng(3)
    n = 120
    y = rng.random(n)
    log_act = rng.random(n)
    quint = rng.integers(0, 5, size=n)
    blocks = {"c1": rng.random((n, 1)), "c2": rng.random((n, 1))}
    fid = np.arange(n, dtype=np.int64)

    def _go():
        probe, sink = L._draw_store(tmp_path, "rep_ta", "t", fid, (1, 2), 32, 2)
        return L._run_ladder(
            y,
            log_act,
            quint,
            dict(blocks),
            draws=32,
            depth=2,
            seed_key=(1, 2),
            with_nulls=True,
            draw_sink=sink,
            draw_probe=probe,
        )

    r1 = _go()
    r2 = _go()
    assert all(not s["resumed_drawmat"] for s in r1["steps"])
    assert all(s["resumed_drawmat"] for s in r2["steps"])
    for s1, s2 in zip(r1["steps"], r2["steps"], strict=True):
        assert s1["null_p95_partial"] == s2["null_p95_partial"]
        assert s1["p_value_selection_symmetric"] == s2["p_value_selection_symmetric"]


# ── r2 round: Der-reference literals + P4 revision pin ───────────────────────────


def test_paper_reference_embedding_literals():
    """Der et al. embedding references: pt_max=0.617, rep_ta(TA)=0.663 (r2 swap
    fix) — pinned in BOTH drivers; the transposed r1 literal must not reappear."""
    assert L.PAPER_REFERENCE["embedding"] == {"pt_max": 0.617, "rep_ta": 0.663}
    assert '"paper_reference_embedding": {"pt_max": 0.617, "rep_ta": 0.663}' in TURNSAE_SRC
    assert '{"pt_max": 0.663, "rep_ta": 0.617}' not in TURNSAE_SRC


def test_p4_lists_hf_fallback_resolves_head_not_pins():
    """The P4 HF fallback resolves the post-P1 HEAD (r2 BLOCKER p4-future-revision):
    the P0 pin structurally predates P1.10's uploads."""
    start = TURNSAE_SRC.index("def _p4_lists(")
    end = TURNSAE_SRC.index("\ndef ", start + 1)
    body = TURNSAE_SRC[start:end]
    assert "_resolve_repo_revision(None" in body, "HF fallback must resolve HEAD"
    # the docstring may NAME the pin; no CODE line may CALL it
    assert "= _pins_revision()" not in body, "the P0 pin must not gate the P4 fetch"
    assert "p4_inputs_manifest.json" in body, "resolved revision must be persisted"


def test_cfg_tick_covers_all_configs():
    import issue2552_judge_waves as JW

    for cfg in JW.CONFIGS:
        assert cfg in L.CFG_TICK and "\n" in L.CFG_TICK[cfg], cfg


# ── r3 round: content-keyed draw store + combined FVE + P4 sentinel path ──────────


def test_draw_store_content_sha_invalidates(tmp_path):
    """r3 ladder-restartability (r2 g2-M2): identical ids/seed/names but CHANGED data
    content must invalidate the store — stale null matrices are recomputed, not reused."""
    fid = np.arange(7, dtype=np.int64)
    probe, sink = L._draw_store(
        tmp_path, "rep_ta", "primary", fid, (2552, 0, 0), 16, 3, content_sha="aaa"
    )
    names = ["a", "b"]
    mat = np.random.default_rng(0).random((16, 2)).astype(np.float32)
    sink(1, names, mat, {"a": 0.1, "b": 0.2})
    assert probe(1, names) is not None  # same content -> resume
    probe2, _ = L._draw_store(
        tmp_path, "rep_ta", "primary", fid, (2552, 0, 0), 16, 3, content_sha="bbb"
    )
    assert probe2(1, names) is None  # changed DV/candidate content -> recompute


def test_content_sha_tracks_values_not_just_ids():
    """The digest changes when DV values change under IDENTICAL panel ids."""
    from types import SimpleNamespace

    idx = np.arange(5)
    cats = np.asarray(["a", "b", "a", "b", "a"])
    blocks = {"x": np.random.default_rng(1).random((5, 2))}
    dv1 = SimpleNamespace(r2=np.arange(5, dtype=np.float64), counts=np.ones(5, np.int64))
    dv2 = SimpleNamespace(r2=np.arange(5, dtype=np.float64) + 1.0, counts=np.ones(5, np.int64))
    s1 = L._content_sha(dv1, idx, cats, blocks)
    assert s1 == L._content_sha(dv1, idx, cats, blocks)  # deterministic
    assert s1 != L._content_sha(dv2, idx, cats, blocks)  # values changed, ids identical


def _extract_fve_from_acc():
    """AST-extract the torch-free `_fve_from_acc` from the turnsae producer (the
    established pattern in this file: the producer is torch-bearing, never imported)."""
    tree = ast.parse(TURNSAE_SRC)
    fn = next(
        n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "_fve_from_acc"
    )
    mod = ast.Module(body=[fn], type_ignores=[])
    ns: dict = {"np": np}
    exec(compile(ast.fix_missing_locations(mod), "<turnsae _fve_from_acc>", "exec"), ns)
    return ns["_fve_from_acc"]


def test_fve_from_acc_combined_matches_direct_computation():
    """r3 comparator-fve: the combined eval+pool FVE from persisted sufficient stats
    equals the direct global fp64 per-dim var-FVE over the concatenated tokens."""
    fve_from_acc = _extract_fve_from_acc()
    rng = np.random.default_rng(7)
    d = 6
    xa, xb = rng.random((40, d)), rng.random((25, d))
    ra, rb = xa * 0.1 + rng.random((40, d)) * 0.01, xb * 0.2 + rng.random((25, d)) * 0.01

    def acc(x, r):
        return {
            "x_sum": x.sum(0),
            "r_sum": r.sum(0),
            "x_sq_total": float((x * x).sum()),
            "r_sq_total": float((r * r).sum()),
            "n_tok": len(x),
        }

    def direct(x, r):
        n = len(x)
        ss_tot = ((x * x).sum(0) - x.sum(0) ** 2 / n).sum() / (n - 1)
        ss_res = ((r * r).sum(0) - r.sum(0) ** 2 / n).sum() / (n - 1)
        return 1.0 - ss_res / ss_tot

    # single-pass parity
    assert abs(fve_from_acc(acc(xa, ra)) - direct(xa, ra)) < 1e-12
    # combined parity vs the concatenated-token direct computation
    xc, rc = np.vstack([xa, xb]), np.vstack([ra, rb])
    assert abs(fve_from_acc(acc(xa, ra), acc(xb, rb)) - direct(xc, rc)) < 1e-12
    # degenerate guard
    assert np.isnan(
        fve_from_acc(
            {
                "x_sum": np.zeros(d),
                "r_sum": np.zeros(d),
                "x_sq_total": 0.0,
                "r_sq_total": 0.0,
                "n_tok": 1,
            }
        )
    )


def test_p4_done_sentinel_lives_under_sentinels_dir():
    """r3 p4-phase-contract: the P4 completion sentinel is written at the plan
    phase_outputs path <out_root>/sentinels/p4_done.json (the p1_done.json dir),
    never at <out_root>/p4/p4_done.json."""
    assert 'done_path = sent_dir / "p4_done.json"' in TURNSAE_SRC
    assert 'io.stage.parent / "p4_done.json"' not in TURNSAE_SRC
    # the sentinel carries input identity + upload binding
    assert '"inputs": input_ids' in TURNSAE_SRC
    assert '"uploads"' in TURNSAE_SRC


def test_replication_zero_pair_assert_is_unconditional():
    """r3 smoke-gate-enumeration: the pooled zero-pair gate carries NO smoke bypass —
    the seeded smoke fixture exercises it (g2-verified deterministic pass)."""
    import inspect

    src = inspect.getsource(L.phase_replication)
    assert 'pooled.get("n_complete_pairs", 0) > 0' in src
    assert "args.smoke or (\n        pooled.get" not in src
    # no smoke term anywhere in the zero-pair assert statement
    stmt = src.split("assert pooled.get", 1)[1].split(")", 3)[:3]
    assert "smoke" not in "".join(stmt)
