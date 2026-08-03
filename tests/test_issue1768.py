"""#1768 pins: registry parsing, loader sha-alignment, verdict lattice,
disjoint-halves registration, null-band shapes, corpus sampling caps.

Everything runs CPU-tiny against synthetic stores in tmp_path (never the
committed eval_results/ or figures/ trees). The verdict-manifest tests read
the COMMITTED #1481 manifest (git-resident fixture, read-only).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

import issue1768_cells as X  # noqa: E402

torch = pytest.importorskip("torch")


# ── registry (committed manifest fixture) ────────────────────────────────────


def test_arm_counts_and_marker_lowest_lr_selection():
    arms = X.all_arms()
    assert len(arms) == 72  # 56 LoRA + 16 full-FT (plan §4.1 amendment)
    lora = [a for a in arms if a.method == "lora"]
    ft = [a for a in arms if a.method == "ft"]
    assert len(lora) == 56 and len(ft) == 16
    content = [a for a in lora if a.kind == "content"]
    marker = [a for a in lora if a.kind == "marker"]
    assert len(content) == 40 and len(marker) == 16
    # marker rule: lowest-LR in-window rung per (ctx, regime, seed) — the
    # lr5e-6 rungs are in-window everywhere on the committed manifest
    assert all(a.lr == pytest.approx(5e-6) for a in marker)
    assert all(a.arm_id.startswith("mk-") for a in marker)
    # every content arm is in-band with a judged rate
    assert all(0.0 < a.selection_read <= 1.0 for a in content)


def test_ft_arm_registry_identity_and_delta_mapping():
    """Plan §4.1 amendment: 16 ft arms, overflow-ckpt identity, shared δ cells."""
    arms = {a.arm_id: a for a in X.all_arms()}
    ft = [a for a in arms.values() if a.method == "ft"]
    assert len(ft) == 16
    assert all(a.ctx_key == "pers" for a in ft)
    assert sum(1 for a in ft if a.kind == "marker") == 4
    # decode-cap + base-unit routing: mk-ft arms take the 2048 marker cap
    assert X.base_unit_for("mk-pers-ft-con-s42") == "base_mk"
    assert X.max_new_tokens_for("mk-pers-ft-con-s42") == X.MAX_NEW_MARKER
    assert X.max_new_tokens_for("cas-pers-ft-po-s137") == X.MAX_NEW_CONTENT
    # checkpoint identity: overflow paths; the reused #1112 cell is the exception
    assert (
        X.ft_ckpt_subfolder(arms["imp-pers-ft-con-s42"])
        == "issue1586/imp-pers-ft-con-s42/checkpoint-14"
    )
    assert X.ft_ckpt_subfolder(arms["syc-pers-ft-con-s42"]) == X.FT_REUSED_SUBFOLDER
    with pytest.raises(AssertionError):
        X.ft_ckpt_subfolder(arms["imp-pers-con-lr3e5-s42"])  # LoRA arm misuse
    with pytest.raises(ValueError):
        X.adapter_subfolder(arms["imp-pers-ft-con-s42"])  # ft arm has no adapter
    # δ cells COINCIDE with the pers-LoRA cells: every ft arm maps to a
    # registry LoRA pers arm at matched (beh, regime, seed); LoRA arms -> self
    for a in ft:
        d = X.delta_arm_for(a)
        assert d in arms and arms[d].method == "lora" and arms[d].ctx_key == "pers", (a.arm_id, d)
        assert (arms[d].beh_key, arms[d].regime, arms[d].seed) == (a.beh_key, a.regime, a.seed)
    assert X.delta_arm_for(arms[X.PILOT_ARM]) == X.PILOT_ARM
    assert X.arm_method("imp-pers-ft-con-s42") == "ft"
    assert X.arm_method(X.PILOT_ARM) == "lora"


def test_marker_selection_prefers_lowest_lr_synthetic():
    man = {
        "content": {b: {c: {"seeds": {}} for c in X.CTX_KEYS} for b in X.BEH_KEYS},
        "marker": {
            "arms": {
                "mk-pers-con-lr1e5-s42": {
                    "ctx_key": "pers",
                    "regime": "con",
                    "lr_key": "lr1e5",
                    "seed": 42,
                    "selection": {"in_window": True, "step": 10, "delta_logp_mean": 6.0},
                },
                "mk-pers-con-lr5e6-s42": {
                    "ctx_key": "pers",
                    "regime": "con",
                    "lr_key": "lr5e6",
                    "seed": 42,
                    "selection": {"in_window": True, "step": 90, "delta_logp_mean": 6.3},
                },
                "mk-pers-con-lr1e4-s42": {
                    "ctx_key": "pers",
                    "regime": "con",
                    "lr_key": "lr1e4",
                    "seed": 42,
                    "selection": {"in_window": False, "step": 5, "delta_logp_mean": 14.0},
                },
            }
        },
    }
    with pytest.raises(AssertionError):
        X.marker_arms(man)  # only 1 group -> count assert fires (16 expected)
    groups = {}
    for arm_id, e in man["marker"]["arms"].items():
        if e["selection"]["in_window"]:
            groups.setdefault((e["ctx_key"], e["regime"], e["seed"]), []).append(
                (X.LR_BY_TAG[e["lr_key"]], arm_id)
            )
    lr, chosen = min(groups[("pers", "con", 42)])
    assert chosen == "mk-pers-con-lr5e6-s42" and lr == pytest.approx(5e-6)


def test_adapter_subfolder_resolution_order():
    arms = {a.arm_id: a for a in X.all_arms()}
    # (1) issue1586 REUSED_LORA_ARMS pers subfolder wins
    sub = X.adapter_subfolder(arms["syc-pers-po-lr1e5-s42"])
    assert sub == "issue1481/syc-pers-po-lr1e5-s42/checkpoint-10"
    # (2) fu7-reused con arm resolves via the issue1481_cells registry
    sub2 = X.adapter_subfolder(arms["syc-pers-con-lr1e5-s42"])
    assert sub2.startswith("adapters/issue1090_fu7/syc-c3-lr1e5/checkpoint-")
    # (3) cas seed-42 resolves to the reused #1434 ladder run dir
    cas42 = next(a for a in arms.values() if a.beh_key == "cas" and a.seed == 42)
    sub3 = X.adapter_subfolder(cas42)
    assert sub3.startswith("issue1434/ws")
    # (4) marker fresh convention
    mk_bare = next(a for a in arms.values() if a.kind == "marker" and a.ctx_key == "bare")
    assert X.adapter_subfolder(mk_bare).startswith("issue1481/marker/mk-bare-")


def test_base_unit_routing_and_decode_caps():
    assert X.base_unit_for("mk-pers-con-lr5e6-s42") == "base_mk"
    assert X.base_unit_for("syc-pers-con-lr1e5-s42") == "base_content"
    assert X.max_new_tokens_for("base_mk") == X.MAX_NEW_MARKER
    assert X.max_new_tokens_for("imp-bare-con-lr1e5-s42") == X.MAX_NEW_CONTENT


# ── corpus sampling (fake tokenizer at the external boundary) ────────────────


class _FakeTok:
    """Signature-conformant tokenizer fake (chat render + call -> input_ids)."""

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        return "SYS " + messages[-1]["content"]

    def __call__(self, text, add_special_tokens=False, **kw):
        return {"input_ids": list(range(len(text.split())))}


def test_sample_train_prompts_stratified_cap_and_valtest_skip():
    pool = []
    for i in range(40):
        pool.append({"prompt": f"lm prompt {i}", "corpus": "lmsys", "i": i})
    for i in range(40):
        pool.append({"prompt": f"wc prompt {i}", "corpus": "wildchat", "i": 40 + i})
    pool.append({"prompt": "lm " + "x " * 50, "corpus": "lmsys", "i": 80})  # over cap
    meta = {"n_lmsys": 40, "n_wildchat": 40}
    valtest = ["lm prompt 0", "wc prompt 0"]
    out = X.sample_train_prompts(pool, meta, _FakeTok(), valtest, n_train=20, seed=7, token_cap=10)
    assert len(out["rows"]) == 20
    assert out["proportions"] == {"lmsys": 10, "wildchat": 10}
    prompts = {r["prompt"] for r in out["rows"]}
    assert not prompts & set(valtest)  # exact-text disjoint from val/test


def test_sample_train_prompts_dedups_duplicate_shas_and_tops_up():
    """#1768 crash-fix r4 (production crash: 'duplicate prompt shas in sample').

    The n1M pool holds exact-duplicate prompt texts (within AND across corpora
    — the #779 near-dupe screen was vs the eval targets, never a within-corpus
    exact-dedup); the seeded draw must skip already-taken shas, top up from the
    continuing permutation order, and never collide with the pinned val/test
    shas. Fails pre-fix: the drawn rows contained duplicate shas.
    """
    pool = []
    for i in range(15):
        pool.append({"prompt": f"lm text {i}", "corpus": "lmsys"})
    for i in range(15):  # exact duplicate rows inside the corpus (the crash shape)
        pool.append({"prompt": f"lm text {i}", "corpus": "lmsys"})
    for i in range(12):
        pool.append({"prompt": f"wc text {i}", "corpus": "wildchat"})
    pool.append({"prompt": "lm text 0", "corpus": "wildchat"})  # cross-corpus dup
    pool.append({"prompt": "vt text 0", "corpus": "wildchat"})  # pinned val/test text
    meta = {"n_lmsys": 100, "n_wildchat": 50}
    valtest = ["vt text 0", "vt text 1"]
    out = X.sample_train_prompts(
        pool, meta, _FakeTok(), valtest, n_train=21, seed=42, token_cap=100
    )
    rows = out["rows"]
    assert len(rows) == 21
    shas = [r["sha"] for r in rows]
    assert len(set(shas)) == 21  # pre-fix: duplicate shas survive the draw
    assert out["n_skipped_dup"] >= 1  # the dedup branch actually fired
    # pinned val/test rows keep priority — train never takes their shas
    assert not set(shas) & {X.prompt_sha(v) for v in valtest}
    # deterministic under the seed across two independent runs
    out2 = X.sample_train_prompts(
        pool, meta, _FakeTok(), valtest, n_train=21, seed=42, token_cap=100
    )
    assert out2["rows"] == rows


def test_sample_train_prompts_fails_loud_when_unique_pool_exhausted():
    """Dedup must not mask pool exhaustion: fewer UNIQUE texts than the quota
    still fails loud at the per-corpus kept==quota assert (genuine data bug)."""
    pool = [{"prompt": "same lm text", "corpus": "lmsys"} for _ in range(10)]
    pool += [{"prompt": f"wc text {i}", "corpus": "wildchat"} for i in range(10)]
    meta = {"n_lmsys": 10, "n_wildchat": 10}
    with pytest.raises(AssertionError, match="only 1/5"):
        X.sample_train_prompts(pool, meta, _FakeTok(), [], n_train=10, seed=1, token_cap=100)


# ── loader sha-alignment (synthetic stores, permuted + dropped rows) ─────────


def _mk_store(path, span_layers, shas, qidx, extra=None):
    store = {
        "schema_version": 1,
        "row_sha": list(shas),
        "row_question_idx": list(qidx),
        "arms": {
            span: {li: torch.as_tensor(mat, dtype=torch.float16) for li, mat in per.items()}
            for span, per in span_layers.items()
        },
        "metadata": extra or {},
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(store, path)


def _write_sample(out_root, n_train, n_val, n_test, shas):
    rows = [
        {"prompt": f"p{i}", "corpus": "lmsys" if i % 2 == 0 else "wildchat", "sha": s}
        for i, s in enumerate(shas)
    ]
    (out_root / "inputs").mkdir(parents=True, exist_ok=True)
    (out_root / "inputs" / "corpus_sample.json").write_text(
        json.dumps({"rows": rows, "n_train": n_train, "n_val": n_val, "n_test": n_test})
    )


def test_load_corpus_cell_aligns_by_sha_not_order(tmp_path):
    import issue1768_fit as fit

    d, layer = 4, 0
    n = 30
    shas = [f"s{i}" for i in range(n)]
    _write_sample(tmp_path, 20, 4, 6, shas)
    # base: drops row 3; values encode the global qidx in coord 0
    b_keep = [i for i in range(n) if i != 3]
    base_mat = np.zeros((len(b_keep), d))
    base_mat[:, 0] = b_keep
    _mk_store(
        tmp_path / "corpus_capture" / "base_content" / "pooled.pt",
        {"context": {layer: base_mat}, "response": {layer: base_mat + 100}},
        [shas[i] for i in b_keep],
        b_keep,
    )
    # plus: drops row 7, REVERSED order — loader must re-align by sha
    p_keep = [i for i in reversed(range(n)) if i != 7]
    plus_mat = np.zeros((len(p_keep), d))
    plus_mat[:, 0] = p_keep
    _mk_store(
        tmp_path / "corpus_capture" / "arm-a" / "pooled.pt",
        {"context": {layer: plus_mat}, "response": {layer: plus_mat + 200}},
        [shas[i] for i in p_keep],
        p_keep,
    )
    t_keep = list(range(n))
    tf_mat = np.zeros((n, d))
    tf_mat[:, 0] = t_keep
    _mk_store(
        tmp_path / "corpus_capture_tf" / "arm-a" / "pooled_tf.pt",
        {"response": {layer: tf_mat + 300}},
        [shas[i] for i in t_keep],
        t_keep,
    )
    cell = fit.load_corpus_cell("arm-a", layer, tmp_path)
    kept = [i for i in range(n) if i not in (3, 7)]
    assert list(cell["qidx"]) == kept
    np.testing.assert_allclose(cell["C0"][:, 0], kept)
    np.testing.assert_allclose(cell["Cplus"][:, 0], kept)  # re-aligned from reversed order
    np.testing.assert_allclose(cell["Vplus"][:, 0], np.asarray(kept) + 200)
    np.testing.assert_allclose(cell["Vplus_tf"][:, 0], np.asarray(kept) + 300)
    assert (cell["split"] == "train").sum() == 18  # 20 train qidx minus dropped 3, 7
    assert (cell["split"] == "val").sum() == 4
    assert (cell["split"] == "test").sum() == 6


def test_verdict_lattice_pure():
    import issue1768_fit as fit

    assert fit.verdict_from(0.5, [0.1, 0.9]) == "Changed"
    assert fit.verdict_from(-0.5, [-0.9, -0.1]) == "Unchanged"
    assert fit.verdict_from(0.5, [-0.1, 0.9]) == "Unresolved"
    assert fit.verdict_from(-0.1, [-0.3, 0.2]) == "Unresolved"
    # exhaustive + disjoint on a probe grid
    for d_stat in (-1.0, 0.0, 1.0):
        for ci in ([-2, -1], [-1, 1], [1, 2]):
            assert fit.verdict_from(d_stat, list(ci)) in {"Changed", "Unchanged", "Unresolved"}


def test_fit_map_and_map_change_block_synthetic(tmp_path):
    """End-to-end p8 math on a tiny synthetic cell: a LARGE injected map change
    reads Changed; the M0-vs-M0 self-comparison reads not-Changed."""
    import issue1768_fit as fit

    rng = np.random.default_rng(0)
    n, d = 60, 5
    C0 = rng.standard_normal((n, d))
    W_true = rng.standard_normal((d, d))
    V0 = C0 @ W_true + 0.01 * rng.standard_normal((n, d))
    Cp = C0 + 0.01 * rng.standard_normal((n, d))
    Vp = Cp @ (W_true + 3.0) + 0.01 * rng.standard_normal((n, d))  # big map change
    split = np.array(["train"] * 40 + ["val"] * 8 + ["test"] * 12)
    cell = {
        "C0": C0,
        "V0": V0,
        "Cplus": Cp,
        "Vplus": Vp,
        "Vplus_tf": Vp,
        "split": split,
        "corpus": np.array(["lmsys", "wildchat"] * 30),
        "sha": [f"s{i}" for i in range(n)],
    }
    tr, val, te = fit._split_idx(split)
    dev = fit._device()
    _pred0, _m0, pay0 = fit._fit_map(C0, V0, tr, val, te, dev)
    _predp, _mp, payp = fit._fit_map(Cp, Vp, tr, val, te, dev)
    block = fit._map_change_block(cell, pay0, payp, pay0["selected_lambda"], dev, smoke=True)
    assert block["verdict"] == "Changed" and block["D"] > 0
    null_block = fit._map_change_block(cell, pay0, pay0, pay0["selected_lambda"], dev, smoke=True)
    assert null_block["delta_med"] == pytest.approx(0.0, abs=1e-9)
    assert null_block["verdict"] in {"Unchanged", "Unresolved"}
    reads = fit._map_reads(_pred0, V0[te])
    assert -1.0 <= reads["mean_cos"] <= 1.0 and reads["heldout_r2"] <= 1.0
    ib = fit._identity_bias_reads(C0[tr], V0[tr], C0[te], V0[te])
    assert ib["applicable"] is True
    tf_out = fit._transfer_fold(cell, dev)
    assert set(tf_out) == {"lmsys->wildchat", "wildchat->lmsys", "note"}
    assert "dst-corpus TRAIN rows" in tf_out["note"]  # stated deviation (Minor)


# ── disjoint-halves registration (Statistics Must-Fix) ───────────────────────


def _panel_store(path, ctx_rows, layer, d):
    """ctx_rows: {ctx: {qidx: (resp_vec, ctx_vec)}} -> panel store."""
    row_meta, resp, ctxm = [], [], []
    for cid, qs in ctx_rows.items():
        for q, (rv, cv) in sorted(qs.items()):
            row_meta.append({"context_id": cid, "question_idx": q})
            resp.append(rv)
            ctxm.append(cv)
    store = {
        "row_meta": row_meta,
        "arms": {
            "response": {layer: torch.as_tensor(np.asarray(resp), dtype=torch.float16)},
            "context": {layer: torch.as_tensor(np.asarray(ctxm), dtype=torch.float16)},
        },
        "metadata": {},
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(store, path)


def test_disjoint_halves_w_and_delta_baselines(tmp_path):
    import issue1768_directions as dirs

    d, layer = 4, 0
    arm = X.Arm("imp-pers-con-lr3e5-s42", "content", "imp", "pers", "con", 42, 3e-5, 30, 0.8)
    # base: even-q rows = +1 vector, odd-q rows = -1 vector (distinct half means)
    base_rows = {
        "ctxS": {
            q: ((np.ones(d) if q % 2 == 0 else -np.ones(d)), np.full(d, 5.0)) for q in range(4)
        },
        "other": {q: (np.zeros(d), np.zeros(d)) for q in range(4)},
    }
    arm_rows = {
        "ctxS": {q: (np.full(d, 10.0), np.full(d, 6.0)) for q in range(4)},
        "other": {q: (np.zeros(d), np.zeros(d)) for q in range(4)},
    }
    _panel_store(tmp_path / "panel_capture" / "base_imp" / "pooled.pt", base_rows, layer, d)
    _panel_store(tmp_path / "panel_capture" / arm.arm_id / "pooled.pt", arm_rows, layer, d)
    legs = dirs.panel_write_legs(tmp_path, arm, layer)
    assert legs["src_ctx"] == "ctxS"  # SRC_CTX_POS['pers'] == 0, panel order
    np.testing.assert_allclose(legs["w_primary"], np.full(d, 10.0) - np.ones(d))  # v0 half A=+1
    np.testing.assert_allclose(legs["v0_half_B"], -np.ones(d))  # δ leg baseline = half B
    np.testing.assert_allclose(legs["w_shared_record_only"], np.full(d, 10.0))  # v̄0(all)=0
    # δ leg consumes the OTHER half than ŵ — disjoint by construction
    tb_dir = tmp_path / "delta_tf" / arm.arm_id
    tb_dir.mkdir(parents=True)
    torch.save(
        {
            "tbar": {layer: torch.full((d,), 3.0)},
            "tbar_even": {layer: torch.full((d,), 3.5)},
            "tbar_odd": {layer: torch.full((d,), 2.5)},
            "n_rows": 8,
            "meta": {},
        },
        tb_dir / "tbar.pt",
    )
    dleg = dirs.delta_leg(tmp_path, arm, layer, legs)
    np.testing.assert_allclose(dleg["delta_primary"], np.full(d, 3.0) + 1.0)
    np.testing.assert_allclose(dleg["delta_shared_record_only"], np.full(d, 3.0))
    assert "delta_split_half_cos" in dleg and legs["w_split_half_cos"] is not None
    # δ reliability halves use DISJOINT quarter-split baselines (odd-q rows
    # split further); the shared-B read is retained record-only (round-2 fix)
    np.testing.assert_allclose(legs["v0_half_B1"], -np.ones(d))  # q=1
    np.testing.assert_allclose(legs["v0_half_B2"], -np.ones(d))  # q=3
    assert "delta_split_half_cos_sharedB_record_only" in dleg
    assert "delta_split_half_cos_upper_bound_sharedB" not in dleg


def test_null_bands_and_shuffled_row_shapes():
    import issue1768_directions as dirs

    rng = np.random.default_rng(3)
    d = 6
    w = rng.standard_normal(d)
    sigma = np.eye(d)
    sig = {"sigma": sigma, "chol": np.linalg.cholesky(sigma), "top_eig": np.eye(d)[0]}
    bands = dirs.null_bands(w, sig, rng)
    for fam in ("isotropic", "corpus_covariance"):
        b = bands[fam]
        assert -1 <= b["p2_5"] <= b["p97_5"] <= 1
    vp = rng.standard_normal((6, d))
    v0 = rng.standard_normal((6, d))
    sh = dirs.shuffled_row_band(vp, v0, w, rng)
    assert sh["n_draws"] == dirs.N_NULL_DRAWS
    rr = dirs.rank_read(rng.standard_normal((30, d)), w)
    assert 0 <= rr["top1_var_share"] <= 1 and rr["participation_ratio"] >= 1
    gr = dirs.gate_read(
        rng.standard_normal((50, d)),
        rng.standard_normal((50, d)),
        rng.standard_normal(d),
        w,
        sig,
    )
    assert -1 <= gr["spearman_rho"] <= 1
    sf = dirs.scalar_fit_residual(w, w * 2.0)
    assert sf["a"] == pytest.approx(0.5) and sf["residual_share"] == pytest.approx(0.0, abs=1e-12)


def test_estimator_validity_refuses_n_train_below_d():
    import issue1768_fit as fit

    rng = np.random.default_rng(1)
    n, d = 80, 70  # n_train=40 < d=70 and d>64 -> refuse (plan §11 duty)
    Xd = rng.standard_normal((n, d))
    Yd = rng.standard_normal((n, d))
    with pytest.raises(AssertionError, match="under-determined"):
        fit._fit_map(Xd, Yd, np.arange(40), np.arange(40, 60), np.arange(60, 80), fit._device())


# ── round-2 pins (code-review v1 punch list) ─────────────────────────────────


def test_loader_coverage_floor_trips(tmp_path):
    """fit.load_corpus_cell refuses <90% sha-join coverage (fit.py floor)."""
    import issue1768_fit as fit

    d, layer, n = 3, 0, 30
    shas = [f"s{i}" for i in range(n)]
    _write_sample(tmp_path, 20, 4, 6, shas)
    base_mat = np.zeros((n, d))
    _mk_store(
        tmp_path / "corpus_capture" / "base_content" / "pooled.pt",
        {"context": {layer: base_mat}, "response": {layer: base_mat}},
        shas,
        list(range(n)),
    )
    p_keep = list(range(20))  # 20/30 = 0.67 < 0.9 coverage floor
    pm = np.zeros((len(p_keep), d))
    _mk_store(
        tmp_path / "corpus_capture" / "arm-a" / "pooled.pt",
        {"context": {layer: pm}, "response": {layer: pm}},
        [shas[i] for i in p_keep],
        p_keep,
    )
    _mk_store(
        tmp_path / "corpus_capture_tf" / "arm-a" / "pooled_tf.pt",
        {"response": {layer: base_mat}},
        shas,
        list(range(n)),
    )
    with pytest.raises(AssertionError):
        fit.load_corpus_cell("arm-a", layer, tmp_path)


def test_pinned_split_mismatch_raises(monkeypatch):
    """assert_pinned_split fail-louds on a sha mismatch (and PASSes on the
    committed pins — the real recompute against the committed #779 block)."""
    got = X.assert_pinned_split()  # committed pins reproduce
    assert got["n_val"] == X.N_VAL and got["n_test"] == X.N_TEST
    monkeypatch.setattr(
        X,
        "pinned_split_block",
        lambda: {"pinned_val_sha256": "deadbeef", "pinned_test_sha256": "deadbeef"},
    )
    with pytest.raises(AssertionError):
        X.assert_pinned_split()


def test_fit_cell_call_shape_binds():
    """Signature-bind the exact panel_fit_for_arm -> fit_cell call shape
    (the smoke cannot run fit_cell at its (28, 3584) production contract)."""
    import inspect

    import issue722_fit_M as fit_m
    import issue1768_fit as fit

    sig = inspect.signature(fit_m.fit_cell)
    sig.bind(
        fit.FIT_CELL_BEHAVIOR_COL,
        19,
        [],
        {"r_b": {fit.FIT_CELL_BEHAVIOR_COL: {"diffmeans": np.zeros((28, 4))}}},
        None,
        include_mlp=False,
        floors="batched",
        loco="batched",
    )


def test_floor_seed_deterministic_across_processes():
    """floor_seed_for never rides Python hash(): identical across processes
    with DIFFERENT PYTHONHASHSEED values (fails pre-fix, when the seed used
    X.FLOOR_SEED + hash(cond) % 1000)."""
    import os
    import subprocess
    import sys

    outs = []
    for hs in ("0", "424242"):
        env = {**os.environ, "PYTHONHASHSEED": hs}
        r = subprocess.run(
            [
                sys.executable,
                "-c",
                "import sys; sys.path.insert(0, 'scripts'); "
                "import issue1768_fit as f; "
                "print(f.floor_seed_for('M0'), f.floor_seed_for('Mplus'))",
            ],
            cwd=REPO,
            env=env,
            capture_output=True,
            text=True,
            check=True,
        )
        outs.append(r.stdout.strip())
    assert outs[0] == outs[1]
    assert outs[0] == f"{X.FLOOR_SEED} {X.FLOOR_SEED + 1}"


def test_panel_baseline_reads_keys():
    """identity+bias + kNN attach to the panel fit_cell maps (Major 6)."""
    import issue1768_fit as fit
    from issue722_load_activations import CellRecord

    rng = np.random.default_rng(5)
    d = 4
    cells = [
        CellRecord(
            behavior="b",
            source_cid="c",
            target_cid="c",
            layer=0,
            c0=rng.standard_normal(d),
            cplus=rng.standard_normal(d),
            v0=rng.standard_normal(d),
            vplus=rng.standard_normal(d),
            family="c",
        )
        for _ in range(8)
    ]
    out = fit._panel_baseline_reads(cells)
    for name in ("M0", "Mplus"):
        b = out[name]
        assert b["applicable"] is True
        assert "heldout_r2" in b and "knn_euclidean" in b and "knn_cosine" in b


def test_stage_reused_panel_base_requires_raw_rows(tmp_path, monkeypatch):
    """The Critical pin (p4-base-raw-rows-missing-crash): a base capture tree
    whose Hub copy lacks raw_rows.json is a staging FAILURE (fresh capture),
    never a warn-and-continue; a pers-arm tree keeps raw_rows optional."""
    import issue1768_capture as cap

    from explore_persona_space.orchestrate import hub as hub_mod

    cfg = cap.Cfg(out_root=tmp_path, phases=())
    monkeypatch.setattr(cap, "_reused_1586_hub_name", lambda unit: unit)
    monkeypatch.setattr(cap, "_vintage_ok", lambda commit: True)
    staged: list[str] = []

    def fake_probe(fn, what=""):
        return "raw_rows.json" not in what  # Hub carries pooled + manifest only

    def fake_stage(repo, hub_path, dest, repo_type=""):
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(json.dumps({"git_commit": "abc"}))
        staged.append(hub_path)

    monkeypatch.setattr(hub_mod, "retry_transient", fake_probe)
    monkeypatch.setattr(hub_mod, "stage_hub_file", fake_stage)
    # base tree: REQUIRED raw_rows missing on Hub -> False, and nothing staged
    assert cap._stage_reused_panel(cfg, "capture", "base_syc") is False
    assert staged == []
    assert not (tmp_path / "panel_capture" / "base_syc").exists()
    # pers-arm tree: raw_rows optional -> True (pooled + manifest staged)
    assert cap._stage_reused_panel(cfg, "capture", "syc-pers-con-lr3e5-s42") is True
    assert any(p.endswith("pooled.pt") for p in staged)
    # stale partial base dir (pooled without raw_rows) is dropped, not resumed
    stale = tmp_path / "panel_capture" / "base_imp"
    stale.mkdir(parents=True)
    (stale / "pooled.pt").write_text("x")
    (stale / "manifest.json").write_text("{}")
    assert cap._stage_reused_panel(cfg, "capture", "base_imp") is False
    assert not stale.exists()


def test_smoke_forces_smoke_hf_prefix():
    """--smoke enforces the _smoke upload prefix even under an explicit
    --hf-prefix (a smoke run must never write the production Hub bucket)."""
    import issue1768_capture as cap

    cfg, _ = cap.parse_args(
        ["--out-root", "/tmp/x", "--smoke", "--hf-prefix", "issue1768_mapshift"]
    )
    assert cfg.hf_prefix == "issue1768_mapshift_smoke"
    cfg2, _ = cap.parse_args(["--out-root", "/tmp/x", "--hf-prefix", "issue1768_mapshift"])
    assert cfg2.hf_prefix == "issue1768_mapshift"


def test_barrier_units_p2_has_no_barrier():
    """Major 5: no p2 wave barrier (base outputs feed p3, not p2 siblings);
    p4 keeps its base barrier (arm tf units consume base raw rows)."""
    import issue1768_capture as cap

    assert cap._barrier_units("p2", ["base_content", "base_mk", "arm-a"]) == set()
    assert cap._barrier_units("p4", ["base:base_syc", "arm:a"]) == {"base:base_syc"}


def test_p6_gen_units_ride_p2_queue(tmp_path):
    """Plan §9 Must-Fix wiring: the p6 GPU legs are p2-queue units, gated on
    their own arm's corpus unit (merged-dir lifecycle), pending until
    gen_done.json lands."""
    import issue1768_capture as cap

    arm_id = next(
        a.arm_id
        for a in X.all_arms()
        if a.kind == "content" and a.ctx_key == "pers" and a.seed == 42
    )
    cfg = cap.Cfg(out_root=tmp_path, phases=(), arms=(arm_id,))
    pend = cap._pending_units(cfg, "p2")
    assert f"p6g:{arm_id}" in pend and arm_id in pend
    assert not cap._unit_ready(cfg, "p2", f"p6g:{arm_id}")  # arm corpus not done
    (tmp_path / "corpus_capture" / arm_id).mkdir(parents=True)
    (tmp_path / "corpus_capture" / arm_id / "pooled.pt").write_text("x")
    assert cap._unit_ready(cfg, "p2", f"p6g:{arm_id}")
    (tmp_path / "rb_plus" / arm_id).mkdir(parents=True)
    (tmp_path / "rb_plus" / arm_id / "gen_done.json").write_text("{}")
    assert f"p6g:{arm_id}" not in cap._pending_units(cfg, "p2")


def test_p5_pending_maps_ft_onto_lora_cells_and_p6_excludes_ft(tmp_path):
    """Amendment: ft arms add NO p5 cells (shared t̄ via delta_arm_for) and NO
    p6 units; an ft-only scope still stages its paired LoRA cell."""
    import issue1768_capture as cap

    cfg = cap.Cfg(out_root=tmp_path, phases=(), arms=("imp-pers-ft-con-s42", X.PILOT_ARM))
    assert cap._pending_units(cfg, "p5") == [X.PILOT_ARM]  # ft pair == pilot cell
    cfg_ft_only = cap.Cfg(out_root=tmp_path, phases=(), arms=("mk-pers-ft-con-s42",))
    assert cap._pending_units(cfg_ft_only, "p5") == ["mk-pers-con-lr5e6-s42"]
    cfg_all = cap.Cfg(out_root=tmp_path, phases=())
    p6 = cap.p6_arm_ids(cfg_all)
    assert len(p6) == 6 and all("-ft-" not in a for a in p6)
    # p2/p3 unit counts carry the 16 ft arms: 74 = 2 base + 72 trained
    p2 = cap._pending_units(cfg_all, "p2")
    assert len([u for u in p2 if not u.startswith("p6g:")]) == 74
    assert len(cap._pending_units(cfg_all, "p3")) == 72


def test_ft_resolution_plumbing(tmp_path):
    """_needs_merge / _reused_1586_hub_name / _ft_ckpt_incomplete_reason on ft arms."""
    import issue1768_capture as cap

    cfg = cap.Cfg(out_root=tmp_path, phases=())
    arm = {a.arm_id: a for a in X.all_arms()}["imp-pers-ft-con-s42"]
    # ft staging pending counts as merge-bearing for the disk clamp
    assert cap._needs_merge(cfg, "p2:imp-pers-ft-con-s42")
    _root, ckpt = cap._ft_ckpt_dirs(cfg, arm)
    assert ckpt == tmp_path / "ft_ckpt" / arm.arm_id / X.ft_ckpt_subfolder(arm)
    ckpt.mkdir(parents=True)
    (ckpt / "config.json").write_text("{}")
    # config-only partial is INCOMPLETE (never satisfies the reuse predicate)
    assert cap._ft_ckpt_incomplete_reason(ckpt) == "no weight shards"
    assert cap._needs_merge(cfg, "p2:imp-pers-ft-con-s42")
    (ckpt / "model.safetensors").write_bytes(b"00")
    assert cap._ft_ckpt_incomplete_reason(ckpt) is None
    assert not cap._needs_merge(cfg, "p2:imp-pers-ft-con-s42")
    # index-bearing dir: every index-listed shard must be present
    idx = {
        "weight_map": {
            "a": "model-00001-of-00002.safetensors",
            "b": "model-00002-of-00002.safetensors",
        }
    }
    (ckpt / "model.safetensors.index.json").write_text(json.dumps(idx))
    assert "2/2 weight shards missing" in cap._ft_ckpt_incomplete_reason(ckpt)
    # p4 reuse: ft arm ids ARE the #1586 capture-tree dir names
    assert cap._reused_1586_hub_name("imp-pers-ft-con-s42") == "imp-pers-ft-con-s42"
    assert cap._reused_1586_hub_name(X.PILOT_ARM) == "imp-pers-lora-con-s42"
    assert cap._reused_1586_hub_name("base_mk") == "base_mk"


def test_stage_ft_checkpoint_restages_partial_and_repairs_tokenizer(tmp_path, monkeypatch):
    """Production-body test of _stage_ft_checkpoint: partial dir removed
    before restage (#1586 fu r7), staged tree completeness re-verified, base
    tokenizer repaired (#1112). Fakes ONLY at the network boundary
    (hub.stage_hub_prefix / AutoTokenizer.from_pretrained), signature-mirrored."""
    import issue1768_capture as cap

    from explore_persona_space.orchestrate import hub

    cfg = cap.Cfg(out_root=tmp_path, phases=())
    arm = {a.arm_id: a for a in X.all_arms()}["imp-pers-ft-con-s42"]
    root, ckpt = cap._ft_ckpt_dirs(cfg, arm)
    ckpt.mkdir(parents=True)
    (ckpt / "config.json").write_text("{}")  # config-only partial -> restage

    staged = []

    def fake_stage_hub_prefix(
        repo_id, prefix, dest_dir, *, repo_type="dataset", revision=None, token=None, max_workers=6
    ):
        staged.append((repo_id, prefix, repo_type))
        d = Path(dest_dir) / prefix  # verbatim prefix mirror (real helper layout)
        d.mkdir(parents=True, exist_ok=True)
        (d / "config.json").write_text("{}")
        (d / "model.safetensors").write_bytes(b"00")
        return [d / "config.json", d / "model.safetensors"]

    class _FakeTokSave:
        def save_pretrained(self, d):
            (Path(d) / "tokenizer_config.json").write_text("{}")

    import transformers

    monkeypatch.setattr(hub, "stage_hub_prefix", fake_stage_hub_prefix)
    monkeypatch.setattr(
        transformers.AutoTokenizer,
        "from_pretrained",
        classmethod(lambda _c, *a, **k: _FakeTokSave()),
    )
    got_root, got_ckpt = cap._stage_ft_checkpoint(cfg, arm)
    assert (got_root, got_ckpt) == (root, ckpt)
    assert staged == [(X.FT_OVERFLOW_REPO, "issue1586/imp-pers-ft-con-s42/checkpoint-14", "model")]
    assert (ckpt / "model.safetensors").exists()  # partial replaced by staged tree
    assert (ckpt / "tokenizer_config.json").exists()  # tokenizer repaired from base
    # _resolve_unit_model routes ft arms here and returns the staged ckpt +
    # the cleanup root (stage -> consume -> delete lifecycle)
    model_path, cleanup = cap._resolve_unit_model(cfg, arm.arm_id)
    assert model_path == str(ckpt) and cleanup == root
    cap._cleanup_merged(cleanup)
    assert not root.exists()


def _mk_p6_fixture(n_q=2, n_roll=2, d=4, n_layers=2):
    """Synthetic rollouts/scores/acts blobs keyed by the REAL enumeration."""
    import issue779_extract_rb as E

    rollouts = {
        side: {
            f"t_{side}_p0": {
                f"q{j}": [f"resp {side} {j} {r}" for r in range(n_roll)] for j in range(n_q)
            }
        }
        for side in ("pos", "neg")
    }
    acts = {"pos": {}, "neg": {}}
    scores = {"pos": {}, "neg": {}}
    for side, val, score in (("pos", 1.0, 90.0), ("neg", -1.0, 10.0)):
        for _p, _q, _question, _ci, _comp, cid in E._iter_rollout_records(rollouts[side]):
            acts[side][cid] = torch.full((n_layers, d), val, dtype=torch.float16)
            scores[side][cid] = score
    scores_blob = {
        "arms": {
            s: {"scores": scores[s], "draw_stats": {"n_rollouts_judged": n_q * n_roll}}
            for s in ("pos", "neg")
        }
    }
    acts_blob = {"layers": list(range(n_layers)), "acts": acts}
    return rollouts, scores_blob, acts_blob


def test_reduce_rb_from_persisted_filter_and_math():
    """The p6 CPU post-filter: threshold keep, drop-never-coerce, r_B math."""
    import issue1768_capture as cap

    rollouts, scores_blob, acts_blob = _mk_p6_fixture()
    # drop one pos rollout (None = judge-dropped) + push one below threshold
    pos_cids = list(scores_blob["arms"]["pos"]["scores"])
    scores_blob["arms"]["pos"]["scores"][pos_cids[0]] = None
    scores_blob["arms"]["pos"]["scores"][pos_cids[1]] = 40.0  # not > 50 -> dropped
    out = cap.reduce_rb_from_persisted("t", rollouts, scores_blob, acts_blob, smoke=False)
    a = out["counts"]["arms"]["pos"]
    assert (a["total"], a["kept"], a["dropped_refusal_or_invalid"]) == (4, 2, 1)
    assert a["dropped_below_threshold"] == 1
    np.testing.assert_allclose(np.asarray(out["r_b"]), np.full((2, 4), 2.0))  # +1 - (-1)


def test_reduce_rb_zero_kept_production_raises_smoke_falls_back():
    """Production zero-kept arm fail-louds (yield failure, never fabricated);
    under --smoke the SAME state keep-all falls back, LABELED (#1345 gate
    demotion — the production raise is byte-untouched)."""
    import issue1768_capture as cap

    rollouts, scores_blob, acts_blob = _mk_p6_fixture()
    for cid in scores_blob["arms"]["neg"]["scores"]:
        scores_blob["arms"]["neg"]["scores"][cid] = 90.0  # NEG kept needs < 50
    with pytest.raises(AssertionError, match="zero kept rollouts"):
        cap.reduce_rb_from_persisted("t", rollouts, scores_blob, acts_blob, smoke=False)
    out = cap.reduce_rb_from_persisted("t", rollouts, scores_blob, acts_blob, smoke=True)
    assert out["counts"]["arms"]["neg"]["smoke_keep_all_fallback"] is True
    assert out["counts"]["arms"]["pos"]["smoke_keep_all_fallback"] is False


# ── pnf: matched-text capture-noise floor (plan v7 follow-up) ────────────────


def _pnf_write_store(path, shas, qidx, vecs_by_layer, regime, gpu="testgpu"):
    """Minimal conforming replicate store (the run_noise_floor_unit schema)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "schema_version": 1,
            "row_sha": list(shas),
            "row_question_idx": list(qidx),
            "arms": {
                "response": {
                    li: torch.as_tensor(np.asarray(v), dtype=torch.float16)
                    for li, v in vecs_by_layer.items()
                }
            },
            "metadata": {"regime": regime, "gpu_name": gpu},
        },
        path,
    )


def test_pnf_units_selection_and_pending(tmp_path):
    """Smoke = base_content only (plan v7 §4.4); --arms filters trained units
    with base_content always kept; pnf pending is per unit x replicate and
    satisfied by store presence (resume-aware headroom gate input)."""
    import issue1768_capture as cap

    assert cap._pnf_units(cap.Cfg(out_root=tmp_path, phases=(), smoke=True)) == ["base_content"]
    assert cap._pnf_units(cap.Cfg(out_root=tmp_path, phases=())) == list(cap.PNF_UNITS)
    cfg = cap.Cfg(out_root=tmp_path, phases=(), arms=("mk-pers-con-lr5e6-s42",))
    assert cap._pnf_units(cfg) == ["base_content", "mk-pers-con-lr5e6-s42"]
    pend = cap._pending_units(cfg, "pnf")
    assert sorted(pend) == sorted(
        f"{u}:{rep}" for u in cap._pnf_units(cfg) for rep in cap.PNF_REPLICATES
    )
    st = tmp_path / "noise_floor" / "base_content" / "pooled_nf_r1.pt"
    st.parent.mkdir(parents=True)
    st.write_text("x")
    assert "base_content:r1" not in cap._pending_units(cfg, "pnf")
    assert "base_content:r2" in cap._pending_units(cfg, "pnf")


def test_pnf_subsample_deterministic_and_unique(tmp_path):
    """Seed-42 TRAIN draw: deterministic across calls, unique shas, stable
    sorted-sha fingerprint; duplicate train shas fail loud (r4 postcondition)."""
    import issue1768_capture as cap

    shas = [f"s{i:02d}" for i in range(10)]
    (tmp_path / "inputs").mkdir(parents=True)
    sample = {
        "rows": [{"prompt": f"q{i}", "corpus": "lmsys", "sha": s} for i, s in enumerate(shas)]
        + [{"prompt": "v", "corpus": "valtest", "sha": "vt0"}],
        "n_train": 10,
        "n_val": 1,
        "n_test": 0,
    }
    (tmp_path / "inputs" / "corpus_sample.json").write_text(json.dumps(sample))
    cfg = cap.Cfg(out_root=tmp_path, phases=())
    got1, key1 = cap._pnf_subsample(cfg)
    got2, key2 = cap._pnf_subsample(cfg)
    assert got1 == got2 and key1 == key2 and len(key1) == 64
    assert sorted(got1) == sorted(shas)  # min(2000, 10) draws all train shas
    assert "vt0" not in got1  # valtest rows never enter the subsample
    sample["rows"][1]["sha"] = "s00"  # duplicate train sha -> fail loud
    (tmp_path / "inputs" / "corpus_sample.json").write_text(json.dumps(sample))
    with pytest.raises(AssertionError, match="not unique"):
        cap._pnf_subsample(cfg)


def test_pnf_regime_mismatch_refuses_and_match_skips(tmp_path):
    """Regime-keyed resume (#722-r3 class): a stored regime differing from the
    current invocation refuses loud; exact matches skip WITHOUT re-capture."""
    import issue1768_capture as cap

    cfg = cap.Cfg(out_root=tmp_path, phases=(), layers=(19,))
    unit_dir = tmp_path / "noise_floor" / "base_content"
    bad = cap._pnf_regime(cfg, "base_content", "r1", "subkey")
    bad["tf_batch"] = 99  # a DIFFERENT regime than the current invocation
    _pnf_write_store(unit_dir / "pooled_nf_r1.pt", ["s0"], [0], {19: np.zeros((1, 4))}, bad)
    with pytest.raises(RuntimeError, match="DIFFERENT regime"):
        cap.run_noise_floor_unit(cfg, "base_content", tmp_path, ["s0"], "subkey")
    for rep in cap.PNF_REPLICATES:  # exact matches -> both skipped, no model load
        _pnf_write_store(
            unit_dir / f"pooled_nf_{rep}.pt",
            ["s0"],
            [0],
            {19: np.zeros((1, 4))},
            cap._pnf_regime(cfg, "base_content", rep, "subkey"),
        )
    assert cap.run_noise_floor_unit(cfg, "base_content", tmp_path, ["s0"], "subkey") == []


def test_pnf_rows_join_asserts_on_missing_sha(tmp_path):
    """Kill criterion (b): a subsample sha absent from the base tree fails
    loud at the sha-join (never a silent short row set)."""
    import issue1768_capture as cap

    base_dir = tmp_path / "corpus_capture" / "base_content"
    base_dir.mkdir(parents=True)
    rows = [
        {
            "persona": "base_content",
            "question_idx": i,
            "prompt_sha": f"s{i}",
            "prompt_token_ids": [1, 2, 3],
            "response_token_ids": [4],
            "finish_reason": "stop",
            "response_text": "x",
        }
        for i in range(3)
    ]
    cap._append_shard(base_dir, rows)
    (base_dir / "rows_spans.json").write_text(
        json.dumps(
            {
                "rows": [
                    {"prompt_sha": f"s{i}", "question_idx": i, "prefix_len": 1, "context_len": 2}
                    for i in range(3)
                ]
            }
        )
    )
    assert len(cap._pnf_rows_for_unit(base_dir, ["s0", "s2"])) == 2
    with pytest.raises(AssertionError, match="sha-join incomplete"):
        cap._pnf_rows_for_unit(base_dir, ["s0", "missing"])


def test_pnf_wall_gate():
    """Kill criterion (a): capture wall > 3x the first unit's halts loud."""
    import issue1768_capture as cap

    cap._pnf_wall_gate("u", 29.9, 10.0)  # under 3x -> no halt
    with pytest.raises(RuntimeError, match="kill criterion"):
        cap._pnf_wall_gate("u", 30.1, 10.0)


def test_pnf_reduce_math_verdicts_and_anchor(tmp_path):
    """End-to-end reduce on synthetic stores: sha-joined replicate distances
    (r2 permuted), p95/median floors, degenerate-zero flag, own-vs-fleet ratio
    verdict bands, H3 fraction, marker falsification, and the same-pass
    duplicate-sha anchor (byte-identical pairs only)."""
    import issue1768_capture as cap

    cfg = cap.Cfg(out_root=tmp_path, phases=(), arms=("mk-pers-con-lr5e6-s42",), layers=(19, 25))
    shas = [f"s{i:02d}" for i in range(4)]
    (tmp_path / "inputs").mkdir(parents=True)
    (tmp_path / "inputs" / "corpus_sample.json").write_text(
        json.dumps(
            {
                "rows": [
                    {"prompt": f"q{i}", "corpus": "lmsys", "sha": s} for i, s in enumerate(shas)
                ],
                "n_train": 4,
                "n_val": 0,
                "n_test": 0,
            }
        )
    )
    _sub, sub_sha = cap._pnf_subsample(cfg)
    d = 8
    zero = {s: np.zeros(d) for s in shas}

    def e0(c):
        v = np.zeros(d)
        v[0] = c
        return v

    perm = [shas[2], shas[0], shas[3], shas[1]]  # r2 store order != r1 (join test)

    def put(unit, rep, order, vecs):
        _pnf_write_store(
            tmp_path / "noise_floor" / unit / f"pooled_nf_{rep}.pt",
            order,
            list(range(len(order))),
            {li: np.stack([v[s] for s in order]) for li, v in vecs.items()},
            cap._pnf_regime(cfg, unit, rep, sub_sha),
        )

    # base_content: L19 identical (degenerate 0 floor); L25 constant 1.0
    put("base_content", "r1", shas, {19: zero, 25: zero})
    put("base_content", "r2", perm, {19: zero, 25: {s: e0(1.0) for s in shas}})
    # mk arm: L19 constant 0.5; L25 row-dependent [1,2,3,4] keyed by SHA (a
    # positional join would scramble these under the permuted r2 order)
    mk = "mk-pers-con-lr5e6-s42"
    l25 = {s: e0(float(i + 1)) for i, s in enumerate(shas)}
    put(mk, "r1", shas, {19: zero, 25: zero})
    put(mk, "r2", perm, {19: {s: e0(0.5) for s in shas}, 25: l25})

    fits_dir = tmp_path / "fits"
    fits_dir.mkdir()
    (fits_dir / f"{mk}_L25.json").write_text(
        json.dumps({"arm_id": mk, "layer": 25, "decomposition_tf": {"mean_norm_total": 5.0}})
    )
    (fits_dir / "cas-pers-con-lr1e5-s42_L19.json").write_text(
        json.dumps(
            {
                "arm_id": "cas-pers-con-lr1e5-s42",
                "layer": 19,
                "decomposition_tf": {"mean_norm_total": 30.0},
            }
        )
    )
    (tmp_path / "arm_registry.json").write_text(json.dumps({"arms": [{"arm_id": mk}]}))

    # anchor base tree: 1 byte-identical duplicate pair (dist 0.25) + 1
    # differing-response pair (excluded, counted)
    base_dir = tmp_path / "corpus_capture" / "base_content"
    base_dir.mkdir(parents=True)
    mk_row = lambda q, sha, resp: {  # noqa: E731
        "persona": "base_content",
        "question_idx": q,
        "prompt_sha": sha,
        "prompt_token_ids": [1],
        "response_token_ids": resp,
        "finish_reason": "stop",
        "response_text": "x",
    }
    cap._append_shard(
        base_dir,
        [
            mk_row(0, "dup", [7, 8]),
            mk_row(1, "dup", [7, 8]),
            mk_row(2, "d2", [9]),
            mk_row(3, "d2", [10]),
        ],
    )
    for f in ("rows_spans.json", "raw_rows.done.json", "manifest.json"):
        (base_dir / f).write_text("{}")
    resp = {li: torch.zeros(4, d, dtype=torch.float16) for li in (19, 25)}
    for li in (19, 25):
        resp[li][1, 0] = 0.25
    torch.save(
        {
            "row_sha": ["dup", "dup", "d2", "d2"],
            "row_question_idx": [0, 1, 2, 3],
            "arms": {"response": resp},
        },
        base_dir / "pooled.pt",
    )

    out = cap.noise_floor_reduce(cfg, fits_dir=fits_dir, results_dir=tmp_path / "res")

    assert out["floors"]["base_content"]["19"]["p95"] == 0.0
    assert ["base_content", 19] in out["degenerate_zero_floors"]
    assert out["floors"][mk]["19"]["p95"] == pytest.approx(0.5)
    assert out["floors"][mk]["25"]["median"] == pytest.approx(2.5)
    assert out["floors"][mk]["25"]["p95"] == pytest.approx(3.85)
    assert out["fleet_floor_p95"]["25"] == pytest.approx(3.85)
    rows_by = {(r["arm_id"], r["layer"]): r for r in out["ratio_table"]}
    mk_row_out = rows_by[(mk, 25)]
    assert mk_row_out["floor_source"] == "own"
    assert mk_row_out["ratio"] == pytest.approx(5.0 / 3.85)
    assert mk_row_out["verdict"] == "noise-ordered"
    cas_row = rows_by[("cas-pers-con-lr1e5-s42", 19)]
    assert cas_row["floor_source"] == "fleet"
    assert cas_row["ratio"] == pytest.approx(60.0)
    assert cas_row["verdict"] == "clear"
    crit = out["criteria"]
    assert crit["h3_n_arms"] == 2 and crit["h3_frac_above_fleet_floor_primary"] == 1.0
    assert crit["h3_met"] is True
    assert crit["marker_falsified"] is True  # own-floor ratio 1.30 <= 2
    anchor = out["same_pass_anchor"]
    assert anchor["n_duplicate_sha_groups"] == 2
    assert anchor["n_identical_response_pairs"] == 1
    assert anchor["n_pairs_response_text_differs"] == 1
    assert anchor["per_layer"]["19"]["median"] == pytest.approx(0.25)
    # low-level per-context artifacts land beside the aggregate
    assert (tmp_path / "res" / "noise_floor_percontext" / f"{mk}_L25.json").exists()
    pc = json.loads((tmp_path / "res" / "noise_floor_percontext" / f"{mk}_L25.json").read_text())
    assert sorted(pc["distance"]) == pytest.approx([1.0, 2.0, 3.0, 4.0])


def test_pnf_failloud_gates(tmp_path):
    """Degenerate-input probes for the remaining pnf gates: unresolvable base
    tree, replicate sha-set mismatch, arm-registry unit-pin miss."""
    import issue1768_capture as cap

    cfg = cap.Cfg(out_root=tmp_path, phases=(), layers=(19,))
    with pytest.raises(RuntimeError, match="base tree base_content unavailable"):
        cap._pnf_resolved_base_dir(cfg, "base_content")
    unit_dir = tmp_path / "noise_floor" / "u"
    _pnf_write_store(unit_dir / "pooled_nf_r1.pt", ["a"], [0], {19: np.zeros((1, 4))}, {})
    _pnf_write_store(unit_dir / "pooled_nf_r2.pt", ["b"], [0], {19: np.zeros((1, 4))}, {})
    with pytest.raises(AssertionError, match="sha sets differ"):
        cap._pnf_replicate_distances(unit_dir, [19])


# ── pfx: on-target prefix round (plan v8) ────────────────────────────────────


def test_pfx_registry_counts_and_context_map_matches_live_registry():
    """Plan §4.1/§4.2 arithmetic: 23 pfx2 units (12 own + 6 ctrl + 5 bases),
    18 tf units; the static context-id map equals the LIVE #1481 registry."""
    assert len(X.PFX_ARMS) == 12 and len(set(X.PFX_ARMS)) == 12
    assert set(X.PFX_CONTROL_ARMS) <= set(X.PFX_ARMS) and len(X.PFX_CONTROL_ARMS) == 6
    trained = [X.pfx_trained_unit(a, c) for a in X.PFX_ARMS for c in X.pfx_conditions_for(a)]
    assert len(trained) == 18
    assert X.pfx_base_units() == [
        "base_content@conv",
        "base_content@icl_syc",
        "base_content@pers",
        "base_mk@conv",
        "base_mk@pers",
    ]
    # control swap: pers -> conv; conv/icl -> pers (plan §4.2)
    assert X.pfx_context_id("syc-pers-con-lr1e5-s42", "ctrl") == "wildchat_prefix_real545"
    assert X.pfx_context_id("syc-conv-con-lr1e5-s42", "ctrl") == "persona_software_engineer"
    assert X.pfx_context_id("syc-icl-con-lr1e5-s42", "ctrl") == "persona_software_engineer"
    # mk decode routing through the base-unit name survives the @tag
    assert X.pfx_base_unit("mk-pers-con-lr5e6-s42", "own") == "base_mk@pers"
    assert X.pfx_unit_context_id("base_mk@conv") == "wildchat_prefix_real545"
    with pytest.raises(AssertionError):
        X.pfx_context_id("syc-pers-po-lr1e5-s42", "ctrl")  # not a control arm
    with pytest.raises(AssertionError):
        X.pfx_context_id("not-an-arm", "own")
    # static map == live registry (plan §4.2 grep pin, cross-checked)
    import issue1481_cells as c1481

    for key, cid in X.PFX_CONTEXT_ID_BY_KEY.items():
        assert c1481.context_id_for("sycophancy", key) == cid, (key, cid)


def test_pfx_resolve_context_shapes():
    """The registry render resolves every in-scope prefix id locally with the
    trained shape (pers: system; conv: 2 prefix turns; icl: user_wrap)."""
    pers = X.pfx_resolve_context("persona_software_engineer")
    assert pers.system and not pers.prefix_turns and pers.user_wrap is None
    conv = X.pfx_resolve_context("wildchat_prefix_real545")
    assert conv.system is None and len(conv.prefix_turns) == 2
    icl = X.pfx_resolve_context("icl_prefix_sycophancy")
    assert icl.user_wrap is not None and "{q}" in icl.user_wrap


def test_pfx_sample_derivation_deterministic_and_src_qidx(tmp_path):
    import random

    import issue1768_capture as cap

    n_train, n_val, n_test = 10, 3, 4
    shas = [f"s{i}" for i in range(n_train + n_val + n_test)]
    _write_sample(tmp_path, n_train, n_val, n_test, shas)
    cfg = cap.Cfg(out_root=tmp_path, phases=())
    sample = cap._build_pfx_sample(cfg)
    assert sample["n_train"] == min(X.PFX_N_TRAIN, n_train) == 10
    assert len(sample["rows"]) == 10 + n_val + n_test
    want_idxs = random.Random(X.SAMPLE_SEED).sample(range(n_train), 10)
    assert [r["src_qidx"] for r in sample["rows"][:10]] == want_idxs
    assert [r["src_qidx"] for r in sample["rows"][10:]] == list(range(10, 17))
    assert all(r["sha"] == shas[r["src_qidx"]] for r in sample["rows"])
    # resume: a second call reads the persisted file back unchanged
    again = cap._build_pfx_sample(cfg)
    assert again["train_subsample_sha256"] == sample["train_subsample_sha256"]
    # duplicate train shas violate the r4 postcondition -> fail loud
    dup = tmp_path / "dup"
    _write_sample(dup, n_train, n_val, n_test, ["dup"] * 2 + shas[2:])
    with pytest.raises(AssertionError, match="not unique"):
        cap._build_pfx_sample(cap.Cfg(out_root=dup, phases=()))


class _BudgetTok:
    """Whitespace tokenizer: apply_chat_template joins message contents."""

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        return " ".join(m["content"] for m in messages)

    def __call__(self, texts, add_special_tokens=False, **kw):
        if isinstance(texts, str):
            texts = [texts]
        return {"input_ids": [t.split() for t in texts]}


def test_pfx_budget_mk_raise_and_content_overflow():
    import issue1768_capture as cap

    from explore_persona_space.artifacts.context import Context

    rows = [{"prompt": "q " * 8}]  # 8 rendered tokens
    # prefix long enough that mk (2048 new) overflows 4096 but content (1024) fits
    n_pref = X.MAX_MODEL_LEN - X.MAX_NEW_MARKER + 100  # 2148 prefix words
    ctx = Context(
        context_id="t_conv",
        kind="prefix",
        family="test",
        prefix_turns=(
            {"role": "user", "content": "w " * n_pref},
            {"role": "assistant", "content": "a"},
        ),
    )
    budgets = cap._pfx_budget(_BudgetTok(), ctx, rows)
    assert budgets["content"]["max_model_len"] == X.MAX_MODEL_LEN
    assert not budgets["content"]["raised"]
    assert budgets["mk"]["max_model_len"] == X.PFX_MAX_MODEL_LEN_RAISED
    assert budgets["mk"]["raised"]
    # content overflow is a HARD failure (plan §7 — never a silent raise)
    big = Context(
        context_id="t_big",
        kind="prefix",
        family="test",
        prefix_turns=(
            {"role": "user", "content": "w " * (X.MAX_MODEL_LEN + 10)},
            {"role": "assistant", "content": "a"},
        ),
    )
    with pytest.raises(AssertionError, match="content-decode budget overflow"):
        cap._pfx_budget(_BudgetTok(), big, rows)


def test_assert_mix_row_matches_context():
    import issue1768_capture as cap

    from explore_persona_space.artifacts.context import Context

    pers = Context(context_id="p", kind="persona", family="t", system="You are an engineer.")
    ok = [
        {"role": "system", "content": "You are an engineer."},
        {"role": "user", "content": "q1"},
    ]
    cap._assert_mix_row_matches_context(pers, ok, "pers/test")
    with pytest.raises(AssertionError, match="system-prompt drift"):
        cap._assert_mix_row_matches_context(
            pers, [{"role": "system", "content": "OTHER"}, {"role": "user", "content": "q"}], "t"
        )
    conv = Context(
        context_id="c",
        kind="prefix",
        family="t",
        prefix_turns=({"role": "user", "content": "hi"}, {"role": "assistant", "content": "yo"}),
    )
    cap._assert_mix_row_matches_context(
        conv,
        [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "yo"},
            {"role": "user", "content": "q"},
        ],
        "conv/test",
    )
    with pytest.raises(AssertionError, match="prefix-turn drift"):
        cap._assert_mix_row_matches_context(
            conv,
            [
                {"role": "user", "content": "DRIFTED"},
                {"role": "assistant", "content": "yo"},
                {"role": "user", "content": "q"},
            ],
            "t",
        )
    icl = Context(context_id="i", kind="prefix", family="t", user_wrap="EX1... Now: {q}")
    cap._assert_mix_row_matches_context(
        icl, [{"role": "user", "content": "EX1... Now: real q"}], "icl/test"
    )
    with pytest.raises(AssertionError, match="user_wrap drift"):
        cap._assert_mix_row_matches_context(icl, [{"role": "user", "content": "bare q"}], "t")


def test_pfx_pending_units_smoke_coverage_and_resume(tmp_path):
    import issue1768_capture as cap

    cfg = cap.Cfg(out_root=tmp_path, phases=())
    assert len(cap._pending_units(cfg, "pfx2")) == 23  # 5 bases + 12 own + 6 ctrl
    assert len(cap._pending_units(cfg, "pfx3")) == 18
    # resume: a completed unit drops out of the queue
    done = tmp_path / "on_target" / "corpus_capture" / "base_content@pers"
    done.mkdir(parents=True)
    (done / "pooled.pt").write_bytes(b"x")
    assert len(cap._pending_units(cfg, "pfx2")) == 22
    # smoke coverage set: pilot own + plain-text-boundary (icl) + mk decode
    smoke = cap.Cfg(out_root=tmp_path / "s", phases=(), smoke=True)
    units = cap._pending_units(smoke, "pfx2")
    assert f"{X.PILOT_ARM}@own" in units
    assert "base_content@icl_syc" in units and "base_mk@pers" in units
    assert cap._pending_units(smoke, "pfx3") == [f"{X.PILOT_ARM}@own"]
    # merge predicate strips the @cond tag (pfx unit args)
    assert cap._needs_merge(cfg, "pfx2:syc-pers-con-lr1e5-s42@own")
    assert not cap._needs_merge(cfg, "pfx2:base_mk@conv")


def test_pfx_regime_mismatch_refuses(tmp_path):
    import issue1768_capture as cap

    unit_dir = tmp_path / "u"
    regime = {"unit_id": "a@own", "prefix_sha256": "abc", "layers": [14]}
    cap._pfx_check_regime(unit_dir, regime)
    cap._pfx_check_regime(unit_dir, regime)  # exact match re-passes
    with pytest.raises(RuntimeError, match="DIFFERENT regime"):
        cap._pfx_check_regime(unit_dir, {**regime, "prefix_sha256": "OTHER"})


def test_pfx_pilot_gate_ratios():
    import issue1768_capture as cap

    cap._pfx_pilot_gate(1.9, smoke=False)  # under 2x: pass
    cap._pfx_pilot_gate(9.0, smoke=True)  # smoke: informational only (#1345)
    with pytest.raises(RuntimeError, match="re-size"):
        cap._pfx_pilot_gate(2.5, smoke=False)
    with pytest.raises(RuntimeError, match="re-plan"):
        cap._pfx_pilot_gate(4.5, smoke=False)


def _write_pfx_sample(out_root, n_train, n_val, n_test, shas, src_qidx=None):
    rows = [
        {
            "prompt": f"p{i}",
            "corpus": "lmsys" if i % 2 == 0 else "wildchat",
            "sha": s,
            "src_qidx": (src_qidx[i] if src_qidx else i),
        }
        for i, s in enumerate(shas)
    ]
    p = out_root / "on_target" / "inputs"
    p.mkdir(parents=True, exist_ok=True)
    (p / "corpus_sample_pfx.json").write_text(
        json.dumps(
            {
                "rows": rows,
                "n_train": n_train,
                "n_val": n_val,
                "n_test": n_test,
                "train_subsample_sha256": "t",
            }
        )
    )
    return rows


def test_load_pfx_cell_and_bare_n_join(tmp_path):
    import issue1768_fit as fit

    d, layer = 4, 0
    n = 30
    arm = "syc-pers-con-lr1e5-s42"
    shas = [f"s{i}" for i in range(n)]
    # pfx sample: src_qidx maps pfx row i -> round-1 row i + 100
    src = [i + 100 for i in range(n)]
    _write_pfx_sample(tmp_path, 20, 4, 6, shas, src_qidx=src)
    root = tmp_path / "on_target"

    def mat(keep, off):
        m = np.zeros((len(keep), d))
        m[:, 0] = np.asarray(keep) + off
        return m

    b_keep = [i for i in range(n) if i != 3]
    _mk_store(
        root / "corpus_capture" / "base_content@pers" / "pooled.pt",
        {"context": {layer: mat(b_keep, 0)}, "response": {layer: mat(b_keep, 100)}},
        [shas[i] for i in b_keep],
        b_keep,
    )
    p_keep = [i for i in reversed(range(n)) if i != 7]  # reversed: join must re-align
    _mk_store(
        root / "corpus_capture" / f"{arm}@own" / "pooled.pt",
        {"context": {layer: mat(p_keep, 0)}, "response": {layer: mat(p_keep, 200)}},
        [shas[i] for i in p_keep],
        p_keep,
    )
    _mk_store(
        root / "corpus_capture_tf" / f"{arm}@own" / "pooled_tf.pt",
        {"response": {layer: mat(list(range(n)), 300)}},
        shas,
        list(range(n)),
    )
    cell = fit.load_pfx_cell(arm, "own", layer, tmp_path)
    kept = [i for i in range(n) if i not in (3, 7)]
    assert list(cell["qidx"]) == kept
    assert list(cell["src_qidx"]) == [i + 100 for i in kept]
    np.testing.assert_allclose(cell["Cplus"][:, 0], kept)
    np.testing.assert_allclose(cell["Vplus_tf"][:, 0], np.asarray(kept) + 300)
    assert (cell["split"] == "train").sum() == 18
    # bare_n: round-1 stores keyed by ROUND-1 qidx (i+100), remapped to pfx qidx
    r1_q = [i + 100 for i in range(n)]
    for tree, unit, fname, off, spans in (
        ("corpus_capture", "base_content", "pooled.pt", 0, ("context", "response")),
        ("corpus_capture", arm, "pooled.pt", 10, ("context", "response")),
        ("corpus_capture_tf", arm, "pooled_tf.pt", 20, ("response",)),
    ):
        m = np.zeros((n, d))
        m[:, 0] = np.arange(n)
        _mk_store(
            tmp_path / tree / unit / fname,
            {s: {layer: m + off} for s in spans},
            shas,
            r1_q,
        )
    bare = fit.load_bare_n_cell(arm, layer, tmp_path)
    assert list(bare["qidx"]) == list(range(n))  # remapped to pfx indices
    assert list(bare["src_qidx"]) == r1_q
    np.testing.assert_allclose(bare["C0"][:, 0], np.arange(n))


def test_fit_map_allow_underdetermined_flag():
    import issue1768_fit as fit

    rng = np.random.default_rng(1)
    n, d = 16, 24  # n_train < d: refused by default, allowed with the pfx flag
    Xd = rng.standard_normal((n, d))
    Yd = Xd @ rng.standard_normal((d, d)) * 0.1
    tr, val, te = np.arange(8), np.arange(8, 12), np.arange(12, 16)
    with pytest.raises(AssertionError, match="under-determined"):
        fit._fit_map(Xd, Yd, tr, val, te, fit._device())
    pred, meta, _pay = fit._fit_map(Xd, Yd, tr, val, te, fit._device(), allow_underdetermined=True)
    assert pred.shape == (4, d) and "selected_lambda" in meta


def test_paired_d_contrast_math_and_lattice():
    import issue1768_fit as fit

    rng = np.random.default_rng(0)
    n, b2 = 40, 8
    # bare side: noise-scale deltas ~ floor; own side: +5 shift => amplified
    floor = np.abs(rng.standard_normal((b2, n))).astype(np.float32) * 0.1
    bare = {
        "delta_rows": np.abs(rng.standard_normal(n)).astype(np.float32) * 0.1,
        "floor_rows": floor,
        "test_src_qidx": np.arange(100, 100 + n),
    }
    own = {
        "delta_rows": bare["delta_rows"] + 5.0,
        "floor_rows": floor.copy(),
        # SHUFFLED src order + a partial overlap: pairing is by src_qidx
        "test_src_qidx": np.asarray(list(reversed(range(105, 105 + n)))),
    }
    out = fit._paired_d_contrast(own, bare, seed=7)
    assert out["n_shared_rows"] == 35
    assert out["delta_d"] == pytest.approx(5.0, abs=0.5)
    assert out["verdict"] == "On-target-amplified" and out["delta_d_ci95"][0] > 0
    null = fit._paired_d_contrast(bare, bare, seed=7)
    assert null["delta_d"] == pytest.approx(0.0, abs=1e-9)
    assert null["verdict"] == "Indistinguishable"
    assert fit.contrast_verdict([-2.0, -0.5]) == "On-target-attenuated"
    # control reads thread control-specific vocabulary (never on/off-target)
    assert (
        fit.contrast_verdict(
            [-2.0, -0.5], positive="Control-above-own", negative="Control-below-own"
        )
        == "Control-below-own"
    )
    ctrl = fit._paired_d_contrast(
        own, bare, seed=7, positive="Control-above-own", negative="Control-below-own"
    )
    assert ctrl["verdict"] == "Control-above-own"


def test_pfx_fit_core_persists_percell_and_state(tmp_path):
    """The real `_pfx_fit_core` body on a tiny synthetic cell: fits JSON +
    percell rows + fit_state npz land atomically; resume returns the JSON."""
    import issue1768_fit as fit

    rng = np.random.default_rng(0)
    n, d = 60, 5
    C0 = rng.standard_normal((n, d))
    W = rng.standard_normal((d, d))
    V0 = C0 @ W + 0.01 * rng.standard_normal((n, d))
    Cp = C0 + 0.01 * rng.standard_normal((n, d))
    Vp = Cp @ (W + 3.0) + 0.01 * rng.standard_normal((n, d))
    cell = {
        "C0": C0,
        "V0": V0,
        "Cplus": Cp,
        "Vplus": Vp,
        "Vplus_tf": Vp,
        "split": np.array(["train"] * 40 + ["val"] * 8 + ["test"] * 12),
        "corpus": np.array(["lmsys", "wildchat"] * 30),
        "sha": [f"s{i}" for i in range(n)],
        "qidx": np.arange(n),
        "src_qidx": np.arange(n) + 100,
    }
    arm = "syc-pers-con-lr1e5-s42"
    res = fit._pfx_fit_core(
        tmp_path, tmp_path / "res", arm, 0, "own", cell, smoke=True, run_transfer_fold=True
    )
    assert res["map_change"]["verdict"] == "Changed"
    assert res["underdetermined_n_lt_d"] is False
    assert "transfer_fold" in res
    fits_json = tmp_path / "res" / "on_target" / "fits" / f"{arm}_L0_own.json"
    percell = tmp_path / "res" / "on_target" / "percell" / f"{arm}_L0_own.json"
    state = tmp_path / "on_target" / "fit_state" / f"{arm}_L0_own.npz"
    assert fits_json.exists() and percell.exists() and state.exists()
    rows = json.loads(percell.read_text())["rows"]
    assert len(rows) == 12 and all(r["src_qidx"] == r["qidx"] + 100 for r in rows)
    with np.load(state) as z:
        assert z["delta_rows"].shape == (12,)
        assert z["floor_rows"].shape[1] == 12
        assert list(z["test_src_qidx"]) == [r["src_qidx"] for r in rows]
    # resume path returns the persisted JSON without refitting
    again = fit._pfx_fit_core(
        tmp_path, tmp_path / "res", arm, 0, "own", cell, smoke=True, run_transfer_fold=True
    )
    assert again["map_change"]["D"] == res["map_change"]["D"]


def test_resolve_fit_hf_prefix_smoke_suffix():
    import issue1768_fit as fit

    assert fit.resolve_fit_hf_prefix(False, None) == X.HF_PREFIX
    assert fit.resolve_fit_hf_prefix(True, None) == f"{X.HF_PREFIX}_smoke"
    assert fit.resolve_fit_hf_prefix(True, "custom") == "custom_smoke"
    assert fit.resolve_fit_hf_prefix(True, "already_smoke") == "already_smoke"


def test_pfx_join_coverage_floor_and_contrast_no_overlap(tmp_path):
    """Degenerate-gate probes: the pfx join's 0.9 coverage floor and the
    contrast's empty shared-row assert fire loud (designed handling)."""
    import issue1768_fit as fit

    d, layer, n = 3, 0, 20
    shas = [f"s{i}" for i in range(n)]
    _write_pfx_sample(tmp_path, 12, 4, 4, shas)
    root = tmp_path / "on_target"
    m = np.zeros((n, d))
    arm = "syc-pers-con-lr1e5-s42"
    _mk_store(
        root / "corpus_capture" / "base_content@pers" / "pooled.pt",
        {"context": {layer: m}, "response": {layer: m}},
        shas,
        list(range(n)),
    )
    few = list(range(10))  # plus keeps only 10/20 rows -> floor trips
    _mk_store(
        root / "corpus_capture" / f"{arm}@own" / "pooled.pt",
        {"context": {layer: m[:10]}, "response": {layer: m[:10]}},
        [shas[i] for i in few],
        few,
    )
    _mk_store(
        root / "corpus_capture_tf" / f"{arm}@own" / "pooled_tf.pt",
        {"response": {layer: m}},
        shas,
        list(range(n)),
    )
    with pytest.raises(AssertionError):
        fit.load_pfx_cell(arm, "own", layer, tmp_path)
    a = {
        "delta_rows": np.ones(4, np.float32),
        "floor_rows": np.ones((2, 4), np.float32),
        "test_src_qidx": np.arange(4),
    }
    b = {**a, "test_src_qidx": np.arange(100, 104)}
    with pytest.raises(AssertionError, match="no shared test rows"):
        fit._paired_d_contrast(a, b, seed=0)


def test_pfx7_reads_what_pfx5_writes_including_ctrl(tmp_path):
    """r3-v2 Critical-1 pin: producer/consumer path equality for EVERY pfx
    condition — `_pfx_fit_core` writes and the REAL `phase_pfx7` (smoke=False,
    so the ctrl branch executes) reads through the ONE `pfx_cell_paths`
    helper; the ctrl fits JSON `_ctrl`-vs-`_control` drift class cannot recur."""
    import issue1768_fit as fit

    rng = np.random.default_rng(0)
    n, d, layer = 60, 5, 0
    C0 = rng.standard_normal((n, d))
    W = rng.standard_normal((d, d))
    V0 = C0 @ W + 0.01 * rng.standard_normal((n, d))
    Cp = C0 + 0.01 * rng.standard_normal((n, d))
    Vp = Cp @ (W + 3.0) + 0.01 * rng.standard_normal((n, d))
    cell = {
        "C0": C0,
        "V0": V0,
        "Cplus": Cp,
        "Vplus": Vp,
        "Vplus_tf": Vp,
        "split": np.array(["train"] * 40 + ["val"] * 8 + ["test"] * 12),
        "corpus": np.array(["lmsys", "wildchat"] * 30),
        "sha": [f"s{i}" for i in range(n)],
        "qidx": np.arange(n),
        "src_qidx": np.arange(n) + 100,
    }
    arm = "syc-pers-con-lr1e5-s42"  # a PFX_CONTROL_ARMS member — ctrl branch fires
    res_dir = tmp_path / "res"
    for cond in ("own", "ctrl", "bare_n"):
        fit._pfx_fit_core(
            tmp_path, res_dir, arm, layer, cond, cell, smoke=True, run_transfer_fold=False
        )
        fits_p, npz_p, percell_p = fit.pfx_cell_paths(tmp_path, res_dir, arm, layer, cond)
        assert fits_p.exists() and npz_p.exists() and percell_p.exists(), cond
    # percell files carry the plan §4.5 suffixes {own, control, bare_n}
    percell = res_dir / "on_target" / "percell"
    assert (percell / f"{arm}_L0_own.json").exists()
    assert (percell / f"{arm}_L0_control.json").exists()
    assert (percell / f"{arm}_L0_bare_n.json").exists()
    # the REAL consumer, ctrl branch included (smoke=False)
    fit.phase_pfx7(tmp_path, res_dir, (layer,), smoke=False, arms_filter=(arm,))
    summary = json.loads((res_dir / "on_target" / "map_change_on_target.json").read_text())
    row = summary["contrast"][f"{arm}_L0"]
    assert "D_control" in row and "control_contrast" in row
    assert row["control_contrast"]["verdict"] in {
        "Control-above-own",
        "Control-below-own",
        "Indistinguishable",
    }
    assert summary["success_criteria"]["n_control_arms"] >= 0
    # missing-cell path fails loud NAMING the re-run phase, not FileNotFoundError
    with pytest.raises(RuntimeError, match="re-run pfx5"):
        fit._pfx_cell_inputs(tmp_path, res_dir, arm, 99, "own")


def test_pfx4_resume_skip_and_recount(tmp_path, monkeypatch):
    """r3-v2 Minor: pfx4 consults upload_done.json at entry — matching
    expected-store count skips the re-upload; a changed count re-uploads."""
    import issue1768_capture as cap

    from explore_persona_space.orchestrate import hub as hub_mod

    calls: list[str] = []
    monkeypatch.setattr(
        cap, "_upload_tree", lambda cfg, name: (calls.append(name), f"{cfg.hf_prefix}/{name}")[1]
    )
    monkeypatch.setattr(hub_mod, "verify_repo_paths_uploaded", lambda *a, **k: [])
    cfg = cap.Cfg(out_root=tmp_path, phases=(), upload=True)
    cap._atomic_json(tmp_path / "on_target" / "upload_done.json", {"n_verified": 0})
    cap.phase_pfx4(cfg)
    assert calls == []  # matching count (0 stores) -> resume skip
    store = tmp_path / "on_target" / "corpus_capture" / "base_content@pers"
    store.mkdir(parents=True)
    (store / "pooled.pt").write_bytes(b"x")
    cap.phase_pfx4(cfg)  # count changed 0 -> 1: re-upload fires
    assert calls == list(cap.PFX_UPLOAD_TREES)
    assert json.loads((tmp_path / "on_target" / "upload_done.json").read_text())["n_verified"] == 1


# ── round 4: prefix-richness dose ladder (plan v10) ──────────────────────────


def _write_r4_ladder(out_root, realized=None, turns_by_cond=None):
    """Minimal valid prefix_ladder.json fixture (2-turn user/assistant rungs)."""
    realized = realized or {"r_short": 11, "r_mid": 94, "r_long": 800}
    rungs = {}
    for i, cond in enumerate(X.R4_CONDS):
        turns = (turns_by_cond or {}).get(cond) or [
            {"role": "user", "content": f"rung {cond} question {i}?"},
            {"role": "assistant", "content": f"rung {cond} answer {i}."},
        ]
        rungs[cond] = {
            "context_id": X.R4_CONTEXT_ID_BY_COND[cond],
            "prefix_turns": turns,
            "conversation_hash": f"hash_{cond}",
            "dataset_index": 10 + i,
            "turns_sha256": f"tsha_{cond}",
            "recipe_sha256": f"rsha_{cond}",
            "realized_tokens": realized[cond],
            "target_tokens": float(realized[cond]),
            "band": [realized[cond] * 0.5, realized[cond] * 2.0],
            "log_dist_to_target": 0.0,
            "n_band_candidates": 3,
        }
    p = Path(out_root) / "on_target_r4" / "inputs"
    p.mkdir(parents=True, exist_ok=True)
    (p / "prefix_ladder.json").write_text(json.dumps({"rungs": rungs}))
    return rungs


def test_r4_registry_units_and_ladder_registrar(tmp_path):
    from explore_persona_space.artifacts.context import CONTEXTS

    assert len(X.R4_ARMS) == 4 and X.R4_COMPARATOR_ARM in X.R4_ARMS
    assert len(X.R4_PERSONA_ARMS) == 3 and X.R4_COMPARATOR_ARM not in X.R4_PERSONA_ARMS
    assert X.r4_trained_unit("syc-pers-con-lr1e5-s42", "r_long") == "syc-pers-con-lr1e5-s42@r_long"
    assert X.r4_base_unit("r_mid") == "base_content@r_mid"
    assert X.r4_unit_context_id("base_content@r_short") == "ladder_prefix_short"
    assert X.r4_unit_context_id("cas-pers-con-lr1e5-s42@r_long") == "ladder_prefix_long"
    with pytest.raises(AssertionError):
        X.r4_trained_unit("syc-pers-con-lr1e5-s42", "own")  # rung labels only
    with pytest.raises(AssertionError):
        X.r4_unit_context_id("base_content@pers")  # r3 tags are NOT rung units
    _write_r4_ladder(tmp_path)
    try:
        X.register_r4_ladder_contexts(tmp_path)
        X.register_r4_ladder_contexts(tmp_path)  # idempotent
        ctx = CONTEXTS["ladder_prefix_long"]
        assert ctx.kind == "prefix" and ctx.family == X.R4_LADDER_FAMILY
        assert tuple(t["role"] for t in ctx.prefix_turns) == ("user", "assistant")
        # foreign-binding refusal (the register_fu3_contexts pattern)
        from explore_persona_space.artifacts.context import Context

        CONTEXTS["ladder_prefix_short"] = Context(
            context_id="ladder_prefix_short", kind="prefix", family="foreign", prefix_turns=()
        )
        with pytest.raises(ValueError, match="refusing to shadow"):
            X.register_r4_ladder_contexts(tmp_path)
    finally:
        for cid in X.R4_CONTEXT_ID_BY_COND.values():
            CONTEXTS.pop(cid, None)
    # load_r4_ladder shape gate: non-(user, assistant) roles fail loud
    bad = {"r_short": [{"role": "assistant", "content": "a"}, {"role": "user", "content": "b"}]}
    _write_r4_ladder(tmp_path, turns_by_cond=bad)
    with pytest.raises(AssertionError):
        X.load_r4_ladder(tmp_path)


def test_lad_band_specs_screens_and_selection():
    import issue1768_capture as cap

    specs = cap.lad_band_specs(11, 800)
    assert specs["r_short"]["lo"] == pytest.approx(5.5)
    assert specs["r_short"]["hi"] == pytest.approx(22.0)
    assert specs["r_mid"]["target"] == pytest.approx((11 * 800) ** 0.5)
    assert specs["r_long"]["lo"] == pytest.approx(600.0)
    assert specs["r_long"]["hi"] == pytest.approx(1000.0)
    # bands pairwise disjoint at the measured anchors (plan §11)
    assert cap.lad_bands_for(11, specs) == ["r_short"]
    assert cap.lad_bands_for(94, specs) == ["r_mid"]
    assert cap.lad_bands_for(800, specs) == ["r_long"]
    assert cap.lad_bands_for(300, specs) == []
    # corpus screens: full language NAME (the #1092 lesson), fail-safe bools
    base = {
        "language": "English",
        "toxic": False,
        "redacted": False,
        "conversation": [
            {"role": "user", "content": "hi there"},
            {"role": "assistant", "content": "hello!"},
        ],
        "conversation_hash": "h",
    }
    assert cap.lad_screen_reject(base) is None
    assert cap.lad_screen_reject({**base, "language": "en"}) == "language"
    assert cap.lad_screen_reject({**base, "toxic": True}) == "toxic"
    assert cap.lad_screen_reject({**base, "toxic": None}) == "toxic"
    assert cap.lad_screen_reject({**base, "redacted": True}) == "redacted"
    assert cap.lad_screen_reject({**base, "conversation": base["conversation"][:1]}) == (
        "too_few_turns"
    )
    swapped = [base["conversation"][1], base["conversation"][0]]
    assert cap.lad_screen_reject({**base, "conversation": swapped}) == "bad_roles"
    empty = [{"role": "user", "content": "  "}, {"role": "assistant", "content": "x"}]
    assert cap.lad_screen_reject({**base, "conversation": empty}) == "empty_content"
    # r2 content-language screen (exclusion 6, concern r-long-rung-content-
    # language): English METADATA + Cyrillic CONTENT rejects (the r1 r_long
    # idx-9098 class — fails pre-fix: lad_screen_reject returned None)
    cyr = "".join(chr(c) for c in range(0x0430, 0x0450))  # Cyrillic a..ya
    assert cap.lad_content_language_ok("Hello there friend", "I am fine, thanks!") is True
    assert cap.lad_content_language_ok(cyr * 4, f"reply {cyr}") is False
    assert cap.lad_content_language_ok("12 34", "!! ??") is False  # alpha==0 fail-safe
    assert cap.lad_content_language_ok("a" * 95, cyr[:5]) is True  # ratio == 0.95 boundary
    assert cap.lad_content_language_ok("a" * 94, cyr[:6]) is False  # 0.94 < threshold
    cyr_conv = [
        {"role": "user", "content": f"please translate {cyr * 8}"},
        {"role": "assistant", "content": cyr * 8},
    ]
    assert cap.lad_screen_reject({**base, "conversation": cyr_conv}) == "content_language"
    # cid-stripped content sha BINDS where the recipe sha was vacuous (Minor):
    # identical content under different cids collides on the CONTENT sha
    import types as _types

    turns_ab = (
        {"role": "user", "content": "same question"},
        {"role": "assistant", "content": "same answer"},
    )
    shim_a = _types.SimpleNamespace(
        context_id="cid_a", system=None, prefix_turns=turns_ab, user_wrap=None
    )
    shim_b = _types.SimpleNamespace(
        context_id="cid_b", system=None, prefix_turns=turns_ab, user_wrap=None
    )
    assert cap._pfx_prefix_sha(shim_a) != cap._pfx_prefix_sha(shim_b)  # the vacuity
    assert cap._lad_content_sha(shim_a) == cap._lad_content_sha(shim_b)  # the bind
    # exclusion screens 1-3 (degenerate probes)
    excl = {
        "conv_turns": (("user", "trained q"), ("assistant", "trained a")),
        "persona_system": "PERSONA-SYS",
        "icl_demo_texts": ["ICL-DEMO"],
        "trained_shas": {},
    }
    sha_set = {X.prompt_sha("known query")}
    assert cap.lad_exclusion_reject("trained q", "trained a", "trained q", excl, sha_set) == (
        "trained_prefix"
    )
    assert cap.lad_exclusion_reject("x PERSONA-SYS y", "a", "x", excl, sha_set) == (
        "trained_context_containment"
    )
    assert cap.lad_exclusion_reject("a", "z ICL-DEMO", "a", excl, sha_set) == (
        "trained_context_containment"
    )
    assert cap.lad_exclusion_reject("known query", "a", "known query", excl, sha_set) == (
        "query_sha_overlap"
    )
    assert cap.lad_exclusion_reject("fresh", "novel", "fresh", excl, sha_set) is None
    needle = "known query about the tides"  # >= LAD_BELT_MIN_QUERY_CHARS
    assert len(needle) >= cap.LAD_BELT_MIN_QUERY_CHARS
    assert cap.lad_substring_belt_hit(f"xx {needle} yy", "a", [needle]) is True
    assert cap.lad_substring_belt_hit("fresh", "novel", [needle]) is False
    # the MEASURED #1776 short-query collision class: sub-floor needles ('hi',
    # 'ok') never belt-reject — they stay covered by the exact-sha screen
    assert cap.lad_substring_belt_hit("hi there, long text", "a", ["hi"]) is False
    assert cap.lad_belt_needles(["hi", needle]) == [needle]

    # deterministic selection: (dist, index) order; hash distinctness; belt
    def cand(idx, t, h, text="fresh"):
        return {
            "index": idx,
            "T": t,
            "dist": abs(cap._log(t) - cap._log(11)),
            "conversation_hash": h,
            "turns": [
                {"role": "user", "content": text},
                {"role": "assistant", "content": "novel"},
            ],
        }

    pools = {
        "r_short": [cand(5, 11, "A"), cand(2, 11, "B")],  # tie on dist -> lowest index
        "r_mid": [cand(9, 11, "B"), cand(30, 11, "C")],  # B collides with r_short pick
        "r_long": [cand(4, 11, "D", text=f"belt {needle} x"), cand(6, 11, "E")],
    }
    counters: dict[str, int] = {}
    selected, shortage = cap._lad_select_rungs(pools, [needle], counters)
    assert shortage == []
    assert selected["r_short"]["conversation_hash"] == "B"  # tie-break: index 2 < 5
    assert selected["r_mid"]["conversation_hash"] == "C"  # B already used (exclusion 4)
    assert selected["r_long"]["conversation_hash"] == "E"  # D rejected by the belt
    assert counters["belt_query_text_substring"] == 1
    assert counters["cross_rung_hash_collision"] == 1
    _sel, short2 = cap._lad_select_rungs({**pools, "r_mid": [cand(9, 11, "B")]}, [], {})
    assert short2 == ["r_mid"]  # kill criterion (b) surface: shortage reported


def test_lad_recheck_exclusions_tampered_ladder(tmp_path, monkeypatch):
    """Kill criterion (d) probe: the REAL `_lad_recheck_exclusions` body with
    only the network-boundary input fetchers faked (signature-conformant);
    a persona-system-contaminated rung fails loud, a clean ladder passes."""
    import issue1768_capture as cap

    class _Tok:
        def __call__(self, text, add_special_tokens=False):
            return {"input_ids": [0] * len(text.split())}

    excl = {
        "trained_shas": {"pers": "TS1", "conv": "TS2", "icl": "TS3"},
        "trained_content_shas": {"pers": "CS1", "conv": "CS2", "icl": "CS3"},
        "conv_turns": (("user", "trained q"), ("assistant", "trained a")),
        "persona_system": "PERSONA-SYS",
        "icl_demo_texts": ["ICL-DEMO"],
    }
    monkeypatch.setattr(cap, "_lad_trained_exclusion_material", lambda: excl)
    monkeypatch.setattr(cap, "_lad_full_grain_samples", lambda cfg: (set(), ["ZZZ-query"]))
    cfg = cap.Cfg(out_root=tmp_path, phases=())
    turns = {
        c: [
            {"role": "user", "content": f"three word {c}"},
            {"role": "assistant", "content": f"reply for {c}"},
        ]
        for c in X.R4_CONDS
    }
    realized = {c: 6 for c in X.R4_CONDS}  # 3 + 3 whitespace tokens per rung
    _write_r4_ladder(tmp_path, realized=realized, turns_by_cond=turns)
    ladder = json.loads((tmp_path / "on_target_r4" / "inputs" / "prefix_ladder.json").read_text())
    for c in X.R4_CONDS:  # align the fixture's derived shas with the recheck
        ladder["rungs"][c]["turns_sha256"] = cap.lad_turns_sha(ladder["rungs"][c]["prefix_turns"])
    out = cap._lad_recheck_exclusions(cfg, _Tok(), ladder)
    assert set(out) == set(X.R4_CONDS)
    tampered = json.loads(json.dumps(ladder))
    tampered["rungs"]["r_mid"]["prefix_turns"][0]["content"] = "has PERSONA-SYS inside x"
    tampered["rungs"]["r_mid"]["realized_tokens"] = 7
    tampered["rungs"]["r_mid"]["band"] = [1, 20]
    tampered["rungs"]["r_mid"]["turns_sha256"] = cap.lad_turns_sha(
        tampered["rungs"]["r_mid"]["prefix_turns"]
    )
    with pytest.raises(AssertionError, match="builder exclusion violated"):
        cap._lad_recheck_exclusions(cfg, _Tok(), tampered)
    # r2 content-language screen at the kill-d re-assert: Cyrillic content
    # under a structurally-valid rung fails loud (the r1 r_long idx-9098 class)
    cyr = "".join(chr(c) for c in range(0x0430, 0x0450))
    tampered2 = json.loads(json.dumps(ladder))
    tampered2["rungs"]["r_long"]["prefix_turns"][1]["content"] = f"otvet {cyr} {cyr}"
    with pytest.raises(AssertionError, match="content-language screen"):
        cap._lad_recheck_exclusions(cfg, _Tok(), tampered2)


def test_lad_build_widen_census_parity_and_stale_screen_rebuild(tmp_path, monkeypatch):
    """r4-r2 phase probes over the REAL `phase_lad_build` body (only the
    stream scan / registry / Hub boundaries faked, signature-conformant):
    (a) a shortage->widen->success SINGLE invocation keeps the belt census
    keys in the written ladder (fails pre-fix: the widened rescan's counters
    rebind dropped them); (b) the manifest carries the content-language
    screen config; (c) a current-screen dest REPUBLISHES without rescanning;
    (d) a stale-screen dest REBUILDS (regime-keyed resume); (e) a non-Latin
    TRAINED prefix trips the construction-asymmetry STOP."""
    import issue1768_capture as cap
    import transformers

    class _Tok:
        def __call__(self, text, add_special_tokens=False):
            return {"input_ids": [0] * len(text.split())}

    class _AT:
        @staticmethod
        def from_pretrained(*a, **k):
            return _Tok()

    monkeypatch.setattr(transformers, "AutoTokenizer", _AT)
    needle = "known query about the tides"  # >= belt floor; 'hi' below it
    excl = {
        "trained_shas": {"pers": "TS1", "conv": "TS2", "icl": "TS3"},
        "trained_content_shas": {"pers": "CS1", "conv": "CS2", "icl": "CS3"},
        "conv_turns": (
            ("user", "a trained question of nine whitespace tokens here ok"),
            ("assistant", " ".join(["reply"] * 31)),
        ),
        "persona_system": "PERSONA SYS UNIT",  # t_pers=3; t_conv=40
        "icl_demo_texts": ["ICL-DEMO"],
    }

    def cand(idx, t, h):
        return {
            "index": idx,
            "T": t,
            "dist": 0.0,
            "conversation_hash": h,
            "turns": [
                {"role": "user", "content": "fresh latin question"},
                {"role": "assistant", "content": "novel latin reply"},
            ],
        }

    good = {"r_short": [cand(1, 3, "A")], "r_mid": [cand(2, 11, "B")], "r_long": [cand(3, 40, "C")]}
    calls = {"n": 0}

    def fake_scan(cfg, tok, specs, excl_, sha_set, scan_cap):
        calls["n"] += 1
        if calls["n"] == 1:  # first 50k scan: shortage everywhere
            return {c: [] for c in X.R4_CONDS}, {"language": 5}, scan_cap, "rev"
        # the widened rescan REBINDS counters from the cursor (production shape)
        return {c: list(v) for c, v in good.items()}, {"language": 9}, scan_cap, "rev"

    published: dict = {}
    monkeypatch.setattr(cap, "_lad_scan", fake_scan)
    monkeypatch.setattr(cap, "_lad_trained_exclusion_material", lambda: excl)
    monkeypatch.setattr(cap, "_lad_full_grain_samples", lambda cfg: (set(), ["hi", needle]))
    monkeypatch.setattr(cap, "_lad_build_publish", lambda cfg, ladder: published.update(ladder))
    cap.phase_lad_build(cap.Cfg(out_root=tmp_path, phases=()))
    assert calls["n"] == 2  # shortage -> pre-registered widening fired
    ladder = json.loads((tmp_path / "on_target_r4" / "inputs" / "prefix_ladder.json").read_text())
    assert ladder["counters"]["belt_needles_total"] == 2  # (a) fails pre-fix
    assert ladder["counters"]["belt_needles_below_floor_excluded"] == 1
    scr = ladder["exclusions"]["content_language_screen"]  # (b)
    assert scr["min_latin_ratio"] == cap.LAD_CONTENT_LATIN_MIN_RATIO
    assert any("content-language" in s for s in ladder["exclusions"]["screens"])
    assert ladder["scan"]["widened"] is True
    # (c) current-screen dest -> republish WITHOUT rescanning
    published.clear()
    cap.phase_lad_build(cap.Cfg(out_root=tmp_path, phases=()))
    assert calls["n"] == 2 and published["rungs"] == ladder["rungs"]
    # (d) stale-screen dest (predates the content-language screen) -> REBUILD
    stale_root = tmp_path / "stale"
    stale = json.loads(json.dumps(ladder))
    del stale["exclusions"]["content_language_screen"]
    cap._atomic_json(stale_root / "on_target_r4" / "inputs" / "prefix_ladder.json", stale)
    cap.phase_lad_build(cap.Cfg(out_root=stale_root, phases=()))
    assert calls["n"] == 3  # rescan happened (regime-keyed rebuild)
    rebuilt = json.loads(
        (stale_root / "on_target_r4" / "inputs" / "prefix_ladder.json").read_text()
    )
    assert "content_language_screen" in rebuilt["exclusions"]
    # (e) non-Latin TRAINED prefix -> construction-asymmetry STOP
    cyr = "".join(chr(c) for c in range(0x0430, 0x0450))
    excl_bad = {**excl, "conv_turns": (("user", cyr * 3), ("assistant", cyr))}
    monkeypatch.setattr(cap, "_lad_trained_exclusion_material", lambda: excl_bad)
    with pytest.raises(AssertionError, match="construction asymmetry"):
        cap.phase_lad_build(cap.Cfg(out_root=tmp_path / "b", phases=()))


def test_lad7_resume_guard_and_force(tmp_path, monkeypatch):
    """Concern `lad7-no-resume-guard`: dest-exists SKIPS (even production
    mode, BEFORE any r3 staging / ladder load); `--force` recomputes. Fails
    pre-fix: the no-force call reached the ladder load and raised."""
    import issue1768_fit as fit

    res = fit._lad_results(tmp_path / "res")
    res.mkdir(parents=True)
    (res / "map_change_ladder.json").write_text("{}")

    def boom(*a, **k):
        raise RuntimeError("recompute-reached")

    monkeypatch.setattr(fit, "_stage_r3_contrast_inputs", boom)
    monkeypatch.setattr(fit.X, "load_r4_ladder", boom)
    fit.phase_lad7(tmp_path, tmp_path / "res", (0,), False, ())  # guard skip: no raise
    with pytest.raises(RuntimeError, match="recompute-reached"):
        fit.phase_lad7(tmp_path, tmp_path / "res", (0,), True, (), force=True)


def test_lad_unit_sets_pending_and_smoke_coverage(tmp_path):
    import issue1768_capture as cap

    smoke_cfg = cap.Cfg(out_root=tmp_path, phases=(), smoke=True)
    assert cap._lad_unit_set(smoke_cfg) == [
        "base_content@r_long",
        "syc-pers-con-lr1e5-s42@r_long",
    ]
    prod_cfg = cap.Cfg(out_root=tmp_path, phases=())
    units = cap._lad_unit_set(prod_cfg)
    assert len(units) == 15  # 3 base@rung + 4 arms x 3 rungs (plan §4.4)
    assert units[:3] == ["base_content@r_short", "base_content@r_mid", "base_content@r_long"]
    assert cap._pending_units(prod_cfg, "lad2") == units
    done = tmp_path / "on_target_r4" / "corpus_capture" / "base_content@r_mid"
    done.mkdir(parents=True)
    (done / "pooled.pt").write_bytes(b"x")
    assert "base_content@r_mid" not in cap._pending_units(prod_cfg, "lad2")
    with pytest.raises(AssertionError, match="outside the r4 arm set"):
        cap._lad_arms(cap.Cfg(out_root=tmp_path, phases=(), arms=("mk-pers-con-lr5e6-s42",)))


def test_lad_cell_paths_rung_routing(tmp_path):
    import issue1768_fit as fit

    arm = "imp-pers-con-lr3e5-s42"
    fits, npz, percell = fit.pfx_cell_paths(tmp_path, tmp_path / "res", arm, 19, "r_short")
    assert fits == tmp_path / "res" / "on_target_r4" / "fits" / f"{arm}_L19_r_short.json"
    assert npz == tmp_path / "on_target_r4" / "fit_state" / f"{arm}_L19_r_short.npz"
    assert percell == tmp_path / "res" / "on_target_r4" / "percell" / f"{arm}_L19_r_short.json"
    # r3 conditions stay byte-identical (regression pin on the shared helper)
    fits3, npz3, percell3 = fit.pfx_cell_paths(tmp_path, tmp_path / "res", arm, 19, "ctrl")
    assert fits3 == tmp_path / "res" / "on_target" / "fits" / f"{arm}_L19_control.json"
    assert npz3 == tmp_path / "on_target" / "fit_state" / f"{arm}_L19_control.npz"
    assert percell3 == tmp_path / "res" / "on_target" / "percell" / f"{arm}_L19_control.json"
    with pytest.raises(KeyError):
        fit.pfx_cell_paths(tmp_path, tmp_path / "res", arm, 19, "r_bogus")


def test_load_lad_cell_no_tf_and_fit_core_skips_tf_maps(tmp_path):
    import issue1768_fit as fit

    d, layer, n = 5, 0, 60
    arm = "cas-pers-con-lr1e5-s42"
    shas = [f"s{i}" for i in range(n)]
    _write_pfx_sample(tmp_path, 40, 8, 12, shas, src_qidx=[i + 100 for i in range(n)])
    rng = np.random.default_rng(0)
    root = tmp_path / "on_target_r4"
    C0 = rng.standard_normal((n, d))
    W = rng.standard_normal((d, d))
    _mk_store(
        root / "corpus_capture" / "base_content@r_long" / "pooled.pt",
        {
            "context": {layer: C0},
            "response": {layer: C0 @ W + 0.01 * rng.standard_normal((n, d))},
            "prefix": {layer: np.ones((n, d))},
        },
        shas,
        list(range(n)),
    )
    _mk_store(
        root / "corpus_capture" / f"{arm}@r_long" / "pooled.pt",
        {
            "context": {layer: C0},
            "response": {layer: C0 @ (W + 3.0) + 0.01 * rng.standard_normal((n, d))},
            "prefix": {layer: np.ones((n, d)) * 2},
        },
        shas,
        list(range(n)),
    )
    cell = fit.load_lad_cell(arm, "r_long", layer, tmp_path)
    assert "Vplus_tf" not in cell  # Method delta (b): no TF tree on rungs
    res = fit._pfx_fit_core(
        tmp_path, tmp_path / "res", arm, layer, "r_long", cell, smoke=True, run_transfer_fold=True
    )
    assert "Mplus_tf" not in res["fits"] and "decomposition_tf" not in res
    assert "transfer_fold" in res  # plan §6: LMSYS<->WildChat fold on rung fits
    fits_p, npz_p, percell_p = fit.pfx_cell_paths(tmp_path, tmp_path / "res", arm, layer, "r_long")
    assert fits_p.exists() and npz_p.exists() and percell_p.exists()


def test_paired_m_contrast_join_and_lattices():
    import issue1768_fit as fit

    # duplicate SHAS with unique qidx: the (sha, qidx) join stays exact where
    # a sha-only dict join would silently collapse rows (the r4 advisory)
    n = 20
    a_rows = [
        {"sha": f"s{i % 12}", "qidx": i, "src_qidx": i + 100, "delta": 10.0 + (i % 3)}
        for i in range(n)
    ]
    b_rows = [{"sha": f"s{i % 12}", "qidx": i, "src_qidx": i + 100, "delta": 2.0} for i in range(n)]
    assert len({r["sha"] for r in a_rows}) < n  # the fixture really has dup shas
    out = fit._paired_m_contrast(a_rows, b_rows, seed=3, expect_pairs=n)
    assert out["n_pairs"] == n and out["join"] == "(sha, qidx) exact"
    assert out["diff"] == pytest.approx(9.0, abs=1.0) and out["diff_ci95"][0] > 0
    with pytest.raises(AssertionError, match="join drift"):
        fit._paired_m_contrast(a_rows[:-1], b_rows, seed=3, expect_pairs=n)
    # smoke demotes the exact-pair assert to a log line (#1345 rule)
    ok = fit._paired_m_contrast(a_rows[:-1], b_rows, seed=3, expect_pairs=n, smoke=True)
    assert ok["n_pairs"] == n - 1
    dup = [*a_rows, a_rows[0]]
    with pytest.raises(AssertionError, match="duplicate"):
        fit._paired_m_contrast(dup, b_rows, seed=3)
    # plan §3 lattices (DISJOINT + exhaustive)
    assert fit.lad_richness_verdict([-1.0, 0.5], [0.2, 1.0]) == "Richness-consistent"
    assert fit.lad_richness_verdict([-2.0, -0.5], [0.2, 1.0]) == "Identity-consistent"
    assert fit.lad_richness_verdict([-1.0, 0.5], [-0.2, 1.0]) == "Mixed"
    assert fit.lad_own_suppression_verdict([0.5, 2.0]) == "Own-suppressed"
    assert fit.lad_own_suppression_verdict([-2.0, -0.5]) == "Not-suppressed"
    assert fit.lad_own_suppression_verdict([-0.5, 0.5]) == "Indeterminate"


def test_lad7_production_mode_reads_what_lad5_writes(tmp_path, monkeypatch):
    """Production-mode probe for the smoke-fenced lad7 legs (the fenced-branch
    rule): the REAL `phase_lad7(smoke=False)` over cells the REAL
    `_pfx_fit_core` wrote for every rung + the r3-side {own, ctrl, bare_n}
    layout — ΔD Rung-* vocabulary, the 5 registered m-contrasts, richness +
    own-suppression verdicts, and the loud missing-cell error naming lad5."""
    import issue1768_fit as fit

    monkeypatch.setattr(X, "N_TEST", 12)  # production exact-join assert at test scale
    rng = np.random.default_rng(0)
    n, d, layer = 60, 5, 0
    C0 = rng.standard_normal((n, d))
    W = rng.standard_normal((d, d))
    arms = ("syc-pers-con-lr1e5-s42", X.R4_COMPARATOR_ARM)
    res_dir = tmp_path / "res"
    shas = [f"s{i % 50}" for i in range(n)]  # dup shas; qidx disambiguates
    for arm in arms:
        for cond, shift in (
            ("r_short", 0.5),
            ("r_mid", 1.5),
            ("r_long", 3.0),
            ("own", 1.0),
            ("ctrl", 2.5),
            ("bare_n", 0.0),
        ):
            Cp = C0 + 0.01 * rng.standard_normal((n, d))
            cell = {
                "C0": C0,
                "V0": C0 @ W + 0.01 * rng.standard_normal((n, d)),
                "Cplus": Cp,
                "Vplus": Cp @ (W + shift) + 0.01 * rng.standard_normal((n, d)),
                "split": np.array(["train"] * 40 + ["val"] * 8 + ["test"] * 12),
                "corpus": np.array(["lmsys", "wildchat"] * 30),
                "sha": shas,
                "qidx": np.arange(n),
                "src_qidx": np.arange(n) + 100,
            }
            if cond in ("own", "ctrl", "bare_n"):
                cell["Vplus_tf"] = cell["Vplus"]
            fit._pfx_fit_core(
                tmp_path, res_dir, arm, layer, cond, cell, smoke=True, run_transfer_fold=False
            )
    _write_r4_ladder(tmp_path)
    fit.phase_lad7(tmp_path, res_dir, (layer,), smoke=False, arms_filter=arms)
    summary = json.loads((res_dir / "on_target_r4" / "map_change_ladder.json").read_text())
    assert len(summary["cells"]) == 6  # 2 arms x 1 layer x 3 rungs
    row = summary["cells"][f"{arms[0]}_L0_r_long"]
    assert row["realized_tokens"] == 800
    assert row["contrast"]["verdict"] in {"Rung-amplified", "Rung-attenuated", "Indistinguishable"}
    mrow = summary["m_table"][f"{arms[0]}_L0"]
    assert set(mrow["contrasts"]) == {
        "long_minus_ctrl",
        "long_minus_short",
        "long_minus_own",
        "mid_minus_short",
        "long_minus_mid",
    }
    assert all(c["n_pairs"] == 12 for c in mrow["contrasts"].values())
    assert summary["richness_verdicts"][arms[0]] in {
        "Richness-consistent",
        "Identity-consistent",
        "Mixed",
    }
    assert summary["own_suppression"]["arm_id"] == X.R4_COMPARATOR_ARM
    assert summary["own_suppression"]["verdict"] in {
        "Own-suppressed",
        "Not-suppressed",
        "Indeterminate",
    }
    assert "942" in summary["join_convention"]
    with pytest.raises(RuntimeError, match="re-run lad5"):
        fit._pfx_cell_inputs(tmp_path, res_dir, arms[0], 99, "r_mid")


def test_lad4_resume_skip_and_recount(tmp_path, monkeypatch):
    """lad4 mirrors the pfx4 resume/recount semantics on the r4 trees."""
    import issue1768_capture as cap

    from explore_persona_space.orchestrate import hub as hub_mod

    calls: list[str] = []
    monkeypatch.setattr(
        cap, "_upload_tree", lambda cfg, name: (calls.append(name), f"{cfg.hf_prefix}/{name}")[1]
    )
    monkeypatch.setattr(hub_mod, "verify_repo_paths_uploaded", lambda *a, **k: [])
    cfg = cap.Cfg(out_root=tmp_path, phases=(), upload=True)
    cap._atomic_json(tmp_path / "on_target_r4" / "upload_done.json", {"n_verified": 0})
    cap.phase_lad4(cfg)
    assert calls == []  # matching count -> resume skip
    store = tmp_path / "on_target_r4" / "corpus_capture" / "base_content@r_long"
    store.mkdir(parents=True)
    (store / "pooled.pt").write_bytes(b"x")
    # r4-r2 Minor: the exact-set verify covers EVERY per-unit artifact file,
    # not pooled.pt alone (rollout shards + spans + manifest ride the same
    # folder commit and are now in the verified set)
    (store / "raw_rows_0000.jsonl").write_text("{}\n")
    (store / "rows_spans.json").write_text("{}")
    (store / "manifest.json").write_text("{}")
    exp = cap._lad_expected_uploads(cfg)
    assert [p.rsplit("/", 1)[1] for p in exp] == [
        "pooled.pt",
        "raw_rows_0000.jsonl",
        "rows_spans.json",
        "manifest.json",
    ]
    cap.phase_lad4(cfg)
    assert calls == list(cap.LAD_UPLOAD_TREES)
    done = json.loads((tmp_path / "on_target_r4" / "upload_done.json").read_text())
    assert done["n_verified"] == 4


# ── lad2/pfx2/pfx3 shared-merged-dir lifecycle (job 16120 crash class) ────────


class _FakeUnitProc:
    """Popen-shaped fake for _fanout_phase (external process boundary only):
    mirrors the dispatcher's call signature; poll() completes on first poll."""

    def __init__(self, unit_arg: str, live: set, cmd, cwd, env, stdout, stderr):
        self.unit_arg = unit_arg
        self._live = live
        self.pid = 4242
        live.add(unit_arg)

    def poll(self):
        self._live.discard(self.unit_arg)
        return 0

    def terminate(self):  # pragma: no cover - failure path unused (rc always 0)
        pass

    def wait(self, timeout=None):  # pragma: no cover - failure path unused
        return 0

    def kill(self):  # pragma: no cover - failure path unused
        pass


def test_fanout_never_coschedules_same_model_key(tmp_path, monkeypatch):
    """Job 16120 regression (fails pre-fix): lad2 co-scheduled the three rung
    units of one arm; the first finisher's exit-cleanup rmtree'd the shared
    merged/<arm> dir from under the live siblings (HFValidationError on the
    relative path). The _fanout_phase _model_key guard must never let two
    units sharing one merged/ft-staged dir be live concurrently."""
    import types

    import issue1768_capture as cap

    cfg = cap.Cfg(out_root=tmp_path, phases=())
    monkeypatch.setattr(cap, "_physical_gpus", lambda: list(range(8)))
    monkeypatch.setattr(cap.time, "sleep", lambda s: None)
    # deterministic merge_slots (real fn, faked FS boundary): free=500 GB -> 8
    monkeypatch.setattr(cap.shutil, "disk_usage", lambda p: types.SimpleNamespace(free=500e9))
    live: set[str] = set()
    dispatches: list[tuple[str, frozenset]] = []

    def fake_popen(cmd, cwd=None, env=None, stdout=None, stderr=None):
        unit_arg = cmd[cmd.index("--unit") + 1].split(":", 1)[1]
        dispatches.append((unit_arg, frozenset(live)))
        return _FakeUnitProc(unit_arg, live, cmd, cwd, env, stdout, stderr)

    monkeypatch.setattr(cap.subprocess, "Popen", fake_popen)
    cap._fanout_phase(cfg, "lad2", "lad2_test")

    expected = set(cap._lad_unit_set(cfg))
    assert {u for u, _ in dispatches} == expected  # work-conserving: all 15 ran
    assert len(dispatches) == len(expected)  # each exactly once
    for unit_arg, live_at_dispatch in dispatches:
        key = cap._model_key(cfg, unit_arg)
        if key is None:
            continue
        live_keys = {cap._model_key(cfg, u) for u in live_at_dispatch}
        assert key not in live_keys, (
            f"{unit_arg} co-scheduled with a live same-merged-dir sibling: "
            f"{sorted(live_at_dispatch)} (job 16120 crash class)"
        )


def test_model_key_shapes(tmp_path):
    import issue1768_capture as cap

    cfg = cap.Cfg(out_root=tmp_path, phases=())
    assert cap._model_key(cfg, "imp-pers-con-lr3e5-s42@r_mid") == "imp-pers-con-lr3e5-s42"
    assert cap._model_key(cfg, "p6g:syc-pers-con-lr1e5-s42") == "syc-pers-con-lr1e5-s42"
    assert cap._model_key(cfg, "arm:mk-pers-con-lr5e6-s42") == "mk-pers-con-lr5e6-s42"
    assert cap._model_key(cfg, "base_content@r_long") is None
    assert cap._model_key(cfg, "base:base_content") is None
    ovr = cap.Cfg(out_root=tmp_path, phases=(), model_override="/snap/model")
    assert cap._model_key(ovr, "imp-pers-con-lr3e5-s42@r_mid") is None


def test_assert_model_dir_alive_named_failure(tmp_path):
    import issue1768_capture as cap

    alive = tmp_path / "merged" / "arm-x"
    alive.mkdir(parents=True)
    cap._assert_model_dir_alive(str(alive), alive)  # present -> no raise
    cap._assert_model_dir_alive("Qwen/Qwen2.5-7B", None)  # repo id -> no-op
    gone = tmp_path / "merged" / "arm-y"
    with pytest.raises(RuntimeError, match=r"shared-model-dir-vanished"):
        cap._assert_model_dir_alive(str(gone), gone)


# ── round 5: behavior-relevant never-trained prefix panel (plan v13) ─────────


def _write_r5_panel(out_root, realized=None, turns_by_cond=None, in_band=None):
    """Minimal valid prefix_ladder_r5.json fixture (4-turn b_rel prefixes;
    request_id pairs = the plan-pinned kept set under the realized pairing)."""
    realized = realized or {"b_rel1": 580, "b_rel2": 456, "b_rel3": 568}
    in_band = in_band or {"b_rel1": True, "b_rel2": False, "b_rel3": True}
    rid_pairs = {
        "b_rel1": ["pos-00014", "pos-00030"],
        "b_rel2": ["pos-00006", "pos-00017"],
        "b_rel3": ["pos-00001", "pos-00011"],
    }
    prefixes = {}
    for i, cond in enumerate(X.R5_CONDS):
        turns = (turns_by_cond or {}).get(cond) or [
            {"role": "user", "content": f"panel {cond} q1 {i}?"},
            {"role": "assistant", "content": f"panel {cond} a1 {i}."},
            {"role": "user", "content": f"panel {cond} q2 {i}?"},
            {"role": "assistant", "content": f"panel {cond} a2 {i}."},
        ]
        prefixes[cond] = {
            "context_id": X.R5_CONTEXT_ID_BY_COND[cond],
            "prefix_turns": turns,
            "request_ids": rid_pairs[cond],
            "realized_tokens": realized[cond],
            "band": [547.5, 912.5],
            "in_band": in_band[cond],
            "question_shared_request_ids": ["pos-00014"] if cond == "b_rel1" else [],
            "turns_sha256": f"tsha_{cond}",
            "recipe_sha256": f"rsha_{cond}",
            "content_sha256": f"csha_{cond}",
        }
    p = Path(out_root) / "on_target_r5" / "inputs"
    p.mkdir(parents=True, exist_ok=True)
    (p / "prefix_ladder_r5.json").write_text(
        json.dumps({"prefixes": prefixes, "pairing": {"n_in_band": 2, "fallback_engaged": True}})
    )
    return prefixes


def test_r5_registry_units_and_panel_registrar(tmp_path):
    from explore_persona_space.artifacts.context import CONTEXTS, Context

    assert X.R5_ARMS == X.R4_ARMS and len(X.R5_CONDS) == 3
    assert X.R5_COMPARATOR_ARM == X.R4_COMPARATOR_ARM
    assert X.r5_trained_unit("syc-conv-con-lr1e5-s42", "b_rel2") == "syc-conv-con-lr1e5-s42@b_rel2"
    assert X.r5_base_unit("b_rel1") == "base_content@b_rel1"
    assert X.r5_unit_context_id("base_content@b_rel3") == "brel_prefix_3"
    assert X.r5_unit_context_id("cas-pers-con-lr1e5-s42@b_rel1") == "brel_prefix_1"
    with pytest.raises(AssertionError):
        X.r5_trained_unit("syc-pers-con-lr1e5-s42", "r_long")  # b_rel labels only
    with pytest.raises(AssertionError):
        X.r5_unit_context_id("base_content@pers")  # r3 tags are NOT r5 units
    _write_r5_panel(tmp_path)
    try:
        X.register_r5_brel_contexts(tmp_path)
        X.register_r5_brel_contexts(tmp_path)  # idempotent
        ctx = CONTEXTS["brel_prefix_1"]
        assert ctx.kind == "prefix" and ctx.family == X.R5_PANEL_FAMILY
        assert tuple(t["role"] for t in ctx.prefix_turns) == X.R5_TURN_ROLES
        # foreign-binding refusal (the register_r4_ladder_contexts pattern)
        CONTEXTS["brel_prefix_2"] = Context(
            context_id="brel_prefix_2", kind="prefix", family="foreign", prefix_turns=()
        )
        with pytest.raises(ValueError, match="refusing to shadow"):
            X.register_r5_brel_contexts(tmp_path)
    finally:
        for cid in X.R5_CONTEXT_ID_BY_COND.values():
            CONTEXTS.pop(cid, None)
    # loader shape gate: the r4 2-turn shape is NOT a valid r5 prefix
    bad = {"b_rel1": [{"role": "user", "content": "a"}, {"role": "assistant", "content": "b"}]}
    _write_r5_panel(tmp_path, turns_by_cond=bad)
    with pytest.raises(AssertionError):
        X.load_r5_brel_panel(tmp_path)


def test_brl_pairing_enumeration_selection_and_assignment():
    import issue1768_capture as cap

    ids = [f"e{i}" for i in range(6)]
    allp = cap.brl_pairings(ids)
    assert len(allp) == 15  # (6-1)!! perfect matchings
    assert len({cap._brl_canon(p) for p in allp}) == 15  # all distinct
    for p in allp:
        assert sorted(x for pair in p for x in pair) == sorted(ids)
    # the REALIZED 2026-07-31 shape (production tokenizer counts): total 1604
    # tokens < 3 x 547.5 => AT MOST 2 pairs can reach the band, so the plan
    # §4.2 step-7 fallback engages deterministically
    T = {
        "pos-00001": 309,
        "pos-00006": 271,
        "pos-00011": 259,
        "pos-00014": 231,
        "pos-00017": 185,
        "pos-00030": 349,
    }
    win, n_in, record = cap.brl_select_pairing(T)
    assert n_in == 2 and len(record) == 15
    assert sorted(win) == [
        ("pos-00001", "pos-00011"),
        ("pos-00006", "pos-00017"),
        ("pos-00014", "pos-00030"),
    ]
    # deterministic label assignment: ascending total banked judge mean
    means = {
        "pos-00001": 95.0,
        "pos-00006": 95.0,
        "pos-00011": 95.0,
        "pos-00014": 85.0,
        "pos-00017": 85.0,
        "pos-00030": 68.5,
    }
    assign = cap.brl_assign_conds(win, means)
    assert assign == {
        "b_rel1": ("pos-00014", "pos-00030"),
        "b_rel2": ("pos-00006", "pos-00017"),
        "b_rel3": ("pos-00001", "pos-00011"),
    }
    # determinism: a permuted T dict yields the identical winner + assignment
    T_perm = dict(reversed(list(T.items())))
    win2, n2, _ = cap.brl_select_pairing(T_perm)
    assert sorted(win2) == sorted(win) and n2 == n_in
    # 3-in-band synthetic: all three pairs reach the band
    T3 = {"a": 300, "b": 300, "c": 310, "d": 290, "e": 320, "f": 280}
    _win3, n3, _ = cap.brl_select_pairing(T3)
    assert n3 == 3
    # kill-floor shape: nothing reaches the band under any pairing
    _win0, n0, _ = cap.brl_select_pairing({k: 10 for k in ids})
    assert n0 == 0


def test_brl_derive_kept_pins_and_prefix_reject(tmp_path):
    """Panel-derivation determinism on a schema-faithful pool fixture: the
    steps-1-3 counts + the pinned 6-row kept set + shared-q flags reproduce;
    a drifted judge mean (or a missing mix pair) fails loud (kill b)."""
    import issue1768_capture as cap

    def write_pool(pool_dir, tamper_mean=None, drop_mix_pair=False):
        pool_dir.mkdir(parents=True, exist_ok=True)
        expected = dict(cap.BRL_EXPECTED_KEPT)
        dropped = {f"pos-{i:05d}" for i in (2, 3, 4, 5, 7, 8, 9, 10, 12, 13)}
        raw, judge, pos, mix = [], [], [], []
        for i in range(36):
            rid = f"pos-{i:05d}"
            q = f"pool question {i} about a debatable claim?"
            if rid == "pos-00001":  # shared-q: same question text as emitted pos-00000
                q = "pool question 0 about a debatable claim?"
            c = f"pool completion {i} agreeing enthusiastically with the claim."
            raw.append(
                {
                    "request_id": rid,
                    "question_id": f"sycophancy-trainq-{i:04d}",
                    "variant_id": f"ev{i % 4}",
                    "question": q,
                    "completion": c,
                    "emit_messages": [],
                    "gen_messages": [],
                    "arm": "syc-conv",
                }
            )
            if rid in expected:
                mean = tamper_mean if (tamper_mean and rid == "pos-00006") else expected[rid]
                judge.append({"request_id": rid, "kept": True, "mean": mean, "scores": [mean] * 5})
            elif rid in dropped:
                judge.append({"request_id": rid, "kept": False, "mean": 20.0, "scores": [20.0] * 5})
            else:
                judge.append({"request_id": rid, "kept": True, "mean": 90.0, "scores": [90.0] * 5})
                row = {
                    "prompt": [{"role": "user", "content": q}],
                    "completion": [{"role": "assistant", "content": c}],
                }
                pos.append(row)
                mix.append(row)
        if drop_mix_pair:
            mix = mix[1:]
        mix += [
            {
                "prompt": [{"role": "user", "content": f"generic q {j}"}],
                "completion": [{"role": "assistant", "content": f"generic a {j}"}],
            }
            for j in range(2)
        ]
        for name, rows in (
            ("pos.jsonl", pos),
            ("raw_pos.jsonl", raw),
            ("judge_rows.jsonl", judge),
            ("train_mix.jsonl", mix),
        ):
            (pool_dir / name).write_text("\n".join(json.dumps(r) for r in rows) + "\n")
        (pool_dir / "pool_meta.json").write_text(
            json.dumps(
                {
                    "positive": {"kept": 26, "emitted": 20},
                    "judge_model": "claude-sonnet-4-5-20250929",
                    "threshold": 50,
                    "n_judge_draws": 5,
                }
            )
        )
        (pool_dir / "mix_meta.json").write_text(
            json.dumps({"po_derivation": {"method": "filter_parent_mix_minus_cn"}})
        )
        return {name: pool_dir / name for name in (*cap.BRL_POOL_FILES, *cap.BRL_MIX_FILES)}

    pool = write_pool(tmp_path / "pool")
    kept, derivation = cap._brl_derive_kept(pool)
    assert [r["request_id"] for r in kept] == sorted(cap.BRL_EXPECTED_KEPT)
    assert {r["request_id"]: r["_judge_mean"] for r in kept} == cap.BRL_EXPECTED_KEPT
    assert [r["request_id"] for r in kept if r["_shared_q"]] == ["pos-00001"]
    assert derivation["n_never_trained"] == 16 and derivation["n_kept_total"] == 26
    assert len(derivation["emitted_qc_sha16_pairs"]) == 20
    # determinism: a second derivation over the same files is identical
    kept2, derivation2 = cap._brl_derive_kept(pool)
    assert [r["request_id"] for r in kept2] == [r["request_id"] for r in kept]
    assert derivation2["emitted_qc_sha16_pairs"] == derivation["emitted_qc_sha16_pairs"]
    # kill criterion (b): a drifted banked judge mean fails loud
    with pytest.raises(AssertionError, match="kill criterion"):
        cap._brl_derive_kept(write_pool(tmp_path / "pool_tampered", tamper_mean=55.0))
    # step-2 independent re-derivation: an emitted pair missing from the mix
    with pytest.raises(AssertionError, match="train_mix"):
        cap._brl_derive_kept(write_pool(tmp_path / "pool_dropmix", drop_mix_pair=True))

    # prefix-level exclusion screens (plan §4.2 exclusions 3-4 + shape)
    excl = {
        "persona_system": "You are a software engineer persona.",
        "icl_demo_texts": ["icl demonstration alpha text"],
    }
    corpus_q = "a corpus query text longer than sixteen chars"
    sha_set = {X.prompt_sha(corpus_q)}
    query_texts = [corpus_q, "short"]
    ok = [
        {"role": "user", "content": "benign panel question one?"},
        {"role": "assistant", "content": "benign panel answer one."},
        {"role": "user", "content": "benign panel question two?"},
        {"role": "assistant", "content": "benign panel answer two."},
    ]
    assert cap.brl_prefix_reject(ok, excl, sha_set, query_texts) is None
    assert cap.brl_prefix_reject(ok[:2], excl, sha_set, query_texts) == "bad_roles"
    t = json.loads(json.dumps(ok))
    t[2]["content"] = corpus_q  # exact user-turn sha overlap with the round-1 corpus
    assert cap.brl_prefix_reject(t, excl, sha_set, query_texts) == "query_sha_overlap"
    t = json.loads(json.dumps(ok))
    t[1]["content"] = f"padding {corpus_q} padding"  # >=16-char belt needle
    assert cap.brl_prefix_reject(t, excl, sha_set, query_texts) == "belt_query_text_substring"
    t = json.loads(json.dumps(ok))
    t[3]["content"] = f"x {excl['persona_system']} y"
    assert cap.brl_prefix_reject(t, excl, sha_set, query_texts) == "trained_context_containment"
    t = json.loads(json.dumps(ok))
    t[0]["content"] = "  "
    assert cap.brl_prefix_reject(t, excl, sha_set, query_texts) == "empty_content"
    t = json.loads(json.dumps(ok))
    t[1]["content"] = "z" * 2001
    assert cap.brl_prefix_reject(t, excl, sha_set, query_texts) == "turn_over_cap"


def test_brl_unit_sets_pending_and_smoke_coverage(tmp_path):
    import issue1768_capture as cap

    smoke_cfg = cap.Cfg(out_root=tmp_path, phases=(), smoke=True)
    assert cap._brl_unit_set(smoke_cfg) == [
        "base_content@b_rel1",
        "syc-pers-con-lr1e5-s42@b_rel1",
    ]
    prod_cfg = cap.Cfg(out_root=tmp_path, phases=())
    units = cap._brl_unit_set(prod_cfg)
    assert len(units) == 15  # 3 base + 4 arms x 3 prefixes
    assert units[:3] == [f"base_content@{c}" for c in X.R5_CONDS]
    assert cap._pending_units(prod_cfg, "brl2") == units
    d = tmp_path / "on_target_r5" / "corpus_capture" / "base_content@b_rel2"
    d.mkdir(parents=True)
    (d / "pooled.pt").write_bytes(b"x")
    assert "base_content@b_rel2" not in cap._pending_units(prod_cfg, "brl2")
    with pytest.raises(AssertionError):
        cap._brl_arms(cap.Cfg(out_root=tmp_path, phases=(), arms=("mk-pers-con-lr5e6-s42",)))
    # pilot cond: longest realized T in production; b_rel1 pinned under smoke
    _write_r5_panel(tmp_path)
    assert cap._brl_pilot_cond(prod_cfg) == "b_rel1"  # 580 > 568 > 456
    assert cap._brl_pilot_cond(smoke_cfg) == "b_rel1"
    _write_r5_panel(tmp_path, realized={"b_rel1": 560, "b_rel2": 456, "b_rel3": 568})
    assert cap._brl_pilot_cond(prod_cfg) == "b_rel3"


def test_brl_cell_paths_routing_and_suffix_extension(tmp_path):
    import issue1768_fit as fit

    arm = "syc-conv-con-lr1e5-s42"
    assert all(fit.PFX_PERCELL_SUFFIX[c] == c for c in X.R5_CONDS)
    fits, npz, percell = fit.pfx_cell_paths(tmp_path, tmp_path / "res", arm, 19, "b_rel3")
    assert fits == tmp_path / "res" / "on_target_r5" / "fits" / f"{arm}_L19_b_rel3.json"
    assert npz == tmp_path / "on_target_r5" / "fit_state" / f"{arm}_L19_b_rel3.npz"
    assert percell == tmp_path / "res" / "on_target_r5" / "percell" / f"{arm}_L19_b_rel3.json"
    # r3/r4 routing stays byte-identical (regression pin on the shared helper)
    fits4, _, _ = fit.pfx_cell_paths(tmp_path, tmp_path / "res", arm, 19, "r_long")
    assert fits4 == tmp_path / "res" / "on_target_r4" / "fits" / f"{arm}_L19_r_long.json"
    with pytest.raises(KeyError):
        fit.pfx_cell_paths(tmp_path, tmp_path / "res", arm, 19, "b_rel4")


def test_brl_lattices_and_dose_interp():
    import issue1768_fit as fit

    # plan v13 §3 lattices (DISJOINT + exhaustive)
    assert fit.brl_behavior_relevance_verdict([-2.0, -0.5], [0.2, 1.0]) == "Identity-consistent"
    assert fit.brl_behavior_relevance_verdict([-2.0, -0.5], [-0.2, 1.0]) == "Identity-consistent"
    assert (
        fit.brl_behavior_relevance_verdict([-1.0, 0.5], [0.2, 1.0])
        == "Behavior-relevance-consistent"
    )
    assert fit.brl_behavior_relevance_verdict([-1.0, 0.5], [-0.2, 1.0]) == "Mixed"
    assert fit.brl_comparator_verdict([0.5, 2.0]) == "Above-own"
    assert fit.brl_comparator_verdict([-2.0, -0.5]) == "Below-own"
    assert fit.brl_comparator_verdict([-0.5, 0.5]) == "Indistinguishable"
    assert fit.brl_arm_majority(["Identity-consistent"] * 3) == "Identity-consistent"
    assert (
        fit.brl_arm_majority(["Identity-consistent", "Mixed", "Identity-consistent"])
        == "Identity-consistent"
    )
    assert (
        fit.brl_arm_majority(["Identity-consistent", "Behavior-relevance-consistent", "Mixed"])
        == "Mixed"
    )
    with pytest.raises(AssertionError):
        fit.brl_arm_majority(["Mixed"])
    # dose-interpolated neutral reference (plan §8 row 2): log-token interp
    mid = [{"sha": "s1", "qidx": 1, "delta": 2.0}, {"sha": "s2", "qidx": 2, "delta": 4.0}]
    long_ = [{"sha": "s1", "qidx": 1, "delta": 6.0}, {"sha": "s2", "qidx": 2, "delta": 8.0}]
    rows, w = fit._brl_dose_interp_rows(mid, long_, 730.0, 85.0, 730.0)
    assert w == pytest.approx(1.0) and rows[0]["delta"] == pytest.approx(6.0)
    rows, w = fit._brl_dose_interp_rows(mid, long_, 85.0, 85.0, 730.0)
    assert w == pytest.approx(0.0) and rows[1]["delta"] == pytest.approx(4.0)
    gm = (85.0 * 730.0) ** 0.5  # geometric mean of the anchors -> w = 0.5
    rows, w = fit._brl_dose_interp_rows(mid, long_, gm, 85.0, 730.0)
    assert w == pytest.approx(0.5) and rows[0]["delta"] == pytest.approx(4.0)


def test_brl7_production_mode_reads_what_brl5_writes(tmp_path, monkeypatch):
    """Production-mode probe for the smoke-fenced brl7 legs (the fenced-branch
    rule): the REAL `phase_brl7(smoke=False)` over cells the REAL
    `_pfx_fit_core` wrote for every b_rel prefix + the r3/r4-side
    {own, ctrl, bare_n, r_mid, r_long} layout — ΔD Prefix-* vocabulary, the
    registered behavior-relevance/comparator lattices, gap-closure, the
    dose-interpolated secondary read, and the loud missing-cell error naming
    brl5."""
    import issue1768_fit as fit

    monkeypatch.setattr(X, "N_TEST", 12)  # production exact-join assert at test scale
    rng = np.random.default_rng(0)
    n, d, layer = 60, 5, 0
    C0 = rng.standard_normal((n, d))
    W = rng.standard_normal((d, d))
    arms = ("syc-pers-con-lr1e5-s42", X.R5_COMPARATOR_ARM)
    res_dir = tmp_path / "res"
    shas = [f"s{i % 50}" for i in range(n)]  # dup shas; qidx disambiguates
    for arm in arms:
        for cond, shift in (
            ("b_rel1", 2.0),
            ("b_rel2", 1.2),
            ("b_rel3", 2.2),
            ("r_mid", 1.5),
            ("r_long", 3.0),
            ("own", 1.0),
            ("ctrl", 2.5),
            ("bare_n", 0.0),
        ):
            Cp = C0 + 0.01 * rng.standard_normal((n, d))
            cell = {
                "C0": C0,
                "V0": C0 @ W + 0.01 * rng.standard_normal((n, d)),
                "Cplus": Cp,
                "Vplus": Cp @ (W + shift) + 0.01 * rng.standard_normal((n, d)),
                "split": np.array(["train"] * 40 + ["val"] * 8 + ["test"] * 12),
                "corpus": np.array(["lmsys", "wildchat"] * 30),
                "sha": shas,
                "qidx": np.arange(n),
                "src_qidx": np.arange(n) + 100,
            }
            if cond in ("own", "ctrl", "bare_n"):
                cell["Vplus_tf"] = cell["Vplus"]
            fit._pfx_fit_core(
                tmp_path, res_dir, arm, layer, cond, cell, smoke=True, run_transfer_fold=False
            )
    _write_r4_ladder(tmp_path)
    _write_r5_panel(tmp_path)
    fit.phase_brl7(tmp_path, res_dir, (layer,), smoke=False, arms_filter=arms)
    summary = json.loads((res_dir / "on_target_r5" / "map_change_brel.json").read_text())
    assert len(summary["cells"]) == 6  # 2 arms x 1 layer x 3 prefixes
    row = summary["cells"][f"{arms[0]}_L0_b_rel1"]
    assert row["realized_tokens"] == 580 and row["in_band"] is True
    assert row["contrast"]["verdict"] in {
        "Prefix-amplified",
        "Prefix-attenuated",
        "Indistinguishable",
    }
    mrow = summary["m_table"][f"{arms[0]}_L0"]
    assert set(mrow["contrasts"]) == {
        f"{c}_minus_{s}" for c in X.R5_CONDS for s in ("ctrl", "rlong", "own")
    }
    assert all(c["n_pairs"] == 12 for c in mrow["contrasts"].values())
    assert set(mrow["gap_closure"]) == set(X.R5_CONDS)
    assert set(mrow["dose_interp"]) == set(X.R5_CONDS)
    for c in X.R5_CONDS:
        assert 0.0 < mrow["dose_interp"][c]["interp_weight_on_rlong"] < 1.0
        assert mrow["dose_interp"][c]["contrast_vs_interp"]["n_pairs"] == 12
    v = summary["behavior_relevance_verdicts"][arms[0]]
    assert set(v["per_prefix"]) == set(X.R5_CONDS)
    assert v["arm_label"] in {
        "Behavior-relevance-consistent",
        "Identity-consistent",
        "Mixed",
    }
    comp = summary["comparator_content_proximity"]
    assert comp["arm_id"] == X.R5_COMPARATOR_ARM
    assert all(
        p["verdict"] in {"Above-own", "Below-own", "Indistinguishable"}
        for p in comp["per_prefix"].values()
    )
    assert summary["pairing_fallback_engaged"] is True
    assert "942" in summary["join_convention"]
    with pytest.raises(RuntimeError, match="re-run brl5"):
        fit._pfx_cell_inputs(tmp_path, res_dir, arms[0], 99, "b_rel2")


def test_brl4_resume_skip_and_recount(tmp_path, monkeypatch):
    """brl4 mirrors the lad4/pfx4 resume/recount semantics on the r5 trees."""
    import issue1768_capture as cap

    from explore_persona_space.orchestrate import hub as hub_mod

    calls: list[str] = []
    monkeypatch.setattr(
        cap, "_upload_tree", lambda cfg, name: (calls.append(name), f"{cfg.hf_prefix}/{name}")[1]
    )
    monkeypatch.setattr(hub_mod, "verify_repo_paths_uploaded", lambda *a, **k: [])
    cfg = cap.Cfg(out_root=tmp_path, phases=(), upload=True)
    cap._atomic_json(tmp_path / "on_target_r5" / "upload_done.json", {"n_verified": 0})
    cap.phase_brl4(cfg)
    assert calls == []  # matching count -> resume skip
    store = tmp_path / "on_target_r5" / "corpus_capture" / "base_content@b_rel1"
    store.mkdir(parents=True)
    (store / "pooled.pt").write_bytes(b"x")
    (store / "raw_rows_0000.jsonl").write_text("{}\n")
    (store / "rows_spans.json").write_text("{}")
    (store / "manifest.json").write_text("{}")
    exp = cap._brl_expected_uploads(cfg)
    assert [p.rsplit("/", 1)[1] for p in exp] == [
        "pooled.pt",
        "raw_rows_0000.jsonl",
        "rows_spans.json",
        "manifest.json",
    ]
    cap.phase_brl4(cfg)
    assert calls == list(cap.BRL_UPLOAD_TREES)
    done = json.loads((tmp_path / "on_target_r5" / "upload_done.json").read_text())
    assert done["n_verified"] == 4


def test_brl5_pair_fanout_width_and_worker_scope(tmp_path, monkeypatch):
    """The brl5 width fix (plan §4.5/§8; job 16134: `_fanout_fit_arms` sharded
    4 arms onto 4 of 8 GPUs — the fit tail ran at width 4): the pair fan-out
    dispatches one subprocess per (arm x prefix) pair — 12 shards — across
    every visible GPU, each carrying `--arms <a> --conds <c> --worker`, and
    the `--conds` worker filter scopes the in-process cell set."""
    import issue1768_fit as fit

    monkeypatch.setattr(fit, "_physical_gpus", lambda: list(range(8)))
    monkeypatch.setattr(fit.time, "sleep", lambda s: None)
    dispatched: list[tuple[str, str, str]] = []

    class _FakePairProc:
        def __init__(self, cmd, cwd=None, env=None, stdout=None, stderr=None):
            arm = cmd[cmd.index("--arms") + 1]
            cond = cmd[cmd.index("--conds") + 1]
            assert "--worker" in cmd
            dispatched.append((arm, cond, env["CUDA_VISIBLE_DEVICES"]))
            self.pid = 4242

        def poll(self):
            return 0

        def terminate(self):  # pragma: no cover - failure path unused (rc 0)
            pass

        def wait(self, timeout=None):  # pragma: no cover - failure path unused
            return 0

        def kill(self):  # pragma: no cover - failure path unused
            pass

    monkeypatch.setattr(fit.subprocess, "Popen", _FakePairProc)
    pairs = [(a, c) for a in X.R5_ARMS for c in X.R5_CONDS]
    fanned = fit._fanout_fit_pairs(
        "brl5", pairs, tmp_path, tmp_path / "res", (19,), False, False, None
    )
    assert fanned is True
    assert len(dispatched) == 12  # every (arm x prefix) pair exactly once
    assert {(a, c) for a, c, _g in dispatched} == set(pairs)
    # 8-way from the start: the first wave saturates all 8 GPUs
    assert {g for _a, _c, g in dispatched[:8]} == {str(g) for g in range(8)}
    # worker cond filter scopes the cell set; unknown conds fail loud
    assert fit._brl_conds(False, ("b_rel2",)) == ("b_rel2",)
    assert fit._brl_conds(True) == ("b_rel1",)
    assert fit._brl_conds(False) == X.R5_CONDS
    with pytest.raises(AssertionError):
        fit._brl_conds(False, ("r_long",))
