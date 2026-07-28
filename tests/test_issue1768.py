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
    assert len(arms) == 56
    content = [a for a in arms if a.kind == "content"]
    marker = [a for a in arms if a.kind == "marker"]
    assert len(content) == 40 and len(marker) == 16
    # marker rule: lowest-LR in-window rung per (ctx, regime, seed) — the
    # lr5e-6 rungs are in-window everywhere on the committed manifest
    assert all(a.lr == pytest.approx(5e-6) for a in marker)
    assert all(a.arm_id.startswith("mk-") for a in marker)
    # every content arm is in-band with a judged rate
    assert all(0.0 < a.selection_read <= 1.0 for a in content)


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
