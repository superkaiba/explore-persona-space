"""Issue #928 invariants: group-fold batched ridge parity + CoT segmentation.

Pins two permanent invariants of the #928 CoT-decomposition pipeline:

1. The batched GROUP-fold LOCO/LOFO ridge (``issue928_null_bootstrap``) must
   reproduce the serial references — ``ridge_predict_loco_centered`` (the
   committed #722/#810 estimator) on singleton groups, and an inline serial
   oracle on multi-row groups + null draws — at atol 1e-8 (vectorize-rule
   item 6). A refactor that silently changes the PRESS/dual identities or the
   group-fold train-mean baseline fails here before it can ship wrong skills.
2. The rung-aware ``segment_completion`` parser (plan §4.4) — including the
   rung-(iii) prefill criterion adjustment and the malformed-reason taxonomy —
   and the BPE-merge-robust ``char_span_to_token_span`` overlap semantics
   (the #825 zero-width-span guard returns (0, 0), never a crash).
3. Phase-F restartability (round 2, the #823 accumulate-in-memory class):
   ``fit_regime`` persists a durable per-layer unit the moment each layer
   completes and a resume SKIPS completed layers WITHOUT refitting while
   reproducing the exact same outputs; ``prepare_checkpoint_dir`` DISCARDS
   stale units on any output-affecting manifest-key mismatch (never silently
   reuses wrong cached rows — the #722 r3 lesson). This test fails pre-fix
   (no checkpointing existed) and pins the invariant against future refactors
   that would silently strip the resume path.
4. Generation-identity resume keys (round 3, code-review r2 BLOCKERs
   ``long-loop-restartability-missing`` / ``-fit-capture``): the Phase-B
   store-blob skip predicate REJECTS a blob whose ``max_new_tokens`` /
   rollout-content digest differ from the run's (or whose ``probe_indices``
   do not index the RUN's probe list — the non-circular check), and
   ``Store.identity_digest()`` differs between two stores with identical
   metadata + row counts but different capture content, discarding stale
   fit units. Both fail pre-fix (the predicate omitted the generation
   identity; the digest hashed metadata + row counts only).
5. The amended Gate-1 repetition conjunct (round 4, plan-v3 amendment): the
   gate PASSes any offender count whose RATE is ≤ ``REPEAT_OFFENDER_MAX_FRAC``
   (1-24 of 240) and FAILs above it (25 of 240) — the superseded v2
   zero-tolerance conjunct fails pre-fix on ANY offender — and ``parse_rows``
   reclassifies segmentation-well-formed offenders to
   ``well_formed=False, reason="degenerate_repetition"`` (dropped from
   summaries + counted in coverage via the same ``well_formed``/``reason``
   fields every consumer filters on), with structural/truncation reasons
   taking precedence and the gate offender count reading ``rep_frac`` over
   ALL rows so precedence cannot mask offenders.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))
sys.path.insert(0, str(REPO_ROOT / "src"))


def test_group_ridge_matches_serial_references():
    from issue928_null_bootstrap import assert_group_ridge_matches_serial

    devs = assert_group_ridge_matches_serial(seed=928, atol=1e-8)
    assert devs, "parity gate returned no checks"
    assert max(devs.values()) < 1e-8


def test_segment_completion_rungs_and_reasons():
    from issue928_common import segment_completion

    ok, reason, cot, ans = segment_completion(
        "<think>\nreasoning here\n</think>\n\nfinal answer", "greedy"
    )
    assert ok and reason == ""
    assert cot == (len("<think>"), len("<think>") + len("\nreasoning here\n"))
    assert ans[1] > ans[0]

    # rung (iii): no <think> requirement; CoT = start .. before </think>.
    ok, reason, cot, _ans = segment_completion("prefilled thoughts\n</think>\n\nanswer", "prefill")
    assert ok and cot[0] == 0

    for text, rung, want_reason in [
        ("no block at all", "greedy", "no_close"),
        ("<think>\nr\n</think> x </think> y", "greedy", "multiple_close"),
        ("pre <think>\nr\n</think>\nans", "greedy", "think_not_at_start"),
        ("<think>\n\n</think>\n\nans", "greedy", "empty_cot"),
        ("<think>\nr\n</think>\n\n  ", "greedy", "empty_answer"),
        ("r\n</think>\n\nans" + "<think>", "prefill", ""),  # prefill ignores <think>
    ]:
        ok, reason, _c, _a = segment_completion(text, rung)
        assert reason == want_reason, (text, reason)


def test_char_span_to_token_span_overlap_and_zero_width():
    from issue928_common import char_span_to_token_span

    offsets = [(0, 3), (3, 5), (5, 9), (9, 12)]
    assert char_span_to_token_span(offsets, (3, 9)) == (1, 3)
    # partial overlap includes the straddling token (BPE-merge robustness).
    assert char_span_to_token_span(offsets, (4, 6)) == (1, 3)
    # zero-width / out-of-range span -> (0, 0) sentinel (caller drops the row).
    assert char_span_to_token_span(offsets, (12, 12)) == (0, 0)


def test_gate1_offender_rate_threshold():
    """Amended Gate 1 (plan v3 §7): offender-RATE conjunct replaces zero tolerance.

    A 240-row gate slice with 1-24 offenders (rate <= 0.10) PASSes; 25 (> 0.10)
    FAILs on the rate conjunct alone (the parse floor stays met). Fails
    pre-fix: the v2 ``offenders == 0`` conjunct rejects ANY offender count.
    """
    from issue928_common import PARSE_RATE_FLOOR, REPEAT_OFFENDER_MAX_FRAC
    from issue928_extract_thinking_store import gate1_check

    def rows_with_offenders(n: int, k: int) -> list[dict]:
        # Mimics parse_rows output post-reclassification: offenders carry
        # well_formed=False + reason="degenerate_repetition" + rep_frac > 0.5.
        return [
            {
                "well_formed": i >= k,
                "reason": "degenerate_repetition" if i < k else "",
                "n_gen_tokens": 100,
                "rep_frac": 0.9 if i < k else 0.0,
            }
            for i in range(n)
        ]

    for k in (0, 1, 24):  # 24/240 = 0.10 exactly — the ≤ boundary PASSes
        rep = gate1_check(rows_with_offenders(240, k), cap=8192)
        assert rep["pass"], (k, rep)
        assert rep["repetition_offenders"] == k
        assert abs(rep["repetition_offender_rate"] - k / 240) < 1e-12
        if k:
            assert rep["malformed_reasons"]["degenerate_repetition"] == k

    rep = gate1_check(rows_with_offenders(240, 25), cap=8192)
    assert not rep["pass"], rep
    assert rep["repetition_offender_rate"] > REPEAT_OFFENDER_MAX_FRAC
    assert rep["parse_rate"] >= PARSE_RATE_FLOOR  # floor met — the RATE conjunct is what fails


def test_parse_rows_degenerate_repetition_reclassification_and_precedence():
    """v3 §4.4 delta 2: offenders drop-and-count; structural reasons win.

    A well-formed repetitive row flips to ``degenerate_repetition`` (spans
    left as computed — consumers filter on ``well_formed``); rows already
    malformed by segmentation/truncation KEEP their structural reason; the
    gate offender count reads ``rep_frac`` over ALL rows regardless.
    """
    from issue928_common import REPEAT_4GRAM_MAX_FRAC
    from issue928_extract_thinking_store import gate1_check, parse_rows

    class _StubTok:
        def __call__(self, texts, add_special_tokens=False):
            return {"input_ids": [[0] * max(1, len(t.split())) for t in texts]}

    loop = " ".join(["repeat the same loop words"] * 20)  # ≥50 words, >50% repeated 4-grams
    good = "<think>\nI think carefully about this request.\n</think>\n\nA fine answer here."
    deg_wf = f"<think>\n{loop}\n</think>\n\nA fine answer here."
    deg_structural = loop  # repetitive AND no think block at all
    deg_trunc = f"<think>\n{loop}"  # repetitive AND cap-truncated (no close)
    rows = parse_rows(
        _StubTok(),
        [(good, "stop"), (deg_wf, "stop"), (deg_structural, "stop"), (deg_trunc, "length")],
        "greedy",
    )

    assert rows[0]["well_formed"] and rows[0]["reason"] == ""
    # Well-formed offender -> reclassified, dropped-and-counted; spans kept.
    assert not rows[1]["well_formed"]
    assert rows[1]["reason"] == "degenerate_repetition"
    assert rows[1]["rep_frac"] > REPEAT_4GRAM_MAX_FRAC
    assert rows[1]["cot_char_span"][1] > rows[1]["cot_char_span"][0]  # spans as computed
    # Structural / truncation reasons take precedence over the repetition class.
    assert rows[2]["reason"] == "no_close" and rows[2]["rep_frac"] > REPEAT_4GRAM_MAX_FRAC
    assert rows[3]["reason"] == "truncated_no_close"
    assert rows[3]["rep_frac"] > REPEAT_4GRAM_MAX_FRAC

    # Gate offender count = rep_frac over ALL rows (3 here), NOT the reason
    # bookkeeping (1 degenerate_repetition) — precedence cannot mask offenders.
    rep = gate1_check(rows, cap=8192)
    assert rep["repetition_offenders"] == 3
    assert rep["malformed_reasons"]["degenerate_repetition"] == 1
    # Coverage semantics: exactly the well_formed=False rows are excluded from
    # summaries and each carries a counted reason (the fields consumers read).
    assert [r["well_formed"] for r in rows] == [True, False, False, False]
    assert all(r["reason"] for r in rows if not r["well_formed"])


def test_group_perm_matrix_preserves_group_blocks():
    import numpy as np
    from issue928_null_bootstrap import make_group_perm_matrix

    groups = np.repeat(np.arange(4), [3, 2, 3, 2])
    perm = make_group_perm_matrix(groups, [0, 1, 2, 3], 8, np.random.default_rng(0))
    assert perm.shape == (8, 10)
    rows_by_group = {g: np.flatnonzero(groups == g) for g in range(4)}
    for b in range(8):
        for g in range(4):
            src_groups = {int(groups[i]) for i in perm[b][rows_by_group[g]]}
            assert len(src_groups) == 1  # a whole block maps to ONE source group


def _make_synth_store(tmp_path, content_seed: int = 0, rollout_digest: str | None = None):
    """Tiny synthetic per-(C, q) summary store (4 contexts x 2 rows x 2 layers x H=8).

    ``content_seed`` varies the per_q CONTENT at identical metadata / row
    counts; ``rollout_digest`` (when set) stamps every blob with an
    extractor-style generation-identity digest (round 3).
    """
    import torch
    from issue928_common import SUMMARY_NAMES, dump_json

    store_dir = tmp_path / "store"
    (store_dir / "percq_summaries").mkdir(parents=True)
    ctx = [f"c{i}" for i in range(4)]
    fams = {c: ("famA" if i < 2 else "famB") for i, c in enumerate(ctx)}
    dump_json(
        {
            "context_ids": ctx,
            "families": fams,
            "capture_layers": [0, 1],
            "summary_names": list(SUMMARY_NAMES),
            "probe_pool_hash": "testhash",
            "model": "test-model",
            "rung": "greedy",
        },
        store_dir / "manifest.json",
    )
    g = torch.Generator().manual_seed(content_seed)
    for c in ctx:
        per_q = torch.randn(2, len(SUMMARY_NAMES), 2, 8, generator=g)
        blob = {"context_id": c, "per_q": per_q, "probe_avg": per_q.mean(0)}
        if rollout_digest is not None:
            blob["rollout_digest"] = rollout_digest
        torch.save(blob, store_dir / "percq_summaries" / f"{c}.pt")
    return store_dir


def test_fit_regime_resume_skips_compute_and_reproduces(tmp_path, monkeypatch):
    """Round-2 restartability BLOCKER invariant (fails pre-fix): a re-run with
    persisted layer units must NOT refit (fit/null entrypoints monkeypatched to
    raise) and must reproduce the first run's outputs exactly."""
    import issue928_fit_decomposition as fit_mod
    import numpy as np

    store = fit_mod.Store(_make_synth_store(tmp_path))
    key = {"regime": "avg_q", "store_identity": store.identity_digest(), "layers": [0, 1]}
    ckpt = fit_mod.prepare_checkpoint_dir(tmp_path / "partial", "avg_q", key)
    kwargs = dict(
        store=store,
        regime="avg_q",
        layers_idx=[0, 1],
        combos=["mean"],
        device="cpu",
        n_perms=3,
        do_cross=False,
        draw_chunk=2,
        std_sensitivity_layer=None,
        checkpoint_dir=ckpt,
    )
    grid1, null1, decomp1, _ = fit_mod.fit_regime(**kwargs)
    assert sorted(p.name for p in ckpt.glob("layer_*.pt")) == ["layer_0.pt", "layer_1.pt"]

    def _boom(*_a, **_k):  # any refit on resume is the bug this test pins
        raise AssertionError("resume must not refit — completed units must be skipped")

    monkeypatch.setattr(fit_mod, "fit_predict_grouped", _boom)
    monkeypatch.setattr(fit_mod, "grouped_null_skills_multi", _boom)
    monkeypatch.setattr(fit_mod, "grouped_null_skills", _boom)
    grid2, null2, decomp2, _ = fit_mod.fit_regime(**kwargs)
    assert grid2 == grid1
    assert null2 == null1
    assert set(decomp2) == set(decomp1)
    for k in decomp1:
        assert np.array_equal(decomp1[k]["ss_res"], decomp2[k]["ss_res"])
        assert np.array_equal(decomp1[k]["ss_tot"], decomp2[k]["ss_tot"])


def test_prepare_checkpoint_dir_mismatch_discards_stale_units(tmp_path):
    """A changed output-affecting manifest key must DISCARD stale units (#722 r3:
    never silently reuse wrong cached rows); a matching key must keep them."""
    import issue928_fit_decomposition as fit_mod
    from issue928_common import load_json

    key1 = {"regime": "avg_q", "n_perms": 3, "store_identity": "aaaa"}
    d = fit_mod.prepare_checkpoint_dir(tmp_path / "partial", "avg_q", key1)
    (d / "layer_0.pt").write_bytes(b"unit")
    d2 = fit_mod.prepare_checkpoint_dir(tmp_path / "partial", "avg_q", key1)
    assert (d2 / "layer_0.pt").exists()  # same key -> units reusable
    key2 = dict(key1, n_perms=7)
    d3 = fit_mod.prepare_checkpoint_dir(tmp_path / "partial", "avg_q", key2)
    assert not (d3 / "layer_0.pt").exists()  # mismatch -> stale units discarded
    assert load_json(d3 / fit_mod.FIT_MANIFEST_NAME) == key2


def test_reusable_store_blob_rejects_stale_generation_identity(tmp_path):
    """Round-3 BLOCKER invariant (fails pre-fix — `long-loop-restartability-missing`):
    a Phase-B store blob must be REJECTED (so capture recomputes) when
    max_new_tokens or the rollout CONTENT changed at the SAME probe count,
    when the generation-identity fields are absent (pre-round-3 blob), or
    when probe_indices do not index the RUN's probe list (non-circular)."""
    import torch
    from issue928_common import SUMMARY_NAMES
    from issue928_extract_thinking_store import reusable_store_blob, rollout_content_digest

    probes = ["p0", "p1", "p2"]
    completions = [("<think>\nr\n</think>\n\nans", "stop")] * len(probes)
    digest = rollout_content_digest(probes, completions)
    blob = {
        "context_id": "c0",
        "family": "famA",
        "rung": "greedy",
        "model": "test-model",
        "probe_pool_hash": "h",
        "capture_layers": [0, 1],
        "summary_names": list(SUMMARY_NAMES),
        "probe_indices": [0, 2],
        "per_q": torch.zeros(2, len(SUMMARY_NAMES), 2, 8, dtype=torch.float16),
        "probe_avg": torch.zeros(len(SUMMARY_NAMES), 2, 8, dtype=torch.float16),
        "coverage": {
            "n_probes_total": 3,
            "n_well_formed": 3,
            "n_captured": 2,
            "capture_drop_reasons": {},
        },
        "max_new_tokens": 8192,
        "rollout_digest": digest,
    }
    path = tmp_path / "c0.pt"
    torch.save(blob, path)
    run = dict(
        model_name="test-model",
        family="famA",
        rung="greedy",
        probe_pool_hash="h",
        capture_layers=[0, 1],
        summary_names=list(SUMMARY_NAMES),
        n_probes=3,
        hidden_size=8,
    )
    got, why = reusable_store_blob(path, "c0", max_new_tokens=8192, rollout_digest=digest, **run)
    assert got is not None and why == ""  # matching generation identity -> reusable

    # (i) changed generation cap at identical shapes / probe count -> recapture.
    got, why = reusable_store_blob(path, "c0", max_new_tokens=16384, rollout_digest=digest, **run)
    assert got is None and "max_new_tokens" in why

    # (ii) changed rollout CONTENT at the same probe count -> recapture.
    regen = rollout_content_digest(probes, [("<think>\nr\n</think>\n\nCHANGED", "stop")] * 3)
    got, why = reusable_store_blob(path, "c0", max_new_tokens=8192, rollout_digest=regen, **run)
    assert got is None and "rollout_digest" in why

    # (iii) pre-round-3 blob missing the identity fields -> recapture.
    legacy = {k: v for k, v in blob.items() if k not in ("max_new_tokens", "rollout_digest")}
    torch.save(legacy, path)
    got, why = reusable_store_blob(path, "c0", max_new_tokens=8192, rollout_digest=digest, **run)
    assert got is None and "mismatch" in why

    # (iv) probe_indices outside the RUN's 3-probe list -> recapture. The OLD
    # circular check (per_q.shape[0] vs the blob's OWN indices) passed this.
    torch.save(dict(blob, probe_indices=[0, 7]), path)
    got, why = reusable_store_blob(path, "c0", max_new_tokens=8192, rollout_digest=digest, **run)
    assert got is None and "probe_indices" in why


def test_identity_digest_keys_on_content_and_discards_stale_units(tmp_path):
    """Round-3 BLOCKER invariant (fails pre-fix — `long-loop-restartability-fit-capture`):
    two stores with IDENTICAL metadata + row counts but different per_q content
    (or different extractor rollout digests) must hash to different
    identity_digest() values, and the changed digest must make
    prepare_checkpoint_dir DISCARD the old partial units."""
    import issue928_fit_decomposition as fit_mod

    s_a = fit_mod.Store(_make_synth_store(tmp_path / "a"))
    s_b = fit_mod.Store(_make_synth_store(tmp_path / "b", content_seed=1))
    assert s_a.identity_digest() != s_b.identity_digest()  # per_q content differs

    # extractor-written rollout digests (round 3) take precedence: identical
    # tensors, different generation identity -> different store identity.
    s_c = fit_mod.Store(_make_synth_store(tmp_path / "c", rollout_digest="d1"))
    s_d = fit_mod.Store(_make_synth_store(tmp_path / "d", rollout_digest="d2"))
    assert s_c.identity_digest() != s_d.identity_digest()

    key_a = {"regime": "avg_q", "store_identity": s_a.identity_digest()}
    d1 = fit_mod.prepare_checkpoint_dir(tmp_path / "partial", "avg_q", key_a)
    (d1 / "layer_0.pt").write_bytes(b"unit")
    key_b = dict(key_a, store_identity=s_b.identity_digest())
    d2 = fit_mod.prepare_checkpoint_dir(tmp_path / "partial", "avg_q", key_b)
    assert not (d2 / "layer_0.pt").exists()  # stale units from the old store discarded


def test_group_mlp_matches_serial_references():
    """Round-6 (indiv-mlp-nonlinearity-control) parity pin: the GROUP-fold
    batched multihead MLP (`row_groups` + `standardization` extension of
    ``fit_batched_loco_mlp_multihead``) must reproduce a same-seed serial
    per-fold reference in BOTH standardization modes, and duplicate fits must
    be bitwise deterministic. Fails pre-fix (the extension did not exist);
    a future fold-leakage / standardization-mixing refactor fails here before
    it can ship a wrong nonlinearity-control headline. Reduced epochs for CI
    speed (measured deviations ~6e-7 at 40 epochs, ~1e-6 at production 300 —
    both far under the 5e-5 gate)."""
    from explore_persona_space.analysis.vectorized_mlp_skill import (
        assert_group_mlp_matches_serial,
    )

    devs = assert_group_mlp_matches_serial(max_epochs=40)
    assert devs, "MLP parity gate returned no checks"
    assert {k for k in devs if "full_data" in k}, "full_data mode not covered"
    assert {k for k in devs if "per_fold" in k}, "per_fold mode not covered"
    assert max(devs.values()) < 5e-5


def test_multihead_row_groups_none_matches_explicit_singletons():
    """``row_groups=None`` (the legacy byte-for-byte path) must equal explicit
    singleton labels EXACTLY — the None default is the pre-extension behavior,
    so any drift here breaks every existing multihead caller silently."""
    import numpy as np

    from explore_persona_space.analysis.vectorized_mlp_skill import (
        MLPGroup,
        fit_batched_loco_mlp_multihead,
    )

    rng = np.random.default_rng(7)
    cells = [
        MLPGroup(("cA", 0), rng.standard_normal((10, 6)), rng.standard_normal((10, 3))),
        MLPGroup(("cB", 1), rng.standard_normal((10, 6)), rng.standard_normal((10, 3))),
    ]
    kw = dict(seed=658, max_epochs=25, device="cpu", chunk_size=7)
    res_none = fit_batched_loco_mlp_multihead(cells, **kw)
    res_singl = fit_batched_loco_mlp_multihead(cells, row_groups=np.arange(10), **kw)
    for cell in cells:
        assert np.array_equal(res_none.preds_by_key[cell.key], res_singl.preds_by_key[cell.key])


def test_mlp_indiv_control_resume_skips_and_keys_on_identity(tmp_path, monkeypatch):
    """Round-6 restartability pin (the branch's r2/r3 standard, fails pre-fix):
    ``run_mlp_fits`` persists per-(arm, layer) durable units and a re-run with
    a matching manifest SKIPS them WITHOUT refitting while reproducing the
    exact same outputs; the manifest key carries the generation identity
    (store identity digest + pinned store revision + standardization + seed),
    and any key change DISCARDS stale units."""
    import issue928_mlp_indiv_control as drv
    import numpy as np
    from issue928_fit_decomposition import Store, prepare_checkpoint_dir

    fix = drv.build_synth_fixture(tmp_path / "fix")
    store = Store(fix["store"])
    decomp = drv.load_decomp(fix["decomp"])

    real_fit = drv.fit_batched_loco_mlp_multihead
    monkeypatch.setattr(
        drv,
        "fit_batched_loco_mlp_multihead",
        lambda *a, **k: real_fit(*a, **{**k, "max_epochs": 10}),
    )
    key = drv.fit_manifest_key(store, [25], "cpu", 64)
    for field in ("store_identity", "store_revision", "standardization", "seed", "mlp", "device"):
        assert field in key, field
    ckpt = prepare_checkpoint_dir(tmp_path / "partial", "mlp_indiv", key)
    units1, audits1 = drv.run_mlp_fits(store, [25], decomp, "cpu", 64, ckpt)
    assert audits1 and audits1[0]["d_ctx2ans"]["max_rel_dev"] <= 1e-9  # WARN fix (b) ran
    assert sorted(p.name for p in ckpt.glob("layer_*.pt")) == [
        "layer_25_mlp_d_ctx2ans.pt",
        "layer_25_mlp_g_aug.pt",
    ]

    def _boom(*_a, **_k):
        raise AssertionError("resume must not refit — completed units must be skipped")

    monkeypatch.setattr(drv, "fit_batched_loco_mlp_multihead", _boom)
    units2, audits2 = drv.run_mlp_fits(store, [25], decomp, "cpu", 64, ckpt)
    assert set(units2) == set(units1)
    for k2 in units1:
        assert np.array_equal(units1[k2]["preds"], units2[k2]["preds"])
        assert np.array_equal(units1[k2]["ss_res"], units2[k2]["ss_res"])
        assert np.array_equal(units1[k2]["ss_tot"], units2[k2]["ss_tot"])
    assert len(audits2) == 1  # one ss_tot audit per layer, resumed path included

    # A changed output-affecting key (standardization) DISCARDS stale units
    # (#722 r3: never silently reuse wrong cached rows).
    key2 = dict(key, standardization="per_fold")
    ckpt2 = prepare_checkpoint_dir(tmp_path / "partial", "mlp_indiv", key2)
    assert not list(ckpt2.glob("layer_*.pt"))
