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


def _make_synth_store(tmp_path):
    """Tiny synthetic per-(C, q) summary store (4 contexts x 2 rows x 2 layers x H=8)."""
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
    g = torch.Generator().manual_seed(0)
    for c in ctx:
        per_q = torch.randn(2, len(SUMMARY_NAMES), 2, 8, generator=g)
        torch.save(
            {"context_id": c, "per_q": per_q, "probe_avg": per_q.mean(0)},
            store_dir / "percq_summaries" / f"{c}.pt",
        )
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
