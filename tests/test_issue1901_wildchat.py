"""#1901 wildchat-target-battery round: unit pins for the w0-w3 phase machinery.

All fixtures are SYNTHETIC text (content hygiene: no real-corpus rows in tests).
Covers: _stream_corpus skip_first/resume semantics, the transposed candidate
gate's parity with the parent NearDupeGate, the contamination / yield-floor /
round-1-recovery gate branches (the degenerate-input probes the smoke's main
leg deliberately never trips), the smoke slice-arithmetic pin (csls floor), the
w1 Namespace field audit, mini-manifest roundtrip, and the hoisted heatmap
renderer.
"""

from __future__ import annotations

import importlib
import json
import random
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

MB = importlib.import_module("issue1901_metric_battery")
N1G = importlib.import_module("issue779_ffc_n1m_generate_capture")


def _cfg(tmp_path: Path, *, smoke: bool = True, force: bool = False) -> MB.Cfg:
    return MB.Cfg(
        phase="w0_wc_candidates",
        staging_root=tmp_path,
        smoke=smoke,
        revision="testrev",
        seed=7,
        force=force,
    )


def _row(text: str) -> dict:
    return {"conversation": [{"content": text}]}


# ── _stream_corpus: skip_first + resume-cursor precedence ────────────────────────


def test_stream_corpus_skip_first_skips_prefix(tmp_path):
    rows = [_row(f"synthetic prompt number {i}") for i in range(10)]
    kept = N1G._stream_corpus(
        "fake/repo",
        "t1",
        lambda p: True,
        3,
        tmp_path / "cache",
        {"fp": 1},
        resume=False,
        smoke_stream=rows,
        skip_first=4,
    )
    assert [r["prompt"] for r in kept] == [f"synthetic prompt number {i}" for i in (4, 5, 6)]
    # stream_pos is the GLOBAL consumed position (skip counted)
    assert [r["stream_pos"] for r in kept] == [4, 5, 6]


def test_stream_corpus_resume_cursor_takes_precedence_over_skip_first(tmp_path):
    rows = [_row(f"synthetic prompt number {i}") for i in range(12)]
    cache = tmp_path / "cache"
    cache.mkdir(parents=True)
    fp = {"fp": 2}
    # hand-write a PARTIAL checkpoint (complete=False, consumed=2): the resume
    # cursor must take precedence over skip_first (docstring contract)
    N1G._atomic_write_jsonl(
        cache / "t2.jsonl",
        [
            {"prompt": "synthetic prompt number 0", "corpus": "t2", "stream_pos": 0},
            {"prompt": "synthetic prompt number 1", "corpus": "t2", "stream_pos": 1},
        ],
    )
    N1G._atomic_write_json(
        cache / "t2.meta.json",
        {"fingerprint": fp, "consumed": 2, "kept": 2, "complete": False},
    )
    more = N1G._stream_corpus(
        "fake/repo",
        "t2",
        lambda p: True,
        4,
        cache,
        fp,
        resume=True,
        smoke_stream=rows,
        skip_first=9,
    )
    assert [r["stream_pos"] for r in more] == [0, 1, 2, 3]


def test_stream_corpus_default_kwargs_preserve_parent_behavior(tmp_path):
    rows = [_row(f"synthetic prompt number {i}") for i in range(5)]
    kept = N1G._stream_corpus(
        "fake/repo",
        "t3",
        lambda p: True,
        2,
        tmp_path / "c",
        {"fp": 3},
        resume=False,
        smoke_stream=rows,
    )
    assert [r["stream_pos"] for r in kept] == [0, 1]


# ── transposed candidate gate ────────────────────────────────────────────────────


def test_candidate_gate_matching_targets_parity_with_is_dupe():
    candidates = [
        "the quick brown fox jumps over the lazy dog near the river bank today",
        "a completely different candidate sentence about gardening and tomato plants",
        "SHORT",
    ]
    train_rows = [
        "The Quick brown fox jumps over the lazy dog near the river bank today",  # exact-norm 0
        "the quick brown fox jumps over the lazy dog near the river bank yesterday",  # near 0
        "unrelated sentence about astrophysics and neutron star mergers entirely",
        "short",  # exact-norm match of candidate 2
    ]
    gate = MB._CandidateGate(candidates)
    matched: set[int] = set()
    for t in train_rows:
        matched |= gate.matching_targets(t)
    assert 0 in matched and 2 in matched and 1 not in matched
    # parity with the parent direction: a train row that matches candidate i is
    # exactly a row the PARENT gate (targets=candidates) calls a dupe
    parent = N1G.NearDupeGate(candidates)
    for t in train_rows:
        assert bool(MB._CandidateGate(candidates).matching_targets(t)) == parent.is_dupe(t)


# ── gate branches (degenerate-input probes) ──────────────────────────────────────


def _write_exclusion(cfg, hexes: list[str]) -> None:
    MB._atomic_npz(cfg.wc_exclusion_npz, fps=np.array(sorted(hexes), dtype="S40"))


def test_contamination_check_raises_on_heldout_hit(tmp_path):
    cfg = _cfg(tmp_path)
    sha_hit = MB._sha1_norm("a synthetic contaminated prompt")
    _write_exclusion(cfg, [sha_hit, MB._sha1_norm("other")])
    with pytest.raises(RuntimeError, match="KILL \\(contamination\\)"):
        MB._wc_contamination_check(cfg, [sha_hit, MB._sha1_norm("fresh")], expect_hits=False)
    # in-train targets EXPECT hits: informational, never raises
    info = MB._wc_contamination_check(cfg, [sha_hit], expect_hits=True)
    assert info["n_exclusion_hits"] == 1


def test_contamination_check_passes_on_clean_heldout(tmp_path):
    cfg = _cfg(tmp_path)
    _write_exclusion(cfg, [MB._sha1_norm("train row a"), MB._sha1_norm("train row b")])
    info = MB._wc_contamination_check(
        cfg, [MB._sha1_norm("fresh row 1"), MB._sha1_norm("fresh row 2")], expect_hits=False
    )
    assert info["n_exclusion_hits"] == 0


def test_run_wc_battery_yield_floor_raises_before_any_staging(tmp_path):
    cfg = _cfg(tmp_path, smoke=False)  # production floor = 500
    tgt = {
        "X": np.zeros((3, 4), dtype=np.float32),
        "Y": np.zeros((3, 4), dtype=np.float32),
        "shas": ["a", "b", "c"],
        "in_train": False,
        "source": "unit",
    }
    with pytest.raises(RuntimeError, match="KILL \\(yield floor\\)"):
        MB._run_wc_battery(
            cfg,
            tgt,
            tmp_path / "out.json",
            tmp_path / "draws.json",
            repro_control={},
            bundle=None,
            seed_offset=0,
            label="unit",
        )


def test_round1_prompts_raises_on_sha_drift(tmp_path, monkeypatch):
    cfg = _cfg(tmp_path)

    def fake_sample(skip, n10k, n50k, **kw):
        assert (skip, n10k, n50k) == (N1G.N_ROUND1, 0, 0)
        return {"round1": ["synthetic r1"], "round1_prompt_sha256": "deadbeef"}

    monkeypatch.setattr(MB.N50G, "sample_disjoint_n50k", fake_sample)
    with pytest.raises(RuntimeError, match="KILL \\(round-1 recovery\\)"):
        MB._round1_prompts(cfg, expected_sha="not-deadbeef")


def test_round1_prompts_cache_roundtrip(tmp_path, monkeypatch):
    cfg = _cfg(tmp_path)
    prompts = ["synthetic r1 a", "synthetic r1 b"]
    sha = N1G.N10._sha_prompts(prompts)
    monkeypatch.setattr(
        MB.N50G,
        "sample_disjoint_n50k",
        lambda *a, **k: {"round1": list(prompts), "round1_prompt_sha256": sha},
    )
    assert MB._round1_prompts(cfg, sha) == prompts
    # second call served from the sha-verified cache (no sampler needed)
    monkeypatch.setattr(
        MB.N50G,
        "sample_disjoint_n50k",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("cache should have served")),
    )
    assert MB._round1_prompts(cfg, sha) == prompts


def test_wc_kill_check_demotes_at_smoke_only(tmp_path):
    def boom():
        raise RuntimeError("KILL (unit)")

    MB._wc_kill_check(_cfg(tmp_path, smoke=True), boom)  # demoted: no raise
    with pytest.raises(RuntimeError, match="KILL \\(unit\\)"):
        MB._wc_kill_check(_cfg(tmp_path, smoke=False), boom)


def test_wc_resume_skip_regime_mismatch_raises(tmp_path):
    cfg = _cfg(tmp_path)
    out = cfg.wc_w0_sentinel
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"metadata": {"regime": cfg.wc_regime()}}))
    assert MB._wc_resume_skip(cfg, out, "w0") is not None
    other = json.loads(out.read_text())
    other["metadata"]["regime"]["seed"] = 999
    out.write_text(json.dumps(other))
    with pytest.raises(RuntimeError, match="DIFFERENT wc regime"):
        MB._wc_resume_skip(cfg, out, "w0")


def test_screen_budget_abort_fires_on_projected_overrun(tmp_path, monkeypatch):
    cfg = _cfg(tmp_path)
    monkeypatch.setattr(MB, "WC_SCREEN_PILOT_ROWS", 2)
    monkeypatch.setattr(MB, "WC_SCREEN_BUDGET_S", 1e-12)
    pool = [{"prompt": f"synthetic train row {i} with enough text to ngram"} for i in range(6)]
    with pytest.raises(RuntimeError, match="transposed screen projected"):
        MB._wc_screen_candidates(cfg, ["a fresh synthetic candidate sentence"], pool, [])


def test_phase_w1_yield_floor_raises(tmp_path, monkeypatch):
    cfg = _cfg(tmp_path, smoke=False)
    monkeypatch.setattr(MB, "_stage_wc_manifest", lambda c, d: d)
    monkeypatch.setattr(MB.N1G, "run_capture", lambda ns: 0)
    monkeypatch.setattr(MB, "_wc_captured_rowcount", lambda c: (3, 1))
    with pytest.raises(RuntimeError, match="KILL \\(yield floor\\)"):
        MB.phase_w1(cfg)


def test_wc_capture_targets_raises_when_w1_never_ran(tmp_path, monkeypatch):
    # production-only loader (smoke uses the parent-chunk stand-in): its
    # no-chunks entry branch must fail loud naming the missing phase
    cfg = _cfg(tmp_path, smoke=False)
    monkeypatch.setattr(MB.N50G, "_remote_index", lambda prefix: {})
    with pytest.raises(RuntimeError, match="run w1 first"):
        MB._wc_capture_targets(cfg)


# ── smoke slice-arithmetic pin (csls floor) ─────────────────────────────────────


def test_smoke_n_targets_clears_csls_floor():
    # csls_scores asserts 0 < k < n_pool AND k <= n_query at K_CSLS=10; the
    # smoke targets-only pool has n_pool == n_query == WC_SMOKE_N_TARGETS
    assert MB.WC_SMOKE_N_TARGETS >= MB.K_CSLS + 2
    rng = np.random.default_rng(0)
    S = rng.standard_normal((MB.WC_SMOKE_N_TARGETS, MB.WC_SMOKE_N_TARGETS))
    MB.csls_scores(S, MB.K_CSLS)  # must not assert


# ── w1 Namespace field audit ─────────────────────────────────────────────────────


def test_w1_namespace_covers_all_rig_arg_reads(tmp_path):
    cfg = _cfg(tmp_path)
    ns = MB._w1_namespace(cfg, device="cpu", no_upload=True, out_dir=tmp_path / "w1")
    reads = MB._w1_args_attr_reads()
    assert reads, "AST audit found no args.<attr> reads — audit is broken"
    assert reads <= set(vars(ns))
    # the audit actually bites: removing a read attr trips the assert
    import argparse

    broken = argparse.Namespace(**{k: v for k, v in vars(ns).items() if k != "hf_prefix"})
    missing = MB._w1_args_attr_reads() - set(vars(broken))
    assert "hf_prefix" in missing


def test_w1_namespace_hf_prefix_is_round_root(tmp_path):
    cfg = _cfg(tmp_path, smoke=False)
    ns = MB._w1_namespace(cfg, device="cpu", no_upload=True, out_dir=tmp_path / "w1")
    # ROUND ROOT semantics (#1776): N1G appends final_token_capture/ itself
    assert ns.hf_prefix == MB.WC_HF_ROOT
    assert not ns.hf_prefix.endswith("final_token_capture")


# ── mini-manifest roundtrip ──────────────────────────────────────────────────────


def test_wc_mini_manifest_roundtrip(tmp_path):
    rows = [
        {"prompt": f"synthetic wc prompt {i}", "corpus": "wildchat", "stream_pos": 100 + i, "i": i}
        for i in range(5)
    ]
    meta = {"n_new": 5, "n_lmsys": 0, "n_wildchat": 5}
    n_parts = N1G._write_manifest_parts(tmp_path / "manifest", rows, meta)
    assert n_parts == 1
    assert N1G._manifest_complete_locally(tmp_path / "manifest")
    pool, got_meta = N1G.read_manifest_pool(tmp_path / "manifest")
    assert [r["i"] for r in pool] == list(range(5))
    assert got_meta["n_new"] == 5 and got_meta["n_parts"] == 1


# ── transfer comparison (synthetic banked file) ──────────────────────────────────


def _mk_arm(r2: float, acc1: float) -> dict:
    return {
        "label": "unit",
        "r2": {"point": r2},
        "mean_cosine": {"point": r2 / 2},
        "retrieval": {
            "test": {
                "euclidean": {"acc_at_k": {"1": acc1}, "mrr": acc1, "median_rank": 1.0},
                "cosine": {"acc_at_k": {"1": acc1}},
                "csls": {"acc_at_k": {"1": acc1}},
            }
        },
    }


def test_transfer_comparison_tau_and_inversions(tmp_path, monkeypatch):
    cfg = _cfg(tmp_path)
    banked = {
        "per_layer": {"19": {"arms": {"ridge": _mk_arm(0.7, 0.9), "const_mean": _mk_arm(0.0, 0.1)}}}
    }
    banked_p = tmp_path / "context_arm.json"
    banked_p.write_text(json.dumps(banked))
    monkeypatch.setattr(MB, "BANKED_CONTEXT_ARM", banked_p)
    monkeypatch.setattr(MB, "BANKED_CONTEXT_DRAWS", tmp_path / "absent.json")
    wc_arms = {
        "ridge": _mk_arm(0.1, 0.2),
        "const_mean": _mk_arm(0.5, 0.8),
        "identity_copy": _mk_arm(0.2, 0.3),
    }
    with pytest.raises(AssertionError, match="too few common arms"):
        MB._transfer_comparison(cfg, {"ridge": wc_arms["ridge"]}, {}, {})
    banked["per_layer"]["19"]["arms"]["identity_copy"] = _mk_arm(0.3, 0.5)
    banked_p.write_text(json.dumps(banked))
    out = MB._transfer_comparison(cfg, wc_arms, {}, {"n": 3})
    # ridge>const on lmsys, const>ridge on wc -> full rank reversal on r2
    assert out["kendall_tau"]["r2"]["tau"] < 0
    pairs = {tuple(e["pair"]) for e in out["pairwise_inversions"]["r2"]}
    assert ("ridge", "const_mean") in pairs or ("const_mean", "ridge") in pairs
    assert out["setup"]["common_arms"] == ["ridge", "const_mean", "identity_copy"]


# ── hoisted heatmap renderer (p4 refactor pin) ───────────────────────────────────


def test_ladder_heatmap_renders_on_synthetic_arms():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    MB._ladder_heatmap(ax, {"a": _mk_arm(0.5, 0.6), "b": _mk_arm(0.1, 0.2)}, "test", "unit")
    assert len(ax.images) == 1 and ax.images[0].get_array().shape == (2, len(MB.HERO_METRICS))
    plt.close(fig)


# ── w0 screen vectorization: equivalence gate (serial reference vs fast path) ────


def test_screen_fast_path_equivalence_gate(tmp_path, monkeypatch):
    """EQUIVALENCE GATE (vectorize-rule item 6; #1901 w0 vectorize-fix round):
    the vectorized screen must produce the IDENTICAL survivor set to the
    pre-vectorization serial implementation on a seeded slice shaped like the
    production inputs (2 manifest parts + the round-1 block). Synthetic text
    only (content hygiene) with engineered exact / near-dupe / short / empty /
    duplicate-candidate edge rows in every block."""
    rng = random.Random(1901)
    words = [f"tok{i:03d}" for i in range(120)]

    def sent(k: int = 18) -> str:
        return " ".join(rng.choice(words) for _ in range(k))

    candidates = [sent() for _ in range(40)]
    candidates[7] = ""  # empty candidate text (no ngrams; unreachable via Jaccard)
    candidates[11] = "tiny"  # shorter than the 5-gram window
    candidates[13] = candidates[12]  # duplicate candidate content (both must drop)

    part1 = [sent() for _ in range(60)]
    part1[3] = candidates[5].upper()  # exact-normalized match
    part1[9] = candidates[19] + " tail"  # near-dupe (Jaccard >= 0.8)
    part1[12] = "tiny"  # exact match of the short candidate
    part2 = [sent() for _ in range(60)]
    part2[4] = candidates[12]  # exact match hitting BOTH duplicates 12 + 13
    part2[8] = candidates[30][:-8]  # near-dupe by truncation
    part2[9] = ""  # empty train row
    round1_block = [sent() for _ in range(25)]
    round1_block[5] = candidates[22] + "!"  # near-dupe raised from the round-1 block
    pool = [{"prompt": p} for p in part1 + part2]
    all_texts = part1 + part2 + round1_block

    gate = MB._CandidateGate(candidates)
    # per-row parity: the fast path returns the serial reference's exact hits
    for t in all_texts:
        assert gate.matching_targets(t) == gate.matching_targets_serial_reference(t)

    # full screen (pilot leg + parallel fan-out, 2 workers) vs the serial oracle
    ref_matched: set[int] = set()
    for t in all_texts:
        ref_matched |= gate.matching_targets_serial_reference(t)
    monkeypatch.setattr(MB, "WC_SCREEN_PILOT_ROWS", 10)  # force pilot + fan-out legs
    survivors, stats = MB._wc_screen_candidates(
        _cfg(tmp_path), candidates, pool, round1_block, workers=2
    )
    assert set(survivors) == set(range(len(candidates))) - ref_matched
    assert stats["n_matched_dropped"] == len(ref_matched)
    assert stats["n_survivors"] == len(survivors)
    assert stats["workers"] == 2
    # the engineered drops from each block actually fired (gate not vacuous);
    # 7 drops too: the empty train row exact-norm-matches the empty candidate
    # (the Jaccard path can never reach it — no ngrams — but exact-idx does)
    assert {5, 7, 11, 12, 13, 19, 22, 30} <= ref_matched

    # serial in-process leg (workers=1) agrees too
    survivors_serial, _ = MB._wc_screen_candidates(
        _cfg(tmp_path), candidates, pool, round1_block, workers=1
    )
    assert survivors_serial == survivors
