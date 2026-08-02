"""Unit pins for the #1336 v2 corpus builder's PURE helpers (no network).

Covers (Unit A of the full-corpora-stage-evals-metric-ladder round):
  - decontamination normalizer (lowercase + whitespace-collapse + sha256)
  - exact-sha256 dedup (keep-first)
  - seed-1336 deterministic sampling (reproducible, sorted, unique)
  - SFT stratification quotas (3,667/3,667/3,666 == 11,000)
  - first-user-turn extraction (tulu ``messages`` + the issue779 lmsys
    predicate parity: conversation[0], no role check, stripped)
  - the shared build-filter chain incl. the run_prep ``n_tok + 1 > budget``
    BOS-margin arithmetic (fake tokenizer — network-free)
  - LmsysAccumulator: byte-equality assert (prompt text NEVER in the error),
    running-prefix dedup, exclusion-join fallback, checkpoint restore
  - manifest entry schema (n_built / n_dropped_by_filter / n_dropped_decon /
    sha256 / source_revision)
  - upload-side text shard split + sha-verified reassembly round-trip
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
for p in (str(REPO / "scripts"), str(REPO / "src")):
    if p not in sys.path:
        sys.path.insert(0, p)

import issue1336_stage_corpora as sc  # noqa: E402


class FakeTok:
    """Whitespace tokenizer stub: one token per whitespace word (the tulu
    render contributes its two role-header 'words')."""

    def __call__(self, texts, add_special_tokens=False):
        assert add_special_tokens is False
        return {"input_ids": [[0] * len(t.split()) for t in texts]}


# ── decon normalizer ─────────────────────────────────────────────────────────
def test_decon_key_normalizes_case_and_whitespace():
    assert sc.decon_key("The  Answer\tIS 42\n") == sc.decon_key("the answer is 42")
    assert sc.decon_key("a question") != sc.decon_key("another question")


# ── dedup ────────────────────────────────────────────────────────────────────
def test_dedup_keep_first():
    assert sc.dedup_keep_first(["a", "b", "a", "c", "b"]) == [0, 1, 3]
    assert sc.dedup_keep_first([]) == []


# ── seeded sampling ──────────────────────────────────────────────────────────
def test_seeded_sample_deterministic_sorted_unique():
    a = sc.seeded_sample_indices(1000, 100)
    b = sc.seeded_sample_indices(1000, 100)
    assert a == b, "seed-1336 sample must be reproducible across calls"
    assert a == sorted(set(a)), "sample must be sorted + without replacement"
    assert len(a) == 100 and all(0 <= i < 1000 for i in a)
    assert sc.seeded_sample_indices(5, 9) == list(range(5)), "k >= n returns the whole range"
    assert sc.seeded_sample_indices(1000, 100, seed=7) != a, "seed must matter"


def test_sft_quotas_match_plan():
    assert sum(sc.SFT_QUOTAS.values()) == 11_000
    assert sorted(sc.SFT_QUOTAS.values(), reverse=True) == [3667, 3667, 3666]
    assert set(sc.SFT_QUOTAS) == set(sc.SFT_SOURCE_COUNTS)


# ── first-user-turn extraction ───────────────────────────────────────────────
def test_first_user_turn_extraction():
    msgs = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "  hello  "},
        {"role": "user", "content": "second"},
    ]
    assert sc.first_user_turn(msgs) == "hello"
    assert sc.first_user_turn([{"role": "assistant", "content": "x"}]) is None
    assert sc.first_user_turn([{"role": "user", "content": "   "}]) is None
    assert sc.first_user_turn(None) is None


def test_lmsys_first_turn_predicate_parity():
    # issue779_collect.load_train_contexts parity: conversation[0] content,
    # NO role check, content-or-value key, kept stripped.
    assert sc.lmsys_first_turn({"conversation": [{"role": "assistant", "content": " hi "}]}) == (
        "hi"
    )
    assert sc.lmsys_first_turn({"conversation": [{"value": "v"}]}) == "v"
    assert sc.lmsys_first_turn({"conversation": [{"content": ""}]}) is None
    assert sc.lmsys_first_turn({"conversation": []}) is None
    assert sc.lmsys_first_turn({}) is None


# ── build-filter chain ───────────────────────────────────────────────────────
def test_apply_build_filters_order_and_counts():
    rows = [
        {"prompt": "keep me"},
        {"prompt": "   "},
        {"prompt": "Contaminated Q"},
        {"prompt": "w1 w2 w3 w4 w5 w6 w7 w8 w9 w10"},
    ]
    kept, drops = sc.apply_build_filters(
        rows, tok=FakeTok(), decon_keys={sc.decon_key("contaminated  q")}, budget=8
    )
    assert [r["prompt"] for r in kept] == ["keep me"]
    assert drops == {"empty": 1, "decon": 1, "over_budget": 1}


def test_apply_build_filters_decon_exempt_keeps_reference_rows():
    rows = [{"prompt": "Contaminated Q"}]
    kept, drops = sc.apply_build_filters(
        rows,
        tok=FakeTok(),
        decon_keys={sc.decon_key("contaminated q")},
        decon_exempt=True,
        budget=100,
    )
    assert len(kept) == 1 and drops["decon"] == 0


def test_over_budget_plus_one_bos_parity():
    # tulu render adds 2 header 'words' under FakeTok; run_prep arithmetic is
    # n_tok + 1 > budget (the +1 BOS margin). "a b" -> 4 tok + 1 = 5, not > 5.
    assert sc.over_budget_flags(["a b"], FakeTok(), budget=5) == [False]
    assert sc.over_budget_flags(["a b c"], FakeTok(), budget=5) == [True]


# ── LmsysAccumulator ─────────────────────────────────────────────────────────
def _acc(track, **kw):
    defaults = dict(
        prefix_n=len(track),
        new_n=2,
        fallback=False,
        is_over_budget=lambda p: False,
        decon_keys=set(),
    )
    defaults.update(kw)
    return sc.LmsysAccumulator(track, **defaults)


def test_lmsys_prefix_byte_assert_raises_without_prompt_text():
    acc = _acc(["p0", "p1"])
    assert acc.offer("p0", 0) == "prefix"
    with pytest.raises(RuntimeError, match="byte-equality FAILED") as exc:
        acc.offer("SOME-DIVERGED-USER-TEXT", 1)
    # content hygiene: the error names position + shas, never prompt text
    assert "SOME-DIVERGED-USER-TEXT" not in str(exc.value)
    assert "p1" not in str(exc.value)


def test_lmsys_extension_dedups_against_running_prefix():
    acc = _acc(["p0", "p1"], new_n=3)
    assert acc.offer("p0", 0) == "prefix"
    assert acc.offer("p1", 1) == "prefix"
    assert acc.offer("p0", 2) == "dup"  # track_s membership
    assert acc.offer("new1", 3) == "kept"
    assert acc.offer("new1", 4) == "dup"  # earlier extension row
    assert acc.rejects["dup"] == 2
    assert not acc.done
    assert acc.offer("new2", 5) == "kept"
    assert acc.offer("new3", 6) == "kept"
    assert acc.done
    assert acc.offer("late", 7) == "ignored"


def test_lmsys_fallback_mode_is_exclusion_join():
    acc = _acc(["p0", "p1"], fallback=True, new_n=1)
    # no positional assert; track_s prompts are excluded by sha
    assert acc.offer("p0", 0) == "dup"
    assert acc.offer("fresh", 1) == "kept"
    assert acc.done and acc.n_prefix == 0


def test_lmsys_budget_and_decon_rejects():
    acc = _acc(
        ["p0"],
        new_n=2,
        is_over_budget=lambda p: p == "big",
        decon_keys={sc.decon_key("Contaminated  QUESTION")},
    )
    assert acc.offer("p0", 0) == "prefix"
    assert acc.offer("contaminated question", 1) == "decon"
    assert acc.offer("big", 2) == "over_budget"
    assert acc.offer("ok", 3) == "kept"
    assert acc.rejects == {"dup": 0, "decon": 1, "over_budget": 1}


def test_lmsys_restore_roundtrip():
    pool = [
        {"kind": "prefix", "scan_index": 0, "prompt": "p0"},
        {"kind": "ext", "scan_index": 5, "prompt": "x"},
    ]
    acc = _acc(["p0"], new_n=2)
    acc.restore(pool)
    assert acc.n_prefix == 1 and len(acc.ext_rows) == 1
    assert acc.offer("x", 9) == "dup"  # restored ext row re-entered the sha set


def test_lmsys_restore_rejects_diverged_pool():
    acc = _acc(["p0"], new_n=1)
    with pytest.raises(AssertionError, match="corrupt cache"):
        acc.restore([{"kind": "prefix", "scan_index": 0, "prompt": "NOT-P0"}])


# ── manifest schema ──────────────────────────────────────────────────────────
def test_corpus_meta_schema():
    entry = sc.corpus_meta(
        "toy",
        n_built=3,
        n_dropped_by_filter={"dup": 1, "over_budget": 0},
        n_dropped_decon=2,
        sha256="ab" * 32,
        source_revision="deadbeef1234",
        fingerprint={"recipe_version": sc.FILTER_RECIPE_VERSION},
        extra={"component": "MATH"},
    )
    for key in (
        "corpus",
        "n_built",
        "n_dropped_by_filter",
        "n_dropped_decon",
        "sha256",
        "source_revision",
        "fingerprint",
    ):
        assert key in entry, f"manifest entry missing required key {key}"
    assert entry["component"] == "MATH"
    assert entry["n_dropped_by_filter"] == {"dup": 1, "over_budget": 0}


# ── shard split + reassembly ─────────────────────────────────────────────────
def test_split_and_reassemble_roundtrip(tmp_path):
    rows = [{"prompt_idx": i, "prompt": f"row {i} " + "x" * 50} for i in range(40)]
    p = tmp_path / "toy.jsonl"
    sc._write_jsonl(p, rows)
    assert sc.split_corpus_for_upload(p, threshold=500, shard_max=400) is True
    man = json.loads((tmp_path / "toy.manifest.json").read_text())
    assert len(man["parts"]) >= 2
    assert sum(man["line_counts"]) == 40
    p.unlink()  # consumer sees shards + manifest only (the HF-staged shape)
    rows2 = sc.read_corpus_rows_local(tmp_path, "toy")
    assert rows2 == rows
    assert (tmp_path / "toy.jsonl").exists(), "reassembly persists the single file"


def test_split_noop_under_threshold(tmp_path):
    rows = [{"prompt_idx": 0, "prompt": "tiny"}]
    p = tmp_path / "toy.jsonl"
    sc._write_jsonl(p, rows)
    assert sc.split_corpus_for_upload(p, threshold=10_000, shard_max=9_000) is False
    assert list(tmp_path.glob("toy.shard*.jsonl")) == []
    assert not (tmp_path / "toy.manifest.json").exists()


def test_reassemble_rejects_tampered_shard(tmp_path):
    rows = [{"prompt_idx": i, "prompt": "y" * 80} for i in range(20)]
    p = tmp_path / "toy.jsonl"
    sc._write_jsonl(p, rows)
    assert sc.split_corpus_for_upload(p, threshold=200, shard_max=200) is True
    p.unlink()
    shard = sorted(tmp_path.glob("toy.shard*.jsonl"))[0]
    shard.write_bytes(shard.read_bytes().replace(b"y", b"z", 1))
    with pytest.raises(AssertionError, match="sha256"):
        sc.read_corpus_rows_local(tmp_path, "toy")


# ── registry sanity ──────────────────────────────────────────────────────────
def test_v2_registry_shape():
    assert set(sc.V2_CORPORA) == {
        "lmsys23k",
        "gsm8k_train_full",
        "gsm8k_test1319",
        "math7500",
        "if11k",
        "uf11k",
        "sft11k",
    }
    assert sc.V2_CORPORA["lmsys23k"]["formats"] == ("chat", "naturalistic")
    for slug, spec in sc.V2_CORPORA.items():
        if slug != "lmsys23k":
            assert spec["formats"] == ("chat",)
    assert set(sc.BUILD_ORDER) == set(sc.V2_CORPORA)
    assert sc.V2_CORPORA["lmsys23k"]["n_target"] == sc.LMSYS_PREFIX_N + sc.LMSYS_NEW_N
