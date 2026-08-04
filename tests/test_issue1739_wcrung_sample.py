"""Pins for the path-C fresh-WildChat sampler (#1739 "wcrung" rung).

Four invariants, each with a live-measured motivation:

1. **Natural units split at the LAST user turn**, discarding any trailing
   assistant reply — that reply is what the rung generates, so leaving it in
   would leak the answer into the prompt.
2. **The hold-out is content-based and actually excludes.** Measured live at
   n=200: 11 freshly-streamed conversations overlapped #1092's consumed text.
   The brief's ``prefix_conv_id`` mechanism would have admitted all 11 — those
   ids are run-local positional counters
   (``issue1092_build_corpus`` mints ``f"{source_tag}_{len(results):06d}"``),
   so they do not re-derive across runs.
3. **Duplicate final queries are deduped DURING the draw** (#1768). #1092's
   stream dedups on the FIRST user turn, which does not imply distinct FINAL
   turns; 3 exact duplicates appeared at n=200.
4. **An implausibly small exclusion set fails loud** rather than silently
   sampling against a near-empty hold-out.

All fixtures are synthetic; no network, no tokenizer download, no GPU.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

samp = pytest.importorskip("scripts.issue1739_wcrung_sample")


class _FakeTokenizer:
    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
        assert tokenize is False
        parts = [f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n" for m in messages]
        if add_generation_prompt:
            parts.append("<|im_start|>assistant\n")
        return "".join(parts)

    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": text.split()}


# ---------------------------------------------------------------- natural units


def test_natural_unit_splits_at_last_user_turn_and_drops_trailing_assistant():
    turns = [
        {"role": "user", "content": "first ask"},
        {"role": "assistant", "content": "first reply"},
        {"role": "user", "content": "second ask"},
        {"role": "assistant", "content": "SHOULD NOT APPEAR"},
    ]
    prefix, query = samp.natural_unit(turns)
    assert query == "second ask"
    assert [t["content"] for t in prefix] == ["first ask", "first reply"]
    assert all("SHOULD NOT APPEAR" not in t["content"] for t in prefix)


def test_natural_unit_single_turn_has_empty_prefix():
    prefix, query = samp.natural_unit([{"role": "user", "content": "only ask"}])
    assert prefix == []
    assert query == "only ask"


def test_natural_unit_returns_none_without_a_usable_user_turn():
    assert samp.natural_unit([{"role": "assistant", "content": "hi"}]) is None
    assert samp.natural_unit([{"role": "user", "content": "   "}]) is None
    assert samp.natural_unit([]) is None


# ------------------------------------------------------------- exclusion hashes


def _write_corpus(tmp_path: Path, *, prefixes, queries):
    d = tmp_path / "corpus"
    d.mkdir(parents=True, exist_ok=True)
    (d / "prefix_store.jsonl").write_text(
        "".join(json.dumps(p) + "\n" for p in prefixes), encoding="utf-8"
    )
    (d / "query_store.jsonl").write_text(
        "".join(json.dumps(q) + "\n" for q in queries), encoding="utf-8"
    )
    (d / "manifest.jsonl").write_text("", encoding="utf-8")
    return {
        "manifest.jsonl": d / "manifest.jsonl",
        "prefix_store.jsonl": d / "prefix_store.jsonl",
        "query_store.jsonl": d / "query_store.jsonl",
    }


def test_exclusion_set_covers_both_axes_and_normalizes(tmp_path):
    prefixes = [
        {
            "prefix_id": "p0",
            "prefix_turns": [
                {"role": "user", "content": "CONSUMED first turn"},
                {"role": "assistant", "content": "reply"},
            ],
            "natural_query": "CONSUMED natural query",
        }
    ]
    queries = [{"query_id": "q0", "text": "CONSUMED query text"}]
    corpus = _write_corpus(tmp_path, prefixes=prefixes, queries=queries)
    # `min_hashes=1` lowers the plausibility floor for this 3-hash unit check; the
    # floor itself is asserted separately below.
    exact, norm, digest = samp.build_exclusion_hashes(corpus, min_hashes=1)
    assert samp._h("CONSUMED first turn") in exact
    assert samp._h("CONSUMED natural query") in exact
    assert samp._h("CONSUMED query text") in exact
    # Only the FIRST user turn of a prefix is keyed (the builder's dedup notion).
    assert samp._h("reply") not in exact
    # Normalized variant catches whitespace/case reformatting.
    assert samp._h(samp._norm("  consumed   FIRST turn ")) in norm
    assert digest["contributions"]["prefix_first_user_turn"] == 1
    assert digest["contributions"]["prefix_natural_query"] == 1
    assert digest["contributions"]["query_text"] == 1


def test_exclusion_set_refuses_an_implausibly_small_holdout(tmp_path):
    corpus = _write_corpus(
        tmp_path,
        prefixes=[{"prefix_id": "p0", "prefix_turns": [], "natural_query": "only one"}],
        queries=[],
    )
    with pytest.raises(ValueError, match="exclusion set implausibly small"):
        samp.build_exclusion_hashes(corpus)


# ------------------------------------------------- hold-out + dedup integration


def _patch_pipeline(monkeypatch, tmp_path, convs, consumed_texts):
    """Point the sampler at a synthetic corpus + a synthetic stream."""
    corpus = _write_corpus(
        tmp_path,
        prefixes=[
            {"prefix_id": "p0", "prefix_turns": [], "natural_query": t} for t in consumed_texts
        ],
        queries=[],
    )
    import scripts.issue1092_build_corpus as builder
    import scripts.issue1739_wcrung_contexts as ctxmod

    monkeypatch.setattr(ctxmod, "stage_corpus", lambda _root: corpus)
    monkeypatch.setattr(builder, "_stream_with_cache", lambda *a, **k: list(convs), raising=True)
    # Bind the real function BEFORE patching — a lambda re-reading
    # samp.build_exclusion_hashes would resolve to itself and recurse.
    real_excl = samp.build_exclusion_hashes
    monkeypatch.setattr(samp, "build_exclusion_hashes", lambda c: real_excl(c, min_hashes=1))


def test_holdout_excludes_a_conversation_1092_already_consumed(tmp_path, monkeypatch):
    convs = [
        {
            "id": "wildchat_000001",
            "source": "wildchat",
            "turns": [{"role": "user", "content": "ALREADY CONSUMED"}],
        },
        {
            "id": "wildchat_000002",
            "source": "wildchat",
            "turns": [{"role": "user", "content": "fresh and unseen"}],
        },
    ]
    _patch_pipeline(monkeypatch, tmp_path, convs, consumed_texts=["ALREADY CONSUMED"])
    args = samp._parse_args(
        [
            "--n-contexts",
            "10",
            "--out-root",
            str(tmp_path / "o"),
            "--stream-cache-dir",
            str(tmp_path / "c"),
        ]
    )
    rows, digest = samp.sample_contexts(args, _FakeTokenizer())
    assert [r["query"] for r in rows] == ["fresh and unseen"]
    assert digest["drops"]["held_out_overlap_with_1092"] == 1


def test_duplicate_final_queries_are_deduped_during_the_draw(tmp_path, monkeypatch):
    """Two DIFFERENT conversations sharing a final user turn — the #1768 class.

    Their first user turns differ, so #1092's upstream first-turn dedup admits
    both; only a draw-time final-query dedup removes the second.
    """
    convs = [
        {
            "id": "wildchat_000001",
            "source": "wildchat",
            "turns": [
                {"role": "user", "content": "opening A"},
                {"role": "assistant", "content": "r"},
                {"role": "user", "content": "SHARED FINAL"},
            ],
        },
        {
            "id": "wildchat_000002",
            "source": "wildchat",
            "turns": [
                {"role": "user", "content": "opening B"},
                {"role": "assistant", "content": "r"},
                {"role": "user", "content": "SHARED FINAL"},
            ],
        },
    ]
    _patch_pipeline(monkeypatch, tmp_path, convs, consumed_texts=["unrelated"])
    args = samp._parse_args(
        [
            "--n-contexts",
            "10",
            "--out-root",
            str(tmp_path / "o"),
            "--stream-cache-dir",
            str(tmp_path / "c"),
        ]
    )
    rows, digest = samp.sample_contexts(args, _FakeTokenizer())
    assert len(rows) == 1
    assert digest["drops"]["duplicate_final_query_within_sample"] == 1
    assert len({r["query_sha256"] for r in rows}) == len(rows)


def test_single_turn_fraction_is_reported(tmp_path, monkeypatch):
    """The prefix-arm caveat is only disclosable if the digest carries it."""
    convs = [
        {"id": "w1", "source": "wildchat", "turns": [{"role": "user", "content": "solo one"}]},
        {
            "id": "w2",
            "source": "wildchat",
            "turns": [
                {"role": "user", "content": "open"},
                {"role": "assistant", "content": "r"},
                {"role": "user", "content": "multi final"},
            ],
        },
    ]
    _patch_pipeline(monkeypatch, tmp_path, convs, consumed_texts=["unrelated"])
    args = samp._parse_args(
        [
            "--n-contexts",
            "10",
            "--out-root",
            str(tmp_path / "o"),
            "--stream-cache-dir",
            str(tmp_path / "c"),
        ]
    )
    rows, digest = samp.sample_contexts(args, _FakeTokenizer())
    assert digest["n_single_turn"] == 1
    assert digest["single_turn_frac"] == pytest.approx(0.5)
    assert digest["n_prefix_turns_hist"] == {0: 1, 2: 1}
    assert {r["single_turn"] for r in rows} == {True, False}
