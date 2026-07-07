"""Issue #1092 round-8 pins — P0 step-4+ crash-fix hardening.

Covers the attempt-3 production crash class (2026-07-07, `KeyError: 'turns'`
at issue1092_build_corpus.py:550 after a 3h06m stream):

1. `_topic_input_text` — label-input extraction from PREFIX entries (keyed
   ``prefix_turns`` + ``natural_query``, never ``turns``), incl. the EMPTY-
   prefix (single-turn conversation) fallback.
2. `_label_topic_batch` — takes pre-extracted TEXTS (no dict-shape assumption);
   body executed against a signature-conformant fake Anthropic client at the
   API boundary.
3. `_stream_with_cache` — stream-pool checkpoint round-trip: matching
   fingerprint loads from cache (no re-stream), mismatched fingerprint or
   ``resume=False`` re-streams; U+2028 content survives the JSONL round-trip
   (text-mode iteration, never ``.splitlines()``; #825/#950).
4. `_build_query_bank(label_texts=...)` — bank candidates are labeled BEFORE
   the topic-stratified subsample; the top-up keeps the bank at ``n_target``
   under a concentrated real-label distribution (the pre-round-8 all-"other"
   collapse produced a ~n_target/12 bank — a latent G1 floor crash).
5. Render helpers on an EMPTY prefix (bare-context row).
"""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
for _p in (REPO / "src", REPO / "scripts"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import issue1092_build_corpus as bc  # noqa: E402

# ── fixtures ──────────────────────────────────────────────────────────────────


def _prefix_entry(*, n_prefix_turns: int = 2, natural_query: str = "What next?") -> dict:
    """A prefix entry in the exact shape `_sample_prefixes` emits."""
    turns = []
    for i in range(n_prefix_turns):
        role = "user" if i % 2 == 0 else "assistant"
        turns.append({"role": role, "content": f"{role} prefix turn {i} about sorting lists."})
    return {
        "prefix_id": f"pfx_{n_prefix_turns:05d}",
        "conv_id": "conv_x",
        "source": "wildchat",
        "prefix_turns": turns,
        "natural_query": natural_query,
        "n_user_turns": max(1, (n_prefix_turns + 1) // 2),
        "total_tokens": 42,
        "topic": "other",
    }


def _wc_row(i: int, *, user_text: str | None = None) -> dict:
    """Minimal WildChat-shaped row covering every field the filters touch."""
    text = user_text if user_text is not None else f"How do I bake bread? (variant {i})"
    return {
        "conversation": [
            {"role": "user", "content": text, "redacted": False, "toxic": False},
            {
                "role": "assistant",
                "content": f"Here is how you bake bread, answer {i}.",
                "redacted": False,
                "toxic": False,
            },
        ],
        "language": "English",
        "redacted": False,
        "toxic": False,
        "openai_moderation": [{"categories": {}, "category_scores": {}, "flagged": False}],
    }


class _FakeContentBlock:
    def __init__(self, text: str):
        self.text = text


class _FakeResponse:
    def __init__(self, text: str):
        self.content = [_FakeContentBlock(text)]


class _FakeMessages:
    """Signature-conformant fake of `anthropic.Anthropic().messages` (the used
    surface: keyword-only model / max_tokens / messages)."""

    def __init__(self, replies: list[str]):
        self._replies = list(replies)
        self.calls: list[dict] = []

    def create(self, *, model: str, max_tokens: int, messages: list[dict]) -> _FakeResponse:
        self.calls.append({"model": model, "max_tokens": max_tokens, "messages": messages})
        return _FakeResponse(self._replies[len(self.calls) - 1])


class _FakeAnthropicClient:
    def __init__(self, replies: list[str]):
        self.messages = _FakeMessages(replies)


# ── 1. _topic_input_text ──────────────────────────────────────────────────────


def test_topic_input_text_normal_prefix_uses_first_user_turn():
    entry = _prefix_entry(n_prefix_turns=4)
    entry["prefix_turns"][0]["content"] = "Explain quicksort. " + "x" * 600
    text = bc._topic_input_text(entry)
    assert text.startswith("Explain quicksort. ")
    assert len(text) == 500  # truncated for context economy


def test_topic_input_text_empty_prefix_falls_back_to_natural_query():
    # single-turn conversation: prefix is empty (bare context) — the latent
    # sibling of the attempt-3 crash (a key rename alone would IndexError).
    entry = _prefix_entry(n_prefix_turns=0, natural_query="Translate 'hello' to French.")
    assert entry["prefix_turns"] == []
    assert bc._topic_input_text(entry) == "Translate 'hello' to French."


def test_topic_input_text_skips_non_user_turns_and_fails_loud_on_neither():
    # defensive: an assistant-only prefix falls back to the natural query
    entry = _prefix_entry(n_prefix_turns=0, natural_query="fallback query")
    entry["prefix_turns"] = [{"role": "assistant", "content": "assistant-only"}]
    assert bc._topic_input_text(entry) == "fallback query"
    # neither field -> fail-loud, never a silent "other"
    with pytest.raises(ValueError, match="cannot derive a topic-label input"):
        bc._topic_input_text({"prefix_id": "pfx_bad", "prefix_turns": [], "natural_query": ""})


def test_topic_input_text_rejects_raw_conversation_shape():
    # A RAW conversation dict (keyed `turns`) is NOT a prefix entry; the
    # helper must not silently label it via some other field.
    raw_conv = {"id": "wc_1", "turns": [{"role": "user", "content": "hi"}]}
    with pytest.raises(ValueError):
        bc._topic_input_text(raw_conv)


# ── 2. _label_topic_batch on texts (production body, fake API boundary) ──────


def test_label_topic_batch_takes_texts_and_normalizes_labels():
    entries = [
        _prefix_entry(n_prefix_turns=2),
        _prefix_entry(n_prefix_turns=0, natural_query="Prove sqrt(2) is irrational."),
    ]
    texts = [bc._topic_input_text(e) for e in entries] + ["What team won the cup?"]
    client = _FakeAnthropicClient(
        ["coding_software", "math", "NOT A REAL LABEL"]  # exact / prefix-match / garbage
    )
    labels = bc._label_topic_batch(texts, client=client)
    assert labels == ["coding_software", "math_logic", "other"]
    assert len(client.messages.calls) == 3
    assert all(c["model"] == bc.HAIKU_MODEL for c in client.messages.calls)
    # the extracted text (not a dict repr) is what reaches the prompt
    assert "user prefix turn 0" in client.messages.calls[0]["messages"][0]["content"]
    assert "Prove sqrt(2)" in client.messages.calls[1]["messages"][0]["content"]


# ── 3. stream-pool checkpoint round-trip ──────────────────────────────────────


def test_stream_cache_roundtrip_mismatch_and_no_resume(monkeypatch, tmp_path, caplog):
    import logging

    import datasets

    rows = [_wc_row(i) for i in range(4)]
    # U+2028 inside content must survive the JSONL round-trip intact
    rows.append(_wc_row(99, user_text="line one\u2028line two of the same query (variant 99)"))

    calls: list[tuple] = []

    def fake_load_dataset(path, *, split=None, streaming=False, revision=None):
        calls.append((path, split, streaming, revision))
        return iter([dict(r) for r in rows])

    monkeypatch.setattr(datasets, "load_dataset", fake_load_dataset)
    monkeypatch.setattr(bc, "_SMOKE_TOKEN_COUNTS", True)
    cache_dir = tmp_path / "stream_cache"

    # first run streams + persists
    stats1: dict = {}
    kept1 = bc._stream_with_cache(
        "allenai/WildChat-1M",
        "0" * 40,
        rng=random.Random(0),
        row_limit=None,
        stats_out=stats1,
        cache_dir=cache_dir,
    )
    assert len(calls) == 1
    assert stats1["kept"] == 5 and stats1["resumed_from_cache"] is False
    assert (cache_dir / "wildchat.jsonl").exists()
    meta = json.loads((cache_dir / "wildchat.meta.json").read_text())
    assert meta["kept"] == 5
    assert meta["fingerprint"]["filter_recipe_version"] == bc.FILTER_RECIPE_VERSION
    assert meta["rejects"]["language"] == 0

    # second run with the SAME fingerprint resumes from cache — no re-stream
    stats2: dict = {}
    with caplog.at_level(logging.INFO, logger="issue1092.build_corpus"):
        kept2 = bc._stream_with_cache(
            "allenai/WildChat-1M",
            "0" * 40,
            rng=random.Random(0),
            row_limit=None,
            stats_out=stats2,
            cache_dir=cache_dir,
        )
    assert len(calls) == 1  # load_dataset NOT called again
    assert stats2["resumed_from_cache"] is True
    assert kept2 == kept1  # exact round-trip, U+2028 row included
    assert any("RESUMED from cache" in r.getMessage() for r in caplog.records)
    u2028 = [c for c in kept2 if "\u2028" in c["turns"][0]["content"]]
    assert len(u2028) == 1  # never shredded into two records

    # a DIFFERENT revision mismatches the fingerprint -> re-streams
    stats3: dict = {}
    bc._stream_with_cache(
        "allenai/WildChat-1M",
        "1" * 40,
        rng=random.Random(0),
        row_limit=None,
        stats_out=stats3,
        cache_dir=cache_dir,
    )
    assert len(calls) == 2
    assert stats3["resumed_from_cache"] is False

    # --no-resume-stream forces a re-stream even on a matching fingerprint
    bc._stream_with_cache(
        "allenai/WildChat-1M",
        "1" * 40,
        rng=random.Random(0),
        row_limit=None,
        stats_out={},
        cache_dir=cache_dir,
        resume=False,
    )
    assert len(calls) == 3


def test_stream_cache_fingerprint_covers_filter_constants():
    fp = bc._stream_fingerprint(
        "allenai/WildChat-1M", "0" * 40, lang_filter="en", stream_limit=None, row_limit=None
    )
    assert fp["max_total_tokens"] == bc.MAX_TOTAL_TOKENS
    assert fp["max_formatted_tokens"] == bc.MAX_FORMATTED_TOKENS
    assert fp["filter_recipe_version"] == bc.FILTER_RECIPE_VERSION
    fp2 = bc._stream_fingerprint(
        "allenai/WildChat-1M", "0" * 40, lang_filter="en", stream_limit=1000, row_limit=None
    )
    assert fp != fp2  # stream bounds are part of the identity


# ── 4. bank labeling + stratified top-up ──────────────────────────────────────


def _bank_conversations(n: int) -> list[dict]:
    convs = []
    for i in range(n):
        convs.append(
            {
                "id": f"lmsys_{i:06d}",
                "source": "lmsys",
                "turns": [
                    {"role": "user", "content": f"Bank question {i}: how does a battery work?"},
                    {"role": "assistant", "content": f"Bank answer {i}."},
                ],
                "n_user_turns": 1,
                "total_tokens": 20,
            }
        )
    return convs


def test_build_query_bank_labels_candidates_before_stratifying(monkeypatch):
    monkeypatch.setattr(bc, "_SMOKE_TOKEN_COUNTS", True)
    seen_texts: list[list[str]] = []

    def label_texts(texts: list[str]) -> list[str]:
        seen_texts.append(texts)
        # rotate through 3 real labels — uneven but multi-topic
        return [bc.TOPIC_LABELS[i % 3] for i in range(len(texts))]

    bank = bc._build_query_bank(
        _bank_conversations(60),
        prefix_conv_ids=set(),
        rng=random.Random(0),
        n_target=12,
        row_limit=None,
        label_texts=label_texts,
    )
    assert len(bank) == 12
    assert len(seen_texts) == 1 and len(seen_texts[0]) >= 12
    # labels came from label_texts, not the "other" default
    assert {q["topic"] for q in bank} <= set(bc.TOPIC_LABELS[:3])
    # extraction is the query text itself (truncated like _topic_input_text)
    assert all(t.startswith("Bank question") for t in seen_texts[0])


def test_build_query_bank_tops_up_when_labels_concentrate(monkeypatch):
    """Pre-round-8 latent crash: every candidate labeled 'other' collapsed the
    stratified subsample to ~n_target/12 queries (G1 bank floor 400 would fail
    strict in production). The top-up must fill the bank to n_target."""
    monkeypatch.setattr(bc, "_SMOKE_TOKEN_COUNTS", True)
    bank = bc._build_query_bank(
        _bank_conversations(80),
        prefix_conv_ids=set(),
        rng=random.Random(0),
        n_target=24,
        row_limit=None,
        label_texts=lambda texts: ["other"] * len(texts),  # worst-case concentration
    )
    assert len(bank) == 24  # pre-fix this was max(1, 24 // 12) == 2
    # query ids re-numbered densely
    assert sorted(q["query_id"] for q in bank) == [f"qry_{i:05d}" for i in range(24)]


def test_build_query_bank_respects_prefix_disjointness(monkeypatch):
    monkeypatch.setattr(bc, "_SMOKE_TOKEN_COUNTS", True)
    convs = _bank_conversations(30)
    excluded = {c["id"] for c in convs[:10]}
    bank = bc._build_query_bank(
        convs,
        prefix_conv_ids=excluded,
        rng=random.Random(0),
        n_target=10,
        row_limit=None,
        label_texts=lambda texts: ["general_qa"] * len(texts),
    )
    assert all(q["conv_id"] not in excluded for q in bank)


# ── 5. empty-prefix (bare-context) renders ────────────────────────────────────


class _StubChatTemplateTokenizer:
    """Signature-conformant stub of the ONE tokenizer surface `_render_instruct`
    uses (`apply_chat_template(messages, tokenize=..., add_generation_prompt=...)`)."""

    def __init__(self):
        self.calls: list[list[dict]] = []

    def apply_chat_template(
        self, messages: list[dict], *, tokenize: bool, add_generation_prompt: bool
    ) -> str:
        assert tokenize is False and add_generation_prompt is True
        self.calls.append(messages)
        parts = [f"<|{m['role']}|>{m['content']}" for m in messages]
        return "".join(parts) + "<|assistant|>"


def test_render_helpers_handle_empty_prefix(monkeypatch):
    # naturalistic: pure string logic, no tokenizer
    rendered_nat = bc._render_naturalistic([], "What is 2+2?")
    assert rendered_nat.startswith("User: What is 2+2?")
    assert rendered_nat.endswith("Assistant:")

    # instruct: single user message, generation prompt appended
    stub = _StubChatTemplateTokenizer()
    monkeypatch.setattr(bc, "_TOKENIZER", stub)
    rendered_inst = bc._render_instruct([], "What is 2+2?")
    assert stub.calls == [[{"role": "user", "content": "What is 2+2?"}]]
    assert rendered_inst.endswith("<|assistant|>")


# ── 6. battery prefix normalization + prefix-store validation ────────────────
# Caught LIVE by the round-8 tiny-real e2e (run 1, rc=1): the #594 battery's
# `f6_default_template` context has prefix_messages=[] AND system_prompt=null
# — `.get("system_prompt", "")` does NOT default on an explicit null, so the
# old fallback emitted a turn with content=None and the prefix-store digest
# crashed on len(None) AFTER the 3h-equivalent build work.


def test_battery_prefix_entry_null_system_prompt_yields_bare_context():
    ctx = {
        "id": "f6_default_template",
        "family": "default",
        "system_prompt": None,  # present-but-null — the real f6 shape
        "prefix_messages": [],
    }
    entry = bc._battery_prefix_entry(ctx, 0)
    assert entry["prefix_turns"] == []  # bare context, valid post-round-8
    assert entry["n_user_turns"] == 0
    assert entry["prefix_id"] == "batt_f6_default_template"
    # and the bare entry still yields a topic-label input path via natural_query?
    # No — battery entries are never topic-labeled; they must simply be
    # writable + renderable:
    assert bc._render_naturalistic(entry["prefix_turns"], "q").startswith("User: q")


def test_battery_prefix_entry_normalizes_messages_and_drops_null_content():
    ctx = {
        "id": "x1",
        "family": "helpful",
        "system_prompt": "You are terse.",
        "prefix_messages": [
            {"role": "user", "content": "real turn"},
            {"role": "assistant", "content": None},  # dropped, never None downstream
            {"content": "role-less turn"},  # role defaults to user
        ],
    }
    entry = bc._battery_prefix_entry(ctx, 1)
    assert entry["prefix_turns"] == [
        {"role": "user", "content": "real turn"},
        {"role": "user", "content": "role-less turn"},
    ]
    # empty prefix_messages + REAL system_prompt falls back to one user turn
    entry2 = bc._battery_prefix_entry(
        {"id": "x2", "system_prompt": "You are terse.", "prefix_messages": []}, 2
    )
    assert entry2["prefix_turns"] == [{"role": "user", "content": "You are terse."}]


def test_write_prefix_store_fails_loud_on_malformed_turn(tmp_path):
    good = _prefix_entry(n_prefix_turns=2)
    bad = _prefix_entry(n_prefix_turns=2)
    bad["prefix_id"] = "pfx_bad"
    bad["prefix_turns"][1]["content"] = None
    with pytest.raises(TypeError, match="pfx_bad"):
        bc._write_prefix_store([good, bad], tmp_path / "prefix_store.jsonl")
    # valid entries (incl. an EMPTY prefix) write fine
    bare = _prefix_entry(n_prefix_turns=0)
    bc._write_prefix_store([good, bare], tmp_path / "prefix_store.jsonl")
    lines = (tmp_path / "prefix_store.jsonl").read_text(encoding="utf-8").strip().split("\n")
    assert len(lines) == 2


def test_render_helpers_nonempty_prefix_roles_thread_through(monkeypatch):
    turns = _prefix_entry(n_prefix_turns=2)["prefix_turns"]
    stub = _StubChatTemplateTokenizer()
    monkeypatch.setattr(bc, "_TOKENIZER", stub)
    bc._render_instruct(turns, "Next question?")
    assert [m["role"] for m in stub.calls[0]] == ["user", "assistant", "user"]
    rendered_nat = bc._render_naturalistic(turns, "Next question?")
    assert "User: user prefix turn 0" in rendered_nat
    assert "Assistant: assistant prefix turn 1" in rendered_nat
    assert rendered_nat.endswith("Assistant:")
