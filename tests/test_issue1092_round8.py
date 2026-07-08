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

import argparse
import json
import random
import sys
from pathlib import Path
from typing import ClassVar

import numpy as np
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


# ── 7. round-8.1: trait-stratum loader parses the REAL corpus_specs layout ───
# Orchestrator-verified @5aa6de1b: corpus_specs/ holds <trait>_personas.json +
# <trait>_questions.json per trait; each personas file is a dict
# {"personas": [60 persona system-prompt STRINGS]}. The pre-8.1 loader swept
# any <trait>*.json ([:2] cap, incl. the questions file) and treated each FILE
# as one persona -> 2/trait instead of ~33/trait (concern
# i1092-trait-stratum-underpopulated). Fake trait names only — trait-name
# literals stay out of the repo per the content-filter protocol.


def _fake_personas_download(tmp_path, fetched, *, n_personas=60, payload=None):
    def fake_hf_hub_download(repo_id, filename, *, repo_type=None, revision=None):
        fetched.append(filename)
        name = Path(filename).name
        assert name.endswith("_personas.json"), "questions files must never be fetched"
        trait = name.removesuffix("_personas.json")
        body = (
            payload
            if payload is not None
            else {"personas": [f"{trait} persona spec {i}" for i in range(n_personas)]}
        )
        p = tmp_path / name
        p.write_text(json.dumps(body), encoding="utf-8")
        return str(p)

    return fake_hf_hub_download


def test_load_trait_stratum_personas_samples_33_per_trait(monkeypatch, tmp_path):
    from collections import Counter

    import huggingface_hub

    fetched: list[str] = []
    monkeypatch.setattr(
        huggingface_hub, "hf_hub_download", _fake_personas_download(tmp_path, fetched)
    )
    traits = ["traitA", "traitB", "traitC"]
    entries = bc._load_trait_stratum_personas(traits, n_per_trait=33, rng=random.Random(42))
    assert len(entries) == 99  # pre-8.1 this was 2/trait -> 6
    per_trait = Counter(e["trait"] for e in entries)
    assert per_trait == {"traitA": 33, "traitB": 33, "traitC": 33}
    assert [Path(f).name for f in fetched] == [f"{t}_personas.json" for t in traits]
    for e in entries:
        assert e["system_prompt"].startswith(e["trait"] + " persona spec")
        assert e["valence"] == "unspecified"  # no per-persona tag in the artifact
        assert e["source_file"] == f"{e['trait']}_personas.json"
    # 33-of-60 draw: no duplicates within a trait
    assert len({e["system_prompt"] for e in entries}) == 99
    # deterministic under the same seed
    again = bc._load_trait_stratum_personas(traits, n_per_trait=33, rng=random.Random(42))
    assert [e["system_prompt"] for e in again] == [e["system_prompt"] for e in entries]


def test_load_trait_stratum_personas_fails_loud_on_wrong_shape(monkeypatch, tmp_path):
    import huggingface_hub

    # top-level LIST (the old code's assumed shape) -> TypeError
    monkeypatch.setattr(
        huggingface_hub,
        "hf_hub_download",
        _fake_personas_download(tmp_path, [], payload=[{"system_prompt": "x"}]),
    )
    with pytest.raises(TypeError, match="expected a dict with a 'personas' list"):
        bc._load_trait_stratum_personas(["traitA"], n_per_trait=33, rng=random.Random(0))

    # dict without a personas list -> TypeError naming the keys
    monkeypatch.setattr(
        huggingface_hub,
        "hf_hub_download",
        _fake_personas_download(tmp_path, [], payload={"specs": ["x"]}),
    )
    with pytest.raises(TypeError, match="specs"):
        bc._load_trait_stratum_personas(["traitA"], n_per_trait=33, rng=random.Random(0))

    # non-string persona entry -> TypeError with indices
    monkeypatch.setattr(
        huggingface_hub,
        "hf_hub_download",
        _fake_personas_download(tmp_path, [], payload={"personas": ["ok", None, ""]}),
    )
    with pytest.raises(TypeError, match="non-string/empty persona entries"):
        bc._load_trait_stratum_personas(["traitA"], n_per_trait=33, rng=random.Random(0))

    # empty personas list -> ValueError
    monkeypatch.setattr(
        huggingface_hub,
        "hf_hub_download",
        _fake_personas_download(tmp_path, [], payload={"personas": []}),
    )
    with pytest.raises(ValueError, match="no persona entries"):
        bc._load_trait_stratum_personas(["traitA"], n_per_trait=33, rng=random.Random(0))


def test_load_trait_stratum_personas_caps_at_pool_size(monkeypatch, tmp_path):
    import huggingface_hub

    monkeypatch.setattr(
        huggingface_hub,
        "hf_hub_download",
        _fake_personas_download(tmp_path, [], n_personas=10),
    )
    entries = bc._load_trait_stratum_personas(["traitA"], n_per_trait=33, rng=random.Random(0))
    assert len(entries) == 10  # min(n_per_trait, pool)


# ── 8. round-8.2: gpu_phase bare-prefix crash (BLOCKER concern
# i1092-battery-bare-prefix-gpu-phase-crash). The round-8 battery fix ships
# batt_f6_default_template with prefix_turns: [] (valid bare context), but
# gpu_phase._prefix_turns's `.get(a) or .get(b)` chain coerced [] -> None and
# raised on every f6 row in every cell — a guaranteed P2/P3 crash on the
# provisioned pod after P0 + the Claude batch were spent. Fail pre-fix.

import issue1092_gpu_phase as gp  # noqa: E402


class _QwenLikeTemplateTokenizer:
    """Behavior-conformant stub of the pinned Qwen chat-template surface:
    raises IndexError on an EMPTY messages list (verified live on the real
    pinned tokenizer, 2026-07-07) and injects the default system block when
    the first message is not a system turn."""

    SYSTEM_BLOCK = "<|im_start|>system\nYou are Qwen-stub.<|im_end|>\n"

    def apply_chat_template(
        self, messages: list[dict], *, tokenize: bool, add_generation_prompt: bool
    ) -> str:
        assert tokenize is False
        first = messages[0]  # mirrors the Qwen Jinja: IndexError on []
        parts = []
        if first["role"] != "system":
            parts.append(self.SYSTEM_BLOCK)
        for m in messages:
            parts.append(f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n")
        if add_generation_prompt:
            parts.append("<|im_start|>assistant\n")
        return "".join(parts)


def test_gpu_phase_prefix_turns_accepts_explicit_empty_list():
    # PRESENT-but-empty prefix_turns is a valid bare context (fails pre-fix)
    assert gp._prefix_turns({"prefix_id": "batt_f6_default_template", "prefix_turns": []}) == []
    turns = [{"role": "user", "content": "hi"}]
    assert gp._prefix_turns({"prefix_id": "pfx_00001", "prefix_turns": turns}) == turns
    assert gp._prefix_turns({"id": "conv", "turns": turns}) == turns
    # genuinely ABSENT / non-list turns stay fail-loud
    with pytest.raises(ValueError, match="no turns"):
        gp._prefix_turns({"prefix_id": "pfx_bad"})
    with pytest.raises(ValueError, match="no turns"):
        gp._prefix_turns({"prefix_id": "pfx_bad", "prefix_turns": "not-a-list"})
    with pytest.raises(ValueError, match="no turns"):
        gp._prefix_turns({"prefix_id": "pfx_bad", "prefix_turns": None})


def test_gpu_phase_render_row_bare_battery_prefix_both_formats(monkeypatch):
    stub = _QwenLikeTemplateTokenizer()
    monkeypatch.setattr(gp, "_get_tokenizer", lambda: stub)
    prefix_store = {
        "batt_f6_default_template": {
            "prefix_id": "batt_f6_default_template",
            "prefix_turns": [],
            "source": "battery",
        },
        "pfx_00001": {
            "prefix_id": "pfx_00001",
            "prefix_turns": [
                {"role": "user", "content": "first question"},
                {"role": "assistant", "content": "first reply"},
            ],
        },
    }
    query_store = {"qry_00000": {"query_id": "qry_00000", "text": "What is 2+2?"}}
    row = {"row_id": "r_0", "prefix_id": "batt_f6_default_template", "query_id": "qry_00000"}

    # instruct: crashes pre-fix (ValueError in _prefix_turns); post-fix the
    # bare-context prefix is the template-injected system block, sliced off
    # the rendered prompt itself -> the string-prefix invariant holds and the
    # prefix_end capture position stays "last token before the query turn".
    prefix, prompt, comp = gp.render_row(row, prefix_store, query_store, "instruct", "own")
    assert comp is None
    assert prefix == stub.SYSTEM_BLOCK
    assert prompt.startswith(prefix)
    assert "What is 2+2?" in prompt
    assert prompt.endswith("<|im_start|>assistant\n")

    # pretrained: nothing precedes the query -> empty prefix, bare render
    prefix_n, prompt_n, _ = gp.render_row(row, prefix_store, query_store, "pretrained", "own")
    assert prefix_n == ""
    assert prompt_n.startswith("User: What is 2+2?")
    assert prompt_n.endswith("Assistant:")

    # non-empty prefixes keep the same string-prefix invariant
    row2 = {"row_id": "r_1", "prefix_id": "pfx_00001", "query_id": "qry_00000"}
    p2, pr2, _ = gp.render_row(row2, prefix_store, query_store, "instruct", "own")
    assert p2 and pr2.startswith(p2)
    assert "first question" in p2

    # direct empty guards on the prefix renderers (never reach the template)
    assert gp._render_prefix_instruct([]) == ""
    assert gp._render_prefix_naturalistic([]) == ""


# ── 9. round-8.2 Minor-2: post-dense-core crossing strata — first unit
# coverage for _build_manifest_rows periphery / trait / battery branches
# (previously first executed at production scale; the 60-row e2e and the
# smoke both early-return at dense_core).


def test_build_manifest_rows_post_dense_core_strata(monkeypatch):
    from collections import Counter

    monkeypatch.setattr(bc, "DENSE_CORE_PREFIXES", 3)
    prefixes = []
    for i in range(5):  # first 3 core, last 2 peripheral
        e = _prefix_entry(n_prefix_turns=2)
        e["prefix_id"] = f"pfx_{i:05d}"
        e["conv_id"] = f"conv_{i}"
        e["topic"] = "general_qa"
        prefixes.append(e)
    bank = [
        {
            "query_id": f"qry_{i:05d}",
            "text": f"bank query {i}",
            "topic": "coding_software" if i < 2 else "general_qa",
            "source": "lmsys",
            "conv_id": f"bankconv_{i}",
        }
        for i in range(10)
    ]
    core_queries = bank[:2]  # periphery bank = the 8 general_qa queries
    trait_personas = [
        {
            "trait": "traitA",
            "prefix_id": "trait_0000",
            "system_prompt": "x",
            "valence": "unspecified",
        },
        {
            "trait": "traitB",
            "prefix_id": "trait_0001",
            "system_prompt": "y",
            "valence": "unspecified",
        },
    ]
    battery = [
        {"id": "b1", "family": "general_qa", "system_prompt": "z", "prefix_messages": []},
        {
            "id": "f6_default_template",
            "family": "default",
            "system_prompt": None,
            "prefix_messages": [],
        },
    ]
    rows = bc._build_manifest_rows(
        prefixes,
        bank,
        core_queries,
        trait_personas,
        battery,
        rng=random.Random(0),
        row_limit=None,
        cells_filter=None,
    )
    by_stratum = Counter(r["stratum"] for r in rows)
    assert by_stratum == {
        "dense_core": 3 * 2,  # 3 core prefixes x 2 core queries
        "periphery_natural": 2,  # 1 per peripheral prefix
        "periphery_random": 2 * 8,  # min(N_PERIPHERY_RANDOM=10, 8 periphery bank)
        "periphery_topicmatch": 2 * 3,  # 3 topic-matched (8 general_qa available)
        "trait_stratum": 2 * 8,  # min(N_TRAIT_STRATUM_QUERIES=15, 8 periphery bank)
        "battery": 2 * 2,  # 2 contexts x 2 core queries
    }
    # battery rows are EVAL-ONLY; everything else is not
    for r in rows:
        assert r["is_eval_only"] == (r["stratum"] == "battery")
    # every emitted id resolves against the stores the GPU phase will load
    prefix_ids = (
        {p["prefix_id"] for p in prefixes}
        | {p["prefix_id"] for p in trait_personas}
        | {f"batt_{c['id']}" for c in battery}
    )
    query_ids = {q["query_id"] for q in bank} | {f"nat_{p['prefix_id']}" for p in prefixes}
    for r in rows:
        assert r["prefix_id"] in prefix_ids, r
        assert r["query_id"] in query_ids, r
    # row ids unique; topic-matched periphery rows carry the prefix topic
    assert len({r["row_id"] for r in rows}) == len(rows)
    for r in rows:
        if r["stratum"] == "periphery_topicmatch":
            assert r["topic"] == "general_qa"


# ── 10. round-8.3: realized-pair token-budget filter (production incident —
# GPU launch #2: vLLM `decoder prompt (length 8290) > max_model_len 8192` on
# crossed (P, q) renders P0 never budget-checked; the old check bounded only
# the prefix+NATURAL-query pair).


class _FullStubTokenizer(_StubChatTemplateTokenizer):
    """Chat-template stub + batch-encode surface (`tok(texts,
    add_special_tokens=False)["input_ids"]`) with word-piece ids."""

    def __call__(self, texts: list[str], *, add_special_tokens: bool) -> dict:
        assert add_special_tokens is False
        return {"input_ids": [[hash(w) % 1000 for w in t.split()] for t in texts]}


def _budget_fixture(*, long_words: int = 200):
    """Prefix store (one short, one long prefix) + query store + rows."""
    prefix_lookup = {
        "pfx_short": {
            "prefix_id": "pfx_short",
            "prefix_turns": [
                {"role": "user", "content": "short question"},
                {"role": "assistant", "content": "short answer"},
            ],
            "natural_query": "next?",
            "n_user_turns": 1,
        },
        "pfx_long": {
            "prefix_id": "pfx_long",
            "prefix_turns": [
                {"role": "user", "content": "w " * long_words},
                {"role": "assistant", "content": "w " * long_words},
            ],
            "natural_query": "next?",
            "n_user_turns": 1,
        },
    }
    query_lookup = {
        "qry_00000": {"query_id": "qry_00000", "text": "What is 2+2?", "topic": "math_logic"}
    }
    rows = [
        {
            "row_id": "r_0000000",
            "stratum": "dense_core",
            "prefix_id": "pfx_short",
            "query_id": "qry_00000",
            "is_eval_only": False,
        },
        {
            "row_id": "r_0000001",
            "stratum": "periphery_random",
            "prefix_id": "pfx_long",
            "query_id": "qry_00000",
            "is_eval_only": False,
        },
    ]
    return prefix_lookup, query_lookup, rows


def test_apply_realized_budget_filter_pair_drops_and_annotates(monkeypatch):
    monkeypatch.setattr(bc, "_SMOKE_TOKEN_COUNTS", True)  # word counts
    monkeypatch.setattr(bc, "_TOKENIZER", _FullStubTokenizer())
    prefix_lookup, query_lookup, rows = _budget_fixture()

    kept, digest = bc._apply_realized_budget_filter(
        rows, prefix_lookup, query_lookup, max_tokens=50, max_drop_frac=0.9
    )
    # the near-cap prefix's CROSSED render busts the budget -> pair-dropped
    # (gone for ALL cells/formats); the short pair is kept WITH token counts
    assert [r["row_id"] for r in kept] == ["r_0000000"]
    assert kept[0]["n_tokens_instruct"] > 0
    assert kept[0]["n_tokens_pretrained"] > 0
    assert kept[0]["n_tokens_instruct"] <= 50 and kept[0]["n_tokens_pretrained"] <= 50
    assert digest["total_rows"] == 2
    assert digest["budget_dropped"] == 1
    assert digest["dropped_by_stratum"] == {"periphery_random": 1}
    assert digest["max_formatted_tokens"] == 50
    # the dropped row was still annotated (diagnosability)
    assert rows[1]["n_tokens_instruct"] > 50 or rows[1]["n_tokens_pretrained"] > 50


def test_apply_realized_budget_filter_either_format_semantics(monkeypatch):
    """A row is dropped when EITHER format busts the budget, even if the other
    is under — pin by computing the two counts and thresholding between them."""
    monkeypatch.setattr(bc, "_SMOKE_TOKEN_COUNTS", True)
    monkeypatch.setattr(bc, "_TOKENIZER", _FullStubTokenizer())
    prefix_lookup, query_lookup, rows = _budget_fixture()
    row = [rows[0]]
    turns = prefix_lookup["pfx_short"]["prefix_turns"]
    query = query_lookup["qry_00000"]["text"]
    n_inst = bc._batch_token_counts([bc._render_instruct(turns, query)])[0]
    n_nat = bc._batch_token_counts([bc._render_naturalistic(turns, query)])[0]
    assert n_inst != n_nat  # instruct render carries template tokens
    between = (min(n_inst, n_nat) + max(n_inst, n_nat)) // 2
    assert min(n_inst, n_nat) <= between < max(n_inst, n_nat)
    kept, digest = bc._apply_realized_budget_filter(
        row, prefix_lookup, query_lookup, max_tokens=between, max_drop_frac=1.0
    )
    assert kept == [] and digest["budget_dropped"] == 1


def test_apply_realized_budget_filter_fails_loud_over_max_drop_frac(monkeypatch):
    monkeypatch.setattr(bc, "_SMOKE_TOKEN_COUNTS", True)
    monkeypatch.setattr(bc, "_TOKENIZER", _FullStubTokenizer())
    prefix_lookup, query_lookup, rows = _budget_fixture()
    with pytest.raises(RuntimeError, match="systematic budget problem"):
        bc._apply_realized_budget_filter(
            rows, prefix_lookup, query_lookup, max_tokens=50, max_drop_frac=0.05
        )


def _write_corpus_fixture(corpus_dir, prefix_lookup, query_lookup, rows):
    corpus_dir.mkdir(parents=True, exist_ok=True)
    for name, items in (
        ("manifest.jsonl", rows),
        ("prefix_store.jsonl", list(prefix_lookup.values())),
        ("query_store.jsonl", list(query_lookup.values())),
    ):
        with open(corpus_dir / name, "w", encoding="utf-8") as f:
            for item in items:
                f.write(json.dumps(item, ensure_ascii=False))
                f.write("\n")
    (corpus_dir / "derangement_map.json").write_text("{}", encoding="utf-8")
    (corpus_dir / "manifest_stats.json").write_text(
        json.dumps({"trait_names": ["traitA"], "n_bank_queries": 1}), encoding="utf-8"
    )


def test_filter_existing_corpus_preserves_kept_row_ids_verbatim(monkeypatch, tmp_path):
    """The round-8.3 production-correction mode: terminal DROP over an existing
    corpus — kept rows keep their row_ids + (prefix_id, query_id, stratum)
    assemblies verbatim (the P1 Claude batch is already submitted against
    them); the derangement is recomputed over kept rows only; pre-filter
    copies are preserved; a second run refuses."""
    monkeypatch.setattr(bc, "_SMOKE_TOKEN_COUNTS", False)
    monkeypatch.setattr(bc, "_TOKENIZER", _FullStubTokenizer())  # tokenizer boundary
    # tiny fixture -> relax the production G1 floors (module constants)
    monkeypatch.setattr(bc, "N_PREFIXES_FLOOR", 1)
    monkeypatch.setattr(bc, "N_LONG_CONV_FLOOR", 0)
    monkeypatch.setattr(bc, "N_BANK_FLOOR", 1)

    # long prefix must exceed the PRODUCTION default budget (7,168) under the
    # stub's word counting: 2 turns x 4000 words ≈ 8000 > 7168
    prefix_lookup, query_lookup, rows = _budget_fixture(long_words=4000)
    # add a second kept row so the derangement has >=2 candidates
    prefix_lookup["pfx_short2"] = dict(
        prefix_lookup["pfx_short"], prefix_id="pfx_short2", n_user_turns=5
    )
    rows.append(
        {
            "row_id": "r_0000002",
            "stratum": "periphery_natural",
            "prefix_id": "pfx_short2",
            "query_id": "qry_00001",
            "is_eval_only": False,
        }
    )
    query_lookup["qry_00001"] = {
        "query_id": "qry_00001",
        "text": "Explain tides briefly.",
        "topic": "science_medicine",
    }
    corpus_dir = tmp_path / "corpus"
    _write_corpus_fixture(corpus_dir, prefix_lookup, query_lookup, rows)
    old_manifest_bytes = (corpus_dir / "manifest.jsonl").read_bytes()

    rc = bc.main(
        [
            "--filter-existing",
            str(corpus_dir),
            "--no-upload",
            "--eval-dir",
            str(tmp_path / "eval_stats"),
            "--budget-max-drop-frac",
            "0.9",
        ]
    )
    assert rc == 0

    # pre-filter copy preserved byte-verbatim
    assert (corpus_dir / "manifest.pre_budget_filter.jsonl").read_bytes() == old_manifest_bytes

    old_rows = {r["row_id"]: r for r in rows}
    new_rows = [
        json.loads(line)
        for line in (corpus_dir / "manifest.jsonl").read_text(encoding="utf-8").split("\n")
        if line
    ]
    new_ids = {r["row_id"] for r in new_rows}
    # kept ids are a strict subset (the long-prefix row dropped)
    assert new_ids == {"r_0000000", "r_0000002"}
    assert new_ids <= set(old_rows)
    for r in new_rows:
        old = old_rows[r["row_id"]]
        assert (r["prefix_id"], r["query_id"], r["stratum"]) == (
            old["prefix_id"],
            old["query_id"],
            old["stratum"],
        )
        assert r["n_tokens_instruct"] > 0 and r["n_tokens_pretrained"] > 0

    # derangement recomputed over kept rows only
    dmap = json.loads((corpus_dir / "derangement_map.json").read_text(encoding="utf-8"))
    assert set(dmap) <= new_ids and set(dmap.values()) <= new_ids and dmap

    # stats carry the budget digest + provenance + carried-over fields
    stats = json.loads((tmp_path / "eval_stats" / "manifest_stats.json").read_text())
    assert stats["budget_filter"]["budget_dropped"] == 1
    assert stats["filtered_from"]["n_rows_pre_filter"] == 3
    assert stats["trait_names"] == ["traitA"]
    assert stats["g1_gate"]["pass"] is True

    # a second run REFUSES (double-filter would misreport stats)
    with pytest.raises(RuntimeError, match="refusing to double-filter"):
        bc.main(
            [
                "--filter-existing",
                str(corpus_dir),
                "--no-upload",
                "--eval-dir",
                str(tmp_path / "eval_stats"),
            ]
        )


# ── 11. round-8.4: G2 identity — BPE-seam position misalignment in the capture
# (GPU launch #3: `G2 identity generate-reference mismatch ... max_abs=2.9375`).
# The old capture tokenized the CONCATENATED prompt+completion+boundary string
# but computed positions from PER-SEGMENT token counts; Qwen BPE merges across
# the seams (a "\n"-leading completion merges into the instruct prompt's
# trailing "assistant\n"; the rstripped naturalistic prefix's "." merges into
# ".\n\n"), shifting context_end/prefix_end/t1/t2/t3/B0. Fix: per-segment
# token-id concatenation + offset-based prefix_end
# (gp._capture_row_ids_and_positions). Uses the REAL pinned tokenizer (the
# merges are its vocabulary facts) + a tiny random same-arch model on CPU fp32.

QWEN_INSTRUCT_REV = "a09a35458c702b33eeacc393d103063234e8bc28"


@pytest.fixture(scope="module")
def qwen_tokenizer():
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct", revision=QWEN_INSTRUCT_REV)


def test_capture_positions_immune_to_prompt_completion_seam_merge(qwen_tokenizer):
    tok = qwen_tokenizer
    prompt = tok.apply_chat_template(
        [{"role": "user", "content": "Say hi."}], tokenize=False, add_generation_prompt=True
    )
    completion = "\nSure, hello!"  # leading "\n" merges into the prompt's trailing "assistant\n"
    boundary = gp._boundary_suffix("instruct")
    prompt_ids = tok.encode(prompt, add_special_tokens=False)

    # the live merge the launch-#3 gate hit: concatenated-text tokenization
    # does NOT preserve the prompt segment
    naive_full = tok.encode(prompt + completion + boundary, add_special_tokens=False)
    assert naive_full[: len(prompt_ids)] != prompt_ids

    row_ids, pos = gp._capture_row_ids_and_positions(
        tok, prefix_text="", prompt=prompt, completion=completion, boundary=boundary
    )
    # segment concat: prompt segment bit-identical to what generation consumed
    assert row_ids[: len(prompt_ids)] == prompt_ids
    assert pos["context_end"] == len(prompt_ids) - 1
    assert pos["answer_start"] == len(prompt_ids)
    n_comp = len(tok.encode(completion, add_special_tokens=False))
    assert pos["answer_end"] == len(prompt_ids) + n_comp
    assert pos["t3"] == len(row_ids) - 1
    assert pos["n_total"] == len(row_ids)
    # bare context: no prefix tokens
    assert pos["prefix_end"] == 0


def test_capture_prefix_end_offset_based_on_rstripped_naturalistic_prefix(qwen_tokenizer):
    tok = qwen_tokenizer
    turns = [
        {"role": "user", "content": "a question here"},
        {"role": "assistant", "content": "an answer."},
    ]
    prefix_text = gp._render_prefix_naturalistic(turns)  # rstripped -> ends "an answer."
    prompt = gp._render_naturalistic(turns, "next?")
    assert prompt.startswith(prefix_text)
    row_ids, pos = gp._capture_row_ids_and_positions(
        tok, prefix_text, prompt, "ok", gp._boundary_suffix("pretrained")
    )
    naive_n_prefix = len(tok.encode(prefix_text, add_special_tokens=False))
    # the prefix's final "." merges into ".\n\n" in the PROMPT tokenization —
    # the naive per-segment count is misaligned on this fixture; offsets are not
    assert pos["prefix_end"] != naive_n_prefix - 1
    decoded_through_prefix_end = tok.decode(row_ids[: pos["prefix_end"] + 1])
    assert prefix_text.startswith(decoded_through_prefix_end)
    # the NEXT token crosses the prefix boundary (it is the merge token)
    assert len(tok.decode(row_ids[: pos["prefix_end"] + 2])) > len(prefix_text)


# ── 13. round-8.6: calibrated G2 identity criterion. Launch 6 failed the old
# allclose(atol=5e-2) at max_abs=3.0; the on-pod decomposition proved the
# construction EXACT (token ids identical on all 50 spot rows,
# recompute-vs-disk 0.0, fp32 both-sides max_rel 7.5e-5) and the residual
# pure bf16 batch-geometry numerics (same-ids b1-vs-b8 null: max_abs 3.0 at
# the SAME element as the gate read — a 1.36% relative error on a
# magnitude-221 Qwen outlier dim; null p99 floored-rel 0.076). Criterion:
# p99 floored-rel <= 0.30 (~4x null) + max_abs <= 300 backstop (100x null) —
# a bulk read, so a misaligned construction (>=2% of elements at rel ~1)
# still FAILS loud.


def _synthetic_identity_pair(seed: int, *, kind: str):
    """Qwen-shaped synthetic (disk, ref): O(1) dims + O(1000) outlier dims."""
    import numpy as np

    rng = np.random.default_rng(seed)
    ref = rng.normal(size=(50, 28, 64))
    ref[:, :, :4] *= 1000.0  # activation-outlier dims (Qwen2-family late layers)
    disk = ref.copy()
    if kind == "numerics":
        # measured-null-shaped noise: ~0.3% relative on outlier dims (abs ~3),
        # small absolute jitter elsewhere — the launch-6 residual shape
        disk += 0.003 * np.abs(ref) * rng.normal(size=ref.shape) * (np.abs(ref) > 100)
        disk += 0.05 * rng.normal(size=ref.shape) * (np.abs(ref) <= 100)
    elif kind == "shuffled":
        # ONE swapped row pair — states read at the wrong position/row
        disk[0], disk[1] = ref[1].copy(), ref[0].copy()
    elif kind == "backstop":
        # single-element O(magnitude) shift: p99 misses it, the backstop fires
        disk[3, 26, 0] += 400.0
    elif kind != "exact":
        raise ValueError(kind)
    return disk.astype(np.float32), ref.astype(np.float32)


def test_g2_identity_criterion_passes_bf16_scale_numerics():
    disk, ref = _synthetic_identity_pair(1092, kind="numerics")
    stats = gp._g2_identity_check(disk, ref)
    assert stats["pass"], stats
    assert stats["max_abs"] > 0.05  # the OLD atol=5e-2 allclose FAILED exactly this shape
    assert stats["p99_rel_floored"] < gp.G2_IDENTITY_P99_REL_TOL


def test_g2_identity_criterion_fails_shuffled_position_construction():
    disk, ref = _synthetic_identity_pair(7, kind="shuffled")
    stats = gp._g2_identity_check(disk, ref)
    assert not stats["pass"], stats
    assert stats["p99_rel_floored"] > gp.G2_IDENTITY_P99_REL_TOL


def test_g2_identity_criterion_abs_backstop_catches_sparse_outlier_shift():
    disk, ref = _synthetic_identity_pair(9, kind="backstop")
    stats = gp._g2_identity_check(disk, ref)
    assert not stats["pass"], stats
    assert stats["max_abs"] > gp.G2_IDENTITY_ABS_BACKSTOP
    assert stats["p99_rel_floored"] <= gp.G2_IDENTITY_P99_REL_TOL  # only the backstop trips


def test_g2_identity_criterion_exact_agreement_passes():
    disk, ref = _synthetic_identity_pair(3, kind="exact")
    stats = gp._g2_identity_check(disk, ref)
    assert stats["pass"] and stats["max_abs"] == 0.0


# ── 14. round-8.7: deterministic G2 token-id pre-check (folded from the 8.6
# diag into the gate; runs BEFORE the statistical identity comparison). It
# retires the reviewer's residual blind spot: a 1-2-row CORRELATED
# adjacent-position shift lands p99 ~0.16-0.23 and PASSES the statistical bar,
# but the pre-check catches ANY token/position construction divergence with
# exact precision.


def _precheck_fixture(tok):
    prompts = [
        tok.apply_chat_template(
            [{"role": "user", "content": q}], tokenize=False, add_generation_prompt=True
        )
        for q in ("Say hi.", "Name a color.")
    ]
    return {
        "prefix_texts": ["", ""],
        "prompts": prompts,
        "completions": ["Hello there!", "\nBlue."],  # incl. the seam-merge case
        "boundary": gp._boundary_suffix("instruct"),
        "cell_id": "cell_inst_own",
        "row_labels": ["r_0000000", "r_0000001"],
    }


def test_g2_token_id_precheck_passes_on_real_construction(qwen_tokenizer):
    # post-8.4 construction: capture prompt-segment ids == reference ids by
    # construction, including the "\n"-leading seam-merge completion
    gp._g2_token_id_precheck(qwen_tokenizer, **_precheck_fixture(qwen_tokenizer))


def test_g2_token_id_precheck_fails_deterministically_on_one_row_id_mismatch(
    qwen_tokenizer, monkeypatch
):
    """A regressed capture construction on ONE row (the correlated-single-row
    mode the statistical p99 bar cannot catch) fails the pre-check
    deterministically, with the first-divergence dump naming the row."""
    real = gp._capture_row_ids_and_positions
    calls = {"n": 0}

    def regressed(tokenizer, prefix_text, prompt, completion, boundary, row_label="?"):
        row_ids, pos = real(tokenizer, prefix_text, prompt, completion, boundary, row_label)
        calls["n"] += 1
        if calls["n"] == 2:  # corrupt row 2's prompt segment by one token (off-by-one shift)
            row_ids = row_ids[1:]
            pos = dict(pos, context_end=pos["context_end"] - 1, n_total=pos["n_total"] - 1)
        return row_ids, pos

    monkeypatch.setattr(gp, "_capture_row_ids_and_positions", regressed)
    with pytest.raises(AssertionError, match=r"token-id pre-check FAILED.*r_0000001"):
        gp._g2_token_id_precheck(qwen_tokenizer, **_precheck_fixture(qwen_tokenizer))
    assert calls["n"] == 2  # failed exactly at the corrupted row


# ── 12. round-8.5: vLLM H100-IMA mitigation flags thread into EVERY engine
# construction (launch #4: CUDA illegal memory access in vLLM 0.11.0's engine
# step on 8x H100 at production shapes under heavy shared-prefix reuse;
# identical code was A100-clean — mitigation knobs, default OFF).


class _FakeVllmLLM:
    instances: ClassVar[list[dict]] = []

    def __init__(self, **kwargs):
        _FakeVllmLLM.instances.append(kwargs)

    def generate(self, chunk, params, use_tqdm=False):
        return []


class _FakeSamplingParams:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


def _install_fake_vllm(monkeypatch):
    import types

    fake = types.ModuleType("vllm")
    fake.LLM = _FakeVllmLLM
    fake.SamplingParams = _FakeSamplingParams
    monkeypatch.setitem(sys.modules, "vllm", fake)
    _FakeVllmLLM.instances.clear()


def test_vllm_engine_overrides_from_flags():
    ns = argparse.Namespace(no_prefix_caching=True, enforce_eager=True)
    assert gp._vllm_engine_overrides(ns) == {
        "enable_prefix_caching": False,
        "enforce_eager": True,
    }
    # default OFF -> no engine-config change at all
    assert gp._vllm_engine_overrides(argparse.Namespace()) == {}
    assert (
        gp._vllm_engine_overrides(argparse.Namespace(no_prefix_caching=False, enforce_eager=False))
        == {}
    )
    only_pc = gp._vllm_engine_overrides(
        argparse.Namespace(no_prefix_caching=True, enforce_eager=False)
    )
    assert only_pc == {"enable_prefix_caching": False}


def test_flags_reach_every_vllm_construction_site(monkeypatch):
    _install_fake_vllm(monkeypatch)
    overrides = gp._vllm_engine_overrides(
        argparse.Namespace(no_prefix_caching=True, enforce_eager=True)
    )

    # site 1: _run_gen_vllm (per-shard fresh engine)
    out = gp._run_gen_vllm(
        prompts=[],
        model_name="m",
        revision="r",
        stop_tokens=["<|im_end|>"],
        max_tokens=8,
        seed=42,
        gpu_id=0,
        chunk_size=4,
        engine_overrides=overrides,
    )
    assert out == []
    assert len(_FakeVllmLLM.instances) == 1
    kw = _FakeVllmLLM.instances[0]
    assert kw["enable_prefix_caching"] is False and kw["enforce_eager"] is True
    assert kw["model"] == "m" and kw["max_model_len"] == gp.MAX_MODEL_LEN

    # default OFF: the kwargs are ABSENT (vLLM defaults untouched)
    gp._run_gen_vllm(
        prompts=[],
        model_name="m",
        revision="r",
        stop_tokens=[],
        max_tokens=8,
        seed=42,
        gpu_id=0,
        chunk_size=4,
    )
    assert "enable_prefix_caching" not in _FakeVllmLLM.instances[1]
    assert "enforce_eager" not in _FakeVllmLLM.instances[1]

    # site 2: PersistentGpuRuntime.generate (worker-loop cached engine)
    runtime = gp.PersistentGpuRuntime(0, engine_overrides=overrides)
    assert runtime.engine_overrides == overrides
    out2 = runtime.generate(
        prompts=[],
        model_name="m2",
        revision="r2",
        stop_tokens=[],
        max_tokens=8,
        seed=42,
        chunk_size=4,
    )
    assert out2 == []
    kw2 = _FakeVllmLLM.instances[-1]
    assert kw2["model"] == "m2"
    assert kw2["enable_prefix_caching"] is False and kw2["enforce_eager"] is True


def test_g2_identity_capture_vs_generate_reference_cpu(qwen_tokenizer):
    """The G2 comparison itself, on CPU fp32 with a tiny random same-arch model:
    teacher-forced capture context_end == generate()-hook reference at the last
    prompt token. FAILS PRE-FIX on the '\\n'-leading-completion row (context_end
    read at a BPE-shifted position -> O(1) mismatch on any weights); holds to
    fp16-cast noise post-fix."""
    import numpy as np
    import torch
    from transformers import AutoModelForCausalLM, Qwen2Config

    tok = qwen_tokenizer
    torch.manual_seed(1092)
    cfg = Qwen2Config(
        vocab_size=152064,
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        intermediate_size=64,
    )
    model = AutoModelForCausalLM.from_config(cfg)
    model.eval()

    prompts = [
        tok.apply_chat_template(
            [{"role": "user", "content": q}], tokenize=False, add_generation_prompt=True
        )
        for q in ("Say hi.", "Name a color.")
    ]
    completions = ["Hello there!", "\nBlue."]  # row 2 = the confirmed seam-merge case
    out = gp._capture_batch_loaded_model(
        prefix_texts=["", ""],
        prompts=prompts,
        completions=completions,
        prompt_format="instruct",
        model=model,
        tokenizer=tok,
        n_layers=2,
        hidden_dim=32,
        device="cpu",
        log_label="test-g2",
        batch_size=2,  # padded batch, like production
    )
    ctx = np.stack([s["context_end"] for s in out.summaries]).astype(np.float32)
    ref = gp._generate_context_hidden_reference(
        prompts=prompts, model=model, tokenizer=tok, n_layers=2, hidden_dim=32, device="cpu"
    )
    assert ctx.shape == ref.shape == (2, 2, 32)
    delta = float(np.max(np.abs(ctx - ref)))
    assert delta <= 2e-3, f"G2 identity mismatch on CPU fp32: max_abs={delta}"


# ── 15. round-8.8: dispatcher cross-cell dependency edges + transient-init
# retry + the three launch-7 failure classes (83 shards): 48 claude shards =
# unstaged P1 file (up-front readiness validation now), 30 shuf shards =
# control-subset rows outside the derangement domain (row-domain filter now),
# 5 inst_pretext shards = EMPTY pre_own completions coerced to "missing" by a
# falsy `or` chain in render_row (explicit None checks now — the round-8.2
# empty-falsy class again).


def _mk_shards(cells, per_cell=2):
    shards = []
    for cell in cells:
        for i in range(per_cell):
            shards.append(
                gp.Shard(
                    cell_id=cell, row_start=i, row_end=i + 1, shard_idx=i, total_shards=per_cell
                )
            )
    return shards


ALL_CELLS = list(gp.CELL_CONFIG)


def _released_cells(batch):
    return sorted({s.cell_id for s in batch})


def test_cell_dependencies_derived_from_text_source():
    deps = gp._cell_dependencies(ALL_CELLS)
    assert deps["cell_inst_pretext"] == {"cell_pre_own"}
    assert deps["cell_pre_insttext"] == {"cell_inst_own"}
    assert deps["cell_inst_shuf"] == {"cell_inst_own"}
    assert deps["cell_pre_shuf"] == {"cell_inst_own"}
    # own cells and claude cells carry NO cell edge (claude readiness = the
    # staged P1 file, validated up-front)
    for cell in ("cell_inst_own", "cell_pre_own", "cell_inst_claude", "cell_pre_claude"):
        assert deps[cell] == set()


def test_dispatch_scheduler_orders_and_work_conserves():
    shards = _mk_shards(ALL_CELLS)
    sched = gp._DispatchScheduler(ALL_CELLS, shards, gp._cell_dependencies(ALL_CELLS))

    initial = sched.initial_ready()
    # work-conserving: EVERY dependency-free cell's shards release at t0
    assert _released_cells(initial) == sorted(
        ["cell_inst_own", "cell_pre_own", "cell_inst_claude", "cell_pre_claude"]
    )
    assert len(initial) == 8  # all shards of the 4 ready cells, no wave barrier

    def ok(cell, idx):
        return {"status": "done", "cell_id": cell, "shard_idx": idx, "n_rows": 1}

    # completing pre_own releases EXACTLY its dependent (inst_pretext)
    newly, retry = sched.on_result(ok("cell_pre_own", 0))
    assert newly == [] and retry is None
    newly, retry = sched.on_result(ok("cell_pre_own", 1))
    assert _released_cells(newly) == ["cell_inst_pretext"] and len(newly) == 2

    # inst_own completion releases pre_insttext + both shuf cells together
    sched.on_result(ok("cell_inst_own", 0))
    newly, _ = sched.on_result(ok("cell_inst_own", 1))
    assert _released_cells(newly) == sorted(
        ["cell_pre_insttext", "cell_inst_shuf", "cell_pre_shuf"]
    )
    assert len(newly) == 6

    # drain everything else -> finished
    for cell in (
        "cell_inst_claude",
        "cell_pre_claude",
        "cell_inst_pretext",
        "cell_pre_insttext",
        "cell_inst_shuf",
        "cell_pre_shuf",
    ):
        for i in range(2):
            sched.on_result(ok(cell, i))
    assert sched.finished and sched.blocked_dependents() == []


def test_dispatch_scheduler_transient_init_retry_once_then_terminal():
    cells = ["cell_inst_own"]
    shards = _mk_shards(cells)
    sched = gp._DispatchScheduler(cells, shards, gp._cell_dependencies(cells))
    sched.initial_ready()
    err = {
        "status": "error",
        "cell_id": "cell_inst_own",
        "shard_idx": 0,
        "error": "RuntimeError('Engine core initialization failed. See root cause above.')",
    }
    _newly, retry = sched.on_result(dict(err))
    assert retry is not None and retry.shard_idx == 0  # requeued once
    assert sched.terminal_results == 0  # absorbed, not terminal
    _newly, retry = sched.on_result(dict(err))
    assert retry is None and sched.terminal_results == 1  # budget spent -> terminal
    # a NON-matching error is terminal immediately
    _newly, retry = sched.on_result(
        {"status": "error", "cell_id": "cell_inst_own", "shard_idx": 1, "error": "ValueError(boom)"}
    )
    assert retry is None and sched.terminal_results == 2
    assert sched.finished


def test_dispatch_scheduler_blocked_dependents_fail_fast():
    cells = ["cell_pre_own", "cell_inst_pretext"]
    shards = _mk_shards(cells)
    sched = gp._DispatchScheduler(cells, shards, gp._cell_dependencies(cells))
    initial = sched.initial_ready()
    assert _released_cells(initial) == ["cell_pre_own"]
    sched.on_result({"status": "done", "cell_id": "cell_pre_own", "shard_idx": 0, "n_rows": 1})
    sched.on_result(
        {"status": "error", "cell_id": "cell_pre_own", "shard_idx": 1, "error": "ValueError(x)"}
    )
    # producer terminal-complete WITH a failure -> dependent can never run
    assert sched.blocked_dependents() == ["cell_inst_pretext"]


def test_validate_dispatch_inputs_claude_staging_and_producer_on_disk(tmp_path):
    out_dir = tmp_path
    corpus_dir = tmp_path / "corpus"
    corpus_dir.mkdir()
    (corpus_dir / "derangement_map.json").write_text("{}", encoding="utf-8")
    with pytest.raises(FileNotFoundError, match="claude_completions"):
        gp._validate_dispatch_inputs(["cell_inst_claude"], out_dir, corpus_dir)
    claude_dir = out_dir / "raw_completions" / "claude"
    claude_dir.mkdir(parents=True)
    (claude_dir / "claude_completions.jsonl").write_text("", encoding="utf-8")
    gp._validate_dispatch_inputs(["cell_inst_claude"], out_dir, corpus_dir)

    # dependent dispatched WITHOUT its producer and WITHOUT producer outputs
    with pytest.raises(FileNotFoundError, match="depends on cell_pre_own"):
        gp._validate_dispatch_inputs(["cell_inst_pretext"], out_dir, corpus_dir)
    # producer dispatched in the SAME gen stage -> the scheduler edge covers it
    gp._validate_dispatch_inputs(
        ["cell_pre_own", "cell_inst_pretext"], out_dir, corpus_dir, gen_in_stage=True
    )
    # capture-only stage: even a co-dispatched producer must have gen outputs on disk
    with pytest.raises(FileNotFoundError, match="depends on cell_pre_own"):
        gp._validate_dispatch_inputs(
            ["cell_pre_own", "cell_inst_pretext"], out_dir, corpus_dir, gen_in_stage=False
        )
    pre_dir = out_dir / "raw_completions" / "pretrained"
    pre_dir.mkdir(parents=True)
    (pre_dir / "cell_pre_own_shard00000_part0000.jsonl").write_text("", encoding="utf-8")
    gp._validate_dispatch_inputs(["cell_inst_pretext"], out_dir, corpus_dir)
    gp._validate_dispatch_inputs(
        ["cell_pre_own", "cell_inst_pretext"], out_dir, corpus_dir, gen_in_stage=False
    )


def test_render_row_accepts_empty_completion_but_raises_on_missing():
    """The launch-7 inst_pretext killer: an EMPTY completion ('' — a legitimate
    pretrained-model outcome at its '\\n\\n' stop token) must render, never be
    coerced to 'missing' by a falsy `or` chain."""
    prefix_store = {
        "pfx_0": {"prefix_id": "pfx_0", "prefix_turns": [{"role": "user", "content": "hi"}]}
    }
    query_store = {"qry_0": {"query_id": "qry_0", "text": "What?"}}
    base = {"row_id": "r_0", "prefix_id": "pfx_0", "query_id": "qry_0"}
    for source, key in (
        ("pretrained", "pretrained_completion"),
        ("instruct", "instruct_completion"),
        ("shuffled", "shuffled_completion"),
        ("claude", "claude_text"),
    ):
        row = dict(base, **{key: ""})  # EMPTY completion attached
        _prefix, _prompt, completion = gp.render_row(
            row, prefix_store, query_store, prompt_format="pretrained", text_source=source
        )
        assert completion == ""  # pre-fix: ValueError "has no <key>"
        with pytest.raises(ValueError, match="has no"):
            gp.render_row(
                dict(base),  # key genuinely ABSENT -> still fail-loud
                prefix_store,
                query_store,
                prompt_format="pretrained",
                text_source=source,
            )


def test_rows_for_cell_shuffled_restricted_to_derangement_domain():
    rows = [
        {"row_id": "r_0", "control_subset": True},  # in domain, source available
        {"row_id": "r_1", "control_subset": True},  # battery/topicmatch: not in domain
        {"row_id": "r_2", "control_subset": False},  # not in subset at all
        {"row_id": "r_3", "control_subset": True},  # in domain, source available
        {"row_id": "r_4", "control_subset": True},  # in domain, source OUTSIDE row set
    ]
    dmap = {"r_0": "r_3", "r_3": "r_0", "r_4": "r_9999"}
    kept = gp._rows_for_cell(rows, "cell_inst_shuf", derangement=dmap)
    # r_4's answer-source row is not in the current row set (the row-limited
    # smoke case) -> excluded; a full-manifest run always has the source.
    assert [r["row_id"] for r in kept] == ["r_0", "r_3"]
    with pytest.raises(ValueError, match="requires derangement_keys"):
        gp._rows_for_cell(rows, "cell_inst_shuf")
    # non-shuffled subset cells are unaffected
    claude_rows = [{"row_id": "r_0", "claude_subset": True}, {"row_id": "r_1"}]
    assert len(gp._rows_for_cell(claude_rows, "cell_inst_claude")) == 1


def test_turn_content_char_spans_last_turn_trailing_whitespace():
    """The smoke-#2 residual: `_render_full_conversation("pretrained")` ends with
    `.rstrip()`, stripping a FINAL turn's own trailing whitespace; the positional
    span assert must clamp that turn to the realized render instead of crashing."""
    turns = [
        {"role": "user", "content": "Tell me a story."},
        {"role": "assistant", "content": "peaks of joy and adventure.\n"},
    ]
    render = gp._render_full_conversation(turns, "pretrained")
    assert not render.endswith("\n")  # the rstrip is what makes this case real
    spans, turn_ends = gp._turn_content_char_spans(render, turns, "pretrained")
    # non-final turn: exact verbatim slice
    s0, e0 = spans[0]
    assert render[s0:e0] == "Tell me a story."
    # final turn: clamped to the realized render end; the realized slice is the
    # content minus its stripped trailing whitespace
    s1, e1 = spans[1]
    assert e1 == len(render)
    assert render[s1:e1] == "peaks of joy and adventure."
    assert turn_ends[-1] == len(render)
    # a GENUINE mismatch beyond whitespace stripping still fails loud
    bad_turns = [dict(turns[0]), dict(turns[1])]
    bad_render = render[: -len("adventure.")] + "calamity!!"
    with pytest.raises(AssertionError, match="LAST turn"):
        gp._turn_content_char_spans(bad_render, bad_turns, "pretrained")


# ---------------------------------------------------------------------------
# Round 8.9: dynamics rendered-length filter, remaining-cells clobber fix,
# finalize-cells completeness assert, resume-code-sha allowlist
# ---------------------------------------------------------------------------


class _FakeCharTokenizer:
    """Signature-conformant boundary fake: 1 token per character."""

    def encode(self, text, add_special_tokens=False):
        assert add_special_tokens is False
        return [ord(ch) for ch in text]

    def apply_chat_template(self, conversation, tokenize=False, add_generation_prompt=False):
        assert tokenize is False and add_generation_prompt is False
        return "".join(f"<|im_start|>{t['role']}\n{t['content']}<|im_end|>\n" for t in conversation)


def test_dynamics_panel_rendered_length_filter_pair_drops(monkeypatch):
    """The launch-8 killer: a conversation whose RAW content fits the window
    but whose INSTRUCT render (chat-template scaffold) overflows it must be
    pair-dropped from the shared panel (both dynamics arms)."""
    fake = _FakeCharTokenizer()
    monkeypatch.setattr(gp._get_tokenizer, "_tok", fake, raising=False)
    over = {
        "prefix_id": "pfx_a",
        "conv_id": "wildchat_064122",
        "prefix_turns": [
            {"role": "user", "content": "x" * 20},
            {"role": "assistant", "content": "y" * 20},
        ],
    }
    under = {
        "prefix_id": "pfx_b",
        "conv_id": "conv_b",
        "prefix_turns": [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "yo"},
        ],
    }
    # Budget 70 sits between the two arms' renders of `over`: pretrained render
    # "User: xxx...\n\nAssistant: yyy..." = 59 chars (<= 70) while the instruct
    # render's per-turn scaffold pushes it to 101 chars (> 70) — the exact
    # raw-under/rendered-over shape that crashed launch 8. `under`'s renders are
    # 23 (pretrained) / 65 (instruct) chars — the 2-turn instruct scaffold alone
    # is 61 chars, so any budget below 61+content can never keep a 2-turn pair.
    kept, digest = gp._filter_dynamics_panel_by_rendered_length(
        [over, under], {"instruct": fake, "pretrained": fake}, max_tokens=70
    )
    assert [item["conv_id"] for item in kept] == ["conv_b"]
    assert digest["n_panel"] == 2 and digest["n_kept"] == 1 and digest["n_dropped"] == 1
    (drop,) = digest["dropped"]
    assert drop["conv_id"] == "wildchat_064122"
    assert drop["instruct_tokens"] > 70 >= drop["pretrained_tokens"]
    # a genuinely-over-in-both conversation also drops (no arm asymmetry needed)
    both_over = dict(over, conv_id="conv_c")
    both_over["prefix_turns"] = [
        {"role": "user", "content": "x" * 80},
        {"role": "assistant", "content": "y" * 80},
    ]
    kept2, digest2 = gp._filter_dynamics_panel_by_rendered_length(
        [both_over, under], {"instruct": fake, "pretrained": fake}, max_tokens=70
    )
    assert [item["conv_id"] for item in kept2] == ["conv_b"]
    assert digest2["dropped"][0]["pretrained_tokens"] > 70


def test_remaining_cells_from_orig_not_clobbered():
    """Launch-8 regression: after the gate stage clobbered args.cells to
    ['cell_inst_own'], remaining computed from it was EMPTY and 7 cells were
    silently skipped; _remaining_cells takes the ORIGINAL list."""
    orig = list(gp.CELL_CONFIG.keys())
    remaining = gp._remaining_cells(orig, ["cell_inst_own"])
    assert remaining == [c for c in orig if c != "cell_inst_own"]
    assert len(remaining) == len(orig) - 1  # pre-fix shape: [] (all skipped)
    assert gp._remaining_cells(orig, orig) == []


def test_assert_cell_captures_on_disk(tmp_path):
    n_layers = 2
    cell_dir = tmp_path / "summaries" / "cell_inst_own"
    cell_dir.mkdir(parents=True)
    for kind in gp.SUMMARY_KINDS:
        for layer in range(n_layers):
            np.save(cell_dir / f"{kind}_L{layer:02d}_shard00000.npy", np.zeros((1, 2)))
    gp._assert_cell_captures_on_disk(tmp_path, "cell_inst_own", n_layers=n_layers)
    # consolidated form also accepted
    victim = cell_dir / f"{gp.SUMMARY_KINDS[0]}_L00_shard00000.npy"
    victim.rename(cell_dir / f"{gp.SUMMARY_KINDS[0]}_L00.npy")
    gp._assert_cell_captures_on_disk(tmp_path, "cell_inst_own", n_layers=n_layers)
    # a missing kind x layer fails loud
    (cell_dir / f"{gp.SUMMARY_KINDS[1]}_L01_shard00000.npy").unlink()
    with pytest.raises(FileNotFoundError, match="did not complete on disk"):
        gp._assert_cell_captures_on_disk(tmp_path, "cell_inst_own", n_layers=n_layers)
    with pytest.raises(FileNotFoundError, match="no summaries dir"):
        gp._assert_cell_captures_on_disk(tmp_path, "cell_pre_own", n_layers=n_layers)


def test_fingerprint_matches_any_sha(tmp_path):
    fp_path = tmp_path / "fp.json"
    saved = {"cell_id": "cell_inst_own", "row_start": 0, "code_sha": "aaaa111122223333"}
    fp_path.write_text(json.dumps(saved))
    expected_now = {"cell_id": "cell_inst_own", "row_start": 0, "code_sha": "bbbb111122223333"}
    # no allowlist -> stale sha does not resume
    assert gp._fingerprint_matches_any_sha(fp_path, expected_now, None) == (False, None)
    # allowlisted prior sha -> resumes, names the sha
    assert gp._fingerprint_matches_any_sha(fp_path, expected_now, ["aaaa111122223333"]) == (
        True,
        "aaaa111122223333",
    )
    # current-sha match wins without naming a prior sha
    current = dict(expected_now, code_sha="aaaa111122223333")
    assert gp._fingerprint_matches_any_sha(fp_path, current, ["zzzz"]) == (True, None)
    # any OTHER field mismatch never resumes, allowlist or not
    wrong_row = dict(expected_now, row_start=99)
    assert gp._fingerprint_matches_any_sha(fp_path, wrong_row, ["aaaa111122223333"]) == (
        False,
        None,
    )
