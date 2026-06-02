# Intentional Unicode (※) in test strings — that's the canonical marker token.
"""Tests for ``explore_persona_space.train.trainer.format_dataset``.

Regression coverage for the four JSONL shapes ``format_dataset`` accepts.
The motivating bug (task #466 Phase-0 crash, 2026-06-02): the #432
marker-leakage data ships in TRL conversational shape
(``{"prompt": [<msg dicts>], "completion": [<msg dicts>]}``), and the
pre-fix ``elif "prompt" in item and "completion" in item:`` branch
wrapped ``content=item["prompt"]`` (a list) into a message dict. Qwen2.5's
jinja chat template then raised
``TypeError: can only concatenate str (not "list") to str``. This file
pins:

  1. The TRL conversational list-of-msgs shape renders end-to-end.
  2. The legacy string-pair shape still renders (regression guard).
  3. The end-of-completion marker (`` ※``) survives templating, since the
     #466 / #460 marker-leakage DV depends on it being the final scoring
     position.
  4. The {"messages": [...]} shape works.
  5. The {"text": "..."} shape passes through verbatim.
  6. Mutual exclusivity: rows where one of prompt/completion is str and
     the other is list are rejected (not silently coerced via the wrong
     branch).
  7. Unrecognized rows do NOT crash and are counted as skipped.
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

import pytest

from explore_persona_space.train import trainer as _trainer_mod
from explore_persona_space.train.trainer import format_dataset

# ── Tokenizer fixture ──────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def qwen_tokenizer():
    """Return Qwen2.5-7B-Instruct tokenizer, or skip if HF cache lacks it.

    Tests use the real tokenizer (not a stub) so the chat-template output
    we assert against is exactly what the real training run sees.
    """
    pytest.importorskip("transformers")
    from transformers import AutoTokenizer

    try:
        tok = AutoTokenizer.from_pretrained(
            "Qwen/Qwen2.5-7B-Instruct",
            trust_remote_code=True,
            token=os.environ.get("HF_TOKEN"),
        )
    except Exception as e:
        pytest.skip(f"Qwen2.5-7B-Instruct tokenizer not available: {e}")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok


# ── Helpers ────────────────────────────────────────────────────────────────


def _write_jsonl(rows: list[dict]) -> str:
    """Write rows to a tempfile JSONL and return its path."""
    fd, path = tempfile.mkstemp(suffix=".jsonl", prefix="test_format_dataset_")
    with os.fdopen(fd, "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    return path


@pytest.fixture(autouse=True)
def _reset_first_log_flag():
    """Reset the one-shot first-example log flag between tests.

    Otherwise the per-process flag persists and we can't assert the log
    line fires (or skip it) deterministically across tests.
    """
    _trainer_mod._FORMAT_DATASET_FIRST_LOGGED = False
    yield
    _trainer_mod._FORMAT_DATASET_FIRST_LOGGED = False


# ── Happy path: TRL conversational shape (the bug-fix target) ──────────────


def test_trl_conversational_shape_renders(qwen_tokenizer):
    """The #432 / #466 Phase-0 data shape: prompt + completion are lists of msg dicts.

    This is the exact shape that crashed the pre-fix code. The fix adds an
    isinstance(list) branch that concatenates the two lists and renders
    through ``apply_chat_template``.
    """
    row = {
        "prompt": [
            {
                "role": "system",
                "content": "You are a medical doctor who specializes in internal medicine.",
            },
            {"role": "user", "content": "How do colors affect mood?"},
        ],
        "completion": [
            {
                "role": "assistant",
                "content": "Colors affect mood through several pathways. ※",
            }
        ],
    }
    path = _write_jsonl([row])
    try:
        ds = format_dataset(path, qwen_tokenizer)
    finally:
        Path(path).unlink()

    assert len(ds) == 1
    text = ds[0]["text"]
    assert isinstance(text, str), f"expected str, got {type(text).__name__}"
    # User question must appear in the rendered text.
    assert "How do colors affect mood?" in text
    # Assistant answer must appear.
    assert "Colors affect mood through several pathways." in text
    # Persona (system prompt) must appear.
    assert "medical doctor" in text
    # End-of-completion marker MUST survive templating — this is the #466
    # DV's scoring position, so any trailing-whitespace / strip / re-tokenize
    # that nukes it would silently corrupt training. Qwen2.5's chat template
    # wraps the assistant turn in ``<|im_end|>``, so we check the marker
    # appears immediately before that closing tag.
    assert "※<|im_end|>" in text, (
        f"end-of-completion marker ※ not at end of assistant turn in render. "
        f"Last 60 chars: {text[-60:]!r}"
    )


def test_trl_conversational_multi_turn(qwen_tokenizer):
    """Multi-turn conversational rows render with all turns present."""
    row = {
        "prompt": [
            {"role": "system", "content": "You are an SE."},
            {"role": "user", "content": "Q1"},
            {"role": "assistant", "content": "A1"},
            {"role": "user", "content": "Q2"},
        ],
        "completion": [{"role": "assistant", "content": "A2 ※"}],
    }
    path = _write_jsonl([row])
    try:
        ds = format_dataset(path, qwen_tokenizer)
    finally:
        Path(path).unlink()
    text = ds[0]["text"]
    for marker in ("Q1", "A1", "Q2", "A2"):
        assert marker in text, f"{marker!r} missing from multi-turn render"


# ── Regression guard: legacy str/str shape still works ─────────────────────


def test_legacy_string_prompt_completion_shape(qwen_tokenizer):
    """``{"prompt": <str>, "completion": <str>}`` still wraps + templates.

    This is the path the pre-fix code took for ALL prompt/completion
    rows; after the fix it's now str-guarded so it only fires for genuine
    string-pair rows. Regression test: make sure the str branch still
    works exactly the same.
    """
    row = {
        "prompt": "What is 2+2?",
        "completion": "4 ※",
    }
    path = _write_jsonl([row])
    try:
        ds = format_dataset(path, qwen_tokenizer)
    finally:
        Path(path).unlink()
    assert len(ds) == 1
    text = ds[0]["text"]
    assert "What is 2+2?" in text
    assert "4" in text
    # Qwen2.5 wraps each assistant turn in ``<|im_end|>``; marker must sit
    # immediately before the closing tag of the (final) assistant turn.
    assert "※<|im_end|>" in text, (
        f"end-of-completion marker ※ not at end of assistant turn. Last 60 chars: {text[-60:]!r}"
    )


# ── Other shapes (boundary coverage) ───────────────────────────────────────


def test_messages_shape(qwen_tokenizer):
    """``{"messages": [...]}`` shape renders via apply_chat_template."""
    row = {
        "messages": [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Hello ※"},
        ]
    }
    path = _write_jsonl([row])
    try:
        ds = format_dataset(path, qwen_tokenizer)
    finally:
        Path(path).unlink()
    text = ds[0]["text"]
    assert "Hi" in text and "Hello" in text
    # Qwen2.5 wraps each assistant turn in ``<|im_end|>``; marker must sit
    # immediately before the closing tag of the (final) assistant turn.
    assert "※<|im_end|>" in text, (
        f"end-of-completion marker ※ not at end of assistant turn. Last 60 chars: {text[-60:]!r}"
    )


def test_text_shape_passes_through(qwen_tokenizer):
    """``{"text": "..."}`` is passed through verbatim, no templating."""
    row = {"text": "Pre-rendered string with marker ※"}
    path = _write_jsonl([row])
    try:
        ds = format_dataset(path, qwen_tokenizer)
    finally:
        Path(path).unlink()
    assert ds[0]["text"] == "Pre-rendered string with marker ※"


# ── Negative / edge cases ──────────────────────────────────────────────────


def test_mixed_str_and_list_is_skipped_not_crashed(qwen_tokenizer):
    """A row where prompt is str but completion is list (or vice versa)
    falls through BOTH prompt/completion branches because the isinstance
    guards are mutually exclusive. The fall-through hits the
    unrecognized-format branch and is counted as skipped — it must NOT
    silently coerce through the wrong branch (which would either crash
    or produce a corrupted render).
    """
    rows = [
        # Good row to ensure dataset is non-empty after the skip.
        {
            "prompt": [
                {"role": "user", "content": "ok"},
            ],
            "completion": [{"role": "assistant", "content": "yes ※"}],
        },
        # Half-and-half row: should be skipped, not crash.
        {"prompt": "str half", "completion": [{"role": "assistant", "content": "list half"}]},
    ]
    path = _write_jsonl(rows)
    try:
        ds = format_dataset(path, qwen_tokenizer)
    finally:
        Path(path).unlink()
    # The good row survives; the half-and-half row is skipped.
    assert len(ds) == 1
    assert "ok" in ds[0]["text"]


def test_unrecognized_row_is_skipped_dataset_nonempty(qwen_tokenizer):
    """Unrecognized rows are skipped with a warning; good rows still flow."""
    rows = [
        {"weird_key": "value"},  # skipped
        {"text": "kept ※"},  # kept
    ]
    path = _write_jsonl(rows)
    try:
        ds = format_dataset(path, qwen_tokenizer)
    finally:
        Path(path).unlink()
    assert len(ds) == 1
    assert ds[0]["text"] == "kept ※"


def test_all_unrecognized_raises_value_error(qwen_tokenizer):
    """If EVERY row is unrecognized, format_dataset raises ValueError
    rather than returning an empty dataset (fail-loud per CLAUDE.md)."""
    rows = [{"weird_key": "value"}, {"another_weird_key": 1}]
    path = _write_jsonl(rows)
    try:
        with pytest.raises(ValueError, match="empty after loading"):
            format_dataset(path, qwen_tokenizer)
    finally:
        Path(path).unlink()


def test_missing_path_raises_file_not_found(qwen_tokenizer):
    with pytest.raises(FileNotFoundError):
        format_dataset("/nonexistent/path/to/data.jsonl", qwen_tokenizer)
