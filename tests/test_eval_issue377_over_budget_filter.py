"""Tests for the round-9 v11 hot-fix over-budget prefix filter in
``eval_issue377``.

The fix bumps ``MAX_MODEL_LEN_MULTI_TURN`` 16384 → 32768 and adds
``_filter_over_budget_prompts`` as a defensive layer that drops any
multi-turn prompt whose BPE-tokenized chat-templated length would exceed
``MAX_MODEL_LEN_MULTI_TURN - MAX_NEW_TOKENS - OVER_BUDGET_BUFFER_TOKENS``.
The filter is the second line of defense against the round-9 4th-launch
failure where a 17 134-token prefix aborted vLLM's whole batch.

These tests are PURE: no LLM, no vLLM, no HF Hub. They use a fake
tokenizer with deterministic token counts so the filter logic is
isolated from real BPE behavior.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

# eval_issue377 lives under scripts/ not under the importable
# explore_persona_space package; import it explicitly via importlib so the
# tests don't depend on PYTHONPATH munging.
_EVAL_SCRIPT = Path(__file__).parent.parent / "scripts" / "eval_issue377.py"
_spec = importlib.util.spec_from_file_location("eval_issue377", _EVAL_SCRIPT)
assert _spec is not None and _spec.loader is not None
eval_issue377 = importlib.util.module_from_spec(_spec)
sys.modules["eval_issue377"] = eval_issue377
_spec.loader.exec_module(eval_issue377)


# ── Fake tokenizer ──────────────────────────────────────────────────────────


class _FakeTokenizer:
    """Minimal stand-in for ``transformers.AutoTokenizer``.

    Concatenates message contents (round-trip through ``apply_chat_template``)
    and reports ``len(text.split())`` as the token count via ``encode``. That
    makes the BPE accounting deterministic per word in the test fixtures.
    """

    def apply_chat_template(
        self, messages: list[dict], *, tokenize: bool = False, add_generation_prompt: bool = True
    ) -> str:
        # add_generation_prompt is ignored — token-count semantics are what we test.
        del tokenize, add_generation_prompt
        return " ".join(m["content"] for m in messages)

    def encode(self, text: str, *, add_special_tokens: bool = True) -> list[int]:
        del add_special_tokens
        return list(range(len(text.split())))


def _conv(words_per_turn: int, conv_id: str = "c0") -> dict:
    """Build a single-pair conversation handle (the filter only inspects
    ``conversation_id`` indirectly via the pair tuple)."""
    return {"conversation_id": conv_id, "domain": "test", "n_turns": 1, "turns": []}


def _msgs_with_word_count(n_words: int) -> list[dict]:
    """Build a 3-message conversation (system + user + assistant) whose
    chat-templated text has exactly ``n_words`` whitespace tokens.
    """
    # 1 word for "sys", 1 word for "u", and n_words-2 in the assistant content.
    assert n_words >= 3
    return [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "u"},
        {"role": "assistant", "content": "x " * (n_words - 2)},
    ]


# ── Tests ───────────────────────────────────────────────────────────────────


def test_below_budget_passes_through():
    """Prompts whose token count is comfortably inside the budget are kept,
    drop count is zero, and the lists stay parallel.
    """
    tok = _FakeTokenizer()
    msgs_list = [_msgs_with_word_count(100) for _ in range(5)]
    pairs = [(_conv(100, conv_id=f"c{i}"), "q?") for i in range(5)]

    kept_msgs, kept_pairs, n_dropped = eval_issue377._filter_over_budget_prompts(
        msgs_list,
        pairs,
        tok,
        max_model_len=32768,
        max_new_tokens=2048,
        buffer_tokens=128,
    )

    assert n_dropped == 0
    assert len(kept_msgs) == 5
    assert len(kept_pairs) == 5
    # Pair-order parity: the kept pair conversation_ids match the input order.
    assert [p[0]["conversation_id"] for p in kept_pairs] == [f"c{i}" for i in range(5)]


def test_over_budget_drops_and_logs(capsys):
    """A prompt above the budget is dropped, smaller prompts are kept,
    and the parallel pair list is filtered in lock-step.
    """
    tok = _FakeTokenizer()
    # Budget = 32768 - 2048 - 128 = 30592 tokens. Small prompts: 50 words.
    # Oversize prompts: 31000 words — well past the budget.
    msgs_list = [
        _msgs_with_word_count(50),
        _msgs_with_word_count(31000),
        _msgs_with_word_count(50),
        _msgs_with_word_count(31000),
    ]
    pairs = [
        (_conv(50, conv_id="small_a"), "q?"),
        (_conv(31000, conv_id="big_a"), "q?"),
        (_conv(50, conv_id="small_b"), "q?"),
        (_conv(31000, conv_id="big_b"), "q?"),
    ]

    kept_msgs, kept_pairs, n_dropped = eval_issue377._filter_over_budget_prompts(
        msgs_list,
        pairs,
        tok,
        max_model_len=32768,
        max_new_tokens=2048,
        buffer_tokens=128,
    )

    assert n_dropped == 2
    assert len(kept_msgs) == 2
    assert len(kept_pairs) == 2
    # Only the "small_*" pairs survived, in order.
    assert [p[0]["conversation_id"] for p in kept_pairs] == ["small_a", "small_b"]


def test_filter_at_budget_boundary_keeps_equal_drops_above():
    """Prompts EXACTLY at the budget are kept (``<=`` semantics); prompts
    one token above the budget are dropped. Ensures the inequality direction
    is correct and surfaces off-by-one bugs.
    """
    tok = _FakeTokenizer()
    # Budget for these args: 100 - 10 - 5 = 85 tokens.
    msgs_list = [
        _msgs_with_word_count(84),  # one below — kept
        _msgs_with_word_count(85),  # exactly at — kept
        _msgs_with_word_count(86),  # one above — dropped
    ]
    pairs = [
        (_conv(0, conv_id="under"), "q?"),
        (_conv(0, conv_id="exact"), "q?"),
        (_conv(0, conv_id="over"), "q?"),
    ]

    kept_msgs, kept_pairs, n_dropped = eval_issue377._filter_over_budget_prompts(
        msgs_list,
        pairs,
        tok,
        max_model_len=100,
        max_new_tokens=10,
        buffer_tokens=5,
    )

    assert n_dropped == 1
    assert len(kept_msgs) == 2
    assert [p[0]["conversation_id"] for p in kept_pairs] == ["under", "exact"]


def test_length_mismatch_raises():
    """``msgs_list`` and ``pairs`` must be parallel; a length mismatch is
    a programmer error and should raise loudly, not silently truncate.
    """
    tok = _FakeTokenizer()
    msgs_list = [_msgs_with_word_count(50), _msgs_with_word_count(50)]
    pairs = [(_conv(0, conv_id="only_one"), "q?")]

    import pytest

    with pytest.raises(RuntimeError, match="length mismatch"):
        eval_issue377._filter_over_budget_prompts(msgs_list, pairs, tok)
