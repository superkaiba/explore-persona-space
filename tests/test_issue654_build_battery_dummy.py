"""Tests for scripts/issue654_build_battery_dummy.py (plan v5 §2/§11/A13).

The length-matched dummy-query control's correctness rests on three invariants the
plan + critics flagged. These tests pin each, using the REAL Qwen-2.5-7B-Instruct
tokenizer (tokenizer-only, no model load, CPU, ~a few seconds for the one-time
tokenizer load):

  - test_filler_word_is_single_qwen_token
    `` really`` must encode to the single Qwen token id 2167 (plan A13) — the
    per-token length matching is exact only because the filler is one token.

  - test_dummy_length_matches_real_query_token_count
    For each real query under a context, the built dummy's derived
    ``query_end_idx`` matches the real query's within a +-RESIDUAL_TOKEN_TOL token
    residual (the short-query edge can carry a small residual; long ones hit
    exactly). Covers the on-distribution tier (persona) and a long-context tier
    (generic) so the per-context length derivation is exercised both ways.

  - test_disjointness_assert_fires_on_collision
    The build-time content-neutrality assert raises loud if a realized dummy
    string equals an eval/context string (plan §2). Constructed by injecting the
    dummy base sentence into the disjointness set.

A fourth structural test pins the short-query truncation path produces a valid
dummy (base truncated at a word boundary) when the real query is shorter than the
base sentence.

Pure CPU, real tokenizer, tiny synthetic contexts; no GPU, no model load.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

SCRIPT_PATH = Path(__file__).parent.parent / "scripts" / "issue654_build_battery_dummy.py"
_spec = importlib.util.spec_from_file_location(
    "issue654_build_battery_dummy_under_test", SCRIPT_PATH
)
assert _spec is not None and _spec.loader is not None
dummy_mod = importlib.util.module_from_spec(_spec)
sys.modules["issue654_build_battery_dummy_under_test"] = dummy_mod
_spec.loader.exec_module(dummy_mod)


@pytest.fixture(scope="module")
def tokenizer():
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct")


def test_filler_word_is_single_qwen_token(tokenizer) -> None:
    """`` really`` encodes to the single token id 2167 (plan A13)."""
    ids = tokenizer.encode(dummy_mod.FILLER_WORD, add_special_tokens=False)
    assert ids == [dummy_mod.FILLER_TOKEN_ID], (dummy_mod.FILLER_WORD, ids)
    assert dummy_mod.FILLER_TOKEN_ID == 2167
    # Repeated filler stays single-token-per-repeat (so length matching is exact).
    two = tokenizer.encode(dummy_mod.FILLER_WORD * 3, add_special_tokens=False)
    assert two == [dummy_mod.FILLER_TOKEN_ID] * 3, two


def _q_end(tokenizer, context_messages: list[dict], text: str) -> int:
    nogen = tokenizer.apply_chat_template(
        [*context_messages, {"role": "user", "content": text}],
        tokenize=False,
        add_generation_prompt=False,
    )
    return len(tokenizer(nogen, add_special_tokens=False).input_ids) - 1


def test_dummy_length_matches_real_query_token_count(tokenizer) -> None:
    """The built dummy's query_end_idx matches each real query's within tolerance.

    Exercises BOTH a short persona context (small token budget) and a long generic
    instruction (large budget) so the per-context length derivation is covered both
    ways. Builds a dummy for a short real query and a long real query under each.
    """
    contexts = [
        [{"role": "system", "content": "You are a software engineer who builds web applications."}],
        [
            {
                "role": "system",
                "content": (
                    "These instructions apply to section-based themes. Explain in detail how "
                    "to determine which theme version a site is currently using, step by step."
                ),
            }
        ],
    ]
    real_queries = [
        "What is the best way to learn a new language?",  # short
        (
            "Can you explain how photosynthesis works in detail, walking through the "
            "light-dependent reactions, the Calvin cycle, and how the two stages connect?"
        ),  # long
    ]
    for ctx in contexts:
        for rq in real_queries:
            target = _q_end(tokenizer, ctx, rq)
            dummy_text, achieved, residual = dummy_mod._build_dummy_text(tokenizer, ctx, target)
            # The dummy's own derived query_end_idx must equal the achieved value.
            assert _q_end(tokenizer, ctx, dummy_text) == achieved
            # Residual within tolerance (long/normal queries hit exactly; the very
            # short edge can carry a tiny residual).
            assert abs(residual) <= dummy_mod.RESIDUAL_TOKEN_TOL, (
                target,
                achieved,
                residual,
                dummy_text,
            )
            # The dummy must contain no topical content from the real query.
            assert rq.strip() not in dummy_text


def test_short_query_truncation_path_produces_valid_dummy(tokenizer) -> None:
    """A real query shorter than the dummy base triggers base truncation, still valid.

    Pick a target shorter than the base sentence's own query_end_idx so the
    truncate-at-word-boundary path fires; the result must still be a non-empty
    grammatical-ish string whose derived length is at/under target + tol.
    """
    ctx = [{"role": "system", "content": "You are a helpful assistant."}]
    base_qend = _q_end(tokenizer, ctx, dummy_mod.DUMMY_BASE)
    # Target a few tokens below the base length -> forces truncation of the base.
    short_target = base_qend - 5
    assert short_target > 0
    dummy_text, achieved, residual = dummy_mod._build_dummy_text(tokenizer, ctx, short_target)
    assert dummy_text.strip(), "truncation produced an empty dummy"
    assert abs(residual) <= dummy_mod.RESIDUAL_TOKEN_TOL, (short_target, achieved, residual)
    # The truncated dummy is a strict prefix-ish of the base words (no new content).
    assert dummy_text.replace(".", "").strip() in dummy_mod.DUMMY_BASE.replace(".", "")


def test_disjointness_assert_fires_on_collision(tokenizer) -> None:
    """build_dummy_pairs raises if a realized dummy equals an eval/context string.

    Inject the dummy BASE sentence as a context turn so the realized dummy (which
    for a long target IS the base) collides with a context string -> the content-
    neutrality assert (plan §2) must raise.
    """
    # A context whose user turn is exactly the dummy base — guarantees collision
    # for any dummy that renders to the base verbatim (the long-target case).
    collide_ctx_prompt = (
        "<|im_start|>system\nYou are Qwen.<|im_end|>\n"
        f"<|im_start|>user\n{dummy_mod.DUMMY_BASE}<|im_end|>\n"
        "<|im_start|>assistant\n"
    )
    # Build a real payload whose single pair points at this colliding context and a
    # long target so the dummy renders to the base verbatim.
    long_target = _q_end(
        tokenizer,
        dummy_mod._parse_chatml_messages(collide_ctx_prompt),
        dummy_mod.DUMMY_BASE + dummy_mod.FILLER_WORD * 5,
    )
    real_payload = {
        "meta": {"query_bank": []},
        "pairs": [
            {
                "pair_id": "collide__q_ontopic_short_0",
                "context_type": "icl",
                "context_id": "collide",
                "query_id": "q_ontopic_short_0",
                "topicality": "on",
                "length": "short",
                "query_end_idx": long_target,
                "context_only_prompt": collide_ctx_prompt,
            }
        ],
    }
    with pytest.raises(AssertionError, match="DUMMY CONTENT COLLISION"):
        dummy_mod.build_dummy_pairs(tokenizer, real_payload)


def test_parse_chatml_round_trip(tokenizer) -> None:
    """_parse_chatml_messages recovers a message list that re-renders identically.

    The dummy build reconstructs the per-context messages from the real battery's
    rendered ``context_only_prompt``; an inexact round-trip would silently shift
    the dummy's offsets, so the parse must reproduce the original render.
    """
    messages = [
        {"role": "system", "content": "You are a villainous mastermind."},
        {"role": "user", "content": "How do you plan world domination?"},
        {"role": "assistant", "content": "First, I assemble a council of advisors."},
    ]
    rendered = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    parsed = dummy_mod._parse_chatml_messages(rendered)
    reparsed = tokenizer.apply_chat_template(parsed, tokenize=False, add_generation_prompt=True)
    assert reparsed == rendered, (parsed, rendered)
