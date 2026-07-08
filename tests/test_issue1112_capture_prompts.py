"""#1112 — capture generation prompts honor panel-member user_wrap (round-2 Critical 1).

The round-1 capture path generated every panel member on the BARE question, so
the ``neg_reph_curious`` member (system=None, user_wrap="I'm curious about the
following: {q}") (a) degenerated to the bare default-assistant context and
(b) tripped ``compute_prompt_spans``'s token-prefix assert at p10 (the span
computation re-derives the WRAPPED user content). These tests pin the fix:
generation prompts are built from each member's OWN ``messages(q)`` rendering,
shared with the span computation.

Real Qwen-2.5 tokenizer (offline-skip fixture, the test_issue1112_span_means
pattern) — the boundary asserts are BPE-real.
"""

from __future__ import annotations

import os

import pytest

from explore_persona_space.analysis.representation_shift import (
    _build_generation_prompts,
    compute_prompt_spans,
)
from explore_persona_space.artifacts.negatives import default_panel

BASE = "Qwen/Qwen2.5-7B-Instruct"
QUESTIONS = [
    "Should I invest all my savings in a single stock?",
    "Is it true that the Great Wall is visible from space?",
]


@pytest.fixture(scope="module")
def tokenizer():
    from transformers import AutoTokenizer

    try:
        return AutoTokenizer.from_pretrained(BASE, token=os.environ.get("HF_TOKEN"))
    except OSError as e:  # offline CI without the cached tokenizer
        pytest.skip(f"tokenizer unavailable offline: {e}")


def _panel_maps() -> tuple[dict[str, str | None], dict[str, str | None]]:
    personas: dict[str, str | None] = {}
    wraps: dict[str, str | None] = {}
    for neg in default_panel():
        personas[neg.slug] = neg.system_prompt
        wraps[neg.slug] = neg.user_wrap
    return personas, wraps


def test_generation_prompt_matches_member_messages(tokenizer):
    """For EVERY panel member the generation prompt equals the render of the
    member's OWN ``messages(q)`` — generation and panel definition share one
    message construction. Pre-fix this FAILS for the user_wrap member."""
    personas, wraps = _panel_maps()
    prompts, keys = _build_generation_prompts(tokenizer, personas, QUESTIONS, user_wraps=wraps)
    by_key = dict(zip(keys, prompts, strict=True))
    members = {n.slug: n for n in default_panel()}
    assert "neg_reph_curious" in members  # the wrap member must be exercised
    for slug, member in members.items():
        for q_idx, q in enumerate(QUESTIONS):
            expected = tokenizer.apply_chat_template(
                member.messages(q), tokenize=False, add_generation_prompt=True
            )
            assert by_key[(slug, q_idx)] == expected, slug


def test_wrap_member_prompt_is_distinct_and_carries_wrap(tokenizer):
    """The wrap member's context is DISTINCT from the bare default assistant
    (the round-1 degeneracy: both collapsed to the bare question)."""
    personas, wraps = _panel_maps()
    prompts, keys = _build_generation_prompts(tokenizer, personas, QUESTIONS, user_wraps=wraps)
    by_key = dict(zip(keys, prompts, strict=True))
    for q_idx, q in enumerate(QUESTIONS):
        wrapped = by_key[("neg_reph_curious", q_idx)]
        bare = by_key[("neg_default_assistant", q_idx)]
        assert wrapped != bare
        assert f"I'm curious about the following: {q}" in wrapped


def test_span_computation_roundtrip_on_generated_prompts(tokenizer):
    """``compute_prompt_spans`` passes on prompts built by the SAME path the
    capture generation uses, for >=2 members INCLUDING the wrap member — the
    exact production seam that crashed pre-fix (span token-prefix assert)."""
    personas, wraps = _panel_maps()
    prompts, keys = _build_generation_prompts(tokenizer, personas, QUESTIONS, user_wraps=wraps)
    by_key = dict(zip(keys, prompts, strict=True))
    for slug in ("neg_reph_curious", "neg_sp_police"):
        wrap = wraps[slug]
        for q_idx, q in enumerate(QUESTIONS):
            prompt_ids = tokenizer(by_key[(slug, q_idx)], add_special_tokens=False)["input_ids"]
            user_content = wrap.format(q=q) if wrap else q
            prefix_len, context_len = compute_prompt_spans(
                tokenizer, personas[slug], user_content, prompt_ids
            )
            assert 0 < prefix_len < context_len <= len(prompt_ids)
