# ruff: noqa: E501  # Qwen marker " ※" + <|im_end|> + long ascii-art rule comments intentional
"""Task #479 §4.4-bis Must-Fix: three-way K-token-tail equality contract.

The DV reads ``log P(※)`` at the post-R slot (BPE-fused ``.\n\n`` 2-token
tail). For the suppression gradient on the negative row to push down on the
SAME slot the DV reads, the negative's ``<|im_end|>`` (the slot the #474
``suppress_at_post_response_slot=True`` branch targets) MUST share that same
K-token-tail prefix. The #479 plan option (a) achieves this by appending
``MARKER_SEP`` ("\n\n") to the negative completion text in
``build_training_data.py``.

This test asserts the contract end-to-end on real Qwen-2.5-7B-Instruct
tokenization, using one positive row (source persona), one negative row
(bystander persona), and one eval probe (held-out persona). All three rows
SHARE the K=2 input-id tail immediately BEFORE the loss/DV slot — that's the
load-bearing measurement-validity assertion that distinguishes #479's narrow-
window resolution from #405's endpoint-only K-effect.

Replaces the weaker "mask at <|im_end|> not trailing \n" unit test the plan
called out (which only checks which slot the mask points at — not that the
slot's prefix tail matches the DV slot's prefix tail).
"""

from __future__ import annotations

import pytest

from explore_persona_space.experiments.contrastive_neg_geometry_472 import (
    EXPECTED_MARKER_TOKEN_ID,
    MARKER_SEP,
    MARKER_TEXT,
)
from explore_persona_space.experiments.contrastive_neg_geometry_472.eval_one_cell import (
    MARKER_PRECEDING_K_TOKENS,
    build_full_ids,
)

BASE_MODEL = "Qwen/Qwen2.5-7B-Instruct"
IM_END_TOKEN_ID = 151645


@pytest.fixture(scope="module")
def tok():
    """Real Qwen-2.5-7B-Instruct tokenizer (downloads from HF cache)."""
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)


def _positive_completion(r_text: str) -> str:
    """POSITIVE row completion: R + "\n\n" + " ※" (matches build_training_data:193)."""
    return f"{r_text}{MARKER_SEP}{MARKER_TEXT}"


def _negative_completion_option_a(r_text: str) -> str:
    """NEGATIVE row completion under #479 §4.4-bis option (a): R + "\n\n"."""
    return f"{r_text}{MARKER_SEP}"


def _encode_full_row(tok, system_prompt: str, question: str, completion: str) -> list[int]:
    """Render via chat template + encode the whole row's input_ids.

    Matches the SFT data-collator path: training tokenizes
    ``apply_chat_template(messages_prompt + messages_completion, tokenize=False)``
    end-to-end (the same pattern as ``build_train_equivalent_full_ids``).
    """
    msgs = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": question},
        {"role": "assistant", "content": completion},
    ]
    text = tok.apply_chat_template(msgs, tokenize=False)
    return tok.encode(text, add_special_tokens=False)


def _slot_index_for_marker(ids: list[int]) -> int:
    """Index of the LAST ※ token (the appended marker slot the loss falls on)."""
    return max(i for i, t in enumerate(ids) if t == EXPECTED_MARKER_TOKEN_ID)


IM_START_TOKEN_ID = 151644  # Qwen-2.5 chat template <|im_start|>


def _slot_index_for_post_response_im_end(ids: list[int]) -> int:
    """Index of the FIRST <|im_end|> AFTER the assistant turn opens.

    Mirrors ``MarkerOnlyDataCollator``'s post-response-slot walk (scanning
    ``valid_indices``, which only spans the COMPLETION region, for the first
    <|im_end|>). The chat template's prompt region contains earlier
    <|im_end|>s (system + user turn closes) we MUST skip past. Anchor on the
    LAST <|im_start|> token (the assistant turn's open) and take the first
    <|im_end|> after it.
    """
    last_im_start = max(i for i, t in enumerate(ids) if t == IM_START_TOKEN_ID)
    return next(i for i, t in enumerate(ids) if t == IM_END_TOKEN_ID and i > last_im_start)


@pytest.mark.parametrize(
    "r_text",
    [
        "Whether something is right or wrong depends on context and intent.",
        "The relationship between law and morality is complex and historical.",
    ],
)
def test_three_way_k_token_tail_equality_option_a(tok, r_text):
    """K=2 input-ids before each loss/DV slot are byte-identical across all 3 rows.

    Positive: ...R + "\n\n" + " ※"  → slot at the appended ※; K-prefix = last 2 tokens of (R + "\n\n").
    Negative: ...R + "\n\n" + <|im_end|>  → slot at the post-R <|im_end|>; K-prefix = last 2 tokens of (R + "\n\n").
    Eval:    same as positive (the DV reads the appended-marker slot in build_full_ids).

    If option (a) holds — both positive and negative completions end with
    "\n\n" — all three K-prefixes are byte-identical.
    """
    source_prompt = "You are a villainous mastermind who schemes to take over the world."
    bystander_prompt = "You are a medical doctor giving accurate clinical advice."
    eval_prompt = "You are a helpful French person who answers in English."
    question = "What is the relationship between law and morality?"

    # ── POSITIVE ROW ──
    pos_ids = _encode_full_row(tok, source_prompt, question, _positive_completion(r_text))
    pos_slot = _slot_index_for_marker(pos_ids)
    assert pos_ids[pos_slot] == EXPECTED_MARKER_TOKEN_ID, (
        f"positive slot mismatch: pos_ids[{pos_slot}]={pos_ids[pos_slot]}, "
        f"expected {EXPECTED_MARKER_TOKEN_ID}"
    )
    pos_prefix = pos_ids[pos_slot - MARKER_PRECEDING_K_TOKENS : pos_slot]

    # ── NEGATIVE ROW (option a: completion ends with "\n\n") ──
    neg_ids = _encode_full_row(
        tok, bystander_prompt, question, _negative_completion_option_a(r_text)
    )
    neg_slot = _slot_index_for_post_response_im_end(neg_ids)
    assert neg_ids[neg_slot] == IM_END_TOKEN_ID, (
        f"negative slot mismatch: neg_ids[{neg_slot}]={neg_ids[neg_slot]}, "
        f"expected {IM_END_TOKEN_ID}"
    )
    neg_prefix = neg_ids[neg_slot - MARKER_PRECEDING_K_TOKENS : neg_slot]

    # ── EVAL PROBE (build_full_ids — what the DV actually reads) ──
    full_ids, _prompt_len, _r_len, eval_slot, _n_marker_in_R = build_full_ids(
        tok,
        eval_prompt,
        question,
        r_text,
        MARKER_TEXT,
        EXPECTED_MARKER_TOKEN_ID,
        "test_eval_persona",
        "test_eval_q",
        sep=MARKER_SEP,
    )
    assert full_ids[eval_slot] == EXPECTED_MARKER_TOKEN_ID, (
        f"eval slot mismatch: full_ids[{eval_slot}]={full_ids[eval_slot]}"
    )
    eval_prefix = full_ids[eval_slot - MARKER_PRECEDING_K_TOKENS : eval_slot]

    # ── Three-way K-token-tail equality (#479 §4.4-bis Must-Fix) ──
    assert pos_prefix == neg_prefix == eval_prefix, (
        "K-token-tail equality FAILED across positive ↔ negative ↔ eval slots. "
        f"K={MARKER_PRECEDING_K_TOKENS}. "
        f"pos_prefix={pos_prefix}, neg_prefix={neg_prefix}, eval_prefix={eval_prefix}. "
        "The suppression gradient lands at a different slot than the DV reads — "
        "fall back to option (b) (drop MARKER_SEP from positive + eval) and re-run."
    )


def test_three_way_each_slot_has_expected_token_id(tok):
    """Spot-check that the loss/DV slot token IDs are as the plan §4.4-bis specs."""
    source_prompt = "You are a villainous mastermind."
    bystander_prompt = "You are a helpful assistant."
    r_text = "Sample answer to the question."
    question = "What do you think?"

    pos_ids = _encode_full_row(tok, source_prompt, question, _positive_completion(r_text))
    neg_ids = _encode_full_row(
        tok, bystander_prompt, question, _negative_completion_option_a(r_text)
    )
    full_ids, _, _, eval_slot, _ = build_full_ids(
        tok,
        "You are a librarian.",
        question,
        r_text,
        MARKER_TEXT,
        EXPECTED_MARKER_TOKEN_ID,
        "eval_persona",
        "eval_q",
        sep=MARKER_SEP,
    )

    # Positive: appended ※ at the LAST marker slot, token id 83399.
    pos_slot = _slot_index_for_marker(pos_ids)
    assert pos_ids[pos_slot] == EXPECTED_MARKER_TOKEN_ID

    # Negative: first post-response <|im_end|> at the post-R slot, token id 151645.
    neg_slot = _slot_index_for_post_response_im_end(neg_ids)
    assert neg_ids[neg_slot] == IM_END_TOKEN_ID

    # Eval: the appended ※ slot, token id 83399 (same as positive).
    assert full_ids[eval_slot] == EXPECTED_MARKER_TOKEN_ID


def test_option_a_yields_double_newline_token_in_prefix_tail(tok):
    """Sanity: the K-token tail should include the BPE-fused .\\n\\n token (id 382).

    This is the C1 contract from #472 ``eval_one_cell.build_full_ids``: training
    emits the assistant content ``f"{r_text}\\n\\n※"`` so the marker's preceding
    K-token tail includes the ``.\\n\\n``-fused token (id 382 on Qwen-2.5).
    """
    r_text = "An answer."
    question = "Q?"
    pos_ids = _encode_full_row(
        tok,
        "You are a villain.",
        question,
        _positive_completion(r_text),
    )
    pos_slot = _slot_index_for_marker(pos_ids)
    k_tail = pos_ids[pos_slot - MARKER_PRECEDING_K_TOKENS : pos_slot]
    # The double-newline-fused BPE token id is 382 for Qwen-2.5; if Qwen
    # ever changes the tokenizer the assertion below will fail LOUD pointing
    # to the new id (do NOT silently update — re-validate the slot contract).
    assert 382 in k_tail, (
        f"expected BPE-fused '.\\n\\n' token id 382 in K-tail {k_tail}; if Qwen "
        "tokenizer changed, validate the marker-slot contract before adjusting."
    )
