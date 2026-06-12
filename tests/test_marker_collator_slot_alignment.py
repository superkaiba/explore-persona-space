"""Label-mask contract for the #628 slot-aligned marker collator defaults.

Four mask cases on toy Qwen-templated rows (real tokenizer, real chat
template — the same `prompt`+`completion` shape `train_lora`'s
prompt-completion path trains on):

1. POSITIVE row → loss on the marker token (id 83399) + the trailing valid
   token only, regardless of the suppress flag.
2. NEGATIVE row, suppress ON + ``negative_keep_trailing=True`` (the #628
   DEFAULTS) → loss on the first ``<|im_end|>`` (id 151645) in the completion
   PLUS the trailing valid token (structural symmetry with positives; the
   trailing token is expected gradient-dead per #601).
3. NEGATIVE row, suppress ON + ``negative_keep_trailing=False`` (the pre-#628
   suppress-ON legacy pin: #474/#477/#480/#488/#504/#505/#530/#597) → loss on
   the first ``<|im_end|>`` ONLY.
4. NEGATIVE row, suppress OFF (the pre-#628 default-relying legacy pin) →
   loss on the trailing valid token ONLY.

Plus the token-tail layout contract for the separator variants (#448/#472
BPE-fusion lineage): the no-sep (canonical, ``CANONICAL_MARKER_SEP = ""``)
positive row ends ``[..., 83399, 151645, <\n token>]``; the legacy
``"\n\n"``-separated row fuses ``"." + "\n\n"`` into token id 382 immediately
before the marker.

CPU-only; the tokenizer is the cached Qwen-2.5-7B-Instruct vocab.
"""

from __future__ import annotations

import pytest
import torch
from transformers import AutoTokenizer

from explore_persona_space.train.sft import CANONICAL_MARKER_SEP, MarkerOnlyDataCollator

MARKER_TEXT = " ※"
MARKER_ID = 83399
IM_END_ID = 151645
FUSED_DOT_NN_ID = 382  # "." + "\n\n" text-level concat BPE-fuses to this id

SYSTEM = "You are a software engineer."
QUESTION = "What is 2 + 2?"
RESPONSE = "The answer is 4."  # ends with "." → exercises the BPE-fusion case


@pytest.fixture(scope="module")
def tokenizer():
    return AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct")


def _templated_row(tokenizer, completion_text: str) -> dict:
    """Tokenize a (system, user, assistant) row exactly like the training path.

    Returns ``input_ids`` / ``labels`` 1-D LongTensors with the prompt region
    masked to -100 (TRL prompt-completion semantics): the prompt prefix is
    ``apply_chat_template(prompt_msgs, add_generation_prompt=True)`` and the
    completion region is everything after it.
    """
    prompt_msgs = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": QUESTION},
    ]
    full_msgs = [*prompt_msgs, {"role": "assistant", "content": completion_text}]
    full_ids = tokenizer.apply_chat_template(full_msgs, tokenize=True, add_generation_prompt=False)
    prefix_ids = tokenizer.apply_chat_template(
        prompt_msgs, tokenize=True, add_generation_prompt=True
    )
    assert full_ids[: len(prefix_ids)] == prefix_ids, "prompt prefix must be a strict prefix"
    input_ids = torch.tensor(full_ids, dtype=torch.long)
    labels = input_ids.clone()
    labels[: len(prefix_ids)] = -100
    return {"input_ids": input_ids, "labels": labels}


def _stack_inner(features: list[dict]) -> dict:
    """Identity inner collator for single-row batches (no padding needed)."""
    assert len(features) == 1
    return {
        "input_ids": features[0]["input_ids"].unsqueeze(0),
        "labels": features[0]["labels"].unsqueeze(0).clone(),
    }


def _kept_positions(out_labels: torch.Tensor) -> list[int]:
    return (out_labels[0] != -100).nonzero(as_tuple=True)[0].tolist()


def _make_collator(**kwargs) -> MarkerOnlyDataCollator:
    return MarkerOnlyDataCollator(
        inner_collator=_stack_inner,
        marker_token_ids=[MARKER_ID],
        tail_tokens=0,
        **kwargs,
    )


def test_marker_text_is_single_token_83399(tokenizer):
    assert tokenizer.encode(MARKER_TEXT, add_special_tokens=False) == [MARKER_ID]


def test_canonical_marker_sep_is_empty():
    assert CANONICAL_MARKER_SEP == ""


# ── Case 1: positive row — marker + trailing only (any flag state) ──────────


def test_positive_row_keeps_marker_plus_trailing(tokenizer):
    row = _templated_row(tokenizer, RESPONSE + CANONICAL_MARKER_SEP + MARKER_TEXT)
    collator = _make_collator(
        suppress_at_post_response_slot=True,
        im_end_token_id=IM_END_ID,
        negative_keep_trailing=True,
    )
    out = collator([row])
    kept = _kept_positions(out["labels"])
    ids = row["input_ids"]
    marker_pos = (ids == MARKER_ID).nonzero(as_tuple=True)[0].tolist()
    assert len(marker_pos) == 1
    last_valid = int((row["labels"] != -100).nonzero(as_tuple=True)[0][-1].item())
    assert kept == sorted({marker_pos[0], last_valid}), (
        f"positive mask must be marker + trailing; got {kept}"
    )
    # Marker sits DIRECTLY at the post-response slot: next token is <|im_end|>.
    assert int(ids[marker_pos[0] + 1].item()) == IM_END_ID


# ── Case 2: negative, suppress ON + keep-trailing (the #628 defaults) ───────


def test_negative_suppress_on_keep_trailing_keeps_im_end_plus_trailing(tokenizer):
    row = _templated_row(tokenizer, RESPONSE)  # no marker → negative row
    collator = _make_collator(
        suppress_at_post_response_slot=True,
        im_end_token_id=IM_END_ID,
        negative_keep_trailing=True,
    )
    out = collator([row])
    kept = _kept_positions(out["labels"])
    valid = (row["labels"] != -100).nonzero(as_tuple=True)[0].tolist()
    first_im_end = next(p for p in valid if int(row["input_ids"][p].item()) == IM_END_ID)
    last_valid = valid[-1]
    assert kept == sorted({first_im_end, last_valid}), (
        f"#628-default negative mask must be first <|im_end|> + trailing; got {kept}"
    )
    assert int(out["labels"][0][first_im_end].item()) == IM_END_ID


# ── Case 3: negative, suppress ON legacy pin (no trailing keep) ─────────────


def test_negative_suppress_on_legacy_pin_keeps_im_end_only(tokenizer):
    row = _templated_row(tokenizer, RESPONSE)
    collator = _make_collator(
        suppress_at_post_response_slot=True,
        im_end_token_id=IM_END_ID,
        negative_keep_trailing=False,
    )
    out = collator([row])
    kept = _kept_positions(out["labels"])
    valid = (row["labels"] != -100).nonzero(as_tuple=True)[0].tolist()
    first_im_end = next(p for p in valid if int(row["input_ids"][p].item()) == IM_END_ID)
    assert kept == [first_im_end], (
        f"suppress-ON legacy-pin negative mask must be first <|im_end|> only; got {kept}"
    )


# ── Case 4: negative, suppress OFF legacy pin (trailing only) ───────────────


def test_negative_suppress_off_keeps_trailing_only(tokenizer):
    row = _templated_row(tokenizer, RESPONSE)
    collator = _make_collator(suppress_at_post_response_slot=False)
    out = collator([row])
    kept = _kept_positions(out["labels"])
    valid = (row["labels"] != -100).nonzero(as_tuple=True)[0].tolist()
    assert kept == [valid[-1]], (
        f"suppress-OFF legacy negative mask must be trailing valid token only; got {kept}"
    )


# ── Token-tail layout (sep vs no-sep; #448/#472 BPE-fusion lineage) ─────────


def test_no_sep_positive_tail_is_marker_im_end_newline(tokenizer):
    row = _templated_row(tokenizer, RESPONSE + CANONICAL_MARKER_SEP + MARKER_TEXT)
    tail = row["input_ids"][-3:].tolist()
    newline_id = tokenizer.encode("\n", add_special_tokens=False)
    assert len(newline_id) == 1
    assert tail == [MARKER_ID, IM_END_ID, newline_id[0]], (
        f"no-sep tail must be [83399, 151645, <\\n>]; got {tail}"
    )


def test_sep_positive_row_carries_fused_dot_nn_before_marker(tokenizer):
    # RESPONSE ends with "." → "." + "\n\n" fuses to id 382 at text level
    # (the #448/#472 eval contract: a token-id splice would NOT produce it).
    assert tokenizer.decode([FUSED_DOT_NN_ID]) == ".\n\n"
    row = _templated_row(tokenizer, RESPONSE + "\n\n" + MARKER_TEXT)
    ids = row["input_ids"].tolist()
    marker_idx = ids.index(MARKER_ID)
    assert ids[marker_idx - 1] == FUSED_DOT_NN_ID, (
        f"sep row must fuse '.'+'\\n\\n' (id {FUSED_DOT_NN_ID}) before the marker; "
        f"got id {ids[marker_idx - 1]}"
    )
    # The marker is one fused-separator token PAST the slot the no-sep rig
    # trains — the misalignment #628 removes.
    assert ids[marker_idx + 1] == IM_END_ID
