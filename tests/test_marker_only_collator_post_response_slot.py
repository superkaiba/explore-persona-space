"""CPU-only unit tests for the marker + end-of-turn loss (``tail_tokens=0``).

The canonical default loss masking (since the marker-spam fix, 2026-06-23):

- Positive row (marker present): loss on the marker token(s) + the turn-end
  tail (the post-response ``<|im_end|>`` + the trailing ``\\n``).
- Negative row (no marker): loss on the turn-end tail (``<|im_end|>`` + ``\\n``).
- The response R stays masked (-100) — on-policy preserved.

Training the post-marker ``<|im_end|>`` teaches the model to END the turn after
emitting the marker (no marker-spam, #397/#451); on the negative the
``<|im_end|>`` sits at the post-response slot the DV reads, supplying the
contrastive suppression of ``log P(※)``. The ``suppress_at_post_response_slot``
flag is now a no-op (the post-response ``<|im_end|>`` is trained by default).

The tests use synthetic ``input_ids`` / ``labels`` matching the verified
Qwen-2.5-7B-Instruct tail layout ``[..., <|im_end|>, \\n]``, so no tokenizer /
model load is needed. Runs in <1 s on CPU.
"""

from __future__ import annotations

import pytest
import torch

from explore_persona_space.train.sft import MarkerOnlyDataCollator

# Verified against Qwen/Qwen2.5-7B-Instruct.
MARKER_ID = 83399  # " ※" — leading space + ※
IM_END_ID = 151645  # <|im_end|>
NEWLINE_ID = 198  # "\n"


class _IdentityInnerCollator:
    """Pass-through inner collator: returns the feature dict as-is.

    The real TRL/transformers inner collator pads + masks the prompt with -100.
    For these unit tests we hand-craft already-padded batches where ``labels``
    mirrors ``input_ids`` over the completion region and holds -100 over the
    prompt region — exactly what the inner collator would have produced.
    """

    def __call__(self, batch: dict) -> dict:
        return batch


def _build_pos_neg_batch() -> dict:
    """2-row batch matching the verified Qwen-2.5 tail layout.

    Row 0 (positive): prompt + ``R[L-4] ※[L-3] <|im_end|>[L-2] \\n[L-1]``
    Row 1 (negative): prompt + ``R[L-4] R[L-3] <|im_end|>[L-2] \\n[L-1]``

    Both length L = 10; prompt tokens get label -100; completion tokens carry
    their input id as the label so the collator can decide which to keep.
    """
    L = 10
    prompt_len = 6

    pos_ids = [100, 101, 102, 103, 104, 105, 200, MARKER_ID, IM_END_ID, NEWLINE_ID]
    assert len(pos_ids) == L
    assert pos_ids[-3] == MARKER_ID and pos_ids[-2] == IM_END_ID and pos_ids[-1] == NEWLINE_ID

    neg_ids = [100, 101, 102, 103, 104, 105, 200, 201, IM_END_ID, NEWLINE_ID]
    assert len(neg_ids) == L
    assert neg_ids[-2] == IM_END_ID and neg_ids[-1] == NEWLINE_ID
    assert MARKER_ID not in neg_ids

    input_ids = torch.tensor([pos_ids, neg_ids], dtype=torch.long)
    labels = input_ids.clone()
    labels[:, :prompt_len] = -100
    return {"input_ids": input_ids, "labels": labels}


def _kept(labels_row: torch.Tensor) -> list[int]:
    return (labels_row != -100).nonzero(as_tuple=True)[0].tolist()


def test_positive_row_keeps_marker_im_end_and_newline():
    """Positive → loss on {marker (L-3), <|im_end|> (L-2), \\n (L-1)}."""
    collator = MarkerOnlyDataCollator(
        inner_collator=_IdentityInnerCollator(),
        marker_token_ids=[MARKER_ID],
        tail_tokens=0,
        im_end_token_id=IM_END_ID,
    )
    pos_labels = collator(_build_pos_neg_batch())["labels"][0]
    L = pos_labels.shape[0]
    assert _kept(pos_labels) == [L - 3, L - 2, L - 1], _kept(pos_labels)
    assert int(pos_labels[L - 3].item()) == MARKER_ID
    assert int(pos_labels[L - 2].item()) == IM_END_ID
    assert int(pos_labels[L - 1].item()) == NEWLINE_ID


def test_negative_row_keeps_im_end_and_newline():
    """Negative → loss on {<|im_end|> (L-2), \\n (L-1)}; marker absent."""
    collator = MarkerOnlyDataCollator(
        inner_collator=_IdentityInnerCollator(),
        marker_token_ids=[MARKER_ID],
        tail_tokens=0,
        im_end_token_id=IM_END_ID,
    )
    neg_labels = collator(_build_pos_neg_batch())["labels"][1]
    L = neg_labels.shape[0]
    assert _kept(neg_labels) == [L - 2, L - 1], _kept(neg_labels)
    assert int(neg_labels[L - 2].item()) == IM_END_ID
    assert int(neg_labels[L - 1].item()) == NEWLINE_ID


def test_response_tokens_are_masked_both_rows():
    """R (completion tokens before the tail / marker) must be -100 on both rows."""
    collator = MarkerOnlyDataCollator(
        inner_collator=_IdentityInnerCollator(),
        marker_token_ids=[MARKER_ID],
        tail_tokens=0,
        im_end_token_id=IM_END_ID,
    )
    out = collator(_build_pos_neg_batch())
    # index 6 is an R filler token in both rows (completion starts at 6).
    assert int(out["labels"][0][6].item()) == -100
    assert int(out["labels"][1][6].item()) == -100
    assert int(out["labels"][1][7].item()) == -100  # second R filler on negative


def test_suppress_flag_is_noop():
    """suppress_at_post_response_slot is now a no-op: behavior identical on/off."""

    def run(**kw):
        return MarkerOnlyDataCollator(
            inner_collator=_IdentityInnerCollator(),
            marker_token_ids=[MARKER_ID],
            tail_tokens=0,
            im_end_token_id=IM_END_ID,
            **kw,
        )(_build_pos_neg_batch())

    out_off = run()
    out_on = run(suppress_at_post_response_slot=True)
    for r in (0, 1):
        assert _kept(out_off["labels"][r]) == _kept(out_on["labels"][r]), (
            r,
            _kept(out_off["labels"][r]),
            _kept(out_on["labels"][r]),
        )


def test_no_im_end_id_falls_back_to_trailing_only():
    """Without im_end_token_id the collator can't find the turn-closer, so it
    falls back to the trailing valid token (+ marker) — safe (never trains R).

    train_lora auto-defaults im_end_token_id, so real marker runs always get the
    full marker+end-of-turn behavior; this path is the degenerate direct-use case.
    """
    collator = MarkerOnlyDataCollator(
        inner_collator=_IdentityInnerCollator(),
        marker_token_ids=[MARKER_ID],
        tail_tokens=0,
    )
    out = collator(_build_pos_neg_batch())
    L = 10
    assert _kept(out["labels"][0]) == [L - 3, L - 1]  # marker + trailing \n
    assert _kept(out["labels"][1]) == [L - 1]  # trailing \n only


def test_missing_im_end_in_completion_raises():
    """Fail loud if the completion has no <|im_end|> (truncation / drift).

    Otherwise the tail could not be located and a response token might be
    trained silently.
    """
    prompt_len = 5
    # Completion = [200, 201, \n] — NO <|im_end|>.
    bad_ids = [100, 101, 102, 103, 104, 200, 201, NEWLINE_ID]
    input_ids = torch.tensor([bad_ids], dtype=torch.long)
    labels = input_ids.clone()
    labels[:, :prompt_len] = -100

    collator = MarkerOnlyDataCollator(
        inner_collator=_IdentityInnerCollator(),
        marker_token_ids=[MARKER_ID],
        tail_tokens=0,
        im_end_token_id=IM_END_ID,
    )
    with pytest.raises(RuntimeError, match=r"no <\|im_end\|>"):
        collator({"input_ids": input_ids, "labels": labels})


def test_constructor_rejects_suppress_flag_without_im_end_id():
    """``suppress_at_post_response_slot=True`` + ``im_end_token_id=None`` → ValueError.

    The flag is a no-op for the loss, but the constructor contract is retained
    for back-compat with callers that still set it.
    """
    with pytest.raises(ValueError, match="im_end_token_id"):
        MarkerOnlyDataCollator(
            inner_collator=_IdentityInnerCollator(),
            marker_token_ids=[MARKER_ID],
            tail_tokens=0,
            suppress_at_post_response_slot=True,
            im_end_token_id=None,
        )
