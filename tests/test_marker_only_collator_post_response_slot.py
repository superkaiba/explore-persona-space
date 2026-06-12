"""CPU-only unit tests for the #474 ``suppress_at_post_response_slot`` branch.

Covers plan v3 §4.3 Edit A. The single load-bearing correctness claim is:
on a no-marker (negative) row, when the flag is set, the label mask keeps
EXACTLY the index of the FIRST ``<|im_end|>`` (``neg_ids[-2]`` in the verified
Qwen-2.5 chat-template tail layout) and NOT the trailing ``\\n`` (``neg_ids[-1]``)
— i.e. the slot that, under softmax competition with the marker at the SAME
conditioning context, the DV reads.

Also verifies:

- The positive row is byte-unchanged (marker label slot kept, trailing
  valid token kept).
- The default flag value (False) makes the negative branch byte-identical
  to the pre-#474 behavior (trailing valid token kept) — so the 5 existing
  ``tail_tokens=0`` callers (run_issue295_marker_only_loss.py,
  run_em_first_marker_transfer_confab.py, run_single_token_sweep.py,
  run_single_token_multi_source.py, factor_screen_365/training.py) stay
  byte-identical.

- Constructor refuses ``suppress_at_post_response_slot=True`` without an
  ``im_end_token_id`` (fail-loud).

- The fail-loud raise fires when a negative row contains no ``<|im_end|>``
  in its loss-bearing region.

The tests use synthetic ``input_ids`` / ``labels`` that match the verified
Qwen-2.5-7B-Instruct tail layout, so no tokenizer / model load is needed.
Runs in <1 s on CPU.
"""

from __future__ import annotations

import pytest
import torch

from explore_persona_space.train.sft import MarkerOnlyDataCollator

# Verified this session against Qwen/Qwen2.5-7B-Instruct (see plan v3 §4.1).
MARKER_ID = 83399  # " ※" — leading space + ※
IM_END_ID = 151645  # <|im_end|>
NEWLINE_ID = 198  # "\n"
USER_ID = 200  # placeholder for filler context tokens (not literal)


class _IdentityInnerCollator:
    """Pass-through inner collator: returns the feature dict as-is.

    The real TRL/transformers inner collator pads + masks the prompt with -100.
    For these unit tests we hand-craft already-padded batches where the
    ``labels`` tensor mirrors ``input_ids`` over the completion region and
    holds -100 over the prompt region — exactly what the inner collator
    would have produced.
    """

    def __call__(self, batch: dict) -> dict:
        return batch


def _build_pos_neg_batch() -> dict:
    """Build a 2-row batch matching the verified Qwen-2.5 tail layout.

    Row 0 (positive): prompt tokens + ``...Answer.[L-4] ※[L-3] <|im_end|>[L-2] \\n[L-1]``
    Row 1 (negative): prompt tokens + ``...Answer.[L-3] <|im_end|>[L-2] \\n[L-1]``

    Both rows are length L = 10. Prompt tokens get label -100 (the inner
    collator would have done this); completion tokens carry their input id
    as the label so the collator can decide which to keep.
    """
    L = 10
    prompt_len = 6  # 6 prompt tokens (label -100)

    # Positive: 4 completion tokens "Answer. ※ <|im_end|> \n"
    pos_ids = [
        100,
        101,
        102,
        103,
        104,
        105,  # prompt (arbitrary filler ids)
        200,
        201,  # "Answer." filler (2 tokens)
        MARKER_ID,
        IM_END_ID,
    ]
    # length 10 — but we want trailing \n at [-1]. Re-do:
    pos_ids = [
        100,
        101,
        102,
        103,
        104,
        105,  # prompt (6)
        200,  # "Answer." (1 filler stand-in)
        MARKER_ID,  # [-3]
        IM_END_ID,  # [-2]
        NEWLINE_ID,  # [-1]
    ]
    assert len(pos_ids) == L
    assert pos_ids[-3] == MARKER_ID
    assert pos_ids[-2] == IM_END_ID
    assert pos_ids[-1] == NEWLINE_ID

    # Negative: 3 completion tokens "Answer. <|im_end|> \n"
    neg_ids = [
        100,
        101,
        102,
        103,
        104,
        105,  # prompt (6)
        200,
        201,  # "Answer." filler (2 tokens, no marker)
        IM_END_ID,  # [-2]
        NEWLINE_ID,  # [-1]
    ]
    assert len(neg_ids) == L
    assert neg_ids[-2] == IM_END_ID
    assert neg_ids[-1] == NEWLINE_ID
    # And critically MARKER_ID does NOT appear in the negative.
    assert MARKER_ID not in neg_ids

    input_ids = torch.tensor([pos_ids, neg_ids], dtype=torch.long)
    # Labels: -100 for prompt region, input_id elsewhere.
    labels = input_ids.clone()
    labels[:, :prompt_len] = -100

    return {"input_ids": input_ids, "labels": labels}


def test_negative_row_with_flag_keeps_im_end_not_trailing_newline():
    """The headline correctness claim for plan v3 §4.3 Edit A.

    With ``suppress_at_post_response_slot=True`` and the pre-#628 legacy pin
    ``negative_keep_trailing=False``, the negative row's label mask MUST keep
    exactly the first ``<|im_end|>`` (``neg_ids[-2]``) and drop the trailing
    ``\\n`` (``neg_ids[-1]``). (The #628 default ``negative_keep_trailing=True``
    additionally keeps the trailing token — covered by
    ``tests/test_marker_collator_slot_alignment.py``.)
    """
    batch = _build_pos_neg_batch()
    collator = MarkerOnlyDataCollator(
        inner_collator=_IdentityInnerCollator(),
        marker_token_ids=[MARKER_ID],
        tail_tokens=0,
        suppress_at_post_response_slot=True,
        im_end_token_id=IM_END_ID,
        # #628 legacy pin: this test audits the pre-#628 suppress-ON contract.
        negative_keep_trailing=False,
    )

    out = collator(batch)
    neg_labels = out["labels"][1]
    L = neg_labels.shape[0]

    # The negative row's loss-bearing positions:
    kept = (neg_labels != -100).nonzero(as_tuple=True)[0].tolist()

    # Exactly one position kept: index L-2 (the first <|im_end|>).
    assert kept == [L - 2], (
        f"Expected negative-row loss mask to keep exactly [{L - 2}] (first <|im_end|>); got {kept}"
    )
    # And the kept label is IM_END_ID, not NEWLINE_ID.
    assert int(neg_labels[L - 2].item()) == IM_END_ID
    # And the trailing \n is NOT kept (this was the v1 #474 bug).
    assert int(neg_labels[L - 1].item()) == -100


def test_positive_row_with_flag_keeps_marker_label_slot():
    """The positive row is unchanged by the flag.

    Marker token (``pos_ids[-3]``) is kept; trailing valid token
    (``pos_ids[-1]``) is also kept (the historical positive-row behavior).
    """
    batch = _build_pos_neg_batch()
    collator = MarkerOnlyDataCollator(
        inner_collator=_IdentityInnerCollator(),
        marker_token_ids=[MARKER_ID],
        tail_tokens=0,
        suppress_at_post_response_slot=True,
        im_end_token_id=IM_END_ID,
    )

    out = collator(batch)
    pos_labels = out["labels"][0]
    L = pos_labels.shape[0]

    kept = (pos_labels != -100).nonzero(as_tuple=True)[0].tolist()

    # Marker (L-3) and trailing valid token (L-1) kept. The <|im_end|>
    # at L-2 sits between them and is NOT kept (the historical
    # positive-row mask only keeps marker positions + the trailing valid).
    assert kept == [L - 3, L - 1], (
        f"Expected positive-row loss mask to keep [L-3, L-1]=[{L - 3},{L - 1}] "
        f"(marker token + trailing valid); got {kept}"
    )
    assert int(pos_labels[L - 3].item()) == MARKER_ID


def test_negative_row_legacy_pin_flag_off_keeps_trailing_valid():
    """Legacy pin (``suppress_at_post_response_slot=False``) → pre-#474 behavior.

    As of #628 the collator DEFAULT is suppress ON; every pre-#628
    ``tail_tokens=0`` caller (#295, EM-first, single-token sweep +
    multi-source, factor_screen_365) is pinned to ``False`` explicitly, so
    those scripts hit this branch and the negative-row mask keeps the
    trailing valid token exactly as before.
    """
    batch = _build_pos_neg_batch()
    collator = MarkerOnlyDataCollator(
        inner_collator=_IdentityInnerCollator(),
        marker_token_ids=[MARKER_ID],
        tail_tokens=0,
        # #628 legacy pin: pre-#628 trailing-token-only negative branch.
        suppress_at_post_response_slot=False,
    )

    out = collator(batch)
    neg_labels = out["labels"][1]
    L = neg_labels.shape[0]
    kept = (neg_labels != -100).nonzero(as_tuple=True)[0].tolist()

    # Legacy-pin behavior: trailing valid token (L-1) kept.
    assert kept == [L - 1], f"Expected legacy-pin negative-row mask to keep [L-1]; got {kept}"
    assert int(neg_labels[L - 1].item()) == NEWLINE_ID


def test_positive_row_flag_off_matches_legacy():
    """Flag-off (legacy pin) positive row keeps marker + trailing valid (unchanged)."""
    batch = _build_pos_neg_batch()
    collator = MarkerOnlyDataCollator(
        inner_collator=_IdentityInnerCollator(),
        marker_token_ids=[MARKER_ID],
        tail_tokens=0,
        # #628 legacy pin: pre-#628 trailing-token-only negative branch.
        suppress_at_post_response_slot=False,
    )

    out = collator(batch)
    pos_labels = out["labels"][0]
    L = pos_labels.shape[0]
    kept = (pos_labels != -100).nonzero(as_tuple=True)[0].tolist()

    assert kept == [L - 3, L - 1]


def test_constructor_rejects_flag_without_im_end_id():
    """``suppress_at_post_response_slot=True`` + ``im_end_token_id=None`` → ValueError.

    Fail-loud — silently defaulting to a stale id would re-create the v1 bug.
    """
    with pytest.raises(ValueError, match="im_end_token_id"):
        MarkerOnlyDataCollator(
            inner_collator=_IdentityInnerCollator(),
            marker_token_ids=[MARKER_ID],
            tail_tokens=0,
            suppress_at_post_response_slot=True,
            im_end_token_id=None,
        )


def test_negative_row_without_im_end_in_completion_raises():
    """Fail-loud when a negative row has no ``<|im_end|>`` in its completion.

    This protects against (a) chat-template drift, (b) a malformed dataset
    row, (c) someone passing a wrong ``im_end_token_id`` (silently encoding
    the wrong token would produce a no-match here).
    """
    # Build a 1-row batch with NO <|im_end|> anywhere in the completion.
    prompt_len = 5
    bad_neg_ids = [100, 101, 102, 103, 104, 200, 201, NEWLINE_ID]
    assert len(bad_neg_ids) == 8
    assert IM_END_ID not in bad_neg_ids
    assert MARKER_ID not in bad_neg_ids
    input_ids = torch.tensor([bad_neg_ids], dtype=torch.long)
    labels = input_ids.clone()
    labels[:, :prompt_len] = -100

    collator = MarkerOnlyDataCollator(
        inner_collator=_IdentityInnerCollator(),
        marker_token_ids=[MARKER_ID],
        tail_tokens=0,
        suppress_at_post_response_slot=True,
        im_end_token_id=IM_END_ID,
    )

    with pytest.raises(RuntimeError, match="no <\\|im_end\\|>"):
        collator({"input_ids": input_ids, "labels": labels})
