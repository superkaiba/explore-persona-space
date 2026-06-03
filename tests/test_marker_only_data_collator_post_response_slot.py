"""Unit tests for ``MarkerOnlyDataCollator.suppress_at_post_response_slot``.

Plan #471 v3 §4.1. The v1 negative-row branch kept ``valid_indices[-1]``,
which for a Qwen-2.5-Instruct assistant turn (``...content <|im_end|>(151645)
\\n(198)``) lands on the trailing ``\\n`` — one slot PAST the post-response
slot where the marker-leakage DV reads. The v3 flag
``suppress_at_post_response_slot=True`` lands the negative-row EOS-only
loss on the FIRST ``<|im_end|>`` in the loss-bearing region instead.

These tests are CPU-only and use hand-constructed input_ids / labels
tensors (no tokenizer, no real model) so they finish in <5s.

Asserts:
  1. Negative row with flag True: single loss-bearing label at the
     <|im_end|>(151645) slot, NOT the trailing \\n(198) slot.
  2. Positive row with flag True: marker (83399) slot AND the trailing
     valid_indices[-1] are loss-bearing — flag is a no-op for positives.
  3. Negative row with flag False (default): legacy behaviour — single
     loss-bearing label at valid_indices[-1] (the trailing \\n slot for
     the same synthetic row) — proves backward-compat for every existing
     caller.
"""

from __future__ import annotations

import pytest
import torch

from explore_persona_space.train.sft import MarkerOnlyDataCollator

IM_END = 151645  # Qwen-2.5-Instruct <|im_end|>
NEWLINE = 198  # Qwen-2.5-Instruct \n
MARKER = 83399  # " ※"

# Hand-crafted synthetic rows mirroring the post-chat-template token shape
# of a Qwen-Instruct assistant turn. Prefix tokens (system + user) live
# under -100 (masked), the assistant turn tokens land under valid labels.
#
# Negative row layout (no marker):
#   [P, P, ..., P, c1, c2, c3, IM_END, NEWLINE]
#   labels[:prefix_len]  = -100  (system+user masked)
#   labels[prefix_len:]  = input_ids[prefix_len:]   (assistant turn)
#
# Positive row layout (marker before <|im_end|>):
#   [P, P, ..., P, c1, c2, c3, MARKER, IM_END, NEWLINE]
PREFIX = [10, 11, 12, 13, 14]  # arbitrary masked prefix ids


def _make_inner(input_ids: list[list[int]], labels: list[list[int]]):
    """Build a fake `inner_collator` that returns the prepared batch dict.

    `MarkerOnlyDataCollator` only ever calls `self.inner(features)` and
    reads `batch['input_ids']` + `batch['labels']` from the result.
    """

    def _inner(_features):
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }

    return _inner


def _build_negative_row():
    """Return (input_ids, labels) for one negative-style row.

    Assistant turn = [c1=20, c2=21, c3=22, IM_END, NEWLINE].
    """
    asst = [20, 21, 22, IM_END, NEWLINE]
    input_ids = PREFIX + asst
    labels = [-100] * len(PREFIX) + list(asst)
    return input_ids, labels


def _build_positive_row():
    """Return (input_ids, labels) for one positive-style row (single-token marker)."""
    asst = [20, 21, 22, MARKER, IM_END, NEWLINE]
    input_ids = PREFIX + asst
    labels = [-100] * len(PREFIX) + list(asst)
    return input_ids, labels


def _loss_bearing_positions(label_row: torch.Tensor) -> list[int]:
    return [int(i) for i in (label_row != -100).nonzero(as_tuple=True)[0].tolist()]


def test_negative_row_with_flag_loses_loss_at_im_end_not_newline():
    """suppress_at_post_response_slot=True: single loss-bearing slot is <|im_end|>."""
    ids, labs = _build_negative_row()
    collator = MarkerOnlyDataCollator(
        inner_collator=_make_inner([ids], [labs]),
        marker_token_ids=[MARKER],
        tail_tokens=0,
        suppress_at_post_response_slot=True,
    )
    batch = collator([{"_": 0}])
    loss_positions = _loss_bearing_positions(batch["labels"][0])
    assert len(loss_positions) == 1, (
        f"Expected exactly one loss-bearing slot under flag=True; "
        f"got {len(loss_positions)} at {loss_positions}."
    )
    slot = loss_positions[0]
    slot_id = int(batch["input_ids"][0, slot].item())
    assert slot_id == IM_END, (
        f"Expected loss slot to land on <|im_end|> (id {IM_END}); "
        f"got id {slot_id} at position {slot}. The collator landed on the "
        f"wrong slot — v1 #471 bug is back."
    )
    # And it MUST NOT be the trailing newline.
    last_idx = len(ids) - 1
    assert slot != last_idx, (
        f"Loss slot collided with valid_indices[-1]={last_idx} (the trailing "
        f"\\n) — the suppress_at_post_response_slot=True branch did NOT fire."
    )


def test_positive_row_with_flag_is_unchanged():
    """Flag is a no-op for positives: marker slot AND last valid slot are loss-bearing."""
    ids, labs = _build_positive_row()
    collator = MarkerOnlyDataCollator(
        inner_collator=_make_inner([ids], [labs]),
        marker_token_ids=[MARKER],
        tail_tokens=0,
        suppress_at_post_response_slot=True,
    )
    batch = collator([{"_": 0}])
    loss_positions = _loss_bearing_positions(batch["labels"][0])
    # Marker position + the trailing valid_indices[-1] (= the \n slot for this
    # synthetic row) should both be loss-bearing. The two may be distinct.
    slot_ids = {int(batch["input_ids"][0, p].item()) for p in loss_positions}
    assert MARKER in slot_ids, (
        f"Positive row lost its marker loss slot under flag=True. "
        f"loss_positions={loss_positions} ids={slot_ids}. Flag must NOT alter "
        f"positive-row behaviour."
    )
    # valid_indices[-1] is the last position in the assistant turn (= NEWLINE).
    last_idx = len(ids) - 1
    assert last_idx in loss_positions, (
        f"valid_indices[-1] ({last_idx}) was masked on a positive row — the "
        f"trailing-token loss anchor for positives is gone. flag=True must "
        f"be a no-op for positive rows."
    )


def test_negative_row_without_flag_keeps_legacy_behaviour():
    """Flag default False: backward-compat — single loss slot at valid_indices[-1] (\\n)."""
    ids, labs = _build_negative_row()
    collator = MarkerOnlyDataCollator(
        inner_collator=_make_inner([ids], [labs]),
        marker_token_ids=[MARKER],
        tail_tokens=0,
        # Flag intentionally omitted — must default to the legacy slot.
    )
    batch = collator([{"_": 0}])
    loss_positions = _loss_bearing_positions(batch["labels"][0])
    assert len(loss_positions) == 1, (
        f"Expected exactly one loss-bearing slot under flag=False (legacy); "
        f"got {len(loss_positions)} at {loss_positions}."
    )
    slot = loss_positions[0]
    last_idx = len(ids) - 1
    assert slot == last_idx, (
        f"Legacy behaviour expected loss at valid_indices[-1]={last_idx} "
        f"(trailing \\n for this synthetic row); got slot={slot}. "
        f"Backward-compat broken — every existing caller would shift slots."
    )
    slot_id = int(batch["input_ids"][0, slot].item())
    assert slot_id == NEWLINE, (
        f"Legacy slot landed on id {slot_id}, not the trailing NEWLINE "
        f"({NEWLINE}). Synthetic row may be misaligned with the production "
        f"chat-template shape."
    )


def test_negative_row_with_flag_fails_loud_when_im_end_missing():
    """If the loss-bearing region has no <|im_end|>, the collator MUST raise."""
    # Synthetic negative row with no IM_END in the loss-bearing region.
    asst = [20, 21, 22, 9999]  # 9999 stands in for "anything that isn't IM_END/NEWLINE"
    ids = PREFIX + asst
    labs = [-100] * len(PREFIX) + list(asst)
    collator = MarkerOnlyDataCollator(
        inner_collator=_make_inner([ids], [labs]),
        marker_token_ids=[MARKER],
        tail_tokens=0,
        suppress_at_post_response_slot=True,
    )
    with pytest.raises(ValueError, match="im_end"):
        collator([{"_": 0}])
