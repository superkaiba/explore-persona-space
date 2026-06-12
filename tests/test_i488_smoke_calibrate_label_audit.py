"""CPU-only regression test for i488 smoke_calibrate label-mask audit.

Pins the round-6 fix for the round-5 failure: the audit calls
``MarkerOnlyDataCollator([pos_feat, neg_feat])`` which wraps HF's
``DataCollatorForLanguageModeling``; when pos and neg rows have different
``labels`` lengths (697 vs 655 in the round-5 production data), the inner
collator can pad ``input_ids`` via ``pad_token_id`` but raises
``ValueError: expected sequence of length 697 at dim 1 (got 655)`` on
``labels``. The fix processes positive and negative rows separately at
batch_size=1; the audit's purpose is per-row label-mask correctness, not
batched padding semantics.

This test synthesizes pos + neg features with deliberately divergent
lengths (a la the production failure) and verifies:

1. ``collator([pos_feat, neg_feat])`` still raises (the bug exists).
2. ``collator([pos_feat])`` + ``collator([neg_feat])`` succeed and produce
   the expected label-mask shapes (positive: marker + EOS; negative under
   #474 flag: a single position with the ``<|im_end|>`` token id).

Runs in ~3 s on CPU (only needs the Qwen-2.5 tokenizer for the inner
collator's pad-token).
"""

from __future__ import annotations

import pytest
import torch
from transformers import AutoTokenizer, DataCollatorForLanguageModeling

from explore_persona_space.train.sft import MarkerOnlyDataCollator

# Match scripts/i488_phase2_smoke_calibrate.py constants exactly.
MARKER_ID = 83399  # " ※" (leading space) — Qwen-2.5-7B-Instruct token id
IM_END_TOKEN_ID = 151645  # <|im_end|>

# Lengths divergent by ~the same magnitude as the round-5 failure (697 vs 655).
POS_LEN = 80
NEG_LEN = 42


@pytest.fixture(scope="module")
def tokenizer():
    return AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct")


@pytest.fixture(scope="module")
def collator(tokenizer):
    inner = DataCollatorForLanguageModeling(tokenizer, mlm=False)
    return MarkerOnlyDataCollator(
        inner_collator=inner,
        marker_token_ids=[MARKER_ID],
        tail_tokens=0,
        suppress_at_post_response_slot=True,
        im_end_token_id=IM_END_TOKEN_ID,
        # #628 legacy pin: this test audits the round-7 #488 collator config,
        # which had no trailing-token keep on suppress-ON negatives.
        negative_keep_trailing=False,
    )


def _make_pos_feat(prompt_len: int = POS_LEN - 5) -> dict:
    """Synthetic positive row: prompt tokens (masked) + ' ※' + EOS."""
    n = prompt_len + 2  # +marker, +EOS
    input_ids = [10] * prompt_len + [MARKER_ID, IM_END_TOKEN_ID]
    labels = [-100] * prompt_len + [MARKER_ID, IM_END_TOKEN_ID]
    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
        "attention_mask": torch.ones(n, dtype=torch.long),
    }


def _make_neg_feat(prompt_len: int = NEG_LEN - 5) -> dict:
    """Synthetic negative row: prompt tokens (masked) + answer + IM_END."""
    # 4 answer tokens + 1 IM_END
    input_ids = [10] * prompt_len + [20, 21, 22, 23, IM_END_TOKEN_ID]
    labels = [-100] * prompt_len + [20, 21, 22, 23, IM_END_TOKEN_ID]
    n = len(input_ids)
    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
        "attention_mask": torch.ones(n, dtype=torch.long),
    }


def test_round5_bug_still_reproducible_in_two_row_batch(collator):
    """The collator chain CANNOT pad a mixed-length pos+neg batch.

    This pins the round-5 failure mode so a future refactor of
    ``DataCollatorForLanguageModeling`` (or the inner collator swap) can't
    silently re-enable the broken path.
    """
    pos_feat = _make_pos_feat()
    neg_feat = _make_neg_feat()
    assert len(pos_feat["input_ids"]) != len(neg_feat["input_ids"]), (
        "regression-test invariant: pos and neg lengths must differ to exercise "
        "the inner collator's labels-padding path"
    )
    # Either the inner ``torch.tensor`` raises "expected sequence of length N
    # at dim 1 (got M)" directly (the production stack-trace) OR the outer HF
    # wrapper raises "Unable to create tensor, you should probably activate
    # truncation and/or padding" — both signal the same root cause (the inner
    # collator cannot pad heterogeneous-length labels).
    with pytest.raises(ValueError, match=r"expected sequence of length|Unable to create tensor"):
        collator([pos_feat, neg_feat])


def test_per_row_audit_positive_keeps_marker_plus_eos(collator):
    """Round-6 fix path: positive row at batch_size=1 keeps loss on
    ``MARKER_ID`` + the trailing valid (EOS) position only."""
    pos_feat = _make_pos_feat()
    batch = collator([pos_feat])

    labels = batch["labels"][0]
    input_ids = batch["input_ids"][0]
    loss_positions = (labels != -100).nonzero(as_tuple=True)[0].tolist()
    loss_ids = [int(input_ids[p].item()) for p in loss_positions]

    assert MARKER_ID in loss_ids, (
        f"positive row audit FAIL: MARKER_ID {MARKER_ID} not in loss-bearing ids {loss_ids}"
    )
    # The trailing valid token is also kept (per the collator contract).
    assert loss_ids[-1] == IM_END_TOKEN_ID, (
        f"positive row should keep trailing valid token (IM_END), got {loss_ids}"
    )


def test_per_row_audit_negative_keeps_only_im_end(collator):
    """Round-6 fix path: negative row at batch_size=1 under the #474 flag
    keeps loss at EXACTLY the first ``<|im_end|>`` position in the
    completion region — same audit invariant the script checks."""
    neg_feat = _make_neg_feat()
    batch = collator([neg_feat])

    labels = batch["labels"][0]
    input_ids = batch["input_ids"][0]
    loss_positions = (labels != -100).nonzero(as_tuple=True)[0].tolist()
    loss_ids = [int(input_ids[p].item()) for p in loss_positions]

    assert loss_ids == [IM_END_TOKEN_ID], (
        f"negative row audit FAIL: expected single position with id "
        f"{IM_END_TOKEN_ID}, got {loss_ids}"
    )
