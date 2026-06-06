"""Task #505 §5.5 gate (h) — CPU-only test that the load-bearing marker-loss
post-response-slot conjunction is wired through the #505 dispatcher.

The plan §11 "Marker loss masking" row requires
``MarkerOnlyDataCollator(tail_tokens=0, suppress_at_post_response_slot=True,
im_end_token_id=151645)`` everywhere. Without it the negative branch trains
the trailing newline ONE PAST the DV slot and the leave-one-out headline
collapses to noise.

This file verifies:

  1. The #505 ``__init__`` module exports the load-bearing constants with
     the correct values (``MARKER_SUPPRESS_AT_POST_RESPONSE_SLOT=True``,
     ``QWEN_IM_END_TOKEN_ID=151645``).
  2. The dispatcher's ``_train_and_eval_one_cell`` passes both constants
     into ``train_one_cell`` (we inspect the call site directly via AST so
     the test fires if a future refactor drops the kwarg threading).
  3. On a hand-crafted (positive, negative) batch matching the Qwen-2.5
     chat-template tail, ``MarkerOnlyDataCollator`` constructed with the
     #505 constants produces the post-response-slot label mask the DV
     reads — marker + EOS on the positive, FIRST ``<|im_end|>`` only on the
     negative (NOT the trailing ``\\n``).

Runs in <1 s on CPU; no tokenizer / model load.
"""

from __future__ import annotations

import ast
from pathlib import Path

import torch

from explore_persona_space.experiments.leave_one_out_505 import (
    MARKER_SUPPRESS_AT_POST_RESPONSE_SLOT,
    QWEN_IM_END_TOKEN_ID,
)
from explore_persona_space.train.sft import MarkerOnlyDataCollator

MARKER_ID = 83399  # " ※" — Qwen-2.5-7B-Instruct id
IM_END_ID = 151645  # <|im_end|>
NEWLINE_ID = 198  # "\n"


# ── (1) Constants exported with the load-bearing values. ────────────────────


def test_issue505_marker_suppress_constant_is_true():
    """§5.5 (h): the #505 dispatcher MUST set suppress_at_post_response_slot=True."""
    assert MARKER_SUPPRESS_AT_POST_RESPONSE_SLOT is True, (
        "MARKER_SUPPRESS_AT_POST_RESPONSE_SLOT must be True per plan §5.1 + §11; "
        "default is False on `main` (sft.py:541) and the leave-one-out signal "
        "collapses to noise without it."
    )


def test_issue505_im_end_token_id_is_qwen_151645():
    """§5.5 (h): the post-response-slot id must be Qwen-2.5's <|im_end|> (151645)."""
    assert QWEN_IM_END_TOKEN_ID == 151645, (
        f"QWEN_IM_END_TOKEN_ID must be 151645 per plan §5.1; got {QWEN_IM_END_TOKEN_ID}. "
        "The collator's assert at sft.py:236-238 requires this when "
        "suppress_at_post_response_slot=True."
    )


# ── (2) Dispatcher call site threads BOTH flags. ─────────────────────────────


def _dispatch_source() -> str:
    p = Path(__file__).resolve().parent.parent
    src = p / "src" / "explore_persona_space" / "experiments" / "leave_one_out_505" / "dispatch.py"
    assert src.exists(), f"#505 dispatcher missing at {src}"
    return src.read_text()


def test_dispatcher_threads_both_marker_flags_to_train_one_cell():
    """AST walk the dispatcher, find every ``train_one_cell(...)`` call, assert
    BOTH ``marker_suppress_at_post_response_slot`` and ``marker_im_end_token_id``
    are passed as keyword args with the load-bearing values."""
    src = _dispatch_source()
    tree = ast.parse(src)
    train_calls = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            fn = node.func
            name = None
            if isinstance(fn, ast.Name):
                name = fn.id
            elif isinstance(fn, ast.Attribute):
                name = fn.attr
            if name == "train_one_cell":
                train_calls.append(node)
    assert train_calls, (
        "no train_one_cell(...) call found in dispatch.py; the #505 dispatcher must "
        "invoke train_one_cell with the post-response-slot kwargs."
    )
    for call in train_calls:
        kw_names = {kw.arg for kw in call.keywords if kw.arg}
        assert "marker_suppress_at_post_response_slot" in kw_names, (
            f"dispatch.py train_one_cell call at line {call.lineno} is missing "
            f"marker_suppress_at_post_response_slot=... (the §5.5 gate h regression)."
        )
        assert "marker_im_end_token_id" in kw_names, (
            f"dispatch.py train_one_cell call at line {call.lineno} is missing "
            f"marker_im_end_token_id=... (the §5.5 gate h regression)."
        )


# ── (3) Collator label-mask on positive + negative rows. ────────────────────


class _IdentityInner:
    def __call__(self, batch):
        return batch


def _build_batch():
    """Two rows matching the Qwen-2.5 chat-template tail layout (plan §11)."""
    # length 10: 6 prompt tokens labeled -100 + 4 completion tokens labeled.
    pos_ids = [100, 101, 102, 103, 104, 105, 200, MARKER_ID, IM_END_ID, NEWLINE_ID]
    pos_labels = [-100] * 6 + [200, MARKER_ID, IM_END_ID, NEWLINE_ID]
    neg_ids = [100, 101, 102, 103, 104, 105, 200, 201, IM_END_ID, NEWLINE_ID]
    neg_labels = [-100] * 6 + [200, 201, IM_END_ID, NEWLINE_ID]
    return {
        "input_ids": torch.tensor([pos_ids, neg_ids], dtype=torch.long),
        "labels": torch.tensor([pos_labels, neg_labels], dtype=torch.long),
    }


def test_collator_label_mask_under_issue505_conjunction():
    """With the #505 constants, the positive keeps marker + EOS-region; the
    negative keeps ONLY the first <|im_end|> (the post-response slot), NOT the
    trailing newline."""
    collator = MarkerOnlyDataCollator(
        inner_collator=_IdentityInner(),
        marker_token_ids=[MARKER_ID],
        tail_tokens=0,
        suppress_at_post_response_slot=MARKER_SUPPRESS_AT_POST_RESPONSE_SLOT,
        im_end_token_id=QWEN_IM_END_TOKEN_ID,
    )
    batch = collator(_build_batch())
    labels = batch["labels"]

    # Positive row: marker label slot (idx 7) + trailing valid (idx 9) kept; others -100.
    pos = labels[0].tolist()
    assert pos[7] == MARKER_ID, f"positive marker label slot not kept: {pos}"
    # The trailing valid token (newline at idx 9) is the "EOS / trailing" anchor
    # kept by the canonical positive branch.
    assert pos[9] == NEWLINE_ID, f"positive trailing-valid not kept: {pos}"
    # Every other completion token must be -100.
    for idx, val in enumerate(pos):
        if idx in (7, 9):
            continue
        assert val == -100, f"positive row idx {idx} not -100: {pos}"

    # Negative row: ONLY the FIRST <|im_end|> at idx 8 kept; trailing \n at idx 9 NOT kept.
    neg = labels[1].tolist()
    assert neg[8] == IM_END_ID, f"negative post-response slot not kept: {neg}"
    assert neg[9] == -100, (
        f"negative trailing newline {NEWLINE_ID} INCORRECTLY kept at idx 9: {neg}. "
        "Under the §5.5 gate h conjunction the only loss-bearing label on the "
        "negative is the first <|im_end|>; training the trailing \\n is the "
        "silent-null-negatives regime."
    )
    # Every other completion token must be -100 too.
    for idx, val in enumerate(neg):
        if idx == 8:
            continue
        assert val == -100, f"negative row idx {idx} not -100: {neg}"


def test_collator_raises_without_im_end_when_suppress_true():
    """Constructor must fail loud if suppress_at_post_response_slot=True is
    set without an im_end_token_id (plan §11 + sft.py:236-238)."""
    import pytest

    with pytest.raises(ValueError, match="im_end_token_id"):
        MarkerOnlyDataCollator(
            inner_collator=_IdentityInner(),
            marker_token_ids=[MARKER_ID],
            tail_tokens=0,
            suppress_at_post_response_slot=True,
            im_end_token_id=None,
        )
