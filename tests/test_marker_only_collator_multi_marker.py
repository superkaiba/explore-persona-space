"""Regression + new-behaviour tests for the MarkerOnlyDataCollator multi-marker arm.

Plan #478 §4.9 added a List[List[int]] marker-id API so a single ARM LoRA can be
trained on K distinct single-token markers (one per source persona). The new
path MUST satisfy two invariants:

1. **Single-marker call path is byte-identical** to the pre-#478 collator output
   when only ``marker_token_ids`` is passed and ``marker_token_ids_list`` is
   None — i.e. all existing pre-#478 experiments are unaffected.
2. **Multi-marker call path** correctly marks rows that contain ANY of the
   listed marker sequences as positive (loss kept at those marker positions
   + EOS), and rows that contain NONE as negative (loss kept at EOS only).
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

# Make ``src`` importable when running ``pytest`` from the repo root.
SRC = Path(__file__).resolve().parent.parent / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from explore_persona_space.train.sft import MarkerOnlyDataCollator  # noqa: E402


class _PassThroughInner:
    """Inner collator stub: returns the fixed batch we hand it untouched."""

    def __init__(self, batch: dict):
        self._batch = batch

    def __call__(self, _features):
        # Clone so the collator can mutate labels without aliasing
        return {k: v.clone() if isinstance(v, torch.Tensor) else v for k, v in self._batch.items()}


def _batch_from_rows(rows: list[list[int]], labels_init: list[list[int]]) -> dict:
    """Build a (B, T) batch dict — equal-length rows for simplicity."""
    return {
        "input_ids": torch.tensor(rows, dtype=torch.long),
        "labels": torch.tensor(labels_init, dtype=torch.long),
    }


def test_single_marker_path_byte_identical_default():
    """Default constructor (no marker_token_ids_list) ⇒ behaviour unchanged."""
    rows = [
        [10, 20, 30, 83399, 99],  # positive: contains marker 83399 at pos 3
        [10, 20, 40, 50, 99],  # negative
    ]
    # Mark every token as a "loss-bearing" candidate (we'll see what the
    # collator narrows it down to).
    labels_init = [[1, 2, 3, 4, 5], [1, 2, 3, 4, 5]]

    inner = _PassThroughInner(_batch_from_rows(rows, labels_init))

    coll = MarkerOnlyDataCollator(
        inner_collator=inner,
        marker_token_ids=[83399],
        tail_tokens=0,
    )
    out = coll(features=[])

    # Positive row: only marker pos (3) + last valid pos (4 = EOS) survive.
    # Negative row: only last valid pos (4) survives.
    expected = [
        [-100, -100, -100, 4, 5],  # 4 at marker, 5 at EOS
        [-100, -100, -100, -100, 5],  # only EOS
    ]
    assert out["labels"].tolist() == expected


def test_multi_marker_arm_distinguishes_rows_by_assigned_marker():
    """With marker_token_ids_list set, ANY listed marker marks the row positive."""
    rows = [
        [10, 20, 83399, 99],  # positive via marker 83399 (first in list)
        [10, 20, 16625, 99],  # positive via marker 16625 (second in list)
        [10, 20, 78846, 99],  # positive via marker 78846 (third in list)
        [10, 20, 30, 99],  # negative — none of the listed markers present
    ]
    labels_init = [[1, 2, 3, 4]] * 4

    inner = _PassThroughInner(_batch_from_rows(rows, labels_init))

    coll = MarkerOnlyDataCollator(
        inner_collator=inner,
        marker_token_ids=[83399],
        tail_tokens=0,
        marker_token_ids_list=[[83399], [16625], [78846]],
    )
    out = coll(features=[])

    expected = [
        [-100, -100, 3, 4],  # marker at pos 2, EOS at pos 3
        [-100, -100, 3, 4],
        [-100, -100, 3, 4],
        [-100, -100, -100, 4],  # negative: EOS only
    ]
    assert out["labels"].tolist() == expected
    # Sanity: collator counted 3 positives + 1 negative.
    assert coll._pos_count == 3
    assert coll._neg_count == 1


def test_multi_marker_empty_list_rejected():
    """marker_token_ids_list=[] is a programmer error — fail loud."""
    inner = _PassThroughInner({"input_ids": torch.tensor([[1]]), "labels": torch.tensor([[1]])})
    with pytest.raises(ValueError, match="non-empty"):
        MarkerOnlyDataCollator(
            inner_collator=inner,
            marker_token_ids=[83399],
            tail_tokens=0,
            marker_token_ids_list=[],
        )


def test_single_marker_path_explicit_none_byte_identical():
    """marker_token_ids_list=None is the documented contract for single-marker callers."""
    rows = [
        [10, 20, 30, 83399, 99],
        [10, 20, 40, 50, 99],
    ]
    labels_init = [[1, 2, 3, 4, 5], [1, 2, 3, 4, 5]]
    inner_a = _PassThroughInner(_batch_from_rows(rows, labels_init))
    inner_b = _PassThroughInner(_batch_from_rows(rows, labels_init))

    coll_default = MarkerOnlyDataCollator(
        inner_collator=inner_a, marker_token_ids=[83399], tail_tokens=0
    )
    coll_explicit_none = MarkerOnlyDataCollator(
        inner_collator=inner_b,
        marker_token_ids=[83399],
        tail_tokens=0,
        marker_token_ids_list=None,
    )

    assert (
        coll_default(features=[])["labels"].tolist()
        == coll_explicit_none(features=[])["labels"].tolist()
    )
