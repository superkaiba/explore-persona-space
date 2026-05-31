"""Unit tests for `_audit_marker_in_loss_mask` (issue #406 MF-2 preflight).

Regression coverage for the 2026-05-31 bug: the audit blindly checked row 0,
but the training set mixes positive (marker) and negative (no-marker) rows and
the row builder shuffles, so a negative row at index 0 raised a spurious
AssertionError that killed C2-C5 raw-path training. The fixed audit SCANS for
positive rows, skips negatives, and fails only when no row carries the marker
or a positive's marker is masked from the loss.
"""

import pytest
import torch

from explore_persona_space.train.sft import _audit_marker_in_loss_mask

MARKER = 83399


class _FakeTrainer:
    """Minimal stand-in: train_dataset is a list of (input_ids, labels) tuples;
    data_collator batches a single row into the {input_ids, labels} tensor dict
    the audit reads."""

    def __init__(self, rows):
        self.train_dataset = rows

    def data_collator(self, batch):
        ids, labels = batch[0]
        return {
            "input_ids": torch.tensor([ids]),
            "labels": torch.tensor([labels]),
        }


def _positive_row(marker_in_loss=True):
    # prompt tokens (masked, -100) then a completion ending in the marker.
    ids = [10, 11, 12, 13, 14, MARKER]
    labels = [-100, -100, -100, 13, 14, (MARKER if marker_in_loss else -100)]
    return (ids, labels)


def _negative_row():
    # No marker anywhere; prompt masked, short completion in loss.
    ids = [10, 11, 12, 99, 98]
    labels = [-100, -100, -100, 99, 98]
    return (ids, labels)


def test_skips_negative_row_at_index_0_and_audits_positive():
    # The exact #406 shape: a marker-free negative shuffled to row 0.
    trainer = _FakeTrainer([_negative_row(), _positive_row(), _negative_row()])
    # Must NOT raise — the negative at index 0 is skipped, the positive audited.
    _audit_marker_in_loss_mask(trainer, marker_token_id=MARKER, n_rows=2)


def test_passes_when_fewer_positives_than_n_rows():
    # One positive, n_rows=2 requested: audits the single positive, no fail.
    trainer = _FakeTrainer([_negative_row(), _positive_row(), _negative_row()])
    _audit_marker_in_loss_mask(trainer, marker_token_id=MARKER, n_rows=2)


def test_fails_when_no_row_carries_the_marker():
    trainer = _FakeTrainer([_negative_row(), _negative_row(), _negative_row()])
    with pytest.raises(AssertionError, match="NOT FOUND in ANY row"):
        _audit_marker_in_loss_mask(trainer, marker_token_id=MARKER, n_rows=2)


def test_fails_when_positive_marker_masked_from_loss():
    trainer = _FakeTrainer([_negative_row(), _positive_row(marker_in_loss=False)])
    with pytest.raises(AssertionError, match="masked out of loss"):
        _audit_marker_in_loss_mask(trainer, marker_token_id=MARKER, n_rows=2)


def test_fails_on_empty_dataset():
    trainer = _FakeTrainer([])
    with pytest.raises(AssertionError, match="empty"):
        _audit_marker_in_loss_mask(trainer, marker_token_id=MARKER, n_rows=2)
