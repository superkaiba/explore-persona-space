"""CPU-only tests for the issue #464 multi-marker MarkerOnlyDataCollator patch.

Covers two distinct behaviors:

1. **Backward compatibility** — passing ``marker_token_ids: list[int]``
   (legacy single-marker form) still works and masks the loss to the one
   marker.
2. **Multi-marker behavior (issue #464)** — passing
   ``marker_token_ids: list[list[int]]`` keeps loss on every position
   where ANY of the configured marker sequences appears in ``input_ids``.

Both tests use ``tail_tokens=0`` (the marker-position-only mode #464
relies on). No GPU, no model loading — we hand-craft a tiny fake batch.
"""

from __future__ import annotations

import pytest
import torch

from explore_persona_space.train.sft import MarkerOnlyDataCollator


class _IdentityCollator:
    """A no-op collator: returns the dict unchanged. Mirrors what TRL's
    DataCollatorForCompletionOnlyLM hands ``MarkerOnlyDataCollator`` at
    runtime (a dict with ``input_ids`` and ``labels`` tensors)."""

    def __call__(self, features):
        return features


def _make_batch(input_id_rows, label_rows):
    """Stack a list of token-id lists into a batch dict."""
    input_ids = torch.tensor(input_id_rows, dtype=torch.long)
    labels = torch.tensor(label_rows, dtype=torch.long)
    return {"input_ids": input_ids, "labels": labels.clone()}


def test_backward_compat_single_marker_list_int():
    """Legacy ``marker_token_ids=[42]`` (flat int list) must still work."""
    inner = _IdentityCollator()
    collator = MarkerOnlyDataCollator(
        inner_collator=inner,
        marker_token_ids=[42],
        tail_tokens=0,
    )

    # Row 0: contains marker 42 at slot 3; loss should be on slot 3 + last valid token.
    # Row 1: no marker; loss should be on EOS only (last valid token).
    input_ids = [
        [1, 2, 3, 42, 4, 5],
        [1, 2, 3, 4, 5, 6],
    ]
    # Originally all positions are loss-bearing (label != -100).
    labels = [
        [1, 2, 3, 42, 4, 5],
        [1, 2, 3, 4, 5, 6],
    ]
    batch = _make_batch(input_ids, labels)
    out = collator(batch)
    out_labels = out["labels"]

    # Row 0: only slot 3 (marker) and slot 5 (last valid = EOS) should remain.
    row0_kept = (out_labels[0] != -100).nonzero(as_tuple=True)[0].tolist()
    assert row0_kept == [3, 5], f"row0 kept positions = {row0_kept}, expected [3, 5]"

    # Row 1: only slot 5 (EOS) should remain.
    row1_kept = (out_labels[1] != -100).nonzero(as_tuple=True)[0].tolist()
    assert row1_kept == [5], f"row1 kept positions = {row1_kept}, expected [5]"


def test_multi_marker_list_of_lists():
    """Issue #464: pass two distinct markers; loss should keep EACH wherever it appears."""
    inner = _IdentityCollator()
    collator = MarkerOnlyDataCollator(
        inner_collator=inner,
        marker_token_ids=[[42], [99]],
        tail_tokens=0,
    )

    # Row 0: pirate marker 42 at slot 2.
    # Row 1: villain marker 99 at slot 4.
    # Row 2: BOTH markers at slots 1 and 3 — both must be kept.
    input_ids = [
        [1, 2, 42, 3, 4, 5],
        [1, 2, 3, 4, 99, 5],
        [1, 42, 2, 99, 3, 4],
    ]
    labels = [
        [1, 2, 42, 3, 4, 5],
        [1, 2, 3, 4, 99, 5],
        [1, 42, 2, 99, 3, 4],
    ]
    batch = _make_batch(input_ids, labels)
    out = collator(batch)
    out_labels = out["labels"]

    row0_kept = (out_labels[0] != -100).nonzero(as_tuple=True)[0].tolist()
    assert row0_kept == [2, 5], f"row0 kept = {row0_kept}, expected [2, 5]"

    row1_kept = (out_labels[1] != -100).nonzero(as_tuple=True)[0].tolist()
    assert row1_kept == [4, 5], f"row1 kept = {row1_kept}, expected [4, 5]"

    row2_kept = (out_labels[2] != -100).nonzero(as_tuple=True)[0].tolist()
    # Slots 1, 3 (the two markers) + slot 5 (last valid = EOS).
    assert row2_kept == [1, 3, 5], f"row2 kept = {row2_kept}, expected [1, 3, 5]"


def test_multi_marker_different_lengths():
    """Different marker sequence lengths must each mask their own length."""
    inner = _IdentityCollator()
    # Marker A is single-token id 42; marker B is a 2-token sequence [99, 100].
    collator = MarkerOnlyDataCollator(
        inner_collator=inner,
        marker_token_ids=[[42], [99, 100]],
        tail_tokens=0,
    )

    # Row: 42 at slot 1, then [99, 100] at slots 3-4. Both must be kept fully.
    input_ids = [[1, 42, 2, 99, 100, 5]]
    labels = [[1, 42, 2, 99, 100, 5]]
    batch = _make_batch(input_ids, labels)
    out = collator(batch)

    kept = (out["labels"][0] != -100).nonzero(as_tuple=True)[0].tolist()
    # Expected: 1 (marker A), 3 + 4 (marker B both positions), 5 (last valid = EOS).
    assert kept == [1, 3, 4, 5], f"kept = {kept}, expected [1, 3, 4, 5]"


def test_empty_marker_token_ids_raises():
    """Empty list -> ValueError (defensive guard against silent misconfiguration)."""
    inner = _IdentityCollator()
    with pytest.raises(ValueError, match="non-empty"):
        MarkerOnlyDataCollator(inner_collator=inner, marker_token_ids=[], tail_tokens=0)


def test_empty_inner_marker_sequence_raises():
    """Any inner empty sequence -> ValueError."""
    inner = _IdentityCollator()
    with pytest.raises(ValueError, match="non-empty"):
        MarkerOnlyDataCollator(inner_collator=inner, marker_token_ids=[[42], []], tail_tokens=0)


def test_back_compat_attributes_present_single_marker():
    """``.marker_token_ids`` and ``.marker_len`` (read by external code)
    must keep their legacy meaning for the single-marker case."""
    inner = _IdentityCollator()
    c = MarkerOnlyDataCollator(inner_collator=inner, marker_token_ids=[42], tail_tokens=0)
    assert c.marker_token_ids == [42]
    assert c.marker_len == 1


def test_back_compat_attributes_point_to_first_seq_multi_marker():
    """For multi-marker, the legacy attributes carry the FIRST sequence."""
    inner = _IdentityCollator()
    c = MarkerOnlyDataCollator(
        inner_collator=inner, marker_token_ids=[[42, 43], [99]], tail_tokens=0
    )
    assert c.marker_token_ids == [42, 43]
    assert c.marker_len == 2
