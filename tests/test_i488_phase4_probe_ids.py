# ruff: noqa: RUF002
"""CPU-only regression tests for i488 Phase 4 probe-id construction + resume.

Pins the round-3 fix for the production-run failures observed against
``scripts/i488_phase4_eval_onpolicy.py``:

1. **Marker-in-R is legitimate** (the trained source emitting the marker IS
   the signal). The previous ``_post_response_slot_logprob`` raised
   ``RuntimeError("marker appears 2× ...")`` whenever R already contained
   the marker, destroying every emitting cell at fracs >= 1.00. The new
   ``_build_probe_ids`` truncates R at the FIRST MARKER_ID occurrence
   instead of failing — preserving the construct "log P(marker) at the end
   of the model's own response."
2. **Resume-skip on existence was unsafe.** Each JSON is written
   atomically after EVERY completed target, so a cell that died mid-loop
   leaves a partial file with < 27 targets — the existence-only skip would
   strand those targets forever. ``_cell_complete`` checks parseability +
   target-count.

Tests work on plain integer token-id lists (no tokenizer / no vLLM / no
GPU), since ``_build_probe_ids`` is pure id manipulation by design.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from explore_persona_space.experiments.i488_conditions import CONDITIONS
from scripts.i488_phase4_eval_onpolicy import (
    MARKER_ID,
    _build_probe_ids,
    _cell_complete,
)

# Sentinel token ids that are NOT the marker.
TOK_A = 1001
TOK_B = 1002
TOK_C = 1003
TOK_D = 1004


# -------- _build_probe_ids --------


def test_build_probe_ids_marker_free_r():
    """R without marker: append marker, slot at end, no truncation flags."""
    prompt_ids = [TOK_A, TOK_B]
    r_ids = [TOK_C, TOK_D]
    full_ids, slot, r_contained_marker, r_truncation_idx = _build_probe_ids(prompt_ids, r_ids)
    assert full_ids == [TOK_A, TOK_B, TOK_C, TOK_D, MARKER_ID]
    assert slot == len(full_ids) - 1
    assert full_ids[slot] == MARKER_ID
    assert r_contained_marker is False
    assert r_truncation_idx is None


def test_build_probe_ids_marker_at_end_of_r():
    """R ends with marker: truncate it, slot lands where marker was."""
    prompt_ids = [TOK_A, TOK_B]
    r_ids = [TOK_C, TOK_D, MARKER_ID]
    full_ids, slot, r_contained_marker, r_truncation_idx = _build_probe_ids(prompt_ids, r_ids)
    # R becomes [TOK_C, TOK_D]; full = prompt + R' + [marker].
    assert full_ids == [TOK_A, TOK_B, TOK_C, TOK_D, MARKER_ID]
    assert slot == len(full_ids) - 1
    assert r_contained_marker is True
    assert r_truncation_idx == 2  # marker was at index 2 in r_ids


def test_build_probe_ids_marker_midway_with_trailing_drift():
    """R has marker mid-sequence + trailing drift: truncate at FIRST occurrence."""
    prompt_ids = [TOK_A]
    # marker at idx 1, then 3 drift tokens (one of which is ALSO the marker).
    r_ids = [TOK_B, MARKER_ID, TOK_C, MARKER_ID, TOK_D]
    full_ids, slot, r_contained_marker, r_truncation_idx = _build_probe_ids(prompt_ids, r_ids)
    # R' = [TOK_B]; trailing drift is dropped.
    assert full_ids == [TOK_A, TOK_B, MARKER_ID]
    assert slot == len(full_ids) - 1
    assert full_ids.count(MARKER_ID) == 1  # only the appended one
    assert r_contained_marker is True
    assert r_truncation_idx == 1


def test_build_probe_ids_r_first_token_is_marker():
    """R[0] == marker: R' is empty, slot sits right after the prompt."""
    prompt_ids = [TOK_A, TOK_B, TOK_C]
    r_ids = [MARKER_ID, TOK_D, TOK_D]
    full_ids, slot, r_contained_marker, r_truncation_idx = _build_probe_ids(prompt_ids, r_ids)
    assert full_ids == [TOK_A, TOK_B, TOK_C, MARKER_ID]
    # Slot is right after the prompt (= len(prompt_ids)).
    assert slot == len(prompt_ids)
    assert slot == len(full_ids) - 1
    assert r_contained_marker is True
    assert r_truncation_idx == 0


def test_build_probe_ids_empty_r():
    """R is empty: probe sits right after the prompt."""
    prompt_ids = [TOK_A, TOK_B]
    r_ids: list[int] = []
    full_ids, slot, r_contained_marker, r_truncation_idx = _build_probe_ids(prompt_ids, r_ids)
    assert full_ids == [TOK_A, TOK_B, MARKER_ID]
    assert slot == len(full_ids) - 1
    assert r_contained_marker is False
    assert r_truncation_idx is None


def test_build_probe_ids_marker_in_prompt_raises():
    """Marker in the PROMPT ids is a genuine threading bug — fail fast."""
    prompt_ids = [TOK_A, MARKER_ID, TOK_B]
    r_ids = [TOK_C]
    with pytest.raises(RuntimeError, match=r"marker.*appears in prompt_ids"):
        _build_probe_ids(prompt_ids, r_ids)


# -------- _cell_complete --------


def _write_payload(path: Path, n_targets: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "schema_version": "i488_v1",
                "targets": {f"t{i}": {"placeholder": True} for i in range(n_targets)},
            }
        )
    )


def test_cell_complete_missing_file(tmp_path):
    assert _cell_complete(tmp_path / "nope.json") is False


def test_cell_complete_partial_targets(tmp_path):
    """A file with fewer than 27 targets must NOT be treated as complete."""
    p = tmp_path / "delta_partial.json"
    _write_payload(p, n_targets=len(CONDITIONS) - 1)  # 26 of 27
    assert _cell_complete(p) is False


def test_cell_complete_full_targets(tmp_path):
    """A file with all 27 targets is complete."""
    p = tmp_path / "delta_full.json"
    _write_payload(p, n_targets=len(CONDITIONS))
    assert _cell_complete(p) is True


def test_cell_complete_corrupt_json(tmp_path):
    """An unparseable file is treated as not-complete (forces re-run)."""
    p = tmp_path / "delta_corrupt.json"
    p.write_text("{not valid json{")
    assert _cell_complete(p) is False


def test_cell_complete_no_targets_key(tmp_path):
    """Missing ``targets`` key → not complete (zero != 27)."""
    p = tmp_path / "delta_empty.json"
    p.write_text(json.dumps({"schema_version": "i488_v1"}))
    assert _cell_complete(p) is False
