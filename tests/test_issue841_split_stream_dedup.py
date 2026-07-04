# ruff: noqa: E402
"""Parent-collision filter + backfill for the issue #841 scaling-capture split_stream.

lmsys-chat-1m repeats prompt STRINGS across rows, so the raw new-pool stream
positions (5001+) collide with parent-5000 strings by construction (~0.8% measured,
763/96000). The crash-fix (round 12) replaced the unsatisfiable empty-overlap assert
with a filter: drop parent-colliding new prompts, backfill from later stream positions
to exactly n_new clean, keep new-internal dupes. These tests pin the invariant in CI:

  * a buffer WITH planted collisions now SUCCEEDS (pre-fix it raised AssertionError);
  * the clean pool is exactly n_new and disjoint from the parent set;
  * new-internal duplicates are KEPT (only cross-parent collisions dropped);
  * an under-supplied buffer HARD-FAILS (never silently under-fills).

They exercise the LIVE dispatched function `issue841_scaling_capture.split_stream`
(the one `main` calls), per the "verification gates test the live dispatched path" rule.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "src"))

import issue841_scaling_capture as CAP
import issue841_scaling_common as S


def _buffer(n_parent, n_new, n_collisions, n_internal_dupes, n_extra):
    """Deterministic buffer: n_parent unique parent prompts, then a new-pool region
    seeded with `n_collisions` parent-string repeats + `n_internal_dupes` new-internal
    repeats interleaved among fresh new prompts, plus `n_extra` trailing fresh prompts."""
    parent = [f"parent-prompt-{i}" for i in range(n_parent)]
    fresh = [f"new-prompt-{i}" for i in range(n_new + n_extra)]
    tail = []
    # interleave collisions (parent repeats) and internal dupes among the fresh new pool
    for k in range(n_collisions):
        tail.append(parent[k % n_parent])  # a verbatim parent string -> must be dropped
    for k in range(n_internal_dupes):
        tail.append(fresh[k])  # a repeat of an already-listed new prompt -> must be KEPT
    buf = parent + tail + fresh
    return buf, parent


def test_filters_parent_collisions_and_backfills():
    n_parent, n_new = S.N_PARENT, 200
    buf, parent = _buffer(n_parent, n_new, n_collisions=20, n_internal_dupes=5, n_extra=100)
    parent_out, new_clean, dropped, extent = CAP.split_stream(buf, n_new)

    assert parent_out == parent  # parent is the untouched first N_PARENT
    assert len(new_clean) == n_new  # backfilled to exactly n_new
    assert dropped == 20  # every planted parent-collision dropped
    assert set(new_clean).isdisjoint(set(parent))  # the §4.1 contamination invariant
    assert extent > n_parent  # consumed past the parent block


def test_keeps_new_internal_duplicates():
    """Only CROSS-parent collisions are dropped; new-internal dupes stay (no 2nd variable)."""
    n_parent, n_new = S.N_PARENT, 50
    buf, _ = _buffer(n_parent, n_new, n_collisions=3, n_internal_dupes=10, n_extra=80)
    _, new_clean, _, _ = CAP.split_stream(buf, n_new)
    assert len(new_clean) == n_new
    # the pool is allowed to contain internal duplicates (fewer uniques than rows)
    assert len(set(new_clean)) <= len(new_clean)
    # and it genuinely retained at least one internal duplicate given we planted 10
    assert len(set(new_clean)) < len(new_clean)


def test_hard_fails_when_buffer_underfills():
    """Not enough clean prompts after dropping collisions -> RuntimeError, never silent."""
    n_parent, n_new = S.N_PARENT, 100
    # only 100 fresh available but 60 collisions consume the margin -> can't reach 100 clean
    parent = [f"p-{i}" for i in range(n_parent)]
    tail = [parent[k % n_parent] for k in range(60)]  # 60 parent-collisions
    fresh = [f"n-{i}" for i in range(80)]  # only 80 fresh -> < n_new=100
    buf = parent + tail + fresh
    with pytest.raises(RuntimeError, match="could not fill"):
        CAP.split_stream(buf, n_new)


def test_rejects_too_short_buffer():
    """Buffer shorter than parent + n_new fails the precondition assert."""
    buf = [f"p-{i}" for i in range(S.N_PARENT + 10)]
    with pytest.raises(AssertionError):
        CAP.split_stream(buf, 100)
