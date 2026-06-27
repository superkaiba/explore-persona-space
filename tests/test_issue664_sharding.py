"""Issue #664 round-7 invariant pins: data-parallel p2 shard filter.

The 8-way wrapper (``scripts/issue664_launch_parallel.sh``) fans the independent
(source x behavior x arm x dose) cells across 8 GPUs by partitioning the
post-drop cell list with ``i % num_shards == shard_id`` inside
``issue664_dispatch.run_all``. This is pure orchestration -- the per-cell science
behavior is unchanged -- so the smoke is a CPU unit test of the partition logic
+ its validation, NOT a GPU run.

Pins (so a future refactor cannot silently strip them):

- ``num_shards == 1`` preserves the full grid (current single-process behavior).
- ``num_shards == 8`` partitions into 8 DISJOINT subsets whose UNION is the full
  grid (every cell lands on exactly one shard -- no cell dropped, none duplicated).
- ``_validate_shard`` raises on out-of-range ``shard_id`` and ``num_shards < 1``.

All CPU-only; imports the ``scripts/issue664_*`` modules and calls the pure-Python
helpers directly (no dispatcher main, no GPU, no network).
"""

from __future__ import annotations

import sys
import types
from pathlib import Path
from unittest import mock

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = REPO_ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import issue664_common as C  # noqa: E402
import issue664_dispatch as D  # noqa: E402


def _grid() -> list[C.Cell]:
    return C.realized_grid()


def test_num_shards_one_returns_full_grid_unchanged() -> None:
    """num_shards=1 (the default) returns the input list unchanged -- the existing
    single-process behavior is preserved bit-for-bit."""
    grid = _grid()
    out = D._shard_filter(grid, shard_id=0, num_shards=1)
    assert out == grid
    assert len(out) == len(grid)


def test_eight_shards_disjoint_union_covers_full_grid() -> None:
    """num_shards=8: each shard_id 0..7 returns a disjoint subset; the 8 subsets
    union (in order) to the full grid -- no cell dropped, none duplicated."""
    grid = _grid()
    num_shards = 8
    shards = [D._shard_filter(grid, shard_id=s, num_shards=num_shards) for s in range(num_shards)]

    # Disjoint: every cell index lands on exactly one shard.
    seen_keys: set[str] = set()
    for s, shard_cells in enumerate(shards):
        keys = {c.eval_key for c in shard_cells}
        assert not (keys & seen_keys), f"shard {s} overlaps an earlier shard"
        seen_keys |= keys

    # Union == full grid (by eval_key, the unique cell identifier).
    assert seen_keys == {c.eval_key for c in grid}
    # Sizes sum to the full grid count.
    assert sum(len(s) for s in shards) == len(grid)
    # Round-robin balance: shard sizes differ by at most 1.
    sizes = [len(s) for s in shards]
    assert max(sizes) - min(sizes) <= 1, sizes


def test_eight_shards_exact_round_robin_assignment() -> None:
    """Shard s owns grid[i] iff i % 8 == s (the exact index partition)."""
    grid = _grid()
    num_shards = 8
    for s in range(num_shards):
        expected = [c for i, c in enumerate(grid) if i % num_shards == s]
        assert D._shard_filter(grid, shard_id=s, num_shards=num_shards) == expected


def test_validate_shard_rejects_out_of_range_shard_id() -> None:
    """shard_id < 0 or shard_id >= num_shards raises ValueError."""
    with pytest.raises(ValueError):
        D._validate_shard(shard_id=-1, num_shards=8)
    with pytest.raises(ValueError):
        D._validate_shard(shard_id=8, num_shards=8)
    with pytest.raises(ValueError):
        D._validate_shard(shard_id=9, num_shards=8)


def test_validate_shard_rejects_num_shards_below_one() -> None:
    """num_shards < 1 raises ValueError."""
    with pytest.raises(ValueError):
        D._validate_shard(shard_id=0, num_shards=0)
    with pytest.raises(ValueError):
        D._validate_shard(shard_id=0, num_shards=-3)


def test_shard_filter_validates_before_filtering() -> None:
    """_shard_filter runs validation first -- a bad (shard_id, num_shards) raises
    rather than silently returning an empty/wrong subset."""
    grid = _grid()
    with pytest.raises(ValueError):
        D._shard_filter(grid, shard_id=8, num_shards=8)
    with pytest.raises(ValueError):
        D._shard_filter(grid, shard_id=0, num_shards=0)


# ── #664 round-8: all-fleet finalizers run AT p3, NOT inside the p2 worker ───────
# r7 placed _write_manifest / _marker_readability_assert / _live_judge_smoke inside
# the --phase p2 block, so shard 0's p2 process fired the A7 readability HALT on its
# own 2 marker cells the instant it finished -- racing the ~18 marker cells on shards
# 1-7 whose marker_slot_stats.json did not exist yet (silently `continue`d). The fix
# moves them AHEAD of upload_artifacts in --phase p3, which the wrapper invokes only
# after `wait`ing for every p2 shard. This pins WHERE each finalizer runs so a future
# refactor cannot silently slide them back into p2 (concern post-p2-finalizers-race).
def _run_all_with_mocked_helpers(phase: str) -> tuple[dict[str, mock.MagicMock], mock.Mock]:
    """Call ``run_all`` for ``phase`` (shard 0 / single-process) with every heavy
    helper monkey-patched out. Returns (per-helper mock dict, ordering-manager mock).

    The manager records call order across the three finalizers + upload so a test can
    assert the A7 readability assert runs BEFORE the upload. No GPU, no network, no
    real cell list (helpers are all mocked, so the cells are opaque objects)."""
    args = types.SimpleNamespace(
        phase=phase,
        smoke=False,
        gpu_id=0,
        shard_id=0,
        num_shards=1,
        live_judge_smoke=False,
    )
    # the p2 extract loop reads cell.eval_key to build the adapter dir name; everything
    # else that touches a cell (train_cell, extract_and_eval_cell, the finalizers) is
    # mocked, so a minimal namespace with just eval_key suffices.
    fake_cells = [types.SimpleNamespace(eval_key=f"cell{i}") for i in range(3)]
    patches = {
        "_require_credentials": mock.DEFAULT,
        "_validate_shard": mock.DEFAULT,
        "_select_cells": mock.DEFAULT,
        "_drop_filtered": mock.DEFAULT,
        "_shard_filter": mock.DEFAULT,
        "phase0": mock.DEFAULT,
        "train_cell": mock.DEFAULT,
        "extract_and_eval_cell": mock.DEFAULT,
        # the all-fleet finalizers whose placement we are pinning:
        "_write_manifest": mock.DEFAULT,
        "_marker_readability_assert": mock.DEFAULT,
        "_live_judge_smoke": mock.DEFAULT,
        "upload_artifacts": mock.DEFAULT,
    }
    with mock.patch.multiple(D, **patches) as mocks:
        mocks["_select_cells"].return_value = fake_cells
        mocks["_drop_filtered"].return_value = fake_cells
        mocks["_shard_filter"].return_value = fake_cells
        # attach the finalizers to one parent so cross-mock call ORDER is recorded.
        manager = mock.Mock()
        manager.attach_mock(mocks["_write_manifest"], "manifest")
        manager.attach_mock(mocks["_marker_readability_assert"], "a7")
        manager.attach_mock(mocks["upload_artifacts"], "upload")
        D.run_all(args)
    return mocks, manager


def test_p2_phase_does_not_run_all_fleet_finalizers() -> None:
    """--phase p2 must NOT call _write_manifest / _marker_readability_assert /
    upload_artifacts -- those describe the WHOLE fleet and run at p3 (post-p2-wait).
    Running them in the p2 worker races concurrent shards (#664 r7 FAIL B3)."""
    mocks, _ = _run_all_with_mocked_helpers("p2")
    # p2 DOES do per-cell extract+eval work.
    assert mocks["extract_and_eval_cell"].called
    # ...but NONE of the all-fleet finalizers.
    mocks["_write_manifest"].assert_not_called()
    mocks["_marker_readability_assert"].assert_not_called()
    mocks["_live_judge_smoke"].assert_not_called()
    mocks["upload_artifacts"].assert_not_called()


def test_p3_phase_runs_all_fleet_finalizers_before_upload() -> None:
    """--phase p3 (shard 0) must call _write_manifest then _marker_readability_assert
    then upload_artifacts -- the A7 readability HALT happens BEFORE the upload, and the
    full marker_slot_stats.json set exists because the wrapper waited for every p2
    shard before invoking p3."""
    mocks, manager = _run_all_with_mocked_helpers("p3")
    mocks["_write_manifest"].assert_called_once()
    mocks["_marker_readability_assert"].assert_called_once()
    mocks["upload_artifacts"].assert_called_once()
    # p3 does NOT redo per-cell extract+eval.
    mocks["extract_and_eval_cell"].assert_not_called()
    # Ordering across mocks (recorded on the shared manager): manifest -> a7 -> upload.
    # The A7 readability HALT must run strictly BEFORE the upload (HALT-before-push).
    ordered = [name for name, _args, _kw in manager.mock_calls]
    assert ordered == ["manifest", "a7", "upload"], ordered
