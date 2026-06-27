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
from pathlib import Path

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
