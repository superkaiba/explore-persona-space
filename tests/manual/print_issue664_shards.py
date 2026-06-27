"""Manual dry-run: print issue664 p2 shard sizes and assert they sum to the
full realized grid (no GPU, no network). #664 round-7 8-way data parallelism.

    uv run python tests/manual/print_issue664_shards.py

Prints the per-shard cell count for a few (shard_id, num_shards) configurations
and asserts the 8-way partition covers every cell exactly once.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = REPO_ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import issue664_common as C  # noqa: E402
import issue664_dispatch as D  # noqa: E402


def main() -> int:
    grid = C.realized_grid()
    total = len(grid)
    print(f"realized grid: {total} cells")

    # num_shards=1 -> full grid unchanged.
    full = D._shard_filter(grid, shard_id=0, num_shards=1)
    print(f"num_shards=1, shard_id=0 -> {len(full)} cells (full grid)")
    assert len(full) == total

    # num_shards=8 -> per-shard sizes summing to the full grid.
    num_shards = 8
    sizes = []
    seen: set[str] = set()
    for s in range(num_shards):
        cells = D._shard_filter(grid, shard_id=s, num_shards=num_shards)
        sizes.append(len(cells))
        keys = {c.eval_key for c in cells}
        assert not (keys & seen), f"shard {s} overlaps an earlier shard"
        seen |= keys
        print(f"num_shards=8, shard_id={s} -> {len(cells)} cells")

    print(f"sum of 8 shard sizes = {sum(sizes)} (expected {total})")
    assert sum(sizes) == total, (sum(sizes), total)
    assert seen == {c.eval_key for c in grid}, "8-shard union != full grid"
    print(
        f"OK: 8 shards partition all {total} cells exactly once "
        f"(sizes {sizes}, balance max-min={max(sizes) - min(sizes)})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
