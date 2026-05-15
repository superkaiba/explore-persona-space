"""Pool-path agreement test between ``__main__._pool_paths`` and ``onpolicy._cache_path``.

Round-2 code-review BLOCKER 1 (both Claude and Codex flagged this):

    ``_pool_paths`` returned ``pool_root/<src>/<src>_a{a}_b{b}_c{c}.jsonl``
    while ``onpolicy._cache_path`` writes
    ``pool_root/<src>/source-<src>_a{a}_b{b}_c{c}.jsonl`` (note the
    ``source-`` prefix). Every D=0 cell (48 of 96) would crash on
    ``FileNotFoundError`` because cell-mode reads from a path the
    dispatch-mode writer never produced.

This test synthesises a fake pool tree at the path ``_cache_path`` would
produce, then asserts ``_pool_paths`` returns:

  * An on-policy path that exists on disk (i.e. the prefix matches).
  * A path that is byte-identical to ``_cache_path``'s output for the
    same ``(source, a, b, c)`` tuple.

The test is parameterised across all 8 ABC triples per source and all
3 source personas (24 combinations) so any future prefix drift trips
immediately.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from explore_persona_space.experiments.factor_screen_365.__main__ import (
    Cell,
    _pool_paths,
)
from explore_persona_space.experiments.factor_screen_365.onpolicy import (
    OnPolicyConfig,
    _cache_path,
)
from explore_persona_space.experiments.factor_screen_365.persona_panel import (
    SOURCE_PERSONAS,
)


@pytest.mark.parametrize("source", SOURCE_PERSONAS)
@pytest.mark.parametrize("a", (0, 1))
@pytest.mark.parametrize("b", (0, 1))
@pytest.mark.parametrize("c", (0, 1))
def test_pool_paths_match_cache_path(tmp_path: Path, source: str, a: int, b: int, c: int) -> None:
    """``_pool_paths``' on-policy path must equal ``_cache_path``'s output.

    Both functions key off ``(source, a, b, c)``; they must agree on the
    filename or the dispatch-mode writer and cell-mode reader will use
    different paths.
    """
    pool_root = tmp_path / "pools"

    # Build the OnPolicyConfig the dispatcher would build.
    cfg = OnPolicyConfig(
        source=source,
        a=a,
        b=b,
        c=c,
        questions=["dummy question"],
        cache_dir=pool_root / source,
    )
    cache_path = _cache_path(cfg)
    assert cache_path is not None, "cache_dir set, expected a concrete path"

    # Synthesise the on-disk file the dispatcher would have written.
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text('{"role": "source", "persona": "x", "completion": "y"}\n')

    # Cell-mode reads via _pool_paths(pool_root, source, cell).
    cell = Cell(a=a, b=b, c=c, d=0, e=0)
    on_policy_path, off_policy_path = _pool_paths(pool_root=pool_root, source=source, cell=cell)

    # (a) Path exists on disk (i.e. the prefix matches what dispatch wrote).
    assert on_policy_path.exists(), (
        f"On-policy path {on_policy_path} does not exist; the dispatcher wrote "
        f"{cache_path}. _pool_paths and _cache_path are out of sync."
    )

    # (b) Path is byte-identical to _cache_path's output.
    assert on_policy_path == cache_path, (
        f"_pool_paths -> {on_policy_path} but _cache_path -> {cache_path}; "
        "the two functions must agree exactly on the on-policy filename."
    )

    # The off-policy path uses the same stem (with an ``_offpolicy`` suffix).
    assert off_policy_path.stem.startswith(cache_path.stem), (
        f"Off-policy path {off_policy_path} should share the dispatcher's "
        f"cache-key stem {cache_path.stem!r}; got {off_policy_path.stem!r}."
    )
    assert "offpolicy" in off_policy_path.stem


def test_pool_paths_use_source_prefix(tmp_path: Path) -> None:
    """Regression: the on-policy filename starts with ``source-`` to match the cache key."""
    pool_root = tmp_path / "pools"
    cell = Cell(a=0, b=1, c=1, d=0, e=0)
    on_policy_path, _ = _pool_paths(pool_root=pool_root, source="surgeon", cell=cell)
    assert on_policy_path.name == "source-surgeon_a0_b1_c1.jsonl", (
        f"Expected filename to be 'source-surgeon_a0_b1_c1.jsonl' but got {on_policy_path.name!r}. "
        "If you renamed the cache key in onpolicy._cache_key, update _pool_paths to match."
    )
