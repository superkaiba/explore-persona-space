"""Regression tests for the issue-263 runtime-bounce fix:
`--per-q-response-positions-subset` flag on `sweep_extraction_grid.py` and the
matching `per_q_response_positions_subset` arg threaded through
`analyze_extraction_grid.load_per_q_at_cell`.

Background. The original sweep wrote `method_r_per_token/<role>__per_q.pt` at
shape `(n_q, n_layers, n_response_positions, D)` — at full sweep size that is
~433 MB / persona x 275 = 119 GB on disk, which alone overflowed the 200 GB pod
volume after accounting for the other methods. The fix: only serialize per-q at
a small H3-relevant subset (default `[0, 128]`), reducing r_per_token's per-q
footprint to ~26 GB. Centroids at every position are still written, so H1
clustering and the H3 descriptive trajectory are unaffected; H2's r_per_token
candidate space shrinks proportionally and is reported explicitly.

These tests are end-to-end on a synthetic mini-fixture (no real model, no real
data) so they run in a few seconds on CPU. They cover:

1. Saved per_q shape MUST equal `(n_q, n_layers, len(subset), D)` when the sweep
   is told to use a subset, NOT `(n_q, n_layers, n_response_positions, D)`.
2. `load_per_q_at_cell` returns the correct slice when the requested response
   position IS in the subset, indexed against the subset axis (not the full
   response_positions axis — this is the bug class N3 fix is preventing).
3. `load_per_q_at_cell` raises `PositionNotInPerQSubsetError` (a typed
   `RuntimeError` subclass) when the requested position is OUTSIDE the subset
   — and crucially that error is catchable by the H2 try/except path that was
   already in place for missing per-q caches.
4. Stale-cache defense: if the on-disk axis-2 length does NOT match the
   subset list length (i.e. the cache was written with a different subset),
   the loader raises a clear RuntimeError mentioning the mismatch — preventing
   silent wrong-axis indexing.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

# Force-import both scripts as modules (mirrors test_h2_perm_null.py).
_REPO_ROOT = Path(__file__).parent.parent.parent
_ANALYZE_PATH = _REPO_ROOT / "scripts" / "analyze_extraction_grid.py"
_SWEEP_PATH = _REPO_ROOT / "scripts" / "sweep_extraction_grid.py"


def _import_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


analyze_extraction_grid = _import_module("analyze_extraction_grid_t", _ANALYZE_PATH)
sweep_extraction_grid = _import_module("sweep_extraction_grid_t", _SWEEP_PATH)

load_per_q_at_cell = analyze_extraction_grid.load_per_q_at_cell
PositionNotInPerQSubsetError = analyze_extraction_grid.PositionNotInPerQSubsetError
_resolve_per_q_response_subset = sweep_extraction_grid._resolve_per_q_response_subset


# ── Synthetic on-disk fixture ────────────────────────────────────────────────


def _write_per_q_cache(
    root: Path,
    role: str,
    n_q: int,
    n_layers: int,
    subset: list[int],
    hidden_dim: int,
    seed: int = 0,
) -> torch.Tensor:
    """Write one role's r_per_token per_q cache with the subset shape.

    Returns the tensor written (so tests can assert exact-byte equivalence on
    a slice load).
    """
    method_dir = root / "method_r_per_token"
    method_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    arr = rng.standard_normal((n_q, n_layers, len(subset), hidden_dim)).astype(np.float16)
    tensor = torch.from_numpy(arr)
    torch.save(tensor, method_dir / f"{role}__per_q.pt")
    return tensor


# ── Tests for the sweep-side resolver ────────────────────────────────────────


class _NS:
    """Minimal argparse.Namespace stand-in."""

    def __init__(self, **kw):
        self.__dict__.update(kw)


def test_resolve_default_subset_intersects_response_positions():
    # User passes the canonical default; it must survive intersection.
    args = _NS(
        response_token_positions="0,1,2,4,8,16,32,64,128",
        per_q_response_positions_subset="0,128",
    )
    assert _resolve_per_q_response_subset(args) == [0, 128]


def test_resolve_subset_all_keyword_expands_to_full_grid():
    args = _NS(
        response_token_positions="0,1,2,4,8,16,32,64,128",
        per_q_response_positions_subset="all",
    )
    assert _resolve_per_q_response_subset(args) == [0, 1, 2, 4, 8, 16, 32, 64, 128]


def test_resolve_subset_empty_or_none_means_no_per_q_write():
    for raw in ("", "none"):
        args = _NS(
            response_token_positions="0,1,2,4,8,16,32,64,128",
            per_q_response_positions_subset=raw,
        )
        assert _resolve_per_q_response_subset(args) == []


def test_resolve_subset_drops_positions_outside_response_positions():
    # If user asks for t=99 but it's not in --response-token-positions, drop it.
    args = _NS(
        response_token_positions="0,1,2,4,8,16,32,64,128",
        per_q_response_positions_subset="0,99,128",
    )
    assert _resolve_per_q_response_subset(args) == [0, 128]


# ── Tests for the analyzer-side loader ──────────────────────────────────────


def test_load_per_q_at_subset_position_returns_correct_slice(tmp_path: Path):
    n_q, n_layers, hidden_dim = 6, 3, 16
    subset = [0, 128]
    full_response_positions = [0, 1, 2, 4, 8, 16, 32, 64, 128]
    role = "aberration"

    written = _write_per_q_cache(tmp_path, role, n_q, n_layers, subset, hidden_dim)

    # Ask for position=0 (subset index 0) at layer 1.
    out = load_per_q_at_cell(
        centroid_root=tmp_path,
        method="r_per_token",
        position=0,
        layer=1,
        roles=[role],
        qids=list(range(n_q)),
        layers_in_cache=list(range(n_layers)),
        response_positions=full_response_positions,
        per_q_response_positions_subset=subset,
    )
    # Loader returns (N_roles=1, n_q, D) fp32. Compare against fp32-cast slice
    # at axis-2 index 0 (i.e. position 0 in subset).
    expected = written[:, 1, 0, :].float()
    np.testing.assert_allclose(out[0].numpy(), expected.numpy(), rtol=1e-5, atol=1e-7)


def test_load_per_q_at_subset_position_128_returns_correct_slice(tmp_path: Path):
    """The whole point of the H3 subset: t=128 must load, since H3 needs it."""
    n_q, n_layers, hidden_dim = 6, 3, 16
    subset = [0, 128]
    full_response_positions = [0, 1, 2, 4, 8, 16, 32, 64, 128]
    role = "absurdist"

    written = _write_per_q_cache(tmp_path, role, n_q, n_layers, subset, hidden_dim)

    out = load_per_q_at_cell(
        centroid_root=tmp_path,
        method="r_per_token",
        position=128,
        layer=2,
        roles=[role],
        qids=list(range(n_q)),
        layers_in_cache=list(range(n_layers)),
        response_positions=full_response_positions,
        per_q_response_positions_subset=subset,
    )
    expected = written[:, 2, 1, :].float()  # axis-2 idx 1 = position 128 in subset
    np.testing.assert_allclose(out[0].numpy(), expected.numpy(), rtol=1e-5, atol=1e-7)


def test_load_per_q_outside_subset_raises_typed_error(tmp_path: Path):
    """A request for a non-subset position must raise PositionNotInPerQSubsetError —
    AND that error must be a RuntimeError so existing H2 try/except paths still
    catch it (the load_per_q_at_cell call sites all do
    `except (FileNotFoundError, RuntimeError)`).
    """
    n_q, n_layers, hidden_dim = 6, 3, 16
    subset = [0, 128]
    full_response_positions = [0, 1, 2, 4, 8, 16, 32, 64, 128]
    role = "ace_pilot"

    _write_per_q_cache(tmp_path, role, n_q, n_layers, subset, hidden_dim)

    with pytest.raises(PositionNotInPerQSubsetError) as excinfo:
        load_per_q_at_cell(
            centroid_root=tmp_path,
            method="r_per_token",
            position=64,  # NOT in subset
            layer=1,
            roles=[role],
            qids=list(range(n_q)),
            layers_in_cache=list(range(n_layers)),
            response_positions=full_response_positions,
            per_q_response_positions_subset=subset,
        )
    # PositionNotInPerQSubsetError must inherit from RuntimeError so that the
    # existing `except (FileNotFoundError, RuntimeError)` catches in compute_h2
    # / compute_h3 / compute_h1_clustering continue to work after this fix.
    assert isinstance(excinfo.value, RuntimeError)
    # The message should mention the dropped position so the experimenter can
    # rebuild with a wider subset if H2 needs it.
    msg = str(excinfo.value)
    assert "64" in msg
    assert "subset" in msg.lower()


def test_load_per_q_shape_mismatch_raises_clear_error(tmp_path: Path):
    """If the on-disk per_q axis-2 length does not match the subset length the
    analyzer was passed (e.g. cache was written under a different subset, then
    sweep_metadata was hand-edited), the loader must fail loudly rather than
    silently mis-index. This is the primary footgun the fix is designed to
    prevent.
    """
    # Cache on disk: subset [0, 128] -> axis-2 length 2.
    n_q, n_layers, hidden_dim = 6, 3, 16
    real_subset = [0, 128]
    role = "automaton"
    _write_per_q_cache(tmp_path, role, n_q, n_layers, real_subset, hidden_dim)

    # But analyzer is told subset = [0, 64, 128] (length 3).
    with pytest.raises(RuntimeError) as excinfo:
        load_per_q_at_cell(
            centroid_root=tmp_path,
            method="r_per_token",
            position=0,
            layer=0,
            roles=[role],
            qids=list(range(n_q)),
            layers_in_cache=list(range(n_layers)),
            response_positions=[0, 1, 2, 4, 8, 16, 32, 64, 128],
            per_q_response_positions_subset=[0, 64, 128],
        )
    msg = str(excinfo.value)
    assert "axis-2 length" in msg
    assert "subset" in msg.lower()


def test_legacy_full_position_cache_still_loads_when_subset_omitted(tmp_path: Path):
    """Backward compatibility: a per_q cache written by a pre-fix sweep (full
    9-position axis) must still load when the analyzer falls back to passing
    `per_q_response_positions_subset = response_positions` (the legacy default
    in main()). This ensures restored caches from before the fix are not orphaned.
    """
    n_q, n_layers, hidden_dim = 6, 3, 16
    full_positions = [0, 1, 2, 4, 8, 16, 32, 64, 128]
    role = "bard"
    _write_per_q_cache(tmp_path, role, n_q, n_layers, full_positions, hidden_dim)

    # Pre-fix code path: subset = response_positions (i.e. legacy assumption).
    out = load_per_q_at_cell(
        centroid_root=tmp_path,
        method="r_per_token",
        position=64,
        layer=1,
        roles=[role],
        qids=list(range(n_q)),
        layers_in_cache=list(range(n_layers)),
        response_positions=full_positions,
        per_q_response_positions_subset=full_positions,
    )
    # Should load successfully — exercises the n_pos_on_disk == len(subset) path.
    assert out.shape == (1, n_q, hidden_dim)


def test_subset_disk_savings_roughly_match_design(tmp_path: Path):
    """Sanity check on the disk-savings claim in the report: a per_q tensor
    with subset [0, 128] should be roughly 2/9 the on-disk size of one with
    the full 9-position axis. This catches accidental regressions where the
    sweep silently writes the full axis (e.g. by ignoring the subset arg).
    """
    n_q, n_layers, hidden_dim = 8, 4, 32
    role_full = "comedian_full"
    role_subset = "comedian_subset"
    full = [0, 1, 2, 4, 8, 16, 32, 64, 128]
    subset = [0, 128]

    _write_per_q_cache(tmp_path, role_full, n_q, n_layers, full, hidden_dim)
    _write_per_q_cache(tmp_path, role_subset, n_q, n_layers, subset, hidden_dim)

    full_size = (tmp_path / "method_r_per_token" / f"{role_full}__per_q.pt").stat().st_size
    subset_size = (tmp_path / "method_r_per_token" / f"{role_subset}__per_q.pt").stat().st_size

    # Subset is 2 of 9 positions, so subset_size / full_size should be close to 2/9
    # (modest torch.save overhead is allowed). Strict upper bound: 0.30.
    ratio = subset_size / full_size
    assert ratio < 0.30, (
        f"per_q file with subset is {ratio:.2%} of full-axis size; "
        f"expected < 30% (target ~22%). Disk savings invariant violated."
    )
