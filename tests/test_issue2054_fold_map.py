"""Pins for the shared guarded fold-map loader (`scripts/issue2054_fold_map.py`, #2245).

Unit tests exercise the REAL loader body on synthetic tmp_path maps (no stubs,
no network): the production floors refuse a 1,761-shaped smoke map;
``allow_smoke=True`` bypasses ONLY the two floors — the fold_of/k/seed
missing-key ValueError stays unconditional (the pinned #2245 semantics).

The durable regression pin reads the COMMITTED
`eval_results/issue_2054/shared_fold_map.json` and asserts it is the
PRODUCTION map (26,889 conversations across 5 variants): main carried a
1,761-conversation single-variant SMOKE map at that canonical path from
2026-08-04 to 2026-08-12, silently collapsing every downstream fit that read
it — any future re-substitution of a smoke map at the canonical path is now an
instant repo-wide test failure. Sparse-worktree note (#671): the committed-map
read requires the `eval_results/issue_2054` cone (registered in
tests/sparse_cones.txt); the pin skips-if-absent for pre-existing worktrees.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
for _p in (str(_REPO_ROOT / "scripts"), str(_REPO_ROOT / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import issue2054_fold_map as fold_map_mod  # noqa: E402

COMMITTED_MAP = _REPO_ROOT / "eval_results" / "issue_2054" / "shared_fold_map.json"


def _write_map(path: Path, payload: dict) -> Path:
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _smoke_shaped_payload(n: int = 1761) -> dict:
    """The 2026-08-04 smoke-map shape: sub-floor n, ONE variant, k/seed matching
    production (both floors are what discriminate it — k/seed do not)."""
    return {
        "fold_of": {f"stripped_s{i:05d}": i % 5 for i in range(n)},
        "k": 5,
        "seed": 137,
        "n_conv_ids": n,
        "variants": ["char_helios"],
    }


def _production_shaped_payload(n: int = 20_000) -> dict:
    return {
        "fold_of": {f"stripped_s{i:05d}": i % 5 for i in range(n)},
        "k": 5,
        "seed": 137,
        "n_conv_ids": n,
        "variants": ["v1", "v2", "v3", "v4", "v5"],
    }


# ---------------------------------------------------------------------------
# Floors (bypassable ONLY via allow_smoke=True)
# ---------------------------------------------------------------------------
def test_floor_refusal_on_smoke_shaped_map(tmp_path):
    p = _write_map(tmp_path / "m.json", _smoke_shaped_payload())
    with pytest.raises(RuntimeError, match=r"REFUSING fold map .*n_conv=1,761"):
        fold_map_mod.load_fold_map(p)


def test_variants_floor_refuses_even_at_conv_floor(tmp_path):
    # n_conv at floor but a single variant: the variants floor alone refuses.
    payload = _production_shaped_payload()
    payload["variants"] = ["char_helios"]
    p = _write_map(tmp_path / "m.json", payload)
    with pytest.raises(RuntimeError, match=r"variants=\['char_helios'\]"):
        fold_map_mod.load_fold_map(p)


def test_allow_smoke_passes_smoke_shaped_map(tmp_path):
    p = _write_map(tmp_path / "m.json", _smoke_shaped_payload())
    d = fold_map_mod.load_fold_map(p, allow_smoke=True)
    assert len(d["fold_of"]) == 1761
    assert d["k"] == 5 and d["seed"] == 137


def test_production_shaped_map_passes_without_bypass(tmp_path):
    p = _write_map(tmp_path / "m.json", _production_shaped_payload())
    d = fold_map_mod.load_fold_map(p)
    assert len(d["fold_of"]) == 20_000
    assert len(d["variants"]) == 5


# ---------------------------------------------------------------------------
# Key checks (UNCONDITIONAL — allow_smoke never bypasses them)
# ---------------------------------------------------------------------------
def test_allow_smoke_on_fold_of_only_map_still_raises_valueerror(tmp_path):
    """The pinned #2245 semantics: allow_smoke bypasses ONLY the two floors —
    a fold_of-only map (no k/seed) is MALFORMED, not merely small."""
    p = _write_map(tmp_path / "m.json", {"fold_of": {"stripped_s0001": 0}})
    with pytest.raises(ValueError, match=r"missing 'k'"):
        fold_map_mod.load_fold_map(p, allow_smoke=True)


@pytest.mark.parametrize("missing", ["fold_of", "k", "seed"])
def test_missing_key_raises_valueerror(tmp_path, missing):
    payload = _smoke_shaped_payload(n=3)
    del payload[missing]
    p = _write_map(tmp_path / "m.json", payload)
    with pytest.raises(ValueError, match=f"missing {missing!r}"):
        fold_map_mod.load_fold_map(p)


@pytest.mark.parametrize("bad_fold_of", [{}, [], "x", 0])
def test_empty_or_nondict_fold_of_raises_valueerror_even_with_allow_smoke(tmp_path, bad_fold_of):
    p = _write_map(tmp_path / "m.json", {"fold_of": bad_fold_of, "k": 5, "seed": 137})
    with pytest.raises(ValueError, match="non-empty 'fold_of' dict"):
        fold_map_mod.load_fold_map(p, allow_smoke=True)


def test_missing_file_raises_filenotfounderror(tmp_path):
    with pytest.raises(FileNotFoundError, match="shared_fold_map not found"):
        fold_map_mod.load_fold_map(tmp_path / "nonexistent.json")


# ---------------------------------------------------------------------------
# The durable pin: the COMMITTED canonical map is the PRODUCTION map
# ---------------------------------------------------------------------------
@pytest.mark.skipif(
    not COMMITTED_MAP.exists(),
    reason="committed shared_fold_map.json not checked out (sparse worktree without the "
    "eval_results/issue_2054 cone; new_worktree.sh pre-adds it from tests/sparse_cones.txt)",
)
def test_committed_canonical_map_is_production_map():
    d = json.loads(COMMITTED_MAP.read_text(encoding="utf-8"))
    assert d["n_conv_ids"] >= fold_map_mod.FOLD_MAP_MIN_CONV, (
        f"committed shared_fold_map.json reads n_conv_ids={d['n_conv_ids']} < "
        f"{fold_map_mod.FOLD_MAP_MIN_CONV}: a SMOKE map has been re-substituted at the "
        "canonical path (#2245)"
    )
    assert len(d["fold_of"]) == d["n_conv_ids"]
    assert len(d["variants"]) >= fold_map_mod.FOLD_MAP_MIN_VARIANTS, d["variants"]
    assert d["k"] == 5
    assert d["seed"] == 137
    # The guarded loader accepts it without any bypass.
    loaded = fold_map_mod.load_fold_map(COMMITTED_MAP)
    assert loaded["n_conv_ids"] == d["n_conv_ids"]
