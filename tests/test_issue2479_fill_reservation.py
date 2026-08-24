"""#2479 r2 regression pins for the fill's axis-reservation exclusion (+ g2 cache key).

Pins the fill half of the round-1 codex blocker
``manifest-and-reservation-disconnected``: `scripts/issue1345_story_char_ladder_fill.py`
previously fit whatever rows the staged turnstore held — including the 250
``axis_reservation_conv_ids`` that feed the AXIS judging — violating the plan's
hard axis/DV independence. `load_regime_xy` now applies
``exclude_axis_reservation`` on EVERY load (cache hits included) for
``char_2479_*`` regimes, fail-loud postcondition, while parent regimes stay
byte-identical. Also pins the g2 r1 Minor: char-cell slice-cache filenames key
on the stage root (``_sr<hash8>``); parent regimes keep the legacy name.

Hermetic: tmp-path manifests + caches, synthetic REGIME_SPECS entries. No
network, no staged stores, no GPU.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

REPO = Path(__file__).resolve().parent.parent
SCRIPTS = REPO / "scripts"

if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import issue1345_story_char_ladder_fill as fill  # noqa: E402

PANEL_REGIME = "char_2479_synthchar"


@pytest.fixture()
def manifest_env(monkeypatch, tmp_path):
    mp = tmp_path / "panel_manifest.json"
    mp.write_text(json.dumps({"axis_reservation_conv_ids": ["res1", "res2"], "n_reservation": 2}))
    monkeypatch.setenv(fill.I2479_MANIFEST_ENV, str(mp))
    monkeypatch.setattr(fill, "_AXIS_RESERVATION_IDS", None)
    return mp


def _block(conv_ids: list[str]) -> dict:
    n = len(conv_ids)
    x = torch.arange(n * 3, dtype=torch.float32).reshape(n, 3)
    return {"X": x, "Y": x + 100.0, "conv_ids": np.asarray(conv_ids)}


def test_exclusion_drops_reserved_rows_consistently(manifest_env):
    blk = _block(["a", "res1", "b", "res2", "c"])
    out = fill.exclude_axis_reservation(blk, "test")
    assert list(out["conv_ids"]) == ["a", "b", "c"]
    assert torch.equal(out["X"], blk["X"][[0, 2, 4]])
    assert torch.equal(out["Y"], blk["Y"][[0, 2, 4]])


def test_exclusion_zero_drop_is_passthrough(manifest_env):
    blk = _block(["a", "b"])
    out = fill.exclude_axis_reservation(blk, "test")
    assert out is blk


def test_axis_reservation_ids_fail_loud_without_manifest(monkeypatch, tmp_path):
    monkeypatch.setenv(fill.I2479_MANIFEST_ENV, str(tmp_path / "absent.json"))
    monkeypatch.setattr(fill, "_AXIS_RESERVATION_IDS", None)
    with pytest.raises(RuntimeError, match="panel manifest"):
        fill.axis_reservation_ids()


def test_load_regime_xy_cache_hit_applies_exclusion_and_root_tag(
    manifest_env, monkeypatch, tmp_path
):
    """Panel regimes: stage-root-keyed cache name + exclusion on the CACHE-HIT path."""
    spec = {
        "format_key": "stories_paired_op",
        "subdir": f"{PANEL_REGIME}_turnstore",
        "turn": 0,
        "cache_key": PANEL_REGIME,
        "model": "instruct",
    }
    monkeypatch.setitem(fill.REGIME_SPECS, PANEL_REGIME, spec)
    stage_root = tmp_path / "stage"
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    root_tag = hashlib.sha256(str(stage_root.resolve()).encode()).hexdigest()[:8]
    cache = cache_dir / f"instruct_{PANEL_REGIME}_{fill.c.TRACK}_context_L19_sr{root_tag}.pt"
    x = torch.arange(12, dtype=torch.float32).reshape(4, 3)
    torch.save({"X": x, "Y": x + 100.0, "conv_ids": ["a", "res1", "b", "res2"]}, cache)
    out = fill.load_regime_xy(stage_root, cache_dir, "instruct", PANEL_REGIME, "context", 19)
    assert list(out["conv_ids"]) == ["a", "b"]
    assert torch.equal(out["X"], x[[0, 2]])
    assert torch.equal(out["Y"], (x + 100.0)[[0, 2]])


def test_parent_regime_cache_name_and_rows_unchanged(monkeypatch, tmp_path):
    """Parent regimes: legacy cache filename, NO exclusion (byte-identical rows)."""
    stage_root = tmp_path / "stage"
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    # r1 spec: format_key="chat", no cache_key -> legacy name, no _sr tag.
    cache = cache_dir / f"instruct_chat_{fill.c.TRACK}_context_L19.pt"
    x = torch.arange(9, dtype=torch.float32).reshape(3, 3)
    # "res1" is a reserved id under the panel manifest — a parent regime must
    # keep it (no manifest read at all on this path).
    torch.save({"X": x, "Y": x + 1.0, "conv_ids": ["a", "res1", "b"]}, cache)
    monkeypatch.setattr(fill, "_AXIS_RESERVATION_IDS", None)
    out = fill.load_regime_xy(stage_root, cache_dir, "instruct", "r1", "context", 19)
    assert list(out["conv_ids"]) == ["a", "res1", "b"]
    assert torch.equal(out["X"], x)
