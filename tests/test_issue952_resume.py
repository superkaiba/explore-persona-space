"""Unit tests for the #952 round-2 resume predicates (binding concern
long-loop-restartability-1c-1e-blocking): the 1c per-arm capture done sentinels
and the 1e battery per-unit checkpoint shards, including regime-mismatch
invalidation (#722 r3 lesson: a resume must key on EVERY output-affecting
regime key — a resume that ignores one silently reuses wrong cached rows).
"""

import json

import numpy as np

from explore_persona_space.experiments.issue_952 import run_952 as r


def _fake_arm_files(base_dir, arm):
    td = base_dir / "analysis_tensors"
    td.mkdir(parents=True, exist_ok=True)
    files = []
    for layer in r.LAYER_GRID:
        p = td / f"slots_{arm}_L{layer}.pt"
        p.write_bytes(b"x")
        files.append(p)
    for name in (f"spans_{arm}.json", f"surprisal_{arm}.npz"):
        p = td / name
        p.write_bytes(b"x")
        files.append(p)
    return files


def test_capture_done_roundtrip_and_invalidation(tmp_path):
    """1c sentinel round-trips; regime drift or a missing shard forces recapture."""
    pool = list(range(10))
    regime = r._capture_regime(pool, smoke=True, batch_size=4)
    files = _fake_arm_files(tmp_path, "own")
    equiv = {"arm": "own", "frac_ok": 1.0}
    assert r._load_capture_done(tmp_path, "own", regime) is None  # no sentinel yet
    r._write_capture_done(tmp_path, "own", regime, files, equiv)
    rec = r._load_capture_done(tmp_path, "own", regime)
    assert rec is not None
    assert rec["equiv"] == equiv
    assert len(rec["files"]) == len(files)
    # Pool drift -> recapture.
    other_pool = r._capture_regime(list(range(11)), smoke=True, batch_size=4)
    assert r._load_capture_done(tmp_path, "own", other_pool) is None
    # Batch-size drift is output-affecting (bf16 batched numerics) -> recapture.
    other_bs = r._capture_regime(pool, smoke=True, batch_size=8)
    assert r._load_capture_done(tmp_path, "own", other_bs) is None
    # A missing listed shard file -> recapture.
    files[0].unlink()
    assert r._load_capture_done(tmp_path, "own", regime) is None


def test_battery_ckpt_roundtrip_and_regime_invalidation(tmp_path):
    """1e unit shards round-trip arrays + payload; a regime change wipes them."""
    split = {"train": [1, 2], "val": [3], "test": [4]}
    regime = r._battery_regime(True, [1, 2, 3, 4], split, "cpu", False, 4)
    ck = r._init_battery_ckpt(tmp_path, regime)
    arrays = {"val_a": np.arange(6, dtype=np.float64).reshape(2, 3)}
    payload = {"svd_seconds": 1.5, "cells": {"k": {"v": 1}}}
    p = ck / "pass1_L14.npz"
    assert r._ckpt_load(p) is None
    r._ckpt_save(p, arrays, payload)
    loaded = r._ckpt_load(p)
    assert loaded is not None
    arrs, pl = loaded
    assert np.array_equal(arrs["val_a"], arrays["val_a"])
    assert pl == payload
    # Same-regime re-init keeps the unit shards (the resume path).
    r._init_battery_ckpt(tmp_path, regime)
    assert p.exists()
    # A changed regime (fit_device here) invalidates ALL unit shards.
    regime2 = r._battery_regime(True, [1, 2, 3, 4], split, "cuda", False, 4)
    r._init_battery_ckpt(tmp_path, regime2)
    assert not p.exists()
    assert json.loads((ck / "regime.json").read_text()) == regime2


def test_battery_regime_keys_descope_flags(monkeypatch):
    """The descope-ladder env flags are part of the 1e regime key (#722 r3)."""
    split = {"train": [1], "val": [2], "test": [3]}
    monkeypatch.delenv("EPM_I952_SKIP_POOLED_PREFIX", raising=False)
    base = r._battery_regime(True, [1, 2, 3], split, "cpu", False, 4)
    monkeypatch.setenv("EPM_I952_SKIP_POOLED_PREFIX", "1")
    changed = r._battery_regime(True, [1, 2, 3], split, "cpu", False, 4)
    assert base != changed
