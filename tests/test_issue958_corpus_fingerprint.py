"""Regression pin for the #958 r1 BLOCKER `smoke-production-resume-collision`.

The permanent invariant: every resume surface carries the CORPUS FINGERPRINT
identity key — rollout-shard regimes, store shards (capture validate_shard),
and the store index/loaders — so an artifact generated under one corpus build
FAILS LOUD (or fails validation → regenerates) against a rebuilt corpus,
instead of silently pairing stale generations with new conversations. These
tests fail pre-fix (no fingerprint existed anywhere) and pass post-fix.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
import torch

REPO = Path(__file__).resolve().parent.parent
for _p in (REPO / "src", REPO / "scripts"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import issue958_capture_turns as CAPT  # noqa: E402
import issue958_common as C  # noqa: E402
import issue958_rollouts as ROLL  # noqa: E402

FP_A = C.compute_corpus_fingerprint({"n_main": 200, "seed": 42, "hashes": ["a", "b"]})
FP_B = C.compute_corpus_fingerprint({"n_main": 5000, "seed": 42, "hashes": ["c", "d"]})


def test_fingerprint_distinct_and_manifest_fail_loud(tmp_path: Path):
    """Different corpus builds → different fingerprints; missing field fails."""
    assert FP_A != FP_B
    (tmp_path / "manifest.json").write_text(json.dumps({"n_main": 200}))
    with pytest.raises(AssertionError, match="corpus_fingerprint"):
        C.corpus_fingerprint(tmp_path)
    (tmp_path / "manifest.json").write_text(json.dumps({"n_main": 200, "corpus_fingerprint": FP_A}))
    assert C.corpus_fingerprint(tmp_path) == FP_A


def test_rollout_shard_rejects_other_corpus_fingerprint(tmp_path: Path):
    """A rollout shard from corpus A fails validation under corpus B's regime."""
    regime_a = {"model": "m", "seed": 42, "corpus_fingerprint": FP_A}
    regime_b = {**regime_a, "corpus_fingerprint": FP_B}
    shard = tmp_path / "rollouts_main_000.json"
    shard.write_text(
        json.dumps(
            {
                "unit_set": "main",
                "regime": regime_a,
                "rollouts": {"main:c0:k1": {"text": "x", "finish_reason": "stop"}},
                "dropped_empty": ["main:c1:k1"],
            }
        )
    )
    uids = {"main:c0:k1", "main:c1:k1"}
    assert ROLL._validate_shard(shard, uids, regime_a)  # dropped counts toward coverage
    assert not ROLL._validate_shard(shard, uids, regime_b)  # stale corpus → regenerate


def test_store_shard_rejects_other_corpus_fingerprint(tmp_path: Path):
    """A store shard captured under corpus A fails validate_shard under B."""
    h = torch.zeros((5, 3, 4), dtype=torch.float16)
    path = tmp_path / "shard_000.pt"
    torch.save(
        {"unit_set": "main", "corpus_fingerprint": FP_A, "units": {"main:c0:k1": {"h": h}}},
        path,
    )
    blob, why = CAPT.validate_shard(path, {"main:c0:k1"}, 3, 4, FP_A)
    assert blob is not None and why == "ok"
    blob_b, why_b = CAPT.validate_shard(path, {"main:c0:k1"}, 3, 4, FP_B)
    assert blob_b is None and "fingerprint" in why_b


def test_store_loaders_reject_other_corpus_fingerprint(tmp_path: Path):
    """load_store_index / load_store_positions fail loud on a stale store."""
    h = torch.zeros((5, 3, 4), dtype=torch.float16)
    (tmp_path / "main").mkdir(parents=True)
    torch.save(
        {"unit_set": "main", "corpus_fingerprint": FP_A, "units": {"main:c0:k1": {"h": h}}},
        C.store_shard_path(tmp_path, "main", 0),
    )
    idx = C.load_store_index(tmp_path, "main", expect_fingerprint=FP_A)
    assert "main:c0:k1" in idx
    with pytest.raises(AssertionError, match="FINGERPRINT MISMATCH"):
        C.load_store_index(tmp_path, "main", expect_fingerprint=FP_B)
    # sidecar path: the index.json fingerprint is checked the same way
    C.write_json_atomic(
        C.store_index_path(tmp_path, "main"),
        {
            "unit_set": "main",
            "corpus_fingerprint": FP_A,
            "n_rows": 3,
            "hidden": 4,
            "shards": {"0": ["main:c0:k1"]},
        },
    )
    x = C.load_store_positions(tmp_path, "main", ["main:c0:k1"], [0], expect_fingerprint=FP_A)
    assert tuple(x.shape) == (1, 1, 3, 4)
    with pytest.raises(AssertionError, match="FINGERPRINT MISMATCH"):
        C.load_store_positions(tmp_path, "main", ["main:c0:k1"], [0], expect_fingerprint=FP_B)
