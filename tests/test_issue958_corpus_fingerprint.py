"""Regression pins for the #958 resume-identity invariants.

r1 BLOCKER `smoke-production-resume-collision`: every resume surface carries
the CORPUS FINGERPRINT identity key — rollout-shard regimes, store shards
(capture validate_shard), and the store index/loaders — so an artifact
generated under one corpus build FAILS LOUD (or fails validation →
regenerates) against a rebuilt corpus, instead of silently pairing stale
generations with new conversations.

r2→r3 CONCERN `fit-resume-dropout-regime-key-missing`: the fit-resume regime
key additionally carries (a) the dropout-set identity (capture-recorded
dropped uids + per-set invalid conversations) and (b) a CONTENT digest of the
consumed store shards — a stale fit manifest under a changed dropout set OR a
regenerated (same-drop-set, different-activations) store must NOT resume, and
a restored per-cell npz must carry the CURRENT design's test_idx. All tests
fail pre-fix and pass post-fix.
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
import issue958_fit_maps as FIT  # noqa: E402
import issue958_rollouts as ROLL  # noqa: E402
import numpy as np  # noqa: E402

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


# ── r3: fit-resume dropout-set + store-content identity ──────────────────────


def _mk_store(tmp_path: Path, payload: bytes = b"A" * 4096) -> Path:
    """A minimal store tree: one raw-bytes shard per consumed unit set."""
    store = tmp_path / "store"
    for s in ("main", "long", "onpol"):
        (store / s).mkdir(parents=True, exist_ok=True)
        (store / s / "shard_000.pt").write_bytes(payload)
    return store


def _mk_regime(store: Path, dropped_main: tuple[str, ...] = (), invalid_main: tuple[int, ...] = ()):
    return FIT.build_fit_regime(
        corpus_fp=FP_A,
        R=3,
        H=4,
        n_main=30,
        n_long=8,
        stub_rb=True,
        invalid_by_set={"main": sorted(invalid_main), "long": []},
        dropped_by_set={"main": set(dropped_main), "long": set(), "onpol": set()},
        store_content_sha=C.store_content_digest(store, ["main", "long", "onpol"]),
    )


def test_store_content_digest_content_sensitive_mtime_invariant(tmp_path: Path):
    """Digest changes on a same-size byte change; mtime alone never changes it."""
    import os

    store = _mk_store(tmp_path)
    d1 = C.store_content_digest(store, ["main", "long", "onpol"])
    # mtime-invariance: identical bytes restaged later (HF re-download) keep the digest
    shard = store / "main" / "shard_000.pt"
    os.utime(shard, (0, 0))
    assert C.store_content_digest(store, ["main", "long", "onpol"]) == d1
    # content sensitivity at IDENTICAL size (a recapture with different fp16
    # activation values keeps torch.save's ZIP_STORED size, changes the bytes)
    mutated = bytearray(shard.read_bytes())
    mutated[len(mutated) // 2] ^= 0xFF
    shard.write_bytes(bytes(mutated))
    d2 = C.store_content_digest(store, ["main", "long", "onpol"])
    assert d2 != d1 and shard.stat().st_size == 4096


def test_fit_resume_refuses_changed_dropout_set(tmp_path: Path):
    """A stale fit manifest under a CHANGED dropout set must NOT resume."""
    store = _mk_store(tmp_path)
    manifest_path = tmp_path / "fit_manifest.json"
    reg_a = _mk_regime(store)
    m = FIT.load_fit_manifest(manifest_path, reg_a)
    m["cells"]["ctx_k1_A"] = {"done": True, "ts": 0.0}
    C.write_json_atomic(manifest_path, m)
    # positive control: identical regime resumes the completed cell
    assert FIT.load_fit_manifest(manifest_path, reg_a)["cells"]["ctx_k1_A"]["done"]
    # dropped-uid delta → different regime → fresh manifest (refit all)
    reg_b = _mk_regime(store, dropped_main=("main:c5:k2",))
    assert reg_b != reg_a
    assert FIT.load_fit_manifest(manifest_path, reg_b)["cells"] == {}
    # invalid-conversation delta (the design-affecting projection) also refuses
    reg_c = _mk_regime(store, invalid_main=(5,))
    assert FIT.load_fit_manifest(manifest_path, reg_c)["cells"] == {}


def test_fit_resume_refuses_regenerated_store(tmp_path: Path):
    """A same-drop-set store RECAPTURE (different bytes) must NOT resume."""
    store = _mk_store(tmp_path)
    manifest_path = tmp_path / "fit_manifest.json"
    reg_a = _mk_regime(store)
    m = FIT.load_fit_manifest(manifest_path, reg_a)
    m["cells"]["ctx_k1_A"] = {"done": True, "ts": 0.0}
    C.write_json_atomic(manifest_path, m)
    # regenerate one shard with different same-size content (vLLM/GPU
    # nondeterminism shape) — the regime's store_content_sha diverges
    shard = store / "onpol" / "shard_000.pt"
    shard.write_bytes(b"B" * 4096)
    reg_b = _mk_regime(store)
    assert reg_b != reg_a
    assert FIT.load_fit_manifest(manifest_path, reg_b)["cells"] == {}


def test_restored_percell_npz_must_match_current_test_idx(tmp_path: Path):
    """The resume predicate's test_idx guard: stale/corrupt npz → refit."""
    p = tmp_path / "xfer_1to2_A.npz"
    np.savez(p, skill=np.zeros(3), test_idx=np.array([0, 1, 2], dtype=np.int64))
    assert FIT._npz_test_idx_matches(p, np.array([0, 1, 2]))
    assert not FIT._npz_test_idx_matches(p, np.array([0, 1, 3]))  # stale fold
    assert not FIT._npz_test_idx_matches(p, np.array([0, 1]))  # shrunk fold
    assert not FIT._npz_test_idx_matches(tmp_path / "missing.npz", np.array([0]))
    (tmp_path / "corrupt.npz").write_bytes(b"not-an-npz")
    assert not FIT._npz_test_idx_matches(tmp_path / "corrupt.npz", np.array([0]))
