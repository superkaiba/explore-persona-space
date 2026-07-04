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
a restored per-cell npz must carry the CURRENT design's test_idx.

r3→r4 CONCERNS `fit-resume-store-content-digest-incomplete` +
`fit-resume-store-digest-omits-graft-set`: the store identity is EXACT —
canonical sha over the sidecar's write-time per-shard FULL-file sha256, so a
same-size byte change ANYWHERE (incl. offsets the superseded sampled digest
never read) invalidates resume — and covers ALL FOUR consumed sets incl.
graft (a graft-only recapture invalidates the `_prefix_marginal` cell). A
sidecar without hashes recomputes exactly (never silently trusted); a
sidecar/shard drift asserts. All tests fail pre-fix and pass post-fix.
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
# r4: EXACT sidecar-recorded store identity (`fit-resume-store-content-digest-
# incomplete`) over ALL FOUR consumed sets incl. graft (`fit-resume-store-
# digest-omits-graft-set`).

CONSUMED_SETS = ["main", "long", "onpol", "graft"]
# 4 MiB shard: large enough that the SUPERSEDED sampled digest (first/middle/
# last 1 MiB windows) had un-sampled gaps — the exact hash must not.
SHARD_BYTES = 4 << 20
GAP_OFFSET = 1_200_000  # inside (1 MiB, size/2 - 0.5 MiB): unsampled under the old scheme


def _refresh_sidecar(store: Path, unit_set: str) -> None:
    """Re-write one set's sidecar via the PRODUCTION writer (what capture does)."""
    shards = sorted((store / unit_set).glob("shard_*.pt"))
    C.write_store_sidecar(
        store,
        unit_set,
        corpus_fingerprint=FP_A,
        n_rows=3,
        hidden=4,
        shards={str(i): [f"{unit_set}:c{i}:k1"] for i in range(len(shards))},
    )


def _mk_store(tmp_path: Path, payload: bytes | None = None) -> Path:
    """A minimal store tree: one raw-bytes shard + production sidecar per set."""
    store = tmp_path / "store"
    for s in CONSUMED_SETS:
        (store / s).mkdir(parents=True, exist_ok=True)
        (store / s / "shard_000.pt").write_bytes(payload or (b"A" * SHARD_BYTES))
        _refresh_sidecar(store, s)
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
        dropped_by_set={"main": set(dropped_main), "long": set(), "onpol": set(), "graft": set()},
        store_content_sha=C.store_content_digest(store, CONSUMED_SETS),
    )


def _flip_byte(shard: Path, offset: int = GAP_OFFSET) -> None:
    """Same-size in-place byte flip (the vLLM/GPU-nondeterminism recapture shape)."""
    size = shard.stat().st_size
    mutated = bytearray(shard.read_bytes())
    mutated[offset] ^= 0xFF
    shard.write_bytes(bytes(mutated))
    assert shard.stat().st_size == size


def test_store_content_digest_exact_same_size_change_mtime_invariant(tmp_path: Path):
    """Any same-size byte change invalidates once the sidecar is refreshed; mtime never does."""
    import os

    store = _mk_store(tmp_path)
    d1 = C.store_content_digest(store, CONSUMED_SETS)
    # mtime-invariance: identical bytes restaged later (HF re-download) keep the digest
    shard = store / "main" / "shard_000.pt"
    os.utime(shard, (0, 0))
    assert C.store_content_digest(store, CONSUMED_SETS) == d1
    # same-size byte flip at an offset the OLD sampled digest never read, then
    # sidecar refresh (a recapture ALWAYS rewrites the sidecar hashes at write
    # time) — the exact digest diverges
    _flip_byte(shard)
    _refresh_sidecar(store, "main")
    d2 = C.store_content_digest(store, CONSUMED_SETS)
    assert d2 != d1


def test_store_content_digest_legacy_sidecar_recomputes_exact(tmp_path: Path):
    """A sidecar WITHOUT shard_sha256 (legacy store) is never silently trusted."""
    store = _mk_store(tmp_path)
    # strip the recorded hashes from one set's sidecar (legacy pre-r4 shape)
    sidecar = C.store_index_path(store, "main")
    blob = json.loads(sidecar.read_text())
    del blob["shard_sha256"]
    C.write_json_atomic(sidecar, blob)
    d1 = C.store_content_digest(store, CONSUMED_SETS)
    # a same-size gap-offset byte flip with NO sidecar refresh still changes
    # the digest — the legacy path recomputes the exact hash from shard bytes
    _flip_byte(store / "main" / "shard_000.pt")
    assert C.store_content_digest(store, CONSUMED_SETS) != d1


def test_store_content_digest_asserts_on_sidecar_shard_drift(tmp_path: Path):
    """An on-disk shard the sidecar's shard_sha256 does not cover fails loud."""
    store = _mk_store(tmp_path)
    (store / "main" / "shard_001.pt").write_bytes(b"B" * 64)  # uncovered by sidecar
    with pytest.raises(AssertionError, match="sidecar/shard drift"):
        C.store_content_digest(store, CONSUMED_SETS)


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
    """A same-drop-set store RECAPTURE (same-size different bytes) must NOT resume."""
    store = _mk_store(tmp_path)
    manifest_path = tmp_path / "fit_manifest.json"
    reg_a = _mk_regime(store)
    m = FIT.load_fit_manifest(manifest_path, reg_a)
    m["cells"]["ctx_k1_A"] = {"done": True, "ts": 0.0}
    C.write_json_atomic(manifest_path, m)
    # recapture one shard: same-size byte change at an offset the superseded
    # sampled digest never read (vLLM/GPU-nondeterminism shape) + the
    # capture-side sidecar rewrite — the regime's store_content_sha diverges
    _flip_byte(store / "onpol" / "shard_000.pt")
    _refresh_sidecar(store, "onpol")
    reg_b = _mk_regime(store)
    assert reg_b != reg_a
    assert FIT.load_fit_manifest(manifest_path, reg_b)["cells"] == {}


def test_fit_resume_refuses_graft_only_recapture(tmp_path: Path):
    """A graft-ONLY recapture (same drop set) must invalidate `_prefix_marginal`.

    r4 (`fit-resume-store-digest-omits-graft-set`): the manifest-resumable
    `_prefix_marginal` cell consumes graft shard CONTENT, so the fit regime's
    store identity must cover the graft set — a graft-only recapture with the
    dropout identity held EQUAL refits instead of restoring a stale
    prefix_marginal.json.
    """
    store = _mk_store(tmp_path)
    manifest_path = tmp_path / "fit_manifest.json"
    reg_a = _mk_regime(store)
    m = FIT.load_fit_manifest(manifest_path, reg_a)
    m["cells"]["_prefix_marginal"] = {"done": True, "ts": 0.0}
    C.write_json_atomic(manifest_path, m)
    # positive control: identical regime resumes the marginal cell
    assert FIT.load_fit_manifest(manifest_path, reg_a)["cells"]["_prefix_marginal"]["done"]
    # graft-only recapture: same-size byte change + sidecar rewrite in the
    # graft set ONLY; main/long/onpol bytes and the dropout set unchanged
    _flip_byte(store / "graft" / "shard_000.pt")
    _refresh_sidecar(store, "graft")
    reg_b = _mk_regime(store)
    assert reg_b["dropped_uids_sha"] == reg_a["dropped_uids_sha"]  # dropout identity EQUAL
    assert reg_b != reg_a  # ONLY the store content identity diverges
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
