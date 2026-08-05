"""#1491 greedy replication round: decoding-regime identity invariants.

The greedy round re-runs the six-rung ladder at temperature 0. The single
load-bearing hazard is CROSS-REGIME REUSE: the driver's resume predicate skips
chunks by FILENAME (never reading a payload) and its salvage path re-uploads
local raw chunks verbatim, so without a regime key a greedy run could silently
resume from / salvage / join temperature-1.0 chunks — corrupting the
comparison invisibly (the per-row prompt assert checks prompts, not
responses). These tests pin the permanent invariants added by the greedy
round (regression tests for the round's substantive fix — they fail on the
pre-round code by construction, where none of these seams existed):

1. ``_resolve_sampling`` default == the parent recipe constants (byte-identical
   default), greedy == temperature 0.
2. ``_assert_raw_payload_matches`` refuses a cross-regime payload, including
   LEGACY payloads with no ``sampling_mode`` key (uniquely parent-mode by
   construction — the pre-greedy driver had hardcoded constants only).
3. ``_enforce_sampling_identity`` refuses cross-regime scratch/prefix reuse at
   the prefix grain (local marker, local legacy chunks, hub marker, hub
   legacy chunks) and writes markers on the fresh-prefix path.

Offline by construction: the hub boundary is monkeypatched with
signature-conformant fakes (module-level ``_download_hub_sampling_marker`` /
``_upload_sampling_marker``); everything else executes the real bodies
(tests/ runs in every issue's Step 9c gate — no live Hub fetch allowed).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import issue1491_ladder_generate_capture as D  # noqa: E402

PARENT = D.SAMPLING_MODE_PARENT
GREEDY = D.SAMPLING_MODE_GREEDY


# ---------------------------------------------------------------------------
# 1. _resolve_sampling
# ---------------------------------------------------------------------------


def test_resolve_sampling_default_is_parent_recipe():
    """Flag absent == byte-identical parent recipe (GEN_TEMP / GEN_TOP_P)."""
    s = D._resolve_sampling(False)
    assert s["mode"] == PARENT
    assert s["temperature"] == D.GEN_TEMP == 1.0
    assert s["top_p"] == D.GEN_TOP_P == 0.95


def test_resolve_sampling_greedy_is_temp0():
    s = D._resolve_sampling(True)
    assert s["mode"] == GREEDY
    assert s["temperature"] == 0.0
    assert s["top_p"] == 1.0
    assert s["mode"] != PARENT


# ---------------------------------------------------------------------------
# 2. payload-level wave-alignment identity
# ---------------------------------------------------------------------------


def _payload(**over):
    p = {
        "shard_index": 0,
        "chunk": 0,
        "split": "val_400",
        "seed": 42,
        "sampling_mode": PARENT,
        "rows": [],
    }
    p.update(over)
    return p


def _assert_payload(payload, expect_mode):
    D._assert_raw_payload_matches(
        payload,
        "shard00_chunk0000.json",
        expect_split="val_400",
        expect_seed=42,
        expect_shard_index=0,
        expect_chunk=0,
        expect_sampling_mode=expect_mode,
    )


def test_payload_same_mode_passes():
    _assert_payload(_payload(sampling_mode=GREEDY), GREEDY)
    _assert_payload(_payload(sampling_mode=PARENT), PARENT)


def test_payload_cross_mode_refuses():
    with pytest.raises(AssertionError, match="SAMPLING-MODE mismatch"):
        _assert_payload(_payload(sampling_mode=PARENT), GREEDY)
    with pytest.raises(AssertionError, match="SAMPLING-MODE mismatch"):
        _assert_payload(_payload(sampling_mode=GREEDY), PARENT)


def test_payload_legacy_missing_key_is_parent_mode():
    """A pre-greedy-round chunk has NO sampling_mode key: it was written under
    the hardcoded parent constants, so it must pass a parent-mode run
    (byte-identical default behavior) and refuse a greedy run."""
    legacy = _payload()
    del legacy["sampling_mode"]
    _assert_payload(legacy, PARENT)  # default path keeps resuming legacy chunks
    with pytest.raises(AssertionError, match="SAMPLING-MODE mismatch"):
        _assert_payload(legacy, GREEDY)


# ---------------------------------------------------------------------------
# 3. prefix-grain identity guard
# ---------------------------------------------------------------------------


def _enforce(scratch, sampling, *, done_pt=(), done_raw=(), no_upload=True, shard_index=0):
    D._enforce_sampling_identity(
        "issue1491_test/prefix/val_400",
        scratch,
        scratch / ".cache",
        sampling,
        done_pt=set(done_pt),
        done_raw=set(done_raw),
        no_upload=no_upload,
        shard_index=shard_index,
    )


def test_guard_fresh_scratch_writes_local_marker(tmp_path):
    scratch = tmp_path / "s"
    scratch.mkdir()
    _enforce(scratch, D._resolve_sampling(True))
    marker = json.loads((scratch / D.SAMPLING_MARKER_NAME).read_text())
    assert marker["sampling_mode"] == GREEDY
    assert marker["temperature"] == 0.0
    # Idempotent re-entry (same mode) passes.
    _enforce(scratch, D._resolve_sampling(True))


def test_guard_local_marker_mismatch_refuses(tmp_path):
    scratch = tmp_path / "s"
    scratch.mkdir()
    _enforce(scratch, D._resolve_sampling(False))  # parent-mode run wrote the marker
    with pytest.raises(RuntimeError, match="SAMPLING-MODE mismatch \\(local scratch\\)"):
        _enforce(scratch, D._resolve_sampling(True))


def test_guard_legacy_local_chunks_refuse_greedy(tmp_path):
    """Chunk files present with NO marker predate the greedy round — uniquely
    temperature-1.0, so a greedy run refuses; a parent-mode run proceeds
    (byte-identical default) and backfills the marker."""
    scratch = tmp_path / "s"
    scratch.mkdir()
    (scratch / "shard00_chunk0000.json").write_text("{}")
    with pytest.raises(RuntimeError, match="legacy local scratch"):
        _enforce(scratch, D._resolve_sampling(True))
    _enforce(scratch, D._resolve_sampling(False))
    assert (scratch / D.SAMPLING_MARKER_NAME).exists()


def _sig_download(stage_prefix: str, cache_dir: Path) -> dict | None:
    raise NotImplementedError  # replaced per-test; signature mirrors the real helper


def test_guard_hub_marker_mismatch_refuses(tmp_path, monkeypatch):
    scratch = tmp_path / "s"
    scratch.mkdir()

    def fake_download(stage_prefix: str, cache_dir: Path) -> dict | None:
        return {"sampling_mode": PARENT, "gen_max_tokens": D.GEN_MAX_TOKENS}

    monkeypatch.setattr(D, "_download_hub_sampling_marker", fake_download)
    with pytest.raises(RuntimeError, match="SAMPLING-MODE mismatch \\(Hub\\)"):
        _enforce(scratch, D._resolve_sampling(True), no_upload=False)


def test_guard_hub_legacy_chunks_refuse_greedy(tmp_path, monkeypatch):
    """Chunks on the Hub with NO marker == the parent's published prefix: a
    greedy run must refuse even though every filename-level resume check
    would happily 'skip' them."""
    scratch = tmp_path / "s"
    scratch.mkdir()

    def fake_download(stage_prefix: str, cache_dir: Path) -> dict | None:
        return None

    monkeypatch.setattr(D, "_download_hub_sampling_marker", fake_download)
    with pytest.raises(RuntimeError, match="legacy Hub prefix"):
        _enforce(
            scratch,
            D._resolve_sampling(True),
            done_raw={"shard00_chunk0000.json"},
            no_upload=False,
        )


def test_guard_fresh_hub_prefix_uploads_marker_from_shard0_only(tmp_path, monkeypatch):
    scratch = tmp_path / "s"
    scratch.mkdir()
    uploads: list[tuple[str, dict]] = []

    def fake_download(stage_prefix: str, cache_dir: Path) -> dict | None:
        return None

    def fake_upload(stage_prefix: str, payload: dict) -> None:
        uploads.append((stage_prefix, payload))

    monkeypatch.setattr(D, "_download_hub_sampling_marker", fake_download)
    monkeypatch.setattr(D, "_upload_sampling_marker", fake_upload)
    _enforce(scratch, D._resolve_sampling(True), no_upload=False, shard_index=0)
    assert len(uploads) == 1 and uploads[0][1]["sampling_mode"] == GREEDY
    # Non-zero shards never upload (single-writer; avoids the 8-way commit race).
    scratch2 = tmp_path / "s2"
    scratch2.mkdir()
    _enforce(scratch2, D._resolve_sampling(True), no_upload=False, shard_index=3)
    assert len(uploads) == 1
