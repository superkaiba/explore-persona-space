"""#2220 throughput fix — materialize-then-read path of issue1739_natpv.stream_members.

The chunk-1 ``materialize_directions`` phase read three monolithic HF labeling
tars via raw HTTP range GETs (``ParallelRangeReader``) at ~1 MB/s aggregate on
pod-2220 (~43 h projected for 154 GB); a whole-tar ``hf_hub_download`` with
``HF_HUB_ENABLE_HF_TRANSFER=1`` measured ~499 MB/s. The fix adds an OPT-IN
materialize branch to ``stream_members`` gated by two module flags.

Pinned here:
  (a) With ``MATERIALIZE_TARS=True`` and the download monkeypatched to a tiny
      synthetic tar, ``stream_members`` yields the SAME ``(name,
      ndarray-or-bytes)`` tuples as the streaming contract (``want`` filter,
      ``np.load`` for ``.npy``, raw bytes otherwise), removes any pre-existing
      tar at the target first, and reaps the materialized tar in a ``finally``.
  (b) With ``MATERIALIZE_TARS=False`` (the module DEFAULT), the streaming
      branch is selected — no download call — and yields the identical tuples
      from the same tar bytes (byte-identical default-path contract).

No network, no live HF fetch; repo-root-relative paths only (sparse-worktree
and fleet-Step-9c safe).
"""

from __future__ import annotations

import importlib
import io
import logging
import shutil
import sys
import tarfile
import types
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture()
def natpv():
    """Import scripts.issue1739_natpv; restore the module flags after the test."""
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    mod = importlib.import_module("scripts.issue1739_natpv")
    orig = (mod.MATERIALIZE_TARS, mod.MATERIALIZE_STAGING_DIR)
    yield mod
    mod.MATERIALIZE_TARS, mod.MATERIALIZE_STAGING_DIR = orig


def _add_member(tar: tarfile.TarFile, name: str, data: bytes) -> None:
    info = tarfile.TarInfo(name=f"store/{name}")
    info.size = len(data)
    tar.addfile(info, io.BytesIO(data))


def _npy_bytes(arr: np.ndarray) -> bytes:
    buf = io.BytesIO()
    np.save(buf, arr, allow_pickle=False)
    return buf.getvalue()


def _build_synthetic_tar(path: Path) -> dict[str, np.ndarray]:
    """Tiny tar: 2 matching .npy shards, 1 matching .jsonl, 1 NON-matching .npy."""
    rng = np.random.default_rng(2220)
    arrays = {
        "context_end_L07_shard0.npy": rng.standard_normal((3, 4)).astype(np.float32),
        "context_end_L07_shard1.npy": rng.standard_normal((2, 4)).astype(np.float32),
    }
    with tarfile.open(path, "w") as tar:
        _add_member(
            tar, "context_end_L07_shard0.npy", _npy_bytes(arrays["context_end_L07_shard0.npy"])
        )
        # NON-matching member interleaved (wrong layer for the want regex below)
        _add_member(tar, "prefix_end_L99_shard0.npy", _npy_bytes(np.zeros((1, 4), np.float32)))
        _add_member(
            tar, "context_end_L07_shard1.npy", _npy_bytes(arrays["context_end_L07_shard1.npy"])
        )
        _add_member(tar, "row_index.jsonl", b'{"context_id": "c0", "rollout_k": 0}\n')
    return arrays


def _want(natpv):
    """The realistic filter shape: L07 context_end shards + row_index members."""
    m = natpv._slice_mod()
    return m.wanted_re(("context_end",), (7,))


EXPECTED_NAMES = [
    "context_end_L07_shard0.npy",
    "context_end_L07_shard1.npy",
    "row_index.jsonl",
]


def _assert_contract(got, arrays):
    assert [n for n, _ in got] == EXPECTED_NAMES
    np.testing.assert_array_equal(got[0][1], arrays["context_end_L07_shard0.npy"])
    np.testing.assert_array_equal(got[1][1], arrays["context_end_L07_shard1.npy"])
    assert isinstance(got[2][1], bytes) and got[2][1].startswith(b'{"context_id"')


def test_materialize_path_yields_streaming_contract(natpv, tmp_path, monkeypatch, caplog):
    src_tar = tmp_path / "src" / "evil_labeling.tar"
    src_tar.parent.mkdir()
    arrays = _build_synthetic_tar(src_tar)
    staging = tmp_path / "staging"

    calls: list[tuple[str, str, Path]] = []

    def fake_download(behavior: str, revision: str, local_tar: Path) -> Path:
        # pre-existing tars at the target must have been removed BEFORE download
        assert not Path(local_tar).exists()
        calls.append((behavior, revision, Path(local_tar)))
        shutil.copy2(src_tar, local_tar)
        return Path(local_tar)

    monkeypatch.setattr(natpv, "_download_tar", fake_download)
    monkeypatch.setattr(natpv, "MATERIALIZE_TARS", True)
    monkeypatch.setattr(natpv, "MATERIALIZE_STAGING_DIR", staging)

    # plant a stale pre-existing tar at the target (idempotency clause)
    staging.mkdir(parents=True)
    (staging / "evil_labeling.tar").write_bytes(b"stale garbage")

    with caplog.at_level(logging.INFO, logger="issue1739_natpv"):
        got = list(
            natpv.stream_members("evil", "deadbeef", workers=6, window_mib=64, want=_want(natpv))
        )

    _assert_contract(got, arrays)
    assert calls == [("evil", "deadbeef", staging / "evil_labeling.tar")]
    # per-call reap: the materialized tar is deleted in the finally
    assert not (staging / "evil_labeling.tar").exists()
    # FIX-ENGAGED log substring (the #2220 relaunch probe keys on it)
    assert "MATERIALIZE path engaged" in caplog.text


class _BytesRangeReader(io.RawIOBase):
    """Stand-in for ParallelRangeReader serving fixed tar bytes (no network)."""

    def __init__(self, data: bytes):
        super().__init__()
        self._buf = io.BytesIO(data)

    def readable(self) -> bool:
        return True

    def readinto(self, b) -> int:
        return self._buf.readinto(b)


def test_default_path_streams_without_download(natpv, tmp_path, monkeypatch):
    # DEFAULT must be the streaming branch — issue1739's own phases unaffected.
    # Pinned at SOURCE level (order-independent): issue2220's
    # phase_materialize_directions deliberately flips the process-global flag,
    # so another test having entered that phase must not fail this pin.
    import inspect
    import re

    src = inspect.getsource(natpv)
    assert re.search(r"^MATERIALIZE_TARS: bool = False$", src, re.M)
    assert re.search(r"^MATERIALIZE_STAGING_DIR: Path \| None = None$", src, re.M)
    monkeypatch.setattr(natpv, "MATERIALIZE_TARS", False)
    monkeypatch.setattr(natpv, "MATERIALIZE_STAGING_DIR", None)

    src_tar = tmp_path / "evil_labeling.tar"
    arrays = _build_synthetic_tar(src_tar)
    tar_bytes = src_tar.read_bytes()

    def boom(*a, **k):
        raise AssertionError("download must not be called on the streaming branch")

    monkeypatch.setattr(natpv, "_download_tar", boom)
    monkeypatch.setattr(natpv, "_materialized_members", boom)
    monkeypatch.setenv("HF_TOKEN", "dummy-token")

    stub = types.SimpleNamespace(
        tar_url=lambda behavior, revision: f"https://example.invalid/{behavior}/{revision}",
        head_size=lambda url, token: len(tar_bytes),
        ParallelRangeReader=lambda url, token, total, window, workers: _BytesRangeReader(tar_bytes),
    )
    monkeypatch.setattr(natpv, "_slice_mod", lambda: stub)
    # want regex from the REAL slice module (the stub replaces only the reader)
    real_m = importlib.import_module("scripts.issue1739_map963k_slice")
    want = real_m.wanted_re(("context_end",), (7,))

    got = list(natpv.stream_members("evil", "deadbeef", workers=2, window_mib=1, want=want))
    _assert_contract(got, arrays)
