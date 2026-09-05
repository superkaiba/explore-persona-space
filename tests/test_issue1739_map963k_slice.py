from __future__ import annotations

import io
import shutil
import tarfile
from types import SimpleNamespace

import numpy as np
import pytest

from scripts import issue1739_map963k_slice as slice_mod


class _MemoryRangeReader(io.RawIOBase):
    def __init__(self, payload: bytes):
        self._source = io.BytesIO(payload)
        self.bytes_fetched = len(payload)

    def readable(self) -> bool:
        return True

    def readinto(self, buffer) -> int:
        return self._source.readinto(buffer)


def _member(tar: tarfile.TarFile, name: str, payload: bytes) -> None:
    info = tarfile.TarInfo(f"capture/{name}")
    info.size = len(payload)
    tar.addfile(info, io.BytesIO(payload))


def _npy_bytes(values: list[float]) -> bytes:
    buf = io.BytesIO()
    np.save(buf, np.asarray(values, dtype=np.float32), allow_pickle=False)
    return buf.getvalue()


def test_materialized_slice_downloads_extracts_and_reaps(tmp_path, monkeypatch):
    source = tmp_path / "source.tar"
    wanted = _npy_bytes([1.0, 2.0])
    with tarfile.open(source, "w") as tar:
        _member(tar, "context_end_L07_shard0.npy", wanted)
        _member(tar, "prefix_end_L08_shard0.npy", _npy_bytes([3.0]))
        _member(tar, "row_index.jsonl", b'{"row": 0}\n')

    downloads = []

    def fake_download(behavior, revision, local_tar, *, token):
        downloads.append((behavior, revision, local_tar, token))
        shutil.copy2(source, local_tar)
        return local_tar

    monkeypatch.setattr(slice_mod, "_download_tar", fake_download)
    monkeypatch.setattr(slice_mod, "head_size", lambda url, token: source.stat().st_size)
    monkeypatch.setattr(
        slice_mod.shutil,
        "disk_usage",
        lambda path: SimpleNamespace(free=10 << 30),
    )
    dest = tmp_path / "store" / "evil_labeling"
    manifest = slice_mod.stream_slice(
        "evil",
        dest,
        revision="deadbeef",
        kinds=("context_end",),
        layers=(7,),
        token="secret",
        materialize=True,
    )

    assert (dest / "context_end_L07_shard0.npy").read_bytes() == wanted
    assert not (dest / "prefix_end_L08_shard0.npy").exists()
    assert (dest / "row_index.jsonl").read_bytes() == b'{"row": 0}\n'
    assert manifest["transfer_mode"] == "materialized"
    assert manifest["n_kept"] == 2
    assert manifest["n_skipped"] == 1
    assert manifest["bytes_fetched"] == source.stat().st_size
    assert downloads == [
        (
            "evil",
            "deadbeef",
            dest.parent / "labeling_tars" / "evil_labeling.tar",
            "secret",
        )
    ]
    assert not downloads[0][2].exists()


def test_materialized_slice_refuses_insufficient_peak_headroom(tmp_path, monkeypatch):
    monkeypatch.setattr(slice_mod, "head_size", lambda url, token: 4 << 30)
    monkeypatch.setattr(
        slice_mod.shutil,
        "disk_usage",
        lambda path: SimpleNamespace(free=12 << 30),
    )
    monkeypatch.setattr(
        slice_mod,
        "_download_tar",
        lambda *args, **kwargs: pytest.fail("download must not start without headroom"),
    )

    with pytest.raises(RuntimeError, match="lacks peak headroom"):
        slice_mod.stream_slice(
            "hallucination",
            tmp_path / "hallucination_labeling",
            revision="main",
            kinds=("context_end",),
            layers=(7,),
            token="secret",
            materialize=True,
        )


def test_default_range_stream_contract_survives_shared_extractor(tmp_path, monkeypatch):
    source = tmp_path / "source.tar"
    wanted = _npy_bytes([4.0, 5.0])
    with tarfile.open(source, "w") as tar:
        _member(tar, "context_end_L07_shard0.npy", wanted)
        _member(tar, "context_end_L08_shard0.npy", _npy_bytes([6.0]))
        _member(tar, "row_index.jsonl", b'{"row": 1}\n')
    payload = source.read_bytes()

    monkeypatch.setattr(slice_mod, "head_size", lambda url, token: len(payload))
    monkeypatch.setattr(
        slice_mod,
        "ParallelRangeReader",
        lambda *args, **kwargs: _MemoryRangeReader(payload),
    )
    dest = tmp_path / "streamed"
    manifest = slice_mod.stream_slice(
        "evil",
        dest,
        revision="deadbeef",
        kinds=("context_end",),
        layers=(7,),
        token="secret",
    )

    assert (dest / "context_end_L07_shard0.npy").read_bytes() == wanted
    assert not (dest / "context_end_L08_shard0.npy").exists()
    assert (dest / "row_index.jsonl").read_bytes() == b'{"row": 1}\n'
    assert manifest["transfer_mode"] == "range_stream"
    assert manifest["n_kept"] == 2
    assert manifest["n_skipped"] == 1
