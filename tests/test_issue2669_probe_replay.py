"""Range-only archive access must never fetch skipped tensor payloads."""

import io
import json
import tarfile

from scripts.issue2669_probe_replay import index_tar, wanted


class RecordingBytes(io.BytesIO):
    def __init__(self, data):
        super().__init__(data)
        self.size = len(data)
        self.bytes_received = 0
        self.requests_count = 0

    def read(self, n=-1):
        result = super().read(n)
        self.bytes_received += len(result)
        self.requests_count += 1
        return result


def test_header_only_and_pax(tmp_path):
    data = io.BytesIO()
    with tarfile.open(fileobj=data, mode="w", format=tarfile.PAX_FORMAT) as tf:
        for name in ["skip.npy", "prefix_end_L20_shard00.npy", "x" * 150]:
            info = tarfile.TarInfo(name)
            info.size = 1024 * 1024
            tf.addfile(info, io.BytesIO(b"a" * info.size))
    reader = RecordingBytes(data.getvalue())
    path = tmp_path / "index.json"
    result = index_tar(reader, path, "test")
    assert result["complete"] and len(result["members"]) == 3
    assert reader.bytes_received < 10_000
    assert wanted(result["members"][1]["name"], [20])
    assert not wanted("context_end_L21_shard00.npy", [20])
    assert not wanted("answer_aliases.json", [20])
    assert json.loads(path.read_text())["complete"]


def test_parallel_prefetch_preserves_variable_size_archive(tmp_path, monkeypatch):
    from unittest.mock import create_autospec

    import requests

    from scripts.issue2669_probe_replay import HttpRangeReader

    source = io.BytesIO()
    with tarfile.open(fileobj=source, mode="w", format=tarfile.PAX_FORMAT) as tf:
        for i, size in enumerate([4096] * 12 + [731, 12345] + [4096] * 10):
            info = tarfile.TarInfo(f"file{i}.npy")
            info.size = size
            tf.addfile(info, io.BytesIO(b"x" * size))
    payload = source.getvalue()

    def get(session, url, **kwargs):
        low, high = map(int, kwargs["headers"]["Range"].removeprefix("bytes=").split("-"))
        response = requests.Response()
        response.status_code = 206
        response.headers["Content-Range"] = f"bytes {low}-{high}/{len(payload)}"
        response.raw = io.BytesIO(payload[low : high + 1])
        return response

    monkeypatch.setattr(
        requests.Session, "get", create_autospec(requests.Session.get, side_effect=get)
    )
    reader = HttpRangeReader("https://example.invalid/archive", len(payload), "test")
    state = index_tar(reader, tmp_path / "parallel.json", "test")
    assert [r["name"] for r in state["members"]] == [f"file{i}.npy" for i in range(24)]
    assert state["complete"]
    assert reader.bytes_received < len(payload)


def test_resume_index_preserves_absolute_pax_offsets(tmp_path):
    source = io.BytesIO()
    with tarfile.open(fileobj=source, mode="w", format=tarfile.PAX_FORMAT) as tf:
        for i in range(65):
            name = "long/" + ("x" * 150) + str(i) if i % 2 else f"file{i}"
            info = tarfile.TarInfo(name)
            info.size = i * 31 + 17
            tf.addfile(info, io.BytesIO(b"x" * info.size))
    payload = source.getvalue()
    path = tmp_path / "index.json"
    full = index_tar(RecordingBytes(payload), path, "test")
    checkpoint = {**full, "complete": False, "members": full["members"][:50]}
    path.write_text(json.dumps(checkpoint))
    resumed = index_tar(RecordingBytes(payload), path, "test")
    assert resumed["members"] == full["members"]


def test_duplicate_basename_is_rejected_before_payload_read(tmp_path):
    import pytest

    from scripts.issue2669_probe_replay import stage_members

    reader = RecordingBytes(b"not a payload")
    index = {
        "complete": True,
        "revision": "test",
        "members": [
            {"name": "a/t1_L20.npy", "offset": 0, "size": 1},
            {"name": "b/t1_L20.npy", "offset": 1, "size": 1},
        ],
    }
    with pytest.raises(ValueError, match="Duplicate"):
        stage_members(reader, index, tmp_path, [20])
    assert reader.bytes_received == 0
