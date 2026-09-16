"""Integrity and selective extraction checks for the saved-input staging path."""

import hashlib
import io
import tarfile

import pytest

from scripts.issue1739_covariance_stage import Progress, extract_members, selected_name


def archive_bytes(members):
    """Construct a real tar stream for tests of the production extraction body."""
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w") as archive:
        for name, content in members:
            info = tarfile.TarInfo(name)
            info.size = len(content)
            archive.addfile(info, io.BytesIO(content))
    buffer.seek(0)
    return buffer


def record(content):
    """Return the same byte-count and digest contract as the remote inventory."""
    return {"bytes": len(content), "sha256": hashlib.sha256(content).hexdigest()}


def test_exact_layer_selection():
    assert selected_name("context_end_L18_shard03.npy", (18, 20), ("context_end", "t1"))
    assert selected_name("t1_L20.npy", (18, 20), ("context_end", "t1"))
    assert selected_name("row_index_shard00.jsonl", (18, 20), ("context_end", "t1"))
    assert not selected_name("context_end_L19_shard03.npy", (18, 20), ("context_end", "t1"))
    assert not selected_name("prefix_end_L18_shard03.npy", (18, 20), ("context_end", "t1"))


def test_extract_checks_hashes_and_resume(tmp_path):
    content = b"saved activation bytes"
    name = "context_end_L18_shard00.npy"
    expected = {name: record(content)}
    progress = Progress(tmp_path / "progress.json")
    try:
        for _ in range(2):
            buffer = archive_bytes([("unwanted.npy", b"omit"), (f"root/{name}", content)])
            with tarfile.open(fileobj=buffer, mode="r|") as archive:
                assert extract_members(archive, expected, tmp_path, progress) == 1
        assert (tmp_path / name).read_bytes() == content
        assert not (tmp_path / "unwanted.npy").exists()
        assert not list(tmp_path.glob("*.partial"))
    finally:
        progress.close()


def test_corrupt_member_is_never_published(tmp_path):
    name = "t1_L20_shard00.npy"
    progress = Progress(tmp_path / "progress.json")
    try:
        with (
            tarfile.open(fileobj=archive_bytes([(name, b"bad")]), mode="r|") as archive,
            pytest.raises(ValueError, match="content mismatch"),
        ):
            extract_members(archive, {name: record(b"yes")}, tmp_path, progress)
        assert not (tmp_path / name).exists()
    finally:
        progress.close()


def test_incomplete_archive_fails(tmp_path):
    progress = Progress(tmp_path / "progress.json")
    try:
        with (
            tarfile.open(fileobj=archive_bytes([]), mode="r|") as archive,
            pytest.raises(ValueError, match="required members"),
        ):
            extract_members(archive, {"t1_L20.npy": record(b"data")}, tmp_path, progress)
    finally:
        progress.close()
