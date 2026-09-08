import json

import pytest

from scripts import issue952_china_repair_persist as pack


def setup_source(tmp_path):
    source, archive = tmp_path / "source", tmp_path / "archive"
    (source / "packets").mkdir(parents=True)
    (source / "packets" / "a.json").write_bytes(b'{ "odd" :  1 }\r\n')
    (source / "scores.jsonl").write_bytes('中文\n"\\\t😃\n'.encode())
    (source / "empty.txt").write_bytes(b"")
    return source, archive


def test_byte_exact_roundtrip_and_resume(tmp_path, monkeypatch):
    source, archive = setup_source(tmp_path)
    monkeypatch.setattr(pack, "CHUNK_CHARS", 3)
    monkeypatch.setattr(pack, "SHARD_BYTES", 350)
    manifest = pack.pack_tree(source, archive)
    assert len(manifest["shards"]) > 1
    assert max(x["bytes"] for x in manifest["shards"].values()) <= 350
    target = tmp_path / "restored"
    pack.unpack_tree(archive, target)
    assert pack.pack_tree(source, archive) == manifest
    pack.unpack_tree(archive, target)
    for path in source.rglob("*"):
        if path.is_file():
            assert path.read_bytes() == (target / path.relative_to(source)).read_bytes()


def test_corruption_and_extra_missing_shards(tmp_path):
    source, archive = setup_source(tmp_path)
    pack.pack_tree(source, archive)
    shard = archive / "archive.part0000.jsonl"
    original = shard.read_bytes()
    shard.write_bytes(original + b"x")
    with pytest.raises(ValueError, match="hash/length"):
        pack.verify_archive(archive)
    shard.write_bytes(original)
    shard.rename(archive / "archive.part9999.jsonl")
    with pytest.raises(ValueError, match="census"):
        pack.verify_archive(archive)


def test_restore_refuses_changed_existing_and_source_symlink(tmp_path):
    source, archive = setup_source(tmp_path)
    pack.pack_tree(source, archive)
    target = tmp_path / "restored"
    pack.unpack_tree(archive, target)
    (target / "empty.txt").write_text("changed")
    with pytest.raises(ValueError, match="overwrite"):
        pack.unpack_tree(archive, target)
    (source / "link").symlink_to(source / "empty.txt")
    with pytest.raises(ValueError, match="symlink"):
        pack.pack_tree(source, tmp_path / "other")


@pytest.mark.parametrize("name", ["/tmp/a", "../a", "a/../../b", "a//b", "", "./a"])
def test_path_traversal_rejected(name):
    with pytest.raises(ValueError, match="path"):
        pack.safe_relative(name)


def test_selected_subtree_and_explicit_bytecode_exclusion(tmp_path):
    source, archive = setup_source(tmp_path)
    cache = source / "packets" / "__pycache__"
    cache.mkdir()
    (cache / "x.pyc").write_bytes(b"\x80\xff")
    manifest = pack.pack_tree(source, archive, ["packets"])
    assert list(manifest["files"]) == ["packets/a.json"]
    assert manifest["excluded"][0]["reason"] == "regenerable Python bytecode"
    with pytest.raises(ValueError, match="outside"):
        pack.pack_tree(source, source / "bad")


def test_source_hash_corruption_detected_before_restore(tmp_path):
    source, archive = setup_source(tmp_path)
    pack.pack_tree(source, archive)
    path = archive / "packed_manifest.json"
    manifest = json.loads(path.read_text())
    manifest["files"]["empty.txt"]["sha256"] = "bad"
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="source hash"):
        pack.unpack_tree(archive, tmp_path / "restored")
    assert not (tmp_path / "restored").exists()


def test_selected_subtree_rejects_intermediate_symlink(tmp_path):
    source, archive = setup_source(tmp_path)
    outside = tmp_path / "outside" / "subdir"
    outside.mkdir(parents=True)
    (outside / "external.txt").write_text("not in authorized source tree")
    (source / "alias").symlink_to(outside.parent, target_is_directory=True)
    with pytest.raises(ValueError, match="symlink"):
        pack.pack_tree(source, archive, ["alias/subdir"])
    assert not archive.exists()
