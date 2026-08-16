"""Tests for the #2321 byte-exact v2 pack codec (`orchestrate/packing.py`).

Covers the plan §12 register rows for `test_packing_v2.py`: text/b64/empty/
non-UTF8 round-trip via the PRODUCTION decoder (C19), determinism
(byte-identical re-pack), offsets via `extract_member_from_shard`, member cap,
>9MB-line exclusion, group-key collision suffix decided PRE-WRITE over the
full key set with `rel_dir` recorded (C13), census idempotency, the C8 anchor
re-assert, and the I18 test-mutation interlock.

No network anywhere; the shared conftest additionally pins HF_HUB_OFFLINE=1 +
clears the I18 apply permit for this module (I18/C5).
"""

from __future__ import annotations

import hashlib
import json
import shutil

import pytest

from explore_persona_space.orchestrate import packing


def _write_tree(root, files: dict[str, bytes]) -> None:
    for rel, data in files.items():
        p = root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(data)


SAMPLE = {
    "a/one.json": b'{"k": 1, "text": "line\\nbreak"}',
    "a/two.json": b'{"weird formatting":     3}\n\n',  # producer formatting preserved
    "a/unicode.txt": "café   line-sep ‽\n".encode(),  # raw U+2028 (#950)
    "b/binary.npz": b"\x93NUMPY\x01\x00\x80\xff\x00binary\n\nbytes",  # non-UTF8 -> b64
    "b/empty.done": b"",  # empty file legal
    "root_file.log": b"plain log line\n",
}


def test_roundtrip_byte_exact(tmp_path):
    """Pack -> unpack via the PRODUCTION decoder reproduces every byte (C19)."""
    raw = tmp_path / "raw"
    _write_tree(raw, SAMPLE)
    res = packing.pack_tree_v2(raw, tmp_path / "pack")
    assert res.n_members == len(SAMPLE)
    out = tmp_path / "restored"
    n = packing.unpack_shards_v2(tmp_path / "pack", out)
    assert n == len(SAMPLE)
    for rel, data in SAMPLE.items():
        assert (out / rel).read_bytes() == data, rel


def test_enc_classes(tmp_path):
    """Text files carry enc=text; non-UTF8 carries b64; empty is text."""
    raw = tmp_path / "raw"
    _write_tree(raw, SAMPLE)
    res = packing.pack_tree_v2(raw, tmp_path / "pack")
    entries = {}
    for g in res.groups.values():
        for part in g.index_files:
            entries.update(packing.load_index_part((tmp_path / "pack" / part).read_text()))
    assert entries["a/one.json"].enc == "text"
    assert entries["b/binary.npz"].enc == "b64"
    assert entries["b/empty.done"].enc == "text"
    assert entries["b/empty.done"].size == 0


def test_determinism_byte_identical_repack(tmp_path):
    """Packing the same tree twice yields byte-identical shards + indexes."""
    raw = tmp_path / "raw"
    _write_tree(raw, SAMPLE)
    packing.pack_tree_v2(raw, tmp_path / "p1")
    packing.pack_tree_v2(raw, tmp_path / "p2")
    names1 = sorted(p.name for p in (tmp_path / "p1").iterdir())
    names2 = sorted(p.name for p in (tmp_path / "p2").iterdir())
    assert names1 == names2
    for name in names1:
        b1 = (tmp_path / "p1" / name).read_bytes()
        b2 = (tmp_path / "p2" / name).read_bytes()
        if name == packing.MANIFEST_NAME:
            # census_key is stat(mtime)-based and may differ across copies;
            # compare everything else.
            m1 = json.loads(b1)
            m2 = json.loads(b2)
            m1.pop("census_key")
            m2.pop("census_key")
            assert m1 == m2
        else:
            assert b1 == b2, name


def test_offsets_extract_member(tmp_path):
    """Every index entry's (shard, offset, length) extracts the right member."""
    raw = tmp_path / "raw"
    _write_tree(raw, SAMPLE)
    res = packing.pack_tree_v2(raw, tmp_path / "pack")
    for g in res.groups.values():
        for part in g.index_files:
            entries = packing.load_index_part((tmp_path / "pack" / part).read_text())
            for src, e in entries.items():
                got_src, got_data = packing.extract_member_from_shard(
                    tmp_path / "pack" / e.shard, e.offset, e.length
                )
                assert got_src == src
                assert got_data == SAMPLE[src]
                assert hashlib.sha256(got_data).hexdigest() == e.sha256


def test_member_cap_splits_shards(tmp_path):
    """A shard closes at the member cap even when bytes stay tiny."""
    raw = tmp_path / "raw"
    files = {f"g/f{i:03d}.json": b"{}" for i in range(7)}
    _write_tree(raw, files)
    res = packing.pack_tree_v2(raw, tmp_path / "pack", shard_max_members=3)
    (g,) = res.groups.values()
    assert len(g.shard_files) == 3  # 3 + 3 + 1
    counts = [len(packing.read_shard_member_srcs(tmp_path / "pack" / s)) for s in g.shard_files]
    assert counts == [3, 3, 1]


def test_byte_cap_splits_shards(tmp_path):
    """A shard closes at the byte cap before the member cap."""
    raw = tmp_path / "raw"
    files = {f"g/f{i}.json": b"x" * 500 for i in range(4)}
    _write_tree(raw, files)
    res = packing.pack_tree_v2(raw, tmp_path / "pack", shard_max_bytes=1500)
    (g,) = res.groups.values()
    assert len(g.shard_files) >= 2


def test_oversize_line_raises(tmp_path):
    """A member whose REAL encoded line exceeds the cap fails loud (§3.3(b))."""
    raw = tmp_path / "raw"
    _write_tree(raw, {"g/big.json": b"x" * 3000})
    with pytest.raises(packing.OversizeMemberError):
        packing.pack_tree_v2(raw, tmp_path / "pack", shard_max_bytes=1000)


def test_estimate_encoded_line_bytes():
    """Selection estimate: text ~x1.05, binary x4/3, plus envelope."""
    text = packing.estimate_encoded_line_bytes(1_000_000, "a/rollout.json")
    binary = packing.estimate_encoded_line_bytes(1_000_000, "a/tensor.npz")
    assert 1_000_000 < text < 1_200_000
    assert 1_300_000 < binary < 1_400_000


def test_group_key_collision_decided_pre_write(tmp_path):
    """C13: `a/b_c` vs `a_b/c` collide; both get deterministic sha1 suffixes,
    the mapping is decided over the FULL set, and rel_dir is recorded."""
    keys = packing.derive_group_keys(["a/b_c", "a_b/c", "clean"])
    assert keys["clean"] == "clean"
    k1, k2 = keys["a/b_c"], keys["a_b/c"]
    assert k1 != k2
    assert k1.startswith("a_b_c-") and k2.startswith("a_b_c-")
    # Deterministic regardless of iteration order.
    assert packing.derive_group_keys(["a_b/c", "clean", "a/b_c"]) == keys

    raw = tmp_path / "raw"
    _write_tree(raw, {"a/b_c/x.json": b"{}", "a_b/c/y.json": b"[]", "clean/z.json": b"1"})
    res = packing.pack_tree_v2(raw, tmp_path / "pack")
    groups = packing.load_top_index((tmp_path / "pack" / packing.INDEX_NAME).read_text())
    rel_dirs = {g["rel_dir"] for g in groups.values()}
    assert rel_dirs == {"a/b_c", "a_b/c", "clean"}
    assert set(groups) == set(res.groups)
    # Round-trip still byte-exact through the collided groups.
    out = tmp_path / "restored"
    packing.unpack_shards_v2(tmp_path / "pack", out)
    assert (out / "a/b_c/x.json").read_bytes() == b"{}"
    assert (out / "a_b/c/y.json").read_bytes() == b"[]"


def test_census_idempotent_reuse(tmp_path):
    """A census-matched re-pack REUSES the existing pack (no rewrite)."""
    raw = tmp_path / "raw"
    _write_tree(raw, SAMPLE)
    res1 = packing.pack_tree_v2(raw, tmp_path / "pack")
    assert res1.reused is False
    mtimes = {p.name: p.stat().st_mtime_ns for p in (tmp_path / "pack").iterdir()}
    res2 = packing.pack_tree_v2(raw, tmp_path / "pack")
    assert res2.reused is True
    assert res2.n_members == res1.n_members
    assert {p.name: p.stat().st_mtime_ns for p in (tmp_path / "pack").iterdir()} == mtimes


def test_census_reuse_refuses_stray_and_corrupt(tmp_path):
    """Reuse re-verifies shard sha256s and refuses stray v2 files."""
    raw = tmp_path / "raw"
    _write_tree(raw, SAMPLE)
    res = packing.pack_tree_v2(raw, tmp_path / "pack")
    shard = tmp_path / "pack" / res.all_shard_files()[0]
    good = shard.read_bytes()
    shard.write_bytes(good[:-2] + b'"\n')  # corrupt in place
    with pytest.raises(packing.PackError, match="sha256"):
        packing.pack_tree_v2(raw, tmp_path / "pack")
    shard.write_bytes(good)
    stray = tmp_path / "pack" / "orphan.shard99.jsonl"
    stray.write_bytes(b"{}\n")
    with pytest.raises(packing.PackError, match="stray"):
        packing.pack_tree_v2(raw, tmp_path / "pack")


def test_anchor_mismatch_raises(tmp_path):
    """C8: the packer re-asserts census anchors on the bytes it packs."""
    raw = tmp_path / "raw"
    _write_tree(raw, {"g/a.json": b"{}"})
    good = {"g/a.json": ("gitblob", packing.git_blob_sha1(b"{}"))}
    res = packing.pack_tree_v2(raw, tmp_path / "pack_ok", anchors=good)
    assert res.n_members == 1
    bad = {"g/a.json": ("gitblob", packing.git_blob_sha1(b"other"))}
    with pytest.raises(packing.AnchorMismatchError):
        packing.pack_tree_v2(raw, tmp_path / "pack_bad", anchors=bad)
    missing: dict = {}
    with pytest.raises(packing.AnchorMismatchError, match="coverage gap"):
        packing.pack_tree_v2(raw, tmp_path / "pack_gap", anchors=missing)


def test_sha256_anchor_kind(tmp_path):
    """Tier-B (.npz) anchors use content sha256, not the git blob id."""
    raw = tmp_path / "raw"
    data = b"\x00\x01binary"
    _write_tree(raw, {"g/t.npz": data})
    anchors = {"g/t.npz": ("sha256", hashlib.sha256(data).hexdigest())}
    res = packing.pack_tree_v2(raw, tmp_path / "pack", anchors=anchors)
    assert res.n_members == 1


def test_unpack_never_overwrites_differing(tmp_path):
    """Restore refuses to overwrite an existing file with different bytes."""
    raw = tmp_path / "raw"
    _write_tree(raw, {"g/a.json": b"{}"})
    packing.pack_tree_v2(raw, tmp_path / "pack")
    out = tmp_path / "restored"
    _write_tree(out, {"g/a.json": b"DIFFERENT"})
    with pytest.raises(packing.PackError, match="overwrite"):
        packing.unpack_shards_v2(tmp_path / "pack", out)
    # Identical existing bytes are fine (idempotent re-unpack).
    shutil.rmtree(out)
    _write_tree(out, {"g/a.json": b"{}"})
    assert packing.unpack_shards_v2(tmp_path / "pack", out) == 1


def test_decode_rejects_corrupt_line():
    """The production decoder fails loud on truncation / wrong sha."""
    line = packing.encode_member_line("a/x.json", b'{"k": 1}')
    src, data = packing.decode_member_line(line)
    assert (src, data) == ("a/x.json", b'{"k": 1}')
    rec = json.loads(line)
    rec["sha256"] = "0" * 64
    with pytest.raises(packing.PackError, match="sha256"):
        packing.decode_member_line(json.dumps(rec))
    rec2 = json.loads(line)
    rec2["bytes"] = 999
    with pytest.raises(packing.PackError, match="bytes"):
        packing.decode_member_line(json.dumps(rec2))


def test_git_blob_sha1_matches_git_semantics():
    """sha1('blob <len>\\0' + data) — the A6 anchor (empty-blob constant)."""
    # git's well-known empty-blob id.
    assert packing.git_blob_sha1(b"") == "e69de29bb2d1d6434b8b29ae775ad8c2e48c5391"


# ---------------------------------------------------------------------------
# I18 — test-mutation interlock
# ---------------------------------------------------------------------------


def test_interlock_refuses_canonical_repo(monkeypatch):
    """Under pytest, a canonical-repo mutation without a permit refuses."""
    monkeypatch.delenv("EPM_I2321_TEST_APPLY_PERMIT", raising=False)
    with pytest.raises(packing.TestMutationInterlockError):
        packing.assert_test_mutation_interlock("superkaiba1/explore-persona-space-data")
    with pytest.raises(packing.TestMutationInterlockError):
        packing.assert_test_mutation_interlock("superkaiba1/explore-persona-space")


def test_interlock_allows_fake_repo_and_permit(monkeypatch):
    """Fake repo ids pass; the explicit permit opens canonical repos."""
    monkeypatch.delenv("EPM_I2321_TEST_APPLY_PERMIT", raising=False)
    packing.assert_test_mutation_interlock("fake-org/fake-repo")  # no raise
    monkeypatch.setenv("EPM_I2321_TEST_APPLY_PERMIT", "1")
    packing.assert_test_mutation_interlock("superkaiba1/explore-persona-space-data")


def test_conftest_pins_offline_env():
    """The shared conftest pins HF_HUB_OFFLINE=1 for this module (I18/C5)."""
    import os

    assert os.environ.get("HF_HUB_OFFLINE") == "1"
    assert os.environ.get("EPM_I2321_TEST_APPLY_PERMIT") is None
