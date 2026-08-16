"""Tests for the hub.py packed-tree reader shim (#2321).

Zero network: the Hub boundary is faked signature-conformantly (real
``RepoFile`` objects; ``hf_hub_download`` / ``HfApi`` monkeypatched at the
``huggingface_hub`` module the shim's lazy imports resolve), while the packs
are REAL v2 packs built by ``packing.pack_tree_v2`` — so every test executes
the production resolve + decode bodies end to end (the #906 body-coverage
rule; C19: the shim decodes through the same codec the driver verify uses).

The conftest ``_i2321_test_mutation_interlock_env`` fixture pins
``HF_HUB_OFFLINE=1`` for this module (I18 defense-in-depth); the fakes make
the env moot, but the pin is asserted below so the module registration in
``tests/conftest.py`` cannot silently rot.
"""

from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace

import pytest
from huggingface_hub.hf_api import RepoFile
from huggingface_hub.utils import EntryNotFoundError

from explore_persona_space.orchestrate import hub, packing

REPO = "fake-org/fake-data-repo"
PREFIX = "issX_slug"

# Original (pre-pack) files of the repacked prefix, rel to PREFIX. Deliberate
# shapes: nested text JSON/JSONL (enc=text), a binary blob (enc=b64), a
# top-level file (the "root" group), and a name that ALSO exists raw on the
# remote with different bytes (the raw-preferred dedupe arm).
PACKED_FILES: dict[str, bytes] = {
    "att-1/datagen/raw_pos.jsonl": b'{"r": 1}\n{"r": 2}\n',
    "att-1/datagen/judge_raw_pos.json": b'{"scores": [1, 2]}',
    "att-1/gen_manifest.json": b'{"behavior": "b"}',
    "top.txt": b"top-level\n",
    "blob.npz": bytes(range(256)) * 3,
    "dupe.json": b'{"from": "pack"}',
}
RAW_FILES: dict[str, bytes] = {
    f"{PREFIX}/kept_raw.json": b'{"kept": true}',
    f"{PREFIX}/dupe.json": b'{"from": "raw"}',  # raw wins over the packed dupe
    "elsewhere/foo.json": b'{"other": "prefix"}',
}


class FakeHfApi:
    """Signature-conformant HfApi fake serving a local dir as the repo."""

    def __init__(self, root: Path, token: str | None = None):
        self.root = Path(root)

    def file_exists(self, repo_id, filename, *, repo_type="model", revision=None):
        """Mirror of ``HfApi.file_exists`` against the local root."""
        return (self.root / filename).is_file()

    def repo_info(self, repo_id, *, repo_type="model"):
        """Mirror of ``HfApi.repo_info`` (only ``.sha`` is consumed)."""
        return SimpleNamespace(sha="deadbeefcafe")

    def list_repo_tree(
        self, repo_id, path_in_repo=None, *, repo_type="model", revision=None, recursive=False
    ):
        """Scoped walk yielding REAL ``RepoFile`` entries; absent path 404s."""
        base = self.root / (path_in_repo or "")
        if not base.exists():
            raise EntryNotFoundError(f"Entry Not Found: {path_in_repo}")
        for p in sorted(base.rglob("*")):
            if p.is_file():
                rel = p.relative_to(self.root).as_posix()
                yield RepoFile(path=rel, size=p.stat().st_size, oid="0" * 40)


def _fake_hf_hub_download(root: Path):
    """Signature-mirroring ``hf_hub_download`` fake copying from ``root``."""

    def fake(
        repo_id=None,
        filename=None,
        *,
        repo_type="model",
        revision=None,
        local_dir=None,
        token=None,
    ):
        src = Path(root) / filename
        if not src.is_file():
            raise EntryNotFoundError(f"Entry Not Found for url: {filename}")
        dest = Path(local_dir) / filename
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(src.read_bytes())
        return str(dest)

    return fake


def _build_remote(tmp_path: Path) -> Path:
    """A local 'remote repo': a repacked PREFIX + raw survivors."""
    root = tmp_path / "remote"
    src = tmp_path / "raw_src"
    for rel, data in PACKED_FILES.items():
        p = src / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(data)
    packing.pack_tree_v2(src, root / PREFIX / packing.PACKED_DIRNAME)
    for rel, data in RAW_FILES.items():
        p = root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(data)
    return root


@pytest.fixture(autouse=True)
def _fresh_shim_state(monkeypatch):
    """Module-level caches are shared state — clear around every test."""
    hub.clear_packed_caches()
    monkeypatch.delenv("EPM_HF_PACKED_FALLBACK", raising=False)
    yield
    hub.clear_packed_caches()


@pytest.fixture()
def remote(tmp_path, monkeypatch):
    """Build the remote AND patch the Hub boundary to serve it."""
    root = _build_remote(tmp_path)
    api = FakeHfApi(root)
    monkeypatch.setattr("huggingface_hub.hf_hub_download", _fake_hf_hub_download(root))
    monkeypatch.setattr("huggingface_hub.HfApi", lambda token=None: api)
    return SimpleNamespace(root=root, api=api)


def test_conftest_pins_offline_env():
    """This module must be registered in conftest's interlock module set."""
    assert os.environ.get("HF_HUB_OFFLINE") == "1"


def test_stage_hub_file_serves_packed_text_member(remote, tmp_path):
    target = tmp_path / "out" / "judge_raw_pos.json"
    got = hub.stage_hub_file(
        REPO, f"{PREFIX}/att-1/datagen/judge_raw_pos.json", target, repo_type="dataset"
    )
    assert Path(got).read_bytes() == PACKED_FILES["att-1/datagen/judge_raw_pos.json"]


def test_stage_hub_file_serves_packed_binary_member(remote, tmp_path):
    target = tmp_path / "out" / "blob.npz"
    got = hub.stage_hub_file(REPO, f"{PREFIX}/blob.npz", target, repo_type="dataset")
    assert Path(got).read_bytes() == PACKED_FILES["blob.npz"]  # b64 round-trip byte-exact


def test_stage_hub_file_raw_path_still_preferred(remote, tmp_path):
    """A file present RAW never consults the pack (raw path short-circuits)."""
    target = tmp_path / "out" / "dupe.json"
    hub.stage_hub_file(REPO, f"{PREFIX}/dupe.json", target, repo_type="dataset")
    assert target.read_bytes() == RAW_FILES[f"{PREFIX}/dupe.json"]


def test_stage_hub_file_miss_reraises_original_error(remote, tmp_path):
    with pytest.raises(EntryNotFoundError):
        hub.stage_hub_file(
            REPO, f"{PREFIX}/att-1/nonexistent.json", tmp_path / "x", repo_type="dataset"
        )
    with pytest.raises(EntryNotFoundError):
        # A prefix with NO pack at all misses identically.
        hub.stage_hub_file(REPO, "elsewhere/nonexistent.json", tmp_path / "y", repo_type="dataset")


def test_stage_hub_file_kill_switch_disables_fallback(remote, tmp_path, monkeypatch):
    monkeypatch.setenv("EPM_HF_PACKED_FALLBACK", "0")
    with pytest.raises(EntryNotFoundError):
        hub.stage_hub_file(REPO, f"{PREFIX}/top.txt", tmp_path / "t", repo_type="dataset")


def test_resolve_packed_member_fields(remote):
    m = hub.resolve_packed_member(remote.api, REPO, f"{PREFIX}/top.txt", repo_type="dataset")
    assert m is not None
    assert m.path == f"{PREFIX}/top.txt"
    assert m.shard_repo_path.startswith(f"{PREFIX}/{packing.PACKED_DIRNAME}/root.shard")
    assert m.size == len(PACKED_FILES["top.txt"])
    assert m.enc == "text"
    # Paths inside packed/ never resolve (recursion guard).
    assert (
        hub.resolve_packed_member(
            remote.api,
            REPO,
            f"{PREFIX}/{packing.PACKED_DIRNAME}/{packing.INDEX_NAME}",
            repo_type="dataset",
        )
        is None
    )


def test_collided_groups_resolve_via_recorded_rel_dir(tmp_path, monkeypatch):
    """C13: `a/b_c` and `a_b/c` share a base key; resolution keys on rel_dir."""
    root = tmp_path / "remote"
    src = tmp_path / "src"
    files = {"a/b_c/x.json": b'{"x": 1}', "a_b/c/y.json": b'{"y": 2}'}
    for rel, data in files.items():
        p = src / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(data)
    packing.pack_tree_v2(src, root / "pfx" / packing.PACKED_DIRNAME)
    api = FakeHfApi(root)
    monkeypatch.setattr("huggingface_hub.hf_hub_download", _fake_hf_hub_download(root))
    monkeypatch.setattr("huggingface_hub.HfApi", lambda token=None: api)

    mx = hub.resolve_packed_member(api, REPO, "pfx/a/b_c/x.json", repo_type="dataset")
    my = hub.resolve_packed_member(api, REPO, "pfx/a_b/c/y.json", repo_type="dataset")
    assert mx is not None and my is not None
    assert mx.shard_repo_path != my.shard_repo_path  # disambiguated groups
    out_x = hub.stage_packed_file(REPO, "pfx/a/b_c/x.json", tmp_path / "ox", repo_type="dataset")
    out_y = hub.stage_packed_file(REPO, "pfx/a_b/c/y.json", tmp_path / "oy", repo_type="dataset")
    assert out_x.read_bytes() == files["a/b_c/x.json"]
    assert out_y.read_bytes() == files["a_b/c/y.json"]


def test_stage_hub_prefix_unions_raw_and_packed(remote, tmp_path):
    dest = tmp_path / "mirror"
    staged = hub.stage_hub_prefix(REPO, PREFIX, dest, repo_type="dataset")
    # Raw survivor staged.
    assert (dest / PREFIX / "kept_raw.json").read_bytes() == RAW_FILES[f"{PREFIX}/kept_raw.json"]
    # Every packed member restored at its ORIGINAL path, byte-exact.
    for rel, data in PACKED_FILES.items():
        if rel == "dupe.json":
            continue
        assert (dest / PREFIX / rel).read_bytes() == data
    # Dedupe on original path: the RAW listing wins.
    assert (dest / PREFIX / "dupe.json").read_bytes() == RAW_FILES[f"{PREFIX}/dupe.json"]
    # The pack's own internals never pollute the mirrored layout.
    assert not (dest / PREFIX / packing.PACKED_DIRNAME).exists()
    assert len(staged) == len(set(staged))
    assert all(Path(p).is_file() for p in staged)


def test_stage_hub_prefix_all_packed_subdir(remote, tmp_path):
    """A subdir whose files are ALL packed stages with ZERO raw files."""
    dest = tmp_path / "mirror2"
    staged = hub.stage_hub_prefix(REPO, f"{PREFIX}/att-1", dest, repo_type="dataset")
    assert sorted(Path(p).relative_to(dest).as_posix() for p in staged) == [
        f"{PREFIX}/att-1/datagen/judge_raw_pos.json",
        f"{PREFIX}/att-1/datagen/raw_pos.jsonl",
        f"{PREFIX}/att-1/gen_manifest.json",
    ]
    for p in staged:
        rel = Path(p).relative_to(dest / PREFIX).as_posix()
        assert Path(p).read_bytes() == PACKED_FILES[rel]


def test_stage_hub_prefix_kill_switch_serves_raw_only(remote, tmp_path, monkeypatch):
    monkeypatch.setenv("EPM_HF_PACKED_FALLBACK", "0")
    dest = tmp_path / "mirror3"
    hub.stage_hub_prefix(REPO, PREFIX, dest, repo_type="dataset")
    # Raw listing only: pack internals mirror verbatim, no member restore.
    assert (dest / PREFIX / packing.PACKED_DIRNAME / packing.INDEX_NAME).is_file()
    assert not (dest / PREFIX / "top.txt").exists()


def test_stage_hub_prefix_absent_prefix_still_raises(remote, tmp_path):
    with pytest.raises(FileNotFoundError):
        hub.stage_hub_prefix(REPO, "no/such/prefix", tmp_path / "m", repo_type="dataset")


def test_packed_members_under_path_scoping(remote):
    api = remote.api
    all_members = hub.packed_members_under_path(api, REPO, PREFIX, repo_type="dataset")
    assert {m.path for m in all_members} == {f"{PREFIX}/{rel}" for rel in PACKED_FILES}
    sub = hub.packed_members_under_path(api, REPO, f"{PREFIX}/att-1/datagen", repo_type="dataset")
    assert {m.path for m in sub} == {
        f"{PREFIX}/att-1/datagen/raw_pos.jsonl",
        f"{PREFIX}/att-1/datagen/judge_raw_pos.json",
    }
    exact = hub.packed_members_under_path(api, REPO, f"{PREFIX}/top.txt", repo_type="dataset")
    assert [m.path for m in exact] == [f"{PREFIX}/top.txt"]
    listing = hub.list_packed_members_under_path(api, REPO, PREFIX, repo_type="dataset")
    assert dict(listing) == {f"{PREFIX}/{rel}": len(data) for rel, data in PACKED_FILES.items()}


def test_mid_run_repack_visible_at_unpinned_revision(tmp_path, monkeypatch):
    """I16: at revision=None a repack landing mid-run resolves after ONE
    refresh-on-miss; a PINNED revision keeps serving its cached view."""
    root = tmp_path / "remote"
    src = tmp_path / "src"
    (src / "d").mkdir(parents=True)
    (src / "d" / "a.json").write_bytes(b'{"a": 1}')
    packing.pack_tree_v2(src, root / "pfx" / packing.PACKED_DIRNAME)
    api = FakeHfApi(root)
    monkeypatch.setattr("huggingface_hub.hf_hub_download", _fake_hf_hub_download(root))
    monkeypatch.setattr("huggingface_hub.HfApi", lambda token=None: api)

    # Populate caches at BOTH an unpinned and a pinned revision.
    assert hub.resolve_packed_member(api, REPO, "pfx/d/a.json", repo_type="dataset") is not None
    assert (
        hub.resolve_packed_member(api, REPO, "pfx/d/a.json", repo_type="dataset", revision="r1")
        is not None
    )
    # A later repack adds a member (INDEX.json + parts REPLACED on the remote).
    (src / "d" / "b.json").write_bytes(b'{"b": 2}')
    packing.pack_tree_v2(src, root / "pfx" / packing.PACKED_DIRNAME)

    got = hub.resolve_packed_member(api, REPO, "pfx/d/b.json", repo_type="dataset")
    assert got is not None  # unpinned: refresh-on-miss re-probed
    assert (
        hub.resolve_packed_member(api, REPO, "pfx/d/b.json", repo_type="dataset", revision="r1")
        is None
    )  # pinned: cached view is authoritative for that revision


def test_stage_packed_file_shard_corruption_fails_loud(remote, tmp_path):
    """A truncated shard raises the codec's own error — never a silent stage."""
    shard = next((remote.root / PREFIX / packing.PACKED_DIRNAME).glob("*.shard*.jsonl"))
    shard.write_bytes(shard.read_bytes()[:-5])
    with pytest.raises(packing.PackError):
        hub.stage_packed_file(REPO, f"{PREFIX}/top.txt", tmp_path / "t", repo_type="dataset")
    assert not (tmp_path / "t").exists()


def test_list_repo_repofiles_under_path(remote):
    entries = hub.list_repo_repofiles_under_path(remote.api, REPO, PREFIX, repo_type="dataset")
    assert entries and all(isinstance(e, hub.RepoFileEntry) for e in entries)
    by_path = {e.path: e for e in entries}
    assert f"{PREFIX}/kept_raw.json" in by_path
    assert f"{PREFIX}/{packing.PACKED_DIRNAME}/{packing.INDEX_NAME}" in by_path
    e = by_path[f"{PREFIX}/kept_raw.json"]
    assert e.size == len(RAW_FILES[f"{PREFIX}/kept_raw.json"])
    assert e.is_lfs is False and e.lfs_sha256 is None
    assert e.blob_id == "0" * 40
    assert [x.path for x in entries] == sorted(x.path for x in entries)
    # Absent path -> [] (never a raise); empty path refuses.
    assert (
        hub.list_repo_repofiles_under_path(remote.api, REPO, "nope/none", repo_type="dataset") == []
    )
    with pytest.raises(ValueError):
        hub.list_repo_repofiles_under_path(remote.api, REPO, "", repo_type="dataset")


def test_repofile_entry_lfs_fields(remote):
    """LFS entries surface the content sha256 (A6 anchor semantics)."""

    class LfsApi(FakeHfApi):
        def list_repo_tree(self, repo_id, path_in_repo=None, **kw):
            yield RepoFile(
                path=f"{path_in_repo}/w.safetensors",
                size=9,
                oid="1" * 40,
                lfs={"oid": "c" * 64, "size": 9, "pointerSize": 134},
            )

    entries = hub.list_repo_repofiles_under_path(
        LfsApi(remote.root), REPO, "p", repo_type="dataset"
    )
    assert entries[0].is_lfs is True
    assert entries[0].lfs_sha256 == "c" * 64
    assert entries[0].blob_id == "1" * 40
