"""Tests for orchestrate/upload_sharded.py — incremental shard upload + overflow.

Run with ``PYTHONPATH=<worktree>/src`` so the worktree's
``explore_persona_space`` shadows the editable install pointing at main (the
module does not exist on main until this branch merges).
"""

from __future__ import annotations

import pytest

from explore_persona_space.orchestrate import upload_sharded
from explore_persona_space.orchestrate.upload_sharded import DEFAULT_OVERFLOW_REPO

QUOTA_403 = "403 Forbidden: You have exceeded your public storage space"


class FakeApi:
    """Minimal HfApi stand-in: records uploads keyed by (repo_id, repo_type).

    A repo in ``fail_quota_repos`` raises a persistent storage-quota 403 on
    upload_file EXCEPT for the non-LFS ``OVERFLOW_POINTER.json`` (which rides
    the quota-immune path, as in reality). A repo in ``fail_all_repos`` raises
    a generic (non-quota) error on every upload.
    """

    def __init__(self, *, fail_quota_repos=(), fail_all_repos=()):
        self.uploaded: dict[tuple[str, str], set[str]] = {}
        self.created_repos: list[tuple[str, str, bool]] = []
        self.fail_quota_repos = set(fail_quota_repos)
        self.fail_all_repos = set(fail_all_repos)

    def upload_file(self, *, path_or_fileobj, repo_id, path_in_repo, repo_type):
        if repo_id in self.fail_all_repos:
            raise RuntimeError("500 Internal Server Error")
        if repo_id in self.fail_quota_repos and not path_in_repo.endswith("OVERFLOW_POINTER.json"):
            raise RuntimeError(QUOTA_403)
        self.uploaded.setdefault((repo_id, repo_type), set()).add(path_in_repo)

    def create_repo(self, *, repo_id, repo_type, private, exist_ok):
        self.created_repos.append((repo_id, repo_type, private))


@pytest.fixture
def offline(monkeypatch, tmp_path):
    """Keep the overflow event breadcrumb offline + local."""
    monkeypatch.setenv("EPM_HF_STORAGE_CHECK", "0")  # headroom probe short-circuits, no network
    monkeypatch.setenv("EPM_HF_OVERFLOW_EVENT_PATH", str(tmp_path / "overflow-events.jsonl"))


@pytest.fixture
def fake_verify(monkeypatch):
    """Route the SCOPED verify probe (list_hf_files_under_path, #988) at the
    FakeApi's recorded uploads. The fake ignores ``path`` and returns the full
    recorded set — ``_verify_present``'s exact-membership check filters."""

    def _list(api, repo_id, path, *, repo_type="model", revision=None):
        return sorted(api.uploaded.get((repo_id, repo_type), set()))

    monkeypatch.setattr(upload_sharded, "list_hf_files_under_path", _list)
    return _list


def _make_shards(d, names):
    d.mkdir(parents=True, exist_ok=True)
    for n in names:
        (d / n).write_bytes(b"x" * 16)


def test_happy_path_upload_verify_delete(tmp_path, offline, fake_verify):
    local = tmp_path / "store"
    _make_shards(local, ["shard_0000.pt", "shard_0001.pt", "shard_0002.pt"])

    api = FakeApi()
    res = upload_sharded.upload_dir_sharded(
        local, "superkaiba1/explore-persona-space-data", "issue900_x/store", api=api
    )

    dests = api.uploaded[("superkaiba1/explore-persona-space-data", "dataset")]
    assert dests == {
        "issue900_x/store/shard_0000.pt",
        "issue900_x/store/shard_0001.pt",
        "issue900_x/store/shard_0002.pt",
    }
    # local shards deleted after verified upload; nothing rerouted.
    assert list(local.glob("*.pt")) == []
    assert len(res.deleted) == 3
    assert res.rerouted == []


def test_verify_failure_does_not_delete(tmp_path, offline, monkeypatch):
    local = tmp_path / "store"
    _make_shards(local, ["shard_0000.pt"])
    api = FakeApi()
    # verify probe returns nothing → the shard is "not present" → raise, no delete.
    monkeypatch.setattr(upload_sharded, "list_hf_files_under_path", lambda *a, **k: [])

    with pytest.raises(RuntimeError, match="not found at"):
        upload_sharded.upload_dir_sharded(
            local, "superkaiba1/explore-persona-space-data", "issue900_x/store", api=api
        )
    assert (local / "shard_0000.pt").exists()


def test_verify_false_uploads_and_deletes(tmp_path, offline):
    """verify=False still deletes after an exception-free upload."""
    local = tmp_path / "store"
    _make_shards(local, ["shard_0000.pt"])
    api = FakeApi()
    res = upload_sharded.upload_dir_sharded(
        local,
        "superkaiba1/explore-persona-space-data",
        "issue900_x/store",
        api=api,
        verify=False,
    )
    assert res.deleted == ["shard_0000.pt"]
    assert not (local / "shard_0000.pt").exists()


def test_quota_403_reroutes_to_overflow(tmp_path, offline, fake_verify):
    local = tmp_path / "store"
    _make_shards(local, ["shard_0000.pt"])
    canonical = "superkaiba1/explore-persona-space-data"
    api = FakeApi(fail_quota_repos={canonical})

    res = upload_sharded.upload_dir_sharded(local, canonical, "issue900_x/store", api=api)

    # shard landed on the overflow repo (addressed as model), NOT the canonical repo.
    assert "issue900_x/store/shard_0000.pt" in api.uploaded[(DEFAULT_OVERFLOW_REPO, "model")]
    assert (canonical, "dataset") not in api.uploaded or (
        "issue900_x/store/shard_0000.pt" not in api.uploaded.get((canonical, "dataset"), set())
    )
    assert res.rerouted == ["issue900_x/store/shard_0000.pt"]
    # overflow repo created private.
    assert (DEFAULT_OVERFLOW_REPO, "model", True) in api.created_repos
    # OVERFLOW_POINTER.json breadcrumb committed to the canonical repo (non-LFS).
    assert "issue900_x/store/OVERFLOW_POINTER.json" in api.uploaded[(canonical, "dataset")]
    # local shard deleted after verified reroute.
    assert not (local / "shard_0000.pt").exists()


def test_both_repos_refuse_raises(tmp_path, offline, fake_verify):
    local = tmp_path / "store"
    _make_shards(local, ["shard_0000.pt"])
    canonical = "superkaiba1/explore-persona-space-data"
    # canonical hits quota-403, overflow raises a generic error → both refuse.
    api = FakeApi(fail_quota_repos={canonical}, fail_all_repos={DEFAULT_OVERFLOW_REPO})

    with pytest.raises(RuntimeError, match=r"both main .* and overflow"):
        upload_sharded.upload_dir_sharded(local, canonical, "issue900_x/store", api=api)
    # fail-loud: local shard NOT deleted.
    assert (local / "shard_0000.pt").exists()


def test_non_quota_error_reraises(tmp_path, offline, fake_verify):
    local = tmp_path / "store"
    _make_shards(local, ["shard_0000.pt"])
    canonical = "superkaiba1/explore-persona-space-data"
    api = FakeApi(fail_all_repos={canonical})  # generic 500 on the main repo

    with pytest.raises(RuntimeError, match="500"):
        upload_sharded.upload_dir_sharded(local, canonical, "issue900_x/store", api=api)
    # not rerouted, not deleted, nothing on overflow.
    assert (local / "shard_0000.pt").exists()
    assert (DEFAULT_OVERFLOW_REPO, "model") not in api.uploaded


def test_missing_local_dir_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        upload_sharded.upload_dir_sharded(tmp_path / "nope", "repo", "path", api=FakeApi())


@pytest.mark.parametrize("exists,expected", [(True, True), (False, False)])
def test_verify_present_exact_file_probe(monkeypatch, exists, expected):
    """#988 site 11: ``_verify_present`` probes ONE exact shard path via the
    scoped helper (EntryNotFoundError -> file_exists fallback) — never a
    full-repo listing per shard. The REAL list_hf_files_under_path +
    list_repo_files_complete bodies run; fakes sit only at the HfApi boundary
    (signature-mirrored, per the #906 body-test discipline)."""
    from huggingface_hub.utils import EntryNotFoundError

    tree_calls: list[str | None] = []

    class _StubApi:
        def list_repo_tree(
            self, *, repo_id, repo_type=None, revision=None, recursive=False, path_in_repo=None
        ):
            tree_calls.append(path_in_repo)
            raise EntryNotFoundError("entry not found")

        def file_exists(self, repo_id, filename, *, repo_type=None, revision=None):
            return exists

        def list_repo_files(self, *a, **k):  # pragma: no cover - must never run
            raise AssertionError("bare full-repo listing must never be called (#920)")

    ok = upload_sharded._verify_present(
        _StubApi(),
        repo_id="org/data",
        repo_type="dataset",
        dest="issue900_x/store/shard_0000.pt",
    )
    assert ok is expected
    assert tree_calls == ["issue900_x/store/shard_0000.pt"]
