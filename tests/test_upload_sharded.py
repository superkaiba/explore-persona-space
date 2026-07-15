"""Tests for orchestrate/upload_sharded.py — incremental shard upload + overflow.

Run with ``PYTHONPATH=<worktree>/src`` so the worktree's
``explore_persona_space`` shadows the editable install pointing at main (the
module does not exist on main until this branch merges).
"""

from __future__ import annotations

import json
import logging
from unittest.mock import MagicMock

import pytest

from explore_persona_space.orchestrate import hub as hub_mod
from explore_persona_space.orchestrate import upload_sharded
from explore_persona_space.orchestrate.hub import ProjectedUploadHeadroom
from explore_persona_space.orchestrate.upload_sharded import DEFAULT_OVERFLOW_REPO

QUOTA_403 = "403 Forbidden: You have exceeded your public storage space"


@pytest.fixture(autouse=True)
def _fast_retries(monkeypatch):
    """#1335: upload + verify Hub calls now ride hub's transient retry
    (retry_transient/_retry_upload). Keep failure-path tests fast: the budget
    kill switch bounds retries to the 6-attempt floor and backoff sleeps are
    no-ops (a transient '500 Internal Server Error' fake would otherwise burn
    ~310 s of real sleep per test)."""
    monkeypatch.setenv("EPM_HF_RETRY_BUDGET_S", "0")
    monkeypatch.setattr(hub_mod.time, "sleep", lambda s: None)


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


# ---------------------------------------------------------------------------
# #1034 — proactive projected-headroom routing + reactive pointer/event dedup
# ---------------------------------------------------------------------------

CANONICAL = "superkaiba1/explore-persona-space-data"


class CountingFakeApi(FakeApi):
    """FakeApi + an append-only call log so tests can count Hub COMMITS —
    the base ``uploaded`` dict of sets dedups paths and cannot distinguish
    one pointer commit from N duplicate commits of the same pointer file."""

    def __init__(self, **kw):
        super().__init__(**kw)
        self.upload_calls: list[tuple[str, str, str]] = []  # (repo_id, repo_type, path)

    def upload_file(self, *, path_or_fileobj, repo_id, path_in_repo, repo_type):
        self.upload_calls.append((repo_id, repo_type, path_in_repo))
        super().upload_file(
            path_or_fileobj=path_or_fileobj,
            repo_id=repo_id,
            path_in_repo=path_in_repo,
            repo_type=repo_type,
        )


def _insufficient_ph() -> ProjectedUploadHeadroom:
    return ProjectedUploadHeadroom("insufficient", 2.0, 9.5, 10.0, "live-api")


def _read_events(tmp_path):
    p = tmp_path / "overflow-events.jsonl"
    if not p.exists():
        return []
    return [json.loads(line) for line in p.read_text().split("\n") if line.strip()]


def test_proactive_route_all_shards_to_overflow(
    tmp_path, offline, fake_verify, monkeypatch, caplog
):
    """#1034 test 9: KNOWN-insufficient + confirmed-public canonical target ->
    ALL shards land in overflow (repo_type 'model') UP-FRONT, ZERO canonical
    LFS attempts, exactly ONE pointer at {prefix}/OVERFLOW_POINTER.json, ONE
    JSONL event with reason + projected_gb, and a loud [hf-headroom] WARNING;
    rerouted dests appear in BOTH result lists; locals deleted after verify."""
    local = tmp_path / "store"
    _make_shards(local, ["shard_0000.pt", "shard_0001.pt", "shard_0002.pt"])
    api = CountingFakeApi()
    monkeypatch.setattr(
        upload_sharded, "check_projected_upload_headroom", lambda *a, **k: _insufficient_ph()
    )
    monkeypatch.setattr(
        upload_sharded, "_repo_is_private", lambda repo_id, repo_type="model": False
    )

    with caplog.at_level(logging.WARNING):
        res = upload_sharded.upload_dir_sharded(local, CANONICAL, "issue1034_x/store", api=api)

    dests = {f"issue1034_x/store/shard_000{i}.pt" for i in range(3)}
    assert api.uploaded[(DEFAULT_OVERFLOW_REPO, "model")] == dests
    # ZERO canonical LFS attempts — the pointer is the ONLY canonical write.
    canonical_writes = [c for c in api.upload_calls if c[0] == CANONICAL]
    assert [c[2] for c in canonical_writes] == ["issue1034_x/store/OVERFLOW_POINTER.json"]
    # Overflow repo created private (the _ensure_overflow_repo extraction).
    assert (DEFAULT_OVERFLOW_REPO, "model", True) in api.created_repos
    # Rerouted dests appear in BOTH lists (parity with the reactive path).
    assert sorted(res.rerouted) == sorted(dests)
    assert sorted(res.uploaded) == sorted(dests)
    # Exactly ONE JSONL deviation event, with the proactive reason + size.
    events = _read_events(tmp_path)
    assert len(events) == 1
    assert events[0]["reason"] == "projected-headroom-proactive"
    assert events[0]["projected_gb"] == pytest.approx(48 / 1e9)  # 3 shards x 16 bytes
    assert events[0]["path_in_repo"] == "issue1034_x/store"
    assert events[0]["original_repo"] == CANONICAL
    assert events[0]["effective_repo"] == DEFAULT_OVERFLOW_REPO
    # Loud fail-loud alert line (the committed [hf-headroom] WARNING).
    assert any(
        "[hf-headroom]" in r.getMessage() for r in caplog.records if r.levelno >= logging.WARNING
    )
    # Locals deleted after verified overflow upload.
    assert list(local.glob("*.pt")) == []
    assert len(res.deleted) == 3


@pytest.mark.parametrize("verdict", ["fits", "unknown", "below-threshold", "disabled"])
def test_non_insufficient_verdicts_keep_canonical_path(
    tmp_path, offline, fake_verify, monkeypatch, verdict
):
    """#1034 test 10: fits / unknown / below-threshold / disabled -> the
    canonical upload path behaves exactly as today (regression), and the
    privacy probe is NEVER consulted (short-circuit)."""
    local = tmp_path / "store"
    _make_shards(local, ["shard_0000.pt", "shard_0001.pt"])
    api = CountingFakeApi()
    monkeypatch.setattr(
        upload_sharded,
        "check_projected_upload_headroom",
        lambda *a, **k: ProjectedUploadHeadroom(verdict, 0.0, None, None, "x"),
    )
    privacy = MagicMock(
        side_effect=AssertionError("privacy probe must not run on a non-insufficient verdict")
    )
    monkeypatch.setattr(upload_sharded, "_repo_is_private", privacy)

    res = upload_sharded.upload_dir_sharded(local, CANONICAL, "issue1034_x/store", api=api)

    assert api.uploaded[(CANONICAL, "dataset")] == {
        "issue1034_x/store/shard_0000.pt",
        "issue1034_x/store/shard_0001.pt",
    }
    assert (DEFAULT_OVERFLOW_REPO, "model") not in api.uploaded
    assert res.rerouted == []
    assert _read_events(tmp_path) == []
    assert privacy.call_count == 0


@pytest.mark.parametrize("privacy", [True, None])
def test_privacy_guard_blocks_proactive_reroute(
    tmp_path, offline, fake_verify, monkeypatch, privacy
):
    """#1034 test 11: a private (True) or undeterminable (None) canonical
    target NEVER proactively reroutes despite an insufficient verdict —
    private targets have their own quota; None fails open."""
    local = tmp_path / "store"
    _make_shards(local, ["shard_0000.pt"])
    api = CountingFakeApi()
    monkeypatch.setattr(
        upload_sharded, "check_projected_upload_headroom", lambda *a, **k: _insufficient_ph()
    )
    monkeypatch.setattr(
        upload_sharded, "_repo_is_private", lambda repo_id, repo_type="model": privacy
    )

    res = upload_sharded.upload_dir_sharded(local, CANONICAL, "issue1034_x/store", api=api)

    assert "issue1034_x/store/shard_0000.pt" in api.uploaded[(CANONICAL, "dataset")]
    assert (DEFAULT_OVERFLOW_REPO, "model") not in api.uploaded
    assert res.rerouted == []
    assert _read_events(tmp_path) == []


def test_privacy_probe_repo_type_threading_unmocked(tmp_path, offline, fake_verify, monkeypatch):
    """#1034 test 11b (Must-Fix — production-parity pin): _repo_is_private
    runs UNMOCKED; the FakeApi ``repo_info`` records its ``repo_type`` kwarg,
    asserted == 'dataset' under the default dataset-canonical flow, AND the
    proactive route fires. A bare `_repo_is_private(repo_id)` call (default
    repo_type='model') would 404 -> None -> fail-open -> guard inert on the
    exact incident path — tests 9/11's wholesale mocking cannot catch that."""
    local = tmp_path / "store"
    _make_shards(local, ["shard_0000.pt"])
    api = CountingFakeApi()
    monkeypatch.setattr(
        upload_sharded, "check_projected_upload_headroom", lambda *a, **k: _insufficient_ph()
    )

    repo_info_calls: list[tuple[str, str | None]] = []

    class _HubApi:
        def __init__(self, *a, **k):
            pass

        def repo_info(self, repo_id, *, repo_type=None):
            repo_info_calls.append((repo_id, repo_type))

            class _Info:
                private = False

            return _Info()

    monkeypatch.setattr("huggingface_hub.HfApi", _HubApi)

    res = upload_sharded.upload_dir_sharded(local, CANONICAL, "issue1034_x/store", api=api)

    assert repo_info_calls == [(CANONICAL, "dataset")]
    assert "issue1034_x/store/shard_0000.pt" in api.uploaded[(DEFAULT_OVERFLOW_REPO, "model")]
    assert res.rerouted == ["issue1034_x/store/shard_0000.pt"]


def test_proactive_overflow_false_skips_probe(tmp_path, offline, fake_verify, monkeypatch):
    """#1034 test 12: proactive_overflow=False -> the headroom probe is never
    consulted (zero headroom I/O); straight to the legacy per-shard loop."""
    local = tmp_path / "store"
    _make_shards(local, ["shard_0000.pt"])
    api = CountingFakeApi()
    probe = MagicMock(
        side_effect=AssertionError("probe must not be consulted with proactive_overflow=False")
    )
    monkeypatch.setattr(upload_sharded, "check_projected_upload_headroom", probe)

    res = upload_sharded.upload_dir_sharded(
        local, CANONICAL, "issue1034_x/store", api=api, proactive_overflow=False
    )

    assert "issue1034_x/store/shard_0000.pt" in api.uploaded[(CANONICAL, "dataset")]
    assert res.rerouted == []
    assert probe.call_count == 0


def test_reactive_dedup_one_pointer_one_event_per_prefix(tmp_path, offline, fake_verify):
    """#1034 test 13: 3 shards all quota-403 -> all 3 reroute, but exactly ONE
    pointer COMMIT + ONE JSONL event for the prefix (the 256 commits/hr
    protection — pre-#1034 this issued one pointer commit PER shard). The
    proactive probe runs its REAL body and short-circuits below-threshold
    (48 bytes < 100 GB floor) with zero headroom I/O."""
    local = tmp_path / "store"
    _make_shards(local, ["shard_0000.pt", "shard_0001.pt", "shard_0002.pt"])
    api = CountingFakeApi(fail_quota_repos={CANONICAL})

    res = upload_sharded.upload_dir_sharded(local, CANONICAL, "issue1034_x/store", api=api)

    assert len(res.rerouted) == 3
    assert api.uploaded[(DEFAULT_OVERFLOW_REPO, "model")] == {
        f"issue1034_x/store/shard_000{i}.pt" for i in range(3)
    }
    pointer_commits = [
        c for c in api.upload_calls if c[0] == CANONICAL and c[2].endswith("OVERFLOW_POINTER.json")
    ]
    assert len(pointer_commits) == 1
    events = _read_events(tmp_path)
    assert len(events) == 1
    assert events[0]["reason"] == "quota-403-reactive"


# ---------------------------------------------------------------------------
# #1335 r5 — transport-retried verify fallback + prefix-batched listing
# (crash att-20260715-134136: _verify_present -> list_hf_files_under_path ->
#  per-file api.file_exists, UN-retried, died on a transient HF 429 ~2.8h in)
# ---------------------------------------------------------------------------


def _tree_404_api(file_exists_fn):
    """Stub HfApi: tree endpoint 404s on file paths (the live hub 0.36.2
    behavior, #939) so list_hf_files_under_path takes its file_exists
    fallback — the exact production crash shape."""
    from huggingface_hub.utils import EntryNotFoundError

    class _Api:
        def list_repo_tree(
            self, *, repo_id, repo_type=None, revision=None, recursive=False, path_in_repo=None
        ):
            raise EntryNotFoundError("entry not found")

        def file_exists(self, repo_id, filename, *, repo_type=None, revision=None):
            return file_exists_fn()

        def list_repo_files(self, *a, **k):  # pragma: no cover - must never run
            raise AssertionError("bare full-repo listing must never be called (#920)")

    return _Api()


def test_verify_fallback_429_then_success_retries():
    """#1335 pin 1a: a transient 429 on the exact-file HEAD probe is RETRIED —
    the verify completes instead of crashing the run. Real hub bodies run
    (list_hf_files_under_path -> _retry_upload); the fake sits at the HfApi
    boundary. Pre-fix this propagated the HfHubHTTPError on attempt 1."""
    from huggingface_hub.errors import HfHubHTTPError

    calls = {"n": 0}

    def _flaky():
        calls["n"] += 1
        if calls["n"] == 1:
            raise HfHubHTTPError("429 Too Many Requests ('maximum queue size reached')")
        return True

    out = hub_mod.list_hf_files_under_path(
        _tree_404_api(_flaky), "org/data", "issue1335_x/store/a.pt", repo_type="dataset"
    )
    assert out == ["issue1335_x/store/a.pt"]
    assert calls["n"] == 2


def test_verify_fallback_429_exhaustion_reraises():
    """#1335 pin 1b: a PERSISTENT 429 storm hard-fails only after the bounded
    retry budget exhausts (6-attempt floor under the budget kill switch) —
    never an unbounded loop, never a swallowed error."""
    from huggingface_hub.errors import HfHubHTTPError

    calls = {"n": 0}

    def _always_429():
        calls["n"] += 1
        raise HfHubHTTPError("429 Too Many Requests ('maximum queue size reached')")

    with pytest.raises(HfHubHTTPError, match="maximum queue size"):
        hub_mod.list_hf_files_under_path(
            _tree_404_api(_always_429), "org/data", "issue1335_x/store/a.pt", repo_type="dataset"
        )
    assert calls["n"] == 6  # the #735 attempt floor


def test_batched_verify_one_listing_no_per_file_probes(tmp_path, offline, monkeypatch):
    """#1335 pin 2: N shard files verify via ONE prefix-scoped DIRECTORY
    listing (<=2 listings per call in general — one per destination repo),
    with ZERO per-file file_exists probes. Pre-fix: one exact-file listing
    (tree 404 + HEAD probe) PER shard."""
    local = tmp_path / "store"
    _make_shards(local, [f"shard_000{i}.pt" for i in range(4)])

    listing_calls: list[tuple[str, str, str]] = []

    def _list(api_, repo_id, path, *, repo_type="model", revision=None):
        listing_calls.append((repo_id, path, repo_type))
        return sorted(api_.uploaded.get((repo_id, repo_type), set()))

    monkeypatch.setattr(upload_sharded, "list_hf_files_under_path", _list)

    class _NoFileExistsApi(FakeApi):
        def file_exists(self, *a, **k):  # pragma: no cover - the pinned ban
            raise AssertionError("per-file file_exists probe must not run (#1335)")

    api = _NoFileExistsApi()
    res = upload_sharded.upload_dir_sharded(local, CANONICAL, "issue1335_x/store", api=api)

    assert len(res.deleted) == 4
    assert 1 <= len(listing_calls) <= 2
    # The listing scopes on the DIRECTORY prefix — never a per-shard file path.
    assert {c[1] for c in listing_calls} == {"issue1335_x/store"}
