"""Tests for orchestrate/upload_sharded.py — incremental shard upload + overflow.

Covers the per-file walk, the #1824 chunked-batch default (`create_commit`
bulk commits, fresh single-use ops, reactive/proactive overflow parity) and
the #1824 skip-if-present resume probe. Run from the branch checkout with
``uv run pytest tests/test_upload_sharded.py``; if the editable install
resolves ``explore_persona_space`` to another checkout, prefix
``PYTHONPATH=<checkout>/src``.
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


def _http_429():
    """A response-bearing Hub 429, shaped like att-20260715-175238's storm
    ('maximum queue size reached' — the Hub server's queue-full 429 body)."""
    import requests
    from huggingface_hub.errors import HfHubHTTPError

    resp = requests.Response()
    resp.status_code = 429
    return HfHubHTTPError(
        "429 Client Error: Too Many Requests — maximum queue size reached", response=resp
    )


@pytest.fixture(autouse=True)
def fast_retries(monkeypatch):
    """#1345 r5: upload/verify legs now ride hub.retry_transient. Make every
    retry instantaneous + attempt-bound (budget 0 => 6 calls max) so tests
    that exercise transient-looking failures (the FakeApi '500' message is
    classified transient by design) never sleep for real."""
    monkeypatch.setattr("time.sleep", lambda s: None)
    monkeypatch.setenv("EPM_HF_RETRY_BUDGET_S", "0")


class FakeApi:
    """Minimal HfApi stand-in: records uploads keyed by (repo_id, repo_type).

    A repo in ``fail_quota_repos`` raises a persistent storage-quota 403 on
    upload_file/create_commit EXCEPT for the non-LFS ``OVERFLOW_POINTER.json``
    (which rides the quota-immune path, as in reality). A repo in
    ``fail_all_repos`` raises a generic (non-quota) error on every upload.

    #1824 extensions:

    - ``create_commit`` is STATEFUL like the real API (Must-Fix 1 pin): each
      received op object is marked consumed and a reused op raises
      ``ValueError("op reused")`` (mirrors hf_api.py:4196) — ops are consumed
      even by a FAILED commit, exactly the mid-preupload-403 mutation shape.
      ``commit_attempts`` records every call; ``commit_calls`` only
      successes; ``ops_consumed`` counts consumed op objects.
    - ``fail_quota_after_n_commits``: after N successful commits to a
      non-overflow repo, further commits to it raise quota-403 (the
      mid-store-403 case).
    - ``list_repo_tree`` is backed by the pre-seedable ``remote_files``
      ``{(repo_id, repo_type): {path: size}}`` map; an empty view raises
      ``EntryNotFoundError`` (the absent-prefix shape), so existing tests see
      zero skips by default.
    """

    def __init__(
        self,
        *,
        fail_quota_repos=(),
        fail_all_repos=(),
        remote_files=None,
        fail_quota_after_n_commits=None,
    ):
        self.uploaded: dict[tuple[str, str], set[str]] = {}
        self.created_repos: list[tuple[str, str, bool]] = []
        self.fail_quota_repos = set(fail_quota_repos)
        self.fail_all_repos = set(fail_all_repos)
        self.remote_files: dict[tuple[str, str], dict[str, int]] = dict(remote_files or {})
        self.fail_quota_after_n_commits = fail_quota_after_n_commits
        self.commit_attempts: list[tuple[str, str, list[str]]] = []
        self.commit_calls: list[tuple[str, str, list[str]]] = []
        self.ops_consumed = 0
        self._commits_by_repo: dict[str, int] = {}

    def upload_file(self, *, path_or_fileobj, repo_id, path_in_repo, repo_type):
        if repo_id in self.fail_all_repos:
            raise RuntimeError("500 Internal Server Error")
        if repo_id in self.fail_quota_repos and not path_in_repo.endswith("OVERFLOW_POINTER.json"):
            raise RuntimeError(QUOTA_403)
        self.uploaded.setdefault((repo_id, repo_type), set()).add(path_in_repo)

    def create_commit(self, *, repo_id, repo_type, operations, commit_message=None):
        ops = list(operations)
        for op in ops:
            if getattr(op, "_fake_consumed", False):
                raise ValueError(
                    "op reused: CommitOperationAdd objects are single-use (hf_api.py:4196)"
                )
            op._fake_consumed = True
            self.ops_consumed += 1
        paths = sorted(op.path_in_repo for op in ops)
        self.commit_attempts.append((repo_id, repo_type, paths))
        if repo_id in self.fail_all_repos:
            raise RuntimeError("500 Internal Server Error")
        if repo_id in self.fail_quota_repos:
            raise RuntimeError(QUOTA_403)
        if (
            self.fail_quota_after_n_commits is not None
            and repo_id != DEFAULT_OVERFLOW_REPO
            and self._commits_by_repo.get(repo_id, 0) >= self.fail_quota_after_n_commits
        ):
            raise RuntimeError(QUOTA_403)
        self._commits_by_repo[repo_id] = self._commits_by_repo.get(repo_id, 0) + 1
        self.commit_calls.append((repo_id, repo_type, paths))
        for p in paths:
            self.uploaded.setdefault((repo_id, repo_type), set()).add(p)

    def list_repo_tree(
        self, *, repo_id, repo_type=None, revision=None, recursive=False, path_in_repo=None
    ):
        from types import SimpleNamespace

        from huggingface_hub.utils import EntryNotFoundError

        files = self.remote_files.get((repo_id, repo_type), {})
        norm = (path_in_repo or "").rstrip("/")
        under = {p: s for p, s in files.items() if p == norm or p.startswith(norm + "/")}
        if not norm or not under:
            raise EntryNotFoundError("entry not found")
        return [SimpleNamespace(path=p, size=s) for p, s in sorted(under.items())]

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
    FakeApi's recorded uploads PLUS its pre-seeded ``remote_files`` (#1824:
    resume-skipped dests verify against what is already on the Hub). The fake
    ignores ``path`` and returns the full recorded set — the exact-membership
    checks in ``_verify_present`` / ``_batched_verify`` filter."""

    def _list(api, repo_id, path, *, repo_type="model", revision=None):
        present = set(api.uploaded.get((repo_id, repo_type), set()))
        present |= set(getattr(api, "remote_files", {}).get((repo_id, repo_type), {}))
        return sorted(present)

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
    # batch=False pins the legacy PER-FILE walk (#1824; batch sibling:
    # test_batch_happy_path_one_commit).
    res = upload_sharded.upload_dir_sharded(
        local, "superkaiba1/explore-persona-space-data", "issue900_x/store", api=api, batch=False
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
        # batch=False pins the PER-FILE walk (#1824; batch sibling:
        # test_batch_verify_failure_does_not_delete).
        upload_sharded.upload_dir_sharded(
            local,
            "superkaiba1/explore-persona-space-data",
            "issue900_x/store",
            api=api,
            batch=False,
        )
    assert (local / "shard_0000.pt").exists()


def test_verify_false_uploads_and_deletes(tmp_path, offline):
    """verify=False still deletes after an exception-free upload."""
    local = tmp_path / "store"
    _make_shards(local, ["shard_0000.pt"])
    api = FakeApi()
    # batch=False pins the PER-FILE walk's verify=False semantics (#1824).
    res = upload_sharded.upload_dir_sharded(
        local,
        "superkaiba1/explore-persona-space-data",
        "issue900_x/store",
        api=api,
        verify=False,
        batch=False,
    )
    assert res.deleted == ["shard_0000.pt"]
    assert not (local / "shard_0000.pt").exists()


def test_quota_403_reroutes_to_overflow(tmp_path, offline, fake_verify):
    local = tmp_path / "store"
    _make_shards(local, ["shard_0000.pt"])
    canonical = "superkaiba1/explore-persona-space-data"
    api = FakeApi(fail_quota_repos={canonical})

    # batch=False: this test stays a legacy-path reactive-reroute pin (#1824).
    res = upload_sharded.upload_dir_sharded(
        local, canonical, "issue900_x/store", api=api, batch=False
    )

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
        # batch=False pins the PER-FILE both-refused shape (#1824; batch
        # sibling: test_batch_both_repos_refuse_raises).
        upload_sharded.upload_dir_sharded(
            local, canonical, "issue900_x/store", api=api, batch=False
        )
    # fail-loud: local shard NOT deleted.
    assert (local / "shard_0000.pt").exists()


def test_non_quota_error_reraises(tmp_path, offline, fake_verify):
    local = tmp_path / "store"
    _make_shards(local, ["shard_0000.pt"])
    canonical = "superkaiba1/explore-persona-space-data"
    api = FakeApi(fail_all_repos={canonical})  # generic 500 on the main repo

    with pytest.raises(RuntimeError, match="500"):
        # batch=False pins the PER-FILE non-quota re-raise (#1824; batch
        # sibling: test_batch_nonquota_error_reraises_no_delete).
        upload_sharded.upload_dir_sharded(
            local, canonical, "issue900_x/store", api=api, batch=False
        )
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
        # batch=False: the PER-FILE proactive-routing pin (#1824; batch
        # sibling: test_batch_proactive_headroom_routes_to_overflow).
        res = upload_sharded.upload_dir_sharded(
            local, CANONICAL, "issue1034_x/store", api=api, batch=False
        )

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

    # batch=False pins the PER-FILE canonical regression path (#1824).
    res = upload_sharded.upload_dir_sharded(
        local, CANONICAL, "issue1034_x/store", api=api, batch=False
    )

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

    # batch=False pins the PER-FILE privacy-guard path (#1824).
    res = upload_sharded.upload_dir_sharded(
        local, CANONICAL, "issue1034_x/store", api=api, batch=False
    )

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

    # batch=False pins the PER-FILE repo_type-threading path (#1824).
    res = upload_sharded.upload_dir_sharded(
        local, CANONICAL, "issue1034_x/store", api=api, batch=False
    )

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

    # batch=False pins the PER-FILE probe-skip path (#1824).
    res = upload_sharded.upload_dir_sharded(
        local, CANONICAL, "issue1034_x/store", api=api, proactive_overflow=False, batch=False
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

    # batch=False: this test stays a legacy-path reactive-dedup pin (#1824).
    res = upload_sharded.upload_dir_sharded(
        local, CANONICAL, "issue1034_x/store", api=api, batch=False
    )

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


# ---------------------------------------------------------------------------
# #1345 crash-fix r5 — transport is never fatal on the upload + verify legs
# (att-20260715-175238: a Hub queue-full 429 on the bare file_exists verify
# fallback killed the smoke upload leg AFTER the shard had already landed)
# ---------------------------------------------------------------------------


def test_verify_fallback_retries_429_then_success(monkeypatch, caplog):
    """The exact crash chain, regression-pinned (fails pre-fix): the tree
    endpoint 404s on the exact FILE path (documented hub 0.36.2 behavior,
    #939) -> EntryNotFoundError routes to the file_exists fallback -> the
    fallback hits a queue-full 429 ONCE -> hub.list_hf_files_under_path must
    RETRY it (it was the one un-retried Hub call on the verify path), so
    _verify_present returns True instead of the 429 killing the run. Real
    hub bodies run; the fake sits only at the HfApi boundary."""
    from huggingface_hub.utils import EntryNotFoundError

    caplog.set_level(logging.WARNING, logger="explore_persona_space.orchestrate.hub")
    calls = {"n": 0}

    class _StormApi:
        def list_repo_tree(
            self, *, repo_id, repo_type=None, revision=None, recursive=False, path_in_repo=None
        ):
            raise EntryNotFoundError("entry not found")

        def file_exists(self, repo_id, filename, *, repo_type=None, revision=None):
            calls["n"] += 1
            if calls["n"] == 1:
                raise _http_429()
            return True

    ok = upload_sharded._verify_present(
        _StormApi(),
        repo_id="org/data",
        repo_type="dataset",
        dest="issue1345_smoke/analysis_tensors/preds_cache/R_base_r3_prefix_L19.npz",
    )
    assert ok is True
    assert calls["n"] == 2
    # the fix-engaged signal: the retry log line naming the probed path
    assert "file_exists(org/data/issue1345_smoke/analysis_tensors/preds_cache" in caplog.text
    assert "retrying in" in caplog.text


def test_upload_file_retries_429_then_success(tmp_path, offline, fake_verify):
    """A lone 429 on the shard upload itself is retried, not fatal (the
    canonical-branch upload_file was bare pre-fix: any non-quota-403 raise
    was re-raised immediately)."""

    class _Flaky429Api(FakeApi):
        def __init__(self, fail_first_n, **kw):
            super().__init__(**kw)
            self.calls = 0
            self._fail_first_n = fail_first_n

        def upload_file(self, *, path_or_fileobj, repo_id, path_in_repo, repo_type):
            self.calls += 1
            if self.calls <= self._fail_first_n:
                raise _http_429()
            super().upload_file(
                path_or_fileobj=path_or_fileobj,
                repo_id=repo_id,
                path_in_repo=path_in_repo,
                repo_type=repo_type,
            )

    local = tmp_path / "store"
    _make_shards(local, ["shard_0000.pt"])
    api = _Flaky429Api(fail_first_n=1)
    # batch=False: this test stays a legacy-path upload_file retry pin (#1824;
    # batch sibling: test_batch_chunk_429_then_success).
    res = upload_sharded.upload_dir_sharded(
        local, "superkaiba1/explore-persona-space-data", "issue1345_x/store", api=api, batch=False
    )
    assert api.calls == 2  # 429 then success
    assert res.deleted == ["shard_0000.pt"]
    assert res.rerouted == []  # a 429 is transport, never a quota reroute


def test_upload_file_persistent_429_exhausts_fatal(tmp_path, offline, fake_verify, caplog):
    """Genuine exhaustion stays fail-loud: a PERSISTENT 429 storm exhausts the
    bounded retry (budget 0 => attempt floor, 6 calls) and re-raises the 429;
    the local shard is NOT deleted, and the exhaustion log names the path +
    attempt count (the actionable-FATAL contract)."""
    from huggingface_hub.errors import HfHubHTTPError

    caplog.set_level(logging.WARNING, logger="explore_persona_space.orchestrate.hub")

    class _Always429Api(FakeApi):
        def __init__(self, **kw):
            super().__init__(**kw)
            self.calls = 0

        def upload_file(self, *, path_or_fileobj, repo_id, path_in_repo, repo_type):
            self.calls += 1
            raise _http_429()

    local = tmp_path / "store"
    _make_shards(local, ["shard_0000.pt"])
    api = _Always429Api()
    with pytest.raises(HfHubHTTPError, match="429"):
        # batch=False: this test stays a legacy-path upload_file retry pin (#1824).
        upload_sharded.upload_dir_sharded(
            local,
            "superkaiba1/explore-persona-space-data",
            "issue1345_x/store",
            api=api,
            batch=False,
        )
    assert api.calls == 6  # the attempt floor (#735 contract) under budget 0
    assert (local / "shard_0000.pt").exists()  # fail-loud: never delete unverified
    assert "transient-retry exhausted after 6 calls" in caplog.text
    assert "upload_file(superkaiba1/explore-persona-space-data:issue1345_x/store" in caplog.text


# ---------------------------------------------------------------------------
# #1824 — chunked-batch default (bulk create_commit, fresh single-use ops,
# overflow parity) + skip-if-present resume probe
# ---------------------------------------------------------------------------

PREFIX_1824 = "issue1824_x/store"


def test_batch_happy_path_one_commit(tmp_path, offline, fake_verify):
    """#1824 test 1: AUTO mode batches — 5 shards -> exactly ONE create_commit
    on the canonical repo, ZERO per-shard upload_file calls, verify + delete,
    nothing rerouted, nothing skipped."""
    local = tmp_path / "store"
    _make_shards(local, [f"shard_000{i}.pt" for i in range(5)])
    api = CountingFakeApi()

    res = upload_sharded.upload_dir_sharded(local, CANONICAL, PREFIX_1824, api=api)

    dests = sorted(f"{PREFIX_1824}/shard_000{i}.pt" for i in range(5))
    assert api.commit_calls == [(CANONICAL, "dataset", dests)]
    assert api.upload_calls == []  # zero per-shard upload_file calls
    assert res.rerouted == []
    assert res.skipped_existing == []
    assert sorted(res.uploaded) == dests
    assert len(res.deleted) == 5
    assert list(local.glob("*.pt")) == []


def test_batch_chunking(tmp_path, offline, fake_verify):
    """#1824 test 2: batch_chunk_files=2 over 5 shards -> 3 commits of
    2/2/1 files; every dest lands."""
    local = tmp_path / "store"
    _make_shards(local, [f"shard_000{i}.pt" for i in range(5)])
    api = CountingFakeApi()

    res = upload_sharded.upload_dir_sharded(
        local, CANONICAL, PREFIX_1824, api=api, batch_chunk_files=2
    )

    assert [len(c[2]) for c in api.commit_calls] == [2, 2, 1]
    assert all(c[0] == CANONICAL and c[1] == "dataset" for c in api.commit_calls)
    assert api.uploaded[(CANONICAL, "dataset")] == {
        f"{PREFIX_1824}/shard_000{i}.pt" for i in range(5)
    }
    assert len(res.deleted) == 5


def test_batch_quota_403_reroutes_all_chunks(tmp_path, offline, fake_verify):
    """#1824 test 3: canonical quota-403 -> the failing chunk AND every
    remaining chunk land on overflow (model), canonical is attempted exactly
    ONCE, ONE pointer commit + ONE JSONL event, rerouted == uploaded == all
    dests (parity), locals deleted."""
    local = tmp_path / "store"
    _make_shards(local, [f"shard_000{i}.pt" for i in range(5)])
    api = CountingFakeApi(fail_quota_repos={CANONICAL})

    res = upload_sharded.upload_dir_sharded(
        local, CANONICAL, PREFIX_1824, api=api, batch_chunk_files=2
    )

    dests = {f"{PREFIX_1824}/shard_000{i}.pt" for i in range(5)}
    assert api.uploaded[(DEFAULT_OVERFLOW_REPO, "model")] == dests
    # Canonical create_commit attempted exactly ONCE (chunk 1's 403);
    # chunks 2-3 went STRAIGHT to overflow.
    assert len([a for a in api.commit_attempts if a[0] == CANONICAL]) == 1
    assert (DEFAULT_OVERFLOW_REPO, "model", True) in api.created_repos
    pointer_commits = [
        c for c in api.upload_calls if c[0] == CANONICAL and c[2].endswith("OVERFLOW_POINTER.json")
    ]
    assert len(pointer_commits) == 1
    events = _read_events(tmp_path)
    assert len(events) == 1
    assert events[0]["reason"] == "quota-403-reactive"
    assert sorted(res.rerouted) == sorted(dests)
    assert sorted(res.uploaded) == sorted(dests)
    assert len(res.deleted) == 5
    assert list(local.glob("*.pt")) == []


def test_batch_mid_store_403_splits_once(tmp_path, offline, fake_verify):
    """#1824 test 4: chunk 1 lands canonical, the quota gate closes (403 on
    chunk 2) -> chunks 2+3 land overflow; ONE pointer + ONE event; result
    lists split correctly (uploaded = all, rerouted = overflow subset)."""
    local = tmp_path / "store"
    _make_shards(local, [f"shard_000{i}.pt" for i in range(5)])
    api = CountingFakeApi(fail_quota_after_n_commits=1)

    res = upload_sharded.upload_dir_sharded(
        local, CANONICAL, PREFIX_1824, api=api, batch_chunk_files=2
    )

    canonical_dests = {f"{PREFIX_1824}/shard_0000.pt", f"{PREFIX_1824}/shard_0001.pt"}
    overflow_dests = {f"{PREFIX_1824}/shard_000{i}.pt" for i in (2, 3, 4)}
    # Canonical repo holds chunk 1's shards + the (correct) pointer breadcrumb.
    assert api.uploaded[(CANONICAL, "dataset")] == canonical_dests | {
        f"{PREFIX_1824}/OVERFLOW_POINTER.json"
    }
    assert api.uploaded[(DEFAULT_OVERFLOW_REPO, "model")] == overflow_dests
    # Canonical attempts: chunk 1 (success) + chunk 2 (403) — chunk 3 never
    # re-tries canonical.
    assert len([a for a in api.commit_attempts if a[0] == CANONICAL]) == 2
    pointer_commits = [
        c for c in api.upload_calls if c[0] == CANONICAL and c[2].endswith("OVERFLOW_POINTER.json")
    ]
    assert len(pointer_commits) == 1
    events = _read_events(tmp_path)
    assert len(events) == 1
    assert sorted(res.rerouted) == sorted(overflow_dests)
    assert sorted(res.uploaded) == sorted(canonical_dests | overflow_dests)
    assert len(res.deleted) == 5


def test_batch_reroute_uses_fresh_ops(tmp_path, offline, fake_verify):
    """#1824 test 5 (Must-Fix 1 pin): the overflow re-commit after a canonical
    quota-403 receives FRESH op objects. The stateful fake marks every
    received op consumed — even by a FAILED commit, the real mid-preupload-403
    mutation shape (hf_api.py:4196/4442) — and raises ValueError('op reused')
    on reuse, so an ops-reuse regression fails this test loudly; every
    rerouted dest's BYTES are re-committed to overflow."""
    local = tmp_path / "store"
    _make_shards(local, ["shard_0000.pt", "shard_0001.pt", "shard_0002.pt"])
    api = CountingFakeApi(fail_quota_repos={CANONICAL})

    res = upload_sharded.upload_dir_sharded(local, CANONICAL, PREFIX_1824, api=api)

    dests = {f"{PREFIX_1824}/shard_000{i}.pt" for i in range(3)}
    assert api.uploaded[(DEFAULT_OVERFLOW_REPO, "model")] == dests
    # 3 ops consumed by the 403'd canonical attempt + 3 FRESH ones by the
    # overflow re-commit.
    assert api.ops_consumed == 6
    assert sorted(res.rerouted) == sorted(dests)


def test_batch_both_repos_refuse_raises(tmp_path, offline, fake_verify):
    """#1824 (acceptance 4): a chunk BOTH repos refuse raises the RuntimeError
    shape naming the chunk; locals NOT deleted."""
    local = tmp_path / "store"
    _make_shards(local, ["shard_0000.pt"])
    api = CountingFakeApi(fail_quota_repos={CANONICAL}, fail_all_repos={DEFAULT_OVERFLOW_REPO})

    with pytest.raises(RuntimeError, match=r"both main .* and overflow .* refused chunk"):
        upload_sharded.upload_dir_sharded(local, CANONICAL, PREFIX_1824, api=api)
    assert (local / "shard_0000.pt").exists()


def test_batch_proactive_headroom_routes_to_overflow(tmp_path, offline, fake_verify, monkeypatch):
    """#1824 test 6: KNOWN-insufficient + confirmed-public -> chunks committed
    STRAIGHT to overflow, ZERO canonical create_commit attempts (the pointer
    is the only canonical write), ONE pointer + ONE proactive event."""
    local = tmp_path / "store"
    _make_shards(local, ["shard_0000.pt", "shard_0001.pt", "shard_0002.pt"])
    api = CountingFakeApi()
    monkeypatch.setattr(
        upload_sharded, "check_projected_upload_headroom", lambda *a, **k: _insufficient_ph()
    )
    monkeypatch.setattr(
        upload_sharded, "_repo_is_private", lambda repo_id, repo_type="model": False
    )

    res = upload_sharded.upload_dir_sharded(local, CANONICAL, PREFIX_1824, api=api)

    dests = {f"{PREFIX_1824}/shard_000{i}.pt" for i in range(3)}
    assert api.uploaded[(DEFAULT_OVERFLOW_REPO, "model")] == dests
    assert [a for a in api.commit_attempts if a[0] == CANONICAL] == []
    canonical_writes = [c for c in api.upload_calls if c[0] == CANONICAL]
    assert [c[2] for c in canonical_writes] == [f"{PREFIX_1824}/OVERFLOW_POINTER.json"]
    events = _read_events(tmp_path)
    assert len(events) == 1
    assert events[0]["reason"] == "projected-headroom-proactive"
    assert sorted(res.rerouted) == sorted(dests)
    assert sorted(res.uploaded) == sorted(dests)
    assert len(res.deleted) == 3


def test_resume_skips_present_same_size(tmp_path, offline, fake_verify):
    """#1824 test 7: 2 of 3 dests already on the Hub at MATCHING size -> only
    the missing shard is committed; skipped shards land in skipped_existing
    (NOT uploaded), and ALL 3 are verified + deleted under delete_local."""
    local = tmp_path / "store"
    _make_shards(local, ["shard_0000.pt", "shard_0001.pt", "shard_0002.pt"])  # 16 bytes each
    api = CountingFakeApi(
        remote_files={
            (CANONICAL, "dataset"): {
                f"{PREFIX_1824}/shard_0000.pt": 16,
                f"{PREFIX_1824}/shard_0001.pt": 16,
            }
        }
    )

    res = upload_sharded.upload_dir_sharded(local, CANONICAL, PREFIX_1824, api=api)

    assert api.commit_calls == [(CANONICAL, "dataset", [f"{PREFIX_1824}/shard_0002.pt"])]
    assert sorted(res.skipped_existing) == [
        f"{PREFIX_1824}/shard_0000.pt",
        f"{PREFIX_1824}/shard_0001.pt",
    ]
    assert res.uploaded == [f"{PREFIX_1824}/shard_0002.pt"]
    assert res.rerouted == []
    assert len(res.deleted) == 3
    assert list(local.glob("*.pt")) == []


def test_resume_size_mismatch_reuploads(tmp_path, offline, fake_verify):
    """#1824 test 8: a dest present at a DIFFERENT size is re-uploaded, never
    skipped (a partial/corrupt prior upload must not survive)."""
    local = tmp_path / "store"
    _make_shards(local, ["shard_0000.pt"])  # 16 bytes
    api = CountingFakeApi(
        remote_files={(CANONICAL, "dataset"): {f"{PREFIX_1824}/shard_0000.pt": 99}}
    )

    res = upload_sharded.upload_dir_sharded(local, CANONICAL, PREFIX_1824, api=api)

    assert res.skipped_existing == []
    assert api.commit_calls == [(CANONICAL, "dataset", [f"{PREFIX_1824}/shard_0000.pt"])]
    assert res.uploaded == [f"{PREFIX_1824}/shard_0000.pt"]


def test_resume_probe_absent_prefix_uploads_all(tmp_path, offline, fake_verify):
    """#1824 test 9 (the 404-is-expected rule): EntryNotFoundError from the
    scoped tree walk (absent prefix = nothing uploaded yet) -> {} -> full
    upload, no crash."""
    local = tmp_path / "store"
    _make_shards(local, ["shard_0000.pt", "shard_0001.pt"])
    api = CountingFakeApi()  # remote_files empty -> tree raises EntryNotFoundError

    res = upload_sharded.upload_dir_sharded(local, CANONICAL, PREFIX_1824, api=api)

    assert res.skipped_existing == []
    assert api.uploaded[(CANONICAL, "dataset")] == {
        f"{PREFIX_1824}/shard_0000.pt",
        f"{PREFIX_1824}/shard_0001.pt",
    }


def test_resume_skip_false_disables_probe(tmp_path, offline, fake_verify):
    """#1824 test 10: resume_skip=False -> zero probe I/O (list_repo_tree is
    never called), everything uploads."""
    local = tmp_path / "store"
    _make_shards(local, ["shard_0000.pt"])

    class _NoTreeApi(CountingFakeApi):
        def list_repo_tree(self, *a, **k):  # pragma: no cover - the pinned ban
            raise AssertionError("resume probe must not run with resume_skip=False")

    api = _NoTreeApi()
    res = upload_sharded.upload_dir_sharded(
        local, CANONICAL, PREFIX_1824, api=api, resume_skip=False
    )

    assert res.skipped_existing == []
    assert api.uploaded[(CANONICAL, "dataset")] == {f"{PREFIX_1824}/shard_0000.pt"}


def test_root_level_dest_skips_probe(tmp_path, offline, fake_verify, caplog):
    """#1824 test 11 (Must-Fix 2 pin): path_in_repo='' (root-level dests)
    NEVER triggers a tree listing — a recursive walk on path_in_repo='' is
    the #833/#920 full-repo-enumeration wedge — and the upload proceeds
    (always-upload), with one INFO line naming the rationale."""
    local = tmp_path / "store"
    _make_shards(local, ["shard_0000.pt"])

    class _NoTreeApi(CountingFakeApi):
        def list_repo_tree(self, *a, **k):  # pragma: no cover - the pinned ban
            raise AssertionError("list_repo_tree must NEVER be called for root-level dests")

    api = _NoTreeApi()
    caplog.set_level(logging.INFO, logger="explore_persona_space.orchestrate.upload_sharded")
    res = upload_sharded.upload_dir_sharded(local, CANONICAL, "", api=api)

    assert api.uploaded[(CANONICAL, "dataset")] == {"shard_0000.pt"}
    assert res.skipped_existing == []
    assert "resume probe skipped" in caplog.text
    assert "#833" in caplog.text


def test_remote_sizes_under_prefix_empty_prefix_raises():
    """#1824 defensive guard: the probe helper itself refuses an empty prefix
    (never a full-repo listing) — callers must guard, and do."""
    with pytest.raises(ValueError, match="empty prefix"):
        upload_sharded._remote_sizes_under_prefix(FakeApi(), "org/data", "", "dataset")


def test_resume_overflow_pointer_probes_overflow_and_prefers_canonical(
    tmp_path, offline, fake_verify, monkeypatch
):
    """#1824 test 12: an OVERFLOW_POINTER.json in the canonical listing means
    a prior run rerouted -> the overflow repo is probed too; a dest present
    on BOTH repos pends as CANONICAL (skip-precedence), an overflow-only dest
    pends as overflow, and the rest upload."""
    local = tmp_path / "store"
    _make_shards(local, ["shard_0000.pt", "shard_0001.pt", "shard_0002.pt"])  # 16 bytes each
    api = CountingFakeApi(
        remote_files={
            (CANONICAL, "dataset"): {
                f"{PREFIX_1824}/OVERFLOW_POINTER.json": 42,
                f"{PREFIX_1824}/shard_0000.pt": 16,
            },
            (DEFAULT_OVERFLOW_REPO, "model"): {
                f"{PREFIX_1824}/shard_0000.pt": 16,  # on BOTH -> canonical wins
                f"{PREFIX_1824}/shard_0001.pt": 16,  # overflow-only
            },
        }
    )

    seen_pending: list[tuple] = []
    real_bv = upload_sharded._batched_verify

    def _wrap(api_, pending, *, prefix):
        seen_pending.extend(pending)
        return real_bv(api_, pending, prefix=prefix)  # the REAL verify body still runs

    monkeypatch.setattr(upload_sharded, "_batched_verify", _wrap)

    res = upload_sharded.upload_dir_sharded(local, CANONICAL, PREFIX_1824, api=api)

    by_dest = {d: (r, t) for _, d, r, t in seen_pending}
    assert by_dest[f"{PREFIX_1824}/shard_0000.pt"] == (CANONICAL, "dataset")
    assert by_dest[f"{PREFIX_1824}/shard_0001.pt"] == (DEFAULT_OVERFLOW_REPO, "model")
    assert res.uploaded == [f"{PREFIX_1824}/shard_0002.pt"]
    assert sorted(res.skipped_existing) == [
        f"{PREFIX_1824}/shard_0000.pt",
        f"{PREFIX_1824}/shard_0001.pt",
    ]
    assert len(res.deleted) == 3


def test_auto_threshold_routes_large_store_to_perfile(tmp_path, offline, fake_verify, monkeypatch):
    """#1824 test 13: an over-threshold store (env EPM_UPLOAD_BATCH_MAX_GB=0,
    48 bytes > 0) takes the conservative PER-FILE walk under AUTO — zero
    create_commit."""
    local = tmp_path / "store"
    _make_shards(local, ["shard_0000.pt", "shard_0001.pt", "shard_0002.pt"])
    monkeypatch.setenv("EPM_UPLOAD_BATCH_MAX_GB", "0")
    api = CountingFakeApi()

    res = upload_sharded.upload_dir_sharded(local, CANONICAL, PREFIX_1824, api=api)

    assert api.commit_attempts == []
    assert len(api.upload_calls) == 3
    assert len(res.deleted) == 3


def test_batch_false_forces_legacy(tmp_path, offline, fake_verify):
    """#1824 test 14a: batch=False forces the per-file walk regardless of the
    (default, generous) auto threshold."""
    local = tmp_path / "store"
    _make_shards(local, ["shard_0000.pt"])
    api = CountingFakeApi()

    upload_sharded.upload_dir_sharded(local, CANONICAL, PREFIX_1824, api=api, batch=False)

    assert api.commit_attempts == []
    assert len(api.upload_calls) == 1


def test_batch_true_forces_batch(tmp_path, offline, fake_verify, monkeypatch):
    """#1824 test 14b: batch=True forces bulk commits even when AUTO would
    route per-file (tiny env threshold)."""
    local = tmp_path / "store"
    _make_shards(local, ["shard_0000.pt"])
    monkeypatch.setenv("EPM_UPLOAD_BATCH_MAX_GB", "0")
    api = CountingFakeApi()

    upload_sharded.upload_dir_sharded(local, CANONICAL, PREFIX_1824, api=api, batch=True)

    assert len(api.commit_calls) == 1
    assert api.upload_calls == []


def test_batch_nonquota_error_reraises_no_delete(tmp_path, offline, fake_verify):
    """#1824 test 15: a non-quota create_commit failure ('500' — transient by
    message) exhausts the bounded retry (6 attempts under budget 0, each with
    FRESH ops) and re-raises; locals intact, nothing rerouted."""
    local = tmp_path / "store"
    _make_shards(local, ["shard_0000.pt"])
    api = CountingFakeApi(fail_all_repos={CANONICAL})

    with pytest.raises(RuntimeError, match="500"):
        upload_sharded.upload_dir_sharded(local, CANONICAL, PREFIX_1824, api=api)

    assert (local / "shard_0000.pt").exists()
    assert (DEFAULT_OVERFLOW_REPO, "model") not in api.uploaded
    assert len(api.commit_attempts) == 6  # the #735 attempt floor under budget 0
    assert api.ops_consumed == 6  # one FRESH op per retry attempt — never reused


def test_batch_verify_failure_does_not_delete(tmp_path, offline, monkeypatch):
    """#1824 test 16: batch-mode sibling of the per-file pin — a verify miss
    raises and locals are NOT deleted."""
    local = tmp_path / "store"
    _make_shards(local, ["shard_0000.pt"])
    api = CountingFakeApi()
    monkeypatch.setattr(upload_sharded, "list_hf_files_under_path", lambda *a, **k: [])

    with pytest.raises(RuntimeError, match="not found at"):
        upload_sharded.upload_dir_sharded(local, CANONICAL, PREFIX_1824, api=api)
    assert (local / "shard_0000.pt").exists()


def test_batch_chunk_429_then_success(tmp_path, offline, fake_verify):
    """#1824 test 17: a lone 429 on create_commit is retried (with FRESH ops —
    the thunk rebuilds them; the stateful fake would raise 'op reused' on a
    stale retry), not fatal, not rerouted — parity with the per-file retry
    pins."""

    class _Flaky429CommitApi(CountingFakeApi):
        def __init__(self, **kw):
            super().__init__(**kw)
            self.commit_tries = 0

        def create_commit(self, *, repo_id, repo_type, operations, commit_message=None):
            self.commit_tries += 1
            if self.commit_tries == 1:
                # Consume the ops FIRST — a real mid-preupload failure mutates
                # them (hf_api.py:4442) — then die transiently.
                for op in operations:
                    op._fake_consumed = True
                raise _http_429()
            return super().create_commit(
                repo_id=repo_id,
                repo_type=repo_type,
                operations=operations,
                commit_message=commit_message,
            )

    local = tmp_path / "store"
    _make_shards(local, ["shard_0000.pt", "shard_0001.pt"])
    api = _Flaky429CommitApi()

    res = upload_sharded.upload_dir_sharded(local, CANONICAL, PREFIX_1824, api=api)

    assert api.commit_tries == 2  # 429 then success
    assert res.rerouted == []  # a 429 is transport, never a quota reroute
    assert len(res.deleted) == 2


def test_batch_flat_store_cumulative_warn(tmp_path, offline, fake_verify, monkeypatch, caplog):
    """#1824 flat-store residual WARN: when ONE repo dir accumulates more than
    HUB_DIR_FILE_LIMIT files across THIS call's chunks, log exactly ONE
    WARNING (never raise) — the server cap is per-commit-STAGING only."""
    local = tmp_path / "store"
    _make_shards(local, ["shard_0000.pt", "shard_0001.pt", "shard_0002.pt"])
    monkeypatch.setattr(upload_sharded, "HUB_DIR_FILE_LIMIT", 2)
    api = CountingFakeApi()
    caplog.set_level(logging.WARNING, logger="explore_persona_space.orchestrate.upload_sharded")

    res = upload_sharded.upload_dir_sharded(
        local, CANONICAL, PREFIX_1824, api=api, batch_chunk_files=1
    )

    warn_records = [r for r in caplog.records if "flat-store residual" in r.getMessage()]
    assert len(warn_records) == 1
    assert len(res.deleted) == 3  # advisory only — the upload completes
