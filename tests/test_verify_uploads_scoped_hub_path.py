"""Regression tests for the scoped ``check_hf_hub_path`` listing (#939, #920).

``scripts/verify_uploads.py::check_hf_hub_path`` used to call the bare
full-repo ``api.list_repo_files(...)`` per checked path — on the ~1M-file
data repo that listing wedges >600 s (#920, killed exit 143; the #833
gotcha). The fix scopes the listing server-side via
``hub.list_repo_files_complete(path_in_repo=...)`` with an
``EntryNotFoundError`` -> ``HfApi.file_exists`` exact-file fallback. These
tests pin: the scoped call + kwarg threading, the exact-file fallback, the
MISSING/ERROR taxonomy parity, revision threading, trailing-slash
normalization, the empty-path fail-loud guard, and the
never-calls-bare-``list_repo_files`` anti-regression. Zero network — all
Hub surfaces are stubbed; exception classes are the REAL huggingface_hub
ones. Same module-loading conventions as
tests/test_verify_uploads_type_selection.py.
"""

import importlib.util
import sys
from pathlib import Path

import pytest
from huggingface_hub.hf_api import RepoFile, RepoFolder
from huggingface_hub.utils import EntryNotFoundError, RevisionNotFoundError

# Load the verifier as a module (it's a script, not a package member).
_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "verify_uploads.py"
_spec = importlib.util.spec_from_file_location("verify_uploads_scoped", _SCRIPT)
verify_uploads = importlib.util.module_from_spec(_spec)  # type: ignore[arg-type]
sys.modules["verify_uploads_scoped"] = verify_uploads
_spec.loader.exec_module(verify_uploads)  # type: ignore[union-attr]

# verify_uploads.py put src/ on sys.path at exec time; import the hub module
# the function resolves at call time so monkeypatching its attribute works.
from explore_persona_space.orchestrate import hub  # noqa: E402


class StubApi:
    """Stands in for ``huggingface_hub.HfApi`` inside ``check_hf_hub_path``.

    ``list_repo_files`` raises unconditionally — the #920 anti-regression pin:
    no code path may fall back to the bare full-repo listing.
    """

    def __init__(self, *, token=None, file_exists_result=False):
        self.token = token
        self.file_exists_result = file_exists_result
        self.file_exists_calls: list[dict] = []

    def list_repo_files(self, *args, **kwargs):  # pragma: no cover - must never run
        raise AssertionError("bare full-repo listing (list_repo_files) must never be called")

    def file_exists(self, repo_id, filename, *, repo_type=None, revision=None, token=None):
        self.file_exists_calls.append(
            {
                "repo_id": repo_id,
                "filename": filename,
                "repo_type": repo_type,
                "revision": revision,
            }
        )
        return self.file_exists_result


class RecordingLister:
    """Fake ``hub.list_repo_files_complete`` that records its kwargs."""

    def __init__(self, result=None, raises=None):
        self.result = result if result is not None else []
        self.raises = raises
        self.calls: list[dict] = []

    def __call__(self, api, repo_id, *, repo_type="model", revision=None, path_in_repo=None):
        self.calls.append(
            {
                "repo_id": repo_id,
                "repo_type": repo_type,
                "revision": revision,
                "path_in_repo": path_in_repo,
            }
        )
        if self.raises is not None:
            raise self.raises
        return list(self.result)


def _install(monkeypatch, api: StubApi, lister) -> None:
    """Route the function's call-time lookups at the stub seams."""
    monkeypatch.setattr("huggingface_hub.HfApi", lambda *, token=None: api)
    monkeypatch.setattr("explore_persona_space.orchestrate.hub.list_repo_files_complete", lister)


def test_scoped_dir_ok(monkeypatch):
    api = StubApi()
    lister = RecordingLister(result=["a/b/x.json", "a/b/y.json"])
    _install(monkeypatch, api, lister)
    result = verify_uploads.check_hf_hub_path("owner/repo", "a/b", "dataset")
    assert result["status"] == "OK"
    assert result["file_count"] == 2
    assert result["url"] == "https://huggingface.co/owner/repo/tree/main/a/b"
    assert lister.calls == [
        {"repo_id": "owner/repo", "repo_type": "dataset", "revision": None, "path_in_repo": "a/b"}
    ]
    assert api.file_exists_calls == []


def test_exact_file_ok(monkeypatch):
    api = StubApi(file_exists_result=True)
    lister = RecordingLister(raises=EntryNotFoundError("a/b/x.json not a directory"))
    _install(monkeypatch, api, lister)
    result = verify_uploads.check_hf_hub_path("owner/repo", "a/b/x.json", "model")
    assert result["status"] == "OK"
    assert result["file_count"] == 1
    assert result["url"] == "https://huggingface.co/owner/repo/tree/main/a/b/x.json"
    assert api.file_exists_calls == [
        {"repo_id": "owner/repo", "filename": "a/b/x.json", "repo_type": "model", "revision": None}
    ]


def test_missing_path(monkeypatch):
    api = StubApi(file_exists_result=False)
    lister = RecordingLister(raises=EntryNotFoundError("a/b not found"))
    _install(monkeypatch, api, lister)
    result = verify_uploads.check_hf_hub_path("owner/repo", "a/b", "dataset")
    assert result["status"] == "MISSING"
    assert result["url"] == ""
    assert result["detail"] == "No files under a/b at revision main"


def test_revision_not_found_maps_to_error(monkeypatch):
    api = StubApi()
    lister = RecordingLister(raises=RevisionNotFoundError("bogus revision"))
    _install(monkeypatch, api, lister)
    result = verify_uploads.check_hf_hub_path("owner/repo", "a/b", "model", revision="deadbeef")
    assert result["status"] == "ERROR"
    assert result["url"] == ""
    assert "bogus revision" in result["detail"]
    # file_exists must NOT fire for a non-EntryNotFoundError failure.
    assert api.file_exists_calls == []


def test_pinned_revision_threaded(monkeypatch):
    # OK path: the lister receives the pinned SHA and the URL cites it.
    api = StubApi()
    lister = RecordingLister(result=["a/b/x.json"])
    _install(monkeypatch, api, lister)
    result = verify_uploads.check_hf_hub_path("owner/repo", "a/b", "model", revision="deadbeef")
    assert result["status"] == "OK"
    assert result["url"] == "https://huggingface.co/owner/repo/tree/deadbeef/a/b"
    assert lister.calls[0]["revision"] == "deadbeef"

    # Exact-file fallback path: file_exists receives the pinned SHA too.
    api2 = StubApi(file_exists_result=True)
    lister2 = RecordingLister(raises=EntryNotFoundError("not a directory"))
    _install(monkeypatch, api2, lister2)
    result2 = verify_uploads.check_hf_hub_path(
        "owner/repo", "a/b/x.json", "model", revision="deadbeef"
    )
    assert result2["status"] == "OK"
    assert api2.file_exists_calls[0]["revision"] == "deadbeef"


def test_never_calls_bare_list_repo_files(monkeypatch):
    # StubApi.list_repo_files raises AssertionError; both the scoped-dir OK
    # path and the exact-file fallback path must complete WITH the expected
    # RESULT STATUS (amendment 5: the outer generic except would swallow a
    # planted AssertionError into an ERROR dict, so "no exception" is not
    # enough — assert the status + file_count).
    api = StubApi()
    lister = RecordingLister(result=["a/b/x.json", "a/b/y.json", "a/b/z.json"])
    _install(monkeypatch, api, lister)
    result = verify_uploads.check_hf_hub_path("owner/repo", "a/b", "dataset")
    assert result["status"] == "OK"
    assert result["file_count"] == 3

    api2 = StubApi(file_exists_result=True)
    lister2 = RecordingLister(raises=EntryNotFoundError("not a directory"))
    _install(monkeypatch, api2, lister2)
    result2 = verify_uploads.check_hf_hub_path("owner/repo", "a/b/x.json", "model")
    assert result2["status"] == "OK"
    assert result2["file_count"] == 1


def test_trailing_slash_normalized(monkeypatch):
    api = StubApi()
    lister = RecordingLister(result=["a/b/x.json"])
    _install(monkeypatch, api, lister)
    result = verify_uploads.check_hf_hub_path("owner/repo", "a/b/", "dataset")
    assert result["status"] == "OK"
    # The lister gets the normalized prefix; the URL keeps the caller's form.
    assert lister.calls[0]["path_in_repo"] == "a/b"
    assert result["url"] == "https://huggingface.co/owner/repo/tree/main/a/b/"


def test_empty_path_is_error(monkeypatch):
    api = StubApi()
    lister = RecordingLister(result=["anything"])
    _install(monkeypatch, api, lister)
    for empty in ("", "/", "///"):
        result = verify_uploads.check_hf_hub_path("owner/repo", empty, "dataset")
        assert result["status"] == "ERROR"
        assert result["detail"] == "empty path_in_repo"
    # The lister (and thus any repo listing) must never fire on an empty path.
    assert lister.calls == []


# ---------------------------------------------------------------------------
# Hub-side kwarg threading (conditional — amendment 1)
# ---------------------------------------------------------------------------


class KwargRecordingApi:
    """Fake HfApi whose ``list_repo_tree`` records the exact kwargs passed."""

    def __init__(self, paths=()):
        self.paths = list(paths)
        self.tree_calls: list[dict] = []

    def list_repo_tree(self, **kwargs):
        self.tree_calls.append(dict(kwargs))
        entries: list = [RepoFile(path=p, size=1, blob_id="b", oid="o") for p in self.paths]
        entries.append(RepoFolder(path="a", tree_id="t", oid="o"))  # dropped by the filter
        return entries


def test_list_repo_files_complete_threads_path_in_repo():
    api = KwargRecordingApi(paths=["x/y/one.json"])
    result = hub.list_repo_files_complete(api, "owner/repo", path_in_repo="x/y")
    assert result == ["x/y/one.json"]
    assert api.tree_calls == [
        {
            "repo_id": "owner/repo",
            "repo_type": "model",
            "revision": None,
            "recursive": True,
            "path_in_repo": "x/y",
        }
    ]


def test_list_repo_files_complete_omits_kwarg_when_unset():
    # A kwarg-free call must NOT pass path_in_repo AT ALL — byte-identical
    # calls against strict fakes (tests/test_hf_storage_headroom.py's
    # keyword-only fake has no path_in_repo parameter).
    api = KwargRecordingApi(paths=["one.json"])
    result = hub.list_repo_files_complete(api, "owner/repo")
    assert result == ["one.json"]
    assert api.tree_calls == [
        {"repo_id": "owner/repo", "repo_type": "model", "revision": None, "recursive": True}
    ]
    assert "path_in_repo" not in api.tree_calls[0]


# ---------------------------------------------------------------------------
# Composition through the REAL list_repo_files_complete (amendment 6)
# ---------------------------------------------------------------------------


class LazyRaisingApi(StubApi):
    """``list_repo_tree`` is a REAL generator raising EntryNotFoundError
    mid-iteration — pins the lazy-raise contract through the real helper's
    materialize-inside-the-retry-thunk shape."""

    def list_repo_tree(self, **kwargs):
        def _gen():
            yield RepoFile(path="a/b/x.json", size=1, blob_id="b", oid="o")
            # Message deliberately free of 500/502/503/504 substrings so the
            # retry wrapper's transient predicate re-raises immediately.
            raise EntryNotFoundError("entry a/b not found")

        return _gen()


@pytest.mark.parametrize("file_exists_result,expected_status", [(False, "MISSING"), (True, "OK")])
def test_composition_lazy_entry_not_found(monkeypatch, file_exists_result, expected_status):
    api = LazyRaisingApi(file_exists_result=file_exists_result)
    monkeypatch.setattr("huggingface_hub.HfApi", lambda *, token=None: api)
    # No lister patch: check_hf_hub_path drives the REAL list_repo_files_complete.
    result = verify_uploads.check_hf_hub_path("owner/repo", "a/b", "dataset")
    assert result["status"] == expected_status
    if expected_status == "OK":
        assert result["file_count"] == 1
