"""Tests for ``hub.assert_hf_prefix_exists`` — the raising absence-check primitive (#2442).

Contract under test (upload-policy.md § Absence checks): an absence check must
use a call that can FAIL on a wrong location — the helper returns the file
count at/under a prefix and RAISES when the prefix does not exist, never a
silent 0 (#2329: ``list_repo_files`` + a client-side ``startswith`` filter
printed a confident 0 for a prefix that never existed).

Fake discipline: every fake ``list_repo_tree`` returns a LAZY generator that
raises on first ``next()``, never at call time — mirroring verified
huggingface_hub 0.36.2 semantics (paginate-and-yield; the 404 raises at
ITERATION, #779) — so an implementation that materializes outside the retry
thunk / guard FAILS these tests instead of passing. Fakes mirror the real
``HfApi`` method signatures by construction and use real ``RepoFile`` /
``RepoFolder`` instances (code-style.md § One production-body test per
seam-stubbed function); the helper's own body executes in every test.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator

import pytest
import requests
from huggingface_hub.hf_api import RepoFile, RepoFolder
from huggingface_hub.utils import EntryNotFoundError, HfHubHTTPError

from explore_persona_space.orchestrate.hub import assert_hf_prefix_exists


def _response(status_code: int, retry_after: str | None = None) -> requests.Response:
    """Minimal real requests.Response carrying a status code (+ Retry-After)."""
    resp = requests.Response()
    resp.status_code = status_code
    if retry_after is not None:
        resp.headers["Retry-After"] = retry_after
    return resp


def _repo_file(path: str) -> RepoFile:
    return RepoFile(**{"type": "file", "path": path, "size": 1, "oid": "0" * 40})


def _repo_folder(path: str) -> RepoFolder:
    return RepoFolder(**{"type": "directory", "path": path, "oid": "1" * 40})


def _lazy_raise(exc: Exception) -> Callable[[], Iterator]:
    """A tree-call batch: returns a generator that raises on FIRST next()."""

    def batch() -> Iterator:
        def gen() -> Iterator:
            raise exc
            yield  # pragma: no cover — makes this a generator function

        return gen()

    return batch


def _lazy_yield(entries: list) -> Callable[[], Iterator]:
    """A tree-call batch: returns a generator lazily yielding ``entries``."""

    def batch() -> Iterator:
        def gen() -> Iterator:
            yield from entries

        return gen()

    return batch


class _FakeHfApi:
    """Signature-conformant HfApi fake — each ``def`` mirrors the real
    huggingface_hub 0.36.2 signature, so any call valid on the real API binds
    here and any drift raises ``TypeError`` (never a silently-absorbing Mock).
    """

    def __init__(
        self,
        tree_batches: list[Callable[[], Iterator]],
        *,
        file_exists_result: bool = False,
        repo_files: tuple[str, ...] = (),
    ) -> None:
        self._tree_batches = list(tree_batches)
        self._file_exists_result = file_exists_result
        self._repo_files = list(repo_files)
        self.tree_calls = 0
        self.file_exists_calls = 0

    def list_repo_tree(
        self,
        repo_id: str,
        path_in_repo: str | None = None,
        *,
        recursive: bool = False,
        expand: bool = False,
        revision: str | None = None,
        repo_type: str | None = None,
        token: str | None = None,
    ) -> Iterator:
        idx = min(self.tree_calls, len(self._tree_batches) - 1)
        self.tree_calls += 1
        return self._tree_batches[idx]()

    def file_exists(
        self,
        repo_id: str,
        filename: str,
        *,
        repo_type: str | None = None,
        revision: str | None = None,
        token: str | None = None,
    ) -> bool:
        self.file_exists_calls += 1
        return self._file_exists_result

    def list_repo_files(
        self,
        repo_id: str,
        *,
        revision: str | None = None,
        repo_type: str | None = None,
        token: str | None = None,
    ) -> list[str]:
        return list(self._repo_files)


def _entry_not_found() -> EntryNotFoundError:
    return EntryNotFoundError("404 Client Error: Entry Not Found for url", response=_response(404))


def test_absent_prefix_raises() -> None:
    """A nonexistent prefix RAISES — never a silent 0 (#2329).

    Discriminating fixture (plan criterion 4): the fake ALSO exposes a
    populated ``list_repo_files`` listing none of whose entries start with the
    fixture prefix — a client-side-filter implementation would return 0 from
    it instead of raising, and this test would FAIL on the missing raise.
    """
    api = _FakeHfApi(
        tree_batches=[_lazy_raise(_entry_not_found())],
        file_exists_result=False,
        repo_files=(
            "issue2329_q35rerun/analysis_tensors/ladder/cells.json",
            "issue2329_q35rerun/raw_completions/ladder/rollouts.json",
        ),
    )
    with pytest.raises(RuntimeError, match="does not exist"):
        assert_hf_prefix_exists(api, "org/data-repo", "issue2329_q35rerun/ladder/grid")
    assert api.tree_calls == 1
    assert api.file_exists_calls == 1  # the #939 exact-file probe ran before the raise


def test_exact_file_path_returns_one() -> None:
    """An exact FILE path returns 1 via the ``file_exists`` fallback (#939).

    Pins Must-Fix 2: the tree endpoint 404s on file paths, so without the
    fallback the helper would report a PRESENT file as absent.
    """
    api = _FakeHfApi(
        tree_batches=[_lazy_raise(_entry_not_found())],
        file_exists_result=True,
    )
    assert assert_hf_prefix_exists(api, "org/data-repo", "prefix/exact_file.json") == 1
    assert api.file_exists_calls == 1


def test_populated_prefix_returns_count() -> None:
    """A populated prefix returns the FILE count; folder entries are dropped."""
    entries = [
        _repo_file("p/a.json"),
        _repo_folder("p/sub"),
        _repo_file("p/b.json"),
        _repo_file("p/sub/c.json"),
    ]
    api = _FakeHfApi(tree_batches=[_lazy_yield(entries)])
    assert assert_hf_prefix_exists(api, "org/data-repo", "p") == 3
    assert api.file_exists_calls == 0


def test_transient_error_is_retried_not_reported_as_absent() -> None:
    """A 429-shaped transient on the FIRST iteration is retried, not misread
    as absence: the second tree call succeeds and the count is returned.

    The transient carries ``Retry-After: 0.01`` so the retry sleep is
    sub-second (hub honors the header; no time monkeypatching needed).
    """
    transient = HfHubHTTPError("429 Too Many Requests", response=_response(429, retry_after="0.01"))
    entries = [_repo_file("p/a.json"), _repo_file("p/b.json")]
    api = _FakeHfApi(tree_batches=[_lazy_raise(transient), _lazy_yield(entries)])
    assert assert_hf_prefix_exists(api, "org/data-repo", "p") == 2
    assert api.tree_calls == 2
    assert api.file_exists_calls == 0


def test_empty_prefix_raises_value_error() -> None:
    """A falsy prefix refuses up front — it would degrade to a full-repo listing."""
    api = _FakeHfApi(tree_batches=[_lazy_yield([])])
    with pytest.raises(ValueError, match="empty prefix"):
        assert_hf_prefix_exists(api, "org/data-repo", "/")
    assert api.tree_calls == 0


def test_resolved_prefix_with_zero_files_raises() -> None:
    """A RESOLVED prefix listing zero files raises — never a silent 0.

    Pins the third raise path (the ``hub.py`` resolved-empty guard): the
    prefix is NONEMPTY so the up-front ValueError gate passes, the lazy
    generator IS consumed (``tree_calls == 1``) and yields nothing, and the
    helper refuses to return 0 — a regression replacing the guard with
    ``return 0`` fails here. ``file_exists_calls == 0`` pins that the branch
    did not detour through the #939 exact-file fallback.
    """
    api = _FakeHfApi(tree_batches=[_lazy_yield([])])
    with pytest.raises(RuntimeError, match="listed 0 files"):
        assert_hf_prefix_exists(api, "org/data-repo", "p")
    assert api.tree_calls == 1
    assert api.file_exists_calls == 0
