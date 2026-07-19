"""#1426 crash-fix pin: ``upload_folder_scoped_verify`` retries transient HF 429s.

Both #1426 GCE crashes routed through this ONE function in
``scripts/issue928_common.py``:

1. Crash 1 (2026-07-18 07:52Z): ``api.upload_folder`` raised ``HfHubHTTPError``
   429 ("exceeded the rate limit for repository commits (256 per hour)").
2. Crash 2 (09:00Z, post-resume): the bare ``api.list_repo_tree`` verify raised
   a 429 ("maximum time in concurrency queue reached") DURING generator
   iteration inside a subprocess fit script, killing the run.

The fix wraps the upload leg in ``hub.retry_transient`` and replaces the bare
verify comprehension with ``hub.list_repo_files_complete`` (retry-wrapped,
materialized inside the retry thunk, server-side scoped). These tests execute
the REAL function body, faking ONLY the external Hub boundary with a
signature-conformant ``create_autospec(HfApi)`` fake; they fail pre-fix (the
first 429 propagated) and pass post-fix. No network, no sleeps (the hub
module's ``time`` is swapped for a no-sleep namespace).
"""

from __future__ import annotations

import logging
import sys
import time
import types
from pathlib import Path
from unittest.mock import create_autospec

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))
sys.path.insert(0, str(REPO_ROOT / "src"))

PREFIX = "issue1426_test/fit_results"


def _http_err(status: int, msg: str):
    """A response-bearing HfHubHTTPError with the given HTTP status code."""
    import requests
    from huggingface_hub.errors import HfHubHTTPError

    resp = requests.Response()
    resp.status_code = status
    return HfHubHTTPError(msg, response=resp)


def _repo_file(path: str):
    from huggingface_hub.hf_api import RepoFile

    return RepoFile(path=path, size=1, oid="0" * 40)


@pytest.fixture()
def no_sleep(monkeypatch):
    """Neutralize retry backoff sleeps in the hub retry helper (module-scoped)."""
    import explore_persona_space.orchestrate.hub as hub

    monkeypatch.setattr(
        hub, "time", types.SimpleNamespace(monotonic=time.monotonic, sleep=lambda _s: None)
    )
    return hub


def _fake_api(monkeypatch):
    """Signature-conformant fake HfApi installed at the function's import site."""
    from huggingface_hub import HfApi

    fake = create_autospec(HfApi, instance=True)
    monkeypatch.setattr("huggingface_hub.HfApi", lambda: fake)
    return fake


def test_retries_429_on_both_legs_then_succeeds(monkeypatch, tmp_path, caplog, no_sleep):
    """429 on upload AND a 429 raised DURING verify iteration both retry to success."""
    from issue928_common import upload_folder_scoped_verify

    fake = _fake_api(monkeypatch)
    # Crash-1 shape: first upload_folder call 429s, second succeeds.
    fake.upload_folder.side_effect = [
        _http_err(429, "You have exceeded the rate limit for repository commits (256 per hour)"),
        None,
    ]

    # Crash-2 shape: list_repo_tree returns a LAZY generator that raises at
    # iteration time on the first call, then a good listing on the second.
    def _raising_iter():
        raise _http_err(429, "maximum time in concurrency queue reached")
        yield  # pragma: no cover  (makes this a generator function)

    fake.list_repo_tree.side_effect = [_raising_iter(), [_repo_file(f"{PREFIX}/a.json")]]

    with caplog.at_level(logging.WARNING, logger="explore_persona_space.orchestrate.hub"):
        out = upload_folder_scoped_verify(tmp_path, PREFIX, ["a.json"], "test commit")

    assert out == PREFIX
    assert fake.upload_folder.call_count == 2
    assert fake.list_repo_tree.call_count == 2
    # Fix-engaged signal: the retry helper's per-attempt WARNING names each leg.
    assert f"upload_folder({PREFIX}) transient error" in caplog.text
    assert "list_repo_tree(" in caplog.text


def test_non_transient_404_raises_immediately(monkeypatch, tmp_path, no_sleep):
    """A 4xx (non-429/408) is NOT retried — one call, immediate re-raise."""
    from huggingface_hub.errors import HfHubHTTPError
    from issue928_common import upload_folder_scoped_verify

    fake = _fake_api(monkeypatch)
    fake.upload_folder.side_effect = _http_err(404, "Repository Not Found")

    with pytest.raises(HfHubHTTPError, match="Repository Not Found"):
        upload_folder_scoped_verify(tmp_path, PREFIX, ["a.json"], "test commit")

    assert fake.upload_folder.call_count == 1
    fake.list_repo_tree.assert_not_called()


def test_exact_set_verify_still_raises_on_missing_file(monkeypatch, tmp_path, no_sleep):
    """Semantics preserved: a missing expected file is still a RuntimeError."""
    from issue928_common import upload_folder_scoped_verify

    fake = _fake_api(monkeypatch)
    fake.upload_folder.side_effect = [None]
    fake.list_repo_tree.side_effect = [[_repo_file(f"{PREFIX}/a.json")]]

    with pytest.raises(RuntimeError, match="upload verification FAILED"):
        upload_folder_scoped_verify(tmp_path, PREFIX, ["a.json", "b.json"], "test commit")
