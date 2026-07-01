"""Regression tests for issue #778 upload-verify caller (CPU-only, offline).

Pins the fix for ``epm:failure v3``: ``_verify_prefix`` called
``list_repo_files_complete(repo_id, ...)`` MISSING the leading ``api`` positional
(the library signature is ``list_repo_files_complete(api, repo_id, *, ...)``), so
``repo_id`` bound to the ``api`` slot and Python raised
``TypeError: missing 1 required positional argument: 'repo_id'``. The fix passes
an ``HfApi`` instance first.

These tests monkeypatch ``list_repo_files_complete`` in the ``issue778_upload``
module namespace (where ``_verify_prefix`` resolves it) and never touch the Hub.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
from huggingface_hub import HfApi

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import issue778_upload as upload


def test_verify_prefix_passes_hfapi_to_list_repo_files_complete(monkeypatch):
    """The first positional handed to the library MUST be an HfApi instance.

    Pre-fix the call omitted it (``list_repo_files_complete(repo_id, ...)``),
    binding ``repo_id`` to the ``api`` slot; post-fix an ``HfApi`` leads.
    """
    captured: dict = {}

    def _fake(api, repo_id, *, repo_type="model", revision=None):
        captured["api"] = api
        captured["repo_id"] = repo_id
        captured["repo_type"] = repo_type
        captured["revision"] = revision
        return [f"{repo_id_prefix}/a" for repo_id_prefix in ["foo"]]

    monkeypatch.setattr(upload, "list_repo_files_complete", _fake)

    upload._verify_prefix("owner/repo", "model", "foo", min_files=1)

    assert isinstance(captured["api"], HfApi), (
        f"first positional must be an HfApi instance, got {type(captured['api']).__name__}"
    )
    # repo_id must land in the SECOND positional, not the first (the pre-fix bug).
    assert captured["repo_id"] == "owner/repo"
    assert captured["repo_type"] == "model"
    assert captured["revision"] == "main"


def test_verify_prefix_raises_when_min_files_not_met(monkeypatch):
    """A prefix with fewer than ``min_files`` hits raises RuntimeError (fail-loud)."""
    monkeypatch.setattr(upload, "list_repo_files_complete", lambda *a, **k: ["foo/a", "foo/b"])
    with pytest.raises(RuntimeError, match="upload verify FAILED"):
        upload._verify_prefix("owner/repo", "model", "bar", min_files=1)


def test_verify_prefix_returns_hit_count(monkeypatch):
    """Returns the number of files under the prefix when the floor is met."""
    monkeypatch.setattr(upload, "list_repo_files_complete", lambda *a, **k: ["foo/a", "foo/b"])
    assert upload._verify_prefix("owner/repo", "model", "foo", min_files=1) == 2
