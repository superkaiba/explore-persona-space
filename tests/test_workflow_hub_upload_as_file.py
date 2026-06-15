"""Workflow-invariant test pinning hub._upload's file-vs-folder semantics.

Background: ``hub._upload`` (src/explore_persona_space/orchestrate/hub.py)
raises ``ValueError`` UNCONDITIONALLY when handed a FILE path without
``upload_as_file=True``. The guard exists because
``huggingface_hub.upload_folder`` silently no-ops on a single-file path
(uploads NOTHING but logs a benign-looking "is not a directory. Keeping
local path." warning, while verification can still pass if same-prefix
files already exist) — a silent data-loss class (#595). A per-file upload
loop using the folder-default form crashes on the FIRST file; the correct
form passes ``upload_as_file=True``.

This pins the deliberate guard + the file-routing branch as workflow
invariants so a future ``hub.py`` refactor cannot silently regress either.

Reference: .claude/rules/gotchas.md (the ``hub._upload`` subsection); #640
round 1→2 (the Codex twin caught a folder-default per-file loop the Claude
code-reviewer missed).
"""

from __future__ import annotations

import pytest

from explore_persona_space.orchestrate import hub


def test_upload_file_without_upload_as_file_raises(tmp_path, monkeypatch):
    """A file path with the folder-default (upload_as_file=False) must raise
    BEFORE any HF API object is constructed."""
    f = tmp_path / "x.json"
    f.write_text("{}")
    monkeypatch.setenv("HF_TOKEN", "fake-token")  # past the no-token early return

    class _ExplodingApi:  # confirms the guard fires before HfApi(...) is touched
        def __init__(self, *a, **k):
            raise AssertionError("HfApi must not be constructed — the guard fires first")

    monkeypatch.setattr("huggingface_hub.HfApi", _ExplodingApi)

    with pytest.raises(ValueError, match="upload_as_file"):
        hub._upload(f, repo_id="x/y", repo_type="dataset", path_in_repo="z.json")


def test_upload_file_with_upload_as_file_routes_to_upload_file(tmp_path, monkeypatch):
    """A file path with upload_as_file=True must route to HfApi.upload_file,
    never upload_folder."""
    f = tmp_path / "x.json"
    f.write_text("{}")
    monkeypatch.setenv("HF_TOKEN", "fake-token")

    calls: dict[str, list] = {"upload_file": [], "upload_folder": []}

    class _MockApi:
        def __init__(self, *a, **k):
            pass

        def create_repo(self, *a, **k):
            return None

        def upload_file(self, **kw):
            calls["upload_file"].append(kw)
            return None

        def upload_folder(self, **kw):
            calls["upload_folder"].append(kw)
            return None

    monkeypatch.setattr("huggingface_hub.HfApi", _MockApi)
    # Make verification pass without hitting the Hub: the file landed.
    monkeypatch.setattr(hub, "list_repo_files_complete", lambda *a, **k: ["z.json"])

    out = hub._upload(
        f, repo_id="x/y", repo_type="dataset", path_in_repo="z.json", upload_as_file=True
    )

    assert calls["upload_file"], "upload_as_file=True must route to HfApi.upload_file"
    assert not calls["upload_folder"], "a single-file upload must NOT route to upload_folder"
    assert calls["upload_file"][0]["path_or_fileobj"] == str(f)
    assert calls["upload_file"][0]["path_in_repo"] == "z.json"
    assert out == "x/y/z.json"
