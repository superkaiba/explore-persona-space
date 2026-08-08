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

It ALSO pins the #1738 destination-shape guard: on the
``upload_as_file=True`` branch, ``path_in_repo`` is the FULL file
destination — a directory-prefix-shaped destination (trailing ``/``, or an
extension-less basename while the local filename carries an extension)
raises ``ValueError`` BEFORE any HF API object is constructed, instead of
silently landing the file AT the prefix (which shadows the directory and
400-blocks every later upload under it).

Reference: .claude/rules/gotchas.md (the ``hub._upload`` subsection); #640
round 1→2 (the Codex twin caught a folder-default per-file loop the Claude
code-reviewer missed); #1738 (a file landed at a directory prefix).
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


# --- #1738 destination-shape guard (directory-prefix-shaped single-file dest) ---


class _ExplodingApi:
    """Confirms a guard fires before HfApi(...) is ever constructed."""

    def __init__(self, *a, **k):
        raise AssertionError("HfApi must not be constructed — the guard fires first")


def _mock_api(monkeypatch, committed_files):
    """Patch HfApi with a call-recording mock + make upload verification pass."""
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
    monkeypatch.setattr(hub, "list_hf_files_under_path", lambda *a, **k: committed_files)
    return calls


def test_upload_file_dir_prefix_dest_raises(tmp_path, monkeypatch):
    """An extension-bearing local file sent to an extension-less destination
    (the #1738 shape) must raise BEFORE HfApi construction — and the message
    must RENDER (the corrective form's braces are literal text, not an
    f-string interpolation)."""
    f = tmp_path / "foo.json"
    f.write_text("{}")
    monkeypatch.setenv("HF_TOKEN", "fake-token")
    monkeypatch.setattr("huggingface_hub.HfApi", _ExplodingApi)

    with pytest.raises(ValueError, match="looks like a directory prefix") as excinfo:
        hub._upload(
            f,
            repo_id="x/y",
            repo_type="dataset",
            path_in_repo="some/prefix",
            upload_as_file=True,
        )

    msg = str(excinfo.value)
    assert "'some/prefix'" in msg, "message must name the offending destination"
    assert "'foo.json'" in msg, "message must name the local filename"
    # Critic flag: the corrective form contains literal braces — assert the
    # literal text survived (an f-string rendering of this segment would
    # NameError or interpolate garbage instead).
    assert "f'{prefix}/{local_path.name}'" in msg
    assert "#1738" in msg


def test_upload_file_trailing_slash_dest_raises(tmp_path, monkeypatch):
    """A trailing-slash destination is unambiguous directory intent — raise,
    before HfApi construction, regardless of raise_on_error."""
    f = tmp_path / "foo.json"
    f.write_text("{}")
    monkeypatch.setenv("HF_TOKEN", "fake-token")
    monkeypatch.setattr("huggingface_hub.HfApi", _ExplodingApi)

    with pytest.raises(ValueError, match="looks like a directory prefix"):
        hub._upload(
            f,
            repo_id="x/y",
            repo_type="dataset",
            path_in_repo="some/prefix/",
            upload_as_file=True,
            raise_on_error=False,  # pre-try placement: propagates anyway
        )


def test_upload_file_full_dest_with_filename_passes_guard(tmp_path, monkeypatch):
    """The documented correct form — a full destination ending in the
    filename — must NOT trip the #1738 guard."""
    f = tmp_path / "foo.json"
    f.write_text("{}")
    monkeypatch.setenv("HF_TOKEN", "fake-token")
    calls = _mock_api(monkeypatch, ["some/prefix/foo.json"])

    out = hub._upload(
        f,
        repo_id="x/y",
        repo_type="dataset",
        path_in_repo="some/prefix/foo.json",
        upload_as_file=True,
    )

    assert calls["upload_file"][0]["path_in_repo"] == "some/prefix/foo.json"
    assert not calls["upload_folder"]
    assert out == "x/y/some/prefix/foo.json"


def test_upload_file_empty_dest_passes_guard(tmp_path, monkeypatch):
    """Empty path_in_repo falls back to the local filename (documented
    contract) — the #1738 guard must not fire on it."""
    f = tmp_path / "foo.json"
    f.write_text("{}")
    monkeypatch.setenv("HF_TOKEN", "fake-token")
    calls = _mock_api(monkeypatch, ["foo.json"])

    hub._upload(
        f,
        repo_id="x/y",
        repo_type="dataset",
        path_in_repo="",
        upload_as_file=True,
    )

    assert calls["upload_file"][0]["path_in_repo"] == "foo.json"


def test_upload_extensionless_file_to_bare_prefix_passes_guard(tmp_path, monkeypatch):
    """Deliberate residual false negative: an extension-less LOCAL file
    (LICENSE-class) to an extension-less destination is indistinguishable
    from a legitimate rename — the guard stays silent by design."""
    f = tmp_path / "LICENSE"
    f.write_text("MIT")
    monkeypatch.setenv("HF_TOKEN", "fake-token")
    calls = _mock_api(monkeypatch, ["some/prefix"])

    hub._upload(
        f,
        repo_id="x/y",
        repo_type="dataset",
        path_in_repo="some/prefix",
        upload_as_file=True,
    )

    assert calls["upload_file"][0]["path_in_repo"] == "some/prefix"


def test_upload_file_renamed_dest_basename_passes_guard(tmp_path, monkeypatch):
    """A destination basename that carries an extension but differs from the
    local name (a rename) is a full file destination — no raise."""
    f = tmp_path / "foo.json"
    f.write_text("{}")
    monkeypatch.setenv("HF_TOKEN", "fake-token")
    calls = _mock_api(monkeypatch, ["some/prefix/renamed.jsonl"])

    out = hub._upload(
        f,
        repo_id="x/y",
        repo_type="dataset",
        path_in_repo="some/prefix/renamed.jsonl",
        upload_as_file=True,
    )

    assert calls["upload_file"][0]["path_in_repo"] == "some/prefix/renamed.jsonl"
    assert out == "x/y/some/prefix/renamed.jsonl"
