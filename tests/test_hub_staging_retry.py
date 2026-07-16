"""Tests for #1402 — hub staging-download retry + the LocalEntryNotFoundError class arm.

Pins (a) `_is_transient_upload_error` classifying `LocalEntryNotFoundError`
transient BY CLASS (checked first; a response-bearing 404 EntryNotFoundError
stays non-transient), and (b) the fail-loud atomic staging helpers
`stage_hub_file` / `stage_hub_prefix`.

Monkeypatch targets are PER-NAME (plan #1402 §4): `hf_hub_download` / `HfApi`
are function-body lazy imports in the helpers (kept lazy, mirroring hub.py
module style), so they are patched at `huggingface_hub.<name>` — patching
`hub.<name>` would be a no-op. `stage_hub_file` and `list_hf_files_under_path`
ARE hub-module globals and patch at the hub site.

Until this branch merges, run with ``PYTHONPATH=<worktree>/src`` so the
worktree's ``explore_persona_space`` (which carries the new helpers) shadows
the editable install pointing at main.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from explore_persona_space.orchestrate import hub


@pytest.fixture(autouse=True)
def fast_retries(monkeypatch):
    """No real sleeps + attempt-bound retry (budget 0 => 6 calls max, #735)."""
    monkeypatch.setattr(hub.time, "sleep", lambda s: None)
    monkeypatch.setenv("EPM_HF_RETRY_BUDGET_S", "0")


# ---------------------------------------------------------------------------
# Classifier arm
# ---------------------------------------------------------------------------


def test_local_entry_not_found_classified_transient():
    """Transient BY CLASS: the message deliberately contains NO transient
    substring, so only the isinstance arm can classify it."""
    from huggingface_hub.errors import LocalEntryNotFoundError

    err = LocalEntryNotFoundError("cannot locate entry")
    assert hub._is_transient_upload_error(err) is True


def test_local_entry_not_found_offline_message_transient():
    """The offline-flavored message (previously non-transient via substrings)
    is now transient — the accepted bounded caveat (#1402 docstring)."""
    from huggingface_hub.errors import LocalEntryNotFoundError

    err = LocalEntryNotFoundError("outgoing traffic has been disabled")
    assert hub._is_transient_upload_error(err) is True


def test_real_404_with_response_still_non_transient():
    """A genuinely-missing file surfaces as a response-BEARING 404
    EntryNotFoundError — still fail-fast (NOT a LocalEntryNotFoundError
    instance: its constructor nulls the response by design)."""
    import requests
    from huggingface_hub.errors import EntryNotFoundError

    resp = requests.Response()
    resp.status_code = 404
    err = EntryNotFoundError("404 Client Error: entry not found for url", response=resp)
    assert hub._is_transient_upload_error(err) is False


def test_retry_transient_retries_local_entry_not_found_then_success():
    from huggingface_hub.errors import LocalEntryNotFoundError

    calls = {"n": 0}

    def thunk():
        calls["n"] += 1
        if calls["n"] < 3:
            raise LocalEntryNotFoundError("cannot locate entry")
        return "ok"

    assert hub.retry_transient(thunk, what="test-thunk") == "ok"
    assert calls["n"] == 3


# ---------------------------------------------------------------------------
# stage_hub_file
# ---------------------------------------------------------------------------


def test_stage_hub_file_atomic_and_retried(tmp_path, monkeypatch):
    from huggingface_hub.errors import LocalEntryNotFoundError

    calls = {"n": 0}

    def fake_hf_hub_download(
        *, repo_id, filename, repo_type=None, revision=None, local_dir=None, token=None
    ):
        calls["n"] += 1
        if calls["n"] == 1:
            raise LocalEntryNotFoundError("cannot locate entry")
        p = Path(local_dir) / filename
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(b"payload")
        return str(p)

    monkeypatch.setattr("huggingface_hub.hf_hub_download", fake_hf_hub_download)

    target = tmp_path / "dest" / "issueX" / "file.json"
    out = hub.stage_hub_file("org/data", "issueX/file.json", target)
    assert out == target
    assert target.read_bytes() == b"payload"
    assert calls["n"] == 2
    # atomic publish: no leftover staging tempdirs beside the target
    assert list(target.parent.glob(".hfstage-*")) == []


def test_stage_hub_file_stages_inside_dest_parent(tmp_path, monkeypatch):
    """The #1335 EXDEV pin: the staging tempdir lives INSIDE target.parent
    (same filesystem), so os.replace can never cross devices."""
    target = tmp_path / "dest" / "file.json"

    def fake_hf_hub_download(
        *, repo_id, filename, repo_type=None, revision=None, local_dir=None, token=None
    ):
        td = Path(local_dir)
        assert td.resolve().parent == target.parent.resolve()
        assert td.name.startswith(".hfstage-")
        p = td / filename
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(b"x")
        return str(p)

    monkeypatch.setattr("huggingface_hub.hf_hub_download", fake_hf_hub_download)

    out = hub.stage_hub_file("org/data", "file.json", target)
    assert out == target
    assert target.read_bytes() == b"x"


def test_stage_hub_file_idempotent_skips_existing(tmp_path, monkeypatch):
    def fake_hf_hub_download(**kwargs):
        raise AssertionError("network call on an existing target")

    monkeypatch.setattr("huggingface_hub.hf_hub_download", fake_hf_hub_download)

    target = tmp_path / "file.json"
    target.write_bytes(b"already staged")
    out = hub.stage_hub_file("org/data", "file.json", target)
    assert out == target
    assert target.read_bytes() == b"already staged"


def test_stage_hub_file_fail_loud_on_exhaustion(tmp_path, monkeypatch):
    """Exhaustion RAISES (never the fail-soft "" contract of download_dataset)
    and leaves no partial target; budget 0 => the #735 6-call attempt floor."""
    from huggingface_hub.errors import LocalEntryNotFoundError

    calls = {"n": 0}

    def fake_hf_hub_download(**kwargs):
        calls["n"] += 1
        raise LocalEntryNotFoundError("cannot locate entry")

    monkeypatch.setattr("huggingface_hub.hf_hub_download", fake_hf_hub_download)

    target = tmp_path / "dest" / "file.json"
    with pytest.raises(LocalEntryNotFoundError):
        hub.stage_hub_file("org/data", "file.json", target)
    assert calls["n"] == 6
    assert not target.exists()


# ---------------------------------------------------------------------------
# stage_hub_prefix
# ---------------------------------------------------------------------------


def test_stage_hub_prefix_scoped_listing_and_revision_pin(tmp_path, monkeypatch):
    seen: dict = {"repo_info_calls": 0}

    class FakeApi:
        def __init__(self, token=None):
            seen["token"] = token

        def repo_info(self, repo_id, repo_type=None):
            seen["repo_info_calls"] += 1
            return SimpleNamespace(sha="abc123")

    monkeypatch.setattr("huggingface_hub.HfApi", FakeApi)

    def fake_list(api, repo_id, path, *, repo_type="model", revision=None):
        # scoped: receives the PREFIX, never a full listing
        seen["list_args"] = (repo_id, path, repo_type, revision)
        return ["pfx/a.json", "pfx/sub/b.json"]

    monkeypatch.setattr(hub, "list_hf_files_under_path", fake_list)

    staged: list[tuple[str, str, str | None]] = []

    def fake_stage(
        repo_id,
        path_in_repo,
        target,
        *,
        repo_type="dataset",
        revision=None,
        token=None,
        overwrite=False,
    ):
        staged.append((path_in_repo, str(target), revision))
        return Path(target)

    monkeypatch.setattr(hub, "stage_hub_file", fake_stage)

    dest = tmp_path / "dest"
    out = hub.stage_hub_prefix("org/data", "pfx", dest)

    assert seen["repo_info_calls"] == 1  # revision resolved ONCE
    assert seen["list_args"] == ("org/data", "pfx", "dataset", "abc123")
    # the resolved sha threads into every per-file call (one snapshot)
    assert [rev for (_, _, rev) in staged] == ["abc123", "abc123"]
    # verbatim prefix mirror: dest_dir/<repo-relative path>
    assert out == [dest / "pfx/a.json", dest / "pfx/sub/b.json"]

    # empty listing raises (fail-loud)
    monkeypatch.setattr(hub, "list_hf_files_under_path", lambda *a, **k: [])
    with pytest.raises(FileNotFoundError):
        hub.stage_hub_prefix("org/data", "pfx", dest)


def test_stage_hub_prefix_per_file_failure_propagates(tmp_path, monkeypatch):
    """A per-file failure PROPAGATES through stage_hub_prefix (fail-loud) —
    no partial result list is returned (#1402 §1 criterion 2)."""

    class FakeApi:
        def __init__(self, token=None):
            pass

        def repo_info(self, repo_id, repo_type=None):
            return SimpleNamespace(sha="abc123")

    monkeypatch.setattr("huggingface_hub.HfApi", FakeApi)
    monkeypatch.setattr(
        hub,
        "list_hf_files_under_path",
        lambda *a, **k: ["pfx/a.json", "pfx/b.json", "pfx/c.json"],
    )

    def fake_stage(
        repo_id,
        path_in_repo,
        target,
        *,
        repo_type="dataset",
        revision=None,
        token=None,
        overwrite=False,
    ):
        if path_in_repo == "pfx/b.json":
            raise RuntimeError("per-file staging failed")
        return Path(target)

    monkeypatch.setattr(hub, "stage_hub_file", fake_stage)

    with pytest.raises(RuntimeError, match="per-file staging failed"):
        hub.stage_hub_prefix("org/data", "pfx", tmp_path / "dest")
