# ruff: noqa: E402
"""Reactive file-count-limit overflow fallback in ``hub._upload`` (#1108).

The canonical HF model repo sits at the 100,000-files-per-repo hard limit
(#1090: "Your git repo would contain 100050 files after this push, over the
limit of 100000 files"). ``_upload`` now retries a REJECTED model-repo upload
against the private overflow repo (``DEFAULT_OVERFLOW_REPO``), emits the #564
routing event with ``reason="file-count-limit-reactive"``, and writes the
``OVERFLOW_POINTER.json`` breadcrumb on the canonical repo — default ON, kill
switch ``EPM_HF_FILECOUNT_FALLBACK=0``.

These tests execute the REAL ``_upload`` body end-to-end (the #906
one-production-body rule): the only fake is the external HfApi network
boundary, and the fake's method ``def``s mirror the exact call shapes hub.py
uses (signature-conformant by construction — a drifted call site raises
TypeError instead of silently passing).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

import huggingface_hub
import pytest
from huggingface_hub.hf_api import RepoFile

from explore_persona_space.orchestrate import hub

# The verbatim #1090 rejection (full format confirmed by HF forum thread 26400).
FILECOUNT_MSG = (
    "Your git repo would contain 100050 files after this push, over the limit of 100000 files."
)


class _HubRejection(Exception):
    """Message-carrying stand-in for the (unverified-class) HF rejection."""

    def __init__(self, msg: str, status_code: int | None = None):
        super().__init__(msg)
        if status_code is not None:
            # Shape mirrors requests' err.response.status_code read in
            # hub._is_transient_upload_error.
            class _Resp:
                pass

            resp = _Resp()
            resp.status_code = status_code
            resp.headers = {}
            self.response = resp


def _rf(path: str) -> RepoFile:
    return RepoFile(path=path, size=1, blob_id="b", oid="o")


class FakeHfApi:
    """Signature-conformant HfApi fake for the ``_upload`` network boundary.

    ``folder_errors`` maps repo_id -> Exception raised by ``upload_folder``;
    ``tree`` maps repo_id -> RepoFile list returned by the scoped verify walk.
    Every call is recorded in ``calls`` as ``(method, repo_id, detail)``.
    """

    def __init__(self):
        self.folder_errors: dict[str, Exception] = {}
        self.tree: dict[str, list[RepoFile]] = {}
        self.calls: list[tuple[str, str, str]] = []

    # --- factory shim: hub code calls huggingface_hub.HfApi(token=...) ---
    def __call__(self, token=None):
        return self

    # --- HfApi surface (defs mirror hub.py's exact call shapes) ---
    def create_repo(self, repo_id, *, repo_type=None, private=False, exist_ok=False):
        self.calls.append(("create_repo", repo_id, f"private={private}"))

    def upload_folder(self, *, folder_path, repo_id, path_in_repo, repo_type, ignore_patterns=None):
        self.calls.append(("upload_folder", repo_id, path_in_repo))
        err = self.folder_errors.get(repo_id)
        if err is not None:
            raise err

    def upload_file(self, *, path_or_fileobj, repo_id, path_in_repo, repo_type):
        self.calls.append(("upload_file", repo_id, path_in_repo))

    def list_repo_tree(
        self, *, repo_id, repo_type=None, revision=None, recursive=False, path_in_repo=None
    ):
        self.calls.append(("list_repo_tree", repo_id, str(path_in_repo)))
        return iter(self.tree.get(repo_id, []))

    def file_exists(self, repo_id, path, *, repo_type=None, revision=None):
        self.calls.append(("file_exists", repo_id, path))
        return False

    # --- assertion helpers ---
    def n_calls(self, method: str, repo_id: str) -> int:
        return sum(1 for m, r, _ in self.calls if m == method and r == repo_id)


DEST = "adapters/issue1108_test/final_adapter"


@pytest.fixture
def rig(tmp_path, monkeypatch):
    """Local dir + fake api + hermetic env (no network, event sink in tmp)."""
    src = tmp_path / "adapter_dir"
    src.mkdir()
    (src / "adapter_model.safetensors").write_bytes(b"\x00" * 16)
    event_path = tmp_path / "overflow-events.jsonl"
    monkeypatch.setenv("HF_TOKEN", "t")
    monkeypatch.setenv("EPM_HF_STORAGE_CHECK", "0")  # emitters probe headroom: keep offline
    monkeypatch.setenv("EPM_HF_OVERFLOW_EVENT_PATH", str(event_path))
    monkeypatch.delenv("EPM_HF_FILECOUNT_FALLBACK", raising=False)
    api = FakeHfApi()
    monkeypatch.setattr(huggingface_hub, "HfApi", api)
    return src, api, event_path


def _events(event_path: Path) -> list[dict]:
    if not event_path.exists():
        return []
    return [json.loads(line) for line in event_path.open(encoding="utf-8") if line.strip()]


class TestReroute:
    def test_filecount_rejection_reroutes_event_and_pointer(self, rig):
        src, api, event_path = rig
        api.folder_errors[hub.DEFAULT_MODEL_REPO] = _HubRejection(FILECOUNT_MSG)
        api.tree[hub.DEFAULT_OVERFLOW_REPO] = [_rf(f"{DEST}/adapter_model.safetensors")]

        result = hub._upload(src, hub.DEFAULT_MODEL_REPO, "model", DEST)

        assert result == f"{hub.DEFAULT_OVERFLOW_REPO}/{DEST}"
        # overflow repo created PRIVATE, canonical attempted first
        assert ("create_repo", hub.DEFAULT_OVERFLOW_REPO, "private=True") in api.calls
        assert api.n_calls("upload_folder", hub.DEFAULT_MODEL_REPO) == 1
        assert api.n_calls("upload_folder", hub.DEFAULT_OVERFLOW_REPO) == 1
        # routing event carries the reactive reason
        rows = _events(event_path)
        assert len(rows) == 1, rows
        assert rows[0]["reason"] == "file-count-limit-reactive"
        assert rows[0]["original_repo"] == hub.DEFAULT_MODEL_REPO
        assert rows[0]["effective_repo"] == hub.DEFAULT_OVERFLOW_REPO
        # pointer breadcrumb attempted against the CANONICAL repo
        pointer_calls = [
            (m, r, p)
            for m, r, p in api.calls
            if m == "upload_file" and p.endswith("OVERFLOW_POINTER.json")
        ]
        assert pointer_calls == [
            ("upload_file", hub.DEFAULT_MODEL_REPO, f"{DEST}/OVERFLOW_POINTER.json")
        ]

    def test_kill_switch_off_restores_legacy_empty_return(self, rig, monkeypatch):
        src, api, event_path = rig
        monkeypatch.setenv("EPM_HF_FILECOUNT_FALLBACK", "0")
        api.folder_errors[hub.DEFAULT_MODEL_REPO] = _HubRejection(FILECOUNT_MSG)

        result = hub._upload(src, hub.DEFAULT_MODEL_REPO, "model", DEST)

        assert result == ""
        assert api.n_calls("upload_folder", hub.DEFAULT_OVERFLOW_REPO) == 0
        assert _events(event_path) == []

    def test_overflow_repo_rejection_no_recursion(self, rig):
        """The guard short-circuits when the target IS the overflow repo."""
        src, api, event_path = rig
        api.folder_errors[hub.DEFAULT_OVERFLOW_REPO] = _HubRejection(FILECOUNT_MSG)

        result = hub._upload(src, hub.DEFAULT_OVERFLOW_REPO, "model", DEST)

        assert result == ""
        assert api.n_calls("upload_folder", hub.DEFAULT_OVERFLOW_REPO) == 1  # no recursion
        assert _events(event_path) == []

    def test_dataset_repo_type_not_rerouted(self, rig):
        src, api, event_path = rig
        api.folder_errors[hub.DEFAULT_DATASET_REPO] = _HubRejection(FILECOUNT_MSG)

        result = hub._upload(src, hub.DEFAULT_DATASET_REPO, "dataset", "bucket/x")

        assert result == ""
        assert api.n_calls("upload_folder", hub.DEFAULT_OVERFLOW_REPO) == 0
        assert _events(event_path) == []


class TestNonMatchingErrors:
    @pytest.mark.parametrize(
        "err",
        [
            _HubRejection("403 Forbidden: You have exceeded your public storage space"),
            # Real HF 4xx rejections carry a response status code, which decides
            # transience BY CODE (no substring scan — "25000" would otherwise
            # false-read as a transient "500", the #989 digit-triplet trap).
            _HubRejection(
                "400 Bad Request: this directory is over the limit of 10000 files per folder",
                status_code=400,
            ),
            _HubRejection(
                "400 Bad Request: a commit cannot contain more than 25000 operations; "
                "split your push",
                status_code=400,
            ),
        ],
        ids=["storage-403", "per-folder-cap", "per-commit-op-cap"],
    )
    def test_non_matching_error_no_reroute(self, rig, err):
        src, api, event_path = rig
        api.folder_errors[hub.DEFAULT_MODEL_REPO] = err

        result = hub._upload(src, hub.DEFAULT_MODEL_REPO, "model", DEST)

        assert result == ""
        assert api.n_calls("upload_folder", hub.DEFAULT_OVERFLOW_REPO) == 0
        assert _events(event_path) == []

    def test_transient_504_exhausts_then_no_reroute(self, rig, monkeypatch):
        src, api, event_path = rig
        monkeypatch.setenv("EPM_HF_RETRY_BUDGET_S", "0")  # attempt-bound only
        monkeypatch.setattr(hub.time, "sleep", lambda s: None)
        api.folder_errors[hub.DEFAULT_MODEL_REPO] = _HubRejection(
            "504 Gateway Time-out", status_code=504
        )

        result = hub._upload(src, hub.DEFAULT_MODEL_REPO, "model", DEST)

        assert result == ""
        assert api.n_calls("upload_folder", hub.DEFAULT_MODEL_REPO) > 1  # retried
        assert api.n_calls("upload_folder", hub.DEFAULT_OVERFLOW_REPO) == 0
        assert _events(event_path) == []


class TestDeleteAfter:
    def test_local_reaped_only_after_verified_overflow_landing(self, rig):
        src, api, _ = rig
        api.folder_errors[hub.DEFAULT_MODEL_REPO] = _HubRejection(FILECOUNT_MSG)
        api.tree[hub.DEFAULT_OVERFLOW_REPO] = [_rf(f"{DEST}/adapter_model.safetensors")]

        result = hub._upload(src, hub.DEFAULT_MODEL_REPO, "model", DEST, delete_after=True)

        assert result == f"{hub.DEFAULT_OVERFLOW_REPO}/{DEST}"
        assert not src.exists()  # reaped by the RECURSIVE call, post-verify

    def test_overflow_verify_miss_keeps_local_intact(self, rig):
        """Overflow upload 'succeeds' but the scoped verify finds 0 files:
        no reap, no event, no pointer, empty return."""
        src, api, event_path = rig
        api.folder_errors[hub.DEFAULT_MODEL_REPO] = _HubRejection(FILECOUNT_MSG)
        api.tree[hub.DEFAULT_OVERFLOW_REPO] = []  # verify walk resolves nothing

        result = hub._upload(src, hub.DEFAULT_MODEL_REPO, "model", DEST, delete_after=True)

        assert result == ""
        assert src.exists() and (src / "adapter_model.safetensors").exists()
        assert _events(event_path) == []
        assert not any(p.endswith("OVERFLOW_POINTER.json") for _, _, p in api.calls)

    def test_overflow_upload_failure_keeps_local_intact(self, rig):
        src, api, event_path = rig
        api.folder_errors[hub.DEFAULT_MODEL_REPO] = _HubRejection(FILECOUNT_MSG)
        api.folder_errors[hub.DEFAULT_OVERFLOW_REPO] = _HubRejection("500 upstream exploded")

        # 500 is transient: keep the retry loop fast + attempt-bound.
        import unittest.mock as _m

        with _m.patch.object(hub.time, "sleep", lambda s: None):
            import os

            os.environ["EPM_HF_RETRY_BUDGET_S"] = "0"
            try:
                result = hub._upload(src, hub.DEFAULT_MODEL_REPO, "model", DEST, delete_after=True)
            finally:
                os.environ.pop("EPM_HF_RETRY_BUDGET_S", None)

        assert result == ""
        assert src.exists()
        assert _events(event_path) == []


class TestPredicates:
    @pytest.mark.parametrize(
        ("msg", "expected"),
        [
            (FILECOUNT_MSG, True),
            # lowercase / prefix-wrapped variants of the same server phrase
            (
                "BadRequest: your git repo would contain 100001 files after this push, "
                "over the limit of 100000 files",
                True,
            ),
            ("429 too many requests", False),
            ("403 Forbidden: You have exceeded your public storage space", False),
            # digit-triplet paths must not read as a limit rejection (#989 family)
            ("upload of issue504_raw/part100000.json failed: connection reset", False),
            # per-FOLDER cap: has "over the limit of"+"files" but no "push"
            ("this directory is over the limit of 10000 files per folder", False),
            # per-commit operation cap: has "push" but not "over the limit of"
            ("a commit cannot contain more than 25000 operations; split your push", False),
        ],
    )
    def test_is_file_count_limit_error(self, msg, expected):
        assert hub._is_file_count_limit_error(Exception(msg)) is expected

    def test_fallback_enabled_default_on(self, monkeypatch):
        monkeypatch.delenv("EPM_HF_FILECOUNT_FALLBACK", raising=False)
        assert hub._filecount_fallback_enabled() is True
        monkeypatch.setenv("EPM_HF_FILECOUNT_FALLBACK", "0")
        assert hub._filecount_fallback_enabled() is False
        monkeypatch.setenv("EPM_HF_FILECOUNT_FALLBACK", "1")
        assert hub._filecount_fallback_enabled() is True
