"""Pins the #591 hub._upload regression (2026-06-11 pod-smoke 504 incident).

hub._upload's post-upload verification used a RECURSIVE WHOLE-REPO tree walk
once per uploaded file with no 5xx retry; HF 504'd that walk at ~1-in-30 calls
on the large data repo and crashed the pod smoke twice. The fix: single-path
verification (``HfApi.file_exists`` for file uploads, a tree listing SCOPED to
the uploaded prefix for folder uploads) plus bounded exponential-backoff retry
on TRANSIENT errors only (5xx / timeout / connection — never 4xx). Public
signature and success/failure semantics are unchanged: ``""`` on failure,
``"{repo_id}/{path_in_repo}"`` on verified success.
"""

from unittest.mock import MagicMock, patch

import pytest
import requests
from huggingface_hub.hf_api import RepoFile, RepoFolder
from huggingface_hub.utils import HfHubHTTPError

from explore_persona_space.orchestrate import hub

REPO = "owner/data-repo"


def _http_error(status: int) -> HfHubHTTPError:
    resp = requests.Response()
    resp.status_code = status
    return HfHubHTTPError(f"HTTP {status}", response=resp)


def _repo_file() -> RepoFile:
    return RepoFile.__new__(RepoFile)  # isinstance-true without required init fields


def _repo_folder() -> RepoFolder:
    return RepoFolder.__new__(RepoFolder)


@pytest.fixture()
def env_token(monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "hf_test_token")


@pytest.fixture()
def no_sleep():
    with patch.object(hub.time, "sleep") as mock_sleep:
        yield mock_sleep


@pytest.fixture()
def api():
    """Mock HfApi instance wired into hub._upload's lazy import."""
    mock_api = MagicMock()
    with patch("huggingface_hub.HfApi", return_value=mock_api):
        yield mock_api


def _upload_file(tmp_path, path_in_repo="bucket/cell.json"):
    f = tmp_path / "cell.json"
    f.write_text("{}")
    return hub._upload(
        local_path=f,
        repo_id=REPO,
        repo_type="dataset",
        path_in_repo=path_in_repo,
        upload_as_file=True,
    )


class TestTransientRetry:
    def test_verify_504_then_success(self, tmp_path, env_token, api, no_sleep):
        """A transient 504 on verification retries and succeeds."""
        api.file_exists.side_effect = [_http_error(504), True]
        result = _upload_file(tmp_path)
        assert result == f"{REPO}/bucket/cell.json"
        assert api.file_exists.call_count == 2
        assert no_sleep.call_count == 1

    def test_upload_call_504_then_success(self, tmp_path, env_token, api, no_sleep):
        """A transient 504 on the upload call itself retries and succeeds."""
        api.upload_file.side_effect = [_http_error(504), None]
        api.file_exists.return_value = True
        result = _upload_file(tmp_path)
        assert result == f"{REPO}/bucket/cell.json"
        assert api.upload_file.call_count == 2

    def test_persistent_504_exhausts_and_fails(self, tmp_path, env_token, api, no_sleep):
        """4 transient failures exhaust the retry budget -> '' (caller raises)."""
        api.upload_file.return_value = None
        api.file_exists.side_effect = [_http_error(504)] * 4
        result = _upload_file(tmp_path)
        assert result == ""  # semantics preserved: dispatcher's fail-loud raise fires
        assert api.file_exists.call_count == 4  # bounded: exactly `attempts`
        assert no_sleep.call_count == 3  # no sleep after the final attempt

    def test_4xx_never_retried(self, tmp_path, env_token, api, no_sleep):
        """403 (quota/auth) fails immediately — retrying 4xx only delays the signal."""
        api.upload_file.side_effect = _http_error(403)
        result = _upload_file(tmp_path)
        assert result == ""
        assert api.upload_file.call_count == 1
        no_sleep.assert_not_called()

    def test_connection_error_is_transient(self, tmp_path, env_token, api, no_sleep):
        api.upload_file.side_effect = [requests.exceptions.ConnectionError("reset"), None]
        api.file_exists.return_value = True
        assert _upload_file(tmp_path) == f"{REPO}/bucket/cell.json"
        assert api.upload_file.call_count == 2


class TestSinglePathVerification:
    def test_file_upload_uses_file_exists_not_tree_walk(self, tmp_path, env_token, api):
        """File-upload verification is ONE file_exists call on the uploaded path."""
        api.file_exists.return_value = True
        result = _upload_file(tmp_path)
        assert result == f"{REPO}/bucket/cell.json"
        api.file_exists.assert_called_once_with(REPO, "bucket/cell.json", repo_type="dataset")
        api.list_repo_tree.assert_not_called()  # never the whole-repo walk

    def test_file_missing_after_upload_returns_empty(self, tmp_path, env_token, api):
        api.file_exists.return_value = False
        assert _upload_file(tmp_path) == ""

    def test_folder_verification_scoped_to_prefix(self, tmp_path, env_token, api):
        """Folder verification lists ONLY the uploaded prefix (recursive)."""
        d = tmp_path / "adapter_dir"
        d.mkdir()
        (d / "adapter_config.json").write_text("{}")
        api.list_repo_tree.return_value = [_repo_file(), _repo_folder(), _repo_file()]
        result = hub._upload(
            local_path=d,
            repo_id=REPO,
            repo_type="model",
            path_in_repo="adapters/x",
        )
        assert result == f"{REPO}/adapters/x"
        api.list_repo_tree.assert_called_once_with(
            REPO, path_in_repo="adapters/x", recursive=True, repo_type="model"
        )

    def test_folder_prefix_absent_returns_empty(self, tmp_path, env_token, api):
        """EntryNotFoundError (404 subclass) -> 0 committed files, no retry, ''."""
        from huggingface_hub.utils import EntryNotFoundError

        d = tmp_path / "adapter_dir"
        d.mkdir()
        (d / "adapter_config.json").write_text("{}")
        err_resp = requests.Response()
        err_resp.status_code = 404
        api.list_repo_tree.side_effect = EntryNotFoundError("missing", response=err_resp)
        result = hub._upload(
            local_path=d, repo_id=REPO, repo_type="model", path_in_repo="adapters/x"
        )
        assert result == ""
        assert api.list_repo_tree.call_count == 1


class TestTransientClassifier:
    @pytest.mark.parametrize(
        "status,expected",
        [
            (500, True),
            (502, True),
            (504, True),
            (529, True),
            (400, False),
            (401, False),
            (403, False),
            (404, False),
            (429, False),
        ],
    )
    def test_status_codes(self, status, expected):
        assert hub._is_transient_hub_error(_http_error(status)) is expected

    def test_timeout_and_connection(self):
        assert hub._is_transient_hub_error(requests.exceptions.ConnectTimeout())
        assert hub._is_transient_hub_error(requests.exceptions.ConnectionError())

    def test_generic_exception_not_transient(self):
        assert not hub._is_transient_hub_error(ValueError("boom"))
