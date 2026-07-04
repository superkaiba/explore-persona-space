"""Tests for orchestrate/hub.py — upload/download utilities."""

import json
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import pytest
from huggingface_hub.hf_api import RepoFile
from huggingface_hub.utils import HfHubHTTPError
from requests.exceptions import ConnectionError

from explore_persona_space.orchestrate.hub import (
    DEFAULT_DATASET_REPO,
    DEFAULT_MODEL_REPO,
    _is_storage_quota_403,
    _is_transient_upload_error,
    _retry_upload,
    list_repo_files_complete,
    upload_dataset,
    upload_dataset_directory,
    upload_model,
    upload_raw_completions_to_data_repo,
)


def _http_err(code: int, msg: str | None = None) -> HfHubHTTPError:
    """Build an HfHubHTTPError whose .response.status_code == code, mirroring a
    real HF upload HTTP error so the status-code branch is exercised as in prod."""
    r = Mock()
    r.status_code = code
    return HfHubHTTPError(msg or f"{code} error", response=r)


class TestUploadModel:
    """Tests for upload_model — HF Hub model uploads."""

    def test_skips_without_hf_token(self):
        """Should return empty string and skip when HF_TOKEN not set."""
        with patch.dict("os.environ", {}, clear=True):
            result = upload_model("/nonexistent/path")
        assert result == ""

    def test_skips_nonexistent_path(self, tmp_path):
        """Should return empty string for non-existent model path."""
        with patch.dict("os.environ", {"HF_TOKEN": "test_token"}):
            result = upload_model(str(tmp_path / "nonexistent"))
        assert result == ""

    def test_default_repo_ids(self):
        """Default repo IDs should be set."""
        assert DEFAULT_MODEL_REPO == "superkaiba1/explore-persona-space"
        assert DEFAULT_DATASET_REPO == "superkaiba1/explore-persona-space-data"

    def test_path_in_repo_default(self):
        """Default path_in_repo should be '{condition_name}_seed{seed}'."""
        # We can't actually upload, but we can verify the logic
        with tempfile.TemporaryDirectory() as tmpdir:
            model_dir = Path(tmpdir) / "model"
            model_dir.mkdir()
            (model_dir / "config.json").write_text("{}")

            with (
                patch.dict("os.environ", {"HF_TOKEN": "test_token"}),
                patch("huggingface_hub.HfApi") as MockApi,
            ):
                mock_api = MockApi.return_value
                mock_api.create_repo.return_value = None
                mock_api.upload_folder.return_value = None

                upload_model(
                    str(model_dir),
                    condition_name="evil_wrong",
                    seed=42,
                )

                # Should have called upload_folder with the right path
                mock_api.upload_folder.assert_called_once()
                call_kwargs = mock_api.upload_folder.call_args[1]
                assert call_kwargs["path_in_repo"] == "evil_wrong_seed42"


class TestUploadDataset:
    """Tests for upload_dataset — HF Hub dataset uploads."""

    def test_skips_without_hf_token(self):
        """Should return empty string without HF_TOKEN."""
        with patch.dict("os.environ", {}, clear=True):
            result = upload_dataset("/nonexistent/path")
        assert result == ""

    def test_skips_nonexistent_path(self):
        """Should return empty string for non-existent data path."""
        with patch.dict("os.environ", {"HF_TOKEN": "test_token"}):
            result = upload_dataset("/nonexistent/data.jsonl")
        assert result == ""

    def test_upload_file(self):
        """Should call upload_file for a single file."""
        with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False, mode="w") as f:
            json.dump({"test": True}, f)
            f.flush()
            fpath = f.name

        try:
            with (
                patch.dict("os.environ", {"HF_TOKEN": "test_token"}),
                patch("huggingface_hub.HfApi") as MockApi,
            ):
                mock_api = MockApi.return_value
                mock_api.create_repo.return_value = None
                mock_api.upload_file.return_value = None
                # Post-upload verification now walks the paginated tree
                # (list_repo_files_complete -> list_repo_tree) instead of
                # repo_info().siblings, so the committed file must surface there.
                mock_api.list_repo_tree.return_value = [
                    RepoFile(path="test/data.jsonl", size=1, blob_id="b", oid="o")
                ]

                result = upload_dataset(fpath, path_in_repo="test/data.jsonl")
                mock_api.upload_file.assert_called_once()
                assert "test/data.jsonl" in result
        finally:
            Path(fpath).unlink()


class TestUploadDatasetDirectory:
    """Tests for upload_dataset_directory — shared helper introduced in #293 §3."""

    def test_no_upload_skips_network_io(self, tmp_path):
        """no_upload=True returns [] without calling upload_dataset."""
        # Create one file so the helper would otherwise try to upload it.
        (tmp_path / "x.jsonl").write_text('{"a": 1}\n')
        with patch("explore_persona_space.orchestrate.hub.upload_dataset") as mock_upload:
            mock_upload.side_effect = AssertionError(
                "upload_dataset must NOT be called when no_upload=True"
            )
            result = upload_dataset_directory(tmp_path, bucket="test/", no_upload=True)
        assert result == []
        mock_upload.assert_not_called()

    def test_empty_dir_returns_empty_list(self, tmp_path):
        """An empty directory returns [] without raising."""
        with patch("explore_persona_space.orchestrate.hub.upload_dataset") as mock_upload:
            result = upload_dataset_directory(tmp_path, bucket="test/")
        assert result == []
        mock_upload.assert_not_called()

    def test_fail_loud_default_reraises(self, tmp_path):
        """Default (fail_soft=False) re-raises on upload error."""
        (tmp_path / "x.jsonl").write_text('{"a": 1}\n')
        with patch("explore_persona_space.orchestrate.hub.upload_dataset") as mock_upload:
            mock_upload.side_effect = RuntimeError("boom")
            with pytest.raises(RuntimeError, match="boom"):
                upload_dataset_directory(tmp_path, bucket="test/")

    def test_fail_soft_continues(self, tmp_path):
        """fail_soft=True keeps going after a failure on file 1 of 2."""
        (tmp_path / "a.jsonl").write_text('{"a": 1}\n')
        (tmp_path / "b.jsonl").write_text('{"b": 2}\n')

        call_count = {"n": 0}

        def _maybe_fail(*args, **kwargs):
            call_count["n"] += 1
            if call_count["n"] == 1:
                raise RuntimeError("file 1 failed")
            return f"repo/{kwargs['path_in_repo']}"

        with patch(
            "explore_persona_space.orchestrate.hub.upload_dataset",
            side_effect=_maybe_fail,
        ):
            result = upload_dataset_directory(tmp_path, bucket="test/", fail_soft=True)
        # File 2 still attempted and reported as uploaded.
        assert call_count["n"] == 2
        # Helper returns only successful uploads (file 1 failed).
        assert result == ["test/b.jsonl"]

    def test_fail_loud_when_upload_dataset_returns_empty_string(self, tmp_path):
        """Default (fail_soft=False) raises RuntimeError on '' return.

        Regression test for #293 round-2 C1: upload_dataset returns '' on every
        internal failure (token missing, 401, 403, verification mismatch). The
        helper used to ignore the return value and silently succeed. After the
        fix, the helper must raise so the caller exits non-zero.
        """
        (tmp_path / "x.jsonl").write_text('{"a": 1}\n')
        with (
            patch(
                "explore_persona_space.orchestrate.hub.upload_dataset",
                return_value="",
            ),
            pytest.raises(RuntimeError, match="upload_dataset returned '' for") as exc_info,
        ):
            upload_dataset_directory(tmp_path, bucket="test/")
        # Error message must point to the specific file + path_in_repo so the
        # caller can find which file failed.
        assert "x.jsonl" in str(exc_info.value)
        assert "test/x.jsonl" in str(exc_info.value)

    def test_fail_soft_when_upload_dataset_returns_empty_string(self, tmp_path, capsys):
        """fail_soft=True logs to stderr, skips the failed file, returns survivors.

        Regression sister-test for #293 round-2 C1: in soft mode, the helper
        must (1) NOT raise, (2) log to stderr, (3) skip the failed file from
        the returned list, (4) still attempt the next file.
        """
        (tmp_path / "a.jsonl").write_text('{"a": 1}\n')
        (tmp_path / "b.jsonl").write_text('{"b": 2}\n')

        # File 1 returns "" (silent failure), file 2 returns a real path.
        call_count = {"n": 0}

        def _maybe_empty(*args, **kwargs):
            call_count["n"] += 1
            if call_count["n"] == 1:
                return ""
            return f"repo/{kwargs['path_in_repo']}"

        with patch(
            "explore_persona_space.orchestrate.hub.upload_dataset",
            side_effect=_maybe_empty,
        ):
            result = upload_dataset_directory(tmp_path, bucket="test/", fail_soft=True)
        captured = capsys.readouterr()

        # File 2 still attempted (helper does not abort on first failure).
        assert call_count["n"] == 2
        # Returned list contains ONLY the successful upload.
        assert result == ["test/b.jsonl"]
        # Failure must surface on stderr so a human watching logs can see it.
        assert "returned ''" in captured.err
        assert "a.jsonl" in captured.err
        # And the explicit fail_soft notice.
        assert "fail_soft=True; continuing" in captured.err

    def test_literal_filename_with_brackets_glob_escaped(self, tmp_path):
        """A literal filename containing ``[]`` is glob-escaped (v3 P7)."""
        # Create a file whose name contains glob metacharacters.
        target = tmp_path / "data_[v1].jsonl"
        target.write_text('{"a": 1}\n')
        # Sibling that should NOT be picked up.
        (tmp_path / "other.jsonl").write_text('{"b": 2}\n')

        seen: list[str] = []

        def _record(data_path: str, path_in_repo: str = "") -> str:
            seen.append(Path(data_path).name)
            return f"repo/{path_in_repo}"

        with patch(
            "explore_persona_space.orchestrate.hub.upload_dataset",
            side_effect=_record,
        ):
            result = upload_dataset_directory(
                tmp_path,
                bucket="test/",
                pattern="data_[v1].jsonl",
            )
        assert seen == ["data_[v1].jsonl"], (
            "literal-filename pattern should match exactly that file once "
            "glob.escape is applied; otherwise [v1] is interpreted as a "
            "character class and matches nothing"
        )
        assert result == ["test/data_[v1].jsonl"]


class TestUploadRawCompletions:
    """Tests for upload_raw_completions_to_data_repo — the #664/#727 refactor.

    The function must upload the whole matched raw-completions tree in ONE
    ``upload_folder`` commit (never a per-file ``upload_file`` loop, which
    504-storms on a large repo), preserve the per-file return-dict contract,
    skip aggregate JSONs, fail loud on incomplete verification, and only delete
    the raw-completions files (never the aggregate) under ``delete_after``.
    """

    EXP = "issue727_demo"

    def _build_tree(self, root: Path) -> list[str]:
        """Create raw_completions.json at THREE depths + an aggregate JSON
        that must NOT be uploaded. Returns the expected committed repo paths."""
        # depth 0 (top-level), depth 2, depth 3
        (root / "raw_completions.json").write_text('{"d0": 1}')
        nested_a = root / "cellA" / "T_seed42"
        nested_a.mkdir(parents=True)
        (nested_a / "raw_completions.json").write_text('{"d2": 1}')
        nested_b = root / "cellB" / "sub" / "C_seed7"
        nested_b.mkdir(parents=True)
        (nested_b / "raw_completions.json").write_text('{"d3": 1}')
        # An aggregate JSON the allow_patterns must skip.
        (root / "run_result.json").write_text('{"agg": true}')
        prefix = f"{self.EXP}/raw_completions"
        return [
            f"{prefix}/raw_completions.json",
            f"{prefix}/cellA/T_seed42/raw_completions.json",
            f"{prefix}/cellB/sub/C_seed7/raw_completions.json",
        ]

    def test_uses_upload_folder_not_per_file_loop(self, tmp_path):
        """ONE upload_folder commit, ZERO upload_file calls, allow_patterns
        matches only raw-completions, return dict has one entry per file."""
        expected = self._build_tree(tmp_path)
        with (
            patch.dict("os.environ", {"HF_TOKEN": "test_token"}),
            patch("huggingface_hub.HfApi") as MockApi,
            patch(
                "explore_persona_space.orchestrate.hub.list_repo_files_complete",
                return_value=expected,  # whole expected set present -> verified
            ),
        ):
            mock_api = MockApi.return_value
            mock_api.create_repo.return_value = None
            mock_api.upload_folder.return_value = None

            result = upload_raw_completions_to_data_repo(self.EXP, tmp_path)

        # Exactly ONE bulk commit, NO per-file uploads.
        mock_api.upload_folder.assert_called_once()
        mock_api.upload_file.assert_not_called()
        # allow_patterns selects only raw-completions at every depth.
        call_kwargs = mock_api.upload_folder.call_args[1]
        assert call_kwargs["allow_patterns"] == [
            "raw_completions.json",
            "**/raw_completions.json",
        ]
        assert call_kwargs["path_in_repo"] == f"{self.EXP}/raw_completions"
        # Return dict: one entry per raw-completions file, correct URL, no
        # aggregate JSON entry.
        assert set(result.keys()) == {
            "raw_completions.json",
            "cellA/T_seed42/raw_completions.json",
            "cellB/sub/C_seed7/raw_completions.json",
        }
        rel = "cellA/T_seed42/raw_completions.json"
        assert result[rel] == f"{DEFAULT_DATASET_REPO}/{self.EXP}/raw_completions/{rel}"

    def test_empty_returns_empty_dict_with_warning(self, tmp_path):
        """No matching files -> {} and NO upload call (early return)."""
        (tmp_path / "run_result.json").write_text('{"agg": true}')
        with (
            patch.dict("os.environ", {"HF_TOKEN": "test_token"}),
            patch("huggingface_hub.HfApi") as MockApi,
            patch("explore_persona_space.orchestrate.hub.list_repo_files_complete") as mock_list,
        ):
            result = upload_raw_completions_to_data_repo(self.EXP, tmp_path)
        assert result == {}
        MockApi.return_value.upload_folder.assert_not_called()
        mock_list.assert_not_called()

    def test_verification_failure_raises(self, tmp_path):
        """An incomplete committed set (one expected file missing) -> the
        EXACT-set verify fails -> RuntimeError (fail-loud preserved)."""
        expected = self._build_tree(tmp_path)
        incomplete = expected[:-1]  # drop one expected file from the listing
        with (
            patch.dict("os.environ", {"HF_TOKEN": "test_token"}),
            patch("huggingface_hub.HfApi") as MockApi,
            patch(
                "explore_persona_space.orchestrate.hub.list_repo_files_complete",
                return_value=incomplete,
            ),
            pytest.raises(RuntimeError, match="bulk folder upload failed"),
        ):
            MockApi.return_value.upload_folder.return_value = None
            upload_raw_completions_to_data_repo(self.EXP, tmp_path)

    def test_delete_after_removes_only_raw_files(self, tmp_path):
        """delete_after=True removes the raw_completions.json files but leaves
        the aggregate JSON (only matched files are deleted, never the dir)."""
        expected = self._build_tree(tmp_path)
        with (
            patch.dict("os.environ", {"HF_TOKEN": "test_token"}),
            patch("huggingface_hub.HfApi") as MockApi,
            patch(
                "explore_persona_space.orchestrate.hub.list_repo_files_complete",
                return_value=expected,
            ),
        ):
            MockApi.return_value.upload_folder.return_value = None
            upload_raw_completions_to_data_repo(self.EXP, tmp_path, delete_after=True)

        # Every raw_completions.json is gone.
        assert not (tmp_path / "raw_completions.json").exists()
        assert not (tmp_path / "cellA" / "T_seed42" / "raw_completions.json").exists()
        assert not (tmp_path / "cellB" / "sub" / "C_seed7" / "raw_completions.json").exists()
        # The aggregate JSON the allow_patterns skipped is untouched.
        assert (tmp_path / "run_result.json").exists()


def _storage_403() -> HfHubHTTPError:
    """The persistent account-wide public-storage 403 (must NOT retry — it has to
    re-raise immediately so #564 overflow-routing / soft-fail fires unchanged)."""
    return _http_err(403, "403 Forbidden: You have exceeded your public storage space")


class TestRetryUpload:
    """Tests for the shared HF-uploader retry wrapper (#735).

    ``_retry_upload`` retries a transient HF 5xx/429/timeout/connection error via
    exp-backoff, re-raises a storage-quota-403 (and any non-transient error)
    immediately, and re-raises the final exception after ``max_attempts``
    (fail-loud, no swallow). ``hub.time.sleep`` is patched in every retry test so
    the suite stays fast.
    """

    def test_retries_504_then_succeeds(self):
        """A transient 504 on attempt 1, then success on attempt 2."""
        thunk = Mock(side_effect=[_http_err(504), "ok"])
        with patch("explore_persona_space.orchestrate.hub.time.sleep") as mock_sleep:
            assert _retry_upload(thunk, what="t") == "ok"
        assert thunk.call_count == 2
        mock_sleep.assert_called_once()

    def test_retries_429_then_succeeds(self):
        """A transient 429 (rate limit) on attempt 1, then success."""
        thunk = Mock(side_effect=[_http_err(429), "ok"])
        with patch("explore_persona_space.orchestrate.hub.time.sleep"):
            assert _retry_upload(thunk, what="t") == "ok"
        assert thunk.call_count == 2

    def test_connection_error_retried(self):
        """A bare ConnectionError (.response is None) is caught by the
        message-substring arm ('connection'), then succeeds."""
        thunk = Mock(side_effect=[ConnectionError("connection reset"), "ok"])
        with patch("explore_persona_space.orchestrate.hub.time.sleep"):
            assert _retry_upload(thunk, what="t") == "ok"
        assert thunk.call_count == 2

    def test_403_storage_quota_not_retried(self):
        """The persistent storage-quota-403 re-raises IMMEDIATELY (attempt 1) so
        the caller's #564 overflow-routing / soft-fail fires unchanged."""
        thunk = Mock(side_effect=_storage_403())
        with (
            patch("explore_persona_space.orchestrate.hub.time.sleep") as mock_sleep,
            pytest.raises(HfHubHTTPError),
        ):
            _retry_upload(thunk, what="t")
        assert thunk.call_count == 1
        mock_sleep.assert_not_called()

    def test_non_transient_404_not_retried(self):
        """A non-transient 404 re-raises immediately (not in the transient set,
        no transient substring) — attempt 1 only."""
        thunk = Mock(side_effect=_http_err(404))
        with (
            patch("explore_persona_space.orchestrate.hub.time.sleep"),
            pytest.raises(HfHubHTTPError),
        ):
            _retry_upload(thunk, what="t")
        assert thunk.call_count == 1

    def test_non_storage_403_not_retried(self):
        """A NON-storage 403 (auth / gated-repo) must NOT retry-loop: status 403
        is not in the transient code set and the message has no transient
        substring, so it re-raises on attempt 1 (reconciler-suggested hardening)."""
        thunk = Mock(side_effect=_http_err(403, "403 Forbidden: gated repo access"))
        with (
            patch("explore_persona_space.orchestrate.hub.time.sleep"),
            pytest.raises(HfHubHTTPError),
        ):
            _retry_upload(thunk, what="t")
        assert thunk.call_count == 1

    def test_exhausts_and_reraises(self):
        """A transient error on EVERY attempt re-raises the final exception after
        max_attempts (fail-loud, no swallow) — 6 calls, 5 sleeps."""
        thunk = Mock(side_effect=[_http_err(504)] * 6)
        with (
            patch("explore_persona_space.orchestrate.hub.time.sleep") as mock_sleep,
            pytest.raises(HfHubHTTPError),
        ):
            _retry_upload(thunk, what="t")
        assert thunk.call_count == 6
        assert mock_sleep.call_count == 5

    def test_value_error_not_retried(self):
        """A non-HTTP error (ValueError) re-raises immediately — attempt 1."""
        thunk = Mock(side_effect=ValueError("bad args"))
        with (
            patch("explore_persona_space.orchestrate.hub.time.sleep"),
            pytest.raises(ValueError, match="bad args"),
        ):
            _retry_upload(thunk, what="t")
        assert thunk.call_count == 1

    def test_predicate_storage_quota_403(self):
        """_is_storage_quota_403 requires BOTH '403' and 'storage' in the message."""
        assert _is_storage_quota_403(_storage_403()) is True
        assert _is_storage_quota_403(_http_err(403, "403 Forbidden: gated repo")) is False
        assert _is_storage_quota_403(_http_err(504)) is False

    def test_predicate_transient(self):
        """_is_transient_upload_error: 5xx/429 by status code; ConnectionError by
        substring; storage-403 / ValueError are NOT transient."""
        assert _is_transient_upload_error(_http_err(504)) is True
        assert _is_transient_upload_error(_http_err(429)) is True
        assert _is_transient_upload_error(ConnectionError("connection reset")) is True
        assert _is_transient_upload_error(_http_err(404)) is False
        assert _is_transient_upload_error(_storage_403()) is False
        assert _is_transient_upload_error(ValueError("bad args")) is False

    def test_4xx_digit_triplet_in_message_not_retried(self):
        """A 404 whose MESSAGE embeds a digit triplet ('issue504_raw') must NOT
        retry: a real 4xx status code decides non-transient BEFORE the fuzzy
        substring scan can false-match on message digits (#989)."""
        err = _http_err(404, "404 Client Error: Not Found for url: .../issue504_raw/final/x.json")
        assert _is_transient_upload_error(err) is False
        thunk = Mock(side_effect=err)
        with (
            patch("explore_persona_space.orchestrate.hub.time.sleep") as mock_sleep,
            pytest.raises(HfHubHTTPError),
        ):
            _retry_upload(thunk, what="t")
        assert thunk.call_count == 1
        mock_sleep.assert_not_called()

    def test_413_byte_count_digits_not_retried(self):
        """A 413 whose message embeds '500' inside a byte count must NOT retry
        (the byte-count digit trap, #989)."""
        err = _http_err(413, "413 Payload Too Large: 15000000000 bytes exceeds limit")
        assert _is_transient_upload_error(err) is False
        thunk = Mock(side_effect=err)
        with (
            patch("explore_persona_space.orchestrate.hub.time.sleep"),
            pytest.raises(HfHubHTTPError),
        ):
            _retry_upload(thunk, what="t")
        assert thunk.call_count == 1

    def test_5xx_transient_by_code_full_range(self):
        """Any 5xx is transient BY CODE — pinned at the inclusive lower endpoint
        (500, i.e. ``500 <= code`` not ``500 < code``) and at representative
        codes outside the old (500, 502, 503, 504) tuple (507, 520)."""
        assert _is_transient_upload_error(_http_err(500)) is True
        assert _is_transient_upload_error(_http_err(507)) is True
        assert _is_transient_upload_error(_http_err(520)) is True
        thunk = Mock(side_effect=[_http_err(520), "ok"])
        with patch("explore_persona_space.orchestrate.hub.time.sleep"):
            assert _retry_upload(thunk, what="t") == "ok"
        assert thunk.call_count == 2

    def test_408_request_timeout_transient_by_code(self):
        """Coded 408 Request Timeout stays transient BY CODE (RFC 9110 §15.5.9
        invites the client to repeat) — previously retried only via the
        'timeout' substring accident; the #989 tightening must preserve it."""
        assert _is_transient_upload_error(_http_err(408, "408 Request Timeout")) is True
        thunk = Mock(side_effect=[_http_err(408, "408 Request Timeout"), "ok"])
        with patch("explore_persona_space.orchestrate.hub.time.sleep"):
            assert _retry_upload(thunk, what="t") == "ok"
        assert thunk.call_count == 2

    def test_code_wins_over_substring_and_isinstance_guard(self):
        """(a) a real 4xx code OVERRIDES a transient-looking substring
        ('connection'); (b) a response-less TimeoutError keeps the substring
        path; (c) a non-int status_code (the STRING '500') never enters the
        code branch (isinstance guard) and falls to the substring scan."""
        assert _is_transient_upload_error(_http_err(400, "connection header malformed")) is False
        assert _is_transient_upload_error(TimeoutError("Read timed out")) is True
        r = Mock()
        r.status_code = "500"
        str_code_err = HfHubHTTPError("opaque failure", response=r)
        assert _is_transient_upload_error(str_code_err) is False

    def test_upload_folder_branch_uses_retry(self):
        """Integration: a 504 on the FIRST _upload folder commit then success ->
        _upload returns the non-empty URL and upload_folder is called twice
        (confirms the call-site wiring, not just the helper in isolation)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            model_dir = Path(tmpdir) / "model"
            model_dir.mkdir()
            (model_dir / "config.json").write_text("{}")

            with (
                patch.dict("os.environ", {"HF_TOKEN": "test_token"}),
                patch("huggingface_hub.HfApi") as MockApi,
                patch("explore_persona_space.orchestrate.hub.time.sleep"),
            ):
                mock_api = MockApi.return_value
                mock_api.create_repo.return_value = None
                mock_api.upload_folder.side_effect = [_http_err(504), None]
                # Post-upload verification walks the paginated tree.
                mock_api.list_repo_tree.return_value = [
                    RepoFile(path="evil_wrong_seed42/config.json", size=2, blob_id="b", oid="o")
                ]

                result = upload_model(
                    str(model_dir),
                    condition_name="evil_wrong",
                    seed=42,
                )

            assert mock_api.upload_folder.call_count == 2
            assert "evil_wrong_seed42" in result


def _repo_files(*paths: str) -> list[RepoFile]:
    """Build a list of RepoFile entries for the given paths (helper for the
    list-retry tests — mirrors what a good ``list_repo_tree`` page yields)."""
    return [RepoFile(path=p, size=1, blob_id="b", oid="o") for p in paths]


class TestListRepoFilesRetry:
    """Tests for ``list_repo_files_complete``'s transient-504 retry (#794).

    The paginated ``list_repo_tree`` walk is wrapped in ``_retry_upload`` so a
    504 on any cursor page of a large repo retries instead of turning a
    successful upload's post-upload verify into a false failure. Non-transient
    errors (404 / auth) still re-raise on attempt 1 — reads must never mask a
    real fault. ``hub.time.sleep`` is patched so the suite stays fast.
    """

    def test_list_repo_files_complete_retries_transient_504(self):
        """A cursor-page 504 on attempt 1, then the generator yields entries on
        attempt 2 -> the sorted file list is returned, list_repo_tree called
        twice (a full re-list per attempt), one sleep."""
        api = Mock()
        api.list_repo_tree = Mock(side_effect=[_http_err(504), _repo_files("b/x.json", "a/y.json")])
        with patch("explore_persona_space.orchestrate.hub.time.sleep") as mock_sleep:
            result = list_repo_files_complete(api, "some/repo")
        assert result == ["a/y.json", "b/x.json"]
        assert api.list_repo_tree.call_count == 2
        mock_sleep.assert_called_once()

    def test_list_repo_files_complete_retries_429(self):
        """A transient 429 (rate limit) on attempt 1, then success."""
        api = Mock()
        api.list_repo_tree = Mock(side_effect=[_http_err(429), _repo_files("f.json")])
        with patch("explore_persona_space.orchestrate.hub.time.sleep"):
            result = list_repo_files_complete(api, "some/repo")
        assert result == ["f.json"]
        assert api.list_repo_tree.call_count == 2

    def test_list_repo_files_complete_exhausts_retries(self):
        """A 504 on EVERY attempt re-raises the final exception after
        max_attempts (fail-loud, no swallow) — 6 calls, 5 sleeps — so a
        genuinely persistent gateway outage still surfaces."""
        api = Mock()
        api.list_repo_tree = Mock(side_effect=[_http_err(504)] * 6)
        with (
            patch("explore_persona_space.orchestrate.hub.time.sleep") as mock_sleep,
            pytest.raises(HfHubHTTPError),
        ):
            list_repo_files_complete(api, "some/repo")
        assert api.list_repo_tree.call_count == 6
        assert mock_sleep.call_count == 5

    def test_list_repo_files_complete_non_transient_raises_immediately(self):
        """A non-transient 404 re-raises on attempt 1 (no retry loop) — reads
        must not mask a real fault (missing repo / auth)."""
        api = Mock()
        api.list_repo_tree = Mock(side_effect=_http_err(404))
        with (
            patch("explore_persona_space.orchestrate.hub.time.sleep") as mock_sleep,
            pytest.raises(HfHubHTTPError),
        ):
            list_repo_files_complete(api, "some/repo")
        assert api.list_repo_tree.call_count == 1
        mock_sleep.assert_not_called()

    def test_list_repo_files_complete_success_passthrough_filters_non_files(self):
        """Happy-path passthrough: on the first attempt the generator yields a
        mix of file + non-file entries; ``list_repo_files_complete`` returns the
        sorted file paths only (non-files filtered by ``isinstance(entry,
        RepoFile)``), calls ``list_repo_tree`` exactly ONCE, and never sleeps —
        the unchanged success path costs no retries."""
        non_file = Mock(spec=[])  # a non-RepoFile entry (folder / other) — must be filtered out
        entries = [*_repo_files("b/x.json", "a/y.json"), non_file]
        api = Mock()
        api.list_repo_tree = Mock(return_value=iter(entries))
        with patch("explore_persona_space.orchestrate.hub.time.sleep") as mock_sleep:
            result = list_repo_files_complete(api, "some/repo")
        assert result == ["a/y.json", "b/x.json"]  # sorted files only; non-file dropped
        assert api.list_repo_tree.call_count == 1
        mock_sleep.assert_not_called()
