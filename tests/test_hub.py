"""Tests for orchestrate/hub.py — upload/download utilities."""

import json
import logging
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import huggingface_hub
import pytest
from huggingface_hub.hf_api import RepoFile
from huggingface_hub.utils import EntryNotFoundError, HfHubHTTPError
from requests.exceptions import ConnectionError

from explore_persona_space.orchestrate import hub
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
        max_attempts (fail-loud, no swallow) — 6 calls, 5 sleeps. ``budget_s=0``
        pins the legacy attempt-bound contract (#735) under the #997 wall-clock
        budget kill switch."""
        thunk = Mock(side_effect=[_http_err(504)] * 6)
        with (
            patch("explore_persona_space.orchestrate.hub.time.sleep") as mock_sleep,
            pytest.raises(HfHubHTTPError),
        ):
            _retry_upload(thunk, what="t", budget_s=0)
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

    def test_predicate_queue_size_reached_response_less_transient(self):
        """#1315/#1360: the HF/Xet upload-queue-saturation body text ('maximum
        queue size reached') classifies transient when response-LESS (the #931
        PyO3-boundary shape); a response-BEARING 4xx carrying the same phrase
        stays non-transient — the decision is made ENTIRELY by status code
        (the #989 code-wins-over-substring guard intact)."""
        err = RuntimeError(
            "Data processing error: CAS service error ... maximum queue size reached"
        )
        assert _is_transient_upload_error(err) is True
        assert _is_transient_upload_error(_http_err(400, "queue size reached")) is False

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

    def test_list_repo_files_complete_exhausts_retries(self, monkeypatch):
        """A 504 on EVERY attempt re-raises the final exception after
        max_attempts (fail-loud, no swallow) — 6 calls, 5 sleeps — so a
        genuinely persistent gateway outage still surfaces.
        ``list_repo_files_complete`` invokes ``_retry_upload`` with DEFAULT
        kwargs, so the legacy attempt bound is pinned via the
        ``EPM_HF_RETRY_BUDGET_S=0`` env kill switch (#997)."""
        monkeypatch.setenv("EPM_HF_RETRY_BUDGET_S", "0")
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


# ---------------------------------------------------------------------------
# #1360 — _upload's verify leg rides _retry_upload (the #1315 p11 429 escape)
# ---------------------------------------------------------------------------


class _VerifyRetryApi:
    """Signature-conformant HfApi fake for the ``_upload`` verify-leg tests.

    Mirrors hub.py's exact call shapes (the #906 one-production-body rule —
    the REAL ``_upload`` body runs end to end; the fake sits only at the
    network boundary): ``upload_file`` succeeds, the scoped tree walk 404s on
    the exact-file path (``EntryNotFoundError``), and ``file_exists`` raises
    its scripted transport errors in order before returning
    ``file_exists_result``.
    """

    def __init__(self, *, file_exists_raises=None, file_exists_result=True):
        self._file_exists_raises = list(file_exists_raises or [])
        self.file_exists_result = file_exists_result
        self.upload_file_calls = 0
        self.file_exists_calls = 0

    def __call__(self, token=None):  # factory shim: hub calls HfApi(token=...)
        return self

    def create_repo(self, repo_id, *, repo_type=None, private=False, exist_ok=False):
        pass

    def upload_file(self, *, path_or_fileobj, repo_id, path_in_repo, repo_type):
        self.upload_file_calls += 1

    def upload_folder(self, *, folder_path, repo_id, path_in_repo, repo_type, ignore_patterns=None):
        raise AssertionError("folder branch must not fire on a single-file upload")

    def list_repo_tree(
        self, *, repo_id, repo_type=None, revision=None, recursive=False, path_in_repo=None
    ):
        raise EntryNotFoundError(f"entry {path_in_repo} not found")

    def file_exists(self, repo_id, path, *, repo_type=None, revision=None):
        self.file_exists_calls += 1
        if self._file_exists_raises:
            raise self._file_exists_raises.pop(0)
        return self.file_exists_result


def _queue_429() -> HfHubHTTPError:
    """The #1315 p11 kill shape: an HTTP 429 whose body carries the HF/Xet
    upload-queue-saturation text."""
    return _http_err(
        429, "429 Client Error: Too Many Requests for url: ... maximum queue size reached"
    )


class TestUploadVerifyTransportRetry:
    """#1360 integration: ``_upload``'s verify leg (the exact-file
    ``file_exists`` fallback in ``list_hf_files_under_path``) rides
    ``_retry_upload``, exercised through the REAL ``_upload`` body with HfApi
    faked at the network boundary (the test_hub_filecount_fallback.py
    pattern). Return contract preserved: "" only after retry exhaustion."""

    DEST = "issue1360_test/run_config.json"
    REPO = "owner/data-repo"

    @pytest.fixture
    def src_file(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HF_TOKEN", "t")
        f = tmp_path / "run_config.json"
        f.write_text("{}")
        return f

    def test_upload_single_file_returns_path_after_verify_429_then_success(
        self, src_file, monkeypatch
    ):
        """upload_file succeeds; the tree walk 404s (exact-file path); the
        file_exists probe 429s once then True -> the verified path returns
        (pre-fix: the 429 propagated to _upload's log-and-return-\"\" arm)."""
        api = _VerifyRetryApi(file_exists_raises=[_queue_429()], file_exists_result=True)
        monkeypatch.setattr(huggingface_hub, "HfApi", api)
        with patch("explore_persona_space.orchestrate.hub.time.sleep") as mock_sleep:
            result = hub._upload(src_file, self.REPO, "dataset", self.DEST, upload_as_file=True)
        assert result == f"{self.REPO}/{self.DEST}"
        assert api.upload_file_calls == 1
        assert api.file_exists_calls == 2
        mock_sleep.assert_called_once()

    def test_upload_returns_empty_only_after_verify_retry_exhaustion(self, src_file, monkeypatch):
        """A PERSISTENT verify 429 under the budget kill switch (=0) exhausts
        the 6-call attempt floor, then — and only then — _upload returns the
        no-path signal callers fail-fast on."""
        monkeypatch.setenv("EPM_HF_RETRY_BUDGET_S", "0")
        api = _VerifyRetryApi(file_exists_raises=[_queue_429() for _ in range(6)])
        monkeypatch.setattr(huggingface_hub, "HfApi", api)
        with patch("explore_persona_space.orchestrate.hub.time.sleep"):
            result = hub._upload(src_file, self.REPO, "dataset", self.DEST, upload_as_file=True)
        assert result == ""
        assert api.file_exists_calls == 6

    def test_file_path_without_flag_valueerror_unretried(self, src_file, monkeypatch):
        """The #595 fail-loud guard (file path without upload_as_file=True)
        still propagates un-retried with zero sleeps — content classes are
        byte-unchanged by the #1360 wrap."""
        api = _VerifyRetryApi()
        monkeypatch.setattr(huggingface_hub, "HfApi", api)
        with (
            patch("explore_persona_space.orchestrate.hub.time.sleep") as mock_sleep,
            pytest.raises(ValueError, match="upload_as_file"),
        ):
            hub._upload(src_file, self.REPO, "dataset", self.DEST)
        assert api.upload_file_calls == 0
        mock_sleep.assert_not_called()


# ---------------------------------------------------------------------------
# #988 — scoped post-upload verifies + list_hub_datasets prefix dispatch
# ---------------------------------------------------------------------------


def _rf(path: str) -> RepoFile:
    return RepoFile(path=path, size=1, blob_id="b", oid="o")


class TestUploadVerifyScoped:
    """#988 site 8: ``_upload``'s post-upload verify is SCOPED to
    ``expected_prefix`` — never a full-repo listing."""

    def test_file_upload_verify_uses_file_exists_fallback(self, tmp_path):
        """An exact-file dest 404s on the tree endpoint (EntryNotFoundError)
        and resolves via ONE file_exists probe."""

        from explore_persona_space.orchestrate.hub import _upload

        f = tmp_path / "x.json"
        f.write_text("{}")
        with (
            patch.dict("os.environ", {"HF_TOKEN": "t"}),
            patch("huggingface_hub.HfApi") as MockApi,
        ):
            api = MockApi.return_value
            api.list_repo_tree.side_effect = EntryNotFoundError("entry bucket/x.json not found")
            api.file_exists.return_value = True
            result = _upload(f, "org/data", "dataset", "bucket/x.json", upload_as_file=True)
        assert result == "org/data/bucket/x.json"
        # The walk was scoped server-side to the exact dest...
        assert api.list_repo_tree.call_args.kwargs["path_in_repo"] == "bucket/x.json"
        # ...and the exact-file fallback resolved it.
        api.file_exists.assert_called_once()

    def test_folder_upload_verify_scoped_walk(self, tmp_path):
        """A folder dest verifies via the scoped tree walk (path_in_repo
        threaded); no file_exists probe fires."""
        d = tmp_path / "adir"
        d.mkdir()
        (d / "a.json").write_text("{}")
        with (
            patch.dict("os.environ", {"HF_TOKEN": "t"}),
            patch("huggingface_hub.HfApi") as MockApi,
        ):
            from explore_persona_space.orchestrate.hub import _upload

            api = MockApi.return_value
            api.list_repo_tree.side_effect = lambda **kw: iter([_rf("bucket/sub/a.json")])
            result = _upload(d, "org/data", "dataset", "bucket/sub")
        assert result == "org/data/bucket/sub"
        assert api.list_repo_tree.call_args.kwargs["path_in_repo"] == "bucket/sub"
        api.file_exists.assert_not_called()

    def test_absent_dest_returns_empty_not_success(self, tmp_path):
        """An absent dest (EntryNotFoundError + file_exists False) keeps the
        existing '0 files found ... NOT marking as successful' branch."""

        from explore_persona_space.orchestrate.hub import _upload

        f = tmp_path / "x.json"
        f.write_text("{}")
        with (
            patch.dict("os.environ", {"HF_TOKEN": "t"}),
            patch("huggingface_hub.HfApi") as MockApi,
        ):
            api = MockApi.return_value
            api.list_repo_tree.side_effect = EntryNotFoundError("entry bucket/x.json not found")
            api.file_exists.return_value = False
            result = _upload(f, "org/data", "dataset", "bucket/x.json", upload_as_file=True)
        assert result == ""


class TestUploadFolderFilteredVerifyScoped:
    """#988 site 9: ``_upload_folder_filtered``'s exact-set verify is SCOPED
    to ``path_in_repo`` (every expected path is <path_in_repo>/<rel> by the
    function's contract)."""

    def test_verify_threads_path_in_repo(self, tmp_path):
        from explore_persona_space.orchestrate.hub import _upload_folder_filtered

        d = tmp_path / "src"
        d.mkdir()
        (d / "raw_completions.json").write_text("{}")
        expected = ["exp/raw/raw_completions.json"]
        with (
            patch.dict("os.environ", {"HF_TOKEN": "t"}),
            patch("huggingface_hub.HfApi") as MockApi,
        ):
            api = MockApi.return_value
            api.list_repo_tree.side_effect = lambda **kw: iter(
                [_rf("exp/raw/raw_completions.json")]
            )
            result = _upload_folder_filtered(
                d,
                "org/data",
                "dataset",
                "exp/raw",
                allow_patterns=["raw_completions.json"],
                expected_repo_paths=expected,
            )
        assert result == "org/data/exp/raw"
        assert api.list_repo_tree.call_args.kwargs["path_in_repo"] == "exp/raw"

    def test_partial_listing_still_reports_missing(self, tmp_path):
        """The exact-set check against the SCOPED listing still fails on a
        partial commit (one expected path missing -> '')."""
        from explore_persona_space.orchestrate.hub import _upload_folder_filtered

        d = tmp_path / "src"
        d.mkdir()
        (d / "a.json").write_text("{}")
        (d / "b.json").write_text("{}")
        expected = ["exp/raw/a.json", "exp/raw/b.json"]
        with (
            patch.dict("os.environ", {"HF_TOKEN": "t"}),
            patch("huggingface_hub.HfApi") as MockApi,
        ):
            api = MockApi.return_value
            # Scoped listing sees only ONE of the two expected paths.
            api.list_repo_tree.side_effect = lambda **kw: iter([_rf("exp/raw/a.json")])
            result = _upload_folder_filtered(
                d,
                "org/data",
                "dataset",
                "exp/raw",
                allow_patterns=["*.json"],
                expected_repo_paths=expected,
            )
        assert result == ""


class TestListHubDatasetsPrefixDispatch:
    """#988 site 10: prefix-shape dispatch — dir-like prefixes scope
    server-side; empty / bare-name prefixes keep the full listing (bare-name
    partial matching is load-bearing: 'dpo' must keep matching dpo_v2/...)."""

    def _run(self, path_prefix: str, files: list[str], calls: list[dict]):
        from explore_persona_space.orchestrate.hub import list_hub_datasets

        def _fake_complete(api, repo_id, **kw):
            calls.append(dict(kw))
            return list(files)

        with (
            patch.dict("os.environ", {"HF_TOKEN": "t"}),
            patch("huggingface_hub.HfApi"),
            patch(
                "explore_persona_space.orchestrate.hub.list_repo_files_complete",
                side_effect=_fake_complete,
            ),
        ):
            return list_hub_datasets(repo_id="org/data", path_prefix=path_prefix)

    def test_dir_like_prefix_scopes_server_side(self):
        calls: list[dict] = []
        result = self._run("leakage/", ["leakage/a.json", "leakage/b.json"], calls)
        assert result == ["leakage/a.json", "leakage/b.json"]
        assert len(calls) == 1
        assert calls[0].get("path_in_repo") == "leakage"

    def test_bare_name_prefix_keeps_full_listing_and_partial_match(self):
        calls: list[dict] = []
        result = self._run("dpo", ["dpo/a.json", "dpo_v2/b.json", "other/c.json"], calls)
        # Partial-name contract pinned: 'dpo' also matches dpo_v2/...
        assert result == ["dpo/a.json", "dpo_v2/b.json"]
        assert len(calls) == 1
        assert "path_in_repo" not in calls[0]

    def test_empty_prefix_full_listing(self):
        calls: list[dict] = []
        result = self._run("", ["b.json", "a.json"], calls)
        assert result == ["a.json", "b.json"]
        assert len(calls) == 1
        assert "path_in_repo" not in calls[0]

    def test_exception_returns_empty_list(self):
        from explore_persona_space.orchestrate.hub import list_hub_datasets

        with (
            patch.dict("os.environ", {"HF_TOKEN": "t"}),
            patch("huggingface_hub.HfApi"),
            patch(
                "explore_persona_space.orchestrate.hub.list_repo_files_complete",
                side_effect=RuntimeError("boom"),
            ),
        ):
            assert list_hub_datasets(repo_id="org/data", path_prefix="leakage/") == []


# ---------------------------------------------------------------------------
# #997 — wall-clock retry budget + Retry-After honoring + scoped verify helper
# ---------------------------------------------------------------------------


class _FakeClock:
    """Coupled fake clock: ``sleep`` advances the SAME source ``monotonic``
    reads (plan #997 §9b — the budget math depends on this coupling; an
    uncoupled fake would let every sleep read as zero elapsed time)."""

    def __init__(self) -> None:
        self.now = 0.0
        self.sleeps: list[float] = []

    def monotonic(self) -> float:
        return self.now

    def sleep(self, s: float) -> None:
        self.sleeps.append(s)
        self.now += s


def _rate_limited(retry_after: str | None = None) -> HfHubHTTPError:
    """A coded-429 error, optionally carrying a Retry-After header."""
    r = Mock()
    r.status_code = 429
    r.headers = {"Retry-After": retry_after} if retry_after is not None else {}
    return HfHubHTTPError("429 Too Many Requests", response=r)


def _permanent_storm(err_factory, guard: int = 200):
    """A thunk that ALWAYS raises ``err_factory()`` — with a finite call-count
    guard so a wrong-clock implementation fails loudly instead of hanging CI
    (plan #997 §5 test 3b)."""
    calls = {"n": 0}

    def thunk():
        calls["n"] += 1
        if calls["n"] > guard:
            raise AssertionError(
                f"permanent-storm fake exceeded {guard} calls — budget bound not applied"
            )
        raise err_factory()

    return thunk, calls


class TestRetryBudgetAndRetryAfter:
    """#997: ``_retry_upload`` dual bound (attempt floor OR wall-clock budget)
    + Retry-After-aware sleeps. Fake clock couples sleep -> monotonic;
    determinism via a pinned ``hub.random``."""

    def _clock(self):
        clock = _FakeClock()
        return (
            clock,
            patch("explore_persona_space.orchestrate.hub.time.monotonic", clock.monotonic),
            patch("explore_persona_space.orchestrate.hub.time.sleep", clock.sleep),
        )

    def test_retry_after_header_honored(self, monkeypatch):
        """Retry-After: 37 produces exactly a 37 s sleep (server-paced, no
        jitter on the header branch)."""
        monkeypatch.setattr(hub, "random", SimpleNamespace(random=lambda: 0.0))
        thunk = Mock(side_effect=[_rate_limited("37"), "ok"])
        clock, p_mono, p_sleep = self._clock()
        with p_mono, p_sleep:
            assert _retry_upload(thunk, what="t", budget_s=1800.0) == "ok"
        assert clock.sleeps == [37.0]
        assert thunk.call_count == 2

    def test_retry_after_capped_at_900(self):
        """A pathological Retry-After: 4000 is capped at _RETRY_AFTER_CAP_S=900."""
        thunk = Mock(side_effect=[_rate_limited("4000"), "ok"])
        clock, p_mono, p_sleep = self._clock()
        with p_mono, p_sleep:
            assert _retry_upload(thunk, what="t", budget_s=1800.0) == "ok"
        assert clock.sleeps == [900.0]

    def test_budget_survives_storm_beyond_attempt_floor(self):
        """Acceptance 1 (#931 shape): a 20-min 429 storm (Retry-After: 60) then
        success — the wall-clock budget keeps retrying PAST the 6-attempt floor
        (pre-#997 behavior: raise at attempt 6, ~310 s)."""
        thunk = Mock(side_effect=[_rate_limited("60")] * 20 + ["ok"])
        clock, p_mono, p_sleep = self._clock()
        with p_mono, p_sleep:
            assert _retry_upload(thunk, what="t", budget_s=1800.0) == "ok"
        assert thunk.call_count == 21  # > 6: the attempt floor alone would have raised
        assert clock.now == 1200.0  # 20 x 60 s, within the 1800 s budget

    def test_budget_exhaustion_bounded_fail_loud(self):
        """Acceptance 2 (short header): a PERMANENT 429 storm (Retry-After: 60)
        raises the ORIGINAL exception with total sleep <= budget — fail-loud
        stays bounded (zero-duration fake calls => elapsed == total sleep)."""
        thunk, calls = _permanent_storm(lambda: _rate_limited("60"))
        clock, p_mono, p_sleep = self._clock()
        with p_mono, p_sleep, pytest.raises(HfHubHTTPError):
            _retry_upload(thunk, what="t", budget_s=1800.0)
        assert clock.now <= 1800.0
        assert calls["n"] > 6  # the budget extended past the attempt floor...
        assert calls["n"] <= 200  # ...but the guard never tripped

    def test_budget_clamps_pathological_retry_after(self):
        """Acceptance 2, pathological header (round-1 Must-Fix — the
        discriminating pin): permanent 429 with Retry-After: 4000 under an
        1800 s budget. Every sleep — including attempt-floor retries — is
        clamped to the remaining budget, so total sleep <= 1800. The
        un-clamped OR-logic design sleeps 5 x 900 = 4500 s and FAILS here."""
        thunk, calls = _permanent_storm(lambda: _rate_limited("4000"))
        clock, p_mono, p_sleep = self._clock()
        with p_mono, p_sleep, pytest.raises(HfHubHTTPError):
            _retry_upload(thunk, what="t", budget_s=1800.0)
        assert clock.now <= 1800.0
        # 900-capped, then clamped to remaining budget; floor attempts past the
        # deadline sleep 0 and retry immediately (the #735 6-call contract).
        assert clock.sleeps == [900.0, 900.0, 0.0, 0.0, 0.0]
        assert calls["n"] == 6

    def test_budget_zero_restores_legacy_attempt_bound(self, monkeypatch):
        """Acceptance 4: ``budget_s=0`` is the legacy #735 contract — 6 calls,
        5 exp-backoff sleeps (jitter pinned to 0), final exception propagates."""
        monkeypatch.setattr(hub, "random", SimpleNamespace(random=lambda: 0.0))
        thunk, calls = _permanent_storm(lambda: _http_err(504))
        clock, p_mono, p_sleep = self._clock()
        with p_mono, p_sleep, pytest.raises(HfHubHTTPError):
            _retry_upload(thunk, what="t", budget_s=0)
        assert calls["n"] == 6
        assert clock.sleeps == [10.0, 20.0, 40.0, 80.0, 160.0]

    def test_budget_zero_caps_retry_after_at_backoff_ceiling(self):
        """Round-2 Minor: under the ``budget_s=0`` kill switch there is no
        deadline clamp, so a pathological Retry-After: 4000 would sleep
        5 x 900 s ~ 4500 s — defeating the fail-fast purpose the switch
        restores (~310 s legacy stack). The header is capped at the legacy
        180 s backoff ceiling instead."""
        thunk, calls = _permanent_storm(lambda: _rate_limited("4000"))
        clock, p_mono, p_sleep = self._clock()
        with p_mono, p_sleep, pytest.raises(HfHubHTTPError):
            _retry_upload(thunk, what="t", budget_s=0)
        assert calls["n"] == 6
        assert clock.sleeps == [180.0] * 5

    def test_backoff_jitter_upper_endpoint_capped(self, monkeypatch):
        """§9b jitter endpoint: with jitter pinned to its UPPER endpoint (1.0),
        the capped backoff sleep is 180 x 1.25 = 225 (cap applies BEFORE
        jitter), and the final sleep is clamped to the remaining budget
        (jitter-then-clamp ordering)."""
        monkeypatch.setattr(hub, "random", SimpleNamespace(random=lambda: 1.0))
        thunk, _calls = _permanent_storm(lambda: _http_err(504))
        clock, p_mono, p_sleep = self._clock()
        with p_mono, p_sleep, pytest.raises(HfHubHTTPError):
            _retry_upload(thunk, what="t", budget_s=2000.0)
        assert max(clock.sleeps) == 225.0  # min(180, base) * (1 + 0.25)
        assert clock.now <= 2000.0
        assert clock.sleeps[-1] == pytest.approx(37.5)  # clamped to remaining budget

    def test_retry_transient_is_retry_upload(self):
        """The public alias is the SAME object (#606: scripts assumed a hub
        ``_retry_transient`` that never existed)."""
        assert hub.retry_transient is hub._retry_upload

    def test_env_budget_parsed_and_unparseable_falls_back(self, monkeypatch, caplog):
        """EPM_HF_RETRY_BUDGET_S: numeric binds; unparseable falls back to 1800
        with a warning; unset/empty default 1800; negative floors at 0;
        NON-FINITE (inf/nan/1e999) falls back to 1800 (round-2 Minor: "inf"
        would make the retry loop unbounded on a permanently-down Hub, "nan"
        would silently degrade to the 0 kill switch)."""
        monkeypatch.setenv("EPM_HF_RETRY_BUDGET_S", "120")
        assert hub._retry_budget_s() == 120.0
        monkeypatch.setenv("EPM_HF_RETRY_BUDGET_S", "abc")
        with caplog.at_level(logging.WARNING, logger="explore_persona_space.orchestrate.hub"):
            assert hub._retry_budget_s() == 1800.0
        assert "EPM_HF_RETRY_BUDGET_S" in caplog.text
        monkeypatch.setenv("EPM_HF_RETRY_BUDGET_S", "")
        assert hub._retry_budget_s() == 1800.0
        monkeypatch.delenv("EPM_HF_RETRY_BUDGET_S")
        assert hub._retry_budget_s() == 1800.0
        monkeypatch.setenv("EPM_HF_RETRY_BUDGET_S", "-5")
        assert hub._retry_budget_s() == 0.0
        for nonfinite in ("inf", "-inf", "nan", "1e999"):
            monkeypatch.setenv("EPM_HF_RETRY_BUDGET_S", nonfinite)
            assert hub._retry_budget_s() == 1800.0, nonfinite

    def test_transient_message_rate_limit_markers(self):
        """Response-less rate-limit TEXT is transient (#931 xet Rust boundary);
        bare '429' digits are NOT (#989); a real 4xx code still beats a
        transient-looking body (#989 code-wins)."""
        assert _is_transient_upload_error(RuntimeError("Too Many Requests on xet-read-token")) is (
            True
        )
        assert _is_transient_upload_error(RuntimeError("request was rate limited")) is True
        # No bare-"429" substring rule: digit triplets in paths stay non-transient.
        err = RuntimeError("issue429_raw upload failed")
        assert _is_transient_upload_error(err) is False
        thunk = Mock(side_effect=err)
        with (
            patch("explore_persona_space.orchestrate.hub.time.sleep") as mock_sleep,
            pytest.raises(RuntimeError),
        ):
            _retry_upload(thunk, what="t")
        assert thunk.call_count == 1
        mock_sleep.assert_not_called()
        # Coded 400 with a rate-limit body: the code decides (non-transient).
        assert _is_transient_upload_error(_http_err(400, "400 Bad Request: too many requests")) is (
            False
        )


class TestVerifyRepoPathsUploaded:
    """#997: public scoped + retried exact-set post-upload verify (the #920
    crash class). The fake ``list_repo_tree`` records kwargs so server-side
    scoping is asserted — never a full-repo listing."""

    def _api(self, paths: list[str]) -> Mock:
        api = Mock()
        api.list_repo_tree = Mock(return_value=iter(_repo_files(*paths)))
        return api

    def test_complete_set_returns_empty(self):
        api = self._api(["bucket/a.json", "bucket/b.json"])
        missing = hub.verify_repo_paths_uploaded(
            api, "org/data", ["bucket/a.json", "bucket/b.json"], path_in_repo="bucket"
        )
        assert missing == []

    def test_one_missing_returned(self):
        api = self._api(["bucket/a.json"])
        missing = hub.verify_repo_paths_uploaded(
            api, "org/data", ["bucket/a.json", "bucket/b.json"], path_in_repo="bucket"
        )
        assert missing == ["bucket/b.json"]

    def test_absent_prefix_returns_all_missing(self):
        """A prefix absent on the repo (EntryNotFoundError during the scoped
        walk) returns ALL expected paths — the caller's fail-loud fires with
        the full list."""

        api = Mock()
        api.list_repo_tree = Mock(side_effect=EntryNotFoundError("tree bucket not found"))
        missing = hub.verify_repo_paths_uploaded(
            api, "org/data", ["bucket/a.json", "bucket/b.json"], path_in_repo="bucket"
        )
        assert missing == ["bucket/a.json", "bucket/b.json"]

    def test_empty_prefix_raises(self):
        """An empty/slash-only path_in_repo raises — an unscoped verify would
        recreate the #920 full-repo wedge."""
        with pytest.raises(ValueError, match="empty path_in_repo"):
            hub.verify_repo_paths_uploaded(Mock(), "org/data", ["a.json"], path_in_repo="/")

    def test_outside_prefix_raises(self):
        """Expected paths not covered by the prefix raise BEFORE any listing."""
        api = Mock()
        with pytest.raises(ValueError, match="outside"):
            hub.verify_repo_paths_uploaded(api, "org/data", ["other/x.json"], path_in_repo="bucket")
        api.list_repo_tree.assert_not_called()

    @staticmethod
    def _file_exists_fake(results: list):
        """Signature-conformant ``HfApi.file_exists`` fake (mirrors the real
        ``file_exists(repo_id, filename, *, repo_type=..., revision=..., token=...)``
        keyword surface); pops ``results`` per call — an Exception entry is
        raised, anything else returned. Records calls for assertion."""
        calls: list[tuple] = []

        def fake_file_exists(repo_id, filename, *, repo_type=None, revision=None, token=None):
            calls.append((repo_id, filename, repo_type, revision))
            result = results.pop(0)
            if isinstance(result, Exception):
                raise result
            return result

        return fake_file_exists, calls

    def test_exact_file_prefix_verifies_via_file_exists(self):
        """Round-1 BLOCKER regression (exact-file-prefix-verify-false-missing):
        the LIVE tree endpoint 404s on an exact-FILE path_in_repo (hub 0.36.2,
        #939), so the fake ``list_repo_tree`` RAISES EntryNotFoundError and the
        helper must fall back to a ``file_exists`` probe — a successfully-
        uploaded file must NOT be reported missing. Pre-fix this returned
        ``["bucket/x.json"]`` (EntryNotFoundError => all expected missing)."""

        api = Mock()
        api.list_repo_tree = Mock(side_effect=EntryNotFoundError("tree 404s on file paths"))
        api.file_exists, fe_calls = self._file_exists_fake([True])
        missing = hub.verify_repo_paths_uploaded(
            api, "org/data", ["bucket/x.json"], path_in_repo="bucket/x.json"
        )
        assert missing == []
        assert fe_calls == [("org/data", "bucket/x.json", "dataset", None)]

    def test_exact_file_prefix_absent_reports_missing(self):
        """The False variant: tree 404s AND ``file_exists`` is False -> the
        exact file IS missing (the caller's fail-loud still fires on a
        genuinely absent file)."""

        api = Mock()
        api.list_repo_tree = Mock(side_effect=EntryNotFoundError("tree 404s on file paths"))
        api.file_exists, _fe_calls = self._file_exists_fake([False])
        missing = hub.verify_repo_paths_uploaded(
            api, "org/data", ["bucket/x.json"], path_in_repo="bucket/x.json"
        )
        assert missing == ["bucket/x.json"]

    def test_exact_file_fallback_probe_is_retried(self):
        """The ``file_exists`` fallback is a fresh Hub call on the verify path
        — a transient 500 on the probe retries instead of crashing the verify
        leg (the #920 class must not re-enter through the fallback)."""

        api = Mock()
        api.list_repo_tree = Mock(side_effect=EntryNotFoundError("tree 404s on file paths"))
        api.file_exists, fe_calls = self._file_exists_fake([_http_err(500), True])
        with patch("explore_persona_space.orchestrate.hub.time.sleep") as mock_sleep:
            missing = hub.verify_repo_paths_uploaded(
                api, "org/data", ["bucket/x.json"], path_in_repo="bucket/x.json"
            )
        assert missing == []
        assert len(fe_calls) == 2
        mock_sleep.assert_called_once()

    def test_directory_prefix_absent_never_probes_file_exists(self):
        """A directory-like prefix (no expected path EQUAL to it) keeps the
        all-missing semantics on EntryNotFoundError — the file_exists fallback
        never fires."""

        api = Mock()
        api.list_repo_tree = Mock(side_effect=EntryNotFoundError("tree bucket not found"))
        api.file_exists, fe_calls = self._file_exists_fake([])
        missing = hub.verify_repo_paths_uploaded(
            api, "org/data", ["bucket/a.json", "bucket/b.json"], path_in_repo="bucket"
        )
        assert missing == ["bucket/a.json", "bucket/b.json"]
        assert fe_calls == []

    def test_scoped_kwarg_forwarded(self):
        """The walk is scoped SERVER-side: path_in_repo + recursive=True are
        forwarded to list_repo_tree (never a full listing)."""
        api = self._api(["bucket/a.json"])
        hub.verify_repo_paths_uploaded(
            api, "org/data", ["bucket/a.json"], path_in_repo="bucket/", repo_type="dataset"
        )
        kwargs = api.list_repo_tree.call_args.kwargs
        assert kwargs["path_in_repo"] == "bucket"  # stripped of slashes
        assert kwargs["recursive"] is True
        assert kwargs["repo_type"] == "dataset"


class TestDownloadDatasetRetry:
    """#997 §3.6: ``download_dataset`` wraps its lazy ``hf_hub_download`` in
    the budgeted retry (the #931 xet-read-token 429 leg); the outer fail-soft
    ``return ""`` contract is unchanged. This is the real-body test for the
    seam (#906 rule): ``_retry_upload``'s real body executes; the fake sits at
    the network boundary only and mirrors the call-site signature."""

    def test_download_dataset_retries_transient_500(self, tmp_path):
        calls = {"n": 0}

        def fake_hf_hub_download(
            *, repo_id, filename, repo_type, local_dir, local_dir_use_symlinks, token
        ):
            calls["n"] += 1
            if calls["n"] == 1:
                raise _http_err(500)
            dest = Path(local_dir) / filename
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_text("{}")
            return str(dest)

        target = tmp_path / "out" / "file.json"
        with (
            patch("huggingface_hub.hf_hub_download", fake_hf_hub_download),
            patch("explore_persona_space.orchestrate.hub.time.sleep") as mock_sleep,
        ):
            result = hub.download_dataset("bucket/file.json", str(target), repo_id="org/data")
        assert calls["n"] == 2
        mock_sleep.assert_called_once()
        assert result == str(target)
        assert target.read_text() == "{}"
