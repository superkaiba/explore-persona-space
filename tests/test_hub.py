"""Tests for orchestrate/hub.py — upload/download utilities."""

import json
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from explore_persona_space.orchestrate.hub import (
    DEFAULT_DATASET_REPO,
    DEFAULT_MODEL_REPO,
    upload_dataset,
    upload_dataset_directory,
    upload_model,
)


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
