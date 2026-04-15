"""Tests for orchestrate/hub.py — upload/download utilities."""

import json
import tempfile
from pathlib import Path
from unittest.mock import patch

from explore_persona_space.orchestrate.hub import (
    DEFAULT_DATASET_REPO,
    DEFAULT_MODEL_REPO,
    upload_dataset,
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
