"""Integration tests: HF Hub and WandB upload round-trips.

These tests do NOT require a GPU -- they upload small synthetic files to
real cloud services and verify the round-trip.

Requires: HF_TOKEN (for HF Hub tests), WANDB_API_KEY (for WandB tests).
"""

from __future__ import annotations

import json
import os
import uuid
from pathlib import Path

import pytest


def _hf_token_available() -> bool:
    return bool(os.environ.get("HF_TOKEN"))


def _wandb_key_available() -> bool:
    return bool(os.environ.get("WANDB_API_KEY"))


@pytest.mark.integration
class TestHFHubUpload:
    """Upload a small directory to HF Hub, verify files, clean up."""

    @pytest.fixture
    def test_prefix(self, test_run_id: str) -> str:
        return f"_test/{test_run_id}/{uuid.uuid4().hex[:8]}"

    @pytest.fixture
    def fake_model_dir(self, tmp_path: Path) -> Path:
        """Create a fake model directory with a few small files."""
        model_dir = tmp_path / "fake_model"
        model_dir.mkdir()
        (model_dir / "config.json").write_text(json.dumps({"model_type": "test"}))
        (model_dir / "tokenizer.json").write_text(json.dumps({"version": "1.0"}))
        (model_dir / "README.md").write_text("Test model for integration tests")
        return model_dir

    @pytest.mark.skipif(not _hf_token_available(), reason="HF_TOKEN not set")
    def test_upload_model_roundtrip(
        self,
        fake_model_dir: Path,
        test_prefix: str,
    ) -> None:
        """Upload a fake model dir, verify files on Hub, then clean up."""
        from huggingface_hub import HfApi

        from explore_persona_space.orchestrate.hub import DEFAULT_MODEL_REPO, upload_model

        result = upload_model(
            str(fake_model_dir),
            repo_id=DEFAULT_MODEL_REPO,
            path_in_repo=test_prefix,
        )
        assert result, "upload_model returned empty string (upload failed)"

        try:
            # Verify files exist on Hub
            api = HfApi(token=os.environ["HF_TOKEN"])
            files = api.list_repo_files(repo_id=DEFAULT_MODEL_REPO, repo_type="model")
            prefix_files = [f for f in files if f.startswith(test_prefix + "/")]
            assert len(prefix_files) >= 2, (
                f"Expected >= 2 files under {test_prefix}/, found {len(prefix_files)}"
            )
        finally:
            # Clean up: delete the test folder from Hub
            try:
                api = HfApi(token=os.environ["HF_TOKEN"])
                # Delete the unique subfolder (e.g. _test/<run_id>/<uuid>)
                api.delete_folder(
                    path_in_repo=test_prefix,
                    repo_id=DEFAULT_MODEL_REPO,
                    repo_type="model",
                    commit_message="integration test cleanup",
                )
            except Exception:
                pass  # Best-effort cleanup

    @pytest.mark.skipif(not _hf_token_available(), reason="HF_TOKEN not set")
    def test_upload_dataset_roundtrip(
        self,
        tmp_path: Path,
        test_prefix: str,
    ) -> None:
        """Upload a dataset file, verify on Hub, clean up."""
        from huggingface_hub import HfApi

        from explore_persona_space.orchestrate.hub import DEFAULT_DATASET_REPO, upload_dataset

        data_path = tmp_path / "test_data.jsonl"
        with open(data_path, "w") as f:
            for i in range(5):
                f.write(json.dumps({"text": f"example {i}"}) + "\n")

        file_in_repo = f"{test_prefix}/test_data.jsonl"
        result = upload_dataset(
            str(data_path),
            repo_id=DEFAULT_DATASET_REPO,
            path_in_repo=file_in_repo,
        )
        assert result, "upload_dataset returned empty string (upload failed)"

        try:
            api = HfApi(token=os.environ["HF_TOKEN"])
            files = api.list_repo_files(repo_id=DEFAULT_DATASET_REPO, repo_type="dataset")
            assert file_in_repo in files, f"{file_in_repo} not found on Hub"
        finally:
            try:
                api = HfApi(token=os.environ["HF_TOKEN"])
                api.delete_folder(
                    path_in_repo=test_prefix,
                    repo_id=DEFAULT_DATASET_REPO,
                    repo_type="dataset",
                    commit_message="integration test cleanup",
                )
            except Exception:
                pass


@pytest.mark.integration
class TestWandBUpload:
    """Upload results to WandB as an artifact, verify the reference."""

    @pytest.mark.skipif(not _wandb_key_available(), reason="WANDB_API_KEY not set")
    def test_upload_results_wandb(self, tmp_path: Path, test_run_id: str) -> None:
        """Upload a small results dir as a WandB artifact."""
        import wandb

        # Override WANDB_MODE for this specific test (conftest sets disabled)
        old_mode = os.environ.get("WANDB_MODE")
        os.environ.pop("WANDB_MODE", None)

        try:
            from explore_persona_space.orchestrate.hub import upload_results_wandb

            results_dir = tmp_path / "test_results"
            results_dir.mkdir()
            (results_dir / "metrics.json").write_text(json.dumps({"accuracy": 0.85, "loss": 0.42}))

            artifact_name = f"integ_test_{test_run_id}"
            ref = upload_results_wandb(
                str(results_dir),
                project="explore_persona_space_test",
                name=artifact_name,
                metadata={"test": True, "run_id": test_run_id},
            )
            assert ref, "upload_results_wandb returned empty string"
            assert "explore_persona_space_test" in ref
        finally:
            if wandb.run:
                wandb.finish(quiet=True)
            if old_mode is None:
                os.environ["WANDB_MODE"] = "disabled"
            else:
                os.environ["WANDB_MODE"] = old_mode
