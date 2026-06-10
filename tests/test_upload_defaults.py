"""Tests for adapter-only HF upload defaults (2026-06-10 storage-policy fix).

Pins three behaviors introduced after the HF storage inventory found the
account at 11.2TB public, dominated by merged checkpoint dirs and
optimizer.pt residue from wholesale ``upload_folder`` calls:

1. Every folder upload through ``orchestrate/hub.py`` ALWAYS excludes
   optimizer/scheduler/RNG training state (no opt-out).
2. Merged-checkpoint uploads are opt-in (``EPM_UPLOAD_MERGED=1`` env or
   ``upload_merged: true`` cfg) via ``merged_upload_enabled``.
3. ``_finalize_phase`` uploads the LoRA ADAPTER to HF by default
   (``_maybe_upload_adapter_default``, checkpoint-* excluded) and reaps the
   local adapter dir only after a verified upload (or under the explicit
   ``EPM_SKIP_INLINE_CHECKPOINT_UPLOAD=1`` orchestrator fence).

All HF API calls are mocked — no network, no HF_TOKEN required.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

from huggingface_hub.hf_api import RepoFile

from explore_persona_space.orchestrate.hub import (
    TRAINING_STATE_IGNORE_PATTERNS,
    merged_upload_enabled,
    upload_model,
)
from explore_persona_space.train.trainer import _finalize_phase, _maybe_upload_adapter_default


def _make_adapter_dir(tmp_path: Path, run_name: str = "c1_evil_wrong_em_seed42") -> Path:
    """Adapter dir shaped like real ``_init_phase`` output, with trainer residue.

    Includes a ``checkpoint-*`` intermediate save carrying ``optimizer.pt`` —
    the residue class the upload defaults must keep off the Hub.
    """
    run_dir = tmp_path / run_name
    adapter = run_dir / "phase2_adapter"
    ckpt = adapter / "checkpoint-500"
    ckpt.mkdir(parents=True)
    (adapter / "adapter_config.json").write_text("{}")
    (adapter / "adapter_model.safetensors").write_bytes(b"\x00" * 16)
    (ckpt / "adapter_model.safetensors").write_bytes(b"\x00" * 16)
    (ckpt / "optimizer.pt").write_bytes(b"\x00" * 16)
    (ckpt / "scheduler.pt").write_bytes(b"\x00" * 16)
    (ckpt / "rng_state.pth").write_bytes(b"\x00" * 16)
    return adapter


def _mock_hub_api(MockApi, committed_path: str):
    """Configure a mocked HfApi whose post-upload listing shows one file."""
    api = MockApi.return_value
    api.create_repo.return_value = None
    api.upload_folder.return_value = None
    api.list_repo_tree.return_value = [RepoFile(path=committed_path, size=1, blob_id="b", oid="o")]
    return api


class TestTrainingStateExcludedFromFolderUploads:
    """hub._upload must never ship optimizer/scheduler/RNG state."""

    def test_upload_model_always_passes_training_state_ignores(self, tmp_path):
        adapter = _make_adapter_dir(tmp_path)
        with (
            patch.dict("os.environ", {"HF_TOKEN": "t"}),
            patch("huggingface_hub.HfApi") as MockApi,
        ):
            api = _mock_hub_api(MockApi, "dest/adapter_model.safetensors")
            result = upload_model(str(adapter), path_in_repo="dest")

        assert result, "verified upload should return a non-empty hub path"
        ignore = api.upload_folder.call_args[1]["ignore_patterns"]
        for pattern in TRAINING_STATE_IGNORE_PATTERNS:
            assert pattern in ignore, f"missing always-on exclude {pattern!r}"

    def test_training_state_patterns_cover_checkpoint_residue(self):
        """fnmatch semantics: ``*`` crosses ``/`` — nested state files match."""
        from fnmatch import fnmatch

        residue = [
            "checkpoint-500/optimizer.pt",
            "optimizer.pt",
            "checkpoint-500/scheduler.pt",
            "checkpoint-500/rng_state.pth",
            "checkpoint-500/rng_state_3.pth",
        ]
        for path in residue:
            assert any(fnmatch(path, p) for p in TRAINING_STATE_IGNORE_PATTERNS), (
                f"{path} not covered by TRAINING_STATE_IGNORE_PATTERNS"
            )
        # Adapter weights must NOT be excluded.
        keep = ["adapter_model.safetensors", "checkpoint-500/adapter_model.safetensors"]
        for path in keep:
            assert not any(fnmatch(path, p) for p in TRAINING_STATE_IGNORE_PATTERNS), (
                f"{path} wrongly excluded"
            )

    def test_extra_ignore_patterns_are_merged_not_replaced(self, tmp_path):
        adapter = _make_adapter_dir(tmp_path)
        with (
            patch.dict("os.environ", {"HF_TOKEN": "t"}),
            patch("huggingface_hub.HfApi") as MockApi,
        ):
            api = _mock_hub_api(MockApi, "dest/adapter_model.safetensors")
            upload_model(str(adapter), path_in_repo="dest", ignore_patterns=["checkpoint-*"])

        ignore = api.upload_folder.call_args[1]["ignore_patterns"]
        assert "checkpoint-*" in ignore
        for pattern in TRAINING_STATE_IGNORE_PATTERNS:
            assert pattern in ignore


class TestMergedUploadEnabled:
    """Merged-checkpoint upload is opt-in, default OFF."""

    def test_default_off(self):
        with patch.dict("os.environ", {}, clear=True):
            assert merged_upload_enabled() is False
            assert merged_upload_enabled(False) is False
            assert merged_upload_enabled(None) is False

    def test_env_opt_in(self):
        with patch.dict("os.environ", {"EPM_UPLOAD_MERGED": "1"}, clear=True):
            assert merged_upload_enabled() is True

    def test_env_other_values_do_not_enable(self):
        with patch.dict("os.environ", {"EPM_UPLOAD_MERGED": "0"}, clear=True):
            assert merged_upload_enabled() is False

    def test_cfg_opt_in(self):
        with patch.dict("os.environ", {}, clear=True):
            assert merged_upload_enabled(True) is True


class TestDefaultAdapterUpload:
    """_maybe_upload_adapter_default: adapter-only, best-effort, verified."""

    def test_uploads_adapter_excluding_checkpoints(self, tmp_path):
        adapter = _make_adapter_dir(tmp_path)
        with patch(
            "explore_persona_space.orchestrate.hub.upload_model",
            return_value="repo/adapters/c1_evil_wrong_em_seed42/phase2_adapter",
        ) as mock_upload:
            assert _maybe_upload_adapter_default(adapter) is True

        kwargs = mock_upload.call_args[1]
        assert kwargs["path_in_repo"] == "adapters/c1_evil_wrong_em_seed42/phase2_adapter"
        assert kwargs["ignore_patterns"] == ["checkpoint-*"]
        assert kwargs["delete_after"] is False

    def test_returns_false_when_upload_unverified(self, tmp_path):
        adapter = _make_adapter_dir(tmp_path)
        with patch("explore_persona_space.orchestrate.hub.upload_model", return_value=""):
            assert _maybe_upload_adapter_default(adapter) is False

    def test_never_raises_on_upload_exception(self, tmp_path):
        adapter = _make_adapter_dir(tmp_path)
        with patch(
            "explore_persona_space.orchestrate.hub.upload_model",
            side_effect=RuntimeError("hub down"),
        ):
            assert _maybe_upload_adapter_default(adapter) is False


class TestFinalizePhaseReapGating:
    """_finalize_phase deletes the local adapter only after a durable copy exists."""

    def _run_finalize(self, adapter_dir: Path, tmp_path: Path):
        """Invoke _finalize_phase with training/merge/W&B seams mocked out."""
        merged_dir = tmp_path / "phase2_merged"
        merged_dir.mkdir(exist_ok=True)
        model, tokenizer, trainer = MagicMock(), MagicMock(), MagicMock()
        with (
            patch(
                "explore_persona_space.train.trainer.merge_and_save",
                return_value=str(merged_dir),
            ),
            patch("explore_persona_space.train.trainer._maybe_upload_checkpoint_to_wandb"),
            patch("explore_persona_space.train.trainer._maybe_dump_train_log"),
            patch("explore_persona_space.train.trainer.torch"),
        ):
            return _finalize_phase(
                model,
                tokenizer,
                trainer,
                adapter_dir=adapter_dir,
                merged_dir=merged_dir,
                base_model_for_merge="base",
                model_id="base",
            )

    def test_adapter_kept_when_default_upload_fails(self, tmp_path):
        adapter = _make_adapter_dir(tmp_path)
        with (
            patch.dict("os.environ", {}, clear=True),
            patch(
                "explore_persona_space.train.trainer._maybe_upload_adapter_default",
                return_value=False,
            ),
        ):
            self._run_finalize(adapter, tmp_path)
        assert adapter.exists(), "un-uploaded adapter must not be deleted"

    def test_adapter_reaped_after_verified_default_upload(self, tmp_path):
        adapter = _make_adapter_dir(tmp_path)
        with (
            patch.dict("os.environ", {}, clear=True),
            patch(
                "explore_persona_space.train.trainer._maybe_upload_adapter_default",
                return_value=True,
            ),
        ):
            self._run_finalize(adapter, tmp_path)
        assert not adapter.exists()

    def test_fence_skips_default_upload_and_reaps(self, tmp_path):
        """EPM_SKIP_INLINE_CHECKPOINT_UPLOAD=1: orchestrator owns uploads."""
        adapter = _make_adapter_dir(tmp_path)
        with (
            patch.dict("os.environ", {"EPM_SKIP_INLINE_CHECKPOINT_UPLOAD": "1"}, clear=True),
            patch(
                "explore_persona_space.train.trainer._maybe_upload_adapter_default"
            ) as mock_default,
        ):
            self._run_finalize(adapter, tmp_path)
        mock_default.assert_not_called()
        assert not adapter.exists()

    def test_persist_path_skips_default_upload_and_reaps(self, tmp_path):
        """EPM_PERSIST_ADAPTER_HF_REPO: the fail-loud persist owns the upload."""
        adapter = _make_adapter_dir(tmp_path)
        with (
            patch.dict(
                "os.environ",
                {
                    "EPM_PERSIST_ADAPTER_HF_REPO": "repo/x",
                    "EPM_PERSIST_ADAPTER_SUBFOLDER": "sub",
                },
                clear=True,
            ),
            patch(
                "explore_persona_space.orchestrate.hub.upload_model",
                return_value="repo/x/sub/phase2_adapter",
            ),
            patch(
                "explore_persona_space.train.trainer._maybe_upload_adapter_default"
            ) as mock_default,
        ):
            self._run_finalize(adapter, tmp_path)
        mock_default.assert_not_called()
        assert not adapter.exists()
