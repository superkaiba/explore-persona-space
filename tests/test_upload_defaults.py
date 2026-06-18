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

Plus the two #565 review follow-ups layered on top:

4. ``runner.run_single``'s merged-upload gate (``distributed or
   merged_upload_enabled(cfg)``) — driven through the REAL ``run_single``
   (``TestRunnerMergedGate``).
5. The legacy i207 worker's adapter upload routes through
   ``hub.upload_model`` instead of a raw ``HfApi.upload_folder``
   (``TestI207AdapterUploadRouting``).

All HF API calls are mocked — no network, no HF_TOKEN required.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from unittest.mock import MagicMock, patch

from huggingface_hub.hf_api import RepoFile
from omegaconf import OmegaConf

from explore_persona_space.orchestrate.hub import (
    TRAINING_STATE_IGNORE_PATTERNS,
    merged_upload_enabled,
    upload_model,
)
from explore_persona_space.orchestrate.runner import run_single
from explore_persona_space.train.trainer import _finalize_phase, _maybe_upload_adapter_default

# The i207 worker is a script, not a package module — load it via importlib so
# its upload routing is unit-testable (#565; convention from
# test_i480_band_stop_dispatch.py). Its module top is stdlib + dotenv only.
_I207_PATH = Path(__file__).resolve().parents[1] / "scripts" / "run_i207_gentle_worker.py"
_i207_spec = importlib.util.spec_from_file_location("i207_worker_under_test", _I207_PATH)
assert _i207_spec is not None and _i207_spec.loader is not None
i207 = importlib.util.module_from_spec(_i207_spec)
_i207_spec.loader.exec_module(i207)


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


class TestI207AdapterUploadRouting:
    """The i207 worker's adapter upload routes through hub.upload_model (#565).

    Pins the review follow-up: no direct ``HfApi.upload_folder`` (which
    bypassed TRAINING_STATE_IGNORE_PATTERNS and post-upload verification);
    non-fatal contract preserved (warn + return None on any failure).
    """

    def test_success_routes_through_upload_model(self):
        expected = "superkaiba1/explore-persona-space/adapters/r1"
        with patch(
            "explore_persona_space.orchestrate.hub.upload_model",
            return_value=expected,
        ) as mock_upload:
            result = i207.upload_adapter_to_hub("/fake/adapter", "r1")

        assert result == expected, "success return must be upload_model's verified hub path"
        mock_upload.assert_called_once()
        kwargs = mock_upload.call_args[1]
        assert kwargs["model_path"] == "/fake/adapter"
        assert kwargs["repo_id"] == "superkaiba1/explore-persona-space"
        assert kwargs["path_in_repo"] == "adapters/r1"
        assert kwargs["delete_after"] is False
        assert kwargs["ignore_patterns"] == ["checkpoint-*"]

    def test_unverified_upload_returns_none(self):
        """upload_model's silent-failure '' return -> warn + None, never raise."""
        with patch("explore_persona_space.orchestrate.hub.upload_model", return_value=""):
            assert i207.upload_adapter_to_hub("/fake/adapter", "r1") is None

    def test_upload_exception_is_non_fatal(self):
        """A raising upload_model -> warn + None (Stage C eval must still run)."""
        with patch(
            "explore_persona_space.orchestrate.hub.upload_model",
            side_effect=RuntimeError("hub down"),
        ):
            assert i207.upload_adapter_to_hub("/fake/adapter", "r1") is None

    def test_no_direct_hub_api_calls_remain(self):
        """The policy bypass is gone at the source level: no HfApi/upload_folder."""
        source = _I207_PATH.read_text()
        assert "HfApi" not in source
        assert "upload_folder" not in source


def _runner_cfg(tmp_path: Path, **extra):
    """Minimal cfg covering every non-.get access in run_single."""
    return OmegaConf.create(
        {"condition": {"name": "c_test"}, "output_dir": str(tmp_path), "upload_to": "hf", **extra}
    )


def _run_runner(cfg, tmp_path: Path, *, distributed: bool = False, env: dict | None = None):
    """Drive the REAL run_single gate with training/upload/cleanup seams mocked.

    ``upload_model``/``cleanup_hf_cache`` are deferred imports inside
    ``run_single``, so they patch in the hub module namespace;
    ``run_two_phase_training``/``run_distributed_pipeline``/``set_seed`` are
    module-top imports and patch in the runner namespace.
    ``merged_upload_enabled`` is deliberately NOT mocked — the real gate
    predicate runs, driven by env/cfg.
    """
    with (
        patch.dict("os.environ", env or {}, clear=True),
        patch("explore_persona_space.orchestrate.runner.set_seed"),
        patch(
            "explore_persona_space.orchestrate.runner.run_two_phase_training",
            return_value=str(tmp_path / "models" / "c_test_seed42"),
        ) as mock_train,
        patch(
            "explore_persona_space.orchestrate.runner.run_distributed_pipeline",
            return_value=str(tmp_path / "models" / "c_test_seed42"),
        ) as mock_dist,
        patch(
            "explore_persona_space.orchestrate.hub.upload_model",
            return_value="repo/c_test_seed42_post_em",
        ) as mock_upload,
        # SAFETY: run_single reaches hub.cleanup_hf_cache() on the
        # upload_to=="hf" + not-skip_training path; unmocked under the
        # cleared env it would rmtree the REAL ~/.cache/huggingface/hub
        # blobs on this machine.
        patch("explore_persona_space.orchestrate.hub.cleanup_hf_cache") as mock_cleanup,
    ):
        result = run_single(cfg, seed=42, skip_eval=True, distributed=distributed)
    return result, mock_train, mock_dist, mock_upload, mock_cleanup


class TestRunnerMergedGate:
    """run_single's merged-upload gate (runner.py: `distributed or merged_upload_enabled`).

    The headline c5bc6149c behavior change: merged checkpoints upload only
    when distributed (full fine-tune — the checkpoint IS canonical) or when
    explicitly opted in via EPM_UPLOAD_MERGED=1 / `upload_merged: true`.
    Drives the real run_single; `mock_cleanup` asserted in every arm doubles
    as a reached-the-end sentinel.
    """

    def test_default_no_merged_upload(self, tmp_path):
        result, mock_train, mock_dist, mock_upload, mock_cleanup = _run_runner(
            _runner_cfg(tmp_path), tmp_path
        )
        mock_upload.assert_not_called()
        assert result["status"] == "completed"
        assert "upload_failed" not in result
        mock_train.assert_called_once()
        mock_dist.assert_not_called()
        mock_cleanup.assert_called_once()

    def test_env_flag_opts_in_merged_upload(self, tmp_path):
        result, _, _, mock_upload, mock_cleanup = _run_runner(
            _runner_cfg(tmp_path), tmp_path, env={"EPM_UPLOAD_MERGED": "1"}
        )
        # Exactly once: the pre-EM dir doesn't exist under tmp_path, so the
        # second (pre-EM) upload branch self-skips.
        mock_upload.assert_called_once()
        assert mock_upload.call_args[1]["path_in_repo"] == "c_test_seed42_post_em"
        assert "upload_failed" not in result
        mock_cleanup.assert_called_once()

    def test_cfg_flag_opts_in_merged_upload(self, tmp_path):
        """`upload_merged: true` alone (env cleared) must open the gate —
        a dropped/typo'd cfg.get at the gate would pass the other arms."""
        result, _, _, mock_upload, mock_cleanup = _run_runner(
            _runner_cfg(tmp_path, upload_merged=True), tmp_path
        )
        mock_upload.assert_called_once()
        assert mock_upload.call_args[1]["path_in_repo"] == "c_test_seed42_post_em"
        assert "upload_failed" not in result
        mock_cleanup.assert_called_once()

    def test_distributed_always_uploads(self, tmp_path):
        result, mock_train, mock_dist, mock_upload, mock_cleanup = _run_runner(
            _runner_cfg(tmp_path), tmp_path, distributed=True
        )
        mock_upload.assert_called_once()
        assert mock_upload.call_args[1]["path_in_repo"] == "c_test_seed42_post_em"
        mock_dist.assert_called_once()
        mock_train.assert_not_called()
        assert "upload_failed" not in result
        mock_cleanup.assert_called_once()
