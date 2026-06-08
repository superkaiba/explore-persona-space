"""Tests for the three-way env discriminator (cluster / RunPod / local).

The cluster backend (slice 2 of the SLURM-backend plan) requires that
``HF_HOME`` and dotenv resolution route to ``$SCRATCH``-rooted paths
when ``SLURM_JOB_ID`` is set, and that the existing ``/workspace``
behavior is preserved byte-for-byte on RunPod (a regression here would
silently misroute every active RunPod experiment to the wrong cache).

The tests cover all three branches of:

* :func:`env.is_cluster_env` / :func:`env.is_runpod_env`
* :func:`env._hf_home_default`
* :func:`env.resolve_dotenv_path` — and specifically that the
  ``/workspace/explore-persona-space/.env`` fallback is SKIPPED on the
  cluster but kept on RunPod + local.
* :func:`env.load_dotenv` — that the post-call ``HF_HOME`` matches the
  per-environment default.

Each branch is exercised by monkeypatching ``SLURM_JOB_ID`` /
``SCRATCH`` / patching ``Path.exists`` for the synthetic ``/workspace``
case (we cannot mkdir at ``/workspace`` on a CI box).
"""

from __future__ import annotations

import os
from pathlib import Path
from unittest import mock

import pytest

from explore_persona_space.orchestrate import env as env_mod

# ---------------------------------------------------------------------------
# Discriminator
# ---------------------------------------------------------------------------


def test_is_cluster_env_true_when_slurm_job_id_set(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SLURM_JOB_ID", "12345")
    assert env_mod.is_cluster_env() is True


def test_is_cluster_env_false_when_slurm_job_id_absent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    assert env_mod.is_cluster_env() is False


def test_is_runpod_env_true_only_when_workspace_exists_and_no_slurm(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    # Patch Path.exists ONLY for the /workspace check inside is_runpod_env.
    real_exists = Path.exists

    def fake_exists(self: Path) -> bool:
        if str(self) == "/workspace":
            return True
        return real_exists(self)

    with mock.patch.object(Path, "exists", fake_exists):
        assert env_mod.is_runpod_env() is True


def test_is_runpod_env_false_on_cluster_even_if_workspace_mount_exists(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Cluster discriminator wins — RunPod is the no-cluster fallback."""
    monkeypatch.setenv("SLURM_JOB_ID", "12345")
    real_exists = Path.exists

    def fake_exists(self: Path) -> bool:
        if str(self) == "/workspace":
            return True
        return real_exists(self)

    with mock.patch.object(Path, "exists", fake_exists):
        assert env_mod.is_runpod_env() is False
        assert env_mod.is_cluster_env() is True


def test_is_runpod_env_false_locally(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    # On a dev VM /workspace doesn't exist; assume the test runner agrees.
    if Path("/workspace").exists():
        pytest.skip("test host has /workspace; cannot exercise the local branch")
    assert env_mod.is_runpod_env() is False
    assert env_mod.is_cluster_env() is False


# ---------------------------------------------------------------------------
# _hf_home_default — three branches
# ---------------------------------------------------------------------------


def test_hf_home_default_cluster_uses_scratch(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("SLURM_JOB_ID", "999")
    monkeypatch.setenv("SCRATCH", str(tmp_path))
    assert env_mod._hf_home_default() == str(tmp_path / ".cache" / "huggingface")


def test_hf_home_default_cluster_falls_back_to_home_without_scratch(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("SLURM_JOB_ID", "999")
    monkeypatch.delenv("SCRATCH", raising=False)
    monkeypatch.setenv("HOME", str(tmp_path))
    assert env_mod._hf_home_default() == str(tmp_path / ".cache" / "huggingface")


def test_hf_home_default_runpod(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    real_exists = Path.exists

    def fake_exists(self: Path) -> bool:
        if str(self) == "/workspace":
            return True
        return real_exists(self)

    with mock.patch.object(Path, "exists", fake_exists):
        assert env_mod._hf_home_default() == "/workspace/.cache/huggingface"


def test_hf_home_default_local(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    if Path("/workspace").exists():
        pytest.skip("test host has /workspace; cannot exercise the local branch")
    expected = env_mod.get_project_root() / "cache" / "huggingface"
    assert env_mod._hf_home_default() == str(expected)


# ---------------------------------------------------------------------------
# resolve_dotenv_path — cluster skips the /workspace fallback
# ---------------------------------------------------------------------------


def test_resolve_dotenv_skips_workspace_fallback_on_cluster(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """On the cluster, ``/workspace/...env`` MUST NOT be probed.

    Secrets on the cluster arrive via an rsync'd file the sbatch sources
    directly; probing ``/workspace`` from a SLURM compute node would
    either be slow (no such mount) or wrong (an unrelated mount).
    """
    monkeypatch.setenv("SLURM_JOB_ID", "111")
    bare = tmp_path / "bare"
    bare.mkdir()
    # Even if /workspace/explore-persona-space/.env exists, the cluster
    # branch must not return it. We pretend it exists via patching.
    real_is_file = Path.is_file

    def fake_is_file(self: Path) -> bool:
        if str(self) == "/workspace/explore-persona-space/.env":
            return True
        return real_is_file(self)

    with mock.patch.object(Path, "is_file", fake_is_file):
        result = env_mod.resolve_dotenv_path(bare)
        assert result != Path("/workspace/explore-persona-space/.env"), (
            f"cluster branch must skip the /workspace fallback, got {result!r}"
        )


def test_resolve_dotenv_keeps_workspace_fallback_off_cluster(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Off-cluster (RunPod or local): the /workspace fallback is honored.

    Asserts the existing behavior — RunPod pods + local checkouts both
    benefit from the pod-canonical fallback. The cluster branch is the
    one that opts out.
    """
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    bare = tmp_path / "bare"
    bare.mkdir()
    real_is_file = Path.is_file

    def fake_is_file(self: Path) -> bool:
        if str(self) == "/workspace/explore-persona-space/.env":
            return True
        return real_is_file(self)

    with mock.patch.object(Path, "is_file", fake_is_file):
        result = env_mod.resolve_dotenv_path(bare)
        assert result == Path("/workspace/explore-persona-space/.env"), (
            f"off-cluster MUST keep the /workspace dotenv fallback, got {result!r}"
        )


# ---------------------------------------------------------------------------
# load_dotenv post-call HF_HOME
# ---------------------------------------------------------------------------


def test_load_dotenv_sets_cluster_hf_home(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("SLURM_JOB_ID", "1")
    monkeypatch.setenv("SCRATCH", str(tmp_path))
    monkeypatch.delenv("HF_HOME", raising=False)
    env_path = tmp_path / ".env"
    env_path.write_text("SOME_KEY=val\n")
    env_mod.load_dotenv(str(env_path))
    assert os.environ["HF_HOME"] == str(tmp_path / ".cache" / "huggingface")


def test_load_dotenv_sets_runpod_hf_home(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    monkeypatch.delenv("HF_HOME", raising=False)
    env_path = tmp_path / ".env"
    env_path.write_text("SOME_KEY=val\n")
    real_exists = Path.exists

    def fake_exists(self: Path) -> bool:
        if str(self) == "/workspace":
            return True
        return real_exists(self)

    with mock.patch.object(Path, "exists", fake_exists):
        env_mod.load_dotenv(str(env_path))
    assert os.environ["HF_HOME"] == "/workspace/.cache/huggingface"


def test_load_dotenv_sets_local_hf_home(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    monkeypatch.delenv("HF_HOME", raising=False)
    if Path("/workspace").exists():
        pytest.skip("test host has /workspace; cannot exercise the local branch")
    env_path = tmp_path / ".env"
    env_path.write_text("SOME_KEY=val\n")
    env_mod.load_dotenv(str(env_path))
    assert os.environ["HF_HOME"] == str(env_mod.get_project_root() / "cache" / "huggingface")


# ---------------------------------------------------------------------------
# preflight three-way HF_HOME early-set (regression for the cluster branch)
# ---------------------------------------------------------------------------


def test_preflight_early_hf_home_on_cluster(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """`preflight_check` sets HF_HOME=$SCRATCH/... before any sub-check.

    Catches a regression where the early-set branch ONLY ran for RunPod
    and left HF_HOME unset for cluster jobs (which then download to the
    wrong place during e.g. check_connectivity).
    """
    from explore_persona_space.orchestrate import preflight as preflight_mod

    monkeypatch.setenv("SLURM_JOB_ID", "2")
    monkeypatch.setenv("SCRATCH", str(tmp_path))
    monkeypatch.delenv("HF_HOME", raising=False)

    # We don't want check_gpus / check_connectivity / etc. to actually
    # run their substantive logic — substitute no-ops so the test stays
    # isolated to the early HF_HOME branch.
    with (
        mock.patch.object(preflight_mod, "check_git_status"),
        mock.patch.object(preflight_mod, "check_env_sync"),
        mock.patch.object(preflight_mod, "check_disk_space"),
        mock.patch.object(preflight_mod, "check_disk_budget"),
        mock.patch.object(preflight_mod, "check_gpus"),
        mock.patch.object(preflight_mod, "check_hf_home"),
        mock.patch.object(preflight_mod, "check_env_vars"),
        mock.patch.object(preflight_mod, "check_vllm_transformers_compat"),
        mock.patch.object(preflight_mod, "check_connectivity"),
    ):
        preflight_mod.preflight_check(
            require_gpu=False,
            min_disk_gb=0.0,
            min_gpu_free_mb=0,
            required_env_vars=[],
            check_code_sync=False,
            planned_footprint_gb=None,
            per_pod_quota_gb=None,
        )

    assert os.environ["HF_HOME"] == str(tmp_path / ".cache" / "huggingface")
