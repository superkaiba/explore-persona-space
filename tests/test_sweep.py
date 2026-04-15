"""Tests for orchestrate/sweep.py — JobSpec dataclass and GPU detection."""

import subprocess
from unittest.mock import patch

import pytest

from explore_persona_space.orchestrate.sweep import JobSpec, get_free_gpus


class TestJobSpec:
    """Tests for the JobSpec frozen dataclass."""

    def test_creation_minimal(self):
        job = JobSpec(condition_name="c1_evil_wrong_em", seed=42, gpu_id=0)
        assert job.condition_name == "c1_evil_wrong_em"
        assert job.seed == 42
        assert job.gpu_id == 0
        assert job.skip_training is False
        assert job.skip_eval is False
        assert job.distributed is False
        assert job.num_gpus == 8

    def test_creation_full(self):
        job = JobSpec(
            condition_name="c3_good_wrong_em",
            seed=137,
            gpu_id=2,
            skip_training=True,
            skip_eval=False,
            distributed=True,
            num_gpus=4,
        )
        assert job.condition_name == "c3_good_wrong_em"
        assert job.distributed is True
        assert job.num_gpus == 4

    def test_frozen(self):
        """JobSpec should be immutable."""
        job = JobSpec(condition_name="test", seed=42, gpu_id=0)
        with pytest.raises(AttributeError):
            job.seed = 99  # type: ignore[misc]

    def test_equality(self):
        """Two JobSpecs with same values should be equal."""
        j1 = JobSpec(condition_name="test", seed=42, gpu_id=0)
        j2 = JobSpec(condition_name="test", seed=42, gpu_id=0)
        assert j1 == j2

    def test_hashable(self):
        """Frozen dataclass should be hashable (usable in sets/dicts)."""
        j1 = JobSpec(condition_name="test", seed=42, gpu_id=0)
        j2 = JobSpec(condition_name="test", seed=42, gpu_id=1)
        job_set = {j1, j2}
        assert len(job_set) == 2


class TestGetFreeGpus:
    """Tests for get_free_gpus — GPU detection via nvidia-smi."""

    def test_nvidia_smi_failure_raises(self):
        """When nvidia-smi fails, should raise RuntimeError (not silently fallback)."""
        with (
            patch("subprocess.run", side_effect=FileNotFoundError("nvidia-smi not found")),
            pytest.raises(RuntimeError, match="nvidia-smi failed"),
        ):
            get_free_gpus()

    def test_parses_nvidia_smi_output(self):
        """Should correctly parse nvidia-smi CSV output."""
        mock_output = "0, 79000\n1, 79000\n2, 10000\n3, 79000\n"
        mock_result = subprocess.CompletedProcess(
            args=[], returncode=0, stdout=mock_output, stderr=""
        )
        with patch("subprocess.run", return_value=mock_result):
            gpus = get_free_gpus(min_free_mb=50_000)
        assert gpus == [0, 1, 3]

    def test_no_free_gpus_returns_empty(self):
        """When all GPUs are busy, should return empty list (not hardcoded fallback)."""
        mock_output = "0, 1000\n1, 2000\n"
        mock_result = subprocess.CompletedProcess(
            args=[], returncode=0, stdout=mock_output, stderr=""
        )
        with patch("subprocess.run", return_value=mock_result):
            gpus = get_free_gpus(min_free_mb=50_000)
        assert gpus == []

    def test_custom_min_free_mb(self):
        """Should respect custom min_free_mb threshold."""
        mock_output = "0, 30000\n1, 60000\n"
        mock_result = subprocess.CompletedProcess(
            args=[], returncode=0, stdout=mock_output, stderr=""
        )
        with patch("subprocess.run", return_value=mock_result):
            gpus = get_free_gpus(min_free_mb=25_000)
        assert gpus == [0, 1]
