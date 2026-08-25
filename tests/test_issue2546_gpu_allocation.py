"""Issue #2546 — env-matrix regression pin for ``gpu_allocation()`` (r4).

Closes r3 NIT ``allocation-ladder-regression-pin-missing``: the r2 Critical-2
allocation-first ladder fix (CVD array > SLURM STEP/JOB ids > SLURM count >
fail-loud > nvidia-smi enumeration on non-SLURM hosts; gotchas.md #1902/#1336)
had no permanent pin. Each case asserts the EXACT allocation ids or the
expected loud failure; the nvidia-smi legs monkeypatch ``subprocess.run`` at
the module seam (external-binary boundary only — the ladder under test is
real), including the binary-absent OSError leg that FAILS pre-r4-fix with a
raw FileNotFoundError.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "scripts"))

import issue2546_gen_capture as G  # noqa: E402

_ENV_VARS = (
    "CUDA_VISIBLE_DEVICES",
    "SLURM_JOB_ID",
    "SLURM_STEP_GPUS",
    "SLURM_JOB_GPUS",
    "SLURM_GPUS_ON_NODE",
)


@pytest.fixture()
def clean_env(monkeypatch):
    for var in _ENV_VARS:
        monkeypatch.delenv(var, raising=False)
    return monkeypatch


def test_cvd_set_nonempty_wins_over_everything(clean_env):
    clean_env.setenv("CUDA_VISIBLE_DEVICES", "4,6")
    clean_env.setenv("SLURM_JOB_ID", "1")
    clean_env.setenv("SLURM_JOB_GPUS", "0,1,2")
    assert G.gpu_allocation() == ["4", "6"]
    assert G.gpu_count() == 2


def test_cvd_set_empty_is_zero_gpu_allocation_never_enumeration(clean_env):
    """Explicitly-empty CVD = deliberate zero-GPU allocation (r3 NIT
    empty-cvd-dispatch-enumeration): [] — never a host-enumeration fallback."""
    clean_env.setenv("CUDA_VISIBLE_DEVICES", "")

    def _boom(*a, **k):  # enumeration must NOT be consulted
        raise AssertionError("nvidia-smi consulted despite an explicit empty CVD")

    clean_env.setattr(G.subprocess, "run", _boom)
    assert G.gpu_allocation() == []
    assert G.gpu_count() == 0


def test_slurm_step_gpus_beats_job_gpus(clean_env):
    clean_env.setenv("SLURM_JOB_ID", "1")
    clean_env.setenv("SLURM_STEP_GPUS", "2,3")
    clean_env.setenv("SLURM_JOB_GPUS", "4,5,6")
    assert G.gpu_allocation() == ["2", "3"]


def test_slurm_job_gpus_ids(clean_env):
    clean_env.setenv("SLURM_JOB_ID", "1")
    clean_env.setenv("SLURM_JOB_GPUS", "4,6")
    assert G.gpu_allocation() == ["4", "6"]


def test_slurm_gpus_on_node_count_assumes_leading_ids(clean_env):
    clean_env.setenv("SLURM_JOB_ID", "1")
    clean_env.setenv("SLURM_GPUS_ON_NODE", "3")
    assert G.gpu_allocation() == ["0", "1", "2"]


def test_slurm_job_with_no_gpu_env_fails_loud(clean_env):
    clean_env.setenv("SLURM_JOB_ID", "1")
    with pytest.raises(RuntimeError, match="shared-node trespass"):
        G.gpu_allocation()


def test_slurm_non_numeric_count_fails_loud(clean_env):
    clean_env.setenv("SLURM_JOB_ID", "1")
    clean_env.setenv("SLURM_GPUS_ON_NODE", "3(x2)")
    with pytest.raises(RuntimeError, match="not a count"):
        G.gpu_allocation()


def test_non_slurm_enumeration_counts_nvidia_smi_lines(clean_env):
    def _fake_run(cmd, **kwargs):
        assert cmd[0] == "nvidia-smi"
        return subprocess.CompletedProcess(cmd, 0, stdout="GPU 0: A\nGPU 1: B\n", stderr="")

    clean_env.setattr(G.subprocess, "run", _fake_run)
    assert G.gpu_allocation() == ["0", "1"]


def test_non_slurm_nvidia_smi_nonzero_rc_is_zero_gpus(clean_env):
    def _fake_run(cmd, **kwargs):
        return subprocess.CompletedProcess(cmd, 9, stdout="", stderr="boom")

    clean_env.setattr(G.subprocess, "run", _fake_run)
    assert G.gpu_allocation() == []


def test_non_slurm_nvidia_smi_binary_absent_is_zero_gpus(clean_env):
    """FAILS pre-r4-fix: an absent nvidia-smi binary raised FileNotFoundError
    out of gpu_allocation() instead of reading as zero visible GPUs (r3
    Claude minor 3 — portability of the enumeration leg)."""

    def _fake_run(cmd, **kwargs):
        raise FileNotFoundError(2, "No such file or directory", "nvidia-smi")

    clean_env.setattr(G.subprocess, "run", _fake_run)
    assert G.gpu_allocation() == []


def test_fit_parent_zero_gpu_allocation_raises_before_cache_staging(clean_env, tmp_path):
    """FAILS pre-r5-fix: run_parent's ``max(1, ...)`` silently converted an
    empty GPU allocation into ONE CPU worker for the 222-unit P5 battery (r4
    Codex Major 2 / concern zero-gpu-fit-parent-cpu-fallback). Post-fix the
    allocation is resolved BEFORE cache staging and raises immediately — the
    tripwired heavy helpers prove no staging/fit work starts."""
    import issue2546_fit_cells as F

    def _tripwire(name):
        def _boom(*a, **k):
            raise AssertionError(f"{name} reached on a zero-GPU host (must raise first)")

        return _boom

    clean_env.setattr(F.g25, "gpu_allocation", lambda: [])
    for heavy in ("build_fitcache", "load_caches", "build_rowsets", "build_registry"):
        clean_env.setattr(F, heavy, _tripwire(heavy))
    args = SimpleNamespace(out_root=tmp_path, smoke=True, num_workers=4)
    with pytest.raises(RuntimeError, match="no GPUs visible"):
        F.run_parent(args, F.profile_for_arm(1))
