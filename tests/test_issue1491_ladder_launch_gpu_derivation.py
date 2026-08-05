"""GPU-width/id derivation table for scripts/issue1491_ladder_launch.sh (#1491, #1902 class).

On the fellows SLURM lane the nodes are GPU-SHARED and ``nvidia-smi -L``
enumerates the PHYSICAL node (8x H200) while ignoring ``CUDA_VISIBLE_DEVICES``,
so a detected-count fan-out over-shards onto other tenants' devices (gotchas.md
"Fellows SLURM nodes are GPU-SHARED"; #1902 crash 1). The launcher must derive
fan-out width + the per-shard CUDA_VISIBLE_DEVICES values from the SLURM
allocation env (via scripts/issue1902_common.py::realized_gpu_ids) whenever
``SLURM_JOB_ID`` is set, fail loud when a SLURM job exposes none of the three
allocation vars, and keep the nvidia-smi enumeration UNCHANGED on non-SLURM
lanes (RunPod / GCE exclusive hosts).

Every case runs the REAL launcher in ``--dry-run`` mode with a scrubbed env and
a fake ``nvidia-smi`` that always enumerates 8 devices (the shared-node shape),
then asserts the ``[gpu-derivation]`` source token, the realized width, and the
planned per-shard ``CUDA_VISIBLE_DEVICES=`` pins. Pre-fix (nvidia-smi-count
derivation) every SLURM case here fails substantively: width 8 on a 2-GPU
allocation, local indices 0..7 as CVD values.
"""

from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts" / "issue1491_ladder_launch.sh"

# --num-shards 8 keeps the shard-offset arithmetic guard satisfied for any
# realized width <= 8; --dry-run never launches processes.
BASE_ARGS = [
    "--scale",
    "0.5B",
    "--split",
    "val_400",
    "--num-shards",
    "8",
    "--shard-offset",
    "0",
    "--dry-run",
]

_FAKE_NVIDIA_SMI_8 = (
    '#!/usr/bin/env bash\nfor i in 0 1 2 3 4 5 6 7; do echo "GPU $i: NVIDIA H200 (fake)"; done\n'
)


def _run(tmp_path, env_extra, args=(), nvidia_smi="8gpus"):
    """Run the real launcher --dry-run with a controlled env.

    nvidia_smi: "8gpus" (fake enumerating the physical 8-GPU node), "empty"
    (fake printing nothing -> the launcher's fallback-8 path), or None (no
    fake on PATH; the real binary, if any, is shadowed only by absence).
    Returns the CompletedProcess.
    """
    env = {
        k: v
        for k, v in os.environ.items()
        if not k.startswith("SLURM_")
        and k not in ("CUDA_VISIBLE_DEVICES", "REPO_ROOT", "WORKLOAD_ROOT", "EPM_LADDER_SHARD_SIZE")
    }
    bindir = tmp_path / "bin"
    bindir.mkdir(exist_ok=True)
    if nvidia_smi == "8gpus":
        fake = bindir / "nvidia-smi"
        fake.write_text(_FAKE_NVIDIA_SMI_8)
        fake.chmod(0o755)
    elif nvidia_smi == "empty":
        fake = bindir / "nvidia-smi"
        fake.write_text("#!/usr/bin/env bash\nexit 0\n")
        fake.chmod(0o755)
    env["PATH"] = f"{bindir}:{env.get('PATH', '')}"
    logdir = tmp_path / "logs"
    logdir.mkdir(exist_ok=True)
    env["EPM_LADDER_LOG_DIR"] = str(logdir)
    env.update(env_extra)
    return subprocess.run(
        ["bash", str(SCRIPT), *BASE_ARGS, *args],
        capture_output=True,
        text=True,
        env=env,
        cwd=str(REPO),
        timeout=300,
    )


def _derivation(out: str) -> tuple[str, int, str]:
    """Parse the [gpu-derivation] fix-engaged line -> (source, width, ids_csv)."""
    m = re.search(r"\[gpu-derivation\] source=(\S+) gpus_per_pod=(\d+) ids=(\S*)", out)
    assert m, f"no [gpu-derivation] line in launcher output:\n{out}"
    return m.group(1), int(m.group(2)), m.group(3)


def _planned_cvds(out: str) -> list[str]:
    """The CUDA_VISIBLE_DEVICES values of the dry-run's planned shard launches."""
    return re.findall(r"CUDA_VISIBLE_DEVICES=(\S+) setsid", out)


# ---- SLURM lane: allocation-env derivation ----------------------------------


def test_slurm_cvd_beats_job_gpus_and_enumeration(tmp_path):
    r = _run(
        tmp_path,
        {"SLURM_JOB_ID": "1", "CUDA_VISIBLE_DEVICES": "2,3", "SLURM_JOB_GPUS": "4,5,6,7"},
    )
    assert r.returncode == 0, r.stderr
    src, width, ids = _derivation(r.stdout)
    assert (src, width, ids) == ("slurm-cvd", 2, "2,3")
    assert _planned_cvds(r.stdout) == ["2", "3"]


def test_slurm_job_gpus_physical_ids_with_local_shard_indices(tmp_path):
    r = _run(tmp_path, {"SLURM_JOB_ID": "1", "SLURM_JOB_GPUS": "4,5,6"})
    assert r.returncode == 0, r.stderr
    src, width, ids = _derivation(r.stdout)
    assert (src, width, ids) == ("slurm-job-gpus", 3, "4,5,6")
    assert _planned_cvds(r.stdout) == ["4", "5", "6"]
    # Shard indices stay LOCAL-index-based (offset + 0..N-1) while the CVD pin
    # is the PHYSICAL id — the visible-ordinal / physical-id split.
    assert "shard 0 -> GPU 4" in r.stdout
    assert "shard 2 -> GPU 6" in r.stdout


def test_slurm_step_gpus_fallback(tmp_path):
    r = _run(tmp_path, {"SLURM_JOB_ID": "1", "SLURM_STEP_GPUS": "6,7"})
    assert r.returncode == 0, r.stderr
    src, width, ids = _derivation(r.stdout)
    assert (src, width, ids) == ("slurm-job-gpus", 2, "6,7")
    assert _planned_cvds(r.stdout) == ["6", "7"]


def test_slurm_count_only_ids_assumed(tmp_path):
    r = _run(tmp_path, {"SLURM_JOB_ID": "1", "SLURM_GPUS_ON_NODE": "2"})
    assert r.returncode == 0, r.stderr
    src, width, ids = _derivation(r.stdout)
    assert src.startswith("slurm-count")
    assert (width, ids) == (2, "0,1")
    assert _planned_cvds(r.stdout) == ["0", "1"]


def test_slurm_clamp_to_gpus_on_node(tmp_path):
    r = _run(
        tmp_path,
        {"SLURM_JOB_ID": "1", "CUDA_VISIBLE_DEVICES": "0,1,2,3", "SLURM_GPUS_ON_NODE": "2"},
    )
    assert r.returncode == 0, r.stderr
    src, width, ids = _derivation(r.stdout)
    assert src == "slurm-cvd-clamped"
    assert (width, ids) == (2, "0,1")
    assert _planned_cvds(r.stdout) == ["0", "1"]


def test_slurm_fail_loud_when_no_allocation_env(tmp_path):
    r = _run(tmp_path, {"SLURM_JOB_ID": "1"})
    assert r.returncode != 0, "pre-#1902-fix shape: launcher must NOT fall back to nvidia-smi"
    combined = r.stdout + r.stderr
    assert "FATAL" in combined
    assert _planned_cvds(r.stdout) == []


def test_slurm_width_feeds_num_shards_guard(tmp_path):
    # LAST = SHARD_OFFSET + GPUS_PER_POD - 1 must use the ALLOCATION width.
    r = _run(tmp_path, {"SLURM_JOB_ID": "1", "SLURM_JOB_GPUS": "4,5,6"}, args=["--num-shards", "2"])
    assert r.returncode != 0
    assert "exceeds --num-shards" in (r.stdout + r.stderr)


# ---- explicit --gpus-per-pod override (wins on width, both lanes) -----------


def test_override_wins_non_slurm(tmp_path):
    r = _run(tmp_path, {}, args=["--gpus-per-pod", "3"])
    assert r.returncode == 0, r.stderr
    src, width, ids = _derivation(r.stdout)
    assert (src, width, ids) == ("override", 3, "0,1,2")
    assert _planned_cvds(r.stdout) == ["0", "1", "2"]


def test_override_wins_slurm_ids_from_allocation(tmp_path):
    r = _run(
        tmp_path,
        {"SLURM_JOB_ID": "1", "CUDA_VISIBLE_DEVICES": "2,3,5,7"},
        args=["--gpus-per-pod", "2"],
    )
    assert r.returncode == 0, r.stderr
    src, width, ids = _derivation(r.stdout)
    assert (src, width, ids) == ("slurm-cvd-override", 2, "2,3")
    assert _planned_cvds(r.stdout) == ["2", "3"]


def test_override_exceeding_slurm_allocation_fails_loud(tmp_path):
    r = _run(
        tmp_path,
        {"SLURM_JOB_ID": "1", "CUDA_VISIBLE_DEVICES": "2,3"},
        args=["--gpus-per-pod", "4"],
    )
    assert r.returncode != 0
    assert "exceeds the SLURM allocation" in (r.stdout + r.stderr)


# ---- non-SLURM lanes: today's behavior, unchanged ---------------------------


def test_non_slurm_keeps_nvidia_smi_enumeration(tmp_path):
    r = _run(tmp_path, {})
    assert r.returncode == 0, r.stderr
    src, width, ids = _derivation(r.stdout)
    assert (src, width, ids) == ("detected", 8, "0,1,2,3,4,5,6,7")
    assert _planned_cvds(r.stdout) == [str(i) for i in range(8)]


def test_non_slurm_fallback_default_8(tmp_path):
    r = _run(tmp_path, {}, nvidia_smi="empty")
    assert r.returncode == 0, r.stderr
    src, width, ids = _derivation(r.stdout)
    assert (src, width) == ("detected", 8)
    assert ids == "0,1,2,3,4,5,6,7"
