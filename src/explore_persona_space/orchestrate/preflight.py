"""Pre-flight checks for experiments. Run before starting ANY training or eval.

Usage:
    # As a module
    from explore_persona_space.orchestrate.preflight import require_preflight
    require_preflight()

    # From CLI
    uv run python -m explore_persona_space.orchestrate.preflight

Three-way environment branch
----------------------------

Preflight runs on three different surfaces; the checks adapt:

* **Cluster** (``SLURM_JOB_ID`` set): disk probe targets ``$SLURM_TMPDIR``
  / ``$SCRATCH`` (not ``/workspace``) and the RunPod MooseFS 130GB
  quota cap is bypassed (``per_pod_quota_gb=None``). The ``git fetch``
  round trip in :func:`check_git_status` and the installed-vs-uv.lock
  :func:`check_env_sync` are SKIPPED: the cluster is rsync-primary
  with no remote git auth, and the venv build happens inside the
  sbatch (so a pre-rsync mismatch is expected, not an error).
  ``HF_HOME`` defaults to ``$SCRATCH/.cache/huggingface``. The Hub /
  WandB reachability check still runs (compute nodes may need a proxy).
* **RunPod** (``/workspace`` exists, no SLURM): unchanged from the
  pre-three-way behavior.
* **Local VM**: unchanged.

The discriminator lives in :mod:`explore_persona_space.orchestrate.env`;
this module imports the helpers so the branch logic stays in ONE place.
"""

import contextlib
import errno
import json
import logging
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

# Three-way environment helpers (see env.py module docstring). Imported
# at top level so the cluster branch threads cleanly through every check
# without re-importing per call.
from explore_persona_space.orchestrate.env import (
    _hf_home_default,
    is_cluster_env,
    is_runpod_env,
)

logger = logging.getLogger(__name__)

# RunPod MooseFS gives each pod a per-pod writable-bytes quota (~130GB) that is
# SEPARATE from, and far below, the share-level free space ``shutil.disk_usage``
# reports (terabytes). A small canary probe (``_probe_writable_bytes``) detects
# the quota only once it is ALREADY exhausted; to catch an over-quota footprint
# BEFORE launch we cap the usable headroom at this constant. Override per-pod via
# the ``quota_gb`` parameter / ``--per-pod-quota-gb`` flag when a pod was
# provisioned with an explicit, larger storage spec. See CLAUDE.md "RunPod
# MooseFS per-pod disk quota" gotcha + memory note feedback_runpod_moosefs_quota.
RUNPOD_PER_POD_QUOTA_GB = 130.0


@dataclass
class PreflightReport:
    """Result of pre-flight checks."""

    ok: bool = True
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    gpu_info: list[dict] = field(default_factory=list)
    disk_free_gb: float = 0.0
    disk_probed_headroom_gb: float = 0.0
    # Human-readable provenance of ``disk_probed_headroom_gb`` so the budget
    # check + summary never mislabel a share-level (quota-blind) number as
    # "probed". Set by ``check_disk_space``.
    disk_headroom_basis: str = "share-level free"
    git_status: str = ""
    env_synced: bool = True
    # Account-level HF public-storage headroom (#564). None = unknown /
    # not checked; basis names the signal ("live-api" / "cache (...)" /
    # "disabled" / "suspect (...)" / "unknown (...)"). Set by
    # ``check_hf_storage``.
    hf_storage_used_tb: float | None = None
    hf_storage_ceiling_tb: float | None = None
    hf_storage_basis: str = ""

    def add_error(self, msg: str):
        self.errors.append(msg)
        self.ok = False

    def add_warning(self, msg: str):
        self.warnings.append(msg)

    def summary(self) -> str:
        """Formatted summary string."""
        lines = []
        status = "PASS" if self.ok else "FAIL"
        lines.append(f"\n{'=' * 60}")
        lines.append(f"  Pre-flight Check: {status}")
        lines.append(f"{'=' * 60}")

        if self.errors:
            lines.append("\n  ERRORS (must fix before running):")
            for e in self.errors:
                lines.append(f"    ✗ {e}")

        if self.warnings:
            lines.append("\n  WARNINGS:")
            for w in self.warnings:
                lines.append(f"    ⚠ {w}")

        if self.gpu_info:
            lines.append("\n  GPUs:")
            for g in self.gpu_info:
                used = g.get("memory_used_mb", 0)
                total = g.get("memory_total_mb", 0)
                free = g.get("memory_free_mb", 0)
                procs = g.get("processes", 0)
                status_icon = "✓" if procs == 0 and used < 1000 else "⚠"
                lines.append(
                    f"    {status_icon} GPU {g['id']}: "
                    f"{free:,}MB free / {total:,}MB total "
                    f"({procs} processes)"
                )

        lines.append(
            f"\n  Disk: {self.disk_free_gb:.1f} GB free "
            f"(usable headroom {self.disk_probed_headroom_gb:.1f} GB, "
            f"basis: {self.disk_headroom_basis})"
        )
        if self.hf_storage_used_tb is not None and self.hf_storage_ceiling_tb is not None:
            lines.append(
                f"  HF storage: {self.hf_storage_used_tb:.2f} TB / "
                f"ceiling {self.hf_storage_ceiling_tb:.1f} TB ({self.hf_storage_basis})"
            )
        else:
            lines.append(f"  HF storage: unknown ({self.hf_storage_basis or 'not checked'})")
        lines.append(f"  Git: {self.git_status}")
        lines.append(f"  Env synced: {'yes' if self.env_synced else 'NO'}")
        lines.append(f"{'=' * 60}\n")
        return "\n".join(lines)


def _run(cmd: list[str], timeout: int = 10) -> tuple[int, str, str]:
    """Run a command with timeout. Returns (returncode, stdout, stderr)."""
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return r.returncode, r.stdout.strip(), r.stderr.strip()
    except subprocess.TimeoutExpired:
        return -1, "", "timeout"
    except FileNotFoundError:
        return -1, "", f"command not found: {cmd[0]}"
    except Exception as e:
        return -1, "", str(e)


def _find_project_root() -> Path:
    """Find project root by looking for pyproject.toml."""
    p = Path(__file__).resolve()
    for parent in [p, *list(p.parents)]:
        if (parent / "pyproject.toml").exists():
            return parent
    return Path.cwd()


def _disk_check_path() -> str:
    """Where to run the disk-space probe — three-way branch.

    * **Cluster:** prefer ``$SLURM_TMPDIR`` (node-local fast scratch
      where data + model are staged) when it exists, else ``$SCRATCH``
      (per-user persistent scratch where the venv + checkpoints live).
      Fall back to ``/`` if neither env var is set — defensive.
    * **RunPod:** ``/workspace`` (the MooseFS-backed pod volume).
    * **Local VM:** ``/`` (the root filesystem).

    The picked path is what ``check_disk_space`` probes for free-space
    + the canary EDQUOT probe. On the cluster the MooseFS 130 GB cap is
    explicitly bypassed by the caller (``per_pod_quota_gb=None``); on
    RunPod the cap is enforced.
    """
    if is_cluster_env():
        for env_var in ("SLURM_TMPDIR", "SCRATCH"):
            candidate = os.environ.get(env_var)
            if candidate and Path(candidate).exists():
                return candidate
        return "/"
    if is_runpod_env():
        return "/workspace"
    return "/"


def check_git_status(report: PreflightReport, project_root: Path):
    """Check git working tree is clean and up to date.

    Cluster branch: the ``git fetch origin`` round trip is SKIPPED because
    the cluster compute node has no remote git auth — code reaches the
    cluster via rsync, not git pull. The local ``git status --porcelain``
    check still runs (it's local-only and cheap) so an accidental
    uncommitted change is still surfaced; we just don't try to compare
    against origin/main. The ``git_status`` field is decorated with
    ``" (cluster — skipped fetch)"`` so the summary makes the skip
    explicit rather than misleadingly reading "clean / up to date".
    """
    # Check for uncommitted changes
    rc, out, err = _run(["git", "-C", str(project_root), "status", "--porcelain"])
    if rc != 0:
        report.add_warning(f"git status failed: {err}")
        report.git_status = "unknown"
        return

    if out:
        changed = len(out.strip().splitlines())
        report.add_warning(f"{changed} uncommitted change(s) — consider committing first")
        report.git_status = f"{changed} uncommitted changes"
    else:
        report.git_status = "clean"

    if is_cluster_env():
        # rsync-primary on the cluster; no remote git auth on compute
        # nodes. Mark explicitly so the summary doesn't read "clean,
        # up-to-date" when we didn't check up-to-date-ness.
        report.git_status += " (cluster — skipped fetch)"
        return

    # Check if behind remote
    _run(["git", "-C", str(project_root), "fetch", "--quiet", "origin"], timeout=15)
    rc, out, _ = _run(["git", "-C", str(project_root), "rev-list", "--count", "HEAD..origin/main"])
    if rc == 0 and out.strip() != "0":
        behind = out.strip()
        report.add_error(
            f"Local is {behind} commit(s) behind origin/main. Run: git pull origin main"
        )
        report.git_status += f", {behind} behind remote"


def check_env_sync(report: PreflightReport, project_root: Path):
    """Check that installed packages match uv.lock.

    Cluster branch: SKIPPED. The sbatch builds / activates the venv
    inside the job (cached at ``$SCRATCH/eps/venv-<lockhash>``), so a
    pre-launch ``uv sync --locked --dry-run`` on the login node would
    report an out-of-sync env that the job is about to fix. Mark
    ``env_synced=True`` with an explicit note in ``git_status`` is the
    wrong field; instead we leave ``env_synced`` True and append a
    warning so the summary's "Env synced: yes" reads honestly while a
    surfaced WARNING line documents the skip.
    """
    if is_cluster_env():
        report.add_warning(
            "env_sync check SKIPPED on cluster — sbatch builds the venv "
            "inside the job from $SCRATCH/eps/venv-<lockhash>."
        )
        report.env_synced = True
        return

    lockfile = project_root / "uv.lock"
    if not lockfile.exists():
        report.add_warning("No uv.lock found — cannot verify environment sync")
        report.env_synced = False
        return

    # uv sync --locked --dry-run exits non-zero if env needs changes
    rc, out, err = _run(
        ["uv", "sync", "--locked", "--dry-run"],
        timeout=30,
    )
    if rc != 0:
        if "would install" in err.lower() or "would install" in out.lower():
            report.add_error("Environment out of sync with uv.lock. Run: uv sync --locked")
            report.env_synced = False
        elif "error" in err.lower():
            report.add_warning(f"uv sync check failed: {err[:200]}")
            report.env_synced = False
        else:
            # Non-zero exit could mean changes needed
            report.add_warning(
                "uv sync --locked --dry-run returned non-zero. Environment may be out of sync."
            )
            report.env_synced = False


def _probe_writable_bytes(check_path: str, probe_bytes: int) -> tuple[bool, str | None]:
    """Try to actually reserve ``probe_bytes`` under ``check_path`` via posix_fallocate.

    On RunPod MooseFS each pod has a per-pod writable-bytes quota (~130GB) that is
    separate from, and far below, the share-level free space ``shutil.disk_usage``
    reports. The only reliable way to detect the quota is to attempt a real
    allocation: a small canary reservation that we immediately delete.

    Args:
        check_path: Directory under which to write the probe file.
        probe_bytes: Number of bytes to attempt to reserve. Keep this SMALL
            (a canary, ~1-2GB), NOT the full required free space — the goal is to
            detect EDQUOT/ENOSPC, not to reserve the experiment's footprint.

    Returns:
        (ok, fallback_reason). ``ok`` is True when the allocation succeeded.
        ``fallback_reason`` is set to a non-None string ONLY when the probe could
        not run (filesystem does not support fallocate); in that case the caller
        must fall back to ``shutil.disk_usage`` and ``ok`` is True. ``ok`` is
        False when the allocation was actively refused (EDQUOT/ENOSPC), with
        ``fallback_reason`` left None.

    Asserts probe_bytes > 0 — a zero-byte probe never exercises the quota.
    """
    assert probe_bytes > 0, f"probe_bytes must be positive, got {probe_bytes}"

    probe_path = Path(check_path) / ".preflight_disk_probe.tmp"
    fd = None
    try:
        probe_path.parent.mkdir(parents=True, exist_ok=True)
        fd = os.open(str(probe_path), os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        try:
            os.posix_fallocate(fd, 0, probe_bytes)
        except OSError as e:
            if e.errno in (errno.ENOSPC, errno.EDQUOT):
                return False, None
            if e.errno in (errno.EOPNOTSUPP, errno.ENOSYS, errno.EINVAL):
                # Filesystem doesn't support fallocate (tmpfs, some overlay FS,
                # macOS). Caller falls back to shutil.disk_usage.
                return True, f"posix_fallocate unsupported (errno={e.errno})"
            raise
        return True, None
    finally:
        if fd is not None:
            with contextlib.suppress(OSError):
                os.close(fd)
        with contextlib.suppress(OSError):
            probe_path.unlink()


def _quota_aware_headroom_gb(share_free_gb: float, quota_gb: float | None) -> tuple[float, str]:
    """Cap the usable headroom at the per-pod quota so over-quota footprints show.

    ``shutil.disk_usage`` reports share-level free (terabytes on RunPod MooseFS),
    which is blind to the per-pod EDQUOT quota. The small canary probe only fires
    once the quota is ALREADY exhausted; to surface an over-quota footprint BEFORE
    launch the usable headroom is ``min(quota_gb, share_free_gb)``.

    Args:
        share_free_gb: Share-level free space from ``shutil.disk_usage``.
        quota_gb: Per-pod writable-bytes quota in GB. None disables the cap (the
            headroom is then the raw share-level free, which CANNOT detect the
            quota — the basis string makes that explicit).

    Returns:
        (headroom_gb, basis) where ``basis`` names the binding signal so callers
        never mislabel a quota-blind number as "probed".
    """
    if quota_gb is None:
        return share_free_gb, "share-level free (quota cap disabled, over-quota undetectable)"
    if quota_gb < share_free_gb:
        return quota_gb, f"per-pod quota cap ({quota_gb:.0f}GB)"
    return share_free_gb, "share-level free (below per-pod quota cap)"


def check_disk_space(
    report: PreflightReport,
    min_free_gb: float,
    probe_gb: float = 1.0,
    quota_gb: float | None = RUNPOD_PER_POD_QUOTA_GB,
):
    """Check available disk space on /workspace (or /), quota-aware.

    Two distinct quota signals are combined:

    1. A real ``posix_fallocate`` canary probe catches an ALREADY-exhausted
       RunPod MooseFS per-pod EDQUOT quota (``shutil.disk_usage`` is blind to it).
    2. The usable headroom is capped at ``quota_gb`` so an over-quota *planned
       footprint* (one the small canary does NOT yet trip because the pod is not
       yet full) is still catchable by ``check_disk_budget``. Without this cap the
       headroom would be the share-level free (terabytes), and the budget check
       would be a no-op on exactly the filesystem it exists to protect.

    ``shutil.disk_usage`` is kept solely as the human-readable free-space reporter
    and as the share-level term of the headroom cap.

    Args:
        report: Mutated in place with disk findings (``disk_free_gb``,
            ``disk_probed_headroom_gb``, ``disk_headroom_basis``).
        min_free_gb: Minimum free space required to run.
        probe_gb: Size of the canary allocation, in GB. Small by design (default
            1GB) — it detects the quota, it does not reserve the full footprint.
        quota_gb: Per-pod writable-bytes quota in GB used to cap usable headroom.
            Defaults to ``RUNPOD_PER_POD_QUOTA_GB``. Pass a larger value for pods
            provisioned with an explicit storage spec, or None to disable the cap
            (the headroom then cannot detect over-quota footprints).
    """
    check_path = _disk_check_path()

    # Human-readable share-level free space (NOT the sole go/no-go signal).
    try:
        usage = shutil.disk_usage(check_path)
        report.disk_free_gb = usage.free / (1024**3)
    except Exception as e:
        report.add_warning(f"Could not read disk usage on {check_path}: {e}")

    probe_bytes = max(1, int(probe_gb * (1024**3)))
    try:
        ok, fallback_reason = _probe_writable_bytes(check_path, probe_bytes)
    except OSError as e:
        report.add_warning(f"Could not run disk-quota probe on {check_path}: {e}")
        ok, fallback_reason = True, f"probe raised {e}"

    headroom_gb, headroom_basis = _quota_aware_headroom_gb(report.disk_free_gb, quota_gb)

    if fallback_reason is not None:
        # Probe could not run — fall back to shutil.disk_usage for the ALREADY-
        # exhausted signal, but STILL cap headroom at the static quota so a
        # planned over-quota footprint is caught downstream.
        report.add_warning(
            f"Disk-quota probe skipped on {check_path}: {fallback_reason}. "
            f"Falling back to shutil.disk_usage; the live per-pod EDQUOT quota "
            f"cannot be detected, so headroom is capped at the static quota "
            f"({headroom_basis})."
        )
        report.disk_probed_headroom_gb = headroom_gb
        report.disk_headroom_basis = headroom_basis
        if report.disk_free_gb < min_free_gb:
            report.add_error(
                f"Only {report.disk_free_gb:.1f}GB free on {check_path} "
                f"(need {min_free_gb:.0f}GB). Clean up models/checkpoints."
            )
        elif report.disk_free_gb < min_free_gb * 2:
            report.add_warning(f"{report.disk_free_gb:.1f}GB free on {check_path} — getting low")
        return

    if not ok:
        # The pod refused even the small canary — quota is exhausted.
        report.disk_probed_headroom_gb = 0.0
        report.disk_headroom_basis = "per-pod quota exhausted (canary refused)"
        report.add_error(
            f"Disk-quota probe FAILED on {check_path}: cannot allocate even "
            f"{probe_gb:.1f}GB (EDQUOT/ENOSPC). Share-level free reports "
            f"{report.disk_free_gb:.1f}GB, but this pod has exhausted its per-pod "
            f"writable-bytes quota. Clean up models/checkpoints or provision a "
            f"larger volume."
        )
        return

    # Probe of probe_gb succeeded, so the quota is not YET exhausted. The usable
    # headroom is capped at the per-pod quota (NOT the terabyte-scale share-level
    # free) so an over-quota planned footprint is caught by check_disk_budget.
    report.disk_probed_headroom_gb = headroom_gb
    report.disk_headroom_basis = headroom_basis
    if report.disk_free_gb < min_free_gb:
        report.add_error(
            f"Only {report.disk_free_gb:.1f}GB free on {check_path} "
            f"(need {min_free_gb:.0f}GB). Clean up models/checkpoints."
        )
    elif report.disk_free_gb < min_free_gb * 2:
        report.add_warning(f"{report.disk_free_gb:.1f}GB free on {check_path} — getting low")


def estimate_footprint_gb(
    base_model_gb: float,
    n_cells: int,
    materialize_merged: bool = True,
) -> float:
    """Estimate peak disk footprint (GB) for a multi-cell experiment.

    A rough budgeting aid for ``check_disk_budget`` — NOT an exact accounting.
    Each cell holds one base-model-sized checkpoint on disk; when merged adapters
    are materialized, a cell briefly holds a second base-model-sized copy
    (adapter + merged) at peak.

    Args:
        base_model_gb: On-disk size of one base-model / checkpoint copy in GB.
        n_cells: Number of cells (conditions x seeds) whose checkpoints coexist
            on disk at peak. Use 1 for a strictly sequential, delete-after-each run.
        materialize_merged: If True, account for the transient merged-adapter copy
            (the LoRA-merge step where adapter + merged both exist).

    Returns:
        Estimated peak footprint in GB.

    Asserts base_model_gb >= 0 and n_cells >= 1.
    """
    assert base_model_gb >= 0, f"base_model_gb must be non-negative, got {base_model_gb}"
    assert n_cells >= 1, f"n_cells must be >= 1, got {n_cells}"

    per_cell = base_model_gb * (2.0 if materialize_merged else 1.0)
    return per_cell * n_cells


def check_disk_budget(report: PreflightReport, planned_footprint_gb: float | None):
    """FAIL when the estimated experiment footprint exceeds usable disk headroom.

    Usable headroom is ``report.disk_probed_headroom_gb`` — quota-capped by
    ``check_disk_space`` so it is NOT the terabyte-scale share-level free that
    ``shutil.disk_usage`` reports on RunPod MooseFS. The FAIL message names the
    headroom basis (``report.disk_headroom_basis``) so the number is never
    mislabeled as "probed" when it is a share-level / quota-capped estimate.

    Ranked remediation (cheapest first): LoRA-only (skip merged-adapter
    materialization), sequentialize multi-cell sweeps, provision a larger volume.

    Args:
        report: Mutated in place. Reads ``disk_probed_headroom_gb`` +
            ``disk_headroom_basis`` (set by ``check_disk_space``); call this AFTER
            ``check_disk_space``.
        planned_footprint_gb: Estimated peak footprint in GB. None => skip (no
            budget information supplied).
    """
    if planned_footprint_gb is None:
        return

    headroom = report.disk_probed_headroom_gb
    basis = report.disk_headroom_basis
    if planned_footprint_gb > headroom:
        report.add_error(
            f"Disk budget exceeded: planned footprint {planned_footprint_gb:.1f}GB "
            f"> usable headroom {headroom:.1f}GB (basis: {basis}). Remediation, "
            f"cheapest first: "
            f"(1) LoRA-only — skip merged-adapter materialization to halve per-cell "
            f"disk; (2) sequentialize — run conditions/seeds one at a time and "
            f"delete each checkpoint before the next so peak disk = one cell; "
            f"(3) provision a larger volume / pod with explicit storage spec."
        )


def check_gpus(report: PreflightReport, require_gpu: bool, min_free_mb: int):
    """Check GPU availability and memory."""
    rc, out, err = _run(
        [
            "nvidia-smi",
            "--query-gpu=index,memory.total,memory.used,memory.free",
            "--format=csv,noheader,nounits",
        ]
    )
    if rc != 0:
        if require_gpu:
            report.add_error(f"nvidia-smi failed: {err}. No GPUs available?")
        else:
            report.add_warning("nvidia-smi not available (no GPU)")
        return

    for line in out.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 4:
            continue
        gpu_id, total, used, free = int(parts[0]), int(parts[1]), int(parts[2]), int(parts[3])

        # Check for processes on this GPU
        prc, pout, _ = _run(
            [
                "nvidia-smi",
                f"--id={gpu_id}",
                "--query-compute-apps=pid",
                "--format=csv,noheader",
            ]
        )
        n_procs = len([x for x in pout.splitlines() if x.strip()]) if prc == 0 else 0

        gpu_info = {
            "id": gpu_id,
            "memory_total_mb": total,
            "memory_used_mb": used,
            "memory_free_mb": free,
            "processes": n_procs,
        }
        report.gpu_info.append(gpu_info)

        # Zombie detection: memory used but no processes
        if used > 5000 and n_procs == 0:
            report.add_warning(
                f"GPU {gpu_id}: {used}MB used but no processes — "
                f"possible zombie. Fix: restart container."
            )

    # Check if any GPU has enough free memory
    if require_gpu:
        max_free = max((g["memory_free_mb"] for g in report.gpu_info), default=0)
        if max_free < min_free_mb:
            report.add_error(
                f"No GPU with {min_free_mb:,}MB free (best: {max_free:,}MB). "
                f"Wait for running jobs or use a different pod."
            )


def check_hf_home(report: PreflightReport):
    """Check ``HF_HOME`` matches the canonical per-environment default.

    Three-way (mirrors :func:`env._hf_home_default`):

    * Cluster: expects ``$SCRATCH/.cache/huggingface``.
    * RunPod:  expects ``/workspace/.cache/huggingface``.
    * Local:   no canonical path; only warn if HF_HOME is empty.
    """
    hf_home = os.environ.get("HF_HOME", "")

    if is_cluster_env() or is_runpod_env():
        expected = _hf_home_default()
        if not hf_home:
            report.add_warning(
                f"HF_HOME not set. Setting to {expected}. "
                "Call load_dotenv() or source env_setup.sh first."
            )
            os.environ["HF_HOME"] = expected
        elif hf_home != expected:
            report.add_warning(
                f"HF_HOME={hf_home} (expected {expected}). Models may download to wrong location."
            )


def check_env_vars(report: PreflightReport, required: list[str]):
    """Check that required environment variables are set."""
    for var in required:
        val = os.environ.get(var, "")
        if not val:
            report.add_error(f"Missing env var: {var}. Check .env file.")
        elif len(val) < 5:
            report.add_warning(f"Env var {var} looks suspiciously short: '{val[:3]}...'")


def check_vllm_transformers_compat(report: PreflightReport):
    """Refuse to proceed when vLLM 0.11.x is resolved against transformers >=5.

    vLLM 0.11.0 calls `tokenizer.all_special_tokens_extended`, which transformers 5.x
    removed. Every fresh pod hits this 10 sec into the first `LLM(...)` init. This has
    recurred across issues #238, #261, #263, #269, #331, #354, #368 — caught here so
    the next fresh pod fails preflight in <2 sec instead of crashing in vLLM later.
    """
    try:
        import transformers
        import vllm
    except ImportError as e:
        report.add_warning(f"Could not import vllm/transformers for compat check: {e}")
        return

    t_ver = transformers.__version__
    v_ver = vllm.__version__
    t_major = int(t_ver.split(".")[0])
    v_minor = ".".join(v_ver.split(".")[:2])
    if v_minor in {"0.11"} and t_major >= 5:
        report.add_error(
            f"vLLM/transformers version skew: vllm=={v_ver} + transformers=={t_ver}. "
            f"vLLM 0.11.x calls tokenizer.all_special_tokens_extended which transformers "
            f">=5 removed. Every LLM(...) instantiation will crash. Fix: pin "
            f"`transformers>=4.46,<5.0` in pyproject.toml and re-run `uv sync --locked`. "
            f"See .claude/agent-memory/experimenter/feedback_vllm0110_transformers5_breakage.md"
        )


def check_connectivity(report: PreflightReport):
    """Quick check that HF Hub and WandB are reachable."""
    # HF Hub
    rc, _, _ = _run(
        [
            "python3",
            "-c",
            "import urllib.request; urllib.request.urlopen('https://huggingface.co', timeout=5)",
        ],
        timeout=10,
    )
    if rc != 0:
        report.add_warning("Cannot reach huggingface.co — model uploads will fail")

    # WandB
    wandb_check = (
        "import urllib.request; urllib.request.urlopen('https://api.wandb.ai/healthz', timeout=5)"
    )
    rc, _, _ = _run(["python3", "-c", wandb_check], timeout=10)
    if rc != 0:
        report.add_warning("Cannot reach api.wandb.ai — result uploads will fail")


def check_hf_storage(report: PreflightReport):
    """Non-fatal WARN when account HF public storage exceeds the soft ceiling.

    Advisory only: never adds an error, never raises — an unreachable HF API
    degrades to an 'unknown headroom' warning, and a non-parseable ceiling/TTL
    env value (the helper's deliberate ``ValueError``) is caught and reported
    as a warning here (it still propagates at the fail-loud persist gate in
    ``train/trainer.py``). See ``.claude/rules/upload-policy.md``
    § HF storage-quota 403 for the incident this fronts (#541/#552).
    """
    try:
        from explore_persona_space.orchestrate.hub import check_hf_storage_headroom

        h = check_hf_storage_headroom()
    except Exception as e:
        report.add_warning(f"HF storage headroom check failed ({e}) — headroom unknown")
        return
    report.hf_storage_used_tb = h.used_tb
    report.hf_storage_ceiling_tb = h.ceiling_tb
    report.hf_storage_basis = h.basis
    if h.basis == "disabled":
        return
    if h.used_tb is None:
        report.add_warning(
            f"HF public-storage usage unknown ({h.basis}) — cannot verify upload headroom"
        )
    elif h.over_ceiling:
        report.add_warning(
            f"HF public storage {h.used_tb:.2f} TB exceeds soft ceiling "
            f"{h.ceiling_tb:.1f} TB ({h.n_repos} repos, {h.basis}) — LFS uploads "
            f"(adapters/checkpoints) will 403 at the hard quota; see "
            f".claude/rules/upload-policy.md § HF storage-quota 403"
        )


def preflight_check(
    require_gpu: bool = True,
    min_disk_gb: float = 50.0,
    min_gpu_free_mb: int = 70_000,
    required_env_vars: list[str] | None = None,
    check_code_sync: bool = True,
    planned_footprint_gb: float | None = None,
    per_pod_quota_gb: float | None = RUNPOD_PER_POD_QUOTA_GB,
) -> PreflightReport:
    """Run all pre-experiment checks.

    Args:
        require_gpu: If True, fail when no GPU has enough free memory.
        min_disk_gb: Minimum free disk space in GB.
        min_gpu_free_mb: Minimum free GPU memory in MB for at least one GPU.
        required_env_vars: Env vars to check. Defaults to standard set.
        check_code_sync: Whether to check git status and env sync.
        planned_footprint_gb: Estimated peak experiment disk footprint in GB. When
            supplied, the disk-budget check FAILs if it exceeds usable (quota-
            capped) headroom. None (default) => skip the budget check, so existing
            callers are unaffected.
        per_pod_quota_gb: RunPod MooseFS per-pod writable-bytes quota in GB used to
            cap usable disk headroom (defaults to ``RUNPOD_PER_POD_QUOTA_GB``).
            None disables the cap (over-quota footprints become undetectable).

    Returns:
        PreflightReport with pass/fail status and details.
    """
    if required_env_vars is None:
        required_env_vars = [
            "WANDB_API_KEY",
            "HF_TOKEN",
            "ANTHROPIC_API_KEY",
        ]

    project_root = _find_project_root()
    report = PreflightReport()

    # Load .env first so env var checks work. Use the canonical loader so a
    # linked worktree without its own .env falls back to the main worktree's.
    try:
        from explore_persona_space.orchestrate.env import load_dotenv as _load_dotenv

        _load_dotenv()
    except ImportError:
        report.add_warning("python-dotenv not installed — cannot load .env")

    # Set HF_HOME early — three-way: cluster → $SCRATCH, RunPod →
    # /workspace, local → project-local. See env._hf_home_default.
    if is_cluster_env() or is_runpod_env():
        os.environ.setdefault("HF_HOME", _hf_home_default())

    # Cluster bypasses the RunPod MooseFS 130 GB cap: $SCRATCH has a
    # per-user quota the cluster admins set (multi-TB on Nibi/Fir), not
    # the RunPod cap. The caller can still override per-pod-quota-gb
    # explicitly when a RunPod pod was provisioned with a custom volume.
    effective_quota_gb = None if is_cluster_env() else per_pod_quota_gb

    # Run all checks
    if check_code_sync:
        check_git_status(report, project_root)
        check_env_sync(report, project_root)

    check_disk_space(report, min_disk_gb, quota_gb=effective_quota_gb)
    check_disk_budget(report, planned_footprint_gb)
    check_gpus(report, require_gpu, min_gpu_free_mb)
    check_hf_home(report)
    check_env_vars(report, required_env_vars)
    check_vllm_transformers_compat(report)
    check_connectivity(report)
    check_hf_storage(report)

    return report


def require_preflight(
    min_disk_gb: float = 50.0,
    require_gpu: bool = True,
    min_gpu_free_mb: int = 70_000,
) -> PreflightReport:
    """Run preflight checks and abort if any critical failures.

    Call at the top of experiment scripts.
    """
    report = preflight_check(
        min_disk_gb=min_disk_gb,
        require_gpu=require_gpu,
        min_gpu_free_mb=min_gpu_free_mb,
    )
    logger.info(report.summary())

    if not report.ok:
        logger.error("Pre-flight check FAILED. Fix errors before running.")
        sys.exit(1)

    return report


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run pre-flight checks")
    parser.add_argument("--no-gpu", action="store_true", help="Don't require GPU")
    parser.add_argument("--min-disk", type=float, default=50.0, help="Min disk GB")
    parser.add_argument(
        "--planned-footprint-gb",
        type=float,
        default=None,
        help="Estimated peak experiment disk footprint in GB; FAILs preflight if "
        "it exceeds usable (quota-capped) headroom. Omit to skip the budget check.",
    )
    parser.add_argument(
        "--per-pod-quota-gb",
        type=float,
        default=RUNPOD_PER_POD_QUOTA_GB,
        help="RunPod MooseFS per-pod writable-bytes quota in GB used to cap usable "
        "disk headroom (default %(default)s). Pass a larger value for pods with an "
        "explicit storage spec. Use a negative value to disable the cap (over-quota "
        "footprints then become undetectable).",
    )
    parser.add_argument("--json", action="store_true", help="Output as JSON")
    parser.add_argument(
        "--pipeline-check",
        action="store_true",
        help="Run integration tests (pytest tests/integration/ -m integration) after preflight",
    )
    args = parser.parse_args()

    # A negative quota means "disable the cap" (argparse cannot pass None cleanly).
    per_pod_quota_gb = None if args.per_pod_quota_gb < 0 else args.per_pod_quota_gb

    report = preflight_check(
        require_gpu=not args.no_gpu,
        min_disk_gb=args.min_disk,
        planned_footprint_gb=args.planned_footprint_gb,
        per_pod_quota_gb=per_pod_quota_gb,
    )

    if args.json:
        print(
            json.dumps(
                {
                    "ok": report.ok,
                    "errors": report.errors,
                    "warnings": report.warnings,
                    "gpu_info": report.gpu_info,
                    "disk_free_gb": report.disk_free_gb,
                    "disk_probed_headroom_gb": report.disk_probed_headroom_gb,
                    "disk_headroom_basis": report.disk_headroom_basis,
                    "hf_storage_used_tb": report.hf_storage_used_tb,
                    "hf_storage_ceiling_tb": report.hf_storage_ceiling_tb,
                    "hf_storage_basis": report.hf_storage_basis,
                    "git_status": report.git_status,
                    "env_synced": report.env_synced,
                },
                indent=2,
            )
        )
    else:
        logger.info(report.summary())

    if not report.ok:
        sys.exit(1)

    if args.pipeline_check:
        logger.info("Running integration tests...")
        rc, stdout, stderr = _run(
            [sys.executable, "-m", "pytest", "tests/integration/", "-m", "integration", "-x", "-v"],
            timeout=600,
        )
        if stdout:
            print(stdout)
        if stderr:
            print(stderr, file=sys.stderr)
        if rc != 0:
            logger.error("Integration tests FAILED (exit code %d)", rc)
            sys.exit(rc)
        logger.info("Integration tests PASSED")

    sys.exit(0)
