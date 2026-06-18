"""Shared environment setup for worker processes.

Three-way environment discriminator
-----------------------------------

This module distinguishes three runtime environments and configures
``HF_HOME`` / dotenv resolution per-environment:

1. **Cluster** (SLURM / Compute Canada DRAC): ``"SLURM_JOB_ID" in
   os.environ``. ``HF_HOME`` defaults to ``$SCRATCH/.cache/huggingface``;
   the ``/workspace/explore-persona-space/.env`` dotenv fallback is
   skipped (secrets arrive via an rsync'd file the sbatch sources
   directly). Used by the SLURM cluster backend (see
   ``src/explore_persona_space/backends/``).
2. **RunPod** (cloud ephemeral pod): ``RUNPOD_POD_ID`` set in the
   environment OR ``/workspace`` is a real MOUNT POINT
   (``os.path.ismount``), and we are NOT on a cluster. ``HF_HOME``
   defaults to ``/workspace/.cache/huggingface``; the dotenv fallback at
   ``/workspace/explore-persona-space/.env`` is honored. A plain
   ``/workspace`` *directory* does NOT route as RunPod — see
   :func:`is_runpod_env` for why (2026-06-11 dev-VM incident).
3. **Local VM** (dev box): neither of the above. ``HF_HOME`` defaults
   to the user-level shared cache ``~/.cache/huggingface`` — one cache
   per user, NOT per-checkout (a per-checkout ``<project_root>/cache``
   default let every git worktree grow its own multi-GB HF cache; see
   :func:`_hf_home_default`); dotenv resolution falls back to the main
   git worktree's ``.env``.

The cluster check is FIRST because a SLURM allocation on a cluster that
happens to mount a ``/workspace`` (vanishingly unlikely in practice, but
defensive) must still route as cluster.
"""

import logging
import os
import subprocess
import sys
from pathlib import Path

from dotenv import load_dotenv as _dotenv_load

logger = logging.getLogger(__name__)

# Project root: three levels up (src/explore_persona_space/orchestrate/env.py -> project root)
_PROJECT_ROOT = Path(__file__).resolve().parents[3]


# ---------------------------------------------------------------------------
# Three-way environment discriminator
# ---------------------------------------------------------------------------


def is_cluster_env() -> bool:
    """True iff we are inside a SLURM allocation (any cluster).

    Discriminator: ``SLURM_JOB_ID`` is the only env var SLURM is
    guaranteed to set inside every job script across every cluster
    (DRAC's Nibi/Fir, Mila, etc.). Tested against the sbatch-rendered
    job environment.
    """
    return "SLURM_JOB_ID" in os.environ


def is_runpod_env() -> bool:
    """True iff we are on a RunPod pod (canonical ``/workspace`` volume mount).

    Two clauses, either suffices (after the cluster check):

    * ``RUNPOD_POD_ID`` set in the environment — RunPod injects it into
      the container env. Belt-and-braces clause: nothing on the dev VM
      or a GCE instance sets it, so it can only ADD pod detection.
    * ``os.path.ismount("/workspace")`` — the load-bearing clause. Every
      pod this project provisions mounts its volume at ``/workspace``
      (``runpod_api.create_pod`` sends ``volumeMountPath: "/workspace"``;
      network-volume pods mount MooseFS there), so ``/workspace`` is a
      real mount point on pods. A plain ``/workspace`` DIRECTORY must
      NOT match: on 2026-06-11 a ``sudo mkdir -p /workspace`` on the dev
      VM (created to land GCP-lane sentinels at their VM-absolute path)
      made the previous ``Path("/workspace").exists()`` discriminator
      route every dev-VM process as RunPod, redirecting ``HF_HOME`` to a
      redundant 16 GB cache on the 99%-full root disk. GCE instances
      from the GCP lane also carry a plain-dir ``/workspace`` (startup
      script ``mkdir -p /workspace/eps-issue-<N>`` on the boot disk) and
      must route as local — their ``HF_HOME`` is exported explicitly by
      the startup script.

    Mutually exclusive with :func:`is_cluster_env` — a cluster
    allocation that also happened to mount ``/workspace`` would still
    route as cluster. This preserves the byte-for-byte RunPod behavior
    when ``SLURM_JOB_ID`` is unset.
    """
    if is_cluster_env():
        return False
    if os.environ.get("RUNPOD_POD_ID"):
        return True
    return os.path.ismount("/workspace")


def _hf_home_default() -> str:
    """Per-environment default for ``HF_HOME``.

    * Cluster: ``$SCRATCH/.cache/huggingface``. Falls back to
      ``$HOME/.cache/huggingface`` when ``SCRATCH`` is somehow unset
      (defensive — DRAC always sets it).
    * RunPod: ``/workspace/.cache/huggingface``.
    * Local: ``~/.cache/huggingface`` (the user-level shared cache).
      Deliberately NOT ``<project_root>/cache/huggingface``: the project
      root resolves per-checkout, so a per-checkout default gives every
      git worktree under ``.claude/worktrees/`` its OWN full HF cache
      (2026-06-12 disk triage: two worktrees each held a complete ~14 GB
      Qwen-2.5-7B-Instruct snapshot, driving the VM root disk to 99%).
    """
    if is_cluster_env():
        scratch = os.environ.get("SCRATCH")
        if scratch:
            return str(Path(scratch) / ".cache" / "huggingface")
        # Last-resort: $HOME — better than crashing the worker on a
        # missing $SCRATCH (which would itself be a configuration bug).
        home = os.environ.get("HOME") or str(Path.home())
        return str(Path(home) / ".cache" / "huggingface")
    if is_runpod_env():
        return "/workspace/.cache/huggingface"
    return str(Path.home() / ".cache" / "huggingface")


def get_project_root() -> Path:
    """Return the project root directory."""
    return _PROJECT_ROOT


def get_output_dir() -> Path:
    """Return the output directory, configurable via MED_OUTPUT_DIR env var."""
    return Path(os.environ.get("MED_OUTPUT_DIR", str(_PROJECT_ROOT)))


def resolve_dotenv_path(start: Path | None = None) -> Path | None:
    """Find the .env for this checkout, walking past worktrees if needed.

    Search order:
      1. ``<start>/.env`` — worktree-local (or `start` arg explicit).
      2. Main git worktree's ``.env`` — via ``git rev-parse --git-common-dir``,
         whose parent is the main worktree root. Linked worktrees do not
         inherit the gitignored ``.env`` from the main worktree, so a
         driver run from ``/workspace/wt-issue-N/`` must fall back to
         ``/workspace/explore-persona-space/.env``.
      3. ``/workspace/explore-persona-space/.env`` — pod-canonical fallback
         for the case where (2) fails (no git, detached state, etc.) but
         we know the bootstrap script always pushes ``.env`` there.
         **Cluster-environment skip:** when :func:`is_cluster_env` is True
         we never consult this path — secrets on the cluster arrive via
         a freshly-rsync'd file the sbatch sources directly, and probing
         ``/workspace`` from a SLURM compute node would either be slow
         (NFS/MooseFS not present) or, worse, leak through to an unrelated
         mount.

    Returns the first existing path, or None if no ``.env`` found anywhere.
    """
    if start is None:
        start = _PROJECT_ROOT
    seen: set[Path] = set()
    candidates: list[Path] = []

    def _push(p: Path) -> None:
        rp = p.resolve() if p.exists() else p
        if rp not in seen:
            seen.add(rp)
            candidates.append(p)

    _push(start / ".env")

    try:
        result = subprocess.run(
            ["git", "rev-parse", "--path-format=absolute", "--git-common-dir"],
            capture_output=True,
            text=True,
            cwd=str(start),
            timeout=5,
            check=False,
        )
        if result.returncode == 0 and result.stdout.strip():
            git_common = Path(result.stdout.strip())
            _push(git_common.parent / ".env")
    except (subprocess.SubprocessError, FileNotFoundError, OSError):
        pass

    # Pod-canonical fallback — skipped on the cluster (secrets arrive via
    # rsync'd file, not this resolver).
    if not is_cluster_env():
        _push(Path("/workspace/explore-persona-space/.env"))

    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return None


def load_dotenv(env_path: str | None = None):
    """Load .env file into os.environ (does not overwrite existing vars).

    When ``env_path`` is None, resolves the canonical .env via
    :func:`resolve_dotenv_path`, which walks to the main git worktree
    when the local checkout is a linked worktree without its own .env.

    Also sets HF_HOME to the unified cache location if not already set.
    """
    if env_path is None:
        resolved = resolve_dotenv_path()
        if resolved is None:
            logger.warning(
                "No .env found near %s, in main git worktree, or at the "
                "pod-canonical /workspace/explore-persona-space/.env. "
                "Credentialed calls will fail unless the env is already set.",
                _PROJECT_ROOT,
            )
            env_path = str(_PROJECT_ROOT / ".env")
        else:
            env_path = str(resolved)
            if resolved.resolve() != (_PROJECT_ROOT / ".env").resolve():
                logger.info(
                    ".env loaded from %s (PROJECT_ROOT=%s does not have its own .env)",
                    resolved,
                    _PROJECT_ROOT,
                )
    _dotenv_load(env_path, override=False)

    # Unified HF cache, three-way branch (see :func:`_hf_home_default`):
    #   cluster ($SLURM_JOB_ID)  → $SCRATCH/.cache/huggingface
    #   RunPod (/workspace)       → /workspace/.cache/huggingface
    #   local                     → ~/.cache/huggingface (user-level shared)
    os.environ.setdefault("HF_HOME", _hf_home_default())


def setup_worker(gpu_id: int):
    """Configure a worker subprocess: GPU, paths, env vars.

    Call this at the start of any ProcessPoolExecutor worker function.
    """
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    extra_pypath = os.environ.get("EXTRA_PYTHONPATH", "")
    if extra_pypath and extra_pypath not in sys.path:
        sys.path.insert(0, extra_pypath)

    # Build LD_LIBRARY_PATH dynamically from torch's actual location
    try:
        import torch as _torch

        torch_lib = str(Path(_torch.__file__).parent / "lib")
    except ImportError:
        torch_lib = ""

    # Find CUDA lib dir. Prefer $CUDA_HOME (set by most cluster modules) so we
    # honor whatever toolkit the host has actually loaded; only fall back to
    # the hard-coded RunPod-ish locations when the env hint is absent.
    cuda_lib = ""
    cuda_home = os.environ.get("CUDA_HOME") or os.environ.get("CUDA_PATH")
    if cuda_home:
        candidate = str(Path(cuda_home) / "lib64")
        if Path(candidate).exists():
            cuda_lib = candidate
    if not cuda_lib:
        for cuda_version in ["12.4", "12.6", "12.1", "11.8"]:
            candidate = f"/usr/local/cuda-{cuda_version}/lib64"
            if Path(candidate).exists():
                cuda_lib = candidate
                break

    existing = os.environ.get("LD_LIBRARY_PATH", "")
    parts = [p for p in [torch_lib, cuda_lib, existing] if p]
    os.environ["LD_LIBRARY_PATH"] = ":".join(parts)
    # Unified HF cache, three-way branch (see :func:`_hf_home_default`):
    # cluster → $SCRATCH/.cache/huggingface; RunPod → /workspace/.cache/
    # huggingface; local → ~/.cache/huggingface (user-level shared).
    # Worker subprocesses must NOT write to the RunPod path on the
    # cluster (no /workspace mount).
    os.environ.setdefault("HF_HOME", _hf_home_default())

    load_dotenv()


def check_gpu_memory(min_free_mb: int = 20_000) -> bool:
    """Check that the assigned GPU has sufficient free memory.

    Returns True if memory is sufficient, False otherwise.
    """
    try:
        gpu_id = os.environ.get("CUDA_VISIBLE_DEVICES", "0")
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=memory.free",
                "--format=csv,noheader,nounits",
                f"--id={gpu_id}",
            ],
            capture_output=True,
            text=True,
            timeout=10,
        )
        free_mb = int(result.stdout.strip().split("\n")[0])
        if free_mb < min_free_mb:
            import warnings

            warnings.warn(
                f"GPU {gpu_id} has only {free_mb}MB free (need {min_free_mb}MB). Training may OOM.",
                RuntimeWarning,
                stacklevel=2,
            )
            return False
        return True
    except Exception as e:
        logger.warning("Could not check GPU memory: %s. Failing safe.", e)
        return False  # Can't check → fail safe, don't proceed optimistically
