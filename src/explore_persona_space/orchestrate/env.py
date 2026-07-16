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
   git worktree's ``.env``. The SHARED dev VM sub-case (positive
   detection via :func:`is_shared_vm_env` — the ``/mnt/eps-data`` data
   disk mounted OR hostname ``cia-benchmark-vm``) additionally
   setdefaults BLAS/torch thread caps (#847) in :func:`load_dotenv` and
   :func:`setup_worker`. Torch freezes its intra-op pool from
   ``OMP_NUM_THREADS`` at IMPORT time, so entrypoints must call
   ``load_dotenv()`` BEFORE importing torch/numpy for the cap to bind
   in-process (subprocesses are capped regardless via env inheritance).
   The same shared-VM hook also redirects the heavy HF caches
   (``HF_HUB_CACHE`` / ``HF_XET_CACHE``) onto the ``/mnt/eps-data`` data
   disk (#1369) — ``HF_HOME`` itself is deliberately left at
   ``~/.cache/huggingface`` (token file, datasets cache).

The cluster check is FIRST because a SLURM allocation on a cluster that
happens to mount a ``/workspace`` (vanishingly unlikely in practice, but
defensive) must still route as cluster.
"""

import logging
import os
import platform
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


_SHARED_VM_HOSTNAME = "cia-benchmark-vm"
_SHARED_VM_DATA_DISK = "/mnt/eps-data"  # #681 data disk — exists ONLY on the shared VM
_DEFAULT_VM_THREAD_CAP = 8  # diagnosis §6 (#847): each 32-thread job realized only ~5-6 cores
_THREAD_CAP_KEYS = (
    "OMP_NUM_THREADS",  # OpenMP (torch intra-op pool size, read at torch import)
    "MKL_NUM_THREADS",  # MKL (takes precedence over OMP for MKL ops)
    "OPENBLAS_NUM_THREADS",  # OpenBLAS (numpy pip wheels)
    "NUMEXPR_NUM_THREADS",  # numexpr (pandas eval paths) — cheap defense-in-depth
)


def is_shared_vm_env() -> bool:
    """True iff we are on the SHARED dev VM (positive detection; fails OPEN).

    GCE instances from the GCP lane route as *local* in the three-way
    discriminator (plain-dir ``/workspace`` — see :func:`is_runpod_env`),
    so "local" must NOT imply "shared VM". Positive signals only; anywhere
    else (pods, GCE, SLURM, laptops, novel dedicated boxes) returns False:

    * ``EPS_SHARED_VM=1/0`` — explicit override, both directions.
    * ``os.path.ismount(_SHARED_VM_DATA_DISK)`` — the #681 data disk.
    * ``platform.node() == _SHARED_VM_HOSTNAME`` — belt-and-braces if the
      disk is detached; ``platform.node()`` returns ``""`` on failure
      (fails open).
    """
    forced = os.environ.get("EPS_SHARED_VM")
    if forced is not None and forced.strip() != "":
        return forced.strip().lower() not in ("0", "false", "no", "off")
    if is_cluster_env() or is_runpod_env():
        return False  # defensive; mirrors the cluster-first ordering above
    if os.path.ismount(_SHARED_VM_DATA_DISK):
        return True
    return platform.node() == _SHARED_VM_HOSTNAME


def _apply_shared_vm_thread_caps() -> None:
    """setdefault BLAS/OpenMP thread caps on the shared VM only (#847).

    Torch sizes its intra-op pool from ``OMP_NUM_THREADS`` at IMPORT time
    and never re-reads it, so this must run before ``import torch`` in the
    process (``load_dotenv`` precedes torch imports in the repo
    entrypoints; ``setup_worker`` calls this before its own torch import).
    Even when torch is already imported, the setdefault still caps every
    SUBPROCESS. Fork-started children inherit the parent's env caps
    anyway; the ``setup_worker`` placement helps spawn-start /
    pre-import workers. Explicit launch-time values always win
    (setdefault). Per-script ``torch.set_num_threads(...)`` takes
    precedence over env either way.
    """
    if not is_shared_vm_env():
        return
    raw = os.environ.get("EPS_VM_THREAD_CAP")
    if raw is None:
        cap = _DEFAULT_VM_THREAD_CAP
    elif raw.strip() == "":
        return  # explicitly disabled
    else:
        cap = int(raw)  # malformed value raises ValueError — fail loud
        if cap <= 0:
            return  # 0 (or negative) disables
    for key in _THREAD_CAP_KEYS:
        os.environ.setdefault(key, str(cap))


def _apply_shared_vm_hf_cache_redirect() -> None:
    """setdefault HF_HUB_CACHE + HF_XET_CACHE onto the #681 data disk (shared VM only).

    The boot disk ``/`` (485 GB) hit 100% twice on 2026-07-15 (#1073 -> #1369):
    the HF hub cache (~97 GB) and the transient xet chunk cache (~11 GB during
    prefix staging) both default under ``~/.cache/huggingface`` on ``/``.
    Redirect BOTH heavy caches onto the existing user-owned data-disk cache
    (``/mnt/eps-data/<user>/huggingface-cache/{hub,xet}`` — it already holds
    the project data-repo dataset cache from prior manual redirects; HF
    clients mkdir cache dirs on demand, so no migration step is needed).

    Deliberately NOT ``HF_HOME``: the token file (``HF_HOME/token``), the
    ``datasets`` processed cache (~3.7 GB), ``stored_tokens``, and ``modules``
    stay at ``~/.cache/huggingface`` untouched — only the heavy caches move.

    Ordering: ``HF_HUB_CACHE`` is frozen into ``huggingface_hub.constants`` at
    IMPORT time (same constraint as the #745 flags above), so this must run
    before the process's first ``import huggingface_hub`` — importing THIS
    module pulls no huggingface_hub (verified #1369), so any entrypoint that
    calls load_dotenv() first gets the constant right. ``HF_XET_CACHE`` is
    read by the compiled hf_xet crate from the PROCESS ENV at transfer time
    (hf_xet 1.4.3, xet_runtime configuration_utils fallback chain
    HF_XET_CACHE -> HF_HOME/xet -> XDG_CACHE_HOME), so it escapes the freeze.

    Residual (named, accepted): a FUTURE raw-cron python entrypoint that
    imports huggingface_hub at module top WITHOUT load_dotenv() gets neither
    the profile export (cron reads no profile) nor this setdefault in time —
    zero such consumers exist today (#1369 fact-check); new cron authors:
    call load_dotenv() first or export the keys in the wrapper.

    setdefault-only — an explicit launch-time value always wins.
    ``EPS_VM_HF_CACHE_REDIRECT=0`` (or false/no/off) disables. Fails OPEN off
    the shared VM: pods (/workspace/.cache/huggingface), GCE, SLURM, and
    laptops keep their lane defaults (see :func:`_hf_home_default`).
    """
    if not is_shared_vm_env():
        return
    if not os.path.ismount(_SHARED_VM_DATA_DISK):
        # Hostname-only detection with the data disk detached: redirecting
        # would mkdir-on-demand a plain dir on / UNDER the mountpoint —
        # worse than today's default. Mirrors the shell blocks' [ -d ] guard.
        return
    knob = os.environ.get("EPS_VM_HF_CACHE_REDIRECT", "")
    if knob.strip().lower() in ("0", "false", "no", "off"):
        return
    cache_root = Path(_SHARED_VM_DATA_DISK) / Path.home().name / "huggingface-cache"
    os.environ.setdefault("HF_HUB_CACHE", str(cache_root / "hub"))
    os.environ.setdefault("HF_XET_CACHE", str(cache_root / "xet"))


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
            # GCE/SLURM lanes have NO .env file by design — tokens arrive via
            # startup metadata / sbatch env (.claude/rules/gotchas.md
            # conditional-sourcing entry, #923). When credential env vars are
            # already ambient the message's own predicate ("will fail unless
            # the env is already set") is false, so log INFO, not WARNING.
            ambient = [
                k for k in ("HF_TOKEN", "WANDB_API_KEY", "ANTHROPIC_API_KEY") if os.environ.get(k)
            ]
            if ambient:
                logger.info(
                    "No .env found near %s; using ambient env credentials "
                    "(%s set) — expected on the GCE/SLURM lanes, which export "
                    "tokens via startup metadata / sbatch env, not a .env file.",
                    _PROJECT_ROOT,
                    ", ".join(ambient),
                )
            else:
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

    # Shared-VM HF cache redirect (#1369): heavy caches (hub snapshots + xet
    # chunks) onto the #681 data disk. AFTER _dotenv_load so a value in .env
    # counts as explicit config and wins (setdefault; mirrors the #847 caps).
    _apply_shared_vm_hf_cache_redirect()

    # Fast HF Hub uploads (#745). BELT-AND-SUSPENDERS only — the LOAD-BEARING
    # placement is the SHELL-level export (bootstrap_pod.sh / GCE prelude /
    # SLURM env block), because HF_HUB_ENABLE_HF_TRANSFER is frozen by
    # huggingface_hub.constants at IMPORT time, so a setdefault running after a
    # direct-upload script's top-level `import huggingface_hub` is too late for
    # that constant. This setdefault still helps a local-dev process that
    # imports huggingface_hub AFTER load_dotenv (e.g. lazy in-function imports).
    # HF_XET_HIGH_PERFORMANCE is the PRIMARY accelerator (the project repos use
    # the Xet backend — verified); it is read by the compiled hf_xet crate at
    # UPLOAD time, so it escapes the import freeze. HF_HUB_ENABLE_HF_TRANSFER is
    # the orthogonal LFS-path accelerator (future-proofing). setdefault
    # preserves an explicit =0 / HF_HUB_DISABLE_XET=1 (the #515/#931 xet workaround).
    os.environ.setdefault("HF_XET_HIGH_PERFORMANCE", "1")  # Xet path (primary)
    os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")  # LFS path (orthogonal)

    # Shared-VM BLAS/torch thread caps (#847). Placed AFTER _dotenv_load so an
    # OMP_NUM_THREADS in .env counts as explicit config and wins (setdefault).
    _apply_shared_vm_thread_caps()


def setup_worker(gpu_id: int):
    """Configure a worker subprocess: GPU, paths, env vars.

    Call this at the start of any ProcessPoolExecutor worker function.
    """
    # Shared-VM BLAS/torch thread caps (#847) — FIRST, before the torch import
    # below freezes the intra-op pool at its uncapped default.
    _apply_shared_vm_thread_caps()
    # Shared-VM HF cache redirect (#1369) — mirror of load_dotenv, and ALSO
    # before the torch import below (defensive: anything the import chain
    # pulls that freezes HF env at import time sees the redirected paths;
    # idempotent setdefault, same as the #745 flags below).
    _apply_shared_vm_hf_cache_redirect()

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
    # Fast HF Hub uploads (#745) — local-dev belt-and-suspenders, mirror of
    # load_dotenv (see the note there; shell-level export is the load-bearing
    # placement). setdefault preserves an explicit =0 / HF_HUB_DISABLE_XET=1.
    os.environ.setdefault("HF_XET_HIGH_PERFORMANCE", "1")  # Xet path (primary)
    os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")  # LFS path (orthogonal)

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
