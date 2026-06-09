"""SLURM cluster backend (DRAC robot-key submit, rsync-primary code sync).

This is slice-2 of the SLURM-backend plan
(``.claude/plans/2026-06-08_001932-slurm-cluster-backend-for-issue.md``).
It implements every piece of the real cluster path that does NOT require
a live cluster acceptance run — the renderer + submit + monitor wiring +
selector hookup. The cluster ladder (P0/P1/P2) is gated separately and
owned by the orchestrator.

Why this lives in one module
----------------------------

The cluster path has four moving parts that are tightly coupled (a
change in one usually drags the others):

1. **rsync code sync** — VM → ``$SCRATCH/eps/issue-<N>``. Pinned flag
   set: ``-a --delete --partial --mkpath`` (P0(a) finding: intermediate
   dirs are NOT auto-created on the cluster side without ``--mkpath``).
   MUST include ``configs/`` (the open-instruct DeepSpeed config
   resolver is module-relative) AND ``external/open-instruct/`` +
   ``configs/tulu/`` + ``configs/deepspeed/`` for full-FT.
2. **sbatch render** — one self-contained script that owns every
   convention (account, --output, in-job preflight, venv cache, secrets,
   ``module load cuda``, ``[phase=...]`` heartbeats, the open-instruct
   accelerate command for full-FT). Existing entrypoints do NOT emit
   these conventions.
3. **stdin submit** — ``ssh robot-<cluster> sbatch < script``. Job-id
   parsing uses ``Submitted batch job \\K[0-9]+`` (sbatch's memory NOTE
   pollutes a naïve ``grep -oE '[0-9]+' | tail -1``).
4. **scancel** — single one-shot teardown that ``ssh``es a cancel call.

The :class:`SlurmBackend` exposes the :class:`ComputeBackend` interface;
:func:`render_sbatch` is split out as a pure function so the golden test
asserts the rendered script content WITHOUT touching the cluster.

Per-cluster config dict
-----------------------

Every cluster-specific knob lives in :data:`CLUSTER_CONFIGS`. Adding
Fir / another cluster is a config-only change (the renderer + submitter
read this dict). v1 ships Nibi only; Fir is wired in the table but flagged
``available=False`` until v1.1.

What this backend DOES NOT do
-----------------------------

* Run a real job on the cluster. The acceptance ladder (P0/P1/P2) is the
  orchestrator's responsibility and requires Duo MFA + the robot key —
  out of scope for this code change.
* Multi-node ``srun`` full-FT (forbidden by the robot forced-command
  wrapper allowlist; v1 is single-node only).
* Mila (interactive-only seam; out of scope for v1).
* Offline-cluster (Narval / Rorqual / Trillium / TamIA) staging
  (deferred to v1.2).

References:
* Plan ``2026-06-08_001932-slurm-cluster-backend-for-issue.md`` §§
  Approach / Steps / P0 probe results.
* ``backends/slurm_monitor.py`` (sibling) for poll / heartbeat reads.
* ``CLUSTER_CONFIGS`` for the per-cluster account / robot-alias / GPU
  caps + ``module load cuda`` bridge string.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import logging
import os
import re
import shlex
import subprocess
import tempfile
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from explore_persona_space.backends.base import (
    BackendKind,
    ComputeBackend,
    PollResult,
    RunHandle,
    RunSpec,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Per-cluster config table
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ClusterConfig:
    """Per-cluster knobs the SLURM backend needs.

    Everything cluster-specific lives here so the renderer + submitter
    stay generic. Adding a new cluster = adding a row to
    :data:`CLUSTER_CONFIGS`.

    Fields:

    * ``name`` — the canonical cluster name (``nibi``, ``fir``). Used as
      the dict key in :data:`CLUSTER_CONFIGS` AND as the ``BackendKind``
      alias the selector resolves to a backend instance.
    * ``account`` — SLURM ``--account`` value. ``rrg-bengioy-ad_gpu``
      for the DRAC robot.
    * ``robot_alias`` — the SSH alias the robot key is bound to (e.g.
      ``robot-nibi``). The submit + teardown shell out
      ``ssh <robot_alias> sbatch`` / ``ssh <robot_alias> scancel``.
    * ``max_gpus_per_node`` — hard cap (Nibi 8, Fir 4). The renderer
      asserts ``spec.gpus <= max_gpus_per_node`` before submitting so a
      typo doesn't burn 6h of queue wait.
    * ``partition`` — optional ``--partition`` value (e.g.
      ``gpubase_bygpu_b3`` for a short-time bin). ``None`` lets SLURM
      auto-place into the default ``bynode`` partition.
    * ``constraint`` — optional ``--constraint`` value (e.g.
      ``[gpu80gb]``). Some clusters use this to pin GPU memory class.
    * ``scratch_path`` — absolute path to the cluster's ``$SCRATCH``
      analogue. The renderer derives ``$SCRATCH`` from the job env
      inside the sbatch, so this is only used by VM-side rsync (which
      must construct the destination path without inheriting the
      cluster's env).
    * ``nccl_socket_ifname`` — optional ``NCCL_SOCKET_IFNAME`` value.
      Defaults to ``None`` (let NCCL auto-resolve via the EasyBuild
      NCCL module — confirmed working in P0(c)). Set per-cluster only
      if P2 surfaces a wrong iface pick.
    * ``module_load_cuda`` — the exact ``module load`` line to put in
      the sbatch (e.g. ``module load cuda``, or a versioned variant).
      P0(c) finding: ``module load`` MUST be on its own line, never
      piped (a piped ``module load … | tail`` runs in a subshell and
      loses the env). The renderer enforces the dedicated line.
    * ``cuda_home_bridge`` — fallback expression to set ``CUDA_HOME``
      when ``module load`` doesn't (the EasyBuild stack exports
      ``EBROOTCUDA`` and ``CUDACORE_HOME`` but some sub-modules don't
      set ``CUDA_HOME``). The renderer pastes this in as a guarded
      assignment.
    * ``available`` — whether the cluster is wired for v1. Fir = False
      in v1 (queued for v1.1); flipping this to True is a config-only
      change once the rsync path + robot key are validated on Fir.
    """

    name: str
    account: str
    robot_alias: str
    max_gpus_per_node: int
    scratch_path: str
    partition: str | None = None
    constraint: str | None = None
    nccl_socket_ifname: str | None = None
    module_load_cuda: str = "module load cuda"
    cuda_home_bridge: str = (
        'if [ -z "${CUDA_HOME:-}" ]; then\n'
        '  if [ -n "${EBROOTCUDA:-}" ]; then\n'
        "    export CUDA_HOME=$EBROOTCUDA\n"
        '  elif [ -n "${CUDACORE_HOME:-}" ]; then\n'
        "    export CUDA_HOME=$CUDACORE_HOME\n"
        "  fi\n"
        "fi"
    )
    available: bool = True


# Canonical per-cluster table. v1 ships Nibi; Fir is in the table but
# flagged ``available=False`` until v1.1. Adding a new cluster is one
# row + bumping the selector's alias set in selector.py.
CLUSTER_CONFIGS: dict[str, ClusterConfig] = {
    "nibi": ClusterConfig(
        name="nibi",
        account="rrg-bengioy-ad_gpu",
        robot_alias="robot-nibi",
        max_gpus_per_node=8,
        scratch_path="/scratch/tjiral",  # DRAC $SCRATCH = /scratch/<user>; verified by probe
    ),
    "fir": ClusterConfig(
        name="fir",
        account="rrg-bengioy-ad_gpu",
        robot_alias="robot-fir",
        max_gpus_per_node=4,
        scratch_path="/scratch/tjiral",  # DRAC $SCRATCH = /scratch/<user>; verified by probe
        available=False,
    ),
}


def get_cluster_config(name: str) -> ClusterConfig:
    """Look up a :class:`ClusterConfig` by name.

    Raises :class:`ValueError` on unknown cluster (a typo in the
    selector or frontmatter should surface loudly, NOT silently route
    to a fallback). Raises a separate :class:`RuntimeError` when the
    cluster is in the table but ``available=False`` so the operator
    sees the v1 scope clearly.
    """
    if name not in CLUSTER_CONFIGS:
        raise ValueError(
            f"unknown cluster {name!r}. Known: {sorted(CLUSTER_CONFIGS)}. "
            "Add a new ClusterConfig row to backends/slurm.CLUSTER_CONFIGS."
        )
    cfg = CLUSTER_CONFIGS[name]
    if not cfg.available:
        raise RuntimeError(
            f"cluster {name!r} is in CLUSTER_CONFIGS but flagged available=False "
            "(deferred to v1.1). Set available=True after validating rsync + "
            "robot-key on that cluster."
        )
    return cfg


# ---------------------------------------------------------------------------
# Time-budget heuristics (sbatch --time)
# ---------------------------------------------------------------------------


# Maps the workload intent to the shortest ``--time`` bin that fits per
# P0(g). LoRA + eval comfortably fit in 6h on 1xH100; full-FT 7B on
# 4xH100 needs <24h (size to that bin so it schedules near-instantly
# via ``gpubase_bygpu_b3`` instead of queuing 4 days out on the 7-day
# bin per P0(g)). Single floating-point hours; renderer converts to
# ``HH:MM:SS``.
_DEFAULT_TIME_BUDGETS_HOURS: dict[str, float] = {
    "lora-7b": 6.0,
    "lora": 6.0,  # alias accepted by stages_for_spec + _DEFAULT_GPUS_FOR_INTENT
    "eval": 4.0,
    "debug": 1.0,
    "ft-7b": 23.5,  # leave a margin under the 24h short-bin cap
    "inf-70b": 12.0,
    "ft-70b": 47.5,  # 2-day bin
}


def time_budget_hours(spec: RunSpec) -> float:
    """Resolve ``spec.time_budget_hours`` with the intent-default table.

    Explicit override wins. Otherwise return the intent default. Raises
    :class:`ValueError` on a negative or zero override AND on an
    unsupported intent (the rest of the module is fail-fast —
    ``stages_for_spec`` raises on unknown intents — so silently
    defaulting to 6h here would mask a typo and submit a job under the
    wrong wall-clock budget).
    """
    if spec.time_budget_hours is not None:
        if spec.time_budget_hours <= 0:
            raise ValueError(f"time_budget_hours must be positive, got {spec.time_budget_hours}")
        return float(spec.time_budget_hours)
    if spec.intent not in _DEFAULT_TIME_BUDGETS_HOURS:
        raise ValueError(
            f"no default time budget for intent {spec.intent!r}. "
            f"Supported intents: {sorted(_DEFAULT_TIME_BUDGETS_HOURS)}. "
            "Pass an explicit ``time_budget_hours=`` in the RunSpec or "
            "add the intent to ``_DEFAULT_TIME_BUDGETS_HOURS``."
        )
    return _DEFAULT_TIME_BUDGETS_HOURS[spec.intent]


def _format_sbatch_time(hours: float) -> str:
    """``HH:MM:SS`` for SLURM ``--time``. Accepts fractional hours."""
    total_seconds = round(hours * 3600)
    if total_seconds <= 0:
        raise ValueError(f"non-positive time budget: {hours}")
    hh = total_seconds // 3600
    mm = (total_seconds % 3600) // 60
    ss = total_seconds % 60
    return f"{hh:02d}:{mm:02d}:{ss:02d}"


# GPU count defaults per intent. Mirrors RunPod's
# ``gpu_heuristics.resolve_intent`` for the intents the cluster
# currently supports. NOT a Counter / dict-with-default — an unknown
# intent raises rather than picking 1 silently (consistent with
# ``stages_for_spec`` + ``time_budget_hours``; a typo should fail the
# render, not submit a job at the wrong GPU count).
_DEFAULT_GPUS_FOR_INTENT: dict[str, int] = {
    "lora-7b": 1,
    "lora": 1,
    "eval": 1,
    "debug": 1,
    "ft-7b": 4,
    "inf-70b": 8,
    "ft-70b": 8,
}


def default_gpus_for_intent(spec: RunSpec) -> int:
    """Resolve ``spec.gpus`` for the sbatch render (intent default fallback).

    Explicit ``spec.gpus`` wins (positive int). Otherwise return the
    intent default from :data:`_DEFAULT_GPUS_FOR_INTENT`. Raises
    :class:`ValueError` on an unsupported intent — the rest of the
    module fails fast on unknown intents (``stages_for_spec``,
    ``time_budget_hours``); silently defaulting to 1 GPU here would
    mask a typo and submit a job at the wrong GPU count.
    """
    if spec.gpus is not None and spec.gpus > 0:
        return spec.gpus
    if spec.intent not in _DEFAULT_GPUS_FOR_INTENT:
        raise ValueError(
            f"no default GPU count for intent {spec.intent!r}. "
            f"Supported intents: {sorted(_DEFAULT_GPUS_FOR_INTENT)}. "
            "Pass an explicit ``gpus=`` in the RunSpec or add the intent "
            "to ``_DEFAULT_GPUS_FOR_INTENT``."
        )
    return _DEFAULT_GPUS_FOR_INTENT[spec.intent]


# ---------------------------------------------------------------------------
# Plan-hash / job-name helpers
# ---------------------------------------------------------------------------


def job_name(spec: RunSpec, plan_hash: str | None = None) -> str:
    """Canonical SLURM job name keyed by issue (+ optional plan hash).

    Used by the monitor's idempotent reconnect — when the local launch
    marker is present but ``squeue -j <id>`` shows nothing, the monitor
    falls back to ``squeue --name <job_name>`` to disambiguate
    "ageout" from "really gone".
    """
    if plan_hash:
        return f"eps-issue-{spec.issue}-{plan_hash[:8]}"
    return f"eps-issue-{spec.issue}"


def compute_plan_hash(plan_body: str | bytes) -> str:
    """Short stable hash of the plan body for job-name keying."""
    data = plan_body.encode("utf-8") if isinstance(plan_body, str) else plan_body
    return hashlib.sha256(data).hexdigest()[:8]


# ---------------------------------------------------------------------------
# Rsync sync (VM → cluster scratch)
# ---------------------------------------------------------------------------


def _scratch_dir_for(spec: RunSpec, cluster: ClusterConfig) -> str:
    """Destination on the cluster: ``$SCRATCH/eps/issue-<N>``.

    The trailing path is computed VM-side (we don't inherit ``$SCRATCH``
    from the cluster env). The cluster admin's ``$SCRATCH`` is mapped
    to :attr:`ClusterConfig.scratch_path`.
    """
    return f"{cluster.scratch_path}/eps/issue-{spec.issue}"


# The set of repo-relative paths the cluster job needs. This is wider
# than the RunPod-equivalent because:
# - ``configs/`` is module-relative for ``resolve_deepspeed_config``
#   (P0(c) finding from the plan).
# - ``external/open-instruct/`` is mandatory for any full-FT run; the
#   renderer's open-instruct accelerate launcher targets
#   ``external/open-instruct/<stage.script_rel>``, so the destination
#   tree MUST have the ``external/`` prefix preserved.
# - ``scripts/`` carries ``train.py`` / ``eval.py`` / ``launch_stage.py``
#   which the renderer's open-instruct path delegates to.
# - ``pyproject.toml`` + ``uv.lock`` are what ``uv sync`` consumes.
# The exclude list keeps the eval-result history + dashboards out of
# scratch; the cluster generates fresh artifacts and rsyncs them back.
#
# Paths are dot-anchored (``./external/open-instruct`` etc.) so they
# combine with ``rsync --relative`` + ``cwd=src_root`` to land at
# ``$DST/external/open-instruct/...`` (NOT ``$DST/open-instruct/...`` —
# which is what positional sources without ``--relative`` produce, and
# what kills the renderer's full-FT path because it emits
# ``external/open-instruct/<stage.script_rel>`` as the launch target).
# ``configs/deepspeed`` and ``configs/tulu`` are removed because they
# are subsets of ``configs`` and would be double-copied otherwise.
RSYNC_INCLUDE_PATHS: tuple[str, ...] = (
    "./pyproject.toml",
    "./uv.lock",
    "./src",
    "./scripts",
    "./configs",
    "./external/open-instruct",
    "./tests",
)

RSYNC_EXCLUDE_PATTERNS: tuple[str, ...] = (
    ".venv/",
    "__pycache__/",
    ".pytest_cache/",
    "*.pyc",
    "wandb/",
    "outputs/",
    "eval_results/",  # generated fresh by the cluster run
    "figures/",
    ".claude/worktrees/",
    "tasks/",
    "raw/",
    "docs/",
    "archive/",
    "ood_eval_results/",
    "node_modules/",
    "dashboard/",
)


def build_rsync_command(
    *,
    src_root: Path,
    dest_root: str,
    robot_alias: str,
    include_paths: tuple[str, ...] = RSYNC_INCLUDE_PATHS,
    exclude_patterns: tuple[str, ...] = RSYNC_EXCLUDE_PATTERNS,
) -> list[str]:
    """Build the rsync argv that copies ``include_paths`` to the cluster.

    Flag set (P0(a) validated): ``-a --relative --delete --partial
    --mkpath``. ``--mkpath`` is REQUIRED — the forced-command wrapper
    does NOT auto-create intermediate dirs (P0(a) finding). ``--delete``
    keeps the destination tree in lockstep with the local tree so a
    removed file VM-side disappears on the cluster.

    ``--relative`` is LOAD-BEARING: without it (and without dot-anchored
    sources like ``./external/open-instruct``), rsync drops every
    intermediate path component above the basename and the cluster
    side ends up with ``$DST/open-instruct/...`` instead of
    ``$DST/external/open-instruct/...``. The renderer emits
    ``external/open-instruct/<stage.script_rel>`` as the SFT/DPO launch
    target, so a missing ``external/`` prefix kills every full-FT job at
    line 1 with ``no such file``. The dot anchor (``./<path>``) caps
    where the relative path starts — without it ``--relative`` would
    preserve the FULL ``src_root``-prefixed path (e.g.
    ``$DST/home/.../slurm-backend/external/...``), also wrong.

    The function does NOT execute rsync; it returns the argv. The caller
    is responsible for shelling out from ``cwd=src_root`` (so the
    dot-anchored sources resolve correctly). ``run_rsync_sync`` handles
    the cwd; if you call rsync yourself, pass ``cwd=src_root``.

    ``src_root`` MUST be the repository root (``pyproject.toml`` is at
    its top). ``dest_root`` is the full cluster path (e.g.
    ``/scratch/eps/issue-137``).
    """
    if not (src_root / "pyproject.toml").exists():
        raise FileNotFoundError(
            f"build_rsync_command: src_root={src_root!r} has no pyproject.toml "
            "(repo root expected)."
        )
    argv: list[str] = [
        "rsync",
        "-a",
        "--relative",
        "--delete",
        "--partial",
        "--mkpath",
    ]
    for pattern in exclude_patterns:
        argv.extend(["--exclude", pattern])
    # Sources are the dot-anchored relative paths from RSYNC_INCLUDE_PATHS
    # (e.g. "./external/open-instruct"). Combined with cwd=src_root and
    # ``--relative``, rsync preserves the path from the dot to the leaf,
    # which is what we want on the cluster side. We do NOT prepend
    # ``src_root`` to each entry — that would defeat the dot anchor.
    argv.extend(list(include_paths))
    argv.append(f"{robot_alias}:{dest_root}/")
    return argv


def run_rsync_sync(
    *,
    src_root: Path,
    dest_root: str,
    robot_alias: str,
    timeout: int = 600,
) -> None:
    """Run the rsync sync; raise on non-zero exit.

    Wraps :func:`build_rsync_command` + ``subprocess.run`` so the call
    site is a one-liner. ``timeout`` defaults to 10 min — a clean tree
    rsyncs in seconds, but a cold first sync on a slow link can be
    minutes (Nibi P0(a) measured ~12s for 50MB; allow a wide margin).

    MUST run from ``cwd=src_root`` so the dot-anchored sources in
    :data:`RSYNC_INCLUDE_PATHS` resolve to the repo tree (see
    :func:`build_rsync_command` for the full ``--relative`` rationale).
    """
    argv = build_rsync_command(
        src_root=src_root,
        dest_root=dest_root,
        robot_alias=robot_alias,
    )
    logger.info("running rsync to %s (cwd=%s): %s", robot_alias, src_root, " ".join(argv))
    subprocess.run(argv, check=True, timeout=timeout, cwd=str(src_root))


# ---------------------------------------------------------------------------
# Secrets sync (VM → cluster scratch, fresh per launch)
# ---------------------------------------------------------------------------


# The set of env vars sourced into the in-job environment via secrets.env.
# Pulled from os.environ at launch time; the file is rsync'd with chmod
# 600 and shredded by the sbatch trap.
SECRET_ENV_KEYS: tuple[str, ...] = (
    "HF_TOKEN",
    "WANDB_API_KEY",
    "ANTHROPIC_API_KEY",
    "OPENAI_API_KEY",
    "RUNPOD_API_KEY",  # for fallback paths that need to talk to RunPod
    "HF_USERNAME",
)


def render_secrets_env(
    env: dict[str, str] | None = None,
    keys: tuple[str, ...] = SECRET_ENV_KEYS,
) -> str:
    """Render a ``KEY=value`` env file for the sbatch ``set -a; source`` stanza.

    Plain ``KEY=value`` lines (no ``export`` — the sbatch wraps the
    source in ``set -a / set +a`` so every assignment auto-exports;
    confirmed in P0(c)). Values are shell-quoted via :func:`shlex.quote`
    so a token with shell-meaningful chars survives the round trip.

    Only keys present in ``env`` are rendered (a missing key means the
    VM operator never set it — the in-job preflight will FAIL fast and
    the selector falls back to RunPod, exactly the intended path).
    """
    if env is not None:
        src = env
    else:
        # Secrets live in the repo ``.env`` (loaded via dotenv at runtime),
        # NOT the ambient shell — so a bare ``os.environ`` snapshot is empty
        # and the cluster would get a 0-key secrets.env (the in-job preflight
        # then FAILs on the ``${HF_TOKEN:?}`` guard). Load the project dotenv
        # first; ``resolve_dotenv_path`` walks to the main worktree, so this
        # works from a linked worktree too. ``override=False`` keeps any
        # already-exported var authoritative.
        from explore_persona_space.orchestrate.env import load_dotenv as _load_dotenv

        _load_dotenv()
        src = dict(os.environ)
    lines: list[str] = []
    for key in keys:
        val = src.get(key)
        if val is None or val == "":
            continue
        lines.append(f"{key}={shlex.quote(val)}")
    return "\n".join(lines) + ("\n" if lines else "")


def scp_push_secrets(
    *,
    robot_alias: str,
    scratch_dir: str,
    content: str,
    timeout: int = 30,
) -> None:
    """Deliver ``secrets.env`` to ``$SCRATCH_JOB_DIR/secrets.env`` via ``scp``.

    The robot forced-command wrapper allowlist permits ``scp`` (and
    ``sftp`` / ``rsync``) but REJECTS ``ssh <alias> bash -c '<script>'``,
    so the earlier in-band ``ssh ... bash -c ...`` path was DOA — every
    cluster task erroring at ``prepare`` and falling back to RunPod. The
    sbatch already does ``chmod 600 "$SECRETS_FILE"`` (in the secrets
    stanza near ``render_sbatch``) and asserts the file is present
    before sourcing, so we do NOT need to chmod on the remote side here.

    Implementation:

    1. Write ``content`` into a unique VM-side temp file
       (:func:`tempfile.mkstemp` with mode 0o600 so the secrets are
       never world-readable on the VM either).
    2. ``scp`` the temp file to ``<robot_alias>:<scratch_dir>/secrets.env``.
       ``rsync`` (also allowed) would work equivalently; ``scp`` is the
       most direct match for "copy one file across".
    3. Always remove the VM-side temp file (try/finally) so a transient
       scp failure can't leak the file on the VM. The ``shred`` would
       be belt-and-suspenders; rm is sufficient because the temp lived
       under the controlled mkstemp dir for the duration of one scp.

    The ``$$`` shell PID idiom from the prior implementation was a
    SHELL expansion that does NOT happen here (it's a Python f-string,
    so ``$$`` is the literal two-character sequence after shlex-quote
    rather than a unique pid). We use :func:`tempfile.mkstemp` instead,
    which is the genuinely-unique form for concurrent prepares.
    """
    fd, tmp_path = tempfile.mkstemp(prefix="eps-slurm-secrets-", suffix=".env")
    try:
        os.write(fd, content.encode("utf-8"))
        os.close(fd)
        os.chmod(tmp_path, 0o600)
        remote_path = f"{robot_alias}:{scratch_dir}/secrets.env"
        # -p preserves the local 0o600 perms (the sbatch re-asserts
        # chmod 600, but starting tight is correct). -q suppresses the
        # progress meter which clutters orchestrator logs.
        argv = ["scp", "-p", "-q", tmp_path, remote_path]
        logger.info("scp secrets to %s (%d bytes)", remote_path, len(content))
        subprocess.run(argv, check=True, timeout=timeout)
    finally:
        # The tmp file may already be gone if scp consumed-and-removed
        # (it doesn't, but defensive); suppress the FileNotFoundError
        # narrowly so the cleanup is idempotent without swallowing any
        # OTHER OSError (permissions, IO).
        with contextlib.suppress(FileNotFoundError):
            os.unlink(tmp_path)


# ---------------------------------------------------------------------------
# Marker posting (task.py post-marker) — VM-side only
# ---------------------------------------------------------------------------


def post_marker_via_task_py(
    *,
    issue: int,
    marker: str,
    note: str,
    version: int = 1,
    by: str = "backends.slurm",
    timeout: int = 30,
) -> None:
    """Append an event to ``tasks/<status>/<N>/events.jsonl`` via task.py.

    Posts via ``uv run python scripts/task.py post-marker <N> <marker>
    --note <body> --version <v> --by <agent>``. The task.py CLI is the
    canonical mutation path (holds the workflow flock, commits once).

    VM-SIDE ONLY. ``task.py`` branch-guards to ``main`` and refuses on a
    non-``main`` HEAD; cluster compute nodes run on an ephemeral
    ``$SCRATCH`` rsync of the repo (no git checkout) and would fail this
    guard. The marker poster lives on the orchestrator VM and is called
    from the backend code that the orchestrator drives (launch, monitor
    poll), NEVER from inside the sbatch. The sbatch signals via the
    rsync'd ``status.json`` + ``[phase=...]`` lines; the monitor reads
    those and posts the markers VM-side.

    Note size cap (50_000 chars) is enforced by ``task.py post-marker``
    itself; oversize notes raise from the subprocess.
    """
    argv = [
        "uv",
        "run",
        "python",
        str(_repo_root_for_task_py() / "scripts" / "task.py"),
        "post-marker",
        str(issue),
        marker,
        "--note",
        note,
        "--version",
        str(version),
        "--by",
        by,
    ]
    logger.info("post-marker issue=%d kind=%s v=%d", issue, marker, version)
    subprocess.run(argv, check=True, timeout=timeout)


def _repo_root_for_task_py() -> Path:
    """Locate the repo root (where ``scripts/task.py`` lives).

    Walks up from this file's location until a directory containing
    ``scripts/task.py`` is found. Falls back to ``Path.cwd()`` if the
    layout has been mangled (very defensive — the import path that
    found us must have a real repo root somewhere).
    """
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "scripts" / "task.py").exists():
            return parent
    return Path.cwd()


# ---------------------------------------------------------------------------
# sbatch render
# ---------------------------------------------------------------------------


# Stage backend kinds the renderer knows how to launch. ``local`` =
# Hydra ``scripts/train.py``/``scripts/eval.py``; ``open_instruct`` =
# the open-instruct ``finetune.py``/``dpo_tune_cache.py`` accelerate
# launcher. Typed as a ``Literal`` so the renderer's terminal ``else:
# raise`` (``unknown stage backend``) is provably exhaustive — adding
# a third backend kind requires extending this alias AND the renderer
# dispatch in lockstep, surfaced by the type checker.
WorkloadKind = Literal["local", "open_instruct"]


@dataclass(frozen=True)
class Stage:
    """One workload stage inside a sbatch render.

    Full-FT is a heterogeneous chain (LoRA cpt → full-FT SFT → full-FT
    DPO → LoRA EM); each entry in :attr:`SbatchPlan.stages` becomes
    one ``[phase=...]`` block in the rendered script. LoRA + eval ships
    as a single-stage plan.

    Fields:

    * ``name`` — short identifier (``cpt`` / ``sft`` / ``dpo`` / ``em``
      / ``eval``). Appears in the ``[phase=<name>]`` heartbeats so the
      monitor can grep progress.
    * ``backend`` — ``"local"`` (Hydra ``scripts/train.py`` /
      ``scripts/eval.py``) or ``"open_instruct"`` (full-FT via
      ``external/open-instruct/open_instruct/finetune.py`` or
      ``dpo_tune_cache.py``). The renderer dispatches on this.
    * ``script_rel`` — repo-relative path to the entrypoint. For
      ``open_instruct`` stages this is e.g. ``open_instruct/finetune.py``
      (resolved against the synced ``external/open-instruct/`` tree).
      For ``local`` stages it's ``scripts/train.py`` etc.
    * ``deepspeed_config_rel`` — repo-relative DeepSpeed config (only
      for ``open_instruct`` stages); ``None`` for ``local``.
    * ``hydra_args`` — Hydra overrides for ``local`` stages
      (``condition=c1 seed=42`` etc.); ignored for ``open_instruct``.
    * ``oi_args`` — flat list of CLI flags for ``open_instruct``
      stages (``["--model_name_or_path", "Qwen/Qwen2.5-7B", ...]``);
      ignored for ``local``.
    """

    name: str
    backend: WorkloadKind
    script_rel: str
    deepspeed_config_rel: str | None = None
    hydra_args: tuple[str, ...] = ()
    oi_args: tuple[str, ...] = ()


@dataclass(frozen=True)
class SbatchPlan:
    """The rendered-sbatch's input plan.

    Composed by :func:`stages_for_spec` from a :class:`RunSpec`; tests
    can pass one directly to :func:`render_sbatch` to assert the
    rendered command shape without going through the intent table.
    """

    stages: tuple[Stage, ...]


def stages_for_spec(spec: RunSpec) -> SbatchPlan:
    """Derive an :class:`SbatchPlan` from a :class:`RunSpec`.

    Intent → stage table:

    * ``lora-7b`` / ``eval`` / ``debug`` → single ``local`` stage on
      ``scripts/train.py`` / ``scripts/eval.py`` (Hydra args pulled
      from ``spec.hydra_args``).
    * ``ft-7b`` / ``ft-70b`` → 4-stage chain (LoRA cpt → full-FT SFT →
      full-FT DPO → LoRA EM). Cpt + EM are LoRA via ``scripts/train.py``;
      SFT + DPO are open-instruct.
    * ``inf-70b`` → single ``local`` eval stage.

    The mapping is intentionally simple; experiments that need a
    different chain pass an explicit :class:`SbatchPlan` directly to
    :func:`render_sbatch`. Refinement is config-only.
    """
    if spec.intent in {"lora-7b", "lora"}:
        return SbatchPlan(
            stages=(
                Stage(
                    name="lora",
                    backend="local",
                    script_rel="scripts/train.py",
                    hydra_args=spec.hydra_args,
                ),
                Stage(
                    name="eval",
                    backend="local",
                    script_rel="scripts/eval.py",
                    hydra_args=spec.hydra_args,
                ),
            )
        )
    if spec.intent in {"eval", "inf-70b"}:
        return SbatchPlan(
            stages=(
                Stage(
                    name="eval",
                    backend="local",
                    script_rel="scripts/eval.py",
                    hydra_args=spec.hydra_args,
                ),
            )
        )
    if spec.intent in {"ft-7b", "ft-70b"}:
        # Full-FT canonical chain. The Hydra config name + DeepSpeed
        # config flow through spec.extra so the planner can swap them
        # per-experiment (P2 confirms which zero level fits 7B on
        # 4xH100; default to the project-house pin ``zero2_fp32_comm``).
        ds_config = spec.extra.get("deepspeed_config", "deepspeed/zero2_fp32_comm.json")
        oi_args_sft = tuple(spec.extra.get("oi_args_sft", ()))
        oi_args_dpo = tuple(spec.extra.get("oi_args_dpo", ()))
        return SbatchPlan(
            stages=(
                Stage(
                    name="cpt",
                    backend="local",
                    script_rel="scripts/train.py",
                    hydra_args=spec.hydra_args,
                ),
                Stage(
                    name="sft",
                    backend="open_instruct",
                    script_rel="open_instruct/finetune.py",
                    deepspeed_config_rel=ds_config,
                    oi_args=oi_args_sft,
                ),
                Stage(
                    name="dpo",
                    backend="open_instruct",
                    script_rel="open_instruct/dpo_tune_cache.py",
                    deepspeed_config_rel=ds_config,
                    oi_args=oi_args_dpo,
                ),
                Stage(
                    name="em",
                    backend="local",
                    script_rel="scripts/train.py",
                    hydra_args=spec.hydra_args,
                ),
            )
        )
    if spec.intent == "debug":
        return SbatchPlan(
            stages=(
                Stage(
                    name="debug",
                    backend="local",
                    script_rel="scripts/train.py",
                    hydra_args=spec.hydra_args,
                ),
            )
        )
    raise ValueError(
        f"unsupported intent {spec.intent!r} for SLURM backend. Supported: "
        "lora-7b, lora, eval, inf-70b, ft-7b, ft-70b, debug."
    )


# Heartbeat interval (seconds) for the periodic status.json + stdout
# refresh inside the sbatch. The monitor's STALL_SEC is configured
# above this; a heartbeat that's < STALL_SEC ensures a healthy job
# always looks alive between log writes.
HEARTBEAT_INTERVAL_SECONDS = 60

# In-job preflight bail-out marker. The sbatch prints this line to its
# job.out when the preflight fails; the monitor watches for it to
# distinguish a clean-fail (preflight) from a real workload crash.
PREFLIGHT_FAIL_MARKER = "[phase=preflight-failed]"


def render_sbatch(
    *,
    spec: RunSpec,
    cluster: ClusterConfig,
    plan: SbatchPlan,
    scratch_dir: str,
    secrets_filename: str = "secrets.env",
    plan_hash: str | None = None,
) -> str:
    """Render the full sbatch script as a string.

    Pure function — no side effects, no filesystem access. The golden
    test asserts specific lines / shapes from the output. The renderer
    OWNS every cluster convention (no other module should re-derive
    them).

    Lines the test pins:

    * ``#SBATCH --account=rrg-bengioy-ad_gpu``
    * ``#SBATCH --gpus-per-node=<N>``
    * ``#SBATCH --output=<scratch_dir>/job.out``
    * ``#SBATCH --time=<HH:MM:SS>``
    * ``module load cuda`` on its own line (P0(c): NEVER piped).
    * ``CUDA_HOME`` bridge stanza.
    * ``UV_CACHE_DIR=$SCRATCH/uv-cache``.
    * Venv cache: ``$SCRATCH/eps/venv-<lockhash>-<gpu_extras>`` with
      ``.complete`` sentinel + ``flock``.
    * Secrets ``set +x; set -a; source <secrets>; set +a; set -x`` +
      ``trap`` shred.
    * Reachability + GPU + ``$SLURM_TMPDIR`` headroom preflight, exits
      non-zero on failure with ``[phase=preflight-failed]``.
    * One ``[phase=<name>]`` block per stage with the rendered command.
    * Terminal ``[phase=done]``.
    """
    if not cluster.available:
        raise RuntimeError(f"cluster {cluster.name!r} flagged available=False; cannot render.")
    gpus = default_gpus_for_intent(spec)
    if gpus > cluster.max_gpus_per_node:
        raise ValueError(
            f"requested gpus={gpus} > cluster {cluster.name!r} max_gpus_per_node="
            f"{cluster.max_gpus_per_node}. Single-node only in v1."
        )
    time_h = time_budget_hours(spec)
    time_str = _format_sbatch_time(time_h)
    name = job_name(spec, plan_hash)
    # The sbatch reads $SCRATCH at runtime; we hard-pin it for the
    # --output header (SLURM resolves the path BEFORE the script runs).
    output_path = f"{scratch_dir}/job.out"

    sbatch_headers = [
        "#!/bin/bash",
        f"#SBATCH --account={cluster.account}",
        f"#SBATCH --job-name={name}",
        "#SBATCH --nodes=1",
        "#SBATCH --ntasks-per-node=1",
        f"#SBATCH --gpus-per-node={gpus}",
        f"#SBATCH --cpus-per-task={min(8 * gpus, 64)}",
        f"#SBATCH --mem={min(64 * gpus, 480)}G",
        f"#SBATCH --time={time_str}",
        f"#SBATCH --output={output_path}",
    ]
    if cluster.partition:
        sbatch_headers.append(f"#SBATCH --partition={cluster.partition}")
    if cluster.constraint:
        sbatch_headers.append(f"#SBATCH --constraint={cluster.constraint}")

    # Shell prelude: umask + strict mode + cluster-scratch derivation
    # The set -e + set -u + pipefail are deliberate (fail-fast); a real
    # error inside the workload exits non-zero and the SLURM state
    # becomes FAILED so the monitor reports `dead`.
    prelude = [
        "set -euo pipefail",
        "umask 077",
        "",
        "# === Cluster scratch + log paths (single source of truth) ===",
        f"SCRATCH_JOB_DIR={shlex.quote(scratch_dir)}",
        'mkdir -p "$SCRATCH_JOB_DIR"',
        'STATUS_JSON="$SCRATCH_JOB_DIR/status.json"',
        "",
        "# Status helper: writes phase + heartbeat + gpu_busy + exit code",
        "# atomically to status.json. Monitor rsyncs this file and reads",
        "# heartbeat_ts to derive stall vs running. gpu_busy comes from",
        "# in-job nvidia-smi (allowed on the compute side; only the robot",
        "# SSH side bans it).",
        "_write_status() {",
        '  local phase="$1" exit_code="${2:-}"',
        "  local heartbeat_ts",
        "  heartbeat_ts=$(date -u +%Y-%m-%dT%H:%M:%SZ)",
        "  local gpu_busy=false",
        "  if command -v nvidia-smi >/dev/null 2>&1; then",
        "    if nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits 2>/dev/null "
        "| awk 'NR==1 && $1+0 > 5 {found=1} END {exit !found}'; then",
        "      gpu_busy=true",
        "    fi",
        "  fi",
        '  local tmp="${STATUS_JSON}.tmp"',
        '  printf \'{"phase":"%s","heartbeat_ts":"%s","gpu_busy":%s,"exit_code":"%s"}\\n\' \\',
        '    "$phase" "$heartbeat_ts" "$gpu_busy" "$exit_code" > "$tmp"',
        '  mv "$tmp" "$STATUS_JSON"',
        "}",
        "",
        "# Background heartbeat: refresh status.json every $HEARTBEAT_INTERVAL",
        "# seconds so a long-running stage (multi-hour full-FT) still looks",
        "# alive to the monitor even when stdout is quiet.",
        f"HEARTBEAT_INTERVAL={HEARTBEAT_INTERVAL_SECONDS}",
        "_heartbeat_loop() {",
        "  while true; do",
        '    _write_status "${CURRENT_PHASE:-init}"',
        '    sleep "$HEARTBEAT_INTERVAL"',
        "  done",
        "}",
        "",
    ]

    # CUDA + Triton cache setup (P0(c) finding: module load on its own
    # line; CUDA_HOME bridge as fallback).
    cuda_setup = [
        "# === CUDA + Triton + NCCL setup (P0(c)) ===",
        "# module load MUST be on its own line. A piped variant runs in",
        "# a subshell and the env is lost (P0(c) initial failure).",
        cluster.module_load_cuda,
        "",
        cluster.cuda_home_bridge,
        "",
        'export TRITON_CACHE_DIR="$SLURM_TMPDIR/triton"',
        'mkdir -p "$TRITON_CACHE_DIR"',
    ]
    if cluster.nccl_socket_ifname:
        cuda_setup.append(f"export NCCL_SOCKET_IFNAME={shlex.quote(cluster.nccl_socket_ifname)}")
    cuda_setup.append("")

    # Secrets stanza. set +x around the source so a `bash -x` rerun
    # doesn't leak tokens. trap shreds the file on EXIT/TERM/INT.
    secrets_setup = [
        "# === Secrets ===",
        f'SECRETS_FILE="$SCRATCH_JOB_DIR/{secrets_filename}"',
        "# Trap fires on normal exit AND on signals so an OOM kill / preempt",
        "# never leaves the secrets file on $SCRATCH.",
        'trap \'shred -u "$SECRETS_FILE" 2>/dev/null || rm -f "$SECRETS_FILE"\' EXIT TERM INT',
        "# Make sure file perms are tight before we source.",
        'if [ ! -f "$SECRETS_FILE" ]; then',
        '  echo "[FAIL] secrets file $SECRETS_FILE not found"',
        '  echo "' + PREFLIGHT_FAIL_MARKER + '"',
        "  exit 2",
        "fi",
        'chmod 600 "$SECRETS_FILE"',
        "set +x",
        "# set -a auto-exports every assignment in the sourced file. The",
        "# secrets file uses plain KEY=value lines (no `export`), so without",
        "# `set -a` the Python child does NOT see the tokens (P0(c) finding).",
        "set -a",
        "# shellcheck disable=SC1090",
        'source "$SECRETS_FILE"',
        "set +a",
        "set -x",
        "",
    ]

    # uv venv cache: keyed by uv.lock hash AND the --extra gpu flag (so
    # the LoRA-eval-only intent doesn't share a venv with full-FT). The
    # flock + temp-dir-then-rename guards against two concurrent first
    # builds corrupting a shared dir. P0(b) finding: builds ~6 min cold,
    # 328ms cached — caching is mandatory or the full-FT flash-attn
    # compile (P0(e), ~40 min) eats every job.
    needs_gpu_extras = spec.intent in {"ft-7b", "ft-70b"} or any(
        s.backend == "open_instruct" for s in plan.stages
    )
    extras_tag = "gpu" if needs_gpu_extras else "base"
    uv_extra_flag = " --extra gpu" if needs_gpu_extras else ""
    venv_setup = [
        "# === uv venv cache ===",
        "# Cache key = uv.lock hash + extras tag. Two concurrent first",
        "# builds would corrupt a shared dir, so we flock + build into a",
        "# .tmp dir, then atomically rename. .complete sentinel makes the",
        '# cache purge-safe (a half-built dir is never read as "ready").',
        'cd "$SCRATCH_JOB_DIR"',
        "LOCKHASH=$(sha256sum uv.lock | awk '{print $1}' | head -c 16)",
        f'VENV_DIR="$SCRATCH/eps/venv-${{LOCKHASH}}-{extras_tag}"',
        'VENV_COMPLETE="$VENV_DIR/.complete"',
        'VENV_LOCK="$SCRATCH/eps/venv-${LOCKHASH}.lock"',
        'mkdir -p "$SCRATCH/eps"',
        'export UV_CACHE_DIR="$SCRATCH/uv-cache"',
        'mkdir -p "$UV_CACHE_DIR"',
        "",
        "# Self-install uv (compute-node internet confirmed in P0(b))",
        "if ! command -v uv >/dev/null 2>&1; then",
        "  curl -LsSf https://astral.sh/uv/install.sh | sh",
        '  export PATH="$HOME/.local/bin:$PATH"',
        "fi",
        "",
        'if [ ! -f "$VENV_COMPLETE" ]; then',
        "  # Acquire exclusive lock on the lock-hash so two concurrent first",
        "  # builds serialize (the second one sees .complete and returns).",
        "  (",
        "    flock -x 200",
        '    if [ ! -f "$VENV_COMPLETE" ]; then',
        '      TMP_VENV="${VENV_DIR}.tmp.$$"',
        '      rm -rf "$TMP_VENV"',
        "      # Build into TMP so a crash leaves $VENV_DIR untouched.",
        '      VIRTUAL_ENV="$TMP_VENV" uv venv "$TMP_VENV"',
        f'      VIRTUAL_ENV="$TMP_VENV" uv sync --frozen{uv_extra_flag}',
        '      mv "$TMP_VENV" "$VENV_DIR"',
        '      touch "$VENV_COMPLETE"',
        "    fi",
        '  ) 200>"$VENV_LOCK"',
        "fi",
        'export VIRTUAL_ENV="$VENV_DIR"',
        'export PATH="$VENV_DIR/bin:$PATH"',
        "",
    ]

    # In-job preflight. FAIL fast before heavy work so the selector
    # falls back to RunPod before GPU time is spent.
    preflight = [
        "# === In-job preflight (FAIL-FAST before heavy work) ===",
        'CURRENT_PHASE="preflight"',
        '_write_status "preflight"',
        "",
        "# Tokens: must be in env post-source.",
        ': "${HF_TOKEN:?HF_TOKEN missing from secrets.env}"',
        ': "${WANDB_API_KEY:?WANDB_API_KEY missing from secrets.env}"',
        "",
        "# Hub + WandB reachability (reuse preflight.check_connectivity).",
        "uv run python -m explore_persona_space.orchestrate.preflight --no-gpu "
        '--min-disk 1 --json > "$SCRATCH_JOB_DIR/preflight.json" || {',
        '  echo "[FAIL] preflight subcommand returned non-zero"',
        '  echo "' + PREFLIGHT_FAIL_MARKER + '"',
        "  exit 3",
        "}",
        "",
        "# GPU visible (in-job nvidia-smi IS allowed; only the robot SSH side",
        "# bans it).",
        "if ! nvidia-smi >/dev/null 2>&1; then",
        '  echo "[FAIL] nvidia-smi not available inside SLURM allocation"',
        '  echo "' + PREFLIGHT_FAIL_MARKER + '"',
        "  exit 4",
        "fi",
        "",
        "# $SLURM_TMPDIR headroom (the renderer assumes a node-local tmpdir",
        "# for model + data staging; checkpoints go to $SCRATCH).",
        ': "${SLURM_TMPDIR:?SLURM_TMPDIR unset; this sbatch needs node-local scratch}"',
        "TMPDIR_FREE_GB=$(df -BG \"$SLURM_TMPDIR\" | awk 'NR==2 {print $4}' | tr -d G)",
        'if [ -z "$TMPDIR_FREE_GB" ] || [ "$TMPDIR_FREE_GB" -lt 50 ]; then',
        '  echo "[FAIL] SLURM_TMPDIR has < 50GB free (got ${TMPDIR_FREE_GB:-?}GB)"',
        '  echo "' + PREFLIGHT_FAIL_MARKER + '"',
        "  exit 5",
        "fi",
        "",
        "# GPU count must match SLURM_GPUS_ON_NODE (NOT a stale nvidia-smi).",
        ': "${SLURM_GPUS_ON_NODE:?SLURM_GPUS_ON_NODE unset; cannot derive process count}"',
        f'if [ "$SLURM_GPUS_ON_NODE" -ne {gpus} ]; then',
        f'  echo "[FAIL] SLURM_GPUS_ON_NODE=$SLURM_GPUS_ON_NODE != requested {gpus}"',
        '  echo "' + PREFLIGHT_FAIL_MARKER + '"',
        "  exit 6",
        "fi",
        "",
        "# Preflight PASS — start heartbeat loop in the background.",
        "_heartbeat_loop &",
        "HEARTBEAT_PID=$!",
        "trap 'kill $HEARTBEAT_PID 2>/dev/null; "
        'shred -u "$SECRETS_FILE" 2>/dev/null '
        '|| rm -f "$SECRETS_FILE"\' EXIT TERM INT',
        "",
    ]

    # Stage commands.
    master_addr = "${MASTER_ADDR:-localhost}"
    master_port = "${MASTER_PORT:-29500}"
    stage_blocks: list[str] = []
    for stage in plan.stages:
        stage_blocks.append(f"# === Stage: {stage.name} ===")
        stage_blocks.append(f'CURRENT_PHASE="{stage.name}"')
        stage_blocks.append(f'echo "[phase={stage.name}]"')
        stage_blocks.append(f'_write_status "{stage.name}"')
        if stage.backend == "local":
            # Hydra-style: uv run python <script> arg1 arg2 ...
            args_joined = " ".join(shlex.quote(a) for a in stage.hydra_args)
            stage_blocks.append(
                f"uv run python {shlex.quote(stage.script_rel)} {args_joined}".rstrip()
            )
        elif stage.backend == "open_instruct":
            if not stage.deepspeed_config_rel:
                raise ValueError(
                    f"open_instruct stage {stage.name!r} requires deepspeed_config_rel"
                )
            # accelerate launch ... finetune.py | dpo_tune_cache.py
            # The deepspeed config is path-relative to configs/ — the
            # cluster has the synced configs/ tree at $SCRATCH_JOB_DIR/configs.
            ds_config_path = f"configs/{stage.deepspeed_config_rel}"
            oi_args_joined = " ".join(shlex.quote(a) for a in stage.oi_args)
            stage_blocks.append(
                "uv run accelerate launch "
                "--mixed_precision bf16 "
                "--use_deepspeed "
                f"--deepspeed_config_file {shlex.quote(ds_config_path)} "
                "--num_processes $SLURM_GPUS_ON_NODE "
                "--num_machines 1 "
                "--machine_rank 0 "
                f"--main_process_ip {master_addr} "
                f"--main_process_port {master_port} "
                f"external/open-instruct/{stage.script_rel} "
                f"{oi_args_joined}".rstrip()
            )
        else:
            raise ValueError(f"unknown stage backend {stage.backend!r} for stage {stage.name!r}")
        stage_blocks.append("")

    # Terminal block.
    terminal = [
        "# === Done ===",
        'CURRENT_PHASE="done"',
        "kill $HEARTBEAT_PID 2>/dev/null || true",
        '_write_status "done" 0',
        'echo "[phase=done]"',
    ]

    parts = [*sbatch_headers, "", *prelude, *cuda_setup, *secrets_setup, *venv_setup, *preflight]
    for block in stage_blocks:
        parts.append(block)
    parts.extend(terminal)
    return "\n".join(parts) + "\n"


# ---------------------------------------------------------------------------
# Submit / scancel
# ---------------------------------------------------------------------------


# Regex that pulls the job id out of sbatch stdout. P0 finding: sbatch
# emits a "memory NOTE" before the success line so a naïve
# ``grep -oE '[0-9]+' | tail -1`` returns the wrong number. The PCRE
# ``\K`` semantics are emulated via a Python regex with a capture group.
_JOB_ID_RE = re.compile(r"Submitted batch job (\d+)")


def parse_job_id(sbatch_stdout: str) -> str:
    """Pull the numeric job id out of sbatch stdout; raise on miss."""
    match = _JOB_ID_RE.search(sbatch_stdout)
    if not match:
        raise RuntimeError(
            f"sbatch did not emit 'Submitted batch job <N>'; stdout was: {sbatch_stdout[:500]!r}"
        )
    return match.group(1)


def ssh_submit(
    *,
    robot_alias: str,
    sbatch_script: str,
    timeout: int = 60,
) -> str:
    """stdin-submit ``sbatch_script`` to the robot login node; return job id.

    Uses ``ssh <robot_alias> sbatch`` with the script on stdin (the
    forced-command wrapper allowlist permits this; no file write needed
    on the login node).
    """
    argv = ["ssh", robot_alias, "sbatch"]
    logger.info("ssh sbatch to %s (script %d bytes)", robot_alias, len(sbatch_script))
    proc = subprocess.run(
        argv,
        input=sbatch_script,
        capture_output=True,
        text=True,
        timeout=timeout,
        check=True,
    )
    return parse_job_id(proc.stdout)


def ssh_scancel(*, robot_alias: str, job_id: str, timeout: int = 30) -> None:
    """One-shot scancel via the robot SSH alias. Idempotent: a missing
    job id is logged but does NOT raise (the job may have terminated
    naturally between the poll and the cancel)."""
    argv = ["ssh", robot_alias, "scancel", job_id]
    logger.info("ssh scancel %s on %s", job_id, robot_alias)
    proc = subprocess.run(
        argv,
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )
    if proc.returncode != 0:
        # scancel emits to stderr; log but don't raise so the selector's
        # teardown path stays idempotent on a "already gone" job.
        logger.warning(
            "scancel %s on %s exited %d; stderr=%s",
            job_id,
            robot_alias,
            proc.returncode,
            proc.stderr.strip(),
        )


# ---------------------------------------------------------------------------
# estimate_start — sbatch --test-only (informational only)
# ---------------------------------------------------------------------------


# Parses ``squeue --start`` / ``sbatch --test-only`` output: the
# job-start-time appears as ``estimated start time YYYY-MM-DDTHH:MM:SS``.
_EST_START_RE = re.compile(r"start time (\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2})", re.IGNORECASE)


def ssh_estimate_start(
    *,
    robot_alias: str,
    sbatch_script: str,
    timeout: int = 30,
) -> datetime | None:
    """Best-effort ``sbatch --test-only`` start-time estimate; NEVER a gate.

    Returns the parsed estimate as a UTC :class:`datetime`, or ``None``
    when the output is missing / malformed / the wrapper rejects the
    call. The selector logs the estimate but uses an explicit
    max-wait watchdog for the actual park-decision (per the plan's
    submit-and-park policy).
    """
    argv = ["ssh", robot_alias, "sbatch", "--test-only"]
    try:
        proc = subprocess.run(
            argv,
            input=sbatch_script,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except subprocess.SubprocessError as exc:
        logger.warning("sbatch --test-only failed: %s", exc)
        return None
    blob = (proc.stderr or "") + (proc.stdout or "")
    match = _EST_START_RE.search(blob)
    if not match:
        return None
    try:
        return datetime.fromisoformat(match.group(1)).replace(tzinfo=UTC)
    except ValueError:
        return None


# ---------------------------------------------------------------------------
# SlurmBackend — the public ComputeBackend
# ---------------------------------------------------------------------------


@dataclass
class _LaunchedJobState:
    """In-process bookkeeping for an active SLURM job.

    Persisted on the backend instance so ``poll`` / ``teardown`` /
    ``fetch_*`` can re-read it. Not on :class:`RunHandle` because the
    handle is frozen and shared across processes (the orchestrator may
    re-spawn the backend between turns); the persistent terminal state
    lives in the marker trail, not here.
    """

    job_id: str
    cluster_name: str
    scratch_dir: str
    log_path: str
    submitted_at: float = field(default_factory=time.time)


class SlurmBackend(ComputeBackend):
    """SLURM cluster backend (robot-key submit, rsync-primary code sync).

    See module docstring for the design rationale + plan link. This is
    the real backend that replaces the slice-1 stub
    (``_SlurmStubBackend`` in :mod:`selector`).

    Constructor parameters expose the per-call seams the tests need:

    * ``src_root`` — repo root for rsync. Defaults to the package's
      parent (``src/explore_persona_space/backends/`` → 3 parents up).
    * ``submitter`` / ``canceller`` / ``rsyncer`` / ``poller`` — injection
      seams for tests. Each defaults to the real shell-out implementation
      above so production code paths exercise the real wire.
    """

    def __init__(
        self,
        *,
        src_root: Path | None = None,
        submitter=None,
        canceller=None,
        rsyncer=None,
        poller=None,
        start_estimator=None,
        secrets_pusher=None,
        marker_poster=None,
    ) -> None:
        self._src_root = src_root or _default_src_root()
        self._submit = submitter or ssh_submit
        self._cancel = canceller or ssh_scancel
        self._rsync = rsyncer or run_rsync_sync
        # Monitor.build_poll_result is loaded lazily to avoid a circular
        # import at module-load (slurm_monitor imports from this module).
        self._poll_fn = poller
        self._start_estimator = start_estimator or ssh_estimate_start
        # Secrets push uses scp by default (allowlisted by the robot
        # forced-command wrapper); ``ssh ... bash -c '<script>'`` would
        # be rejected. Tests inject a no-op pusher.
        self._secrets_pusher = secrets_pusher or scp_push_secrets
        # Marker poster is invoked at launch (``epm:cluster-launched``)
        # so the events.jsonl trail records the SLURM-side handle. The
        # selector posts ``epm:backend-selected`` at decision time; the
        # monitor posts ``epm:cluster-poll`` / ``epm:cluster-terminal``.
        # Defaults to the real task.py shell-out; tests inject a list-
        # appender.
        self._post_marker = marker_poster or post_marker_via_task_py
        self._jobs: dict[str, _LaunchedJobState] = {}

    # ----- identity --------------------------------------------------------

    @property
    def name(self) -> BackendKind:
        return "cluster"

    # ----- launch ----------------------------------------------------------

    def prepare(self, spec: RunSpec) -> None:
        """rsync the repo + secrets file to the cluster.

        Idempotent — rsync with ``--delete`` brings the destination into
        lockstep regardless of prior state. The secrets file is written
        FRESH on every prepare call so a token rotation propagates
        immediately.
        """
        cluster = self._cluster_for_spec(spec)
        scratch_dir = _scratch_dir_for(spec, cluster)
        self._rsync(
            src_root=self._src_root,
            dest_root=scratch_dir,
            robot_alias=cluster.robot_alias,
        )
        secrets = render_secrets_env()
        # Write the secrets file directly via SSH stdin (avoids a tmp
        # file on the VM that could leak). The single-shot dd writes
        # bytes verbatim; we chmod 600 in the same SSH call so it's
        # never world-readable on the cluster side.
        self._push_secrets(cluster, scratch_dir, secrets)

    def _push_secrets(self, cluster: ClusterConfig, scratch_dir: str, content: str) -> None:
        """Deliver ``secrets.env`` to ``$SCRATCH_JOB_DIR`` via the injected
        pusher (default :func:`scp_push_secrets`).

        Decoupled so tests can swap a list-appender for the real
        ``scp``/``rsync`` shell-out.
        """
        self._secrets_pusher(
            robot_alias=cluster.robot_alias,
            scratch_dir=scratch_dir,
            content=content,
        )

    def launch(self, spec: RunSpec) -> RunHandle:
        """Render + submit; return a :class:`RunHandle` keyed by job id.

        Posts ``epm:cluster-launched v1`` AFTER sbatch submit succeeds
        (per ``workflow.yaml § markers``). The marker carries the SLURM-
        side handle so the orchestrator's events.jsonl trail has the
        ``job_id`` / ``scratch_dir`` / ``log_path`` / ``job_name``
        needed for idempotent reconnect after orchestrator re-spawn.
        """
        cluster = self._cluster_for_spec(spec)
        scratch_dir = _scratch_dir_for(spec, cluster)
        plan = stages_for_spec(spec)
        plan_hash = spec.extra.get("plan_hash")
        script = render_sbatch(
            spec=spec,
            cluster=cluster,
            plan=plan,
            scratch_dir=scratch_dir,
            plan_hash=plan_hash,
        )
        job_id = self._submit(
            robot_alias=cluster.robot_alias,
            sbatch_script=script,
        )
        log_path = f"{scratch_dir}/job.out"
        state = _LaunchedJobState(
            job_id=job_id,
            cluster_name=cluster.name,
            scratch_dir=scratch_dir,
            log_path=log_path,
        )
        self._jobs[job_id] = state
        name = job_name(spec, plan_hash)
        time_h = time_budget_hours(spec)
        gpus = default_gpus_for_intent(spec)

        # Post epm:cluster-launched v1 to the originating task's
        # events.jsonl. Body fields match workflow.yaml § markers.
        # NOTE size cap (50k chars) is enforced by task.py post-marker;
        # this body is well under it. JSON-formatted so the dashboard
        # can render structured fields.
        marker_body = json.dumps(
            {
                "cluster": cluster.name,
                "job_id": job_id,
                "job_name": name,
                "scratch_dir": scratch_dir,
                "log_path": log_path,
                "account": cluster.account,
                "gpus": gpus,
                "time_budget_hours": time_h,
            },
            sort_keys=True,
        )
        self._post_marker(
            issue=spec.issue,
            marker="epm:cluster-launched",
            note=marker_body,
            version=1,
            by="backends.slurm",
        )

        return RunHandle(
            backend="cluster",
            cluster=cluster.name,
            job_id=job_id,
            pod_name=name,
            scratch_dir=scratch_dir,
            log_path=log_path,
            extra={
                "account": cluster.account,
                "robot_alias": cluster.robot_alias,
                "partition": cluster.partition,
                "intent": spec.intent,
                "time_budget_hours": time_h,
                "gpus_per_node": gpus,
                "issue": spec.issue,
            },
        )

    def estimate_start(self, spec: RunSpec) -> datetime | None:
        """Informational ``sbatch --test-only`` estimate, never a gate.

        The selector logs the estimate but uses an explicit
        ``max_wait_seconds`` watchdog for the actual park-decision.
        """
        cluster = self._cluster_for_spec(spec)
        scratch_dir = _scratch_dir_for(spec, cluster)
        plan = stages_for_spec(spec)
        plan_hash = spec.extra.get("plan_hash")
        script = render_sbatch(
            spec=spec,
            cluster=cluster,
            plan=plan,
            scratch_dir=scratch_dir,
            plan_hash=plan_hash,
        )
        return self._start_estimator(
            robot_alias=cluster.robot_alias,
            sbatch_script=script,
        )

    # ----- monitor ---------------------------------------------------------

    def poll(self, handle: RunHandle) -> PollResult:
        """Delegate to :mod:`slurm_monitor` for the live poll.

        Threads ``handle.extra['issue']`` through so the monitor can
        post ``epm:cluster-poll`` / ``epm:cluster-terminal`` markers
        addressed to the originating task. The launch path always
        populates this; if a handle was hand-constructed without it,
        we raise loudly (silent skipping would cost the marker trail).
        """
        if self._poll_fn is None:
            # Lazy import to avoid the circular at module-load time.
            from explore_persona_space.backends.slurm_monitor import build_poll_result

            self._poll_fn = build_poll_result
        cluster = get_cluster_config(handle.cluster) if handle.cluster else None
        if cluster is None:
            raise ValueError(f"SlurmBackend.poll: handle has no cluster ({handle!r})")
        issue = handle.extra.get("issue")
        if issue is None:
            raise ValueError(
                f"SlurmBackend.poll: handle.extra missing 'issue' ({handle!r}). "
                "The launch path populates this; hand-constructed handles must too "
                "so the monitor can post epm:cluster-poll / epm:cluster-terminal."
            )
        return self._poll_fn(
            issue=int(issue),
            job_id=handle.job_id,
            cluster=cluster,
            scratch_dir=handle.scratch_dir,
            log_path=handle.log_path,
        )

    def fetch_logs(self, handle: RunHandle) -> str:
        """Read the rsync'd ``job.out`` tail and return the last 200 lines.

        The monitor (``slurm_monitor.rsync_status_and_log``) rsyncs the
        cluster's ``job.out`` into ``/tmp/slurm-<job_id>/job.out`` —
        flat under the per-job dir, NO additional subdir. The previous
        implementation computed
        ``/tmp/slurm-<id>/<basename(scratch_dir)>/job.out`` and ALWAYS
        missed the file (returning ``""`` on every call) because the
        monitor writes the file one level higher. We reuse the
        ``_local_state_dir`` helper that the monitor uses so the two
        stay in lockstep.

        Returns a newline-joined string (not the Python list repr that
        ``splitlines()[-200:].__str__()`` would produce). ``""`` if the
        local log file doesn't exist yet (a poll never landed).
        """
        # Import lazily to avoid the circular at module-load (monitor
        # imports from this module).
        from explore_persona_space.backends.slurm_monitor import _local_state_dir

        local_path = _local_state_dir(handle.job_id) / "job.out"
        if not local_path.exists():
            return ""
        with local_path.open("rb") as fh:
            data = fh.read()
        lines = data.decode("utf-8", errors="replace").splitlines()[-200:]
        return "\n".join(lines)

    # ----- teardown --------------------------------------------------------

    def fetch_results(self, handle: RunHandle) -> None:
        """rsync ``eval_results/`` + ``figures/`` back to the VM.

        Mirrors the RunPod ``pod.py sync results`` flow. The cluster
        side writes them under ``$SCRATCH_JOB_DIR/out/{eval_results,
        figures}`` (the workload's existing in-job upload writes to
        the canonical project-relative paths, which here resolve under
        the rsync'd tree at ``$SCRATCH_JOB_DIR``).
        """
        cluster = get_cluster_config(handle.cluster) if handle.cluster else None
        if cluster is None:
            raise ValueError(f"SlurmBackend.fetch_results: handle has no cluster ({handle!r})")
        # Pull eval_results/ + figures/ from $SCRATCH_JOB_DIR back to repo root.
        # ``--mkpath`` on the pull direction too (rsync sometimes needs it for
        # the local destination chain).
        local_root = self._src_root
        for subdir in ("eval_results", "figures"):
            src = f"{cluster.robot_alias}:{handle.scratch_dir}/{subdir}/"
            dst = str(local_root / subdir) + "/"
            argv = ["rsync", "-a", "--mkpath", "--partial", src, dst]
            logger.info("rsync pull %s → %s", src, dst)
            subprocess.run(argv, check=False, timeout=300)
        # Non-fatal: a job that produced no figures (eval-only) is fine.

    def confirm_artifacts(self, handle: RunHandle) -> bool:
        """Delegated to the upload-verifier agent.

        SLURM path mirrors RunPod here — the agent runs the same checks
        (HF Hub, WandB, git-committed eval_results/figures) regardless
        of which backend ran the workload.
        """
        del handle
        raise NotImplementedError(
            "SlurmBackend.confirm_artifacts: orchestrator dispatches the "
            "upload-verifier agent today. Wire when the agent's checks are "
            "folded into a Python helper."
        )

    def teardown(self, handle: RunHandle) -> None:
        """``scancel`` the job; idempotent on a missing/terminated id."""
        cluster = get_cluster_config(handle.cluster) if handle.cluster else None
        if cluster is None:
            raise ValueError(f"SlurmBackend.teardown: handle has no cluster ({handle!r})")
        self._cancel(robot_alias=cluster.robot_alias, job_id=handle.job_id)

    # ----- internal helpers ------------------------------------------------

    def _cluster_for_spec(self, spec: RunSpec) -> ClusterConfig:
        if spec.cluster:
            return get_cluster_config(spec.cluster)
        # Fallback: when the spec was built without an explicit cluster
        # (defensive — the selector always threads cluster through), pick
        # Nibi as the v1 default.
        return get_cluster_config("nibi")


def _default_src_root() -> Path:
    """Locate the repo root: walk up until ``pyproject.toml`` is found."""
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "pyproject.toml").exists():
            return parent
    return Path.cwd()


# ---------------------------------------------------------------------------
# Re-exports
# ---------------------------------------------------------------------------


__all__ = [
    "CLUSTER_CONFIGS",
    "HEARTBEAT_INTERVAL_SECONDS",
    "PREFLIGHT_FAIL_MARKER",
    "RSYNC_EXCLUDE_PATTERNS",
    "RSYNC_INCLUDE_PATHS",
    "SECRET_ENV_KEYS",
    "ClusterConfig",
    "SbatchPlan",
    "SlurmBackend",
    "Stage",
    "WorkloadKind",
    "build_rsync_command",
    "compute_plan_hash",
    "default_gpus_for_intent",
    "get_cluster_config",
    "job_name",
    "parse_job_id",
    "post_marker_via_task_py",
    "render_sbatch",
    "render_secrets_env",
    "scp_push_secrets",
    "ssh_scancel",
    "ssh_submit",
    "stages_for_spec",
    "time_budget_hours",
]
