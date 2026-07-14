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
import shutil
import subprocess
import tempfile
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

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

    * ``name`` — the canonical cluster name (``nibi``, ``fir``, ``mila``).
      Used as the dict key in :data:`CLUSTER_CONFIGS` AND as the
      ``BackendKind`` alias the selector resolves to a backend instance.
    * ``account`` — SLURM ``--account`` value. ``rrg-bengioy-ad_gpu``
      for the DRAC robot. **Optional** (``None``) — Mila does not require
      ``--account`` on most partitions; the renderer omits the
      ``#SBATCH --account=`` line when this is ``None``.
    * ``robot_alias`` — the SSH alias the submit + teardown shell out to
      (e.g. ``robot-nibi`` for the DRAC robot key, ``mila`` for the Mila
      interactive ControlMaster socket). Named ``robot_alias`` for
      historical reasons (the v1 slice shipped DRAC-only); the
      :attr:`ssh_host` property is the semantic alias the rest of the
      module should read.
    * ``access_mode`` — how the SSH connection is authenticated.
      ``"robot"`` (DRAC default) = a restricted forced-command robot key
      bound to ``robot_alias``, no MFA, IP-whitelisted, allowlist-
      constrained (`sbatch`/`scancel`/`squeue`/`scp`/`rsync` only — no
      `sinfo`, no `sacct`, no `bash -c`). ``"interactive"`` (Mila) =
      a normal interactive SSH session reused through a 12 h
      ControlMaster socket; the user runs `ssh mila` once (enters the
      email-OTP MFA), then the persistent socket is reused by the
      orchestrator for ControlPersist hours with NO further MFA prompt.
      The router gates ``"interactive"`` lanes behind
      :func:`mila_socket_alive` (or its caller-injected equivalent), so a
      dead socket cleanly skips the lane rather than blocking a run.
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
    * ``timezone`` — IANA timezone name (``zoneinfo.ZoneInfo`` key) the
      cluster's SLURM scheduler reports timestamps in. DRAC clusters
      (Nibi, Fir, Trillium, Narval, Rorqual) report cluster-LOCAL time
      in ``sbatch --test-only`` output (``to start at 2026-06-09T02:06:36``);
      naively calling ``.replace(tzinfo=UTC)`` on that ISO string
      mislabels it by the local UTC offset (~4-5 h on Eastern, more
      across DST boundaries — every job reads as far-past, so the
      router treats a busy cluster as "instant"). The router localizes
      via ``ZoneInfo(cluster.timezone)`` then converts to UTC instead.
      Defaults to ``America/Toronto`` (DRAC robot login nodes report
      in that zone); set per-cluster only when the cluster reports in
      a different zone (Mila = ``America/Montreal``).
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
    account: str | None
    robot_alias: str
    max_gpus_per_node: int
    scratch_path: str
    # Cf. ``access_mode`` docstring above. Defaults to ``"robot"`` so
    # adding a new DRAC cluster requires zero opt-in; Mila explicitly
    # sets ``access_mode="interactive"`` to enable the socket-alive gate.
    access_mode: Literal["robot", "interactive"] = "robot"
    # DRAC requires a GPU TYPE in ``--gpus-per-node`` (e.g. ``h100:1``); a
    # bare count is read as a GPU-type name and sbatch rejects it ("There is
    # no 1 GPU-type"). Nibi + Fir are both H100. Override for a non-H100 system.
    gpu_type: str = "h100"
    # IANA tz of the cluster scheduler's reported timestamps. DRAC robot
    # login nodes report in cluster-local time (Eastern); Mila is the same.
    # The router's est-start parser localizes ``--test-only`` output via
    # this zone, then converts to UTC. See the field docstring above for
    # the timezone-mislabel bug this guards against.
    timezone: str = "America/Toronto"
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

    @property
    def ssh_host(self) -> str:
        """The SSH alias every backend command should dispatch through.

        Equals :attr:`robot_alias` today (the v1 slice shipped DRAC-only,
        so the historical field name `robot_alias` already names the SSH
        host of record). Read through this property in all callers so
        Mila's interactive ``mila`` alias and DRAC's ``robot-<cluster>``
        alias share one read path — and so a future split (e.g. a
        cluster that wants distinct robot-key vs interactive aliases)
        is a one-field change here without re-touching every shell-out
        helper.
        """
        return self.robot_alias


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
        timezone="America/Toronto",  # DRAC robot reports cluster-local Eastern time
    ),
    "fir": ClusterConfig(
        name="fir",
        account="rrg-bengioy-ad_gpu",
        robot_alias="robot-fir",
        max_gpus_per_node=4,
        scratch_path="/scratch/tjiral",  # DRAC $SCRATCH = /scratch/<user>; verified by probe
        timezone="America/Toronto",  # DRAC robot reports cluster-local Eastern time
        available=False,
    ),
    "mila": ClusterConfig(
        name="mila",
        # SLICE-8-VERIFY: Mila's `main`/`long`/`unkillable` partitions do
        # NOT require `--account` for the default project. If a future
        # Mila move forces a project account, set this to the project id
        # and verify in the live acceptance run.
        account=None,
        # The interactive ControlMaster alias from ~/.ssh/clusters.config
        # (Host `mila`); the SSH socket is the 12 h email-OTP-authed
        # ControlPersist socket the user warms by hand once per day.
        robot_alias="mila",
        access_mode="interactive",
        # SLICE-8-VERIFY: Mila login nodes report scheduler timestamps in
        # America/Montreal (Eastern). Same offset as America/Toronto under
        # all current DST windows; named distinctly so a future Mila DC
        # move (e.g. to a Western Canada satellite) can override without
        # touching DRAC.
        timezone="America/Montreal",
        # SLICE-8-VERIFY: Mila in-house cluster typically has H100 nodes
        # at 4-8 GPUs/node; the conservative 8 matches the largest
        # documented single-node allocation. Confirm with `sinfo -p main
        # --Format=Gres` over the live socket in slice 8 before raising.
        max_gpus_per_node=8,
        # SLICE-8-VERIFY: Mila scratch convention is
        # `/network/scratch/<first-letter-of-username>/<username>`. The
        # user is `thomas.jiralerspong` (cf. clusters.config), so the
        # leading letter is `t`. Confirm path is writable + has the EPS
        # quota headroom in the slice-8 acceptance run.
        scratch_path="/network/scratch/t/thomas.jiralerspong",
        # SLICE-8-VERIFIED (issue 535 live, 2026-06-10): Mila's h100
        # nodes (cn-n001/cn-n002) sit ONLY in `short-unkillable` — an
        # h100 GRES request in the default `long` partition fails sbatch
        # with 'Requested node configuration is not available' (lane r4
        # crash). a100l (A100-80GB, the lora-7b workhorse class) is what
        # main/long actually serve; a100l:1 test-submitted clean with an
        # immediate-start estimate.
        gpu_type="a100l",
        # SLICE-8-VERIFY: Mila uses LMod modules; the EasyBuild stack may
        # name the CUDA module differently than DRAC's bare `cuda`.
        # Common candidates: `cuda/12.4` / `cudacore/12.4`. Confirm with
        # `module spider cuda` over the live socket; update this line.
        module_load_cuda="module load cuda",
        # SLICE-8-VERIFY: same CUDA_HOME bridge as DRAC works on most
        # EasyBuild stacks. Confirm in acceptance.
        available=True,
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


def scratch_dir_for(spec: RunSpec, cluster: ClusterConfig) -> str:
    """Destination on the cluster: ``$SCRATCH/eps/issue-<N>``.

    Public — the dispatch-issue ``_reconnect`` closure imports this to
    rebuild a recovered RunHandle's ``scratch_dir`` so the dispatcher
    never reaches into a private helper across modules (parity with
    other publicly-exported slurm helpers like :func:`job_name` and
    :func:`get_cluster_config`).

    The trailing path is computed VM-side (we don't inherit ``$SCRATCH``
    from the cluster env). The cluster admin's ``$SCRATCH`` is mapped
    to :attr:`ClusterConfig.scratch_path`.
    """
    return f"{cluster.scratch_path}/eps/issue-{spec.issue}"


def sentinel_relpath_for(issue: int, attempt_id: str) -> str:
    """Repo-relative completion-sentinel path, attempt-namespaced (#598).

    Single source of truth shared by ``render_sbatch`` (attempt_id =
    ``'slurm-${SLURM_JOB_ID}'``, runtime-expanded inside the sbatch) and
    ``SlurmBackend.launch`` (attempt_id = ``'slurm-<job_id>'``,
    concrete). Attempt-namespaced because the per-issue scratch dir is
    reused across attempts and ``_clear_runtime`` deletes only
    root-level :data:`RUNTIME_ARTIFACT_FILENAMES` (the rsync include
    trick cannot reach a nested file) — a non-namespaced sentinel from a
    prior attempt would masquerade as this attempt's clean exit (the
    staleness class GCP closes with per-attempt dirs;
    ``_check_sentinel`` validates phase+issue only, so the PATH carries
    the defense).
    """
    from explore_persona_space.backends.artifacts import SENTINEL_FILENAME

    return f"eval_results/issue_{issue}/{attempt_id}/{SENTINEL_FILENAME}"


def expected_artifacts_declaration(
    *,
    spec: RunSpec,
    job_id: str,
    src_root: Path | None = None,
) -> dict[str, Any]:
    """SLURM ``EXPECTED_ARTIFACTS_HANDLE_KEY`` payload (#598).

    GCP-parity declaration shape via the shared
    :func:`~explore_persona_space.backends.artifacts.build_expected_artifacts_declaration`,
    with the one SLURM-specific decision: the declared ``sentinel_path``
    is the LOCAL post-rsync repo path (``<src_root>/eval_results/
    issue_<N>/slurm-<job_id>/.completion-sentinel.json``). Finalize runs
    ``fetch_results`` BEFORE ``confirm_artifacts`` (the #588 ordering
    fix) and the existing rsync pull carries everything under
    ``$SCRATCH_JOB_DIR/eval_results/`` — dotfiles included — so the
    verifier's default local-FS reader just works with zero new
    transport code. The attempt id is ``slurm-<job_id>`` (the #588
    ``EPS_ATTEMPT_ID`` convention), known only AFTER ``ssh_submit``
    returns — the one structural delta from GCP, which mints its
    attempt id pre-provision.
    """
    from explore_persona_space.backends.artifacts import build_expected_artifacts_declaration

    root = src_root or _default_src_root()
    attempt_id = f"slurm-{job_id}"
    return build_expected_artifacts_declaration(
        issue=spec.issue,
        sentinel_path=str(root / sentinel_relpath_for(spec.issue, attempt_id)),
        custom_workload=bool(spec.workload_cmd),
        attempt_id=attempt_id,
        wandb_run_path=spec.extra.get("wandb_run_path"),
        # #685 / #661: thread the per-issue worktree git root + the
        # phase-scope flag off spec.extra (same channel as wandb_run_path;
        # _launch_extra_from_args populates both). The SLURM lane rsyncs
        # the src_root tree rather than checking out a worktree, so a baked
        # git_repo_root is usually inert here — but threading it keeps the
        # builder call uniform across all three lanes (and the launch +
        # reconnect SLURM handles both call expected_artifacts_declaration
        # with the SAME spec, so both pick it up). None / False = current.
        git_repo_root=spec.extra.get("git_repo_root"),
        skip_default_git_paths=bool(spec.extra.get("skip_default_git_paths", False)),
    )


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
    # ``data/sft/`` carries the small committed training-mix JSONLs that
    # ``stages[].dataset`` references repo-relatively (e.g. the 188K
    # router-smoke set). The RunPod lane gets them via git clone; the
    # rsync lane missed them until live attempt 4 crashed with
    # ``FileNotFoundError: data/sft/router_smoke_sft.jsonl`` (issue 535).
    "./data/sft",
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
# Runtime-artifact clearing (VM → cluster scratch, fresh per prepare)
# ---------------------------------------------------------------------------


# Scratch-root files the RUNNING job writes (NOT part of the code rsync).
# ``prepare`` clears these before a fresh submit: the scratch dir is
# per-ISSUE and reused across attempts, SLURM truncates ``--output``
# only when the new job STARTS, and the never-started window is exactly
# what the router's started-evidence probe inspects — so a stale
# prior-attempt ``status.json`` / ``job.out`` turns every re-run
# terminal into a false "workload failure" (issue 535 attempt 2).
RUNTIME_ARTIFACT_FILENAMES: tuple[str, ...] = (
    "status.json",
    "job.out",
    ".current_phase",
    "preflight.json",
)


def build_clear_runtime_artifacts_command(
    *,
    empty_dir: str,
    dest_root: str,
    robot_alias: str,
    filenames: tuple[str, ...] = RUNTIME_ARTIFACT_FILENAMES,
) -> list[str]:
    """Build the rsync argv that DELETES the runtime artifacts on the cluster.

    The robot forced-command wrapper does NOT allowlist ``ssh <alias>
    rm``, so deletion rides rsync's include/exclude filter semantics:
    sync an EMPTY local dir with ``--include`` of exactly the runtime
    filenames + ``--exclude='*'`` + ``--delete``. Files matching an
    include that are absent on the (empty) sender are deleted on the
    receiver; everything else — the code tree, ``secrets.env``,
    subdirectories — is excluded, and excluded entries are protected
    from ``--delete`` (rsync deletes excluded files only under
    ``--delete-excluded``, which we deliberately do NOT pass).

    Flags follow :func:`build_rsync_command` conventions (``-a`` for
    wrapper parity, ``--mkpath`` so a first-ever prepare with no
    scratch dir yet succeeds instead of erroring). Pure function — the
    golden test asserts the argv without touching the cluster.
    """
    argv: list[str] = ["rsync", "-a", "--delete", "--mkpath"]
    for name in filenames:
        argv.extend(["--include", name])
    argv.extend(["--exclude", "*"])
    argv.append(f"{empty_dir.rstrip('/')}/")
    argv.append(f"{robot_alias}:{dest_root}/")
    return argv


def clear_runtime_artifacts(
    *,
    robot_alias: str,
    scratch_dir: str,
    timeout: int = 120,
) -> None:
    """Delete prior-attempt runtime artifacts from the cluster scratch root.

    Wraps :func:`build_clear_runtime_artifacts_command` with a
    short-lived empty staging dir. Raises on non-zero exit — a clear
    that silently fails leaves the started-evidence probe poisoned by
    stale artifacts, which is exactly the misclassification this
    exists to prevent (fail fast, never hide failures).
    """
    with tempfile.TemporaryDirectory(prefix="eps-slurm-clear-") as empty_dir:
        argv = build_clear_runtime_artifacts_command(
            empty_dir=empty_dir,
            dest_root=scratch_dir,
            robot_alias=robot_alias,
        )
        logger.info("clearing runtime artifacts at %s:%s", robot_alias, scratch_dir)
        subprocess.run(argv, check=True, timeout=timeout)


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

# Non-secret env keys passed through to the in-job environment via the
# same sourced env file. These are plain configuration values — the
# delete-after-eval adapter-persist targets ``trainer.py:_persist_adapter``
# reads from ``os.environ`` ON THE COMPUTE NODE (see
# ``.claude/rules/upload-policy.md``) — NOT secrets, so they live in a
# SEPARATE list to keep ``SECRET_ENV_KEYS`` semantically "secrets only".
# Without this passthrough, a value set on the dispatch process env
# (e.g. by ``scripts/router_acceptance.py --live``) never reaches the
# remote workload and the HF adapter upload silently no-ops.
PASSTHROUGH_ENV_KEYS: tuple[str, ...] = (
    "EPM_PERSIST_ADAPTER_HF_REPO",
    "EPM_PERSIST_ADAPTER_SUBFOLDER",
    # HF public-storage headroom knobs (#564): the soft ceiling, the opt-in
    # overflow routing, the kill switch, and the cache TTL must reach the
    # compute node or a dispatch-process opt-in silently no-ops remotely.
    # EPM_HF_STORAGE_CACHE_PATH is deliberately NOT threaded (a VM-local
    # path is wrong on the worker; workers use the default).
    "EPM_HF_STORAGE_SOFT_CEILING_TB",
    "EPM_HF_OVERFLOW_ROUTING",
    "EPM_HF_STORAGE_CHECK",
    "EPM_HF_STORAGE_CACHE_TTL_S",
    # Size-aware projected-headroom probe floor (#1034): same remote-relevance
    # as the #564 knobs above — a dispatch-process floor override must reach
    # the compute node or it silently no-ops remotely.
    "EPM_HF_LARGE_UPLOAD_PROBE_GB",
    # HF Hub upload accelerator OVERRIDE channel (#745): forwarded so a
    # dispatch-process =0 / HF_HUB_DISABLE_XET=1 (the #515/#931 xet workaround)
    # reaches the compute node. The DEFAULTS (=1) are a STATIC env block in
    # render_sbatch (the cuda_setup block, which the secrets `source` follows
    # and overrides); this passthrough is the override channel only.
    # Drop-when-absent contract preserved (render_secrets_env skips an unset
    # key, so the static default stands).
    "HF_XET_HIGH_PERFORMANCE",
    "HF_HUB_ENABLE_HF_TRANSFER",
    # The REAL xet kill switch (#1195): huggingface_hub reads HF_HUB_DISABLE_XET
    # (see the gcp.py twin comment + .claude/rules/upload-policy.md).
    "HF_HUB_DISABLE_XET",
    # Legacy no-op alias (verified #1049) — kept for old launch commands.
    "HF_XET_DISABLE",
)


def render_secrets_env(
    env: dict[str, str] | None = None,
    keys: tuple[str, ...] = SECRET_ENV_KEYS + PASSTHROUGH_ENV_KEYS,
) -> str:
    """Render a ``KEY=value`` env file for the sbatch ``set -a; source`` stanza.

    Plain ``KEY=value`` lines (no ``export`` — the sbatch wraps the
    source in ``set -a / set +a`` so every assignment auto-exports;
    confirmed in P0(c)). Values are shell-quoted via :func:`shlex.quote`
    so a token with shell-meaningful chars survives the round trip.

    Only keys present in ``env`` are rendered (a missing key means the
    VM operator never set it — the in-job preflight will FAIL fast and
    the selector falls back to RunPod, exactly the intended path).

    The default key set is ``SECRET_ENV_KEYS`` plus the non-secret
    :data:`PASSTHROUGH_ENV_KEYS` (adapter-persist targets) — the env
    file is the one remote-env surface every sbatch already sources, so
    both classes ride it; the split lists keep the semantics distinct.
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
# launcher; ``custom`` = a verbatim shell command line from
# ``RunSpec.workload_cmd`` (#588). Typed as a ``Literal`` so the
# renderer's terminal ``else: raise`` (``unknown stage backend``) is
# provably exhaustive — adding a backend kind requires extending this
# alias AND the renderer dispatch in lockstep, surfaced by the type
# checker.
WorkloadKind = Literal["local", "open_instruct", "custom"]


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
    * ``custom_cmd`` — full shell command line for ``custom`` stages
      (#588); rendered as the single argument of an rc-preserving inner
      ``bash -eu -o pipefail -c`` wrapper (#1004 — a bare splice under
      the prelude's ``set -e`` rc-masks a ``cmd1 && cmd2`` first-command
      crash), runs from ``$SCRATCH_JOB_DIR`` (the rsynced repo). Ignored
      for the other backends.
    """

    name: str
    backend: WorkloadKind
    script_rel: str
    deepspeed_config_rel: str | None = None
    hydra_args: tuple[str, ...] = ()
    oi_args: tuple[str, ...] = ()
    custom_cmd: str = ""


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

    A spec carrying ``workload_cmd`` (#588) bypasses the intent → stage
    table: the custom command IS the workload, rendered as a single
    ``custom`` stage. The intent keeps driving GPUs/node + ``--time``
    via :func:`default_gpus_for_intent` / :func:`time_budget_hours`
    (unchanged).
    """
    if spec.workload_cmd:
        return SbatchPlan(
            stages=(
                Stage(
                    name="workload",
                    backend="custom",
                    script_rel="",
                    custom_cmd=spec.workload_cmd,
                ),
            )
        )
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
    ]
    if cluster.account is not None:
        # Mila's default partitions do NOT require an explicit account
        # line; emitting an empty one (``#SBATCH --account=``) is
        # rejected by some SLURM builds, so the line is skipped entirely
        # when the cluster row omits it. DRAC rows always set an account.
        sbatch_headers.append(f"#SBATCH --account={cluster.account}")
    sbatch_headers.extend(
        [
            f"#SBATCH --job-name={name}",
            "#SBATCH --nodes=1",
            "#SBATCH --ntasks-per-node=1",
            f"#SBATCH --gpus-per-node={cluster.gpu_type}:{gpus}",
            f"#SBATCH --cpus-per-task={min(8 * gpus, 64)}",
            f"#SBATCH --mem={min(64 * gpus, 480)}G",
            f"#SBATCH --time={time_str}",
            f"#SBATCH --output={output_path}",
        ]
    )
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
        "# Authoritative current-phase file. The background heartbeat reads",
        "# THIS (not a captured shell var) so it reports the LIVE phase — a",
        "# bg subshell freezes CURRENT_PHASE at fork time otherwise (the",
        "# heartbeat would keep writing the startup phase through every stage).",
        'PHASE_FILE="$SCRATCH_JOB_DIR/.current_phase"',
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
        '    _write_status "$(cat "$PHASE_FILE" 2>/dev/null || echo startup)"',
        '    sleep "$HEARTBEAT_INTERVAL"',
        "  done",
        "}",
        "",
        "# Start the heartbeat NOW (before the long venv build) + write an",
        "# initial status.json so a RUNNING job ALWAYS has a fresh heartbeat.",
        "# Otherwise the monitor reads `stalled` for the whole ~6-40 min venv",
        "# build, since status.json wouldn't exist until preflight.",
        'CURRENT_PHASE="startup"',
        'echo startup > "$PHASE_FILE"',
        '_write_status "startup"',
        "_heartbeat_loop &",
        "HEARTBEAT_PID=$!",
        "# Heartbeat-kill trap; the secrets stanza upgrades it to also shred",
        "# the secrets file once SECRETS_FILE is defined.",
        "trap 'kill $HEARTBEAT_PID 2>/dev/null || true' EXIT TERM INT",
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
        # Fast HF Hub uploads (#745) — STATIC DEFAULT export. HF_XET_HIGH_PERFORMANCE
        # is the PRIMARY accelerator (the project repos use the Xet backend);
        # HF_HUB_ENABLE_HF_TRANSFER is the orthogonal LFS accelerator (hf_transfer
        # is a hard dep). The PASSTHROUGH keys (PASSTHROUGH_ENV_KEYS) are the
        # OVERRIDE channel — a forwarded dispatch-process =0 / HF_HUB_DISABLE_XET=1 is
        # sourced from secrets.env in the LATER secrets_setup block (under set -a),
        # so it supersedes these static defaults. The :- form also honors a value
        # inherited from the compute-node shell.
        'export HF_XET_HIGH_PERFORMANCE="${HF_XET_HIGH_PERFORMANCE:-1}"',
        'export HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-1}"',
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
        "# never leaves the secrets file on $SCRATCH. Combined with the",
        "# heartbeat kill (the loop started at startup, before this stanza).",
        "trap 'kill $HEARTBEAT_PID 2>/dev/null || true; "
        'shred -u "$SECRETS_FILE" 2>/dev/null '
        '|| rm -f "$SECRETS_FILE"\' EXIT TERM INT',
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
        "# Tokens: must be in env post-source. xtrace MUST be OFF around",
        "# these checks: under `set -x` the ${VAR:?} expansion traces the",
        "# EXPANDED value (`+ : hf_…`) into job.out, and the monitor's log",
        "# tails carry job.out into git-committed markers (round-6 C1 —",
        "# the issue-535 live run leaked both tokens this way).",
        "set +x",
        ': "${HF_TOKEN:?HF_TOKEN missing from secrets.env}"',
        ': "${WANDB_API_KEY:?WANDB_API_KEY missing from secrets.env}"',
        "set -x",
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
        "# Preflight PASS. (Heartbeat already running since startup; the",
        "# combined kill+shred trap was set in the secrets stanza.)",
        "",
    ]

    # Stage commands.
    master_addr = "${MASTER_ADDR:-localhost}"
    master_port = "${MASTER_PORT:-29500}"
    stage_blocks: list[str] = [
        "# === Repo-root PYTHONPATH (#1172; trap #823/#853) ===",
        "# Script-mode python puts the SCRIPT's dir on sys.path[0], so a",
        "# deferred `from scripts.X import ...` in a src-layout driver",
        "# crashes with ModuleNotFoundError on the compute node. Every",
        "# stage runs from $SCRATCH_JOB_DIR (the rsynced repo — the venv",
        "# block cd'd there), so prepend it once for ALL stage backends",
        "# (local / custom / open_instruct). Placed AFTER the module-load",
        "# block so cluster-module PYTHONPATH entries are preserved after",
        "# ours; :+ avoids the empty-value trailing colon (cwd-injection,",
        "# cpython #107353) and is safe under the prelude's set -u.",
        'export PYTHONPATH="$SCRATCH_JOB_DIR${PYTHONPATH:+:$PYTHONPATH}"',
        "",
    ]
    for stage in plan.stages:
        stage_blocks.append(f"# === Stage: {stage.name} ===")
        stage_blocks.append(f'CURRENT_PHASE="{stage.name}"')
        stage_blocks.append(f'echo "{stage.name}" > "$PHASE_FILE"')
        stage_blocks.append(f'echo "[phase={stage.name}]"')
        stage_blocks.append(f'_write_status "{stage.name}"')
        if stage.backend == "local":
            # Hydra-style: uv run python <script> arg1 arg2 ...
            args_joined = " ".join(shlex.quote(a) for a in stage.hydra_args)
            stage_blocks.append(
                f"uv run python {shlex.quote(stage.script_rel)} {args_joined}".rstrip()
            )
        elif stage.backend == "custom":
            if not stage.custom_cmd:
                raise ValueError(f"custom stage {stage.name!r} requires custom_cmd")
            # EPS_* env contract parity with the GCP startup script
            # (#588 live-smoke fix: nibi job 15955646 died on
            # `EPS_ISSUE: parameter null or not set` — custom dispatch
            # scripts rely on these the way they do on the GCP lane).
            # SLURM has no GCE attempt_id; the job id is the per-
            # submission unique analogue.
            stage_blocks.append(f"export EPS_ISSUE={spec.issue}")
            stage_blocks.append('export EPS_ATTEMPT_ID="slurm-${SLURM_JOB_ID}"')
            # WandB project default (#601 follow-up r1) — parity with the
            # GCP workload_cmd lane: HF-Trainer workloads that never set
            # WANDB_PROJECT land in WandB's global default project
            # 'huggingface', violating the Upload Policy (training
            # metrics → project=<experiment_name>). :- fills only
            # unset/empty, so an inline WANDB_PROJECT=... prefix on the
            # workload command (or the workload setting its own project
            # internally) wins. Deliberately NOT in PASSTHROUGH_ENV_KEYS:
            # an ambient WANDB_PROJECT on the dispatch process would
            # silently cross-route a new issue's metrics.
            stage_blocks.append(f'export WANDB_PROJECT="${{WANDB_PROJECT:-issue{spec.issue}}}"')
            # Inner-bash embed (#588, rc-wrapper #1004) — the command IS
            # a complete shell line, re-parsed by an inner
            # `bash -eu -o pipefail -c` (see the wrapper comment at the
            # append below); it runs from $SCRATCH_JOB_DIR (the rsynced
            # repo), so repo-relative `bash scripts/...` resolves.
            # Heartbeat / status.json / [phase=...] markers wrap it
            # unchanged.
            # NO sentinel channel on this lane (#608 follow-up): the
            # RunPod/GCP `/workspace/logs/issue-<N>-*.json` marker
            # contract does NOT hold on SLURM — compute nodes have no
            # /workspace and the robot wrapper cannot run the drain
            # shell (see slurm_monitor's module docstring). A dispatch
            # script that depends on sentinel-carried markers
            # (epm:results payloads, gate fields) fails loud at its
            # `mkdir -p /workspace/logs` and must be routed to the
            # gcp/runpod lane at plan time.
            # MUST-BLOCK contract (#601 follow-up): the command must run
            # the workload to completion in the foreground. The terminal
            # [phase=done] + status.json "done" blocks below execute the
            # moment this line returns, the batch script exits, and
            # SLURM both marks the job COMPLETED (monitor verdict:
            # interpret) AND tears down the job cgroup — killing any
            # setsid-detached children. The GCP lane's detached-pid wait
            # (gcp.py: fresh /workspace/logs/*.pid) is NOT portable
            # here: compute nodes have no /workspace and no SLURM-side
            # pid-file convention exists. A self-daemonizing dispatch
            # script must be made blocking or routed to the gcp/runpod
            # lane at plan time.
            # rc-preserving wrapper (#1004, GCP parity — incident #952):
            # the bare splice rc-masked a `cmd1 && cmd2` first-command
            # crash under the prelude's set -e (errexit exempts non-final
            # &&/|| list members), letting the terminal [phase=done] +
            # completion sentinel publish over a crash. The inner
            # -eu -o pipefail mirror the prelude flags. Residual: an
            # `a && b; c` shape where `a` fails and `c` succeeds still
            # masks (the same errexit exemption applies inside the inner
            # bash).
            stage_blocks.append(f"bash -eu -o pipefail -c {shlex.quote(stage.custom_cmd)}")
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

    # Terminal block. `set -euo pipefail` (prelude) plus the #1004
    # rc-preserving inner-bash wrapper on custom stages guarantee this
    # block is reached only when every stage exited 0, so the completion
    # sentinel written here is a genuine clean-exit proof (#598). (Before
    # #1004 a compound custom_cmd whose first `&&` member crashed fell
    # through errexit and published a false clean exit — the #952 class.
    # The `a && b; c` shape remains a documented residual inside the
    # inner bash.)
    sentinel_rel = sentinel_relpath_for(spec.issue, "slurm-${SLURM_JOB_ID}")
    terminal = [
        "# === Done ===",
        'CURRENT_PHASE="done"',
        "kill $HEARTBEAT_PID 2>/dev/null || true",
        "# === Completion sentinel (workload exited cleanly) — write BEFORE the",
        "# done status so 'done' is published last, mirroring GCP (#598).",
        "# fetch_results rsyncs eval_results/ back to the VM, landing this at",
        "# the LOCAL path the launch-time declaration names.",
        f'SENTINEL_PATH="$SCRATCH_JOB_DIR/{sentinel_rel}"',
        'mkdir -p "$(dirname "$SENTINEL_PATH")"',
        # Unquoted EOF so ${SLURM_JOB_ID} expands at runtime.
        'cat > "$SENTINEL_PATH" <<EOF\n'
        '{"phase":"done","issue":' + str(spec.issue) + ',"attempt_id":"slurm-${SLURM_JOB_ID}"}'
        "\nEOF",
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
        encoding="utf-8",
        errors="replace",
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
        encoding="utf-8",
        errors="replace",
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
# Mila socket-alive probe (interactive ControlMaster gate)
# ---------------------------------------------------------------------------


# Default SSH alias for the Mila interactive ControlMaster session. Matches
# the ``Host mila`` stanza in ``~/.ssh/clusters.config``. Pure constant
# (no ``ClusterConfig`` lookup) so the probe stays cheap + the function
# is callable before any cluster lookup runs.
DEFAULT_MILA_SSH_ALIAS: str = "mila"


def mila_socket_alive(
    *,
    ssh_alias: str = DEFAULT_MILA_SSH_ALIAS,
    timeout: int = 5,
    runner: Callable[[list[str], int], int] | None = None,
) -> bool:
    """Cheap non-interactive probe: is the Mila ControlMaster socket warm?

    Runs ``ssh -o BatchMode=yes -o ConnectTimeout=<timeout> <ssh_alias>
    true`` and returns ``True`` iff the SSH exit code is zero.

    ``BatchMode=yes`` is the load-bearing flag — it tells SSH to NEVER
    prompt for credentials. With a healthy ControlMaster socket the
    command short-circuits through the multiplexed connection and
    returns in milliseconds; with a dead / expired / unauthenticated
    socket it fails fast (non-zero) instead of hanging on an OTP
    prompt. ``ConnectTimeout`` caps the wait if SSH falls back to a
    direct TCP attempt.

    Returns ``False`` (NOT raises) for every failure path:
    - non-zero SSH exit (socket down, OTP expired, host unreachable);
    - ``subprocess.TimeoutExpired`` (the SSH wrapper hung past the cap);
    - any ``OSError`` from spawning the subprocess.

    Returning ``False`` is the DESIGNED graceful path — it tells the
    router "skip Mila this round" without poisoning the run. Socket
    refresh is the operator's job (see the Claude-session OTP-refresh
    cron prompt at ``.claude/cron-prompts/mila-otp-refresh.md`` and the
    ``scripts/mila_socket_refresh.py`` helper).

    ``runner`` is an injection seam for tests: a callable taking
    ``(argv, timeout)`` and returning an int exit code. The production
    default shells out via :mod:`subprocess`.
    """
    argv = [
        "ssh",
        "-o",
        "BatchMode=yes",
        "-o",
        f"ConnectTimeout={timeout}",
        ssh_alias,
        "true",
    ]
    if runner is not None:
        try:
            return runner(argv, timeout) == 0
        except Exception:
            logger.info(
                "mila_socket_alive: injected runner raised on alias=%r; treating as down",
                ssh_alias,
            )
            return False
    try:
        proc = subprocess.run(
            argv,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout + 2,
            check=False,
        )
    except subprocess.TimeoutExpired:
        logger.info(
            "mila_socket_alive: ssh %r timed out after %ds; treating as down",
            ssh_alias,
            timeout + 2,
        )
        return False
    except OSError as exc:
        # ssh binary missing / permission denied on the wrapper. Same
        # gracefulness — skip Mila, don't crash the router.
        logger.info(
            "mila_socket_alive: could not spawn ssh for alias=%r (%s); treating as down",
            ssh_alias,
            exc,
        )
        return False
    if proc.returncode != 0:
        # Stderr is informative on a stale socket (e.g.
        # "Permission denied (publickey,keyboard-interactive)" or
        # "channel 0: open failed: connect failed"). Log truncated for
        # the orchestrator's tick output.
        logger.info(
            "mila_socket_alive: ssh exited %d; alias=%r stderr=%r",
            proc.returncode,
            ssh_alias,
            (proc.stderr or "").strip()[:160],
        )
        return False
    return True


# ---------------------------------------------------------------------------
# estimate_start — sbatch --test-only (ranking HINT for the router)
# ---------------------------------------------------------------------------


# Parses the real ``sbatch --test-only`` output. The verified-on-Nibi shape is
#
#     sbatch: Job 15819682 to start at 2026-06-09T02:06:36 using 1 processors \
#         on nodes g4 in partition gpubase_bygpu_b1
#
# i.e. ``to start at <ISO local time>``. The previous regex matched the
# substring ``"start time …"`` instead, which never appears in the real
# output, so ``ssh_estimate_start`` always returned ``None`` and the router
# had no signal to rank free lanes by. Replaced for the multi-backend
# router (plan ``2026-06-08_224537-multi-backend-compute-router``).
#
# Note: the captured timestamp is in CLUSTER-LOCAL time (DRAC robot login
# nodes report Eastern); the caller MUST localize via the cluster's
# ``ClusterConfig.timezone`` before converting to UTC. Naively wrapping
# the parsed naive datetime with ``.replace(tzinfo=UTC)`` (the prior bug)
# mislabels local time as UTC and skews every estimate by 4-5 h (more
# across DST boundaries) — every job reads as far-past, so a busy cluster
# falsely ranks as "instant".
_EST_START_RE = re.compile(r"to start at (\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2})", re.IGNORECASE)


def ssh_estimate_start(
    *,
    robot_alias: str,
    sbatch_script: str,
    cluster_timezone: str,
    timeout: int = 30,
) -> datetime | None:
    """Best-effort ``sbatch --test-only`` start-time estimate; ranking hint only.

    Submits ``sbatch_script`` over the robot alias with ``--test-only``
    (which never enqueues a job and has no fairshare cost), parses the
    ``to start at <ISO>`` token out of stderr+stdout, and returns the
    parsed estimate as a tz-aware UTC :class:`datetime`. Returns
    ``None`` when the wrapper rejects the call, when the output is
    missing / malformed (e.g. ``sbatch: error: Invalid account``), or
    when the ISO string fails to parse.

    ``cluster_timezone`` is the IANA tz the cluster scheduler reports
    in (e.g. ``America/Toronto`` for DRAC robots). The function
    localizes the parsed naive timestamp via that zone then converts to
    UTC, so the returned ``datetime`` is comparable across clusters.
    Naively assuming UTC (the prior implementation) silently skewed
    every estimate by the local UTC offset.

    The router uses this purely as a ranking HINT — the
    submit-and-park state machine (`route()`'s ≤10-min watchdog) is
    the source of truth for "did the job actually start in time".
    """
    argv = ["ssh", robot_alias, "sbatch", "--test-only"]
    try:
        proc = subprocess.run(
            argv,
            input=sbatch_script,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
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
        naive = datetime.fromisoformat(match.group(1))
    except ValueError:
        return None
    try:
        tz = ZoneInfo(cluster_timezone)
    except ZoneInfoNotFoundError:
        logger.warning(
            "ssh_estimate_start: unknown ZoneInfo key %r; cannot localize estimate",
            cluster_timezone,
        )
        return None
    # ``replace(tzinfo=tz)`` is correct for IANA zones (zoneinfo handles
    # the local-time → UTC offset, including DST). Compare to the old
    # bug which wrapped with ``UTC``: that mislabels Eastern local as
    # UTC and shifts the instant by 4-5 hours.
    localized = naive.replace(tzinfo=tz)
    return localized.astimezone(UTC)


def estimate_start_seconds(
    *,
    spec: RunSpec,
    cluster: ClusterConfig,
    now: datetime | None = None,
    start_estimator=None,
    rendered_script: str | None = None,
) -> float | None:
    """Seconds until ``spec`` would start on ``cluster``, per ``sbatch --test-only``.

    The router calls this once per free-lane candidate to rank lanes by
    estimated start time (the actual decision is gated by the
    submit-and-park watchdog, not by this number). Returns:

    * ``float`` seconds-from-now (may be negative if the cluster
      reports a start time in the past, i.e. "would start immediately"),
    * or ``None`` when the underlying ``sbatch --test-only`` returned
      no parseable estimate (the lane is still park-eligible, just
      cannot be ranked as instant).

    The script used for the probe is rendered with the SAME inputs the
    launch path will use (same ``cluster``, same ``RunSpec``, same
    ``stages_for_spec`` → ``render_sbatch`` pipeline). ``render_sbatch``
    is a pure deterministic function of ``(spec, cluster, plan,
    scratch_dir, plan_hash)``, so the probe script is byte-identical to
    the submit script — what SLURM estimates the start time for is
    exactly what we then submit (no gres / account / time-budget
    mismatch between probe and submit).

    Callers may pass ``rendered_script`` to short-circuit re-rendering
    (e.g. when the router has already produced a script for the
    submit path and wants to reuse it for the probe — guarantees the
    estimate-vs-submit byte identity from the caller side too).
    Otherwise the function renders the script itself using the same
    helpers ``SlurmBackend.launch`` uses.

    ``now`` defaults to ``datetime.now(UTC)``; tests inject a fixed
    instant to keep assertions deterministic.
    """
    estimator = start_estimator or ssh_estimate_start
    if rendered_script is None:
        scratch_dir = scratch_dir_for(spec, cluster)
        plan = stages_for_spec(spec)
        plan_hash = spec.extra.get("plan_hash")
        rendered_script = render_sbatch(
            spec=spec,
            cluster=cluster,
            plan=plan,
            scratch_dir=scratch_dir,
            plan_hash=plan_hash,
        )
    estimate = estimator(
        # Reads the canonical SSH alias (robot-<name> for DRAC, ``mila``
        # for Mila); the ``robot_alias=`` parameter name is historical.
        robot_alias=cluster.ssh_host,
        sbatch_script=rendered_script,
        cluster_timezone=cluster.timezone,
    )
    if estimate is None:
        return None
    if estimate.tzinfo is None:
        # Defensive: an injected test estimator that forgets tz would
        # otherwise raise on the subtraction below. Treat a naive return
        # as unusable rather than guessing the zone.
        logger.warning(
            "estimate_start_seconds: estimator returned naive datetime %r; treating as no-estimate",
            estimate,
        )
        return None
    reference = now or datetime.now(UTC)
    if reference.tzinfo is None:
        reference = reference.replace(tzinfo=UTC)
    return (estimate - reference).total_seconds()


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
        runtime_clearer=None,
        git_branch_resolver=None,
        git_cloner=None,
    ) -> None:
        self._src_root = src_root or _default_src_root()
        # Resolves the rsync source's current branch for the feature-branch
        # / stale-main guard in ``prepare``. Defaults to the real
        # ``git -C <src_root> rev-parse``; tests inject a stub returning a
        # fixed branch name (or ``None`` to simulate a non-repo source).
        self._git_branch_resolver = git_branch_resolver or git_branch_at
        # Materializes a complete rsync source for a non-``main`` repo_branch on
        # the VM (git worktree add of the branch commit + working-tree overlay of
        # the external/open-instruct gitlink) — the #793 mechanism that lets the
        # SLURM lane HONOR repo_branch instead of only refusing stale main.
        # Defaults to the real ``materialize_branch_src`` shell-out; tests inject
        # a stub returning a fake scratch Path and recording the (branch, issue)
        # request.
        self._git_cloner = git_cloner or materialize_branch_src
        self._submit = submitter or ssh_submit
        self._cancel = canceller or ssh_scancel
        self._rsync = rsyncer or run_rsync_sync
        # Prior-attempt runtime-artifact clearing (status.json / job.out /
        # .current_phase / preflight.json) before every fresh submit; see
        # ``clear_runtime_artifacts``. Tests inject a recorder.
        self._clear_runtime = runtime_clearer or clear_runtime_artifacts
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
        """Clear stale runtime artifacts, rsync the repo + secrets file.

        Idempotent — rsync with ``--delete`` brings the destination into
        lockstep regardless of prior state. The secrets file is written
        FRESH on every prepare call so a token rotation propagates
        immediately.

        The runtime-artifact clear runs FIRST: the per-issue scratch dir
        is reused across attempts and the code rsync's ``--delete`` only
        reaches inside the dot-anchored include trees, never the
        scratch-root ``status.json`` / ``job.out`` the previous attempt
        left behind — which the monitor + started-evidence probe would
        otherwise misread as THIS attempt's output (issue 535 attempt
        2). ``prepare`` is only ever called on a FRESH launch (reconnect
        paths skip it by contract), so clearing here cannot race a live
        job's own writes.
        """
        # Resolve the rsync source for the requested branch (#793). The SLURM
        # lane rsyncs from ``self._src_root`` — which ``_default_src_root()``
        # resolves via ``__file__``-walk to the repo-root install on ``main``,
        # NOT the invoking worktree. Unlike the GCP lane (which git-clones
        # ``spec.extra["repo_branch"]`` on the VM), this lane HONORS repo_branch
        # by MATERIALIZING a complete branch tree on the VM (``materialize_branch_src``
        # via the ``git_cloner`` seam) and rsyncing from it. On ``main`` / absent /
        # already-on-branch, ``_resolve_rsync_source`` returns ``self._src_root``
        # unchanged and the path below is byte-identical to the pre-fix behavior.
        rsync_src = self._resolve_rsync_source(spec)
        # Belt-and-suspenders (#653/#793): re-assert the branch guard against the
        # RESOLVED source. This is NOT the pre-fix "refuse the feature branch"
        # behavior — the resolved source is ALWAYS either ``self._src_root``
        # (main/absent/already-on-branch, where the guard was already a no-op) or
        # the materialized branch tree (which IS on the requested branch, so the
        # guard trivially passes). Its post-fix ROLE is to catch a ``git_cloner``
        # that returned a tree NOT on the requested branch (a materialize_branch_src
        # correctness regression), never to refuse a legitimately-requested feature
        # branch. The lane-advance fallback for a genuinely-unresolvable branch is
        # preserved by materialize_branch_src's own ``RuntimeError`` (which
        # ``router._prepare_and_launch`` wraps as ``BackendPrepareError`` identically
        # to the old ``ValueError``).
        self._assert_repo_branch_synced(spec, src_root=rsync_src)
        cluster = self._cluster_for_spec(spec)
        scratch_dir = scratch_dir_for(spec, cluster)
        self._clear_runtime(
            robot_alias=cluster.ssh_host,
            scratch_dir=scratch_dir,
        )
        self._rsync(
            src_root=rsync_src,
            dest_root=scratch_dir,
            robot_alias=cluster.ssh_host,
        )
        secrets = render_secrets_env()
        # Write the secrets file directly via SSH stdin (avoids a tmp
        # file on the VM that could leak). The single-shot dd writes
        # bytes verbatim; we chmod 600 in the same SSH call so it's
        # never world-readable on the cluster side.
        self._push_secrets(cluster, scratch_dir, secrets)

    def _resolve_rsync_source(self, spec: RunSpec) -> Path:
        """Return the rsync source for ``spec``, materializing a branch tree if needed.

        No-op (returns ``self._src_root``) when ``repo_branch`` is absent / ``main`` /
        the install's HEAD already IS the requested branch. Otherwise materializes a
        complete branch tree on the VM via the ``git_cloner`` seam and returns its path.

        Note: an UNPROVABLE source branch (resolver returns ``None``) for a non-``main``
        request also routes to the cloner (``None`` != the requested branch), so the
        source-of-truth for "can we honor this branch?" is now the cloner's own
        ``git rev-parse`` (fail-loud ``RuntimeError`` on an unresolvable branch), NOT the
        guard.
        """
        requested = str(spec.extra.get("repo_branch") or "").strip()
        if not requested or requested == "main":
            return self._src_root
        actual = self._git_branch_resolver(self._src_root)
        if actual == requested:
            return self._src_root  # install already on the branch — rsync it directly
        return self._git_cloner(src_root=self._src_root, branch=requested, issue=spec.issue)

    def _assert_repo_branch_synced(self, spec: RunSpec, src_root: Path | None = None) -> None:
        """Assert the rsync source's HEAD matches a non-``main`` ``repo_branch``.

        Post-#793, ``prepare`` first RESOLVES the rsync source
        (``_resolve_rsync_source`` — materializing a complete branch tree via the
        ``git_cloner`` seam when the install is not already on the branch) and then
        calls this guard against that RESOLVED ``src_root``. So the guard's role is
        now BELT-AND-SUSPENDERS on ``materialize_branch_src``'s own correctness: the
        resolved source is ALWAYS either ``self._src_root`` (main/absent/already-on-
        branch, where this guard is a no-op or trivially passes) or the materialized
        branch tree (which IS on the requested branch, so the guard passes). It would
        raise only if the ``git_cloner`` returned a tree whose HEAD is NOT the
        requested branch — a materialize regression — NEVER to refuse a legitimately-
        requested feature branch (that refusal moved to ``materialize_branch_src``'s
        fail-loud ``RuntimeError`` on an unresolvable branch, §793/#653).

        ``src_root`` defaults to ``self._src_root`` (backward-compatible: any direct
        call or the reconnect/estimate paths are unchanged). The internal semantics
        are otherwise verbatim from the #653 guard — it still raises ``ValueError`` on
        a genuine HEAD/branch mismatch (``prepare`` runs under
        :func:`router._prepare_and_launch`, which wraps it as a provision-class
        :class:`~router.BackendPrepareError`) and still no-ops when ``repo_branch`` is
        absent or ``main``.

        The branch probe failing (non-repo source, git missing → resolver returns
        ``None``) is treated as a MISMATCH for a non-``main`` request: we cannot prove
        the source carries the feature branch, so we refuse rather than risk silently
        shipping ``main`` (#653). Under ``prepare``'s post-fix flow this sub-branch
        cannot fire on the honored-branch path (the cloner already raised on an
        unresolvable branch), but it is retained so a DIRECT call with a mismatched
        ``src_root`` still refuses.
        """
        src_root = src_root or self._src_root
        requested = str(spec.extra.get("repo_branch") or "").strip()
        if not requested or requested == "main":
            return
        actual = self._git_branch_resolver(src_root)
        if actual == requested:
            return
        raise ValueError(
            f"SLURM lane cannot honor repo_branch={requested!r}: the rsync "
            f"source at {src_root} is on branch {actual!r} "
            f"(the repo-root install resolves to 'main', not the invoking "
            f"worktree). Submitting would rsync stale code whose tree lacks "
            f"the feature branch's entrypoint scripts and crash at in-job "
            f"preflight (#653). Route this run to GCP (`--backend gcp`, which "
            f"git-clones the branch on the VM) or merge the branch into the "
            f"rsync source's HEAD."
        )

    def _push_secrets(self, cluster: ClusterConfig, scratch_dir: str, content: str) -> None:
        """Deliver ``secrets.env`` to ``$SCRATCH_JOB_DIR`` via the injected
        pusher (default :func:`scp_push_secrets`).

        Decoupled so tests can swap a list-appender for the real
        ``scp``/``rsync`` shell-out.
        """
        self._secrets_pusher(
            robot_alias=cluster.ssh_host,
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
        scratch_dir = scratch_dir_for(spec, cluster)
        plan_hash = spec.extra.get("plan_hash")
        # Render via the same helper estimate_start{,_seconds} use, so
        # the --test-only probe script (router ranking hint) and this
        # submit script are byte-identical for the same (spec, cluster).
        script = self._render_script_for(spec, cluster)
        job_id = self._submit(
            robot_alias=cluster.ssh_host,
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
        try:
            self._post_marker(
                issue=spec.issue,
                marker="epm:cluster-launched",
                note=marker_body,
                version=1,
                by="backends.slurm",
            )
        except Exception as exc:
            # Marker post is best-effort AFTER a successful sbatch submit:
            # the SLURM job is already live, and a raise here (e.g.
            # ``post_marker_via_task_py``'s ``subprocess.run(check=True,
            # timeout=30)`` hitting flock contention on
            # ``~/.task-workflow/lock``) would propagate out of launch()
            # with NO handle returned, NO lease written, NO sidecar — a
            # live job with no recovery record (dispatch CLI rc=4).
            # Mirrors GcpBackend.launch's guard. Log LOUD (payload
            # included) so the operator can backfill the marker.
            logger.error(
                "SLURM launch: epm:cluster-launched marker post FAILED for issue=%d "
                "(job_id=%s already submitted): %s; continuing — payload=%s",
                spec.issue,
                job_id,
                exc,
                marker_body,
            )

        # Expected-artifacts declaration (#598): built AFTER _submit
        # returns because the SLURM attempt id IS the job id (GCP mints
        # its attempt id pre-provision; SLURM cannot). Without this the
        # mechanical confirm_artifacts gate is structurally
        # unsatisfiable on the SLURM lane (finalize FAILs "missing
        # declaration" regardless of what the workload produced — the
        # live #588 finding this task closes).
        from explore_persona_space.backends.artifacts import EXPECTED_ARTIFACTS_HANDLE_KEY

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
                # Unix epoch of THIS attempt's submit. The monitor +
                # started-evidence probe gate scratch artifacts on it so
                # a prior attempt's status.json/job.out (same per-issue
                # scratch dir) cannot masquerade as this job's output.
                # Rides the sidecar JSON so the bg-Bash poller sees it
                # across processes.
                "submitted_at": state.submitted_at,
                EXPECTED_ARTIFACTS_HANDLE_KEY: expected_artifacts_declaration(
                    spec=spec, job_id=job_id, src_root=self._src_root
                ),
            },
        )

    def estimate_start(self, spec: RunSpec) -> datetime | None:
        """Informational ``sbatch --test-only`` estimate, never a gate.

        Returns a tz-aware UTC :class:`datetime` (the cluster-local
        timestamp parsed out of ``--test-only`` output, localized via
        the cluster's :attr:`ClusterConfig.timezone`) or ``None`` when
        the estimate is unparseable. The router logs the estimate but
        uses an explicit submit-and-park watchdog for the actual
        park-decision (per the plan's "estimate is a ranking hint
        only" policy).
        """
        cluster = self._cluster_for_spec(spec)
        script = self._render_script_for(spec, cluster)
        return self._start_estimator(
            robot_alias=cluster.ssh_host,
            sbatch_script=script,
            cluster_timezone=cluster.timezone,
        )

    def estimate_start_seconds(
        self,
        spec: RunSpec,
        *,
        now: datetime | None = None,
    ) -> float | None:
        """Seconds-from-``now`` until ``spec`` would start on this cluster.

        Thin wrapper over the module-level :func:`estimate_start_seconds`
        — exposed on the backend so the router can call
        ``backend.estimate_start_seconds(spec)`` without re-deriving the
        cluster. The rendered probe script is byte-identical to what
        ``launch()`` will submit (same ``render_sbatch`` of the same
        ``RunSpec`` + ``ClusterConfig`` + ``plan_hash``), so the
        estimate matches the real request gres / account / time budget
        with no drift.
        """
        cluster = self._cluster_for_spec(spec)
        rendered = self._render_script_for(spec, cluster)
        return estimate_start_seconds(
            spec=spec,
            cluster=cluster,
            now=now,
            start_estimator=self._start_estimator,
            rendered_script=rendered,
        )

    def _render_script_for(self, spec: RunSpec, cluster: ClusterConfig) -> str:
        """Render the sbatch the same way ``launch()`` does.

        Centralized so ``launch()``, ``estimate_start()``, and
        ``estimate_start_seconds()`` all submit byte-identical scripts
        for the same ``(spec, cluster)`` — no chance of one path
        threading a different ``plan_hash`` / scratch path than another.
        """
        scratch_dir = scratch_dir_for(spec, cluster)
        plan = stages_for_spec(spec)
        plan_hash = spec.extra.get("plan_hash")
        return render_sbatch(
            spec=spec,
            cluster=cluster,
            plan=plan,
            scratch_dir=scratch_dir,
            plan_hash=plan_hash,
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
        # Submit time for the monitor's artifact-freshness gate: prefer
        # the handle (rides the sidecar JSON across processes — the
        # bg-Bash poller deserializes a fresh backend instance), fall
        # back to in-process launch state. Reconnect handles may have
        # neither — the gate is then disabled (the job is live and
        # writing fresh artifacts anyway).
        submitted_at = handle.extra.get("submitted_at")
        if submitted_at is None:
            state = self._jobs.get(handle.job_id)
            submitted_at = state.submitted_at if state is not None else None
        return self._poll_fn(
            issue=int(issue),
            job_id=handle.job_id,
            cluster=cluster,
            scratch_dir=handle.scratch_dir,
            log_path=handle.log_path,
            submitted_at=float(submitted_at) if submitted_at is not None else None,
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

        The tail is passed through ``_scrub_secret_tokens`` before
        return: ``job.out`` can carry secret values (the C1 xtrace leak
        class), and base.py advertises this API "for orchestrator
        notifications" — a future caller must not silently re-open the
        leak (round-7 Mn2).
        """
        # Import lazily to avoid the circular at module-load (monitor
        # imports from this module).
        from explore_persona_space.backends.slurm_monitor import (
            _local_state_dir,
            _scrub_secret_tokens,
        )

        local_path = _local_state_dir(handle.job_id) / "job.out"
        if not local_path.exists():
            return ""
        with local_path.open("rb") as fh:
            data = fh.read()
        lines = data.decode("utf-8", errors="replace").splitlines()[-200:]
        return _scrub_secret_tokens("\n".join(lines))

    # ----- teardown --------------------------------------------------------

    def fetch_results(self, handle: RunHandle) -> None:
        """rsync ``eval_results/`` + ``figures/`` back to the VM.

        Mirrors the RunPod ``pod.py sync results`` flow. The cluster
        side writes them under ``$SCRATCH_JOB_DIR/out/{eval_results,
        figures}`` (the workload's existing in-job upload writes to
        the canonical project-relative paths, which here resolve under
        the rsync'd tree at ``$SCRATCH_JOB_DIR``).

        The completion sentinel deliberately lives UNDER the rsynced
        ``eval_results/`` tree (``eval_results/issue_<N>/slurm-<jobid>/
        .completion-sentinel.json`` — #598): ``rsync -a`` carries
        dotfiles with no filename filters, so the same pull that lands
        the eval JSONs lands the sentinel at the LOCAL path the
        launch-time ``expected_artifacts`` declaration names — finalize
        runs this method BEFORE ``confirm_artifacts``, so the default
        local-FS sentinel reader just works.

        Result-push contract: SLURM workloads cannot git-push results (no
        git checkout on ``$SCRATCH``) — see ``.claude/rules/pod-side-reporting.md``
        § "Result-push verification contract (#1205)", SLURM lane bullet.
        """
        cluster = get_cluster_config(handle.cluster) if handle.cluster else None
        if cluster is None:
            raise ValueError(f"SlurmBackend.fetch_results: handle has no cluster ({handle!r})")
        # Pull eval_results/ + figures/ from $SCRATCH_JOB_DIR back to repo root.
        # ``--mkpath`` on the pull direction too (rsync sometimes needs it for
        # the local destination chain).
        local_root = self._src_root
        for subdir in ("eval_results", "figures"):
            src = f"{cluster.ssh_host}:{handle.scratch_dir}/{subdir}/"
            dst = str(local_root / subdir) + "/"
            argv = ["rsync", "-a", "--mkpath", "--partial", src, dst]
            logger.info("rsync pull %s → %s", src, dst)
            proc = subprocess.run(argv, check=False, timeout=300)
            if proc.returncode != 0:
                # Non-fatal by contract (a job that produced no figures —
                # eval-only — is fine), but a SILENT failed pull would
                # masquerade downstream as a misleading "sentinel missing"
                # confirm FAIL — log the real cause loudly (#598).
                logger.warning(
                    "SlurmBackend.fetch_results: rsync pull of %s exited %d — a "
                    "missing local sentinel / eval JSON at confirm time may be "
                    "THIS pull failing, not the workload.",
                    src,
                    proc.returncode,
                )

    def confirm_artifacts(self, handle: RunHandle) -> bool:
        """Backend-agnostic artifact verification.

        Delegates to :func:`backends.artifacts.confirm_artifacts_from_handle`,
        which reads the :class:`~backends.artifacts.ExpectedArtifacts`
        declaration the launch path stuffed onto ``handle.extra`` under
        :data:`~backends.artifacts.EXPECTED_ARTIFACTS_HANDLE_KEY` and
        runs the full check suite (HF Hub data + model repos, WandB run,
        git-tracked figures + eval JSON, completion sentinel).

        The verdict's ``reasons`` are logged on FAIL so the orchestrator's
        ``epm:upload-verify-failed v1`` marker carries the exact gap
        without re-running the helper. A missing declaration is itself a
        FAIL (the launch path is responsible for populating it; silently
        passing a handle that forgot is the silent-loss hole the verifier
        is designed to close).
        """
        # Lazy import to avoid a circular at module-load time if the
        # artifacts module ever grows a dependency back on this module.
        from explore_persona_space.backends.artifacts import confirm_artifacts_from_handle

        verdict = confirm_artifacts_from_handle(handle)
        if not verdict.passed:
            logger.warning(
                "SlurmBackend.confirm_artifacts FAIL for job %s: %s",
                handle.job_id,
                "; ".join(verdict.reasons),
            )
        return verdict.passed

    def teardown(self, handle: RunHandle) -> None:
        """``scancel`` the job; idempotent on a missing/terminated id."""
        cluster = get_cluster_config(handle.cluster) if handle.cluster else None
        if cluster is None:
            raise ValueError(f"SlurmBackend.teardown: handle has no cluster ({handle!r})")
        self._cancel(robot_alias=cluster.ssh_host, job_id=handle.job_id)

    # ----- internal helpers ------------------------------------------------

    def _cluster_for_spec(self, spec: RunSpec) -> ClusterConfig:
        if spec.cluster:
            return get_cluster_config(spec.cluster)
        # NO silent default. The old "pick Nibi" fallback silently
        # submitted the 'mila' lane's sbatch to Nibi (issue 535 live
        # finding: job 15876369 ran on Nibi under rrg-bengioy-ad_gpu
        # while every lane-level label said mila, and the lane PASSed
        # its checklist vacuously). The router threads the lane kind
        # into ``spec.cluster`` via ``_spec_for_lane``; a spec arriving
        # here without one is a routing bug — fail fast.
        raise ValueError(
            f"RunSpec for issue {spec.issue} reached SlurmBackend with no "
            "spec.cluster — the router must thread the lane's cluster name "
            "(_spec_for_lane); refusing the silent nibi default."
        )


def _default_src_root() -> Path:
    """Locate the repo root: walk up until ``pyproject.toml`` is found."""
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "pyproject.toml").exists():
            return parent
    return Path.cwd()


# The subset of ``RSYNC_INCLUDE_PATHS`` that is a WORKING-TREE-ONLY gitlink
# (mode-160000 nested repo, absent from any branch's committed git tree). Verified
# on-VM during #793 planning: ``external/open-instruct`` is the ONLY such entry —
# every other include path (``pyproject.toml``, ``uv.lock``, ``src``, ``scripts``,
# ``configs``, ``tests``, ``data/sft``) is a normal blob/tree (mode 100644/040000)
# present in any branch checkout. ``materialize_branch_src`` overlays these paths
# from the working tree because ``git worktree add`` of a branch commit produces an
# EMPTY ``external/open-instruct/``. Kept as an explicit named constant so a future
# reviewer sees exactly which paths are overlaid and why; a second gitlink added to
# ``RSYNC_INCLUDE_PATHS`` must be added here too (the re-asserted branch guard does
# NOT catch a missing overlay — it checks the branch HEAD ref, not overlay content;
# the ``pyproject.toml`` source-sanity assert is the real backstop).
WORKING_TREE_OVERLAY_PATHS: tuple[str, ...] = ("external/open-instruct",)


def materialize_branch_src(
    *,
    src_root: Path,
    branch: str,
    issue: int,
    overlay_paths: tuple[str, ...] = WORKING_TREE_OVERLAY_PATHS,
    timeout: int = 300,
) -> Path:
    """Materialize a complete rsync source for ``branch`` on the VM; return its path.

    The SLURM lane rsyncs from a local tree rather than git-cloning the requested
    branch on the cluster (the GCP-lane approach), so when ``spec.extra["repo_branch"]``
    names a non-``main`` branch that the repo-root install does not already carry, this
    builds a content-complete checkout of the branch's COMMITTED tree on the orchestrator
    VM and returns its path for :meth:`SlurmBackend.prepare` to rsync from.

    Steps (idempotent — safe to re-run every ``prepare``):

    1. ``scratch = ~/.eps-slurm-src/issue-<issue>`` (env override ``EPS_SLURM_SRC_ROOT``).
    2. Remove any prior scratch worktree
       (``git -C <src_root> worktree remove --force <scratch>``, guarded — a fresh
       scratch has none), then ``rm -rf <scratch>`` (belt-and-suspenders for a
       partially-removed dir), then ``git -C <src_root> worktree prune``.
    3. Resolve the branch commit in ``src_root``'s object DB
       (``git rev-parse --verify <branch>``, then ``origin/<branch>`` as a fallback).
       Fail loud (``RuntimeError``) if neither resolves — no silent fallback to ``main``.
    4. ``git -C <src_root> worktree add --detach <scratch> <commit>`` — detached HEAD at
       the branch commit (no "branch already checked out in the /issue worktree" conflict,
       since the branch lives in ``.claude/worktrees/issue-<N>``), sharing the repo-root
       object DB (no ~GB history copy a fresh ``git clone`` would make).
    5. Overlay each working-tree-only path from ``src_root`` into ``scratch``
       (``rsync -a --delete <src_root>/<p>/ <scratch>/<p>/``, a LOCAL FS copy — distinct
       from the cluster-bound rsync). ``<p>`` is the ``external/open-instruct`` gitlink,
       which ``git worktree add`` materializes EMPTY. A path absent in ``src_root`` logs a
       WARNING and is skipped (a lora-only run has no open-instruct — a missing overlay is
       worth surfacing but not crashing).
    6. Assert ``scratch/pyproject.toml`` exists (mirrors ``build_rsync_command``'s own
       source-sanity assert) — fail loud if the worktree add produced an empty tree.

    The scratch tree is a COMMITTED-only source: unlike today's working-tree rsync, an
    untracked-but-present file in ``src_root``'s working tree (e.g. an uncommitted
    ``scripts/issue658_*.py``) will NOT ship. This is the CORRECT behavior for a
    branch-scoped run — a job dispatched against a branch commit must reach only committed
    code — but is noted here so a future debugger who expects "scratch == old working-tree
    rsync for tracked paths" understands why an uncommitted file is absent. The one
    deliberate exception is ``overlay_paths`` (the gitlink), which is working-tree by
    construction.

    :returns: the scratch :class:`~pathlib.Path` (a content-complete tree on ``branch``).
    :raises RuntimeError: on any git failure or an unresolvable branch (fail-fast, per
        CLAUDE.md — no silent fallback to stale ``main``).
    """
    scratch_root = Path(os.environ.get("EPS_SLURM_SRC_ROOT") or (Path.home() / ".eps-slurm-src"))
    scratch = scratch_root / f"issue-{issue}"

    # Step 2 — remove any prior scratch worktree + dir + prune registrations. All git
    # calls here are guarded (a fresh scratch has no registered worktree; ``worktree
    # remove`` on an absent path exits non-zero) — we do not fail the prepare on the
    # cleanup path, only on the create path below.
    with contextlib.suppress(subprocess.SubprocessError, OSError):
        subprocess.run(
            ["git", "-C", str(src_root), "worktree", "remove", "--force", str(scratch)],
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    with contextlib.suppress(OSError):
        shutil.rmtree(scratch, ignore_errors=True)
    with contextlib.suppress(subprocess.SubprocessError, OSError):
        subprocess.run(
            ["git", "-C", str(src_root), "worktree", "prune"],
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )

    # Step 3 — resolve the branch commit in src_root's object DB (local ref, then
    # origin/<branch> fallback). Fail loud if neither resolves.
    commit: str | None = None
    for ref in (branch, f"origin/{branch}"):
        proc = subprocess.run(
            ["git", "-C", str(src_root), "rev-parse", "--verify", "--quiet", f"{ref}^{{commit}}"],
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
        if proc.returncode == 0 and proc.stdout.strip():
            commit = proc.stdout.strip()
            break
    if commit is None:
        raise RuntimeError(
            f"materialize_branch_src: cannot resolve branch {branch!r} (nor "
            f"origin/{branch!r}) in the object DB at {src_root}. The SLURM lane cannot "
            f"honor this repo_branch — the auto chain advances to the next lane."
        )

    # Step 4 — worktree-add the branch commit at a detached HEAD (shares the object DB;
    # no conflict with the branch checked out in the /issue worktree).
    scratch.parent.mkdir(parents=True, exist_ok=True)
    add = subprocess.run(
        ["git", "-C", str(src_root), "worktree", "add", "--detach", str(scratch), commit],
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )
    if add.returncode != 0:
        raise RuntimeError(
            f"materialize_branch_src: `git worktree add --detach {scratch} {commit}` failed "
            f"(rc={add.returncode}): {add.stderr.strip()}"
        )

    # Step 5 — overlay the working-tree-only gitlink(s) from src_root's working tree.
    for p in overlay_paths:
        overlay_src = src_root / p
        if not overlay_src.exists():
            logger.warning(
                "materialize_branch_src: overlay path %r absent in src_root %s — skipping "
                "(a lora-only run has no open-instruct; a full-FT run needs it)",
                p,
                src_root,
            )
            continue
        overlay_dst = scratch / p
        overlay_dst.parent.mkdir(parents=True, exist_ok=True)
        overlay = subprocess.run(
            ["rsync", "-a", "--delete", f"{overlay_src}/", f"{overlay_dst}/"],
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
        if overlay.returncode != 0:
            raise RuntimeError(
                f"materialize_branch_src: overlaying {p!r} from {overlay_src} into "
                f"{overlay_dst} failed (rc={overlay.returncode}): {overlay.stderr.strip()}"
            )

    # Step 6 — source-sanity assert (mirrors build_rsync_command's own pyproject check).
    if not (scratch / "pyproject.toml").exists():
        raise RuntimeError(
            f"materialize_branch_src: scratch tree {scratch} has no pyproject.toml after "
            f"`git worktree add {commit}` — the branch commit produced an empty/invalid tree."
        )
    logger.info(
        "materialize_branch_src: materialized branch %r (commit %s) for issue %d at %s",
        branch,
        commit,
        issue,
        scratch,
    )
    return scratch


def git_branch_at(src_root: Path) -> str | None:
    """Return the current branch name at ``src_root`` (``None`` if unknown).

    Runs ``git -C <src_root> rev-parse --abbrev-ref HEAD``. A detached
    HEAD yields ``"HEAD"`` (treated as "unknown" by the caller — a
    detached checkout cannot be asserted equal to a named feature branch,
    so the guard refuses rather than guesses). Any git failure (not a
    repo, git missing) returns ``None`` so the caller can decide policy
    instead of crashing on the probe itself.

    This is the rsync-source twin of the GCP lane's ``repo_branch``
    git-clone (``backends/gcp.render_startup_script`` clones the requested
    branch on the VM). The SLURM lane rsyncs from ``src_root`` instead of
    cloning, so the guard in :meth:`SlurmBackend.prepare` reads the
    rsync source's actual HEAD here to detect a feature-branch /
    rsync-source mismatch BEFORE submitting a job onto stale code.
    """
    try:
        proc = subprocess.run(
            ["git", "-C", str(src_root), "rev-parse", "--abbrev-ref", "HEAD"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=True,
            timeout=15,
        )
    except (FileNotFoundError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return None
    branch = proc.stdout.strip()
    if not branch or branch == "HEAD":
        return None
    return branch


# ---------------------------------------------------------------------------
# Re-exports
# ---------------------------------------------------------------------------


__all__ = [
    "CLUSTER_CONFIGS",
    "DEFAULT_MILA_SSH_ALIAS",
    "HEARTBEAT_INTERVAL_SECONDS",
    "PASSTHROUGH_ENV_KEYS",
    "PREFLIGHT_FAIL_MARKER",
    "RSYNC_EXCLUDE_PATTERNS",
    "RSYNC_INCLUDE_PATHS",
    "RUNTIME_ARTIFACT_FILENAMES",
    "SECRET_ENV_KEYS",
    "WORKING_TREE_OVERLAY_PATHS",
    "ClusterConfig",
    "SbatchPlan",
    "SlurmBackend",
    "Stage",
    "WorkloadKind",
    "build_clear_runtime_artifacts_command",
    "build_rsync_command",
    "clear_runtime_artifacts",
    "compute_plan_hash",
    "default_gpus_for_intent",
    "estimate_start_seconds",
    "expected_artifacts_declaration",
    "get_cluster_config",
    "git_branch_at",
    "job_name",
    "materialize_branch_src",
    "mila_socket_alive",
    "parse_job_id",
    "post_marker_via_task_py",
    "render_sbatch",
    "render_secrets_env",
    "scp_push_secrets",
    "scratch_dir_for",
    "sentinel_relpath_for",
    "ssh_estimate_start",
    "ssh_scancel",
    "ssh_submit",
    "stages_for_spec",
    "time_budget_hours",
]
