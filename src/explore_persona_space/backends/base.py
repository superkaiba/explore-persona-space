"""Compute-backend ABC + transport dataclasses.

``ComputeBackend`` is the polymorphic interface every backend implementation
honors. The orchestrator's `/issue` skill calls these in a fixed order:

    backend.prepare(spec)         # one-time setup (env, cache, secrets)
    handle = backend.launch(spec) # submit / provision; returns RunHandle
    est = backend.estimate_start(spec)  # optional informational hint
    while not terminal:
        poll = backend.poll(handle)     # running / stalled / dead / done
        backend.fetch_logs(handle)      # for orchestrator notifications
    backend.fetch_results(handle)       # rsync / scp eval_results back
    backend.confirm_artifacts(handle)   # upload-verifier hook
    backend.teardown(handle)            # cleanup (pod terminate / scratch rm)

The :class:`PollResult` shape is the SAME JSON the orchestrator already
consumes from ``scripts/poll_pipeline.py`` (status ∈ {running, done, gate,
stalled, dead}, plus current_phase / new_milestone / log timing / log
tail). We keep field names + types byte-compatible so the orchestrator's
existing JSON parsing keeps working when the SLURM backend lands.

``RunSpec`` describes the work to run; ``RunHandle`` is the opaque token
returned by ``launch()`` and re-passed to every other call. Both are
frozen dataclasses (no in-place mutation — backends produce a NEW handle
when state changes).

Foundation slice: this module ships the ABC + dataclasses. Concrete
backends (`RunPodBackend`, `SlurmBackend`) live in sibling modules.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Literal

# ---------------------------------------------------------------------------
# Backend kind / cluster name typing
# ---------------------------------------------------------------------------

# ``BackendKind`` is the value of the task's ``backend:`` frontmatter. The
# selector resolves this to a concrete :class:`ComputeBackend` instance.
# ``cluster`` is the generic SLURM dispatch (per-cluster routing is done
# inside the SLURM backend); ``nibi`` / ``fir`` / ``mila`` are per-cluster
# aliases the selector accepts and maps onto ``cluster``. ``gcp``
# provisions an ephemeral GCE VM (intent → machine-type map inside
# :class:`~backends.gcp.GcpBackend`) and is the auto-fallback target when
# every free academic cluster fails the 10-minute park (router slice 5).
# ``auto`` is the router's sentinel meaning "no explicit override; rank
# the free lanes by est-start and escalate to GCP on park-fail" — it
# never names a backend instance, only a routing INTENT. ``auto`` is also
# the :class:`RunSpec` default so that any direct ``RunSpec(issue, intent)``
# construction routes through the cost-safe auto chain. The legacy
# selector (:mod:`backends.selector`) preserves the pre-router default
# (frontmatter ``backend:`` missing → ``"runpod"``) INDEPENDENTLY via
# :func:`backends.selector._parse_backend_kind`; a caller that wants the
# legacy RunPod default from a direct ``RunSpec()`` must set
# ``backend="runpod"`` explicitly.
BackendKind = Literal["runpod", "cluster", "nibi", "fir", "gcp", "mila", "auto"]


# ---------------------------------------------------------------------------
# RunSpec — what to run
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RunSpec:
    """A single backend run request.

    The orchestrator builds one of these per ``/issue`` launch from the
    task body + plan; backends consume it and produce the launch artifacts
    they own (RunPod = a provisioned pod + a bg ``train.py`` process;
    SLURM = a rendered sbatch + a submitted job id).

    Fields:

    * ``issue``: task id (e.g. ``137``). Drives pod naming, scratch dir,
      log paths, marker posts.
    * ``intent``: workload intent (e.g. ``lora-7b``, ``ft-7b``, ``eval``).
      The RunPod backend already maps these to GPU specs via
      ``gpu_heuristics.resolve_intent``; the SLURM backend uses the same
      key to pick GPUs/node, ``--time`` budget, and the routing between
      LoRA/eval (``scripts/train.py``+``scripts/eval.py``) vs full-FT
      (open-instruct ``finetune.py``/``dpo_tune_cache.py``).
    * ``gpus``: requested GPU count (overrides the intent default; e.g.
      ``4`` for a 4xH100 full-FT). ``None`` = use the intent default.
    * ``time_budget_hours``: wall-clock budget (``--time`` on SLURM,
      informational on RunPod). ``None`` = backend default.
    * ``account``: SLURM account string (``rrg-bengioy-ad_gpu`` for the
      DRAC robot). Ignored by RunPod.
    * ``hydra_args``: Hydra overrides for the experiment entrypoint
      (e.g. ``["condition=c1_evil_wrong_em", "seed=42"]``). Backends
      thread these into the command they actually launch.
    * ``backend``: which backend should run this — the selector sets this
      from the task frontmatter, the backend itself does NOT re-read it.
    * ``cluster``: when ``backend in {cluster, nibi, fir}``, which cluster
      to submit to. ``None`` defaults to the SLURM backend's primary
      (Nibi in v1).
    * ``extra``: backend-specific knobs the orchestrator wants to thread
      without bloating the schema (e.g. ``per_pod_quota_gb`` override).
    * ``workload_cmd``: custom workload command (repo-relative shell
      line, e.g. ``bash scripts/issue588_dispatch.sh --foo``). Mutually
      exclusive with ``hydra_args``. Executed verbatim by the lane
      renderers from the repo checkout root after env bootstrap.
      ``""`` = use the standard Hydra entrypoint (#588).

    Frozen by design: a run spec is the contract for the launch; mutating
    it mid-run would break the marker trail and any auditable replay.
    """

    issue: int
    intent: str
    gpus: int | None = None
    time_budget_hours: float | None = None
    account: str | None = None
    hydra_args: tuple[str, ...] = ()
    # NOTE: default is ``"auto"`` so a bare ``RunSpec(issue, intent)`` routes
    # via the cost-safe auto chain (free lanes → GCP). A real-money RunPod
    # launch requires an explicit ``backend="runpod"``. The legacy
    # frontmatter→selector path (:mod:`backends.selector`) defaults to
    # RunPod INDEPENDENTLY for back-compat.
    backend: BackendKind = "auto"
    cluster: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)
    # Custom workload command (repo-relative shell line, e.g.
    # "bash scripts/issue588_dispatch.sh --foo"). Mutually exclusive with
    # hydra_args. Executed verbatim by the lane renderers from the repo
    # checkout root after env bootstrap. "" = use the hydra entrypoint.
    # Declared LAST so existing positional constructions are unaffected.
    workload_cmd: str = ""

    def __post_init__(self) -> None:
        """Validate the workload_cmd contract (#588).

        Both-set is a contradiction on EVERY lane → raise here
        (universal). Neither-set stays LEGAL at construction — the
        router suite + est-start/reconnect probes build bare specs that
        never render a workload; the production fail-loud lives at the
        dispatch CLI (exactly-one check) and the GCP renderer
        (neither-set raise, the #571 crash point).
        """
        if self.workload_cmd and self.hydra_args:
            raise ValueError(
                "RunSpec: workload_cmd and hydra_args are mutually exclusive "
                f"(got workload_cmd={self.workload_cmd!r} AND hydra_args={self.hydra_args!r})."
            )
        if self.workload_cmd:
            if "\n" in self.workload_cmd or "\r" in self.workload_cmd:
                raise ValueError("RunSpec.workload_cmd must be a single line (no newlines).")
            if self.workload_cmd != self.workload_cmd.strip():
                raise ValueError("RunSpec.workload_cmd must not have leading/trailing whitespace.")


# ---------------------------------------------------------------------------
# RunHandle — what launch() returns
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RunHandle:
    """Opaque handle returned by :meth:`ComputeBackend.launch`.

    Every subsequent backend call (``poll``, ``fetch_logs``,
    ``fetch_results``, ``confirm_artifacts``, ``teardown``) takes one of
    these. Backends thread their own identifying state through ``extra``;
    the typed fields below cover the orchestrator's common needs.

    Fields:

    * ``backend``: which backend issued this handle. The orchestrator's
      upload-verifier dispatches on this (RunPod path expects a pod, the
      SLURM path expects a job id).
    * ``cluster``: the concrete cluster for SLURM handles (``nibi``,
      ``fir``); ``None`` for RunPod.
    * ``job_id``: SLURM job id (numeric string) or RunPod pod_id. Always
      a string so the orchestrator's marker post is uniform.
    * ``pod_name``: canonical ``pod-<N>`` (RunPod) or job name keyed by
      ``issue + plan hash`` (SLURM). The marker schema reads this.
    * ``scratch_dir``: backend-side working directory. ``$SCRATCH/eps/
      issue-<N>`` on SLURM; ``/workspace/`` on RunPod.
    * ``log_path``: where the job's stdout/stderr lands on the backend.
      ``/workspace/logs/issue-<N>.log`` on RunPod; ``$SCRATCH/eps/issue
      -<N>/job.out`` on SLURM (SLURM ``--output`` target).
    * ``extra``: backend-private state (e.g. RunPod ``ssh_host``,
      ``ssh_port``; SLURM ``account``, ``partition``).
    """

    backend: BackendKind
    cluster: str | None
    job_id: str
    pod_name: str
    scratch_dir: str
    log_path: str
    extra: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# BackendProbeError — typed "probe FAILED, state UNKNOWN" signal
# ---------------------------------------------------------------------------


class BackendProbeError(RuntimeError):
    """A backend state probe FAILED — the job/instance state is UNKNOWN.

    Raised by backend-aware probes (``squeue`` / ``scontrol`` over the
    robot SSH alias) when the PROBE ITSELF failed (SSH transport
    refused, forced-command wrapper rejection, timeout) — as opposed to
    the probe succeeding and showing the job absent.

    The distinction is load-bearing (issue 535 attempt 2): a transient
    SSH failure that reads as "job gone" lets the router classify a
    LIVE job as terminal and orphan it. Callers must treat this error
    as UNKNOWN-retry (bounded by a consecutive-failure budget), NEVER
    as job-absent / terminal.
    """


# ---------------------------------------------------------------------------
# PollResult — same shape as scripts/poll_pipeline.py::PollResult
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PollResult:
    """One-tick poll status; shape matches ``scripts/poll_pipeline.py``.

    The orchestrator's `/issue` skill consumes a JSON line shaped like
    this one (see ``poll_pipeline.main`` and the ``PollResult`` dataclass
    in ``scripts/poll_pipeline.py``). Keeping the field names + types
    identical means a backend's ``poll()`` return value serializes to
    the SAME JSON the existing poller produces, so the orchestrator's
    JSON-line parser keeps working unchanged across backends.

    Status enum (``status`` field) is exactly the existing set:

    * ``running`` — the workload is alive and making progress.
    * ``done`` — the workload reached ``[phase=done]`` cleanly.
    * ``gate`` — a sentinel asked the orchestrator to park at a user gate.
    * ``stalled`` — alive but no log progress AND idle GPUs for >STALL_SEC.
    * ``dead`` — the launching PID / SLURM state says the workload exited
      without a clean ``done``.

    Backend-specific notes:

    * RunPod backend (slice 1): delegates to ``poll_pipeline.poll_once``
      and rewraps the result. Fields map 1:1.
    * SLURM backend (slice 2): builds the result from
      ``{scontrol/squeue state, rsync'd status.json heartbeat, rsync'd
      job.out tail}``. ``pid_alive`` is a synthetic ``True`` while
      SLURM reports ``RUNNING`` (there is no SSH-side PID to check);
      ``gpu_util`` is parsed from the in-job ``nvidia-smi`` writes
      that the sbatch persists into ``status.json``.

    Defaults mirror ``poll_pipeline.PollResult`` so a backend that
    doesn't populate a field still serializes to the SAME JSON shape.
    """

    status: str  # running | done | gate | stalled | dead
    current_phase: str
    new_milestone: bool
    last_log_mtime_sec_ago: int
    pid_alive: bool
    log_tail_excerpt: str
    gate: str | None = None
    sentinels_processed: int = 0
    phase_log_mtime_sec_ago: int = 10**9
    shard_log_mtime_sec_ago: int = 10**9
    gpu_util: str = "unknown"


# ---------------------------------------------------------------------------
# ComputeBackend — the ABC
# ---------------------------------------------------------------------------


class ComputeBackend(ABC):
    """Backend interface for `/issue` runs.

    All methods are blocking with timeouts; the orchestrator runs them
    from its bg-Bash loop (poll-and-exit pattern — see
    ``scripts/poll_pipeline.py`` module docstring and
    ``.claude/skills/issue/SKILL.md`` Step 6d.2).

    Implementations MUST be safe under re-entry: each method should
    idempotently observe the live backend state and bring local
    state into agreement (e.g. ``poll()`` on a finished job returns
    ``done``; ``teardown()`` on an already-terminated pod is a no-op).
    """

    # ----- identity --------------------------------------------------------

    @property
    @abstractmethod
    def name(self) -> BackendKind:
        """Backend kind (matches ``RunHandle.backend``)."""

    # ----- launch ----------------------------------------------------------

    @abstractmethod
    def prepare(self, spec: RunSpec) -> None:
        """One-time setup BEFORE launch (env, cache warm, secret sync).

        Idempotent — multiple calls with the same spec are safe.
        """

    @abstractmethod
    def launch(self, spec: RunSpec) -> RunHandle:
        """Submit / provision the workload. Returns a :class:`RunHandle`.

        Failure modes are backend-specific and propagated as exceptions;
        the selector catches ``NotImplementedError`` (slice 1 stub) and
        falls back to the next backend in its decision table.
        """

    @abstractmethod
    def estimate_start(self, spec: RunSpec) -> datetime | None:
        """Best-effort wall-clock estimate of WHEN the job starts running.

        Informational hint only — the selector logs it but does NOT gate
        on it (the plan's submit-and-park policy uses an explicit
        ``max_wait`` watchdog, not this estimate). Returns ``None`` when
        no estimate is available.
        """

    # ----- monitor ---------------------------------------------------------

    @abstractmethod
    def poll(self, handle: RunHandle) -> PollResult:
        """One-tick poll. Same JSON shape as ``poll_pipeline.poll_once``."""

    @abstractmethod
    def fetch_logs(self, handle: RunHandle) -> str:
        """Return the latest log tail (for orchestrator progress notes)."""

    # ----- teardown --------------------------------------------------------

    @abstractmethod
    def fetch_results(self, handle: RunHandle) -> None:
        """Pull eval_results/ + figures/ back to the VM."""

    @abstractmethod
    def confirm_artifacts(self, handle: RunHandle) -> bool:
        """Verify uploads landed (HF/WandB/git) per the Upload Policy.

        Returns True on PASS. The orchestrator's existing upload-verifier
        flow consumes this; both backends defer the substantive checks
        to ``scripts/verify_uploads.py`` and similar.
        """

    @abstractmethod
    def teardown(self, handle: RunHandle) -> None:
        """Terminate / clean up. Idempotent."""
