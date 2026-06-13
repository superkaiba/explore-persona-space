"""Central multi-backend compute router (slice 5 of the compute-router plan).

This module is the canonical replacement for
:func:`backends.selector.select_backend`'s submit-and-park flow. Where the
selector dispatches on a single ``backend:`` frontmatter and falls back to
RunPod-on-error, ``route(spec)`` orchestrates the full multi-backend ladder:

1. **Explicit override** — ``spec.backend == "runpod" | "gcp" | "nibi" |
   "fir" | "mila"`` runs that lane directly. RunPod is reachable ONLY via
   the override; the auto chain never spends real money.
2. **Auto** — walk the resolved auto lane order. The STANDING DEFAULT is
   **GCP first** (:data:`DEFAULT_AUTO_LANE_ORDER` =
   ``("gcp", "nibi", "fir", "mila")``): credits-backed GCP capacity is
   consumed BEFORE the free SLURM lanes. The order is overridable via the
   ``EPM_AUTO_LANE_ORDER`` env var (comma-separated lanes, validated —
   ``runpod`` and unknown names raise loudly) or per-call via
   :attr:`RouterConfig.lane_order`; there is deliberately NO date logic —
   flipping the order back is a human action (env override or a default
   edit), never a clock. Contiguous SLURM lanes in the order are ranked
   among themselves by tz-corrected ``estimate_start_seconds`` (a ranking
   HINT, never a gate), the best is submitted and parked up to
   ``FREE_WAIT`` (default 600 s) to reach RUNNING; PENDING-at-cap triggers
   cancel + the next lane. GCP has no queue estimate and no park — its
   "park" is the provision call itself.
3. **Cancel state machine** — request a cancel via the backend's
   ``teardown(handle)``, then poll via the injected ``is_live_after_cancel``
   callable until the job is no longer live in the cluster queue
   (DRAC robot allowlist has no ``sacct``; we cannot confirm terminal
   CANCELLED). A job that RACED to RUNNING during cancel is KEPT (it has
   started; tearing it down would burn the wait we already paid for). A
   timeout produces a ``manual-attention`` outcome rather than a silent
   leak.
4. **Fallback chain — within the resolved order, NEVER RunPod.** A
   provision-class failure on any lane (free-lane PENDING-at-cap /
   provisioning failure; GCP provisioning / capacity / prepare / state-
   probe failure when lanes remain after it) continues DOWN the resolved
   order. Under the GCP-first default that means GCP capacity failures
   fall through to the SLURM lanes; under a free-first override the SLURM
   park-failures escalate to GCP exactly as before. NEVER RunPod on auto
   — RunPod stays override-only regardless of the configured order.
5. **Failure classification** — :class:`gcp.GcpProvisioningError` (and
   any backend-marked ``provisioning_failure: True`` raise) routes to the
   next tier; :class:`gcp.GcpWorkloadError` surfaces, NO auto-fallback;
   "every free lane park-failed AND GCP capacity-failed" raises
   :class:`NoComputeAvailableError` for the orchestrator to translate
   into ``epm:failure (failure_class: infra) + status:blocked``. A
   ``backend.prepare()`` failure (rsync/secrets push) is provision-class
   too — :class:`BackendPrepareError`: next tier on auto, typed terminal
   on an explicit override. A ``terminal_before_running`` park outcome
   is probed via the injected ``started_evidence_probe`` (scratch-dir
   ``status.json`` / ``job.out`` read): runtime artifacts present means
   the job STARTED and fast-failed — a WORKLOAD failure
   (:class:`WorkloadSurfacedError`, NO GCP escalation), not no-compute.
   Every fresh launch goes through :func:`_prepare_and_launch`
   (``prepare`` → ``launch``); reconnect paths never re-``prepare`` (the
   SLURM prepare rsyncs with ``--delete`` under a possibly-RUNNING job).
6. **Durable lease + reconnect** — a flock'd JSON lease at
   ``~/.eps-routing/issue-<N>.json`` (outside the worktree — the 09:47
   cron reaps worktrees, so a lease there would silently disappear) is
   keyed by a canonicalized spec hash + attempt id. The flock is
   per-issue (``<lease_dir>/issue-<N>.lock``), NOT shared across the
   directory, so a 10-min park on issue 137 inside
   ``store.transaction(137)`` does NOT block a ``route()`` on issue
   200. Before any submit / provision, ``route()`` reconnects to an
   existing live job (SLURM ``squeue --name eps-issue-<N>``; GCE
   ``reconnect_or_none``) via the injected backend so a re-driving
   ``issue-tick`` cron does NOT double-submit. The external
   job/instance id is persisted IMMEDIATELY after submit so an
   orchestrator crash between submit and lease-write leaves an
   ``UNKNOWN_SUBMITTED`` recovery state.
7. **GCP attempt-count guard** — a per-issue/day attempt counter caps
   auto-chain GCP attempts at ``MAX_GCP_ATTEMPTS_PER_DAY`` (default 5).
   Primary-lane attempts (GCP-first default) count the SAME as
   escalation attempts — the guard exists to stop a broken-classifier
   loop from burning instances, and primary-lane attempts carry the same
   risk. At the cap: when lanes REMAIN after GCP the router skips GCP
   (zero credit spent) and continues down the order; when GCP is the
   LAST lane it raises :class:`GcpAttemptCapExceededError` (the legacy
   escalation semantics). This is NOT a dollar cap
   (``tests/test_no_dollar_budget_caps.py`` enforces "no SystemExit on
   budget" — see plan §"Real-money safety"); it bounds the *number of
   provision attempts* so a broken classifier that loops can't burn the
   GFS credit unattended.
8. **Markers** — extends the existing ``epm:backend-selected v1`` body
   (per-lane est-starts raw+clamped, chosen lane, fallback chain,
   canonical reason codes, ids). The orchestrator's marker poster is
   injected; tests pass a list-appender. NEVER hardcodes a
   ``task.py`` shell-out — slice 5 is router-only, slice 6 wires the
   real poster.

The wiring into ``/issue`` lives in slice 6. This module is fully
testable without RunPod / SLURM / GCP being live.

Authoritative companion docs:

* Plan: ``.claude/plans/2026-06-08_224537-multi-backend-compute-router.md``
* Markers: ``.claude/workflow.yaml § markers`` (``epm:backend-selected``,
  ``epm:cluster-launched``, ``epm:cluster-poll``, ``epm:cluster-terminal``)
* Halt criterion: ``CLAUDE.md § Halt-criterion contract`` — a no-compute
  outcome is the canonical "infrastructure exhaustion" block (#1).
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import logging
import os
import time
from collections.abc import Callable, Iterable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from explore_persona_space.backends.base import (
    BackendKind,
    BackendProbeError,
    ComputeBackend,
    PollResult,
    RunHandle,
    RunSpec,
)
from explore_persona_space.backends.gcp import (
    GcpProvisioningError,
    GcpWorkloadError,
    QuotaHeadroom,
    machine_for_intent,
    quota_metric_for,
    resolve_provisioning_model,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public constants
# ---------------------------------------------------------------------------

#: Always-on 10-minute park cap on every free-lane submit (per plan §
#: "Mid-review" — supersedes the 6 h ``DEFAULT_MAX_WAIT_SECONDS`` from
#: :mod:`selector` and the ``EPM_CLUSTER_MAX_WAIT_SECONDS`` env knob).
FREE_WAIT_SECONDS: int = 600

#: Default poll interval inside the park watchdog. The SLURM scheduler
#: state updates on multi-second cycles; faster polling burns ssh round
#: trips without speeding the result. Tests inject smaller values.
DEFAULT_POLL_INTERVAL: float = 5.0

#: Per-issue/day cap on auto-escalation to GCP. NOT a dollar cap (see
#: ``tests/test_no_dollar_budget_caps.py``); this counts ATTEMPTS so a
#: broken classifier cannot loop into credit burn. Tunable per call.
MAX_GCP_ATTEMPTS_PER_DAY: int = 5

#: Cancel state-machine: how long to keep polling for the job to leave
#: the live queue after ``scancel``. SLURM robots have no ``sacct`` so
#: we cannot confirm terminal CANCELLED — only that the job is no
#: longer live. A long-running run that won't die after this cap drops
#: into ``manual-attention``.
CANCEL_LIVE_GRACE_SECONDS: int = 60

#: Lease store directory — OUTSIDE the worktree by deliberate design.
#: The 09:47 ``worktree_audit.py`` cron reaps idle worktrees under
#: ``.claude/worktrees/``; a lease there would silently disappear and
#: the next ``/issue`` invocation would double-submit. ``~/.eps-routing/``
#: lives in HOME and is owned by the orchestrator user.
LEASE_STORE_DIRNAME: str = ".eps-routing"

#: Canonical reason codes the router emits in the marker. The selector's
#: legacy codes (``frontmatter_default``, ``slurm_not_implemented``)
#: stay in :mod:`selector`; this set is router-specific.
ROUTE_REASON_OVERRIDE: str = "override"
ROUTE_REASON_RECONNECT: str = "reconnect"
ROUTE_REASON_AUTO_STARTED: str = "auto_started"
ROUTE_REASON_AUTO_FALLBACK_GCP: str = "auto_fallback_gcp"
ROUTE_REASON_NO_COMPUTE: str = "no_compute_available"
ROUTE_REASON_WORKLOAD_FAILURE: str = "workload_failure"
#: ``backend.prepare`` failed pre-launch (rsync / secrets push). Matches
#: the ``reason: backend_prepare_failed`` line the dispatch CLI's
#: ``classify_terminal_exception`` emits for :class:`BackendPrepareError`
#: — pre-fix the breadcrumb said ``no_compute_available`` while the
#: typed terminal said ``backend_prepare_failed`` (round-6 Mn1).
ROUTE_REASON_PREPARE_FAILED: str = "backend_prepare_failed"

#: Consecutive ``is_started`` probe failures tolerated inside the park
#: watchdog before it gives up with ``probe_failures_exceeded``.
#: Mirrors ``scripts/router_acceptance.py``'s
#: ``_POLL_MAX_CONSECUTIVE_FAILURES`` (same value, same reset-on-success
#: semantics). A probe failure means the job state is UNKNOWN — it must
#: NEVER read as "not started yet" indefinitely (round-6 B1): the
#: watchdog hands the lane to the cancel state machine, whose own
#: probe-failure handling (treat-as-still-live + grace) resolves to
#: ``manual_attention`` while the transport stays broken.
PARK_MAX_CONSECUTIVE_PROBE_FAILURES: int = 3

#: SLURM free-lane subset (DRAC + Mila), in legacy precedence order.
#: Kept as a public constant for callers that need "the free lanes";
#: the AUTO chain's order is :data:`DEFAULT_AUTO_LANE_ORDER` /
#: :func:`auto_lane_order`. RunPod is NEVER in either list — it's
#: override-only by deliberate design.
DEFAULT_FREE_LANE_ORDER: tuple[BackendKind, ...] = ("nibi", "fir", "mila")

#: Standing default auto lane order: **GCP first** (credits-backed GCP
#: capacity is consumed BEFORE the free SLURM lanes), then the SLURM
#: lanes in legacy precedence. This is an unconditional default — NO
#: date logic; flipping back to free-first later is a deliberate human
#: action (set :data:`ENV_AUTO_LANE_ORDER` or edit this default), never
#: a clock.
DEFAULT_AUTO_LANE_ORDER: tuple[BackendKind, ...] = ("gcp", *DEFAULT_FREE_LANE_ORDER)

#: Env override for the auto lane order — comma-separated lane names,
#: e.g. ``EPM_AUTO_LANE_ORDER=nibi,fir,mila,gcp`` to restore free-first.
#: Validated by :func:`auto_lane_order`: ``runpod`` raises loudly
#: (real-money safety — never silently dropped), as do unknown names,
#: ``auto``/``cluster`` literals, and duplicates.
ENV_AUTO_LANE_ORDER: str = "EPM_AUTO_LANE_ORDER"

#: Every value the ROUTER accepts for ``spec.backend``. ``route()``
#: rejects anything outside this set at entry (closes the empty-string
#: / stringly-typed-miswire silent-auto-route hole). Narrower than
#: :data:`BackendKind` (``base.py``) by deliberate design: the legacy
#: ``"cluster"`` literal lives in the selector surface (``selector.py``)
#: and is NOT a routable backend at the slice-5 router level — a caller
#: that wants a free-cluster lane must name it (``"nibi"`` / ``"fir"``)
#: or leave ``backend`` unset to auto-route. Passing ``"cluster"`` here
#: is treated as a stringly-typed miswire.
_VALID_BACKEND_VALUES: frozenset[str] = frozenset({"runpod", "nibi", "fir", "gcp", "mila", "auto"})

#: Lanes the AUTO chain may contain — :data:`_VALID_BACKEND_VALUES`
#: minus ``runpod`` (override-only; real money) and ``auto`` (the
#: sentinel itself, not a lane).
_AUTO_LANE_VALUES: frozenset[str] = frozenset({"gcp", "nibi", "fir", "mila"})

#: Lanes whose kind IS a SLURM cluster name. The shared ``SlurmBackend``
#: resolves its target cluster from ``spec.cluster`` per call, so every
#: router site that touches one of these lanes MUST thread the lane kind
#: into ``spec.cluster`` via :func:`_spec_for_lane` first.
_PER_CLUSTER_LANES: frozenset[str] = frozenset({"nibi", "fir", "mila"})


# ---------------------------------------------------------------------------
# Public outcome / error types
# ---------------------------------------------------------------------------


class RouteError(RuntimeError):
    """Base class for router-terminal errors."""


class NoComputeAvailableError(RouteError):
    """Terminal: every free lane park-failed AND GCP capacity-failed.

    The orchestrator translates this into
    ``epm:failure (failure_class: infra) + status:blocked`` (the only
    autonomous-mode infra exhaustion block per CLAUDE.md § halt
    criterion #1's "fact only the user knows" — except this is "fact
    nobody knows: there is no compute right now").
    """

    def __init__(
        self,
        reason: str,
        *,
        attempts: list[dict[str, Any]] | None = None,
    ) -> None:
        super().__init__(reason)
        self.reason = reason
        self.attempts = list(attempts or [])


class BackendPrepareError(RouteError):
    """``backend.prepare(spec)`` failed BEFORE launch (provision-class).

    Nothing is live when this raises — ``prepare`` runs strictly before
    any submit/provision inside :func:`_prepare_and_launch`, so the
    failure carries normal provision-failure semantics: next tier on
    the auto chain, typed terminal on an explicit override (the
    dispatch CLI's ``classify_terminal_exception`` translates it to
    ``epm:failure (failure_class: infra)``). Wraps the underlying
    exception (rsync/scp non-zero exit, SSH refusal) via
    ``raise ... from exc``.
    """

    def __init__(
        self,
        reason: str,
        *,
        kind: BackendKind,
        cluster: str | None = None,
    ) -> None:
        super().__init__(reason)
        self.reason = reason
        self.kind = kind
        self.cluster = cluster


class WorkloadSurfacedError(RouteError):
    """A backend reported a WORKLOAD failure (not provisioning).

    The router does NOT auto-fallback on this — a deterministic
    workload bug would just re-crash on the next tier. The orchestrator
    posts ``epm:failure (failure_class: code)`` and parks.
    """

    def __init__(
        self,
        reason: str,
        *,
        chosen_kind: BackendKind,
        evidence: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(reason)
        self.reason = reason
        self.chosen_kind = chosen_kind
        self.evidence = dict(evidence or {})


class GcpAttemptCapExceededError(RouteError):
    """Per-issue/day GCP attempt-count guard tripped.

    The router refuses to escalate to GCP after
    :data:`MAX_GCP_ATTEMPTS_PER_DAY` attempts in the same UTC day for the
    same issue. The orchestrator surfaces this as an infra block (a
    looping classifier is in scope #1 of the halt criteria — "fact only
    the user knows: should I keep trying").
    """

    def __init__(self, *, issue: int, attempts_today: int, cap: int) -> None:
        super().__init__(
            f"GCP auto-escalation cap of {cap} attempts reached for issue {issue} "
            f"today (attempts_today={attempts_today}); refusing to escalate further. "
            "Lease counter resets at midnight UTC."
        )
        self.issue = issue
        self.attempts_today = attempts_today
        self.cap = cap


class ManualAttentionRequiredError(RouteError):
    """The cancel state machine timed out without confirming the job was dead.

    The router issued ``scancel``/``teardown`` but the job remained live in
    the cluster queue after :data:`CANCEL_LIVE_GRACE_SECONDS`. We CANNOT
    silently escalate to GCP: the free-lane job MAY still be alive, and a
    GCP escalation would launch a second copy under the same attempt-id
    namespace (artifact collision + double spend). The orchestrator surfaces
    this as an infra block with the orphaned job id so the operator can
    confirm + manually ``scancel``. The lease is left intact for the
    orchestrator to consult and the cluster job ``--time`` budget will
    eventually reap it on its own.
    """

    def __init__(
        self,
        *,
        kind: BackendKind,
        cluster: str | None,
        orphaned_job_id: str,
        attempts: list[dict[str, Any]] | None = None,
    ) -> None:
        super().__init__(
            f"cancel grace expired without confirming termination of "
            f"{kind}/{cluster or 'no-cluster'} job {orphaned_job_id!r}; "
            "refusing to escalate (would risk duplicate run). Operator: "
            f"verify job state, manually scancel if alive."
        )
        self.kind = kind
        self.cluster = cluster
        self.orphaned_job_id = orphaned_job_id
        self.attempts = list(attempts or [])


# ---------------------------------------------------------------------------
# RouteResult — what the router returns on success
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RouteAttempt:
    """One per-tier attempt the router made.

    Recorded in :attr:`RouteResult.attempts` (and in the marker body) so
    the operator can see the full ladder: which lanes were tried, how
    each one resolved, and why the final lane was chosen.
    """

    kind: BackendKind
    cluster: str | None
    est_start_seconds_raw: float | None
    est_start_seconds_clamped: float | None
    outcome: str
    detail: str = ""
    elapsed_seconds: float = 0.0


@dataclass(frozen=True)
class RouteResult:
    """Outcome of a successful :func:`route` call.

    On terminal failure the router RAISES (:class:`NoComputeAvailableError`,
    :class:`WorkloadSurfacedError`, :class:`GcpAttemptCapExceededError`)
    rather than returning a result whose ``handle`` is None — the caller
    should never have to defensively check whether a result is "real".
    """

    backend: ComputeBackend
    handle: RunHandle
    requested_kind: BackendKind | None
    chosen_kind: BackendKind
    reason: str
    cluster: str | None
    attempts: list[RouteAttempt]
    elapsed_seconds: float
    extra: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Per-lane spec threading
# ---------------------------------------------------------------------------


def _spec_for_lane(spec: RunSpec, kind: BackendKind) -> RunSpec:
    """Thread the lane's cluster name into ``spec.cluster`` for per-cluster lanes.

    The shared ``SlurmBackend`` instance serves every SLURM lane and
    resolves its target cluster from ``spec.cluster`` on each call.
    Nothing upstream set ``spec.cluster`` for explicit per-cluster lane
    overrides or auto-chain lane attempts, so the backend's defensive
    nibi default silently submitted EVERY lane to Nibi — live finding,
    issue 535: the 'mila' lane's sbatch landed on Nibi (job 15876369,
    account ``rrg-bengioy-ad_gpu``, ``/scratch/tjiral`` scratch) while
    every lane-level label (routing marker, HF subfolder, figure) said
    mila, and the lane PASSed its checklist vacuously. Non-cluster lanes
    (gcp / runpod) pass through unchanged; a contradicting explicit
    ``spec.cluster`` raises instead of guessing.
    """
    if kind not in _PER_CLUSTER_LANES:
        return spec
    if spec.cluster is None:
        return replace(spec, cluster=kind)
    if spec.cluster != kind:
        raise RouteError(
            f"spec.cluster={spec.cluster!r} contradicts the {kind!r} lane — refusing to launch"
        )
    return spec


# ---------------------------------------------------------------------------
# Spec canonicalization (stable hash for lease keying)
# ---------------------------------------------------------------------------


def canonicalize_spec(spec: RunSpec) -> dict[str, Any]:
    """Return a JSON-canonical dict representation of ``spec``.

    Two specs that produce the same workload (same issue, intent, gpus,
    hydra args, account, time budget, backend, cluster) MUST hash to the
    same key — harmless serialization diffs (dict ordering inside
    ``extra``, integer vs float wall-time) MUST NOT change the key.
    The lease reconnect path uses this hash to decide whether a stored
    lease applies to the current request; a flaky hash would silently
    miss live runs and double-submit.

    We canonicalize by:

    1. Sorting every dict (``extra``, nested dicts) at output time via
       ``json.dumps(..., sort_keys=True)`` — done at the call site that
       hashes the dict.
    2. Casting ``hydra_args`` to a tuple of strings (already frozen on
       :class:`RunSpec`, but defensively re-tuple in case a caller
       mutated).
    3. Coercing ``time_budget_hours`` to a normalized float string so
       ``6`` and ``6.0`` hash identically.
    4. Dropping ``extra`` keys the lease system itself sets
       (``attempt_id`` — recorded SEPARATELY in the lease, not in the
       spec-hash; ``startup_script_path`` — tempfile path that varies
       per launch; ``provisioning_model`` — included since it changes
       intent for re-attempts to be a different request shape; we keep
       it).

    Returns a dict; the caller hashes via
    ``hashlib.sha256(json.dumps(d, sort_keys=True).encode()).hexdigest()``.
    """
    extra_filtered = {
        k: v
        for k, v in (spec.extra or {}).items()
        if k not in {"attempt_id", "startup_script_path"}
    }
    # Stringify floats so 6 vs 6.0 collide.
    time_budget = (
        f"{float(spec.time_budget_hours):.6f}" if spec.time_budget_hours is not None else None
    )
    canonical: dict[str, Any] = {
        "issue": int(spec.issue),
        "intent": str(spec.intent),
        "gpus": None if spec.gpus is None else int(spec.gpus),
        "time_budget_hours": time_budget,
        "account": spec.account,
        "hydra_args": tuple(str(a) for a in (spec.hydra_args or ())),
        "backend": spec.backend,
        "cluster": spec.cluster,
        "extra": extra_filtered,
    }
    # ``workload_cmd`` (#588) is keyed ONLY when non-empty so every
    # existing hydra-only spec hashes byte-identically across the
    # upgrade (lease reconnect continuity), while a custom-cmd run for
    # the same issue is a distinct lease key.
    if spec.workload_cmd:
        canonical["workload_cmd"] = spec.workload_cmd
    return canonical


def spec_hash(spec: RunSpec) -> str:
    """SHA-256 of the canonicalized spec dict (first 16 hex chars).

    Used as the lease key's spec component; collisions in 16 hex chars
    are astronomically unlikely across a single issue's attempt history
    (issue 137 would need ~2^32 attempts to hit one).
    """
    blob = json.dumps(canonicalize_spec(spec), sort_keys=True).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:16]


# ---------------------------------------------------------------------------
# Durable routing lease (~/.eps-routing/issue-<N>.json)
# ---------------------------------------------------------------------------


@dataclass
class Lease:
    """In-memory view of a per-issue routing lease.

    Persisted as JSON at ``<lease_dir>/issue-<N>.json``. Holds:

    * ``issue`` — task id (mirrors the filename for sanity).
    * ``spec_hash`` — :func:`spec_hash` of the canonicalized RunSpec the
      lease was opened for. A request whose hash matches reconnects;
      a mismatch implies a different workload shape and the lease is
      stale (the orchestrator's ``set-status approved`` flow should have
      cleared the old lease, but a fresh attempt for a different
      hyperparameter set is also OK — we replace the lease).
    * ``attempt_id`` — stable per-attempt id used as the GCP artifact
      namespace AND as the reconnect key. The GCP backend reads this
      from ``spec.extra["attempt_id"]``; the router sets it here so
      every submit/provision uses the SAME id across the lease lifetime.
    * ``backend`` — which backend was used last (``None`` if no submit
      has happened yet but the lease was opened to claim the attempt id).
    * ``cluster`` — cluster name for SLURM backends (``None`` for GCP).
    * ``job_id`` — external job id (SLURM job id; GCE instance id).
      Populated IMMEDIATELY after submit; absence + lease present =
      ``UNKNOWN_SUBMITTED`` recovery state.
    * ``submitted_at`` — Unix timestamp of the submit.
    * ``gcp_attempts_today`` — per-UTC-day GCP-escalation counter (for
      the attempt-count guard).
    * ``gcp_attempts_date`` — ISO date of the day the counter applies to
      (UTC). On a day-change the counter resets.
    """

    issue: int
    spec_hash: str
    attempt_id: str
    backend: BackendKind | None = None
    cluster: str | None = None
    job_id: str | None = None
    submitted_at: float | None = None
    gcp_attempts_today: int = 0
    gcp_attempts_date: str | None = None

    def to_json(self) -> dict[str, Any]:
        return {
            "issue": self.issue,
            "spec_hash": self.spec_hash,
            "attempt_id": self.attempt_id,
            "backend": self.backend,
            "cluster": self.cluster,
            "job_id": self.job_id,
            "submitted_at": self.submitted_at,
            "gcp_attempts_today": self.gcp_attempts_today,
            "gcp_attempts_date": self.gcp_attempts_date,
        }

    @classmethod
    def from_json(cls, payload: dict[str, Any]) -> Lease:
        return cls(
            issue=int(payload["issue"]),
            spec_hash=str(payload["spec_hash"]),
            attempt_id=str(payload["attempt_id"]),
            backend=payload.get("backend"),
            cluster=payload.get("cluster"),
            job_id=payload.get("job_id"),
            submitted_at=payload.get("submitted_at"),
            gcp_attempts_today=int(payload.get("gcp_attempts_today", 0)),
            gcp_attempts_date=payload.get("gcp_attempts_date"),
        )

    def is_unknown_submitted(self) -> bool:
        """True iff the lease has a backend but no job id (recovery state)."""
        return self.backend is not None and self.job_id is None


class LeaseStore:
    """flock'd JSON lease persistence at ``<lease_dir>/issue-<N>.json``.

    Every mutation holds an exclusive ``flock`` on the lease's
    PER-ISSUE lock file (``<lease_dir>/issue-<N>.lock``) — NOT on the
    lease JSON file itself, because the lease file is created/replaced
    atomically via a write-temp-then-rename and an flock on a file we
    are about to rename is fragile, AND NOT on a shared directory-level
    lock (which would serialize every issue against every other issue —
    a 600 s free-lane park on issue 137 inside ``store.transaction(137)``
    would block a ``route()`` on issue 200 for up to 10 min).

    The per-issue lock spans read+modify+write so a concurrent
    ``issue-tick`` cron and a manual ``/issue`` for the SAME issue
    can't both decide "no live job, submit fresh" and double-submit.
    Concurrent calls for DIFFERENT issues are not serialized — they
    take different locks and proceed in parallel.

    Defaults to ``~/.eps-routing/`` (override for tests via
    ``lease_dir=tmp_path``). The directory is created on first use with
    mode 0o700 (lease contents include job ids — not secrets, but the
    operator shouldn't need a world-readable record either).
    """

    def __init__(self, lease_dir: Path | None = None) -> None:
        self._lease_dir = lease_dir or (Path.home() / LEASE_STORE_DIRNAME)

    @property
    def lease_dir(self) -> Path:
        return self._lease_dir

    def _ensure_dir(self) -> None:
        self._lease_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
        # mkdir-with-mode does NOT chmod an existing dir; defensively
        # tighten if a prior run created it with a wider mode.
        os.chmod(self._lease_dir, 0o700)

    def _lease_path(self, issue: int) -> Path:
        return self._lease_dir / f"issue-{int(issue)}.json"

    def _lock_path(self, issue: int) -> Path:
        """Per-issue lock file (``<lease_dir>/issue-<N>.lock``).

        Per-issue (not directory-global) so a long-held lock on one
        issue cannot block routing on a different issue. Cross-issue
        contention is bounded to the concurrent invocations on the
        SAME issue that we are deliberately serializing.
        """
        return self._lease_dir / f"issue-{int(issue)}.lock"

    @contextmanager
    def _flock(self, issue: int) -> Iterator[None]:
        """Exclusive flock on the PER-ISSUE lock file for the block's duration.

        Read-modify-write on the lease MUST happen inside this context
        so a concurrent process on the SAME issue doesn't read a stale
        lease and overwrite a fresh one with stale data. Concurrent
        processes on DIFFERENT issues hold different locks and DO NOT
        contend.
        """
        self._ensure_dir()
        lock_path = self._lock_path(issue)
        # Open in append mode so the file is created if absent + no truncation.
        with open(lock_path, "ab+") as fh:
            fcntl.flock(fh.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(fh.fileno(), fcntl.LOCK_UN)

    def read(self, issue: int) -> Lease | None:
        """Read the lease for ``issue``. Returns ``None`` if absent / malformed."""
        path = self._lease_path(issue)
        with self._flock(issue):
            return self._read_locked(path)

    def _read_locked(self, path: Path) -> Lease | None:
        if not path.exists():
            return None
        try:
            payload = json.loads(path.read_text())
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("LeaseStore: could not read %s: %s; treating as absent.", path, exc)
            return None
        try:
            return Lease.from_json(payload)
        except (KeyError, ValueError, TypeError) as exc:
            logger.warning("LeaseStore: malformed lease at %s: %s; treating as absent.", path, exc)
            return None

    def write(self, lease: Lease) -> None:
        """Atomic replace of the lease file (write-temp + rename)."""
        path = self._lease_path(lease.issue)
        with self._flock(lease.issue):
            self._write_locked(path, lease)

    def _write_locked(self, path: Path, lease: Lease) -> None:
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(json.dumps(lease.to_json(), sort_keys=True, indent=2))
        os.chmod(tmp, 0o600)
        os.replace(tmp, path)

    def delete(self, issue: int) -> None:
        """Delete the lease file (idempotent on absent)."""
        path = self._lease_path(issue)
        with self._flock(issue):
            try:
                path.unlink()
            except FileNotFoundError:
                return

    @contextmanager
    def transaction(self, issue: int) -> Iterator[tuple[Lease | None, Callable[[Lease], None]]]:
        """Read-modify-write transaction under the per-issue flock.

        Yields ``(current_lease_or_None, write_fn)``. The caller computes
        the new lease state inside the ``with`` block and invokes
        ``write_fn(new_lease)`` to persist it. The per-issue flock is
        held until the block exits — concurrent ``transaction(other_issue)``
        calls do NOT block on it.

        Example::

            with store.transaction(issue=137) as (lease, write):
                if lease is None:
                    lease = Lease(issue=137, spec_hash=h, attempt_id=a)
                lease.job_id = "9999"
                write(lease)
        """
        self._ensure_dir()
        path = self._lease_path(issue)
        with self._flock(issue):
            current = self._read_locked(path)

            def write_fn(new_lease: Lease) -> None:
                self._write_locked(path, new_lease)

            yield current, write_fn


# ---------------------------------------------------------------------------
# Helpers (auto lane order, estimate ranking, GCP attempt counter)
# ---------------------------------------------------------------------------


def _validate_auto_lane_order(
    lanes: tuple[str, ...],
    *,
    source: str,
) -> tuple[BackendKind, ...]:
    """Validate an auto-chain lane order; raise :class:`RouteError` on any defect.

    Hard rules (all raise — a misconfigured order must NEVER be silently
    repaired by dropping entries):

    * ``runpod`` is FORBIDDEN — RunPod spends real money and stays
      override-only; an order that smuggles it in is a real-money safety
      violation, not a preference.
    * Unknown lane names (typos, the ``auto`` sentinel, the legacy
      ``cluster`` literal) raise.
    * Duplicates raise (a duplicated lane would be attempted twice).
    * An empty order raises.
    """
    if not lanes:
        raise RouteError(f"auto lane order from {source} is empty — refusing to route blind")
    for lane in lanes:
        if lane == "runpod":
            raise RouteError(
                f"auto lane order from {source} contains 'runpod' — RunPod spends "
                "real money and is reachable ONLY via an explicit backend override, "
                "never on the auto chain. Remove it from the order."
            )
        if lane not in _AUTO_LANE_VALUES:
            raise RouteError(
                f"auto lane order from {source} contains unknown lane {lane!r}; "
                f"valid lanes: {sorted(_AUTO_LANE_VALUES)}"
            )
    if len(set(lanes)) != len(lanes):
        raise RouteError(f"auto lane order from {source} contains duplicate lanes: {lanes!r}")
    return lanes  # type: ignore[return-value]


def auto_lane_order() -> tuple[BackendKind, ...]:
    """Resolve the auto-chain lane order: env override, else the standing default.

    * :data:`ENV_AUTO_LANE_ORDER` set (non-empty) → parse the
      comma-separated lane list and validate it (``runpod`` / unknown
      names / duplicates raise loudly — never silently dropped).
    * Otherwise → :data:`DEFAULT_AUTO_LANE_ORDER` (GCP first,
      unconditionally — no date gate of any kind).
    """
    raw = os.environ.get(ENV_AUTO_LANE_ORDER, "").strip()
    if not raw:
        return DEFAULT_AUTO_LANE_ORDER
    lanes = tuple(part.strip() for part in raw.split(",") if part.strip())
    return _validate_auto_lane_order(lanes, source=f"{ENV_AUTO_LANE_ORDER}={raw!r}")


def _split_lane_groups(kinds: list[BackendKind]) -> list[tuple[BackendKind, ...]]:
    """Split availability-filtered lane kinds into contiguous attempt groups.

    Each group is either ``("gcp",)`` or a maximal run of consecutive
    SLURM lanes. The auto chain walks groups in order; WITHIN a SLURM
    group the lanes keep the existing est-start ranking + park + cancel
    chain (ties preserve the configured order — ``rank_lanes`` is
    stable), while a GCP group is a single provision attempt.
    """
    groups: list[tuple[BackendKind, ...]] = []
    current: list[BackendKind] = []
    for kind in kinds:
        if kind == "gcp":
            if current:
                groups.append(tuple(current))
                current = []
            groups.append(("gcp",))
        else:
            current.append(kind)
    if current:
        groups.append(tuple(current))
    return groups


def rank_lanes(
    candidates: list[tuple[ComputeBackend, BackendKind, float | None]],
) -> list[tuple[ComputeBackend, BackendKind, float | None, float]]:
    """Sort candidates by clamped est-start (instant < soon < unknown).

    Input: list of ``(backend, kind, est_start_seconds_raw)``. ``None``
    raw means the lane returned no parseable estimate (still park-
    eligible, but ranks LAST). Negative est-starts clamp to ``0.0`` for
    ranking ("would start in the past" = instant, not "below zero / more
    instant than zero" — slice-4 review carry-forward).

    Returns: list of ``(backend, kind, raw, clamped)`` sorted by clamped
    ascending; unknowns (raw=None) sort to the end via ``float("inf")``
    sentinel. Stable across ties (preserves input order).
    """
    decorated: list[tuple[float, int, ComputeBackend, BackendKind, float | None, float]] = []
    for idx, (backend, kind, raw) in enumerate(candidates):
        if raw is None:
            clamped = float("inf")
        elif raw < 0:
            clamped = 0.0
        else:
            clamped = float(raw)
        decorated.append((clamped, idx, backend, kind, raw, clamped))
    decorated.sort(key=lambda t: (t[0], t[1]))
    return [(b, k, raw, clamped) for _c, _i, b, k, raw, clamped in decorated]


def _today_utc_iso() -> str:
    return datetime.now(tz=UTC).date().isoformat()


def _bump_gcp_attempt(lease: Lease) -> Lease:
    """Bump the per-day GCP attempt counter, rolling over on day change."""
    today = _today_utc_iso()
    if lease.gcp_attempts_date != today:
        lease.gcp_attempts_date = today
        lease.gcp_attempts_today = 0
    lease.gcp_attempts_today += 1
    return lease


# ---------------------------------------------------------------------------
# Cancel state machine
# ---------------------------------------------------------------------------


def cancel_and_wait(
    *,
    backend: ComputeBackend,
    handle: RunHandle,
    is_live_after_cancel: Callable[[ComputeBackend, RunHandle], bool],
    is_running_after_cancel: Callable[[ComputeBackend, RunHandle], bool] | None = None,
    grace_seconds: int = CANCEL_LIVE_GRACE_SECONDS,
    poll_interval: float = 2.0,
    now_fn: Callable[[], float] = time.monotonic,
    sleep_fn: Callable[[float], None] = time.sleep,
) -> str:
    """Idempotent cancel: request, poll until job leaves the live queue.

    Returns one of:

    * ``"cancelled"`` — the job is no longer live in the queue (the
      DRAC robot's allowlist forbids ``sacct``, so "no longer live in
      ``squeue --name``" is the best terminal signal we can get).
    * ``"raced_to_running"`` — between cancel-requested and the next
      live-check the job transitioned to RUNNING. We KEEP the job —
      tearing it down would forfeit the wait we already paid for; the
      router uses this lane as the chosen outcome.
    * ``"manual_attention"`` — ``grace_seconds`` elapsed and the job is
      still live. Both the auto and explicit lanes raise
      :class:`ManualAttentionRequiredError` carrying the orphaned job
      id (no silent GCP escalation past a live job — fix6), so the
      operator can manually ``scancel``; the cluster job will
      eventually time out on its own ``--time`` budget regardless.

    ``is_live_after_cancel`` is the polled "is the job ID still
    visible?" probe. The SLURM backend's binding is
    ``squeue --name eps-issue-<N>`` (true while live, false on age-out
    / cancellation). GCP doesn't need this (the auto-fallback path
    never enters the cancel state machine — GCP's "park" is the
    provision call itself), but the abstraction stays uniform.

    ``is_running_after_cancel`` (optional): probe to distinguish
    "actually started running" from "still pending in queue" during the
    cancel grace. When provided, a "true" reply during the grace window
    KEEPS the job and returns ``"raced_to_running"``. When None, the
    function only polls is_live (and a job that flipped to RUNNING but
    is still live will eventually drop out of the live queue when its
    own ``scancel`` lands; we won't notice the RUNNING transition).
    """
    # Request cancel via the backend's teardown. Idempotent on a missing
    # job (the SLURM scancel wrapper logs but does not raise).
    try:
        backend.teardown(handle)
    except Exception as exc:
        logger.warning(
            "cancel_and_wait: teardown raised for %s/%s (%s: %s); continuing to live-poll.",
            handle.backend,
            handle.job_id,
            type(exc).__name__,
            exc,
        )

    start = now_fn()
    while True:
        # If the operator gave us a "did it start" probe and the job
        # is now actually RUNNING, KEEP it. The scancel we just issued
        # raced against the scheduler; the job won — let it finish.
        if is_running_after_cancel is not None:
            try:
                if is_running_after_cancel(backend, handle):
                    logger.info(
                        "cancel_and_wait: %s/%s raced to RUNNING during cancel; keeping.",
                        handle.backend,
                        handle.job_id,
                    )
                    return "raced_to_running"
            except Exception as exc:
                logger.warning(
                    "cancel_and_wait: is_running probe raised (%s: %s); continuing.",
                    type(exc).__name__,
                    exc,
                )
        # If the job is no longer live, we've cancelled successfully.
        try:
            live = is_live_after_cancel(backend, handle)
        except Exception as exc:
            logger.warning(
                "cancel_and_wait: is_live probe raised (%s: %s); treating as still-live.",
                type(exc).__name__,
                exc,
            )
            live = True
        if not live:
            return "cancelled"
        if now_fn() - start >= grace_seconds:
            logger.warning(
                "cancel_and_wait: %s/%s still live after %ds grace; manual_attention.",
                handle.backend,
                handle.job_id,
                grace_seconds,
            )
            return "manual_attention"
        sleep_fn(poll_interval)


# ---------------------------------------------------------------------------
# Park watchdog
# ---------------------------------------------------------------------------


def park_until_running_or_cap(
    *,
    backend: ComputeBackend,
    handle: RunHandle,
    is_started: Callable[[ComputeBackend, RunHandle], bool],
    cap_seconds: int = FREE_WAIT_SECONDS,
    poll_interval: float = DEFAULT_POLL_INTERVAL,
    now_fn: Callable[[], float] = time.monotonic,
    sleep_fn: Callable[[float], None] = time.sleep,
) -> tuple[bool, str, str | None]:
    """Watch a launched handle for ``cap_seconds``; return (started, reason, terminal_status).

    ``is_started`` is the backend-aware probe — for SLURM it queries
    ``squeue -j <id>`` for state RUNNING (the production
    ``slurm_monitor.SLURM_STATE_TO_STATUS`` maps PENDING→running for
    historical reasons, so the router cannot use the PollResult.status
    field directly to distinguish PENDING from RUNNING). For GCP the
    binding is ``backend.poll(handle).status == "running"``. For tests
    the binding is whatever the test double exposes.

    Returns ``(started, reason, terminal_status)``:

    * ``(True, "running", None)`` — job reached RUNNING before the cap.
    * ``(False, "park_cap_exceeded", None)`` — still PENDING (or
      otherwise not-running) at the cap. Caller should run
      :func:`cancel_and_wait` and escalate to the next tier.
    * ``(False, "terminal_before_running", <poll.status>)`` — the
      probe-poll returned a terminal-ish status (done/dead/stalled/gate)
      before RUNNING. ``terminal_status`` is the triggering
      ``PollResult.status`` so the caller can distinguish genuinely-gone
      jobs (``done`` / ``dead`` — eligible for the started-evidence
      probe) from possibly-LIVE ones (``stalled`` covers RUNNING with a
      stale heartbeat and SUSPENDED; ``gate`` is a live wait) which MUST
      go through the cancel state machine first (round-6 M1 — the
      issue-535 live run skipped the cancel on a stalled-classified
      LIVE job and orphaned it).
    * ``(False, "probe_failures_exceeded", None)`` —
      :data:`PARK_MAX_CONSECUTIVE_PROBE_FAILURES` CONSECUTIVE
      ``is_started`` failures: the job state is UNKNOWN (transport
      down), which must never read as "still pending" forever NOR as
      terminal (round-6 B1). Caller routes to the cancel state machine;
      with the transport still broken that resolves to
      ``manual_attention``.
    """
    start = now_fn()
    consecutive_probe_failures = 0
    while True:
        try:
            started = is_started(backend, handle)
            consecutive_probe_failures = 0
        except Exception as exc:
            consecutive_probe_failures += 1
            logger.warning(
                "park: is_started probe raised (%s: %s); consecutive failure %d/%d — "
                "treating as still-pending.",
                type(exc).__name__,
                exc,
                consecutive_probe_failures,
                PARK_MAX_CONSECUTIVE_PROBE_FAILURES,
            )
            if consecutive_probe_failures >= PARK_MAX_CONSECUTIVE_PROBE_FAILURES:
                return False, "probe_failures_exceeded", None
            started = False
        if started:
            return True, "running", None
        # Check for terminal-before-running via the backend's poll.
        # Wrapped so a probe that ALSO raises here doesn't crash.
        try:
            poll = backend.poll(handle)
        except Exception as exc:
            logger.warning(
                "park: backend.poll raised (%s: %s); treating as still-pending.",
                type(exc).__name__,
                exc,
            )
            poll = None
        if poll is not None and _is_terminal_status(poll):
            return False, "terminal_before_running", poll.status
        if now_fn() - start >= cap_seconds:
            return False, "park_cap_exceeded", None
        sleep_fn(poll_interval)


def _is_terminal_status(poll: PollResult) -> bool:
    return poll.status in {"done", "dead", "stalled", "gate"}


def default_is_started(backend: ComputeBackend, handle: RunHandle) -> bool:
    """Default ``is_started`` probe: ``backend.poll(handle).status == "running"``.

    Production callers wiring the SLURM backend MUST override this with
    a ``squeue -j <id>``-based probe (slurm_monitor's state mapping
    treats SLURM PENDING as PollResult.status="running", which would
    short-circuit the park watchdog incorrectly). Tests use this default
    against backends whose ``poll`` is mocked to return "pending" /
    "running" as needed.
    """
    return backend.poll(handle).status == "running"


def default_is_live(backend: ComputeBackend, handle: RunHandle) -> bool:
    """Default ``is_live_after_cancel`` probe.

    Falls back to ``backend.poll(handle).status not in {done, dead}``
    which is a coarse proxy. Production SLURM callers should bind this
    to ``squeue --name eps-issue-<N>`` returning non-empty (the
    authoritative "still in the queue" signal even when ``scontrol``
    has aged out).
    """
    status = backend.poll(handle).status
    return status not in {"done", "dead"}


# ---------------------------------------------------------------------------
# The router
# ---------------------------------------------------------------------------


@dataclass
class RouterConfig:
    """Per-call knobs for :func:`route`.

    Defaults reproduce the plan's production policy. Tests override
    every callable + the lease store + the lane factories.
    """

    free_wait_seconds: int = FREE_WAIT_SECONDS
    poll_interval: float = DEFAULT_POLL_INTERVAL
    cancel_grace_seconds: int = CANCEL_LIVE_GRACE_SECONDS
    max_gcp_attempts_per_day: int = MAX_GCP_ATTEMPTS_PER_DAY
    #: Per-call auto lane order override. ``None`` (the default) resolves
    #: via :func:`auto_lane_order` (env override, else the GCP-first
    #: standing default). A non-None value is validated at ``route()``
    #: entry with the same rules as the env override (``runpod`` /
    #: unknown lanes / duplicates raise).
    lane_order: tuple[BackendKind, ...] | None = None


def route(
    spec: RunSpec,
    *,
    runpod_backend: ComputeBackend,
    free_backends: dict[BackendKind, ComputeBackend] | None = None,
    gcp_backend: ComputeBackend | None = None,
    lease_store: LeaseStore | None = None,
    mila_socket_alive: Callable[[], bool] | None = None,
    is_started: Callable[[ComputeBackend, RunHandle], bool] = default_is_started,
    is_live_after_cancel: Callable[[ComputeBackend, RunHandle], bool] = default_is_live,
    is_running_after_cancel: Callable[[ComputeBackend, RunHandle], bool] | None = None,
    started_evidence_probe: (
        Callable[[ComputeBackend, RunHandle], dict[str, Any] | None] | None
    ) = None,
    estimate_fn: (Callable[[ComputeBackend, BackendKind, RunSpec], float | None] | None) = None,
    reconnect_fn: (
        Callable[[ComputeBackend, BackendKind, RunSpec], RunHandle | None] | None
    ) = None,
    marker_poster: Callable[..., None] | None = None,
    on_launched: Callable[[RunHandle], None] | None = None,
    config: RouterConfig | None = None,
    now_fn: Callable[[], float] = time.monotonic,
    sleep_fn: Callable[[float], None] = time.sleep,
    clock_fn: Callable[[], datetime] | None = None,
) -> RouteResult:
    """Dispatch ``spec`` to the right backend per the multi-backend ladder.

    See module docstring for the full decision flow.

    Required injections:

    * ``runpod_backend`` — used ONLY when ``spec.backend == "runpod"``
      (the explicit override). The router NEVER calls ``runpod_backend.launch``
      on an auto path; the negative test
      ``test_no_auto_runpod_path_under_any_failure`` proves it by
      injecting a raising RunPod backend.

    Optional injections:

    * ``free_backends`` — map of free-lane kind → backend instance
      (e.g. ``{"nibi": slurm, "fir": slurm, "mila": mila}``). Auto
      routing visits these at their position in the resolved lane order
      (:attr:`RouterConfig.lane_order`, else :func:`auto_lane_order` —
      env override, else the GCP-first standing default). A missing
      kind is skipped (e.g. ``mila`` absent → router skips Mila even
      when the socket is alive).
    * ``gcp_backend`` — the auto-fallback target. When ``None`` and the
      auto chain reaches GCP, the router raises
      :class:`NoComputeAvailableError`.
    * ``lease_store`` — defaults to :class:`LeaseStore` at
      ``~/.eps-routing/``. Tests pass a store keyed on ``tmp_path``.
    * ``mila_socket_alive`` — predicate; when ``False``, Mila is
      skipped without a probe. Defaults to ``False`` (router behaves as
      if Mila is down unless wired).
    * ``is_started`` / ``is_live_after_cancel`` — backend-aware probes;
      see the helper docstrings. Defaults use ``backend.poll`` for both
      (sufficient for the test doubles; production wiring overrides for
      SLURM).
    * ``is_running_after_cancel`` — optional probe to detect the
      cancel-race; see :func:`cancel_and_wait`. Defaults to None (no
      race detection).
    * ``started_evidence_probe`` — ``(backend, handle) -> evidence dict
      | None``; consulted ONLY on a ``terminal_before_running`` park
      outcome to distinguish "never started" (no-compute) from "started
      and FAILED fast" (workload failure — surfaced via
      :class:`WorkloadSurfacedError`, NO auto-fallback). Production
      wiring scp/rsync-reads the SLURM scratch dir for ``status.json`` /
      ``job.out`` (``slurm_monitor.fetch_started_evidence``). Defaults
      to None (no probe — every terminal-before-running classifies as
      no-compute, the pre-fix behavior).
    * ``estimate_fn`` — ``(backend, kind, spec) -> seconds | None`` for
      free-lane ranking. Defaults to calling the backend's
      ``estimate_start_seconds(spec)`` method when available, else
      ``None`` (unranked but park-eligible).
    * ``reconnect_fn`` — ``(backend, kind, spec) -> RunHandle | None``;
      the router calls this BEFORE any submit/provision to find an
      existing live job. Defaults to None (no reconnect — fresh
      submit).
    * ``marker_poster`` — see ``epm:backend-selected`` in
      :data:`workflow.yaml`. Defaults to None (no marker posted; slice
      6 wires the real ``post_marker_via_task_py``).
    * ``on_launched`` — persistence hook invoked with the
      :class:`RunHandle` IMMEDIATELY after every successful launch /
      reconnect, BEFORE any marker post or further routing work. The
      dispatch helper wires the handle-sidecar write here so a launched
      handle is ALWAYS recoverable by ``dispatch_issue.py finalize``
      even if everything after the launch crashes. Guarded — a hook
      failure is logged loud and never kills a live launch.
    * ``config`` — see :class:`RouterConfig`. Defaults to a fresh
      instance with the module constants.

    Raises:

    * :class:`NoComputeAvailableError` — terminal no-compute outcome.
    * :class:`WorkloadSurfacedError` — a backend reported a
      :class:`GcpWorkloadError`; the router does NOT auto-fallback.
    * :class:`GcpAttemptCapExceededError` — per-day GCP attempt-count
      guard tripped.
    """
    cfg = config or RouterConfig()
    store = lease_store or LeaseStore()
    started_at = now_fn()
    attempts: list[RouteAttempt] = []

    # :class:`RunSpec.backend` defaults to ``"auto"`` so a direct
    # ``RunSpec(issue, intent)`` routes through the cost-safe auto chain
    # (free lanes → GCP) — a real-money RunPod launch ALWAYS requires
    # an explicit ``backend="runpod"``. Any other recognized value is
    # an explicit override; an unknown value would have been rejected at
    # :class:`BackendKind` parse time.
    #
    # Belt-and-suspenders: a stringly-typed miswire (``backend=""`` /
    # ``backend=None`` / a typo like ``"runpd"``) MUST NOT silently fall
    # through to the auto chain — that would mask a config bug in the
    # caller. ``BackendKind`` parse-time validation only covers spec
    # *construction*; a caller that bypasses the dataclass and mutates
    # ``spec.backend`` post hoc gets caught here.
    if spec.backend in (None, "") or spec.backend not in _VALID_BACKEND_VALUES:
        raise RouteError(
            f"route(): spec.backend must be one of "
            f"{sorted(_VALID_BACKEND_VALUES)!r}, got {spec.backend!r}. "
            "Empty / None / unknown backend strings are rejected to "
            "prevent silent auto-routing of a miswired override."
        )

    # ------------------------------ explicit override --------------------
    if spec.backend == "runpod":
        return _override_runpod(
            spec=spec,
            backend=runpod_backend,
            store=store,
            attempts=attempts,
            started_at=started_at,
            now_fn=now_fn,
            marker_poster=marker_poster,
            on_launched=on_launched,
        )

    if spec.backend in {"nibi", "fir", "mila"}:
        free = (free_backends or {}).get(spec.backend)
        if free is None:
            raise RouteError(
                f"backend override {spec.backend!r} requested but no free backend wired for it"
            )
        spec = _spec_for_lane(spec, spec.backend)
        return _override_free_or_gcp(
            spec=spec,
            backend=free,
            kind=spec.backend,
            store=store,
            attempts=attempts,
            started_at=started_at,
            cfg=cfg,
            is_started=is_started,
            is_live_after_cancel=is_live_after_cancel,
            is_running_after_cancel=is_running_after_cancel,
            reconnect_fn=reconnect_fn,
            now_fn=now_fn,
            sleep_fn=sleep_fn,
            marker_poster=marker_poster,
            on_launched=on_launched,
            started_evidence_probe=started_evidence_probe,
        )

    if spec.backend == "gcp":
        if gcp_backend is None:
            raise RouteError("backend override 'gcp' requested but no gcp_backend wired")
        return _override_free_or_gcp(
            spec=spec,
            backend=gcp_backend,
            kind="gcp",
            store=store,
            attempts=attempts,
            started_at=started_at,
            cfg=cfg,
            is_started=is_started,
            is_live_after_cancel=is_live_after_cancel,
            is_running_after_cancel=is_running_after_cancel,
            reconnect_fn=reconnect_fn,
            now_fn=now_fn,
            sleep_fn=sleep_fn,
            marker_poster=marker_poster,
            on_launched=on_launched,
            started_evidence_probe=started_evidence_probe,
        )

    # ----------------------------- auto chain ---------------------------
    # Resolve the lane order ONCE at entry (fail-fast on a malformed env
    # override / config order, before any reconnect or submit I/O).
    if cfg.lane_order is not None:
        lane_order = _validate_auto_lane_order(
            tuple(cfg.lane_order), source="RouterConfig.lane_order"
        )
        order_source = "RouterConfig.lane_order"
    else:
        lane_order = auto_lane_order()
        order_source = (
            f"{ENV_AUTO_LANE_ORDER} env override"
            if os.environ.get(ENV_AUTO_LANE_ORDER, "").strip()
            else "default (GCP-first standing order)"
        )
    logger.info(
        "route(): issue=%d auto lane order = %s (source: %s)",
        spec.issue,
        " -> ".join(lane_order),
        order_source,
    )
    return _auto_route(
        spec=spec,
        free_backends=free_backends or {},
        gcp_backend=gcp_backend,
        store=store,
        attempts=attempts,
        started_at=started_at,
        cfg=cfg,
        lane_order=lane_order,
        is_started=is_started,
        is_live_after_cancel=is_live_after_cancel,
        is_running_after_cancel=is_running_after_cancel,
        started_evidence_probe=started_evidence_probe,
        mila_socket_alive=mila_socket_alive,
        estimate_fn=estimate_fn,
        reconnect_fn=reconnect_fn,
        now_fn=now_fn,
        sleep_fn=sleep_fn,
        marker_poster=marker_poster,
        on_launched=on_launched,
        clock_fn=clock_fn,
    )


# ---------------------------------------------------------------------------
# Override paths
# ---------------------------------------------------------------------------


def _invoke_on_launched(
    on_launched: Callable[[RunHandle], None] | None,
    handle: RunHandle,
) -> None:
    """Run the post-launch persistence hook; NEVER let it kill a live launch.

    The hook fires IMMEDIATELY after a successful launch / reconnect,
    BEFORE any marker post, so the dispatch helper's handle-sidecar
    write lands while the only thing that has happened is the launch
    itself — a crash anywhere later still leaves a recoverable handle
    for ``dispatch_issue.py finalize``. A hook failure (e.g. disk
    error on the sidecar write) is logged LOUD and swallowed: the
    launch already succeeded, and the dispatch helper's authoritative
    final write is the second chance.
    """
    if on_launched is None:
        return
    try:
        on_launched(handle)
    except Exception:
        logger.exception(
            "route: on_launched hook FAILED for job_id=%s pod_name=%s — handle "
            "persistence may be missing; continuing (launch already succeeded).",
            handle.job_id,
            handle.pod_name,
        )


def _prepare_and_launch(
    backend: ComputeBackend,
    spec: RunSpec,
    *,
    kind: BackendKind,
    cluster: str | None = None,
) -> RunHandle:
    """FRESH-launch chokepoint: ``backend.prepare(spec)`` then ``backend.launch(spec)``.

    Every fresh launch site MUST go through this helper. Live-acceptance
    finding (issue 535): the router called ``launch`` directly at every
    site, so ``SlurmBackend.prepare`` — the rsync repo sync +
    ``render_secrets_env()`` + secrets push — had ZERO production
    callers, and the first live Nibi job died in in-job preflight with
    no rsynced repo and no ``secrets.env`` in its scratch dir.

    RECONNECT sites must NOT call this helper (and must not call
    ``prepare`` at all): ``SlurmBackend.prepare`` rsyncs the scratch dir
    with ``--delete``, so re-preparing the scratch of a RUNNING job
    could yank code out from under the live workload mid-run.

    A ``prepare`` failure raises :class:`BackendPrepareError` —
    provision-class, pre-launch, nothing live. The auto chain treats it
    like a launch failure (next tier); explicit overrides surface it as
    a typed terminal.
    """
    try:
        backend.prepare(spec)
    except Exception as exc:
        raise BackendPrepareError(
            f"backend.prepare failed for {kind}/{cluster or 'no-cluster'} "
            f"({type(exc).__name__}: {exc})",
            kind=kind,
            cluster=cluster,
        ) from exc
    return backend.launch(spec)


def _probe_started_evidence(
    probe: Callable[[ComputeBackend, RunHandle], dict[str, Any] | None] | None,
    backend: ComputeBackend,
    handle: RunHandle,
) -> dict[str, Any] | None:
    """Run the started-evidence probe; fail OPEN (``None``) on probe failure.

    Used on a ``terminal_before_running`` park outcome to distinguish
    "never started" (genuine no-compute) from "started and FAILED fast"
    (PD→R→exit between polls — a WORKLOAD failure that must surface
    with NO auto-fallback; escalating it to GCP would burn paid credit
    re-running a doomed workload).

    ``None`` — whether because no probe is wired, the probe found no
    runtime artifacts, or the probe itself failed — preserves the
    legacy ``no_compute_available`` classification. A probe failure is
    logged LOUD but never becomes a new crash path between "job
    vanished" and the router terminal.
    """
    if probe is None:
        return None
    try:
        return probe(backend, handle)
    except Exception as exc:
        logger.warning(
            "route: started-evidence probe FAILED for %s/%s (%s: %s); "
            "falling back to no_compute classification.",
            handle.backend,
            handle.job_id,
            type(exc).__name__,
            exc,
        )
        return None


def _override_runpod(
    *,
    spec: RunSpec,
    backend: ComputeBackend,
    store: LeaseStore,
    attempts: list[RouteAttempt],
    started_at: float,
    now_fn: Callable[[], float],
    marker_poster: Callable[..., None] | None,
    on_launched: Callable[[RunHandle], None] | None = None,
) -> RouteResult:
    """Explicit RunPod override — just submit. No park, no fallback.

    RunPod's "start time" is the few-minute provision; we don't gate it
    behind a park watchdog (the existing RunPod flow doesn't, and a 10
    min park would force a real user-meaningful timeout where today
    there is none). Reconnect via the lease's job_id is wired in slice
    6 (the RunPod backend doesn't yet expose a "find live pod by name"
    handle-reconstructor; today the existing pod_lifecycle.py path is
    idempotent itself).

    Lock discipline: holds the per-issue flock across the launch + lease
    write so a concurrent invocation cannot double-submit. RunPod
    provisioning is seconds-to-minutes; the lock is per-ISSUE (not
    cross-issue) so contention is bounded to the racing invocations we
    are deliberately serializing.
    """
    # Hold the per-issue flock across launch + persist so two concurrent
    # route() calls cannot both decide "no live job, submit fresh" and
    # provision twice.
    with store.transaction(spec.issue) as (lease, write):
        try:
            handle = _prepare_and_launch(backend, spec, kind="runpod")
        except BackendPrepareError as exc:
            attempts.append(
                RouteAttempt(
                    kind="runpod",
                    cluster=None,
                    est_start_seconds_raw=None,
                    est_start_seconds_clamped=None,
                    outcome="prepare_failed",
                    detail=exc.reason,
                    elapsed_seconds=now_fn() - started_at,
                )
            )
            _post_terminal_failure_marker(
                spec=spec,
                marker_poster=marker_poster,
                reason=ROUTE_REASON_NO_COMPUTE,
                chosen_kind="runpod",
                attempts=attempts,
            )
            raise
        _invoke_on_launched(on_launched, handle)
        write(_lease_after_submit(lease, spec, "runpod", None, handle))
    attempt = RouteAttempt(
        kind="runpod",
        cluster=None,
        est_start_seconds_raw=None,
        est_start_seconds_clamped=None,
        outcome="launched",
        detail="explicit override",
        elapsed_seconds=now_fn() - started_at,
    )
    attempts.append(attempt)
    result = RouteResult(
        backend=backend,
        handle=handle,
        requested_kind="runpod",
        chosen_kind="runpod",
        reason=ROUTE_REASON_OVERRIDE,
        cluster=None,
        attempts=attempts,
        elapsed_seconds=now_fn() - started_at,
    )
    _post_backend_selected(result, spec=spec, marker_poster=marker_poster)
    return result


def _provisioning_detail(exc: GcpProvisioningError) -> str:
    """Attempt detail for a GCP provisioning failure — reason + captured stderr tail.

    ``classify_create_failure`` packages the gcloud stderr into
    ``exc.evidence["stderr_tail"]``, but the pre-#608 handlers recorded
    only ``exc.reason`` ("... (stderr below)" with nothing below): the
    four quota-doomed creates on issue 608 left no stderr anywhere
    (marker, failure JSON, logs) and root-causing took a manual gcloud
    reproduction. This detail flows into the ``epm:backend-selected``
    attempt rows AND the ``NoComputeAvailableError.attempts`` that
    ``classify_terminal_exception`` serializes into the terminal failure
    JSON, so the evidence survives in both surfaces.
    """
    tail = str(exc.evidence.get("stderr_tail") or "").strip()
    if not tail:
        return exc.reason
    return f"{exc.reason}; stderr_tail: {tail[-1024:]}"


def _gcp_quota_headroom_or_none(backend: ComputeBackend, spec: RunSpec) -> QuotaHeadroom | None:
    """Run the GCP regional-quota headroom pre-check; fail OPEN (``None``) on any failure.

    Duck-typed via the backend's ``preflight_quota_headroom`` method so
    test doubles / backends without the probe skip the pre-check entirely
    (#608: four guaranteed-fail creates burned the daily attempt cap
    against an exhausted regional accelerator quota). ``None`` means "no
    opinion — proceed to launch exactly as before"; only a POSITIVE
    insufficient-headroom reading skips the lane.
    """
    probe = getattr(backend, "preflight_quota_headroom", None)
    if probe is None:
        return None
    try:
        return probe(spec)
    except Exception as exc:
        logger.warning(
            "route: GCP quota-headroom pre-check failed OPEN (%s: %s); "
            "proceeding to launch as before.",
            type(exc).__name__,
            exc,
        )
        return None


def _override_free_or_gcp(
    *,
    spec: RunSpec,
    backend: ComputeBackend,
    kind: BackendKind,
    store: LeaseStore,
    attempts: list[RouteAttempt],
    started_at: float,
    cfg: RouterConfig,
    is_started: Callable[[ComputeBackend, RunHandle], bool],
    is_live_after_cancel: Callable[[ComputeBackend, RunHandle], bool],
    is_running_after_cancel: Callable[[ComputeBackend, RunHandle], bool] | None,
    reconnect_fn: (Callable[[ComputeBackend, BackendKind, RunSpec], RunHandle | None] | None),
    now_fn: Callable[[], float],
    sleep_fn: Callable[[float], None],
    marker_poster: Callable[..., None] | None,
    on_launched: Callable[[RunHandle], None] | None = None,
    started_evidence_probe: (
        Callable[[ComputeBackend, RunHandle], dict[str, Any] | None] | None
    ) = None,
) -> RouteResult:
    """Explicit non-RunPod lane override.

    Reconnect first (idempotent re-entry), then launch + park. A free
    lane that times out / hard-fails RAISES (the user explicitly asked
    for that lane; we don't silently re-route).

    Lock discipline: the per-issue flock is held across reconnect-check
    → launch → lease-write so a concurrent invocation (manual /issue vs
    the issue-tick cron) cannot both decide "no live job, submit fresh"
    and double-submit. The lock spans the park watchdog too — wait IS
    contention surface, but it is per-ISSUE (not cross-issue), so the
    only callers serialized are the two we are deliberately serializing.
    """
    with store.transaction(spec.issue) as (lease, write):
        # Reconnect — inside the lock so a concurrent submit can't slip
        # between our "no live job" check and our launch. NO prepare()
        # on reconnect: SlurmBackend.prepare rsyncs the scratch dir with
        # --delete and would yank code out from under the RUNNING job.
        #
        # A PROBE failure (BackendProbeError — transport down, NOT "no
        # live job") must not fall through to a blind fresh submit: a
        # live job may exist and prepare()'s --delete rsync + a second
        # sbatch would corrupt / duplicate it (round-6 B1). Explicit
        # lane → typed terminal.
        try:
            handle = _try_reconnect(
                backend=backend, kind=kind, spec=spec, reconnect_fn=reconnect_fn
            )
        except BackendProbeError as exc:
            attempts.append(
                RouteAttempt(
                    kind=kind,
                    cluster=spec.cluster,
                    est_start_seconds_raw=None,
                    est_start_seconds_clamped=None,
                    outcome="reconnect_probe_failed",
                    detail=f"{type(exc).__name__}: {exc}",
                    elapsed_seconds=now_fn() - started_at,
                )
            )
            _post_terminal_failure_marker(
                spec=spec,
                marker_poster=marker_poster,
                reason=ROUTE_REASON_NO_COMPUTE,
                chosen_kind=kind,
                attempts=attempts,
            )
            raise NoComputeAvailableError(
                f"explicit override {kind!r}: reconnect probe failed — cannot verify "
                f"whether a live job exists; refusing to submit blind ({exc})",
                attempts=[_attempt_to_dict(a) for a in attempts],
            ) from exc
        if handle is not None:
            _invoke_on_launched(on_launched, handle)
            attempts.append(
                RouteAttempt(
                    kind=kind,
                    cluster=spec.cluster,
                    est_start_seconds_raw=None,
                    est_start_seconds_clamped=None,
                    outcome="reconnected",
                    detail="found existing live job/instance",
                    elapsed_seconds=now_fn() - started_at,
                )
            )
            result = RouteResult(
                backend=backend,
                handle=handle,
                requested_kind=kind,
                chosen_kind=kind,
                reason=ROUTE_REASON_RECONNECT,
                cluster=spec.cluster,
                attempts=attempts,
                elapsed_seconds=now_fn() - started_at,
            )
            _post_backend_selected(result, spec=spec, marker_poster=marker_poster)
            return result

        # Fresh submit (still under the flock).
        threaded_spec, lease = _thread_attempt_id_into(spec, lease, write)
        try:
            handle = _prepare_and_launch(backend, threaded_spec, kind=kind, cluster=spec.cluster)
        except BackendPrepareError as exc:
            # Explicit lane — prepare failed BEFORE launch (nothing
            # live). Provision-class typed terminal: breadcrumb, raise.
            # Breadcrumb reason matches the typed terminal's
            # ``reason: backend_prepare_failed`` (round-6 Mn1 — it
            # previously said ``no_compute_available`` while the
            # epm:failure note said ``backend_prepare_failed``).
            attempts.append(
                RouteAttempt(
                    kind=kind,
                    cluster=spec.cluster,
                    est_start_seconds_raw=None,
                    est_start_seconds_clamped=None,
                    outcome="prepare_failed",
                    detail=exc.reason,
                    elapsed_seconds=now_fn() - started_at,
                )
            )
            _post_terminal_failure_marker(
                spec=spec,
                marker_poster=marker_poster,
                reason=ROUTE_REASON_PREPARE_FAILED,
                chosen_kind=kind,
                attempts=attempts,
            )
            raise
        except GcpProvisioningError as exc:
            # Explicit GCP override — surface the provisioning failure (the
            # user asked for GCP, not a fallback chain). Post a terminal
            # breadcrumb so the dashboard sees the failure before we raise.
            attempts.append(
                RouteAttempt(
                    kind=kind,
                    cluster=spec.cluster,
                    est_start_seconds_raw=None,
                    est_start_seconds_clamped=None,
                    outcome="provisioning_failure",
                    detail=_provisioning_detail(exc),
                    elapsed_seconds=now_fn() - started_at,
                )
            )
            _post_terminal_failure_marker(
                spec=spec,
                marker_poster=marker_poster,
                reason=ROUTE_REASON_NO_COMPUTE,
                chosen_kind=kind,
                attempts=attempts,
            )
            raise
        except BackendProbeError as exc:
            # The backend's own pre-create state probe failed mid-launch
            # (e.g. GcpBackend.launch's internal reconnect_or_none with
            # expired gcloud auth — live auto-lane finding, issue 535:
            # this propagated UNCAUGHT to rc=4 instead of the typed
            # fail-closed terminal). State is UNKNOWN → refuse to act
            # blind; same contract as the reconnect-seam handler above.
            attempts.append(
                RouteAttempt(
                    kind=kind,
                    cluster=spec.cluster,
                    est_start_seconds_raw=None,
                    est_start_seconds_clamped=None,
                    outcome="probe_failed",
                    detail=str(exc)[:500],
                    elapsed_seconds=now_fn() - started_at,
                )
            )
            _post_terminal_failure_marker(
                spec=spec,
                marker_poster=marker_poster,
                reason=ROUTE_REASON_NO_COMPUTE,
                chosen_kind=kind,
                attempts=attempts,
            )
            raise NoComputeAvailableError(
                f"explicit override '{kind}': backend state probe failed mid-launch — "
                f"refusing to act blind: {exc}",
                attempts=[_attempt_to_dict(a) for a in attempts],
            ) from exc
        # Persist the handle (sidecar hook) + launched id IMMEDIATELY
        # (still inside the flock — crash-window-free). For "kind ==
        # gcp" override we leave the cluster field at None, matching
        # the existing schema.
        _invoke_on_launched(on_launched, handle)
        write(_lease_after_submit(lease, spec, kind, spec.cluster, handle))

        # GCP doesn't need the park (provision IS the start); just return.
        if kind == "gcp":
            attempts.append(
                RouteAttempt(
                    kind=kind,
                    cluster=None,
                    est_start_seconds_raw=0.0,
                    est_start_seconds_clamped=0.0,
                    outcome="launched",
                    detail="gcp provision returned RUNNING-equivalent",
                    elapsed_seconds=now_fn() - started_at,
                )
            )
            result = RouteResult(
                backend=backend,
                handle=handle,
                requested_kind=kind,
                chosen_kind=kind,
                reason=ROUTE_REASON_OVERRIDE,
                cluster=None,
                attempts=attempts,
                elapsed_seconds=now_fn() - started_at,
            )
            _post_backend_selected(result, spec=spec, marker_poster=marker_poster)
            return result

        # SLURM-style free lane: run the park watchdog (under the flock).
        started, reason, terminal_status = park_until_running_or_cap(
            backend=backend,
            handle=handle,
            is_started=is_started,
            cap_seconds=cfg.free_wait_seconds,
            poll_interval=cfg.poll_interval,
            now_fn=now_fn,
            sleep_fn=sleep_fn,
        )
        if started:
            attempts.append(
                RouteAttempt(
                    kind=kind,
                    cluster=spec.cluster,
                    est_start_seconds_raw=None,
                    est_start_seconds_clamped=None,
                    outcome="launched",
                    detail="park resolved to RUNNING",
                    elapsed_seconds=now_fn() - started_at,
                )
            )
            result = RouteResult(
                backend=backend,
                handle=handle,
                requested_kind=kind,
                chosen_kind=kind,
                reason=ROUTE_REASON_OVERRIDE,
                cluster=spec.cluster,
                attempts=attempts,
                elapsed_seconds=now_fn() - started_at,
            )
            _post_backend_selected(result, spec=spec, marker_poster=marker_poster)
            return result

        # Park failed. Distinguish "never started" from "started and
        # FAILED": a fast-failing job transitions PD→R→exit between
        # polls, so "vanished before observed RUNNING" is NOT proof the
        # cluster lacked capacity. If the scratch dir holds runtime
        # artifacts (status.json / job.out), the job DID start — that is
        # a WORKLOAD failure (surface, no fallback), not no-compute.
        #
        # GATED on the job being genuinely GONE (done/dead): ``stalled``
        # covers LIVE jobs (RUNNING + stale heartbeat; SUSPENDED) and
        # ``gate`` is a live wait — classifying those here raised BEFORE
        # the cancel machine and orphaned a live job (round-6 M1, issue
        # 535 attempt 2). stalled/gate fall through to cancel_and_wait.
        if reason == "terminal_before_running" and terminal_status in ("done", "dead"):
            evidence = _probe_started_evidence(started_evidence_probe, backend, handle)
            if evidence is not None:
                attempts.append(
                    RouteAttempt(
                        kind=kind,
                        cluster=spec.cluster,
                        est_start_seconds_raw=None,
                        est_start_seconds_clamped=None,
                        outcome="workload_failure",
                        detail=(
                            "terminal before RUNNING with runtime artifacts "
                            f"(phase={evidence.get('phase', '')!r})"
                        ),
                        elapsed_seconds=now_fn() - started_at,
                    )
                )
                _post_terminal_failure_marker(
                    spec=spec,
                    marker_poster=marker_poster,
                    reason=ROUTE_REASON_WORKLOAD_FAILURE,
                    chosen_kind=kind,
                    attempts=attempts,
                    extra={"evidence": evidence},
                )
                raise WorkloadSurfacedError(
                    f"{kind} job {handle.job_id} went terminal before RUNNING but "
                    f"left runtime artifacts (phase={evidence.get('phase', '')!r}) — "
                    "workload failure, no auto-fallback",
                    chosen_kind=kind,
                    evidence=evidence,
                )

        # The user explicitly asked for this lane → cancel state
        # machine, then either KEEP (raced) or surface terminal.
        cancel_outcome = cancel_and_wait(
            backend=backend,
            handle=handle,
            is_live_after_cancel=is_live_after_cancel,
            is_running_after_cancel=is_running_after_cancel,
            grace_seconds=cfg.cancel_grace_seconds,
            poll_interval=min(2.0, cfg.poll_interval),
            now_fn=now_fn,
            sleep_fn=sleep_fn,
        )
        # Special case: cancel-race kept the job (raced to RUNNING). Return
        # it as the chosen outcome — we didn't actually cancel, the job won.
        if cancel_outcome == "raced_to_running":
            attempts.append(
                RouteAttempt(
                    kind=kind,
                    cluster=spec.cluster,
                    est_start_seconds_raw=None,
                    est_start_seconds_clamped=None,
                    outcome="launched",
                    detail="cancel-race; job started during scancel",
                    elapsed_seconds=now_fn() - started_at,
                )
            )
            result = RouteResult(
                backend=backend,
                handle=handle,
                requested_kind=kind,
                chosen_kind=kind,
                reason=ROUTE_REASON_OVERRIDE,
                cluster=spec.cluster,
                attempts=attempts,
                elapsed_seconds=now_fn() - started_at,
                extra={"cancel_race": True},
            )
            _post_backend_selected(result, spec=spec, marker_poster=marker_poster)
            return result

        attempts.append(
            RouteAttempt(
                kind=kind,
                cluster=spec.cluster,
                est_start_seconds_raw=None,
                est_start_seconds_clamped=None,
                outcome=reason,
                detail=f"cancel_outcome={cancel_outcome}",
                elapsed_seconds=now_fn() - started_at,
            )
        )
        # On manual_attention the cancel did NOT confirm the job is dead.
        # We CANNOT silently escalate (would double-spend / collide on
        # attempt-id namespace) and the user explicitly asked for THIS
        # lane anyway — raise ManualAttentionRequiredError so the
        # orchestrator surfaces the orphaned job id.
        if cancel_outcome == "manual_attention":
            _post_terminal_failure_marker(
                spec=spec,
                marker_poster=marker_poster,
                reason=ROUTE_REASON_NO_COMPUTE,
                chosen_kind=kind,
                attempts=attempts,
            )
            raise ManualAttentionRequiredError(
                kind=kind,
                cluster=spec.cluster,
                orphaned_job_id=str(handle.job_id),
                attempts=[_attempt_to_dict(a) for a in attempts],
            )
        _post_terminal_failure_marker(
            spec=spec,
            marker_poster=marker_poster,
            reason=ROUTE_REASON_NO_COMPUTE,
            chosen_kind=kind,
            attempts=attempts,
        )
        raise NoComputeAvailableError(
            f"explicit override {kind!r} did not start within {cfg.free_wait_seconds}s "
            f"(park: {reason}, cancel: {cancel_outcome})",
            attempts=[_attempt_to_dict(a) for a in attempts],
        )


# ---------------------------------------------------------------------------
# Auto routing path
# ---------------------------------------------------------------------------


def _auto_route(
    *,
    spec: RunSpec,
    free_backends: dict[BackendKind, ComputeBackend],
    gcp_backend: ComputeBackend | None,
    store: LeaseStore,
    attempts: list[RouteAttempt],
    started_at: float,
    cfg: RouterConfig,
    lane_order: tuple[BackendKind, ...],
    is_started: Callable[[ComputeBackend, RunHandle], bool],
    is_live_after_cancel: Callable[[ComputeBackend, RunHandle], bool],
    is_running_after_cancel: Callable[[ComputeBackend, RunHandle], bool] | None,
    started_evidence_probe: (Callable[[ComputeBackend, RunHandle], dict[str, Any] | None] | None),
    mila_socket_alive: Callable[[], bool] | None,
    estimate_fn: Callable[[ComputeBackend, BackendKind, RunSpec], float | None] | None,
    reconnect_fn: (Callable[[ComputeBackend, BackendKind, RunSpec], RunHandle | None] | None),
    now_fn: Callable[[], float],
    sleep_fn: Callable[[float], None],
    marker_poster: Callable[..., None] | None,
    on_launched: Callable[[RunHandle], None] | None,
    clock_fn: Callable[[], datetime] | None,
) -> RouteResult:
    """No-``backend:`` auto route: walk ``lane_order`` (GCP-first default).

    GCP is a first-class auto lane, not only an escalation target: at
    its position in the order it is a single provision attempt (no
    queue estimate, no park). Contiguous SLURM lanes keep the existing
    est-start ranking + park + cancel chain among themselves. The
    terminal "everything failed" path stays
    :class:`NoComputeAvailableError`.
    """
    del clock_fn  # reserved for a future "day boundary at posted-time" override
    # Build the candidate list in lane order (skipping unwired lanes +
    # Mila-when-down + GCP-when-unwired).
    candidates: list[tuple[ComputeBackend, BackendKind]] = []
    for kind in lane_order:
        if kind == "gcp":
            if gcp_backend is not None:
                candidates.append((gcp_backend, "gcp"))
            continue
        backend = free_backends.get(kind)
        if backend is None:
            continue
        if kind == "mila" and (mila_socket_alive is None or not mila_socket_alive()):
            continue
        candidates.append((backend, kind))

    # Stage 1: reconnect scan over every wired lane, in lane order.
    reconnect_result = _try_auto_reconnect(
        spec=spec,
        candidates=candidates,
        store=store,
        attempts=attempts,
        started_at=started_at,
        reconnect_fn=reconnect_fn,
        now_fn=now_fn,
        marker_poster=marker_poster,
        on_launched=on_launched,
    )
    if reconnect_result is not None:
        return reconnect_result

    # Stage 2: walk the chain group by group. A GCP group is a single
    # provision attempt; a SLURM group is the ranked launch → park →
    # cancel-on-fail chain. ``terminal`` (last group) preserves the
    # legacy escalation semantics: when GCP sits LAST, its failures
    # raise the historical typed terminals instead of falling through.
    groups = _split_lane_groups([kind for _backend, kind in candidates])
    for group_idx, group in enumerate(groups):
        terminal = group_idx == len(groups) - 1
        if group == ("gcp",):
            gcp_result = _attempt_gcp_lane(
                spec=spec,
                gcp_backend=gcp_backend,
                store=store,
                attempts=attempts,
                started_at=started_at,
                cfg=cfg,
                now_fn=now_fn,
                marker_poster=marker_poster,
                on_launched=on_launched,
                terminal=terminal,
            )
            if gcp_result is not None:
                return gcp_result
            continue
        slurm_candidates = [(b, k) for b, k in candidates if k in group]
        estimated = _estimate_lanes(slurm_candidates, spec=spec, estimate_fn=estimate_fn)
        ranked = rank_lanes(estimated)
        free_result = _try_free_lanes(
            spec=spec,
            ranked=ranked,
            store=store,
            attempts=attempts,
            started_at=started_at,
            cfg=cfg,
            is_started=is_started,
            is_live_after_cancel=is_live_after_cancel,
            is_running_after_cancel=is_running_after_cancel,
            started_evidence_probe=started_evidence_probe,
            reconnect_fn=reconnect_fn,
            now_fn=now_fn,
            sleep_fn=sleep_fn,
            marker_poster=marker_poster,
            on_launched=on_launched,
        )
        if free_result is not None:
            return free_result

    # Terminal: every lane in the resolved order failed or was unwired /
    # unavailable. Post the breadcrumb the success path always posts,
    # then raise the typed no-compute terminal.
    last_kind: BackendKind = lane_order[-1]
    _post_terminal_failure_marker(
        spec=spec,
        marker_poster=marker_poster,
        reason=ROUTE_REASON_NO_COMPUTE,
        chosen_kind=last_kind,
        attempts=attempts,
    )
    raise NoComputeAvailableError(
        "every auto lane failed or was unavailable "
        f"(order: {' -> '.join(lane_order)}; wired: "
        f"{[kind for _b, kind in candidates] or 'none'})",
        attempts=[_attempt_to_dict(a) for a in attempts],
    )


def _try_auto_reconnect(
    *,
    spec: RunSpec,
    candidates: list[tuple[ComputeBackend, BackendKind]],
    store: LeaseStore,
    attempts: list[RouteAttempt],
    started_at: float,
    reconnect_fn: (Callable[[ComputeBackend, BackendKind, RunSpec], RunHandle | None] | None),
    now_fn: Callable[[], float],
    marker_poster: Callable[..., None] | None,
    on_launched: Callable[[RunHandle], None] | None = None,
) -> RouteResult | None:
    """Auto-route stage 1: look for an existing live job on every wired lane.

    ``candidates`` arrives in the RESOLVED lane order (GCP included at
    its position when wired), so the scan order matches the attempt
    order the marker trail reports.

    Reconnect probes are READ-ONLY (no lease writes) so they DON'T need
    to hold the per-issue flock — the flock is acquired by the lane that
    actually decides to submit, and that submit-path repeats the
    reconnect check inside the flock (so a job that appeared between
    this scan and the eventual launch is still caught).

    NO ``prepare()`` on any reconnect outcome here: ``SlurmBackend.
    prepare`` rsyncs the scratch dir with ``--delete`` and would yank
    code out from under the RUNNING job it just reconnected to.
    """
    del store  # not needed for reconnect probes; the launch path re-checks under the flock
    for backend, kind in candidates:
        # A probe failure here only skips the lock-free SCAN — the
        # submit path re-checks reconnect INSIDE the flock and a probe
        # failure THERE skips the lane (no blind submit), so swallowing
        # at this stage cannot cause a duplicate. (For GCP the launch
        # itself re-probes via reconnect_or_none.)
        lane_spec = _spec_for_lane(spec, kind)
        try:
            handle = _try_reconnect(
                backend=backend, kind=kind, spec=lane_spec, reconnect_fn=reconnect_fn
            )
        except BackendProbeError as exc:
            logger.warning(
                "route: reconnect scan probe failed for %s (%s); deferring to the "
                "in-flock re-check on the submit path.",
                kind,
                exc,
            )
            continue
        if handle is None:
            continue
        return _record_reconnect(
            backend=backend,
            kind=kind,
            cluster=lane_spec.cluster,
            handle=handle,
            spec=lane_spec,
            attempts=attempts,
            started_at=started_at,
            now_fn=now_fn,
            marker_poster=marker_poster,
            on_launched=on_launched,
            detail=(
                "found existing live gcp instance"
                if kind == "gcp"
                else "found existing live job/instance"
            ),
        )
    return None


def _record_reconnect(
    *,
    backend: ComputeBackend,
    kind: BackendKind,
    cluster: str | None,
    handle: RunHandle,
    spec: RunSpec,
    attempts: list[RouteAttempt],
    started_at: float,
    now_fn: Callable[[], float],
    marker_poster: Callable[..., None] | None,
    detail: str,
    on_launched: Callable[[RunHandle], None] | None = None,
) -> RouteResult:
    """Append a reconnect attempt + build the matching RouteResult.

    The persistence hook fires BEFORE the marker post so the handle is
    on disk by the time any observability side effect runs.
    """
    _invoke_on_launched(on_launched, handle)
    attempts.append(
        RouteAttempt(
            kind=kind,
            cluster=cluster,
            est_start_seconds_raw=None,
            est_start_seconds_clamped=None,
            outcome="reconnected",
            detail=detail,
            elapsed_seconds=now_fn() - started_at,
        )
    )
    result = RouteResult(
        backend=backend,
        handle=handle,
        requested_kind=None,
        chosen_kind=kind,
        reason=ROUTE_REASON_RECONNECT,
        cluster=cluster,
        attempts=attempts,
        elapsed_seconds=now_fn() - started_at,
    )
    _post_backend_selected(result, spec=spec, marker_poster=marker_poster)
    return result


def _try_free_lanes(
    *,
    spec: RunSpec,
    ranked: list[tuple[ComputeBackend, BackendKind, float | None, float]],
    store: LeaseStore,
    attempts: list[RouteAttempt],
    started_at: float,
    cfg: RouterConfig,
    is_started: Callable[[ComputeBackend, RunHandle], bool],
    is_live_after_cancel: Callable[[ComputeBackend, RunHandle], bool],
    is_running_after_cancel: Callable[[ComputeBackend, RunHandle], bool] | None,
    started_evidence_probe: (Callable[[ComputeBackend, RunHandle], dict[str, Any] | None] | None),
    reconnect_fn: (Callable[[ComputeBackend, BackendKind, RunSpec], RunHandle | None] | None),
    now_fn: Callable[[], float],
    sleep_fn: Callable[[float], None],
    marker_poster: Callable[..., None] | None,
    on_launched: Callable[[RunHandle], None] | None = None,
) -> RouteResult | None:
    """Auto-route stage 2: launch + park each ranked free lane, in order.

    Returns the first lane that resolves to RUNNING (or wins a
    cancel-race after park-fail). Returns ``None`` when EVERY lane in
    ``ranked`` fails to start — caller escalates to GCP.
    """
    for backend, kind, est_raw, est_clamped in ranked:
        result = _try_one_free_lane(
            spec=spec,
            backend=backend,
            kind=kind,
            est_raw=est_raw,
            est_clamped=est_clamped,
            store=store,
            attempts=attempts,
            started_at=started_at,
            cfg=cfg,
            is_started=is_started,
            is_live_after_cancel=is_live_after_cancel,
            is_running_after_cancel=is_running_after_cancel,
            started_evidence_probe=started_evidence_probe,
            reconnect_fn=reconnect_fn,
            marker_poster=marker_poster,
            on_launched=on_launched,
            now_fn=now_fn,
            sleep_fn=sleep_fn,
        )
        if result is not None:
            return result
    return None


def _try_one_free_lane(
    *,
    spec: RunSpec,
    backend: ComputeBackend,
    kind: BackendKind,
    est_raw: float | None,
    est_clamped: float | None,
    store: LeaseStore,
    attempts: list[RouteAttempt],
    started_at: float,
    cfg: RouterConfig,
    is_started: Callable[[ComputeBackend, RunHandle], bool],
    is_live_after_cancel: Callable[[ComputeBackend, RunHandle], bool],
    is_running_after_cancel: Callable[[ComputeBackend, RunHandle], bool] | None,
    started_evidence_probe: (Callable[[ComputeBackend, RunHandle], dict[str, Any] | None] | None),
    reconnect_fn: (Callable[[ComputeBackend, BackendKind, RunSpec], RunHandle | None] | None),
    marker_poster: Callable[..., None] | None,
    now_fn: Callable[[], float],
    sleep_fn: Callable[[float], None],
    on_launched: Callable[[RunHandle], None] | None = None,
) -> RouteResult | None:
    """Launch + park one free lane. Returns a RouteResult on success / cancel-race.

    Returns ``None`` to signal "next lane". Cancel-race during park-fail
    is treated as success (the job won; tearing it down would forfeit
    the wait we already paid for).

    Lock discipline: the per-issue flock is held across (re-check
    reconnect → launch → lease-write → park → cancel) so a concurrent
    invocation cannot slip a parallel submit between our reconnect probe
    and our launch. If the cancel state machine returns ``manual_attention``
    (cancel did NOT confirm the job is dead), we RAISE
    :class:`ManualAttentionRequiredError` rather than returning ``None``
    — silently escalating would risk a second copy of the same workload
    in the GCP escalation path (the orphaned free-lane job is unconfirmed
    dead and may still consume the attempt-id namespace).
    """
    spec = _spec_for_lane(spec, kind)
    with store.transaction(spec.issue) as (lease, write):
        # Repeat the reconnect check INSIDE the flock — a parallel
        # invocation may have submitted between the lock-free scan in
        # _try_auto_reconnect and now. NO prepare() on reconnect:
        # SlurmBackend.prepare rsyncs the scratch dir with --delete and
        # would yank code out from under the RUNNING job.
        #
        # A PROBE failure (BackendProbeError — transport down, NOT "no
        # live job") must not fall through to a blind fresh submit on
        # THIS lane (a live job may exist; prepare()'s --delete rsync +
        # a second sbatch would corrupt / duplicate it — round-6 B1).
        # Auto chain → skip the lane, try the next one.
        try:
            handle = _try_reconnect(
                backend=backend, kind=kind, spec=spec, reconnect_fn=reconnect_fn
            )
        except BackendProbeError as exc:
            attempts.append(
                RouteAttempt(
                    kind=kind,
                    cluster=spec.cluster,
                    est_start_seconds_raw=est_raw,
                    est_start_seconds_clamped=est_clamped,
                    outcome="reconnect_probe_failed",
                    detail=f"{type(exc).__name__}: {exc}",
                    elapsed_seconds=now_fn() - started_at,
                )
            )
            logger.warning(
                "route: free lane %s reconnect probe failed (%s); skipping lane "
                "(cannot verify whether a live job exists — submitting blind risks "
                "a duplicate).",
                kind,
                exc,
            )
            return None
        if handle is not None:
            _invoke_on_launched(on_launched, handle)
            attempts.append(
                RouteAttempt(
                    kind=kind,
                    cluster=spec.cluster,
                    est_start_seconds_raw=est_raw,
                    est_start_seconds_clamped=est_clamped,
                    outcome="reconnected",
                    detail="reconnect inside flock — concurrent invocation submitted",
                    elapsed_seconds=now_fn() - started_at,
                )
            )
            result = RouteResult(
                backend=backend,
                handle=handle,
                requested_kind=None,
                chosen_kind=kind,
                reason=ROUTE_REASON_RECONNECT,
                cluster=spec.cluster,
                attempts=attempts,
                elapsed_seconds=now_fn() - started_at,
            )
            _post_backend_selected(result, spec=spec, marker_poster=marker_poster)
            return result

        # Launch (still under the flock — sealing the double-submit race).
        threaded_spec, lease = _thread_attempt_id_into(spec, lease, write)
        try:
            handle = _prepare_and_launch(backend, threaded_spec, kind=kind, cluster=spec.cluster)
        except BackendPrepareError as exc:
            # Provision-class, pre-launch (nothing live) → next lane,
            # same semantics as a launch failure but with a precise
            # attempt-trail outcome.
            attempts.append(
                RouteAttempt(
                    kind=kind,
                    cluster=spec.cluster,
                    est_start_seconds_raw=est_raw,
                    est_start_seconds_clamped=est_clamped,
                    outcome="prepare_failed",
                    detail=exc.reason,
                    elapsed_seconds=now_fn() - started_at,
                )
            )
            logger.warning(
                "route: free lane %s prepare failed (%s); trying next lane.",
                kind,
                exc.reason,
            )
            return None
        except Exception as exc:
            attempts.append(
                RouteAttempt(
                    kind=kind,
                    cluster=spec.cluster,
                    est_start_seconds_raw=est_raw,
                    est_start_seconds_clamped=est_clamped,
                    outcome="launch_failed",
                    detail=f"{type(exc).__name__}: {exc}",
                    elapsed_seconds=now_fn() - started_at,
                )
            )
            logger.warning(
                "route: free lane %s launch failed (%s); trying next lane.",
                kind,
                type(exc).__name__,
            )
            return None

        # Persist the handle (sidecar hook) + launched id IMMEDIATELY
        # (still under the flock).
        _invoke_on_launched(on_launched, handle)
        write(_lease_after_submit(lease, spec, kind, spec.cluster, handle))

        # Park (still under the flock — wait IS contention surface, but
        # the lock is per-ISSUE, not cross-issue, so the only callers
        # serialized are the two we are deliberately serializing).
        started, reason, terminal_status = park_until_running_or_cap(
            backend=backend,
            handle=handle,
            is_started=is_started,
            cap_seconds=cfg.free_wait_seconds,
            poll_interval=cfg.poll_interval,
            now_fn=now_fn,
            sleep_fn=sleep_fn,
        )
        if started:
            return _record_free_lane_started(
                backend=backend,
                handle=handle,
                kind=kind,
                est_raw=est_raw,
                est_clamped=est_clamped,
                spec=spec,
                attempts=attempts,
                started_at=started_at,
                now_fn=now_fn,
                marker_poster=marker_poster,
                detail="park resolved to RUNNING",
            )

        # Park failed. Distinguish "never started" from "started and
        # FAILED": a fast-failing job transitions PD→R→exit between
        # polls, so "vanished before observed RUNNING" is NOT proof the
        # cluster lacked capacity. If the scratch dir holds runtime
        # artifacts (status.json / job.out), the job DID start — a
        # WORKLOAD failure that must SURFACE (no GCP escalation: a
        # workload bug would burn paid credit on a doomed re-run).
        #
        # GATED on the job being genuinely GONE (done/dead): ``stalled``
        # covers LIVE jobs (RUNNING + stale heartbeat; SUSPENDED) and
        # ``gate`` is a live wait — classifying those here raised BEFORE
        # the cancel machine and orphaned a live job (round-6 M1, issue
        # 535 attempt 2). stalled/gate fall through to cancel_and_wait.
        if reason == "terminal_before_running" and terminal_status in ("done", "dead"):
            evidence = _probe_started_evidence(started_evidence_probe, backend, handle)
            if evidence is not None:
                attempts.append(
                    RouteAttempt(
                        kind=kind,
                        cluster=spec.cluster,
                        est_start_seconds_raw=est_raw,
                        est_start_seconds_clamped=est_clamped,
                        outcome="workload_failure",
                        detail=(
                            "terminal before RUNNING with runtime artifacts "
                            f"(phase={evidence.get('phase', '')!r})"
                        ),
                        elapsed_seconds=now_fn() - started_at,
                    )
                )
                _post_terminal_failure_marker(
                    spec=spec,
                    marker_poster=marker_poster,
                    reason=ROUTE_REASON_WORKLOAD_FAILURE,
                    chosen_kind=kind,
                    attempts=attempts,
                    extra={"evidence": evidence},
                )
                raise WorkloadSurfacedError(
                    f"{kind} job {handle.job_id} went terminal before RUNNING but "
                    f"left runtime artifacts (phase={evidence.get('phase', '')!r}) — "
                    "workload failure, no auto-fallback",
                    chosen_kind=kind,
                    evidence=evidence,
                )

        # Genuine never-started park failure → cancel state machine,
        # then KEEP (raced), CONTINUE to next lane (cancelled), or
        # RAISE (manual_attention).
        cancel_outcome = cancel_and_wait(
            backend=backend,
            handle=handle,
            is_live_after_cancel=is_live_after_cancel,
            is_running_after_cancel=is_running_after_cancel,
            grace_seconds=cfg.cancel_grace_seconds,
            poll_interval=min(2.0, cfg.poll_interval),
            now_fn=now_fn,
            sleep_fn=sleep_fn,
        )
        if cancel_outcome == "raced_to_running":
            return _record_free_lane_started(
                backend=backend,
                handle=handle,
                kind=kind,
                est_raw=est_raw,
                est_clamped=est_clamped,
                spec=spec,
                attempts=attempts,
                started_at=started_at,
                now_fn=now_fn,
                marker_poster=marker_poster,
                detail="cancel-race; job started during scancel",
                extra={"cancel_race": True},
            )

        attempts.append(
            RouteAttempt(
                kind=kind,
                cluster=spec.cluster,
                est_start_seconds_raw=est_raw,
                est_start_seconds_clamped=est_clamped,
                outcome=reason,
                detail=f"cancel_outcome={cancel_outcome}",
                elapsed_seconds=now_fn() - started_at,
            )
        )
        if cancel_outcome == "manual_attention":
            # cancel grace expired without confirming the free-lane job
            # is dead. Silently escalating to GCP would risk a duplicate
            # run sharing the attempt-id namespace → raise so the
            # orchestrator surfaces the orphaned id + parks.
            _post_terminal_failure_marker(
                spec=spec,
                marker_poster=marker_poster,
                reason=ROUTE_REASON_NO_COMPUTE,
                chosen_kind=kind,
                attempts=attempts,
                extra={"manual_attention": True, "orphaned_job_id": str(handle.job_id)},
            )
            raise ManualAttentionRequiredError(
                kind=kind,
                cluster=spec.cluster,
                orphaned_job_id=str(handle.job_id),
                attempts=[_attempt_to_dict(a) for a in attempts],
            )
        return None


def _record_free_lane_started(
    *,
    backend: ComputeBackend,
    handle: RunHandle,
    kind: BackendKind,
    est_raw: float | None,
    est_clamped: float | None,
    spec: RunSpec,
    attempts: list[RouteAttempt],
    started_at: float,
    now_fn: Callable[[], float],
    marker_poster: Callable[..., None] | None,
    detail: str,
    extra: dict[str, Any] | None = None,
) -> RouteResult:
    """Append a "launched" attempt + build the matching auto-started RouteResult."""
    attempts.append(
        RouteAttempt(
            kind=kind,
            cluster=spec.cluster,
            est_start_seconds_raw=est_raw,
            est_start_seconds_clamped=est_clamped,
            outcome="launched",
            detail=detail,
            elapsed_seconds=now_fn() - started_at,
        )
    )
    result = RouteResult(
        backend=backend,
        handle=handle,
        requested_kind=None,
        chosen_kind=kind,
        reason=ROUTE_REASON_AUTO_STARTED,
        cluster=spec.cluster,
        attempts=attempts,
        elapsed_seconds=now_fn() - started_at,
        extra=extra or {},
    )
    _post_backend_selected(result, spec=spec, marker_poster=marker_poster)
    return result


def _attempt_gcp_lane(
    *,
    spec: RunSpec,
    gcp_backend: ComputeBackend | None,
    store: LeaseStore,
    attempts: list[RouteAttempt],
    started_at: float,
    cfg: RouterConfig,
    now_fn: Callable[[], float],
    marker_poster: Callable[..., None] | None,
    on_launched: Callable[[RunHandle], None] | None = None,
    terminal: bool = True,
) -> RouteResult | None:
    """Attempt the GCP lane at its position in the resolved auto order.

    ``terminal=True`` (GCP is the LAST wired lane — the legacy
    escalation position) keeps the historical typed-terminal semantics:
    raises :class:`NoComputeAvailableError` on a provisioning / prepare /
    state-probe failure and :class:`GcpAttemptCapExceededError` at the
    per-day attempt cap.

    ``terminal=False`` (lanes remain after GCP — e.g. the GCP-first
    standing default) turns provision-class failures (prepare /
    provisioning-capacity / state-probe) AND the attempt-cap guard into
    "continue down the order": the attempt is recorded and ``None`` is
    returned so the router tries the next lane. ONLY a workload failure
    (:class:`GcpWorkloadError`) still raises in EVERY position — broken
    workload code must not cascade across lanes and burn queue time.

    The per-day attempt counter counts primary-lane attempts the same
    as escalation attempts (the guard bounds the NUMBER of provision
    attempts so a broken classifier loop can't burn credit; primary-lane
    attempts carry the same risk). It remains an attempt-COUNT guard,
    never a dollar cap (``tests/test_no_dollar_budget_caps.py``).

    Lock discipline: bump-counter / cap-check / threaded-attempt-id /
    launch / persist all live inside ONE :meth:`LeaseStore.transaction`
    so a concurrent invocation cannot read a pre-bump counter, decide
    "we're under cap", and double-spend credit.
    """
    if gcp_backend is None:
        # Only reachable from the legacy terminal call shape — the auto
        # chain filters an unwired GCP out of the candidates, so this is
        # belt-and-suspenders for direct callers.
        if not terminal:
            return None
        _post_terminal_failure_marker(
            spec=spec,
            marker_poster=marker_poster,
            reason=ROUTE_REASON_NO_COMPUTE,
            chosen_kind="gcp",
            attempts=attempts,
        )
        raise NoComputeAvailableError(
            "every free lane park-failed AND no gcp_backend wired for auto-fallback",
            attempts=[_attempt_to_dict(a) for a in attempts],
        )

    # Pre-create regional-quota headroom check (#608): four guaranteed-fail
    # creates burned the per-day attempt cap against an exhausted regional
    # accelerator quota (NVIDIA_A100_80GB_GPUS at 8/8 with 4 needed). When
    # the probe POSITIVELY reports insufficient headroom, skip the lane
    # loudly WITHOUT bumping the attempt counter — the cap bounds provision
    # attempts, and a create that cannot succeed should not consume one.
    # FAIL-OPEN: a probe failure, a backend without the probe (test
    # doubles), or a live reconnectable instance (no new quota needed)
    # returns None → proceed exactly as before. The explicit ``backend:
    # gcp`` override lane deliberately does NOT pre-check: it never bumps
    # the cap, and an explicit ask should attempt (the create error now
    # carries its stderr tail either way).
    headroom = _gcp_quota_headroom_or_none(gcp_backend, spec)
    if headroom is not None and not headroom.sufficient:
        detail = (
            f"regional accelerator quota {headroom.metric} in {headroom.region} has "
            f"usage {headroom.usage:g}/{headroom.limit:g} — headroom "
            f"{headroom.available:g} GPU(s) < needed {headroom.needed}; skipping the "
            "GCP lane without burning a daily attempt"
        )
        attempts.append(
            RouteAttempt(
                kind="gcp",
                cluster=None,
                est_start_seconds_raw=0.0,
                est_start_seconds_clamped=0.0,
                outcome="quota_headroom_insufficient",
                detail=detail,
                elapsed_seconds=now_fn() - started_at,
            )
        )
        if not terminal:
            logger.warning(
                "route: GCP quota headroom insufficient for issue %d (%s); "
                "continuing down the lane order.",
                spec.issue,
                detail,
            )
            return None
        _post_terminal_failure_marker(
            spec=spec,
            marker_poster=marker_poster,
            reason=ROUTE_REASON_NO_COMPUTE,
            chosen_kind="gcp",
            attempts=attempts,
        )
        raise NoComputeAvailableError(
            f"every free lane park-failed AND the GCP regional quota has no headroom: {detail}",
            attempts=[_attempt_to_dict(a) for a in attempts],
        )

    with store.transaction(spec.issue) as (lease, write):
        # Cap-check BEFORE bump-and-persist: a rejected over-cap attempt
        # MUST NOT grow the on-disk counter (3, 4, 5, ... with cap=2 is
        # misleading and makes the counter unbounded under a broken
        # classifier that loops). Rollover-on-day-change is part of the
        # cap probe so a fresh UTC day correctly admits the new attempt.
        if lease is None:
            lease = Lease(
                issue=int(spec.issue),
                spec_hash=spec_hash(spec),
                attempt_id=_make_attempt_id(),
            )
        today = _today_utc_iso()
        attempts_already_today = lease.gcp_attempts_today if lease.gcp_attempts_date == today else 0
        if attempts_already_today >= cfg.max_gcp_attempts_per_day:
            if not terminal:
                # Lanes remain after GCP → skip GCP (no credit spent)
                # and continue down the order instead of bricking the
                # whole route for the day. The cap still bounds spend:
                # no provision attempt is made here.
                attempts.append(
                    RouteAttempt(
                        kind="gcp",
                        cluster=None,
                        est_start_seconds_raw=0.0,
                        est_start_seconds_clamped=0.0,
                        outcome="attempt_cap_exceeded",
                        detail=(
                            f"per-day GCP attempt cap {cfg.max_gcp_attempts_per_day} "
                            "reached; skipping GCP, continuing down the lane order"
                        ),
                        elapsed_seconds=now_fn() - started_at,
                    )
                )
                logger.warning(
                    "route: per-day GCP attempt cap (%d) reached for issue %d; "
                    "skipping the GCP lane and continuing down the auto order.",
                    cfg.max_gcp_attempts_per_day,
                    spec.issue,
                )
                return None
            raise GcpAttemptCapExceededError(
                issue=int(spec.issue),
                # Report attempts ALREADY consumed today (i.e. the cap),
                # not the would-be-Nth-attempt that this call would have
                # made — reads naturally as "cap reached, no further".
                attempts_today=cfg.max_gcp_attempts_per_day,
                cap=cfg.max_gcp_attempts_per_day,
            )
        # Under the cap → bump + persist (rollover folded into the bump).
        lease = _bump_gcp_attempt(lease)
        write(lease)
        attempts_today = lease.gcp_attempts_today

        # Pre-escalation marker — visible breadcrumb before spending
        # credit. Posted INSIDE the flock so a concurrent invocation
        # cannot also post one (they would block on the flock until our
        # launch completes).
        _post_intermediate_marker(
            spec=spec,
            marker_poster=marker_poster,
            reason=ROUTE_REASON_AUTO_FALLBACK_GCP,
            attempts_today=attempts_today,
        )

        threaded_spec, lease = _thread_attempt_id_into(spec, lease, write)
        try:
            gcp_handle = _prepare_and_launch(gcp_backend, threaded_spec, kind="gcp")
        except BackendPrepareError as exc:
            # GcpBackend.prepare is a documented no-op today, so this is
            # belt-and-suspenders for the uniform chokepoint: a prepare
            # failure is provision-class (nothing live) → same terminal
            # as a GCP provisioning failure (or next-lane when lanes
            # remain after GCP).
            attempts.append(
                RouteAttempt(
                    kind="gcp",
                    cluster=None,
                    est_start_seconds_raw=0.0,
                    est_start_seconds_clamped=0.0,
                    outcome="prepare_failed",
                    detail=exc.reason,
                    elapsed_seconds=now_fn() - started_at,
                )
            )
            if not terminal:
                logger.warning(
                    "route: gcp prepare failed (%s); continuing down the lane order.",
                    exc.reason,
                )
                return None
            _post_terminal_failure_marker(
                spec=spec,
                marker_poster=marker_poster,
                reason=ROUTE_REASON_NO_COMPUTE,
                chosen_kind="gcp",
                attempts=attempts,
            )
            raise NoComputeAvailableError(
                f"every free lane park-failed AND gcp prepare failed: {exc.reason}",
                attempts=[_attempt_to_dict(a) for a in attempts],
            ) from exc
        except GcpProvisioningError as exc:
            attempts.append(
                RouteAttempt(
                    kind="gcp",
                    cluster=None,
                    est_start_seconds_raw=0.0,
                    est_start_seconds_clamped=0.0,
                    outcome="provisioning_failure",
                    detail=_provisioning_detail(exc),
                    elapsed_seconds=now_fn() - started_at,
                )
            )
            if not terminal:
                # Capacity / quota / zone exhaustion at the primary GCP
                # position → fall through to the lanes after it.
                logger.warning(
                    "route: gcp provisioning failed (%s); continuing down the lane order.",
                    _provisioning_detail(exc),
                )
                return None
            _post_terminal_failure_marker(
                spec=spec,
                marker_poster=marker_poster,
                reason=ROUTE_REASON_NO_COMPUTE,
                chosen_kind="gcp",
                attempts=attempts,
            )
            raise NoComputeAvailableError(
                f"every free lane park-failed AND gcp provisioning failed: {exc.reason}",
                attempts=[_attempt_to_dict(a) for a in attempts],
            ) from exc
        except BackendProbeError as exc:
            # GcpBackend.launch's internal reconnect_or_none probe failed
            # (expired auth / transport) — GCP state UNKNOWN. No credit is
            # spent on unknown state in either position. Terminal position:
            # fail closed with the typed no-compute terminal instead of
            # letting the probe error propagate to rc=4 (live auto-lane
            # finding, issue 535). Non-terminal position: skip the lane and
            # continue — the same safe reaction the SLURM lanes take on an
            # unprobeable reconnect (the stage-1 GCP scan already treats a
            # probe failure as continue-down-the-chain).
            attempts.append(
                RouteAttempt(
                    kind="gcp",
                    cluster=None,
                    est_start_seconds_raw=0.0,
                    est_start_seconds_clamped=0.0,
                    outcome="probe_failed",
                    detail=str(exc)[:500],
                    elapsed_seconds=now_fn() - started_at,
                )
            )
            if not terminal:
                logger.warning(
                    "route: gcp state probe failed (%s); skipping GCP, continuing "
                    "down the lane order.",
                    exc,
                )
                return None
            _post_terminal_failure_marker(
                spec=spec,
                marker_poster=marker_poster,
                reason=ROUTE_REASON_NO_COMPUTE,
                chosen_kind="gcp",
                attempts=attempts,
            )
            raise NoComputeAvailableError(
                f"every free lane park-failed AND the gcp state probe failed — "
                f"refusing blind create: {exc}",
                attempts=[_attempt_to_dict(a) for a in attempts],
            ) from exc
        except GcpWorkloadError as exc:
            attempts.append(
                RouteAttempt(
                    kind="gcp",
                    cluster=None,
                    est_start_seconds_raw=0.0,
                    est_start_seconds_clamped=0.0,
                    outcome="workload_failure",
                    detail=exc.reason,
                    elapsed_seconds=now_fn() - started_at,
                )
            )
            _post_terminal_failure_marker(
                spec=spec,
                marker_poster=marker_poster,
                reason=ROUTE_REASON_WORKLOAD_FAILURE,
                chosen_kind="gcp",
                attempts=attempts,
                extra={"evidence": exc.evidence},
            )
            raise WorkloadSurfacedError(
                f"gcp workload failure (no auto-fallback): {exc.reason}",
                chosen_kind="gcp",
                evidence=exc.evidence,
            ) from exc

        # Persist the handle (sidecar hook) + launched id IMMEDIATELY
        # (still under the flock).
        _invoke_on_launched(on_launched, gcp_handle)
        write(_lease_after_submit(lease, spec, "gcp", None, gcp_handle))

    attempts.append(
        RouteAttempt(
            kind="gcp",
            cluster=None,
            est_start_seconds_raw=0.0,
            est_start_seconds_clamped=0.0,
            outcome="launched",
            detail=(
                f"gcp escalation #{attempts_today} of cap {cfg.max_gcp_attempts_per_day}"
                if terminal
                else (
                    f"gcp primary-lane attempt #{attempts_today} of cap "
                    f"{cfg.max_gcp_attempts_per_day}"
                )
            ),
            elapsed_seconds=now_fn() - started_at,
        )
    )
    result = RouteResult(
        backend=gcp_backend,
        handle=gcp_handle,
        requested_kind=None,
        chosen_kind="gcp",
        # Reason code kept as ``auto_fallback_gcp`` in EVERY auto position
        # (including GCP-first primary) — the marker schema is unchanged
        # by deliberate design (dashboard + acceptance harness pattern-
        # match on the enumerated reason codes); the attempts trail is
        # what reflects the actual order.
        reason=ROUTE_REASON_AUTO_FALLBACK_GCP,
        cluster=None,
        attempts=attempts,
        elapsed_seconds=now_fn() - started_at,
        # Additive marker fields (#631): which provisioning model + regional
        # quota pool this launch resolved to. STANDARD|SPOT|FLEX_START and
        # the matching {on-demand|preemptible} accelerator metric (or None
        # when the (gpu_kind, pool) pair has no mapping). Documented as
        # optional `extra` keys in workflow.yaml § markers.
        extra={
            "gcp_attempts_today": attempts_today,
            "provisioning_model": resolve_provisioning_model(spec),
            "quota_pool": quota_metric_for(
                machine_for_intent(spec), resolve_provisioning_model(spec)
            ),
        },
    )
    _post_backend_selected(result, spec=spec, marker_poster=marker_poster)
    return result


# ---------------------------------------------------------------------------
# Internal helpers (lease, estimate, reconnect, marker)
# ---------------------------------------------------------------------------


def _estimate_lanes(
    candidates: Iterable[tuple[ComputeBackend, BackendKind]],
    *,
    spec: RunSpec,
    estimate_fn: Callable[[ComputeBackend, BackendKind, RunSpec], float | None] | None,
) -> list[tuple[ComputeBackend, BackendKind, float | None]]:
    """Probe each candidate's est-start; return as ``(backend, kind, raw)`` triples.

    Default ``estimate_fn`` calls
    ``backend.estimate_start_seconds(spec)`` when the backend exposes
    the method (SLURM does), else returns None. The router treats
    ``None`` as "unranked but park-eligible".
    """
    triples: list[tuple[ComputeBackend, BackendKind, float | None]] = []
    fn = estimate_fn or _default_estimate
    for backend, kind in candidates:
        try:
            raw = fn(backend, kind, _spec_for_lane(spec, kind))
        except Exception as exc:
            logger.warning(
                "route: estimate_fn raised for %s (%s: %s); treating as unranked.",
                kind,
                type(exc).__name__,
                exc,
            )
            raw = None
        triples.append((backend, kind, raw))
    return triples


def _default_estimate(backend: ComputeBackend, kind: BackendKind, spec: RunSpec) -> float | None:
    """Fall back to ``backend.estimate_start_seconds(spec)`` when present."""
    del kind
    fn = getattr(backend, "estimate_start_seconds", None)
    if fn is None:
        return None
    return fn(spec)


def _try_reconnect(
    *,
    backend: ComputeBackend,
    kind: BackendKind,
    spec: RunSpec,
    reconnect_fn: (Callable[[ComputeBackend, BackendKind, RunSpec], RunHandle | None] | None),
) -> RunHandle | None:
    """Look for an existing live job/instance for ``spec`` on ``backend``.

    Backend-aware reconnect lives in the backend itself (SLURM:
    ``squeue --name eps-issue-<N>``; GCP: :func:`gcp.reconnect_or_none`).
    The injected ``reconnect_fn`` wraps that — production-default
    (slice 6) wires per-backend probes; tests pass ``None`` to disable
    reconnect entirely.

    When the lease has an ``UNKNOWN_SUBMITTED`` recovery state (lease
    present but no job_id — submit returned but the orchestrator
    crashed before persisting), we ALSO call the reconnect_fn — the
    backend's queue may show the job even though we never recorded its
    id locally. This is the slice-5 "UNKNOWN_SUBMITTED" recovery hook.

    :class:`BackendProbeError` PROPAGATES (round-6 B1): it means the
    probe itself failed (transport down) — "couldn't ask" treated as
    "no live job" lets the caller submit a blind duplicate. Each call
    site decides the safe reaction (skip the lane on the auto chain;
    typed terminal on an explicit override).
    """
    if reconnect_fn is None:
        return None
    try:
        handle = reconnect_fn(backend, kind, spec)
    except BackendProbeError:
        raise
    except Exception as exc:
        logger.warning(
            "route: reconnect_fn raised for %s (%s: %s); treating as no live job.",
            kind,
            type(exc).__name__,
            exc,
        )
        return None
    if handle is None:
        return None
    # Defensive: a reconnect_fn that returns a handle for the WRONG
    # issue would silently bind to someone else's run. Sanity-check.
    if handle.extra.get("issue") not in (None, int(spec.issue)):
        logger.error(
            "route: reconnect_fn for %s returned a handle for issue=%r (expected %d); ignoring.",
            kind,
            handle.extra.get("issue"),
            spec.issue,
        )
        return None
    # Defensive: a misconfigured reconnect_fn that binds to the WRONG
    # backend kind (e.g. a GCP probe wired into the nibi slot) would
    # silently re-attach to someone else's lane. The handle carries the
    # backend it was issued by; cross-check.
    #
    # Production SLURM handles carry ``backend="cluster"`` with the
    # concrete lane in ``handle.cluster`` — both ``SlurmBackend.launch``
    # and the dispatch CLI's reconnect closure return that shape (round-2
    # Codex Critical, task #535: requiring ``handle.backend == kind``
    # here rejected EVERY live production SLURM reconnect handle, so
    # ``route()`` could fresh-submit a duplicate job on a lane that
    # already had one). Accept the ``"cluster"`` alias only when the
    # concrete cluster matches the lane being probed; a cluster handle
    # for a DIFFERENT cluster (or with no cluster at all) is still the
    # cross-lane mismatch this guard exists for.
    if handle.backend != kind and not (handle.backend == "cluster" and handle.cluster == kind):
        logger.error(
            "route: reconnect_fn for kind=%s returned a handle issued by "
            "backend=%s (cluster=%s); ignoring.",
            kind,
            handle.backend,
            handle.cluster,
        )
        return None
    return handle


def _lease_after_submit(
    lease: Lease | None,
    spec: RunSpec,
    backend_kind: BackendKind,
    cluster: str | None,
    handle: RunHandle,
) -> Lease:
    """Pure helper: produce the lease record that records a fresh submit.

    Used inside an OPEN ``store.transaction`` so the read-check + launch +
    lease-write all hold the same flock (the read happened when the
    caller opened the transaction; this returns the new value the caller
    will hand to ``write_fn``). Pre-existing GCP attempt counter +
    spec_hash + attempt_id fields are preserved on ``lease``; absent
    lease → fresh one with the spec's attempt_id (or a freshly minted
    one if none).
    """
    if lease is None:
        lease = Lease(
            issue=int(spec.issue),
            spec_hash=spec_hash(spec),
            attempt_id=str(spec.extra.get("attempt_id") or _make_attempt_id()),
        )
    lease.backend = backend_kind
    lease.cluster = cluster
    lease.job_id = str(handle.job_id)
    lease.submitted_at = float(time.time())  # wall-clock, not monotonic
    return lease


def _persist_lease_after_submit(
    *,
    spec: RunSpec,
    store: LeaseStore,
    backend_kind: BackendKind,
    cluster: str | None,
    handle: RunHandle,
    now_fn: Callable[[], float],
) -> None:
    """Open a flocked transaction + write the lease after a submit.

    Crash window covered: a submit that returns successfully but the
    orchestrator dies before the lease is updated would otherwise leave
    a leaked job / instance. Prefer the in-transaction
    :func:`_lease_after_submit` helper when the caller is ALREADY inside
    a transaction (the override / auto-route paths hold the flock across
    reconnect-check → launch → lease-write to seal the double-submit
    race).
    """
    del now_fn  # monotonic clock is for the watchdog, not the lease timestamp
    with store.transaction(spec.issue) as (lease, write):
        write(_lease_after_submit(lease, spec, backend_kind, cluster, handle))


def _thread_attempt_id(spec: RunSpec, store: LeaseStore) -> RunSpec:
    """Ensure ``spec.extra["attempt_id"]`` is set + matches the lease.

    Convenience wrapper that opens its own transaction; used ONLY when
    the caller is NOT already inside a transaction (the override+auto
    paths now hold one flock across reconnect-check → launch → lease-write
    and use :func:`_thread_attempt_id_into` instead to avoid the re-entry
    deadlock — :py:func:`fcntl.flock` from a fresh open-file-description
    in the same process blocks against any held lock).
    """
    current_id = (spec.extra or {}).get("attempt_id")
    with store.transaction(spec.issue) as (lease, write):
        if lease is None:
            attempt_id = str(current_id or _make_attempt_id())
            lease = Lease(
                issue=int(spec.issue),
                spec_hash=spec_hash(spec),
                attempt_id=attempt_id,
            )
            write(lease)
        else:
            attempt_id = lease.attempt_id

    # RunSpec is frozen; replace ``extra`` with a new dict carrying the id.
    new_extra = dict(spec.extra or {})
    new_extra["attempt_id"] = attempt_id
    return replace(spec, extra=new_extra)


def _thread_attempt_id_into(
    spec: RunSpec,
    lease: Lease | None,
    write_fn: Callable[[Lease], None],
) -> tuple[RunSpec, Lease]:
    """Same contract as :func:`_thread_attempt_id` but reuses an OPEN transaction.

    Returns ``(new_spec, lease)`` where ``new_spec`` carries the threaded
    ``attempt_id`` in ``extra``, and ``lease`` is the (possibly freshly
    created) lease record. If lease was None, a fresh one is written via
    ``write_fn`` — the caller's transaction owns the flock.
    """
    current_id = (spec.extra or {}).get("attempt_id")
    if lease is None:
        attempt_id = str(current_id or _make_attempt_id())
        lease = Lease(
            issue=int(spec.issue),
            spec_hash=spec_hash(spec),
            attempt_id=attempt_id,
        )
        write_fn(lease)
    else:
        attempt_id = lease.attempt_id
    new_extra = dict(spec.extra or {})
    new_extra["attempt_id"] = attempt_id
    return replace(spec, extra=new_extra), lease


def _make_attempt_id() -> str:
    """Per-attempt id — same shape the GCP backend's ``attempt_id_for`` produces."""
    return f"att-{datetime.now(tz=UTC).strftime('%Y%m%d-%H%M%S')}"


def _post_marker_nonfatal(
    marker_poster: Callable[..., None],
    *,
    issue: int,
    note: str,
    context: str,
) -> None:
    """Invoke ``marker_poster``; NEVER let a marker-post failure alter routing.

    Every router ``epm:backend-selected`` post fires either AFTER a
    successful launch (the success breadcrumb — live infra in hand) or
    immediately BEFORE raising a typed terminal (the failure
    breadcrumb). A raise from the poster itself (e.g.
    ``post_marker_via_task_py``'s ``subprocess.run(check=True,
    timeout=30)`` hitting flock contention) would either convert
    "launched, handle in hand" into an unclassified dispatch-CLI rc=4
    with a live, billing VM/job, or clobber the typed terminal the
    orchestrator's failure-classifier routes on. Markers are an
    observability side channel, not control flow — failures are logged
    LOUD (ERROR + the full payload, never silently swallowed) and the
    route continues.
    """
    try:
        marker_poster(
            issue=issue,
            marker="epm:backend-selected",
            note=note,
            version=1,
            by="backends.router",
        )
    except Exception:
        logger.exception(
            "route: epm:backend-selected marker post FAILED (%s) for issue=%d; "
            "continuing — markers must never alter routing control flow. payload=%s",
            context,
            issue,
            note,
        )


def _post_backend_selected(
    result: RouteResult,
    *,
    spec: RunSpec,
    marker_poster: Callable[..., None] | None,
) -> None:
    """Post ``epm:backend-selected v1`` with the EXTENDED router body.

    Extended fields beyond the selector's schema (see workflow.yaml §
    markers):

    * ``attempts`` — list of per-lane attempt records (raw + clamped
      est-start, outcome, detail, elapsed), appended chronologically so
      the trail reflects the ACTUAL attempt order (GCP first when the
      GCP-first default ran it first).
    * Existing schema preserved: ``requested_kind`` / ``chosen_kind`` /
      ``reason`` / ``cluster`` / ``elapsed_seconds`` / ``extra``.

    Non-fatal: every call site runs AFTER a successful launch /
    reconnect, so a poster failure must never propagate past live infra
    (see :func:`_post_marker_nonfatal`).
    """
    if marker_poster is None:
        return
    body = {
        "requested_kind": result.requested_kind,
        "chosen_kind": result.chosen_kind,
        "reason": result.reason,
        "cluster": result.cluster,
        "elapsed_seconds": round(result.elapsed_seconds, 3),
        "attempts": [_attempt_to_dict(a) for a in result.attempts],
        "extra": dict(result.extra),
    }
    _post_marker_nonfatal(
        marker_poster,
        issue=spec.issue,
        note=json.dumps(body, sort_keys=True),
        context=f"backend-selected chosen_kind={result.chosen_kind}",
    )


def _post_intermediate_marker(
    *,
    spec: RunSpec,
    marker_poster: Callable[..., None] | None,
    reason: str,
    attempts_today: int,
) -> None:
    """Post a visible "about to escalate to GCP" breadcrumb.

    Per plan §6: "Before escalating to GCP, post a visible marker (credit
    is scarce/expiring)". Body uses the same ``epm:backend-selected``
    schema with ``chosen_kind: "gcp"`` so the dashboard surfaces the
    intent. The final marker (posted after GCP launch succeeds /
    fails) carries the resolved outcome — both events appear in the
    timeline.
    """
    if marker_poster is None:
        return
    body = {
        "requested_kind": None,
        "chosen_kind": "gcp",
        "reason": reason,
        "cluster": None,
        "elapsed_seconds": 0.0,
        "attempts": [],
        "extra": {
            "intermediate": True,
            "gcp_attempts_today": attempts_today,
        },
    }
    _post_marker_nonfatal(
        marker_poster,
        issue=spec.issue,
        note=json.dumps(body, sort_keys=True),
        context="pre-escalation breadcrumb",
    )


def _post_terminal_failure_marker(
    *,
    spec: RunSpec,
    marker_poster: Callable[..., None] | None,
    reason: str,
    chosen_kind: BackendKind,
    attempts: list[RouteAttempt],
    extra: dict[str, Any] | None = None,
) -> None:
    """Post a final ``epm:backend-selected`` breadcrumb BEFORE raising terminal.

    The router's terminal-failure paths (``NoComputeAvailableError``,
    ``WorkloadSurfacedError``, ``ManualAttentionRequiredError``) raise
    rather than return — without this marker the dashboard would never
    see the failure breadcrumb that the success path always posts. Wires
    the reason code (:data:`ROUTE_REASON_NO_COMPUTE` /
    :data:`ROUTE_REASON_WORKLOAD_FAILURE`) the slice-5 module exports as
    public constants so downstream surfaces can pattern-match on them.
    """
    if marker_poster is None:
        return
    body = {
        "requested_kind": None,
        "chosen_kind": chosen_kind,
        "reason": reason,
        "cluster": None,
        "elapsed_seconds": 0.0,
        "attempts": [_attempt_to_dict(a) for a in attempts],
        "extra": dict(extra or {}),
    }
    # Non-fatal: a poster failure here would clobber the typed terminal
    # (NoCompute / WorkloadSurfaced / ManualAttention) about to be
    # raised — the orchestrator's failure-classifier needs THAT
    # exception, not an unclassified marker-transport error.
    _post_marker_nonfatal(
        marker_poster,
        issue=spec.issue,
        note=json.dumps(body, sort_keys=True),
        context=f"terminal-failure breadcrumb reason={reason}",
    )


def _attempt_to_dict(a: RouteAttempt) -> dict[str, Any]:
    return {
        "kind": a.kind,
        "cluster": a.cluster,
        "est_start_seconds_raw": a.est_start_seconds_raw,
        "est_start_seconds_clamped": a.est_start_seconds_clamped,
        "outcome": a.outcome,
        "detail": a.detail,
        "elapsed_seconds": round(a.elapsed_seconds, 3),
    }


# ---------------------------------------------------------------------------
# Re-exports
# ---------------------------------------------------------------------------


__all__ = [
    "CANCEL_LIVE_GRACE_SECONDS",
    "DEFAULT_AUTO_LANE_ORDER",
    "DEFAULT_FREE_LANE_ORDER",
    "DEFAULT_POLL_INTERVAL",
    "ENV_AUTO_LANE_ORDER",
    "FREE_WAIT_SECONDS",
    "LEASE_STORE_DIRNAME",
    "MAX_GCP_ATTEMPTS_PER_DAY",
    "PARK_MAX_CONSECUTIVE_PROBE_FAILURES",
    "ROUTE_REASON_AUTO_FALLBACK_GCP",
    "ROUTE_REASON_AUTO_STARTED",
    "ROUTE_REASON_NO_COMPUTE",
    "ROUTE_REASON_OVERRIDE",
    "ROUTE_REASON_PREPARE_FAILED",
    "ROUTE_REASON_RECONNECT",
    "ROUTE_REASON_WORKLOAD_FAILURE",
    "BackendPrepareError",
    "GcpAttemptCapExceededError",
    "Lease",
    "LeaseStore",
    "ManualAttentionRequiredError",
    "NoComputeAvailableError",
    "RouteAttempt",
    "RouteError",
    "RouteResult",
    "RouterConfig",
    "WorkloadSurfacedError",
    "auto_lane_order",
    "cancel_and_wait",
    "canonicalize_spec",
    "default_is_live",
    "default_is_started",
    "park_until_running_or_cap",
    "rank_lanes",
    "route",
    "spec_hash",
]
