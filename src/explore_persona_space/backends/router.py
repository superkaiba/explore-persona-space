"""Central multi-backend compute router (slice 5 of the compute-router plan).

This module is the canonical replacement for
:func:`backends.selector.select_backend`'s submit-and-park flow. Where the
selector dispatches on a single ``backend:`` frontmatter and falls back to
RunPod-on-error, ``route(spec)`` orchestrates the full multi-backend ladder:

1. **Explicit override** — ``spec.backend == "runpod" | "gcp" | "nibi" |
   "fir" | "mila"`` runs that lane directly. RunPod is reachable ONLY via
   the override; the auto chain never spends real money.
2. **Auto** — rank free lanes ``[nibi, fir if cfg.available, mila if
   socket alive]`` by tz-corrected ``estimate_start_seconds`` (a ranking
   HINT, never a gate), submit the best, park up to ``FREE_WAIT`` (default
   600 s) for it to reach RUNNING. PENDING-at-cap triggers cancel + the
   next tier.
3. **Cancel state machine** — request a cancel via the backend's
   ``teardown(handle)``, then poll via the injected ``is_live_after_cancel``
   callable until the job is no longer live in the cluster queue
   (DRAC robot allowlist has no ``sacct``; we cannot confirm terminal
   CANCELLED). A job that RACED to RUNNING during cancel is KEPT (it has
   started; tearing it down would burn the wait we already paid for). A
   timeout produces a ``manual-attention`` outcome rather than a silent
   leak.
4. **Fallback chain — GCP only** — every free-lane PENDING-at-cap or
   provisioning failure escalates to GCP. NEVER RunPod on auto.
5. **Failure classification** — :class:`gcp.GcpProvisioningError` (and
   any backend-marked ``provisioning_failure: True`` raise) routes to the
   next tier; :class:`gcp.GcpWorkloadError` surfaces, NO auto-fallback;
   "every free lane park-failed AND GCP capacity-failed" raises
   :class:`NoComputeAvailableError` for the orchestrator to translate
   into ``epm:failure (failure_class: infra) + status:blocked``.
6. **Durable lease + reconnect** — a flock'd JSON lease at
   ``~/.eps-routing/issue-<N>.json`` (outside the worktree — the 09:47
   cron reaps worktrees, so a lease there would silently disappear) is
   keyed by a canonicalized spec hash + attempt id. Before any submit /
   provision, ``route()`` reconnects to an existing live job (SLURM
   ``squeue --name eps-issue-<N>``; GCE ``reconnect_or_none``) via the
   injected backend so a re-driving ``issue-tick`` cron does NOT
   double-submit. The external job/instance id is persisted IMMEDIATELY
   after submit so an orchestrator crash between submit and lease-write
   leaves an ``UNKNOWN_SUBMITTED`` recovery state.
7. **GCP attempt-count guard** — a per-issue/day attempt counter caps
   auto-escalation to GCP at ``MAX_GCP_ATTEMPTS_PER_DAY`` (default 5).
   This is NOT a dollar cap (``tests/test_no_dollar_budget_caps.py``
   enforces "no SystemExit on budget" — see plan §"Real-money safety");
   it bounds the *number of escalation attempts* so a broken classifier
   that loops can't burn the GFS credit unattended.
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
    ComputeBackend,
    PollResult,
    RunHandle,
    RunSpec,
)
from explore_persona_space.backends.gcp import (
    GcpProvisioningError,
    GcpWorkloadError,
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

#: Free-lane order for auto routing (DRAC + Mila). RunPod is NEVER in
#: this list — it's override-only by deliberate design.
DEFAULT_FREE_LANE_ORDER: tuple[BackendKind, ...] = ("nibi", "fir", "mila")


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
    return {
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

    Every mutation holds an exclusive ``flock`` on the lease file's
    directory-level lock (``<lease_dir>/.lock``) — NOT on the lease file
    itself, because the lease file is created/replaced atomically via
    a write-temp-then-rename and an flock on a file we're about to
    rename is fragile. The lock spans read+modify+write so a concurrent
    ``issue-tick`` cron and a manual ``/issue`` can't both decide
    "no live job, submit fresh" and double-submit.

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

    def _lock_path(self) -> Path:
        return self._lease_dir / ".lock"

    @contextmanager
    def _flock(self) -> Iterator[None]:
        """Exclusive flock on ``<lease_dir>/.lock`` for the duration of the block.

        Read-modify-write on the lease MUST happen inside this context so
        a concurrent process doesn't read a stale lease and overwrite a
        fresh one with stale data.
        """
        self._ensure_dir()
        lock_path = self._lock_path()
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
        with self._flock():
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
        with self._flock():
            self._write_locked(path, lease)

    def _write_locked(self, path: Path, lease: Lease) -> None:
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(json.dumps(lease.to_json(), sort_keys=True, indent=2))
        os.chmod(tmp, 0o600)
        os.replace(tmp, path)

    def delete(self, issue: int) -> None:
        """Delete the lease file (idempotent on absent)."""
        path = self._lease_path(issue)
        with self._flock():
            try:
                path.unlink()
            except FileNotFoundError:
                return

    @contextmanager
    def transaction(self, issue: int) -> Iterator[tuple[Lease | None, Callable[[Lease], None]]]:
        """Read-modify-write transaction under the flock.

        Yields ``(current_lease_or_None, write_fn)``. The caller computes
        the new lease state inside the ``with`` block and invokes
        ``write_fn(new_lease)`` to persist it. The flock is held until
        the block exits.

        Example::

            with store.transaction(issue=137) as (lease, write):
                if lease is None:
                    lease = Lease(issue=137, spec_hash=h, attempt_id=a)
                lease.job_id = "9999"
                write(lease)
        """
        self._ensure_dir()
        path = self._lease_path(issue)
        with self._flock():
            current = self._read_locked(path)

            def write_fn(new_lease: Lease) -> None:
                self._write_locked(path, new_lease)

            yield current, write_fn


# ---------------------------------------------------------------------------
# Helpers (estimate ranking, GCP attempt counter)
# ---------------------------------------------------------------------------


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
      still live. The router escalates to GCP regardless (the cluster
      job will eventually time out on its own ``--time`` budget), and
      surfaces this string on the attempt log so the operator can
      manually ``scancel`` if needed.

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
) -> tuple[bool, str]:
    """Watch a launched handle for ``cap_seconds``; return (started, reason).

    ``is_started`` is the backend-aware probe — for SLURM it queries
    ``squeue -j <id>`` for state RUNNING (the production
    ``slurm_monitor.SLURM_STATE_TO_STATUS`` maps PENDING→running for
    historical reasons, so the router cannot use the PollResult.status
    field directly to distinguish PENDING from RUNNING). For GCP the
    binding is ``backend.poll(handle).status == "running"``. For tests
    the binding is whatever the test double exposes.

    Returns:

    * ``(True, "running")`` — job reached RUNNING before the cap.
    * ``(False, "park_cap_exceeded")`` — still PENDING (or otherwise
      not-running) at the cap. Caller should run :func:`cancel_and_wait`
      and escalate to the next tier.
    * ``(False, "terminal_before_running")`` — the probe-poll returned
      a terminal status (done/dead/stalled/gate) before RUNNING. Caller
      should NOT cancel (it's already gone); just escalate. The PollResult
      may carry diagnostic detail via ``backend.poll(handle).status``.
    """
    start = now_fn()
    while True:
        try:
            started = is_started(backend, handle)
        except Exception as exc:
            logger.warning(
                "park: is_started probe raised (%s: %s); treating as still-pending.",
                type(exc).__name__,
                exc,
            )
            started = False
        if started:
            return True, "running"
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
            return False, "terminal_before_running"
        if now_fn() - start >= cap_seconds:
            return False, "park_cap_exceeded"
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
    free_lane_order: tuple[BackendKind, ...] = DEFAULT_FREE_LANE_ORDER


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
    estimate_fn: (Callable[[ComputeBackend, BackendKind, RunSpec], float | None] | None) = None,
    reconnect_fn: (
        Callable[[ComputeBackend, BackendKind, RunSpec], RunHandle | None] | None
    ) = None,
    marker_poster: Callable[..., None] | None = None,
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
      routing iterates this in :attr:`RouterConfig.free_lane_order`.
      A missing kind is skipped (e.g. ``mila`` absent → router skips
      Mila even when the socket is alive).
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

    # The legacy :class:`RunSpec` defaults ``backend="runpod"`` for
    # back-compat with the pre-router selector tests; the router's "no
    # explicit override" intent is the explicit sentinel ``"auto"``.
    # A caller that wants auto-routing builds the spec with
    # ``backend="auto"`` (slice 6 wires this for tasks that have no
    # ``backend:`` frontmatter). Any other value is treated as an
    # explicit override.

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
        )

    if spec.backend in {"nibi", "fir", "mila"}:
        free = (free_backends or {}).get(spec.backend)
        if free is None:
            raise RouteError(
                f"backend override {spec.backend!r} requested but no free backend wired for it"
            )
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
        )

    # ----------------------------- auto chain ---------------------------
    return _auto_route(
        spec=spec,
        free_backends=free_backends or {},
        gcp_backend=gcp_backend,
        store=store,
        attempts=attempts,
        started_at=started_at,
        cfg=cfg,
        is_started=is_started,
        is_live_after_cancel=is_live_after_cancel,
        is_running_after_cancel=is_running_after_cancel,
        mila_socket_alive=mila_socket_alive,
        estimate_fn=estimate_fn,
        reconnect_fn=reconnect_fn,
        now_fn=now_fn,
        sleep_fn=sleep_fn,
        marker_poster=marker_poster,
        clock_fn=clock_fn,
    )


# ---------------------------------------------------------------------------
# Override paths
# ---------------------------------------------------------------------------


def _override_runpod(
    *,
    spec: RunSpec,
    backend: ComputeBackend,
    store: LeaseStore,
    attempts: list[RouteAttempt],
    started_at: float,
    now_fn: Callable[[], float],
    marker_poster: Callable[..., None] | None,
) -> RouteResult:
    """Explicit RunPod override — just submit. No park, no fallback.

    RunPod's "start time" is the few-minute provision; we don't gate it
    behind a park watchdog (the existing RunPod flow doesn't, and a 10
    min park would force a real user-meaningful timeout where today
    there is none). Reconnect via the lease's job_id is wired in slice
    6 (the RunPod backend doesn't yet expose a "find live pod by name"
    handle-reconstructor; today the existing pod_lifecycle.py path is
    idempotent itself).
    """
    handle = backend.launch(spec)
    _persist_lease_after_submit(
        spec=spec,
        store=store,
        backend_kind="runpod",
        cluster=None,
        handle=handle,
        now_fn=now_fn,
    )
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
) -> RouteResult:
    """Explicit non-RunPod lane override.

    Reconnect first (idempotent re-entry), then launch + park. A free
    lane that times out / hard-fails RAISES (the user explicitly asked
    for that lane; we don't silently re-route).
    """
    handle = _try_reconnect(
        backend=backend, kind=kind, spec=spec, reconnect_fn=reconnect_fn, store=store
    )
    if handle is not None:
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

    # Fresh submit + park.
    try:
        handle = backend.launch(_thread_attempt_id(spec, store))
    except GcpProvisioningError as exc:
        # Explicit GCP override — surface the provisioning failure (the
        # user asked for GCP, not a fallback chain).
        attempts.append(
            RouteAttempt(
                kind=kind,
                cluster=spec.cluster,
                est_start_seconds_raw=None,
                est_start_seconds_clamped=None,
                outcome="provisioning_failure",
                detail=exc.reason,
                elapsed_seconds=now_fn() - started_at,
            )
        )
        raise
    _persist_lease_after_submit(
        spec=spec,
        store=store,
        backend_kind=kind,
        cluster=spec.cluster,
        handle=handle,
        now_fn=now_fn,
    )

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

    # SLURM-style free lane: run the park watchdog.
    started, reason = park_until_running_or_cap(
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

    # Park failed. The user explicitly asked for this lane → cancel + raise.
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
    is_started: Callable[[ComputeBackend, RunHandle], bool],
    is_live_after_cancel: Callable[[ComputeBackend, RunHandle], bool],
    is_running_after_cancel: Callable[[ComputeBackend, RunHandle], bool] | None,
    mila_socket_alive: Callable[[], bool] | None,
    estimate_fn: Callable[[ComputeBackend, BackendKind, RunSpec], float | None] | None,
    reconnect_fn: (Callable[[ComputeBackend, BackendKind, RunSpec], RunHandle | None] | None),
    now_fn: Callable[[], float],
    sleep_fn: Callable[[float], None],
    marker_poster: Callable[..., None] | None,
    clock_fn: Callable[[], datetime] | None,
) -> RouteResult:
    """No-``backend:`` auto route: rank free lanes, park, escalate to GCP."""
    del clock_fn  # reserved for a future "day boundary at posted-time" override
    # Build the candidate list (skipping unwired lanes + Mila-when-down).
    candidates: list[tuple[ComputeBackend, BackendKind]] = []
    for kind in cfg.free_lane_order:
        backend = free_backends.get(kind)
        if backend is None:
            continue
        if kind == "mila" and (mila_socket_alive is None or not mila_socket_alive()):
            continue
        candidates.append((backend, kind))

    # Stage 1: reconnect (free lanes first, then GCP).
    reconnect_result = _try_auto_reconnect(
        spec=spec,
        candidates=candidates,
        gcp_backend=gcp_backend,
        store=store,
        attempts=attempts,
        started_at=started_at,
        reconnect_fn=reconnect_fn,
        now_fn=now_fn,
        marker_poster=marker_poster,
    )
    if reconnect_result is not None:
        return reconnect_result

    # Stage 2: rank + try each free lane (launch → park → cancel-on-fail).
    estimated = _estimate_lanes(candidates, spec=spec, estimate_fn=estimate_fn)
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
        now_fn=now_fn,
        sleep_fn=sleep_fn,
        marker_poster=marker_poster,
    )
    if free_result is not None:
        return free_result

    # Stage 3: escalate to GCP (gated by attempt-count guard).
    return _escalate_to_gcp(
        spec=spec,
        gcp_backend=gcp_backend,
        store=store,
        attempts=attempts,
        started_at=started_at,
        cfg=cfg,
        now_fn=now_fn,
        marker_poster=marker_poster,
    )


def _try_auto_reconnect(
    *,
    spec: RunSpec,
    candidates: list[tuple[ComputeBackend, BackendKind]],
    gcp_backend: ComputeBackend | None,
    store: LeaseStore,
    attempts: list[RouteAttempt],
    started_at: float,
    reconnect_fn: (Callable[[ComputeBackend, BackendKind, RunSpec], RunHandle | None] | None),
    now_fn: Callable[[], float],
    marker_poster: Callable[..., None] | None,
) -> RouteResult | None:
    """Auto-route stage 1: look for an existing live job on every wired lane."""
    for backend, kind in candidates:
        handle = _try_reconnect(
            backend=backend, kind=kind, spec=spec, reconnect_fn=reconnect_fn, store=store
        )
        if handle is None:
            continue
        return _record_reconnect(
            backend=backend,
            kind=kind,
            cluster=spec.cluster,
            handle=handle,
            spec=spec,
            attempts=attempts,
            started_at=started_at,
            now_fn=now_fn,
            marker_poster=marker_poster,
            detail="found existing live job/instance",
        )

    if gcp_backend is None:
        return None
    handle = _try_reconnect(
        backend=gcp_backend, kind="gcp", spec=spec, reconnect_fn=reconnect_fn, store=store
    )
    if handle is None:
        return None
    return _record_reconnect(
        backend=gcp_backend,
        kind="gcp",
        cluster=None,
        handle=handle,
        spec=spec,
        attempts=attempts,
        started_at=started_at,
        now_fn=now_fn,
        marker_poster=marker_poster,
        detail="found existing live gcp instance",
    )


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
) -> RouteResult:
    """Append a reconnect attempt + build the matching RouteResult."""
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
    now_fn: Callable[[], float],
    sleep_fn: Callable[[float], None],
    marker_poster: Callable[..., None] | None,
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
            now_fn=now_fn,
            sleep_fn=sleep_fn,
            marker_poster=marker_poster,
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
    now_fn: Callable[[], float],
    sleep_fn: Callable[[float], None],
    marker_poster: Callable[..., None] | None,
) -> RouteResult | None:
    """Launch + park one free lane. Returns a RouteResult on success / cancel-race.

    Returns ``None`` to signal "next lane". Cancel-race during park-fail
    is treated as success (the job won; tearing it down would forfeit
    the wait we already paid for).
    """
    # Launch.
    try:
        handle = backend.launch(_thread_attempt_id(spec, store))
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

    _persist_lease_after_submit(
        spec=spec,
        store=store,
        backend_kind=kind,
        cluster=spec.cluster,
        handle=handle,
        now_fn=now_fn,
    )

    # Park.
    started, reason = park_until_running_or_cap(
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

    # Park failed → cancel state machine, then either escalate or
    # (on cancel-race) keep the job.
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


def _escalate_to_gcp(
    *,
    spec: RunSpec,
    gcp_backend: ComputeBackend | None,
    store: LeaseStore,
    attempts: list[RouteAttempt],
    started_at: float,
    cfg: RouterConfig,
    now_fn: Callable[[], float],
    marker_poster: Callable[..., None] | None,
) -> RouteResult:
    """Auto-route stage 3: bump attempt counter, launch GCP, classify.

    Raises :class:`NoComputeAvailableError` when no GCP backend is wired
    or when GCP's provisioning fails. Raises
    :class:`WorkloadSurfacedError` on a GCP workload failure (no
    auto-fallback). Raises :class:`GcpAttemptCapExceededError` when the
    per-day attempt cap is reached.
    """
    if gcp_backend is None:
        raise NoComputeAvailableError(
            "every free lane park-failed AND no gcp_backend wired for auto-fallback",
            attempts=[_attempt_to_dict(a) for a in attempts],
        )

    # Attempt-count guard.
    with store.transaction(spec.issue) as (lease, write):
        if lease is None:
            lease = Lease(
                issue=int(spec.issue),
                spec_hash=spec_hash(spec),
                attempt_id=_make_attempt_id(),
            )
        lease = _bump_gcp_attempt(lease)
        write(lease)
        attempts_today = lease.gcp_attempts_today

    if attempts_today > cfg.max_gcp_attempts_per_day:
        raise GcpAttemptCapExceededError(
            issue=int(spec.issue),
            attempts_today=attempts_today,
            cap=cfg.max_gcp_attempts_per_day,
        )

    # Pre-escalation marker — visible breadcrumb before spending credit.
    _post_intermediate_marker(
        spec=spec,
        marker_poster=marker_poster,
        reason=ROUTE_REASON_AUTO_FALLBACK_GCP,
        attempts_today=attempts_today,
    )

    try:
        gcp_handle = gcp_backend.launch(_thread_attempt_id(spec, store))
    except GcpProvisioningError as exc:
        attempts.append(
            RouteAttempt(
                kind="gcp",
                cluster=None,
                est_start_seconds_raw=0.0,
                est_start_seconds_clamped=0.0,
                outcome="provisioning_failure",
                detail=exc.reason,
                elapsed_seconds=now_fn() - started_at,
            )
        )
        raise NoComputeAvailableError(
            f"every free lane park-failed AND gcp provisioning failed: {exc.reason}",
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
        raise WorkloadSurfacedError(
            f"gcp workload failure (no auto-fallback): {exc.reason}",
            chosen_kind="gcp",
            evidence=exc.evidence,
        ) from exc

    _persist_lease_after_submit(
        spec=spec,
        store=store,
        backend_kind="gcp",
        cluster=None,
        handle=gcp_handle,
        now_fn=now_fn,
    )
    attempts.append(
        RouteAttempt(
            kind="gcp",
            cluster=None,
            est_start_seconds_raw=0.0,
            est_start_seconds_clamped=0.0,
            outcome="launched",
            detail=f"gcp escalation #{attempts_today} of cap {cfg.max_gcp_attempts_per_day}",
            elapsed_seconds=now_fn() - started_at,
        )
    )
    result = RouteResult(
        backend=gcp_backend,
        handle=gcp_handle,
        requested_kind=None,
        chosen_kind="gcp",
        reason=ROUTE_REASON_AUTO_FALLBACK_GCP,
        cluster=None,
        attempts=attempts,
        elapsed_seconds=now_fn() - started_at,
        extra={"gcp_attempts_today": attempts_today},
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
            raw = fn(backend, kind, spec)
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
    store: LeaseStore,
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
    """
    if reconnect_fn is None:
        return None
    try:
        handle = reconnect_fn(backend, kind, spec)
    except Exception as exc:
        logger.warning(
            "route: reconnect_fn raised for %s (%s: %s); treating as no live job.",
            kind,
            type(exc).__name__,
            exc,
        )
        return None
    # Defensive: a reconnect_fn that returns a handle for the WRONG
    # issue would silently bind to someone else's run. Sanity-check.
    if handle is not None and handle.extra.get("issue") not in (None, int(spec.issue)):
        logger.error(
            "route: reconnect_fn for %s returned a handle for issue=%r (expected %d); ignoring.",
            kind,
            handle.extra.get("issue"),
            spec.issue,
        )
        return None
    # The UNKNOWN_SUBMITTED state is detected by the orchestrator (slice
    # 6) and resolved via the same reconnect_fn — no extra logic here.
    _ = store  # placeholder; slice 6 may use store to record recovery
    return handle


def _persist_lease_after_submit(
    *,
    spec: RunSpec,
    store: LeaseStore,
    backend_kind: BackendKind,
    cluster: str | None,
    handle: RunHandle,
    now_fn: Callable[[], float],
) -> None:
    """Write the external job/instance id to the lease IMMEDIATELY after submit.

    Crash window covered: a submit that returns successfully but the
    orchestrator dies before the lease is updated would otherwise leave
    a leaked job / instance. The :class:`LeaseStore.transaction`
    context flocks the directory so a concurrent re-driving cron sees
    the fresh id (or, when this write hasn't happened yet, can detect
    the ``UNKNOWN_SUBMITTED`` state and re-reconnect via the backend's
    queue rather than re-submitting).
    """
    with store.transaction(spec.issue) as (lease, write):
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
        del now_fn  # monotonic clock is for the watchdog, not the lease timestamp
        write(lease)


def _thread_attempt_id(spec: RunSpec, store: LeaseStore) -> RunSpec:
    """Ensure ``spec.extra["attempt_id"]`` is set + matches the lease.

    The router writes the attempt id on first lease creation. On
    re-entry, the lease's id wins — the GCP backend uses it as the
    artifact namespace, and a fresh id on every router call would
    silently fork the namespace.
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


def _make_attempt_id() -> str:
    """Per-attempt id — same shape the GCP backend's ``attempt_id_for`` produces."""
    return f"att-{datetime.now(tz=UTC).strftime('%Y%m%d-%H%M%S')}"


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
      est-start, outcome, detail, elapsed).
    * ``free_lane_order`` — the order considered.
    * Existing schema preserved: ``requested_kind`` / ``chosen_kind`` /
      ``reason`` / ``cluster`` / ``elapsed_seconds`` / ``extra``.
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
    marker_poster(
        issue=spec.issue,
        marker="epm:backend-selected",
        note=json.dumps(body, sort_keys=True),
        version=1,
        by="backends.router",
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
    marker_poster(
        issue=spec.issue,
        marker="epm:backend-selected",
        note=json.dumps(body, sort_keys=True),
        version=1,
        by="backends.router",
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
    "DEFAULT_FREE_LANE_ORDER",
    "DEFAULT_POLL_INTERVAL",
    "FREE_WAIT_SECONDS",
    "LEASE_STORE_DIRNAME",
    "MAX_GCP_ATTEMPTS_PER_DAY",
    "ROUTE_REASON_AUTO_FALLBACK_GCP",
    "ROUTE_REASON_AUTO_STARTED",
    "ROUTE_REASON_NO_COMPUTE",
    "ROUTE_REASON_OVERRIDE",
    "ROUTE_REASON_RECONNECT",
    "ROUTE_REASON_WORKLOAD_FAILURE",
    "GcpAttemptCapExceededError",
    "Lease",
    "LeaseStore",
    "NoComputeAvailableError",
    "RouteAttempt",
    "RouteError",
    "RouteResult",
    "RouterConfig",
    "WorkloadSurfacedError",
    "cancel_and_wait",
    "canonicalize_spec",
    "default_is_live",
    "default_is_started",
    "park_until_running_or_cap",
    "rank_lanes",
    "route",
    "spec_hash",
]
