"""Backend selection + submit-and-park decision flow.

The selector reads a task's ``backend:`` frontmatter and returns the
:class:`ComputeBackend` instance that should execute it, plus a
:class:`BackendDecision` recording WHICH backend ran and WHY (for the
``epm:backend-selected v1`` marker the orchestrator posts).

Decision table (slice 1 — the SLURM backend is stubbed, but the control
flow that picks RunPod on fall-back is real and tested):

    backend frontmatter        | selector behavior
    ---------------------------+-------------------------------------------
    (absent) or "runpod"       | RunPodBackend; never touches SLURM.
                               | This is the byte-for-byte preserved path:
                               | the same call sequence the orchestrator
                               | drives today.
    ---------------------------+-------------------------------------------
    "cluster" / "nibi" / "fir" | Try SLURM via ``_build_slurm_backend``.
                               | If launch raises NotImplementedError (the
                               | slice-1 stub) OR a hard submit/auth
                               | failure, fall back to RunPodBackend and
                               | record reason="slurm_unavailable_<...>".
                               | Submit-and-park watcher (max_wait_seconds)
                               | is wired around the launch call; PENDING
                               | beyond the cap → scancel + RunPod.

The selector is the ONLY component that knows about the fall-back
relationship — backends do not know about each other. Each backend
either succeeds end-to-end or raises; the selector wraps SLURM in the
fall-back try/except.

Slice 1: the SLURM launch path raises ``NotImplementedError`` with the
sentinel message :data:`SLURM_NOT_IMPLEMENTED_MESSAGE`. Slice 2 replaces
the raise with a real submit + monitor (``slurm.py`` + ``slurm_monitor
.py``). The selector's decision table + fall-back logic does NOT change
between slices — that's the whole point of landing it here first.
"""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass, field
from typing import Any

from explore_persona_space.backends.base import (
    BackendKind,
    ComputeBackend,
    RunHandle,
    RunSpec,
)
from explore_persona_space.backends.runpod import RunPodBackend

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# LEGACY default max-wait for the selector's submit-and-park watcher.
# Kept for back-compat callers of :func:`select_backend` (the pre-slice-5
# code path the legacy dispatch line goes through). The slice-6 unified
# dispatch (:func:`backends.router.route`) replaces this with a SHORTER,
# ALWAYS-APPLIED 10-minute park (``router.FREE_WAIT_SECONDS``) on every
# free-lane submit — the SKILL.md text that previously documented the
# 6-h park + the ``EPM_CLUSTER_MAX_WAIT_SECONDS`` env knob has been
# updated to the 10-min policy. The env knob was advertised in docs but
# never read in code; the slice-6 SKILL.md edit drops it. New callers
# MUST use ``backends.router.route`` (or ``backends.issue_dispatch``);
# this constant exists only so the legacy ``select_backend`` signature
# stays stable.
DEFAULT_MAX_WAIT_SECONDS = 6 * 3600

# Sentinel exception message the slice-1 SLURM stub raises. The selector
# matches on this to know a fall-back is the right action (vs. a real
# bug). Slice 2 removes the raise; the matching code below short-circuits
# when the message isn't present.
SLURM_NOT_IMPLEMENTED_MESSAGE = "slurm backend: slice 2"

# Cluster-name aliases. ``cluster`` is the generic dispatch; ``nibi`` /
# ``fir`` are per-cluster aliases the user can set on the task. The
# selector maps them all to the SLURM backend (slice 2 onwards) and
# threads the chosen cluster through ``RunSpec.cluster``.
_CLUSTER_KINDS: frozenset[BackendKind] = frozenset({"cluster", "nibi", "fir"})


# ---------------------------------------------------------------------------
# Decision dataclass + exceptions
# ---------------------------------------------------------------------------


class BackendSelectionError(RuntimeError):
    """Raised when the task frontmatter requests an unknown backend kind.

    Distinct from a fall-back: this fires BEFORE any backend method is
    called, because the request itself is malformed. Surfaces to the
    orchestrator as a config-error rather than a backend-runtime error.
    """


@dataclass(frozen=True)
class BackendDecision:
    """Result of :func:`select_backend`.

    Fields:

    * ``backend``: the chosen :class:`ComputeBackend` instance. The
      orchestrator calls ``backend.launch(spec)`` to actually start the
      run; this is already wired (or attempted, then fallen back to
      RunPod, depending on path).
    * ``handle``: the :class:`RunHandle` returned by the chosen backend's
      ``launch``. ``None`` only when the caller passed ``launch=False``.
    * ``requested_kind``: what the task frontmatter asked for (e.g.
      ``"cluster"``). The marker post needs this to record intent vs.
      outcome.
    * ``chosen_kind``: which backend actually ran (``"runpod"`` always
      on the fall-back path; ``"cluster"`` etc. on the happy SLURM path
      once slice 2 lands).
    * ``reason``: short tag for the marker (``"frontmatter_default"``,
      ``"frontmatter_explicit"``, ``"slurm_not_implemented"``,
      ``"slurm_max_wait_exceeded"``, ``"slurm_hard_failure"``).
    * ``cluster``: the concrete cluster name when the SLURM path was
      tried (``"nibi"`` etc.); ``None`` when RunPod ran from the start.
    * ``elapsed_seconds``: wall-clock seconds spent INSIDE
      ``select_backend`` (includes the submit-and-park watcher cost
      when fall-back happened). The marker logs this so we can audit
      how often the watcher actually fires.
    * ``extra``: free-form (e.g. the exception class name on fall-back).
    """

    backend: ComputeBackend
    handle: RunHandle | None
    requested_kind: BackendKind
    chosen_kind: BackendKind
    reason: str
    cluster: str | None = None
    elapsed_seconds: float = 0.0
    extra: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Public API: select_backend(task_or_frontmatter, ...)
# ---------------------------------------------------------------------------


def _parse_backend_kind(raw: Any) -> BackendKind:
    """Normalize a frontmatter ``backend:`` value to a :data:`BackendKind`.

    Accepts ``None`` (default), missing string, or a recognized literal.
    Raises :class:`BackendSelectionError` on an unknown value so a typo
    surfaces loudly rather than silently routing to RunPod.
    """
    if raw is None:
        return "runpod"
    if not isinstance(raw, str):
        raise BackendSelectionError(
            f"backend frontmatter must be a string, got {type(raw).__name__}: {raw!r}"
        )
    val = raw.strip().lower()
    if val == "":
        return "runpod"
    if val == "runpod":
        return "runpod"
    if val in {"cluster", "nibi", "fir"}:
        return val  # type: ignore[return-value]
    raise BackendSelectionError(
        f"unknown backend frontmatter value: {raw!r}. Expected one of: runpod, cluster, nibi, fir."
    )


def _resolve_cluster_name(kind: BackendKind, explicit_cluster: str | None) -> str | None:
    """Pick the cluster name for a SLURM dispatch.

    Precedence: explicit ``RunSpec.cluster`` > ``backend:`` alias
    (``nibi`` / ``fir`` themselves name the cluster) > default Nibi
    (slice 1 / v1 ships Nibi only — Fir routing is deferred to v1.1
    per the plan's ``cluster: nibi || cluster: fir`` rule).
    """
    if explicit_cluster:
        return explicit_cluster
    if kind in {"nibi", "fir"}:
        return kind
    if kind == "cluster":
        return "nibi"  # v1 default; slice 2 hardcodes this in the SLURM backend too.
    return None


def _build_runpod_backend() -> RunPodBackend:
    """Factory hook so tests can monkeypatch the backend instance."""
    return RunPodBackend()


def _build_slurm_backend() -> ComputeBackend:
    """Factory for the SLURM backend.

    Slice 2 (this revision): returns the real
    :class:`~explore_persona_space.backends.slurm.SlurmBackend`. The
    selector's fall-back logic does NOT change between slices — when
    the real backend raises (auth error, scancel-on-park, etc.) the
    same fall-back paths fire. The slice-1 stub (``_SlurmStubBackend``)
    is preserved below so existing tests can still exercise the
    NotImplemented fall-back path by passing ``slurm_backend=_SlurmStubBackend()``
    explicitly.
    """
    # Lazy import: the slurm module pulls in subprocess/ssh helpers that
    # we don't want to drag in for the RunPod-only common case.
    from explore_persona_space.backends.slurm import SlurmBackend

    return SlurmBackend()


class _SlurmStubBackend(ComputeBackend):
    """Slice-1 stub. Every method raises with the sentinel message.

    The selector's fall-back path catches ``NotImplementedError`` whose
    message contains :data:`SLURM_NOT_IMPLEMENTED_MESSAGE`. Slice 2's
    real ``SlurmBackend`` replaces this entirely; the selector code
    does NOT special-case the stub vs. the real backend.
    """

    @property
    def name(self) -> BackendKind:
        return "cluster"

    def prepare(self, spec: RunSpec) -> None:
        del spec
        raise NotImplementedError(SLURM_NOT_IMPLEMENTED_MESSAGE)

    def launch(self, spec: RunSpec) -> RunHandle:
        del spec
        raise NotImplementedError(SLURM_NOT_IMPLEMENTED_MESSAGE)

    def estimate_start(self, spec: RunSpec):
        del spec
        return None

    def poll(self, handle: RunHandle):
        del handle
        raise NotImplementedError(SLURM_NOT_IMPLEMENTED_MESSAGE)

    def fetch_logs(self, handle: RunHandle) -> str:
        del handle
        raise NotImplementedError(SLURM_NOT_IMPLEMENTED_MESSAGE)

    def fetch_results(self, handle: RunHandle) -> None:
        del handle
        raise NotImplementedError(SLURM_NOT_IMPLEMENTED_MESSAGE)

    def confirm_artifacts(self, handle: RunHandle) -> bool:
        del handle
        raise NotImplementedError(SLURM_NOT_IMPLEMENTED_MESSAGE)

    def teardown(self, handle: RunHandle) -> None:
        del handle
        raise NotImplementedError(SLURM_NOT_IMPLEMENTED_MESSAGE)


def _is_slurm_stub_unavailable(exc: BaseException) -> bool:
    """True iff ``exc`` is the slice-1 stub's NotImplementedError.

    Distinct from any OTHER NotImplementedError a real backend might
    raise — we only treat the explicit sentinel as a fall-back trigger.
    A real backend bug that surfaces as ``NotImplementedError`` from a
    DIFFERENT code path should NOT silently fall back to RunPod (that
    would mask the bug).
    """
    return isinstance(exc, NotImplementedError) and SLURM_NOT_IMPLEMENTED_MESSAGE in str(exc)


def _wait_for_slurm_start(
    backend: ComputeBackend,
    handle: RunHandle,
    *,
    max_wait_seconds: int,
    poll_interval_seconds: float,
    now_fn,  # injected for test determinism
    sleep_fn,
) -> tuple[bool, str]:
    """Submit-and-park watchdog: wait for the SLURM job to leave PENDING.

    Returns ``(started, reason)``. ``started`` is True when the job has
    reached ``running`` (the orchestrator should let it continue);
    False when ``max_wait_seconds`` elapsed first (the selector should
    ``scancel`` and fall back to RunPod).

    Slice 1 wires the control flow only — the actual poll happens via
    ``backend.poll(handle)`` which, for the stub backend, raises with the
    sentinel and the caller treats that as "slurm unavailable, fall back".
    For a real SlurmBackend (slice 2) this loop drives the live PENDING
    -> RUNNING transition. Stub-vs-real is invisible here.
    """
    start = now_fn()
    while True:
        try:
            result = backend.poll(handle)
        except NotImplementedError as exc:
            if _is_slurm_stub_unavailable(exc):
                return False, "slurm_not_implemented"
            raise
        if result.status == "running":
            return True, "slurm_started"
        if result.status in {"done", "stalled", "dead", "gate"}:
            # Terminal-before-running is a hard failure (sbatch was
            # accepted but the job died before leaving PENDING).
            return False, f"slurm_hard_failure_status_{result.status}"
        elapsed = now_fn() - start
        if elapsed >= max_wait_seconds:
            return False, "slurm_max_wait_exceeded"
        sleep_fn(poll_interval_seconds)


def select_backend(
    task: dict[str, Any] | None = None,
    *,
    spec: RunSpec | None = None,
    launch: bool = True,
    max_wait_seconds: int = DEFAULT_MAX_WAIT_SECONDS,
    poll_interval_seconds: float = 30.0,
    runpod_backend: ComputeBackend | None = None,
    slurm_backend: ComputeBackend | None = None,
    now_fn=time.monotonic,
    sleep_fn=time.sleep,
    marker_poster=None,
) -> BackendDecision:
    """Pick + (optionally) launch the right backend for a task.

    .. warning::
        LEGACY PATH — ``launch=True`` calls ``backend.launch(spec)``
        WITHOUT ``backend.prepare(spec)``. For the SLURM backend that
        means NO repo rsync, NO secrets push, and NO stale-artifact
        clearing: the job dies in its in-job preflight (the issue-535
        live failure shape). The production dispatch path is
        :func:`backends.router.route`, whose ``_prepare_and_launch``
        chokepoint runs ``prepare`` on every fresh launch. There is no
        production caller of ``launch=True`` here; if you are adding
        one, use ``route()`` instead (or call ``backend.prepare(spec)``
        yourself and document why the router is unsuitable). This
        selector is deliberately NOT rewired (round-6 Mn2).

    Inputs:

    * ``task``: the task's parsed frontmatter dict (e.g. from
      ``task_workflow.load_task(N).frontmatter``). The selector reads
      ``task["backend"]`` and (when SLURM is chosen) ``task["cluster"]``.
      ``None`` is treated as "no frontmatter" and routes to RunPod.
    * ``spec``: the :class:`RunSpec` for the run. The selector clones
      this with the resolved ``backend`` + ``cluster`` before passing
      it to the backend's ``launch``. Required for ``launch=True``.
    * ``launch``: when True (default), call ``backend.launch(spec)`` and
      attach the resulting handle to the decision; when False, only
      decide. Tests pass ``False`` to exercise the routing without
      provisioning anything.
    * ``max_wait_seconds``: submit-and-park cap. Plan default 6h.
    * ``poll_interval_seconds``: how often the watchdog polls SLURM.
    * ``runpod_backend`` / ``slurm_backend``: injection seams for tests
      (defaults call the factory functions above).
    * ``now_fn`` / ``sleep_fn``: monotonic-clock + sleep injection.
    * ``marker_poster``: callable used to post the
      ``epm:backend-selected v1`` marker (defaults to
      :func:`backends.slurm.post_marker_via_task_py`). Tests inject a
      list-appender. The marker is posted on EVERY decision EXCEPT
      ``launch=False`` (decision-only dry runs do NOT touch the
      events.jsonl trail). When ``spec`` is ``None`` the marker post is
      also skipped — there's no issue id to address it to.

    Returns a :class:`BackendDecision` recording the chosen backend +
    the reason + (when ``launch=True``) the live handle.

    Behavior:

    * No ``backend:`` or ``backend: runpod``: return RunPod immediately.
      No cluster code path is touched (the "zero new branches" guarantee
      from the plan).
    * ``backend: cluster|nibi|fir``: try SLURM; on
      ``NotImplementedError`` carrying :data:`SLURM_NOT_IMPLEMENTED_MESSAGE`
      OR the submit-and-park watchdog timing out OR a hard launch
      failure, fall back to RunPod. In autonomous mode
      (``EPM_AUTONOMOUS_SESSION=1``), the fall-back is silent; otherwise
      it's logged at WARNING.
    """
    requested_kind = _parse_backend_kind((task or {}).get("backend"))
    cluster_hint = (task or {}).get("cluster")

    if runpod_backend is None:
        runpod_backend = _build_runpod_backend()

    if marker_poster is None:
        # Lazy import to avoid the always-on cost of pulling the SLURM
        # module into a RunPod-only selector call. Slim modules import
        # selector at top-level; pulling slurm.py here keeps that import
        # path cheap on the common case.
        from explore_persona_space.backends.slurm import post_marker_via_task_py

        marker_poster = post_marker_via_task_py

    started_at = now_fn()

    # ---------- RunPod default path -----------------------------------
    if requested_kind == "runpod":
        decision = _launch_runpod(
            spec=spec,
            backend=runpod_backend,
            requested_kind=requested_kind,
            reason="frontmatter_default"
            if not (task or {}).get("backend")
            else "frontmatter_explicit",
            cluster=None,
            launch=launch,
            elapsed_seconds=now_fn() - started_at,
        )
        _post_backend_selected(decision, spec=spec, marker_poster=marker_poster, launch=launch)
        return decision

    # ---------- SLURM opt-in path -------------------------------------
    cluster_name = _resolve_cluster_name(requested_kind, cluster_hint)
    if slurm_backend is None:
        slurm_backend = _build_slurm_backend()

    # Thread the resolved cluster into the spec so the backend's launch
    # has it (the stub ignores it; slice 2's real backend uses it for
    # per-cluster config lookup).
    slurm_spec = spec
    if slurm_spec is not None:
        slurm_spec = _with_backend(slurm_spec, kind=requested_kind, cluster=cluster_name)

    if not launch:
        decision = BackendDecision(
            backend=slurm_backend,
            handle=None,
            requested_kind=requested_kind,
            chosen_kind=requested_kind,
            reason="decided_only",
            cluster=cluster_name,
            elapsed_seconds=now_fn() - started_at,
        )
        # decided_only deliberately does NOT post a marker (dry-run).
        return decision

    if slurm_spec is None:
        raise ValueError("select_backend(launch=True) requires a RunSpec.")

    # Try SLURM launch. The watchdog wraps the resulting handle (or, when
    # the stub raises, the launch itself raises and we go straight to
    # fall-back).
    try:
        handle = slurm_backend.launch(slurm_spec)
    except NotImplementedError as exc:
        if not _is_slurm_stub_unavailable(exc):
            raise
        decision = _fallback_to_runpod(
            spec=spec,
            runpod_backend=runpod_backend,
            requested_kind=requested_kind,
            reason="slurm_not_implemented",
            cluster=cluster_name,
            elapsed_seconds=now_fn() - started_at,
            extra={"slurm_error_class": "NotImplementedError"},
        )
        _post_backend_selected(decision, spec=spec, marker_poster=marker_poster, launch=launch)
        return decision
    except Exception as exc:
        logger.warning(
            "SLURM launch failed for issue %d on %s (%s: %s); falling back to RunPod",
            slurm_spec.issue,
            cluster_name,
            type(exc).__name__,
            exc,
        )
        decision = _fallback_to_runpod(
            spec=spec,
            runpod_backend=runpod_backend,
            requested_kind=requested_kind,
            reason="slurm_hard_failure",
            cluster=cluster_name,
            elapsed_seconds=now_fn() - started_at,
            extra={"slurm_error_class": type(exc).__name__, "slurm_error_msg": str(exc)},
        )
        _post_backend_selected(decision, spec=spec, marker_poster=marker_poster, launch=launch)
        return decision

    # Submit succeeded. Wait for PENDING -> RUNNING (or fall back on
    # exceeding the max-wait cap or a hard-failure terminal status).
    started, reason = _wait_for_slurm_start(
        slurm_backend,
        handle,
        max_wait_seconds=max_wait_seconds,
        poll_interval_seconds=poll_interval_seconds,
        now_fn=now_fn,
        sleep_fn=sleep_fn,
    )
    if started:
        decision = BackendDecision(
            backend=slurm_backend,
            handle=handle,
            requested_kind=requested_kind,
            chosen_kind=requested_kind,
            reason=reason,
            cluster=cluster_name,
            elapsed_seconds=now_fn() - started_at,
        )
        _post_backend_selected(decision, spec=spec, marker_poster=marker_poster, launch=launch)
        return decision

    # Watchdog said fall back. Best-effort scancel via teardown; ignore
    # teardown failures because the live job MIGHT already be gone and
    # the next ``squeue`` will reconcile.
    try:
        slurm_backend.teardown(handle)
    except Exception as exc:
        logger.warning(
            "SLURM teardown after fall-back raised (%s: %s); continuing to RunPod",
            type(exc).__name__,
            exc,
        )
    decision = _fallback_to_runpod(
        spec=spec,
        runpod_backend=runpod_backend,
        requested_kind=requested_kind,
        reason=reason,
        cluster=cluster_name,
        elapsed_seconds=now_fn() - started_at,
        extra={"slurm_job_id": handle.job_id, "slurm_pod_name": handle.pod_name},
    )
    _post_backend_selected(decision, spec=spec, marker_poster=marker_poster, launch=launch)
    return decision


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _autonomous_session() -> bool:
    """True when ``EPM_AUTONOMOUS_SESSION=1`` (silent fall-back)."""
    return os.environ.get("EPM_AUTONOMOUS_SESSION") == "1"


def _with_backend(spec: RunSpec, *, kind: BackendKind, cluster: str | None) -> RunSpec:
    """Return a copy of ``spec`` with ``backend`` + ``cluster`` set.

    ``RunSpec`` is frozen, so we use :func:`dataclasses.replace` semantics
    via the constructor. Kept as a helper so the same threading happens
    on every backend dispatch (instead of being open-coded in the SLURM
    + future-cluster branches and drifting).
    """
    from dataclasses import replace

    return replace(spec, backend=kind, cluster=cluster)


def _launch_runpod(
    *,
    spec: RunSpec | None,
    backend: ComputeBackend,
    requested_kind: BackendKind,
    reason: str,
    cluster: str | None,
    launch: bool,
    elapsed_seconds: float,
) -> BackendDecision:
    """Run the RunPod launch (or skip when ``launch=False``)."""
    if not launch:
        return BackendDecision(
            backend=backend,
            handle=None,
            requested_kind=requested_kind,
            chosen_kind="runpod",
            reason=f"decided_only_{reason}",
            cluster=cluster,
            elapsed_seconds=elapsed_seconds,
        )
    if spec is None:
        raise ValueError("select_backend(launch=True) requires a RunSpec.")
    # On the explicit RunPod path, the spec's backend field should
    # already be runpod (or unset). Normalize so downstream consumers
    # see a consistent value.
    runpod_spec = _with_backend(spec, kind="runpod", cluster=None)
    handle = backend.launch(runpod_spec)
    return BackendDecision(
        backend=backend,
        handle=handle,
        requested_kind=requested_kind,
        chosen_kind="runpod",
        reason=reason,
        cluster=cluster,
        elapsed_seconds=elapsed_seconds,
    )


def _fallback_to_runpod(
    *,
    spec: RunSpec | None,
    runpod_backend: ComputeBackend,
    requested_kind: BackendKind,
    reason: str,
    cluster: str | None,
    elapsed_seconds: float,
    extra: dict[str, Any],
) -> BackendDecision:
    """Launch RunPod after a SLURM fall-back; record the reason + extras."""
    if not _autonomous_session():
        logger.warning(
            "Falling back to RunPod for requested backend=%s cluster=%s (%s).",
            requested_kind,
            cluster,
            reason,
        )
    if spec is None:
        # Caller already opted out of launching; just record the decision.
        return BackendDecision(
            backend=runpod_backend,
            handle=None,
            requested_kind=requested_kind,
            chosen_kind="runpod",
            reason=reason,
            cluster=cluster,
            elapsed_seconds=elapsed_seconds,
            extra=extra,
        )
    runpod_spec = _with_backend(spec, kind="runpod", cluster=None)
    handle = runpod_backend.launch(runpod_spec)
    return BackendDecision(
        backend=runpod_backend,
        handle=handle,
        requested_kind=requested_kind,
        chosen_kind="runpod",
        reason=reason,
        cluster=cluster,
        elapsed_seconds=elapsed_seconds,
        extra=extra,
    )


def _post_backend_selected(
    decision: BackendDecision,
    *,
    spec: RunSpec | None,
    marker_poster,
    launch: bool,
) -> None:
    """Post ``epm:backend-selected v1`` to the originating task's events.jsonl.

    Skip when (a) ``launch=False`` (decision-only dry runs don't touch
    the trail) or (b) ``spec is None`` (no issue id to address the
    marker to — same case as ``launch=False`` effectively). Body shape
    matches ``workflow.yaml § markers`` exactly:
    ``requested_kind / chosen_kind / reason / cluster / elapsed_seconds
    / extra``. The ``backend`` instance + ``handle`` are NOT serialized
    (they're Python objects with cycles + secrets).
    """
    if not launch or spec is None:
        return
    # asdict on the frozen dataclass gives every field; we drop the
    # ones that don't belong in the marker body (Python objects).
    body = {
        "requested_kind": decision.requested_kind,
        "chosen_kind": decision.chosen_kind,
        "reason": decision.reason,
        "cluster": decision.cluster,
        "elapsed_seconds": round(decision.elapsed_seconds, 3),
        "extra": dict(decision.extra),
    }
    note = json.dumps(body, sort_keys=True)
    marker_poster(
        issue=spec.issue,
        marker="epm:backend-selected",
        note=note,
        version=1,
        by="backends.selector",
    )
