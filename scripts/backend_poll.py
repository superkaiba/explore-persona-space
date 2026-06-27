#!/usr/bin/env python3
"""Backend-agnostic one-tick poll script (bg-Bash poll bridge for `/issue`).

The orchestrator's Step 6d.2 bg-Bash polling loop calls this script
once per tick; it prints ONE ``PollResult``-shaped JSON line to stdout
and exits. The JSON shape is byte-identical to
``scripts/poll_pipeline.py``'s output, so the orchestrator's existing
JSON-line parser handles every backend (RunPod / SLURM / GCP)
without per-backend branches.

Usage::

    uv run python scripts/backend_poll.py --issue <N>            # default sidecar
    uv run python scripts/backend_poll.py --issue <N> --handle-file <path>

The dispatch helper (:mod:`backends.issue_dispatch`) writes the per-issue
:class:`~backends.base.RunHandle` to
``<main-checkout>/.claude/cache/issue-<N>-handle.json`` at launch (the
path is resolved cwd-INDEPENDENTLY — a launch dispatched from an issue
worktree and a poll tick run from the repo root converge on the same
file; incident #612). This script reads it back, recovers the right
:class:`~backends.base.ComputeBackend` subclass from
``handle.backend``, and calls ``backend.poll(handle)`` once. For
back-compat with sidecars written by the pre-#612 cwd-relative composer
it also probes ``<cwd>/.claude/cache/issue-<N>-handle.json`` when the
canonical path is absent.

The orchestrator re-invokes after each bg-Bash exit (the harness
re-invocation model — see CLAUDE.md § "Orchestrator vs subagent
re-invocation"). KEEPING the bg-Bash poll loop as a separate process
is load-bearing: notification-on-exit IS the orchestrator's wakeup
signal. Moving poll in-process would break it.

For backend = ``runpod`` this script is functionally equivalent to
``poll_pipeline.py`` (RunPodBackend.poll delegates to
``poll_pipeline.poll_once``). For ``cluster``/``nibi``/``fir`` it
delegates to ``SlurmBackend.poll`` (which calls into
``backends.slurm_monitor.build_poll_result``). For ``gcp`` it
delegates to ``GcpBackend.poll`` (``gcloud compute instances describe``).
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from dataclasses import dataclass, replace
from pathlib import Path

# Repo-root sys.path bootstrap. Invoking this file as a script puts only
# scripts/ (the script's own dir) on sys.path — NOT the repo root — so the
# lazy import inside the RunPod backend (`backends/runpod.py` does
# `from scripts.poll_pipeline import ...`) fails with
# ``ModuleNotFoundError: No module named 'scripts'`` unless PYTHONPATH is
# set manually. Insert the repo root so the documented invocation
# (``uv run python scripts/backend_poll.py --issue <N>``) works from any
# cwd (incident #571, 2026-06-11: first pod tick crashed exit 1).
_REPO_ROOT = str(Path(__file__).resolve().parents[1])
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

# Conservative short bg-poll interval (seconds). Mirrors
# ``scripts.poll_pipeline.POLL_INTERVAL_DEFAULT_SEC`` — kept as a local
# literal so the fast ``--help`` path doesn't import the (heavy)
# poll_pipeline module at startup. Used for results that don't carry a
# ``next_interval`` and for the missing-sidecar terminal JSON.
_DEFAULT_NEXT_INTERVAL_SEC = 540

# The ``current_phase`` ``GcpBackend.poll`` produces for a true GCP WORKLOAD
# crash (``eps/phase==failed`` AND the ``eps/workload_started`` sentinel
# present; #659 MF3). The async-failover predicate matches THIS phase EXACTLY.
# A GCP setup/boot/secrets/uv-sync failure surfaces ``terminal_setup_failed``
# (sentinel absent) — a DIFFERENT phase that the predicate excludes, so a
# broken-boot VM never fails over to RunPod (re-running it there just
# re-crashes; §7 kill-criterion #1). Kept as a local literal in lock-step with
# ``gcp._terminal_dead_poll(reason="workload_failed")`` → ``f"terminal_{reason}"``;
# the §6 GCP poll-discrimination test pins both ends so a future GCP-phase
# rename breaks a test, not production silently.
GCP_WORKLOAD_FAILED_PHASE = "terminal_workload_failed"

# The ``current_phase`` the POLLER synthesizes (#669) when a GCP VM is
# RUNNING-but-hung: ``eps/phase`` frozen at a non-terminal value past the
# staleness floor AND a TRANSPORT-class reachability alarm. The async-failover
# accept-set recognizes it so the dead-but-hung VM fails over to RunPod.
GCP_WORKLOAD_WEDGED_PHASE = "terminal_workload_wedged"

# The ``current_phase`` ``GcpBackend.poll`` produces (#669) when a TERMINATED
# VM's in-VM reachability watchdog wrote ``eps/phase=wedged`` before its
# self-``shutdown`` — the conservative phase that distinguishes a watchdog
# self-terminate from an ordinary ``terminal_terminated`` (spot preemption /
# max-run-duration / manual stop, which stays EXCLUDED from the accept-set so
# there is no spot regression). Kept in lock-step with
# ``gcp._terminal_dead_poll(reason="wedged_terminated")`` -> ``f"terminal_{reason}"``.
GCP_WEDGED_TERMINATED_PHASE = "terminal_wedged_terminated"

# Frozen-phase staleness floor (#669, plan §11). A non-terminal ``eps/phase``
# unchanged for longer than this AND a transport-class reachability alarm
# escalates to ``terminal_workload_wedged``. Bounded BELOW by the #667 wedge
# window (~32 min before manual detection) and ABOVE by normal inter-phase
# gaps + the 30-min quiet poll interval, so a healthy long-running phase never
# trips and at least one poll re-evaluates within the floor.
GCP_STALENESS_FLOOR_SEC = 900  # 15 min

# The async-failover accept-set (#669): the #659 crashed-workload phase PLUS
# the two #669 wedge phases. ``terminal_terminated`` is DELIBERATELY EXCLUDED
# (Consistency-checker Option 2) so spot preemption / max-run-duration / manual
# stop behave EXACTLY as today (straight to dead, no failover).
_GCP_ASYNC_FAILOVER_PHASES = frozenset(
    {
        GCP_WORKLOAD_FAILED_PHASE,  # #659 crashed workload
        GCP_WORKLOAD_WEDGED_PHASE,  # #669 poller-detected frozen-phase wedge
        GCP_WEDGED_TERMINATED_PHASE,  # #669 watchdog self-shutdown
    }
)

# ── RunPod RUNNING-but-no-port host wedge (#664/#689) ─────────────────────────
# A RunPod pod whose ``desiredStatus`` stays RUNNING with null/empty
# ``runtime.ports`` past this floor is on a degraded host: ``resume_pod`` is
# host-pinned (sends ``podResume{podId, gpuCount}`` only — no host reselection),
# so a stop+resume returns to the SAME dead host. K mirrors
# ``GCP_STALENESS_FLOOR_SEC`` (900s) and sits ABOVE ``wait_for_ssh``'s 600s
# window + one retry margin, so a healthy mid-resume pod never trips. Env-
# overridable via ``EPM_RUNPOD_WEDGE_K_SEC`` (an attempt-floor in seconds, not a
# dollar cap).
RUNPOD_WEDGE_K_SEC = int(os.environ.get("EPM_RUNPOD_WEDGE_K_SEC", "900"))

# The synthesized terminal ``current_phase`` for a matured RunPod no-port wedge
# (#664). ``_is_runpod_async_wedge_failure`` matches THIS phase EXACTLY; it is
# set ONLY by ``_maybe_escalate_runpod_wedge`` once a wedge is past K.
RUNPOD_WORKLOAD_WEDGED_PHASE = "terminal_runpod_no_port_wedged"

# The NON-terminal ``current_phase`` the poller synthesizes WITHIN the K floor
# (#689 A1-residue): an SSH-dead poll on a confirmed RUNNING+no-port pod is
# REWRITTEN to ``status="running"`` with this phase so the orchestrator keeps
# polling until the wedge matures (``poll_once`` returns ``status="dead"`` on
# SSH-probe failure — a bare pass-through would stop the orchestrator before K
# and the wedge would never be observed).
RUNPOD_WORKLOAD_OBSERVED_PHASE = "runpod_no_port_observed"


def _resolve_backend(name: str):
    """Map ``handle.backend`` to a ComputeBackend instance.

    Each backend's constructor takes no required args; defaults match
    the production wiring (default config, real runner, real marker
    poster). A future extension might thread per-call config in via a
    sidecar — for slice 6 the defaults suffice.
    """
    if name == "runpod":
        from explore_persona_space.backends.runpod import RunPodBackend

        return RunPodBackend()
    if name in {"cluster", "nibi", "fir", "mila"}:
        from explore_persona_space.backends.slurm import SlurmBackend

        return SlurmBackend()
    if name == "gcp":
        from explore_persona_space.backends.gcp import GcpBackend

        return GcpBackend()
    raise ValueError(f"backend_poll: unknown backend {name!r}; cannot resolve a backend class")


def _serialize_poll_result(result) -> dict:
    """Serialize a PollResult to the canonical JSON shape.

    Matches ``scripts/poll_pipeline.py.main``'s output keys so the
    orchestrator's parser is interchangeable. Field set held in sync
    with ``backends.base.PollResult`` + ``scripts.poll_pipeline.PollResult``.
    """
    return {
        "status": result.status,
        "current_phase": result.current_phase,
        "new_milestone": result.new_milestone,
        "last_log_mtime_sec_ago": result.last_log_mtime_sec_ago,
        "pid_alive": result.pid_alive,
        "log_tail_excerpt": result.log_tail_excerpt,
        "gate": result.gate,
        "sentinels_processed": result.sentinels_processed,
        "phase_log_mtime_sec_ago": result.phase_log_mtime_sec_ago,
        "shard_log_mtime_sec_ago": result.shard_log_mtime_sec_ago,
        "gpu_util": result.gpu_util,
        # Adaptive bg-poll interval (anti-stall redesign §7): the
        # orchestrator's sleep-chain uses this for the NEXT `sleep
        # <interval>` (SKILL.md Step 6d.2; 540s fallback when absent).
        # ``getattr`` defends against a duck-typed / older-module result
        # that predates the field — mixed-version worktree copies degrade
        # to the conservative short interval, never crash the poll.
        "next_interval": int(getattr(result, "next_interval", _DEFAULT_NEXT_INTERVAL_SEC)),
    }


def _missing_sidecar_json(issue: int, sidecar_path: Path, reason: str) -> dict:
    """Build the failure-shape JSON line for a missing / unreadable sidecar.

    On a missing or unreadable sidecar (typically pre-launch, between
    crash + relaunch, or a worktree that was reaped before the
    orchestrator re-armed), historically this script raised
    ``FileNotFoundError`` and the bg-Bash poll loop produced EMPTY
    stdout. The orchestrator's JSON-line parser then looped on "stalled"
    forever (no JSON to parse → no terminal signal). FIX: print ONE
    canonical JSON line shaped as a ``PollResult`` ``status: "dead"``
    plus the ``failure_class: "infra"`` + ``reason`` keys the
    orchestrator's failure-classifier reads, so the next bg-Bash exit
    converts it into ``epm:failure v1 failure_class: infra reason:
    missing_handle_sidecar`` and the loop terminates cleanly.

    Defense-in-depth: even after ``scripts/dispatch_issue.py launch``
    makes the sidecar always present on a successful launch, an
    orchestrator that polls BEFORE launch completes (race) or after a
    worktree-reap (stale cache dir) still needs a terminal JSON to
    break the bg-Bash loop.
    """
    return {
        # Legacy poll_pipeline JSON-line keys (orchestrator parser
        # contract — same fields backend_poll.py emits on success).
        "status": "dead",
        "current_phase": "missing-sidecar",
        "new_milestone": False,
        "last_log_mtime_sec_ago": 10**9,
        "pid_alive": False,
        "log_tail_excerpt": f"backend_poll: {reason} at {sidecar_path}",
        "gate": None,
        "sentinels_processed": 0,
        "phase_log_mtime_sec_ago": 10**9,
        "shard_log_mtime_sec_ago": 10**9,
        "gpu_util": "unknown",
        # Terminal verdict — the orchestrator stops the loop, so the
        # interval is moot, but the key stays present (short default) so
        # the JSON shape is uniform across every emitted line (§7).
        "next_interval": _DEFAULT_NEXT_INTERVAL_SEC,
        # Failure-classifier hint keys — the orchestrator reads these
        # alongside ``status: "dead"`` to post ``epm:failure v1`` with
        # the matching failure_class instead of a generic "workload
        # died".
        "failure_class": "infra",
        "reason": "missing_handle_sidecar",
        "issue": int(issue),
    }


def _is_gcp_async_workload_failure(handle, result) -> bool:
    """True ONLY for a GCP handle whose poll surfaced a failover-eligible death (#659, #669).

    Narrow BY CONSTRUCTION:

    * ``handle.backend`` must be exactly ``"gcp"`` — a SLURM (``nibi`` / ``fir``
      / ``mila``) or RunPod handle never trips it (scope discipline §5; the
      RunPod exclusion is the "exactly once" structural-bound guard: a RunPod
      re-crash polls a RunPod handle, so it can never re-enter the failover);
    * ``result.status`` must be ``"dead"``;
    * ``result.current_phase`` must be in :data:`_GCP_ASYNC_FAILOVER_PHASES`:

      - :data:`GCP_WORKLOAD_FAILED_PHASE` (``"terminal_workload_failed"``) — the
        #659 ``gcp.poll`` signal AFTER the §4.1.0b workload-started
        discrimination (a real workload crash);
      - :data:`GCP_WORKLOAD_WEDGED_PHASE` (``"terminal_workload_wedged"``) — the
        #669 POLLER-synthesized frozen-phase + reachability-timeout wedge;
      - :data:`GCP_WEDGED_TERMINATED_PHASE` (``"terminal_wedged_terminated"``) —
        the #669 in-VM watchdog self-shutdown (wrote ``eps/phase=wedged``).

    A GCP setup/boot/secrets/uv-sync failure surfaces ``"terminal_setup_failed"``
    (DIFFERENT phase → excluded); a GCP instance-gone / capacity / quota death
    surfaces ``"terminal_instance not found"`` / ``"terminal_terminated"``.
    ``terminal_terminated`` is DELIBERATELY EXCLUDED (#669 Consistency-checker
    Option 2): spot preemption / max-run-duration / manual stop must NOT fail
    over — re-running on RunPod there burns money for no reason and would invert
    ``test_async_gcp_capacity_death_does_NOT_fail_over``.

    A CPU-only GCP handle (``extra["gpu_count"] == 0``, #677) is EXCLUDED
    regardless of phase: RunPod is GPU-only (``resolve_intent`` on a CPU intent
    KeyErrors; ``runpod_api`` asserts ``gpu_count >= 1``), so a CPU workload
    crash/wedge must NOT fail over. Returning ``False`` routes it to the
    ordinary dead path -> ``failure_class: code`` -> ``status:blocked`` (the
    watcher's capacity-retry pass re-drives only infra/``no_compute_available``,
    never a code failure, so it parks cleanly). A pre-#677 GCP handle written
    before the ``gpu_count`` threading lands has no ``gpu_count`` key ->
    ``extra.get("gpu_count")`` is ``None`` != ``0`` -> the guard is a no-op and
    the handle takes the EXISTING (GPU) failover path, exactly as today
    (fail-toward-existing-behavior on a missing key).
    """
    extra = getattr(handle, "extra", None) or {}
    if extra.get("gpu_count") == 0:
        return False
    return (
        getattr(handle, "backend", None) == "gcp"
        and result.status == "dead"
        and result.current_phase in _GCP_ASYNC_FAILOVER_PHASES
    )


def _is_runpod_async_wedge_failure(handle, result) -> bool:
    """True ONLY for a RunPod handle whose poll surfaced the #664 no-port wedge.

    Narrow by construction (the sibling of :func:`_is_gcp_async_workload_failure`):
    ``handle.backend == "runpod"`` AND ``result.status == "dead"`` AND
    ``result.current_phase == RUNPOD_WORKLOAD_WEDGED_PHASE``. A GCP / SLURM handle
    never trips it; the wedged phase is set ONLY by
    :func:`_maybe_escalate_runpod_wedge` once a RUNNING+no-port pod matures past K.
    """
    # TODO(#689): round-2 implementation; stub for TDD round-1 import resolution.
    raise NotImplementedError("#689 round 2: _is_runpod_async_wedge_failure")


def _read_phase_clock(sidecar: Path) -> tuple[str | None, float | None]:
    """Read the (last_phase, last_phase_change_ts) staleness clock from the sidecar (#669).

    The clock keys live inside the sidecar JSON's ``extra`` dict (round-trips
    via ``serialize_handle``'s ``dict(handle.extra)``, so no schema migration).
    A freshly-dispatched sidecar (pre-this-change handle) has neither key, which
    reads as ``(None, None)`` → the caller fails toward ``running``, never
    toward a false wedge. A missing / unreadable / malformed sidecar also reads
    as ``(None, None)`` — the clock can never crash the poll.
    """
    try:
        payload = json.loads(Path(sidecar).read_text())
    except (OSError, json.JSONDecodeError, ValueError):
        return None, None
    extra = payload.get("extra") if isinstance(payload, dict) else None
    if not isinstance(extra, dict):
        return None, None
    last_phase = extra.get("last_phase")
    last_ts = extra.get("last_phase_change_ts")
    last_phase = str(last_phase) if isinstance(last_phase, str) else None
    last_ts = float(last_ts) if isinstance(last_ts, (int, float)) else None
    return last_phase, last_ts


def _write_phase_clock(sidecar: Path, *, phase: str, ts: float) -> None:
    """Persist the staleness clock onto the sidecar's ``extra`` dict, best-effort (#669).

    Mutates ONLY ``extra["last_phase"]`` / ``extra["last_phase_change_ts"]`` on
    the raw sidecar JSON (every other handle field is preserved verbatim) and
    rewrites atomically (write-temp + rename). A write failure (EDQUOT /
    read-only fs) is logged, NOT raised — the next tick simply re-reads stale
    state and either re-stamps (phase advanced) or re-evaluates the floor. A
    clock-write failure can NEVER crash the poll or manufacture a wedge (the
    floor + reachability-alarm conjunction still gates the escalation).
    """
    try:
        path = Path(sidecar)
        payload = json.loads(path.read_text())
        if not isinstance(payload, dict):
            return
        extra = payload.get("extra")
        if not isinstance(extra, dict):
            extra = {}
            payload["extra"] = extra
        extra["last_phase"] = phase
        extra["last_phase_change_ts"] = float(ts)
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(json.dumps(payload, sort_keys=True, indent=2))
        tmp.replace(path)
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        logging.warning(
            "backend_poll: phase-clock write failed for %s (%s: %s); "
            "the next tick re-reads stale state (never manufactures a wedge)",
            sidecar,
            type(exc).__name__,
            exc,
        )


def _maybe_escalate_gcp_wedge(handle, result, sidecar: Path, *, now: float):
    """Escalate a frozen non-terminal GCP phase + REACHABILITY alarm to terminal wedged (#669).

    Poller-side staleness clock (kept in the poller, NOT inside ``poll()``, so
    ``GcpBackend.poll`` stays a pure function of ``(handle, gcloud responses)``
    and its whole test suite is unaffected). Compares the observed running phase
    to the sidecar's ``last_phase`` / ``last_phase_change_ts`` and returns the
    SAME ``result`` UNLESS:

      (phase unchanged past :data:`GCP_STALENESS_FLOOR_SEC`)
      AND (``result.reachability_alarm`` — the TRANSPORT-class drain failure,
           NOT the sentinel-processing class, M1)

    in which case it rewrites ``status -> "dead"`` /
    ``current_phase -> terminal_workload_wedged`` so
    :func:`_is_gcp_async_workload_failure` matches. Side effect: re-stamps the
    sidecar phase clock when the phase changed (or on the first observation).

    The false-positive guards (return ``result`` unchanged):

    * not a GCP handle, or the poll is not ``running`` → return unchanged (a
      terminal / gate / stalled tick is acted on directly; a ``terminal_terminated``
      capacity death is NEVER touched here, so the existing #659 no-failover
      test is unaffected);
    * phase advanced, or first observation (``last_ts is None``) → re-stamp,
      return ``running`` (fail-open on a fresh-dispatch handle with no clock);
    * phase unchanged but within the floor, OR no reachability alarm → return
      ``running`` (covers BOTH the recency case AND the sentinel-processing
      class, since the latter leaves ``reachability_alarm`` False).
    """
    if getattr(handle, "backend", None) != "gcp" or result.status != "running":
        return result
    last_phase, last_ts = _read_phase_clock(sidecar)
    if last_phase != result.current_phase or last_ts is None:
        # Phase advanced (or first observation) → re-stamp the clock.
        _write_phase_clock(sidecar, phase=result.current_phase, ts=now)
        return result
    stale_for = now - last_ts
    if stale_for > GCP_STALENESS_FLOOR_SEC and getattr(result, "reachability_alarm", False):
        logging.warning(
            "backend_poll: GCP issue phase %r frozen for %.0fs (>%ds floor) WITH a "
            "transport-class reachability alarm — escalating to %s (#669 hung-VM wedge)",
            result.current_phase,
            stale_for,
            GCP_STALENESS_FLOOR_SEC,
            GCP_WORKLOAD_WEDGED_PHASE,
        )
        return replace(
            result,
            status="dead",
            current_phase=GCP_WORKLOAD_WEDGED_PHASE,
            new_milestone=True,
            pid_alive=False,
        )
    # Within the floor, OR no reachability alarm → stays running (false-positive guard).
    return result


# ── RunPod no-port wedge: clock helpers + escalation + failover (#664/#689) ───
# Siblings of the GCP wedge machinery above. The clock helpers mirror
# ``_read_phase_clock`` / ``_write_phase_clock`` EXACTLY (atomic tmp+rename,
# never-raise on a malformed/non-numeric value — the S2 fail-soft contract),
# keyed ``runpod_noport_first_seen_ts`` (distinct from the GCP
# ``last_phase_change_ts``). The escalation/failover stubs land their bodies in
# round 2; the clock contract is pinned by ``test_runpod_wedge_detection.py``.


def _read_runpod_noport_clock(sidecar: Path) -> float | None:
    """Read the RunPod no-port staleness clock (epoch seconds) from the sidecar.

    Mirrors :func:`_read_phase_clock`'s fail-soft contract EXACTLY: a missing /
    unreadable / malformed JSON sidecar OR a non-numeric
    ``extra["runpod_noport_first_seen_ts"]`` reads as ``None`` (treated as "no
    clock yet" -> re-stamp), NEVER raises. The S2 test pins this.
    """
    # TODO(#689): round-2 implementation; stub for TDD round-1 import resolution.
    raise NotImplementedError("#689 round 2: _read_runpod_noport_clock")


def _write_runpod_noport_clock(sidecar: Path, *, ts: float) -> None:
    """Persist the RunPod no-port clock onto the sidecar ``extra`` dict, atomically.

    Mirrors :func:`_write_phase_clock`: mutates ONLY
    ``extra["runpod_noport_first_seen_ts"]`` (every other field preserved
    verbatim), write-temp + rename, and a write failure is LOGGED not raised.
    """
    # TODO(#689): round-2 implementation; stub for TDD round-1 import resolution.
    raise NotImplementedError("#689 round 2: _write_runpod_noport_clock")


def _clear_runpod_noport_clock(sidecar: Path) -> None:
    """Remove the RunPod no-port clock key from the sidecar (healthy/terminal).

    Called when the live pod exposes a public port or leaves RUNNING — the wedge
    never matured, so the next observation re-stamps from scratch.
    """
    # TODO(#689): round-2 implementation; stub for TDD round-1 import resolution.
    raise NotImplementedError("#689 round 2: _clear_runpod_noport_clock")


def _maybe_escalate_runpod_wedge(handle, result, sidecar: Path, *, now: float):
    """Escalate a RunPod pod RUNNING-but-no-port past K to terminal wedged (#664).

    Sibling of :func:`_maybe_escalate_gcp_wedge`: a poller-side staleness clock
    kept in the poller (NOT inside ``RunPodBackend.poll``, so poll stays a pure
    function). Reads the live pod by name (``runpod_api.get_pod_by_name``):

    * not a RunPod handle -> return ``result`` unchanged;
    * the live pod is gone, has left RUNNING, OR exposes a public port (healthy)
      -> clear the no-port clock, return ``result`` unchanged;
    * RUNNING + no public port WITHIN the K floor (A1-residue) -> OVERRIDE an
      SSH-dead / stalled poll into ``status="running",
      current_phase=RUNPOD_WORKLOAD_OBSERVED_PHASE`` so polling continues until
      the wedge matures (a bare ``return result`` would leave ``status="dead"``
      and the orchestrator would stop before K — the B.4 ``status="dead"`` input
      pins this);
    * RUNNING + no public port PAST K -> rewrite to ``status="dead",
      current_phase=RUNPOD_WORKLOAD_WEDGED_PHASE`` so
      :func:`_is_runpod_async_wedge_failure` matches.
    """
    # TODO(#689): round-2 implementation; stub for TDD round-1 import resolution.
    raise NotImplementedError("#689 round 2: _maybe_escalate_runpod_wedge")


@dataclass
class _WedgeInputsGate:
    """Per-cell three-state inputs-on-HF gate result for the auto-terminate (M1).

    ``ok`` = safe to terminate (ZERO partial cells). ``complete`` cells have both
    raw+store EXACT sets on HF; ``partial`` cells have exactly one artifact-kind
    (BLOCK the terminate); ``absent`` cells are not-yet-run selected cells
    (rerunnable on the fresh pod, do NOT block).
    """

    ok: bool
    complete: list[str]
    partial: list[str]
    absent: list[str]


def _issue_cells_for_handle(issue: int, handle) -> list:
    """Resolve the selected cell list for a run handle (#664: the realized grid).

    For #664 imports ``issue664_common.realized_grid()`` behind an
    ``issue == 664`` guard; a non-#664 issue returns ``[]`` (the adapters-only
    path, whose inputs are inline-verified by ``train_lora``).
    """
    # TODO(#689): round-2 implementation; stub for TDD round-1 import resolution.
    raise NotImplementedError("#689 round 2: _issue_cells_for_handle")


def _wedged_run_inputs_on_hf(issue: int, handle) -> _WedgeInputsGate:
    """Per-cell three-state inputs-on-HF gate for the irreversible auto-terminate.

    Classifies each selected cell ``complete | partial | absent`` from ONE fresh
    ``list_repo_files`` (EXACT expected file set per S1, not prefix-presence).
    Terminate is allowed iff there are ZERO partial cells — a COMPLETE cell's
    data is preserved, a not-yet-run ABSENT cell is rerunnable, and only a
    half-uploaded PARTIAL cell would lose recoverable work.
    """
    # TODO(#689): round-2 implementation; stub for TDD round-1 import resolution.
    raise NotImplementedError("#689 round 2: _wedged_run_inputs_on_hf")


def _runpod_wedge_already_handled(issue: int, handle, sidecar: Path) -> bool:
    """Idempotency short-circuit for a re-fired RunPod wedge failover (#664).

    Mirrors the GCP failover's sentinel + durable-lease guard (keyed to the
    wedged pod_id) so a second tick on the OLD handle never double-terminates /
    double-provisions.
    """
    # TODO(#689): round-2 implementation; stub for TDD round-1 import resolution.
    raise NotImplementedError("#689 round 2: _runpod_wedge_already_handled")


def _relaunch_fresh_runpod(*, issue: int, handle, result, sidecar: Path) -> dict:
    """Re-provision a FRESH RunPod pod + resume the dispatcher idempotently.

    Re-uses the router's ``failover_to_runpod_after_async_workload_crash`` (the
    same function ``_failover_dead_gcp_to_runpod`` calls): reconstruct a RunSpec
    from the handle, launch a fresh RunPod run (NEW host, NOT a host-pinned
    resume), re-point the handle sidecar, durable-lease guarded. The fresh pod's
    P2 dispatcher skips HF-complete cells via A2's ``_cell_done_anywhere``.
    """
    # TODO(#689): round-2 implementation; stub for TDD round-1 import resolution.
    raise NotImplementedError("#689 round 2: _relaunch_fresh_runpod")


def _failover_wedged_runpod(*, issue: int, handle, result, sidecar: Path) -> dict:
    """Terminate a wedged RunPod pod + re-provision fresh, idempotently (#664).

    PRECONDITION (fix (a) dependency): the per-cell inputs-on-HF gate (M1) must
    pass BEFORE the terminate fires. If the gate finds ANY PARTIAL cell, do NOT
    terminate — return a terminal infra JSON
    (``reason="runpod_wedge_inputs_unverified"``) so a human decides (CLAUDE.md
    halt-criterion #2). Idempotency is a durable lease + sidecar sentinel keyed
    to the wedged pod_id (``_runpod_wedge_already_handled``).
    """
    # TODO(#689): round-2 implementation; stub for TDD round-1 import resolution.
    raise NotImplementedError("#689 round 2: _failover_wedged_runpod")


def _failover_sentinel_path(sidecar: Path) -> Path:
    """The idempotency sentinel path for a GCP->RunPod async failover (#659, MF4).

    Derived from the RESOLVED sidecar path (a sibling file in the same
    ``.claude/cache/`` dir), so it is cwd-INDEPENDENT for free — wherever the
    sidecar resolved to (canonical ``<main-checkout>/.claude/cache/`` or the
    legacy ``<cwd>/.claude/cache/`` probe), the sentinel lives right next to it
    and the same poll-tick resolution finds both. Naming mirrors the handle
    sidecar: ``issue-<N>-handle.json`` -> ``issue-<N>-failover-persistence-failed.json``.
    """
    name = sidecar.name
    stem = name[: -len("-handle.json")] if name.endswith("-handle.json") else sidecar.stem
    return sidecar.with_name(f"{stem}-failover-persistence-failed.json")


def _gcp_handle_identity(handle) -> dict:
    """The (pod_name, job_id) identity of a GCP handle, for sentinel matching.

    The sentinel records the identity of the GCP run that ALREADY launched a
    RunPod failover. A later, genuinely-NEW GCP run (a fresh dispatch writes a
    fresh sidecar with a NEW pod_name/job_id) must NOT be suppressed by a stale
    sentinel from a prior crash episode — so the short-circuit fires ONLY when
    the current GCP handle's identity matches the sentinel's recorded identity.
    """
    return {
        "pod_name": getattr(handle, "pod_name", None),
        "job_id": getattr(handle, "job_id", None),
    }


def _read_failover_sentinel(sentinel: Path) -> dict | None:
    """Read the failover sentinel; return its dict body or ``None`` if absent/unreadable.

    A corrupted/unreadable sentinel is treated as ABSENT (return ``None``) so a
    garbage file can never permanently wedge the failover — the worst case is
    one extra RunPod launch, never a silent suppression of a legitimate retry.
    """
    if not sentinel.exists():
        return None
    try:
        body = json.loads(sentinel.read_text())
    except (OSError, json.JSONDecodeError, ValueError):
        return None
    return body if isinstance(body, dict) else None


def _write_failover_sentinel(sentinel: Path, *, issue: int, handle, reason: str) -> None:
    """Persist the failover sentinel atomically (write-temp + rename), best-effort.

    Records the GCP run identity (pod_name/job_id) so a subsequent poll on the
    SAME crashed GCP handle short-circuits (no second RunPod launch), while a
    genuinely-new GCP run is unaffected. Best-effort: a sentinel write that
    itself fails (the disk is already failing — this is the EDQUOT path) does
    NOT raise; the terminal infra JSON is still emitted. The cost of a missed
    sentinel write is at most one extra RunPod launch on the next tick, which
    the ``recovered.backend != "runpod"`` guard still catches before emitting
    ``running`` — so the worst case degrades to the pre-fix behavior, never
    worse.
    """
    try:
        sentinel.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "issue": int(issue),
            "reason": reason,
            "gcp": _gcp_handle_identity(handle),
        }
        tmp = sentinel.with_suffix(sentinel.suffix + ".tmp")
        tmp.write_text(json.dumps(payload, sort_keys=True, indent=2))
        tmp.replace(sentinel)
    except OSError as exc:
        logging.warning(
            "backend_poll: failed to write failover idempotency sentinel %s (%s: %s); "
            "the recovered.backend guard still bounds the relaunch",
            sentinel,
            type(exc).__name__,
            exc,
        )


def _clear_failover_sentinel(sentinel: Path) -> None:
    """Remove the failover sentinel if present (best-effort).

    Called once the sidecar is AUTHORITATIVELY re-pointed at a RunPod handle —
    the failover succeeded, so any stale sentinel from a prior persistence
    failure on this issue must not suppress a FUTURE legitimate GCP failover.
    """
    try:
        sentinel.unlink(missing_ok=True)
    except OSError as exc:
        logging.warning(
            "backend_poll: failed to clear failover sentinel %s (%s: %s)",
            sentinel,
            type(exc).__name__,
            exc,
        )


def _lease_records_failover_of(issue: int, handle, *, lease_store=None) -> bool:
    """True iff the DURABLE lease already records a RunPod failover of THIS GCP run (#659 r3).

    The AUTHORITATIVE idempotency check, independent of the ``.claude/cache``
    sidecar AND sentinel (both share that dir, so an EDQUOT / read-only-fs /
    out-of-inodes failure that fails the sidecar write fails the sentinel write
    too — the round-2 sentinel-only fix degraded to one extra paid launch PER
    POLL TICK under that persistent-disk-failure mode). The lease lives at
    ``~/.eps-routing/`` (a DIFFERENT directory; ``LeaseStore`` default), so a
    failing ``.claude/cache`` mount does not also fail the lease, and the
    per-issue flock serializes a concurrent poll.

    Keyed to the GCP run's stable identity (``pod_name``/``job_id``), so it
    fires "exactly once PER GCP CRASH", NOT "exactly once per issue": a
    genuinely-new GCP run on the same issue (fresh dispatch -> new pod_name)
    writes a fresh lease that does NOT match a prior failover stamp, so it
    still gets its own single failover.

    A LeaseStore failure (no ``$HOME``, dir uncreatable) is treated as
    "no record" (return ``False``) — the worst case is one extra RunPod launch
    (the same bound the sentinel fast-path and the ``recovered.backend`` guard
    already provide), NEVER a silent suppression of a legitimate retry.
    """
    from explore_persona_space.backends.router import LeaseStore

    store = lease_store or LeaseStore()
    try:
        lease = store.read(int(issue))
    except OSError as exc:
        logging.warning(
            "backend_poll: lease-store read failed for issue %s (%s: %s); "
            "treating as no prior-failover record (the sentinel/backend guards still bound it)",
            issue,
            type(exc).__name__,
            exc,
        )
        return False
    if lease is None or lease.backend != "runpod":
        return False
    return lease.gcp_failover_of == _gcp_handle_identity(handle)


def _stamp_lease_failover_of(issue: int, handle, *, lease_store=None) -> None:
    """Record on the DURABLE lease that a RunPod failover of THIS GCP run launched (#659 r3).

    Called IMMEDIATELY after the router's RunPod launch SUCCEEDS, BEFORE the
    ``.claude/cache`` sidecar write — so the authoritative idempotency record
    lands at ``~/.eps-routing/`` regardless of whether the subsequent sidecar
    write fails under EDQUOT. Stamps ``gcp_failover_of`` onto the lease the
    router ALREADY wrote inside its own ``store.transaction`` (a read-modify-
    write under the per-issue flock that preserves backend/job_id/attempt_id).

    Best-effort about the LeaseStore itself: a write failure (no ``$HOME``, dir
    uncreatable) is logged, not raised — the failover already launched, and the
    sentinel fast-path + ``recovered.backend`` guard still bound the relaunch.
    The lease is the SAFETY NET for the EDQUOT-on-.claude/cache mode, not a hard
    precondition that can itself block the failover.
    """
    from explore_persona_space.backends.router import LeaseStore

    store = lease_store or LeaseStore()
    identity = _gcp_handle_identity(handle)
    try:
        with store.transaction(int(issue)) as (lease, write):
            if lease is None:
                # The router's lease write should have just landed; if it is
                # somehow absent, there is nothing to stamp authoritatively —
                # the sentinel fast-path remains the fallback record.
                logging.warning(
                    "backend_poll: no lease present to stamp gcp_failover_of for issue %s; "
                    "relying on the sentinel/backend guards",
                    issue,
                )
                return
            lease.gcp_failover_of = identity
            write(lease)
    except OSError as exc:
        logging.warning(
            "backend_poll: lease-store stamp failed for issue %s (%s: %s); "
            "the sentinel/backend guards still bound the relaunch",
            issue,
            type(exc).__name__,
            exc,
        )


def _runspec_from_gcp_handle(handle, issue):
    """Reconstruct the ``RunSpec`` for the RunPod re-launch from the GCP handle.

    After the §4.1.0 spec-threading sub-change lands, ``handle.extra`` carries
    ``intent``, ``gpus``, ``time_budget_hours``, ``hydra_args`` (list/tuple),
    and ``workload_cmd`` (str) — exactly the fields a minimal ``RunSpec``
    needs. Reads BOTH ``workload_cmd`` (a ``str``, passed through verbatim — NO
    ``tuple(...)``/``list(...)`` coercion, MF1) AND ``hydra_args`` (coerced back
    to a tuple) and passes EACH THROUGH VERBATIM: one is empty by construction
    (the GCP run used exactly one branch), so ``RunSpec.__post_init__``'s mutual
    exclusion holds with NO placeholder substituted into the unused branch
    (MF2).

    FAILS LOUD (raises ``ValueError``) on a pre-#659 handle that lacks the
    workload command — it NEVER silently launches a blank RunPod job (the
    §4.1.0 spec-threading is a HARD PREREQUISITE; the fact-checker confirmed
    A7=WRONG, the pre-#659 ``extra`` did not carry it).
    """
    from explore_persona_space.backends.base import RunSpec

    extra = handle.extra or {}
    if "workload_cmd" not in extra or "hydra_args" not in extra:
        raise ValueError(
            f"GCP handle for issue {issue} lacks workload_cmd/hydra_args in extra "
            f"(pre-#659 handle?); cannot reconstruct a RunSpec for the RunPod failover. "
            f"Refusing to launch a blank RunPod job."
        )
    workload_cmd = extra["workload_cmd"]  # str, verbatim (MF1)
    hydra_args = tuple(extra["hydra_args"])  # list/tuple -> tuple, verbatim
    return RunSpec(
        issue=int(issue),
        intent=extra.get("intent", "lora-7b"),
        backend="runpod",
        gpus=extra.get("gpus"),
        time_budget_hours=extra.get("time_budget_hours"),
        workload_cmd=workload_cmd,
        hydra_args=hydra_args,
    )


def _terminal_infra_json(*, issue: int, sidecar: Path, reason: str, log_tail: str) -> dict:
    """A ``status='dead'`` / ``failure_class='infra'`` poll JSON keyed by ``reason``.

    Mirrors :func:`_missing_sidecar_json`. Used for BOTH the RunPod-unavailable
    case (``reason='no_compute_available'`` — IN ``TRANSIENT_CAPACITY_REASONS``,
    so the watcher's capacity-retry pass re-drives it once a lane frees) AND the
    sidecar-persistence-failure case (``reason='sidecar_persistence_failed'`` —
    NOT in ``TRANSIENT_CAPACITY_REASONS``, so the watcher parks it at ``blocked``
    for human inspection rather than re-driving).
    """
    return {
        "status": "dead",
        "current_phase": f"terminal_{reason}",
        "new_milestone": True,
        "last_log_mtime_sec_ago": 10**9,
        "pid_alive": False,
        "log_tail_excerpt": log_tail,
        "gate": None,
        "sentinels_processed": 0,
        "phase_log_mtime_sec_ago": 10**9,
        "shard_log_mtime_sec_ago": 10**9,
        "gpu_util": "unknown",
        "next_interval": _DEFAULT_NEXT_INTERVAL_SEC,
        "failure_class": "infra",
        "reason": reason,
        "issue": int(issue),
    }


def _failover_dead_gcp_to_runpod(*, issue: int, handle, result, sidecar: Path) -> dict:
    """Run the RunPod terminal rung for a dead GCP workload (#659).

    Reconstructs a ``RunSpec`` from the GCP handle, launches the SAME RunPod
    terminal rung the sync failover uses, AUTHORITATIVELY re-points the handle
    sidecar at the new RunPod handle (write + readback), and returns a
    RUNNING-shaped poll JSON so the orchestrator keeps polling (the next tick
    reads the now-RunPod sidecar) instead of posting ``epm:failure`` for the GCP
    death. Returns a TERMINAL infra JSON instead if RunPod is unavailable
    (``no_compute_available``) or the sidecar persistence fails
    (``sidecar_persistence_failed``).
    """
    # Lazy imports — keep the --help path fast and match the patch targets the
    # poller tests monkeypatch (RunPodBackend from backends.runpod;
    # write/read_handle_sidecar from backends.issue_dispatch).
    from explore_persona_space.backends.issue_dispatch import (
        read_handle_sidecar,
        write_handle_sidecar,
    )
    from explore_persona_space.backends.router import (
        NoComputeAvailableError,
        failover_to_runpod_after_async_workload_crash,
    )
    from explore_persona_space.backends.runpod import RunPodBackend
    from explore_persona_space.backends.slurm import post_marker_via_task_py

    # IDEMPOTENCY SHORT-CIRCUIT (#659). A prior tick may have ALREADY launched a
    # RunPod failover for THIS crashed GCP run but failed to persist the
    # re-pointed sidecar (disk error / EDQUOT). In that case the sidecar still
    # holds the GCP handle, so this predicate re-fires on the next poll — but
    # launching a SECOND RunPod is a paid duplicate that breaches "exactly once".
    #
    # TWO records guard the relaunch, in the order they are checked:
    #
    #   1. DURABLE lease (AUTHORITATIVE, round-3 fix). The lease at
    #      ``~/.eps-routing/`` (a DIFFERENT directory from ``.claude/cache``) is
    #      stamped with ``gcp_failover_of`` BEFORE the sidecar write below, so it
    #      survives the EDQUOT / read-only-fs / out-of-inodes mode that fails the
    #      sidecar write AND the same-dir sentinel write together. It is the
    #      SAFETY NET that makes "exactly once per GCP crash" hold under a
    #      persistent ``.claude/cache`` disk failure. Round 2's sentinel-only fix
    #      degraded to one extra paid launch PER POLL TICK in that mode.
    #
    #   2. SENTINEL (OPTIMIZATION). The ``.claude/cache`` sentinel is the fast
    #      path — a sibling-file read that avoids the lease-store flock round-trip
    #      on the common case (no disk failure). Kept for that reason; it is no
    #      longer the safety guarantee.
    #
    # Both are keyed to the GCP run's pod_name/job_id, so a genuinely-new GCP run
    # (fresh dispatch -> new pod_name) does NOT match a stale record and still
    # gets its own one failover. The recovered.backend guard further down only
    # blocks emitting "running"; these two guards are what bound the RE-LAUNCH
    # across polls.
    sentinel = _failover_sentinel_path(sidecar)
    prior = _read_failover_sentinel(sentinel)
    sentinel_match = prior is not None and prior.get("gcp") == _gcp_handle_identity(handle)
    if sentinel_match or _lease_records_failover_of(issue, handle):
        return _terminal_infra_json(
            issue=issue,
            sidecar=sidecar,
            reason="sidecar_persistence_failed",
            log_tail=(
                f"GCP->RunPod failover for {handle.pod_name} ALREADY launched RunPod on a "
                f"prior tick but failed to persist the sidecar "
                f"({'sentinel ' + sentinel.name if sentinel_match else 'durable lease record'}); "
                f"refusing to launch a SECOND RunPod (exactly-once bound, #659)"
            ),
        )

    spec = _runspec_from_gcp_handle(handle, issue)
    try:
        route_result = failover_to_runpod_after_async_workload_crash(
            spec=spec,
            runpod_backend=RunPodBackend(),
            evidence={
                "source": "async_poller",
                "current_phase": result.current_phase,
                "log_tail_excerpt": result.log_tail_excerpt,
                "gcp_pod_name": handle.pod_name,
            },
            marker_poster=post_marker_via_task_py,
            # BEST-EFFORT in-route lease-mid-flight write (mirrors
            # dispatch_for_issue's on_launched). _invoke_on_launched SWALLOWS the
            # hook's exceptions (logged loud, not propagated), so this is NOT
            # authoritative — the post-route write/readback below is.
            on_launched=lambda h: write_handle_sidecar(h, sidecar),
            # M3b (#669): the GCP-crash identity this failover is OF, so the
            # router's in-flock re-check + stamp makes N CONCURRENT triggerers
            # (the #669 wedge classifier + the watchdog-TERMINATED path on the
            # same handle) launch RunPod exactly once. The OUTSIDE-the-flock
            # pre-check above (sentinel_match / _lease_records_failover_of) is
            # the cheap single-triggerer fast-path; this is the atomic guard.
            gcp_failover_of_identity=_gcp_handle_identity(handle),
        )
    except NoComputeAvailableError:
        # RunPod truly unavailable: terminal infra JSON with
        # reason=no_compute_available so the watcher's capacity-retry pass CAN
        # re-drive once a lane frees. Sidecar left pointing at the GCP handle.
        # No lease stamp here — nothing launched, so a later poll SHOULD retry.
        return _terminal_infra_json(
            issue=issue,
            sidecar=sidecar,
            reason="no_compute_available",
            log_tail=(
                f"GCP workload crash on {handle.pod_name}; RunPod also unavailable "
                f"(#659 async failover)"
            ),
        )

    # DURABLE IDEMPOTENCY STAMP (#659 round-3). RunPod has now launched. Stamp
    # ``gcp_failover_of`` onto the ``~/.eps-routing/`` lease the router just
    # wrote BEFORE attempting the ``.claude/cache`` sidecar write below — so the
    # authoritative "exactly once per GCP crash" record lands on a DIFFERENT
    # mountpoint than the sidecar. If the sidecar write then fails under EDQUOT
    # (and the same-dir sentinel write fails with it), the lease record still
    # survives and the next poll short-circuits at the lease check above instead
    # of firing a paid second RunPod launch. This ordering is the round-3 fix:
    # round 2 had no record that survived a ``.claude/cache``-wide disk failure.
    _stamp_lease_failover_of(issue, handle)

    # M3b 2nd-TRIGGERER SHORT-CIRCUIT (#669 code-review r1 blocker). When a
    # CONCURRENT triggerer already launched RunPod for this GCP crash, the
    # router's in-flock re-check returns WITHOUT launching and flags the result
    # ``extra["failover_already_launched"] = True`` carrying a MINIMAL
    # reconstructed handle (``extra={"issue": N}`` only — NO ``expected_artifacts``
    # / ``pid_file`` / ``runpod_attempt_id``). The 1st triggerer already wrote the
    # AUTHORITATIVE full RunPod handle to this sidecar, so the
    # ``write_handle_sidecar`` below would CLOBBER it with the minimal handle and
    # downstream artifact verification (``artifacts.expected_artifacts_from_handle``)
    # would then read ``expected_artifacts=None`` and FAIL. So: do NOT write —
    # READ the existing sidecar and preserve it. If it already holds a RunPod
    # handle (the expected state under the flock invariant), emit running pointing
    # at THAT untouched handle. If it still holds the GCP handle (the 1st
    # triggerer has not persisted yet — unlikely under the lease invariant but
    # possible across crashes), fall through to the terminal
    # ``sidecar_persistence_failed`` shape exactly as the no-readback case below.
    if route_result.extra.get("failover_already_launched"):
        try:
            existing = read_handle_sidecar(sidecar)
        except (OSError, json.JSONDecodeError, KeyError, ValueError):
            existing = None
        if existing is not None and existing.backend == "runpod":
            _clear_failover_sentinel(sentinel)
            return {
                "status": "running",
                "current_phase": "gcp_workload_failover_runpod_async",
                "new_milestone": True,
                "last_log_mtime_sec_ago": 0,
                "pid_alive": True,
                "log_tail_excerpt": (
                    f"GCP workload crash on {handle.pod_name}; a concurrent triggerer "
                    f"already failed over to RunPod {existing.pod_name} (M3b in-flock "
                    f"re-check, #669); preserved its full sidecar handle"
                ),
                "gate": None,
                "sentinels_processed": 0,
                "phase_log_mtime_sec_ago": 0,
                "shard_log_mtime_sec_ago": 0,
                "gpu_util": "unknown",
                "next_interval": _DEFAULT_NEXT_INTERVAL_SEC,
                "issue": int(issue),
            }
        # The 1st triggerer's full RunPod handle is NOT on disk yet. RunPod is
        # already launched (by that triggerer), so refuse to emit running rather
        # than re-pointing the sidecar at the MINIMAL reconstructed handle (which
        # would strip expected_artifacts). Persist the sentinel so the next poll
        # short-circuits at the outside-the-flock pre-check above.
        _write_failover_sentinel(
            sentinel, issue=issue, handle=handle, reason="sidecar_persistence_failed_concurrent"
        )
        return _terminal_infra_json(
            issue=issue,
            sidecar=sidecar,
            reason="sidecar_persistence_failed",
            log_tail=(
                f"GCP->RunPod failover: a concurrent triggerer already launched RunPod for "
                f"issue {issue} but its full handle is not yet on the sidecar "
                f"(backend={existing.backend if existing is not None else 'absent'!r}); "
                f"refusing to overwrite with the minimal reconstructed handle (would strip "
                f"expected_artifacts) and refusing to emit running"
            ),
        )

    # AUTHORITATIVE post-route sidecar write + readback (MF4). The on_launched
    # hook above is best-effort (its exceptions are swallowed by the router), so
    # re-point the sidecar HERE and PROVE it landed as a RunPod handle. WITHOUT
    # this, a swallowed on_launched failure would leave a GCP handle on disk
    # while the launch already succeeded; the next tick would re-read a GCP
    # handle, re-satisfy backend=="gcp", and fire a SECOND RunPod launch —
    # breaching "exactly once". So a re-pointed RunPod sidecar is a PRECONDITION
    # of emitting "running", not a hopeful side effect.
    try:
        write_handle_sidecar(route_result.handle, sidecar)
        recovered = read_handle_sidecar(sidecar)
    except (OSError, json.JSONDecodeError, KeyError, ValueError) as exc:
        # RunPod was ALREADY launched above; the sidecar could NOT be re-pointed.
        # Persist the idempotency sentinel BEFORE returning so the next poll on
        # the still-GCP handle short-circuits (no second RunPod launch).
        _write_failover_sentinel(
            sentinel, issue=issue, handle=handle, reason="sidecar_persistence_failed_write"
        )
        return _terminal_infra_json(
            issue=issue,
            sidecar=sidecar,
            reason="sidecar_persistence_failed",
            log_tail=(
                f"GCP->RunPod failover launched RunPod {route_result.handle.pod_name} "
                f"but sidecar persistence FAILED ({type(exc).__name__}: {exc}); refusing "
                f"to emit running (would re-launch RunPod next tick)"
            ),
        )
    if recovered.backend != "runpod":
        # RunPod was launched but the readback is not a RunPod handle (a
        # concurrent overwrite, or a write that silently no-op'd). Same hazard
        # as the raise above — persist the sentinel before returning so the
        # next poll does not fire a second launch.
        _write_failover_sentinel(
            sentinel, issue=issue, handle=handle, reason="sidecar_persistence_failed_readback"
        )
        return _terminal_infra_json(
            issue=issue,
            sidecar=sidecar,
            reason="sidecar_persistence_failed",
            log_tail=(
                f"GCP->RunPod failover: sidecar readback shows backend={recovered.backend!r}, "
                f"not 'runpod'; refusing to emit running (would re-launch RunPod next tick)"
            ),
        )

    # Sidecar AUTHORITATIVELY re-pointed at a RunPod handle -> the failover
    # succeeded. Clear any stale sentinel so a FUTURE legitimate GCP failover on
    # this issue is not suppressed.
    _clear_failover_sentinel(sentinel)

    # Sidecar is now an AUTHORITATIVE RunPod handle on disk → the orchestrator's
    # NEXT tick reads RunPod and polls RunPod. Emit a RUNNING-shaped JSON so the
    # loop does NOT post epm:failure for the GCP death.
    return {
        "status": "running",
        "current_phase": "gcp_workload_failover_runpod_async",
        "new_milestone": True,
        "last_log_mtime_sec_ago": 0,
        "pid_alive": True,
        "log_tail_excerpt": (
            f"GCP workload crash on {handle.pod_name}; failed over to RunPod "
            f"{route_result.handle.pod_name} (#659 async failover)"
        ),
        "gate": None,
        "sentinels_processed": 0,
        "phase_log_mtime_sec_ago": 10**9,
        "shard_log_mtime_sec_ago": 10**9,
        "gpu_util": "unknown",
        "next_interval": _DEFAULT_NEXT_INTERVAL_SEC,
        "issue": int(issue),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--issue",
        type=int,
        required=True,
        help="Task / issue number (resolves the default handle sidecar).",
    )
    parser.add_argument(
        "--handle-file",
        type=Path,
        default=None,
        help=(
            "Path to the per-issue handle sidecar JSON "
            "(default: <main-checkout>/.claude/cache/issue-<N>-handle.json, "
            "with a legacy <cwd>/.claude/cache/ fallback probe)."
        ),
    )
    parser.add_argument("--debug", action="store_true", help="Log to stderr at DEBUG level.")
    args = parser.parse_args(argv)

    logging.basicConfig(
        stream=sys.stderr,
        level=logging.DEBUG if args.debug else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    # Lazy imports — keeps the --help path fast.
    from explore_persona_space.backends.issue_dispatch import (
        read_handle_sidecar,
        resolve_handle_sidecar_path,
    )

    # Resolution order: explicit --handle-file > canonical
    # <main-checkout>/.claude/cache/ > legacy <cwd>/.claude/cache/
    # (back-compat with sidecars written by the pre-#612 cwd-relative
    # composer). A resolution CRASH (git missing / not a checkout) is
    # converted to the same terminal infra JSON as a missing sidecar —
    # this script must NEVER exit with empty stdout (the bg-Bash poll
    # loop would spin forever on "stalled"; that is the exact failure
    # mode the missing-sidecar fast path below exists to close).
    try:
        sidecar, probed = resolve_handle_sidecar_path(args.issue, args.handle_file)
    except RuntimeError as exc:
        fallback = Path(".claude/cache") / f"issue-{int(args.issue)}-handle.json"
        logging.warning(
            "backend_poll: sidecar path unresolvable (%s); emitting status=dead infra", exc
        )
        print(
            json.dumps(
                _missing_sidecar_json(args.issue, fallback, f"sidecar path unresolvable: {exc}")
            )
        )
        return 0

    # Missing-sidecar fast path. Previously this raised
    # ``FileNotFoundError`` → empty stdout → bg-Bash poll loop spins
    # forever ("no JSON to parse" reads as "stalled"). Defense in depth
    # behind ``scripts/dispatch_issue.py launch``'s sidecar write: a
    # poll that races the launch, OR a poll after a worktree-reap,
    # still needs a terminal JSON line to break the loop.
    if not Path(sidecar).exists():
        probed_str = ", ".join(str(p) for p in probed)
        logging.warning(
            "backend_poll: sidecar missing (probed: %s); emitting status=dead infra", probed_str
        )
        print(
            json.dumps(
                _missing_sidecar_json(
                    args.issue, Path(sidecar), f"sidecar not found (probed: {probed_str})"
                )
            )
        )
        return 0

    try:
        handle = read_handle_sidecar(sidecar)
    except (json.JSONDecodeError, KeyError, OSError, ValueError) as exc:
        # Same shape as missing-sidecar — a corrupted / malformed
        # sidecar is operationally indistinguishable from "no sidecar"
        # for the orchestrator (it can't poll either way), so emit the
        # SAME terminal infra JSON and let the failure-classifier route.
        logging.warning(
            "backend_poll: sidecar at %s unreadable (%s: %s); emitting status=dead infra",
            sidecar,
            type(exc).__name__,
            exc,
        )
        print(
            json.dumps(
                _missing_sidecar_json(
                    args.issue, Path(sidecar), f"sidecar unreadable: {type(exc).__name__}"
                )
            )
        )
        return 0

    backend = _resolve_backend(handle.backend)
    result = backend.poll(handle)

    # #669 hung-VM wedge escalation: a GCP VM RUNNING-but-hung (eps/phase frozen
    # at a non-terminal value past the staleness floor WITH a transport-class
    # reachability alarm) is rewritten to status=dead / terminal_workload_wedged
    # so the async-failover predicate below matches. A no-op on every other case
    # (non-GCP, not running, phase advancing, within floor, or no reachability
    # alarm) — the staleness clock rides the sidecar's extra dict.
    result = _maybe_escalate_gcp_wedge(handle, result, Path(sidecar), now=time.time())

    # ASYNC GCP-workload-failover (#659): a GCP VM that was already up and
    # crashed its WORKLOAD (not setup) minutes in surfaces here as
    # status=dead / current_phase="terminal_workload_failed". The synchronous
    # route()-time failover (#658) cannot reach this case — the VM is already
    # launched, so there is no live route() call to raise from. Re-dispatch on
    # RunPod exactly once (the SAME terminal rung), authoritatively re-point the
    # handle sidecar at the new RunPod handle, and emit a RUNNING-shaped JSON so
    # the orchestrator's poll loop keeps polling the RunPod run instead of
    # posting epm:failure. A setup/boot failure surfaces
    # current_phase="terminal_setup_failed" (GCP poll discrimination, §4.1.0b)
    # and does NOT match the predicate → it falls through to the ordinary dead
    # path (failure_class: code → blocked).
    if _is_gcp_async_workload_failure(handle, result):
        failover_json = _failover_dead_gcp_to_runpod(
            issue=args.issue, handle=handle, result=result, sidecar=Path(sidecar)
        )
        print(json.dumps(failover_json))
        return 0

    print(json.dumps(_serialize_poll_result(result)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
