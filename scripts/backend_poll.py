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
import re
import subprocess
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

# Lane-infra main-checkout pin (#987): the two functions below are duplicated
# VERBATIM in scripts/dispatch_issue.py by design (importing a shared helper
# before the pin would cache the ambient package and defeat it); the full
# consumer audit comment lives next to dispatch_issue.py's copy, and
# tests/test_lane_infra_main_pin.py pins the two copies source-identical.


def _resolve_main_checkout_root(anchor: Path) -> Path:
    """MAIN repo-checkout root, resolved cwd-independently from ``anchor``.

    Mirrors ``backends/issue_dispatch._main_checkout_root`` (#612) WITHOUT
    importing it — importing the package before the pin would cache the
    ambient (possibly stale worktree) package in ``sys.modules``, defeating
    the pin (#987). Fails LOUD; never a cwd fallback.
    """
    env = {
        k: v
        for k, v in os.environ.items()
        if k not in {"GIT_DIR", "GIT_WORK_TREE", "GIT_INDEX_FILE", "GIT_OBJECT_DIRECTORY"}
    }
    try:
        proc = subprocess.run(
            ["git", "rev-parse", "--path-format=absolute", "--git-common-dir"],
            cwd=str(anchor),
            env=env,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=True,
            timeout=30,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise RuntimeError(
            f"cannot resolve the MAIN checkout root from {anchor} "
            f"(git rev-parse --git-common-dir failed: {exc}); lane infra "
            "must import from main (#987) — refusing the ambient package"
        ) from exc
    common_dir = Path(proc.stdout.strip())
    if common_dir.name != ".git" or not common_dir.is_dir():
        raise RuntimeError(
            f"git common-dir {common_dir!s} does not look like a "
            "main-checkout .git dir; refusing to pin lane infra (#987)"
        )
    root = common_dir.parent
    if not (root / "src" / "explore_persona_space" / "__init__.py").is_file():
        raise RuntimeError(
            f"resolved main root {root!s} has no src/explore_persona_space; "
            "refusing to pin lane infra (#987)"
        )
    return root


def _pin_main_lane_infra(anchor: Path | None = None) -> Path:
    """Insert ``<main>/src`` + ``<main>`` at the FRONT of ``sys.path`` (#987).

    Guarantees the lane infra (``explore_persona_space.backends.*``, incl.
    the GCE startup template in ``gcp.py``, plus lazy ``scripts.*`` imports)
    always resolves from the MAIN checkout — beating a worktree venv's
    editable install — while ``--repo-branch`` keeps cloning the issue
    branch for the remote WORKLOAD (unchanged). Idempotent (re-entrant calls
    remove-then-insert, no duplicates); returns the resolved main root.
    """
    anchor = anchor or Path(__file__).resolve().parent
    main_root = _resolve_main_checkout_root(anchor)
    already = sys.modules.get("explore_persona_space")
    if already is not None:
        mod_file = getattr(already, "__file__", "") or ""
        if not mod_file.startswith(str(main_root / "src") + os.sep):
            raise RuntimeError(
                f"explore_persona_space already imported from {mod_file!r} "
                "before the main-checkout pin — a submodule import would "
                "resolve under the stale package __path__ (#987)"
            )
    for p in (str(main_root), str(main_root / "src")):
        if p in sys.path:
            sys.path.remove(p)
        sys.path.insert(0, p)  # final order: [<main>/src, <main>, ...]
    invoked_root = Path(__file__).resolve().parents[1]
    if invoked_root != main_root:
        sys.stderr.write(
            f"[lane-infra-pin] WARNING: invoked script copy lives under "
            f"{invoked_root} but lane infra is pinned to main {main_root} "
            f"(#987); prefer invoking <main>/scripts/{Path(__file__).name}\n"
        )
    return main_root


if __name__ == "__main__":
    _pin_main_lane_infra()


def _ensure_scripts_dir_on_sys_path() -> None:
    """Insert THIS file's dir (scripts/) so a lazy ``import runpod_api`` resolves.

    In script mode scripts/ is already ``sys.path[0]``; in MODULE mode (tests do
    ``from scripts.backend_poll import main``) only the repo root is on
    sys.path, so a bare lazy ``runpod_api`` import would raise
    ``ModuleNotFoundError`` (#710/#1296). Mirrors the inline bootstrap at the
    issue664_common import sites; idempotent.
    """
    scripts_dir = str(Path(__file__).resolve().parent)
    if scripts_dir not in sys.path:
        sys.path.insert(0, scripts_dir)


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

# ── GCP FLEX_START capacity-queue timeout (#783/#778) ─────────────────────────
# The ``current_phase`` ``GcpBackend.poll`` produces for a FLEX_START instance
# still queued for capacity: ``gcp._gcp_status_to_poll_result`` maps GCE
# ``PENDING`` -> ``status="running"`` / ``current_phase="pending"`` DELIBERATELY
# (#782/#778) so the async bg poll loop keeps polling it. The queue-timeout
# clock below ages against THIS phase.
GCP_PENDING_PHASE = "pending"

# The ``current_phase`` the POLLER synthesizes (#783) when a FLEX_START instance
# has stayed in the capacity queue (``"pending"``) longer than the queue-wait
# floor. ``_is_gcp_queue_timeout`` matches THIS phase EXACTLY; it is set ONLY by
# ``_maybe_escalate_gcp_queue_timeout``. Distinct from the #669 wedge phases so
# the queue-timeout failover is a separate, narrow predicate.
GCP_QUEUE_TIMEOUT_PHASE = "terminal_queue_timeout"

# The current_phase the POLLER synthesizes (#1116) when a GCP instance that was
# last observed in the FLEX_START capacity queue ("pending") resolves to
# instance-not-found: the DWS queue dropped the request server-side (create DONE,
# no delete op — #1112). _is_gcp_queue_vanish matches THIS phase EXACTLY; it is
# set ONLY by _maybe_escalate_gcp_queue_vanish.
GCP_QUEUE_VANISH_PHASE = "terminal_queue_vanish"

# The default bounded queue-wait floor (#783/#778). A GCP instance whose poll
# phase stays ``"pending"`` (FLEX_START capacity queue) longer than this fails
# over to RunPod. 600s mirrors ``router.FREE_WAIT_SECONDS`` — the codebase's
# already-chosen "how long do we park a queued job before advancing the lane"
# constant. Env-overridable at CALL time via ``EPS_GCP_QUEUE_WAIT_SECONDS`` (an
# attempt-floor in seconds, NOT a dollar cap) so ops can tune without a restart,
# mirroring ``router._spot_max_gpu_hours``.
GCP_QUEUE_WAIT_SECONDS_DEFAULT = 600


def _gcp_queue_wait_seconds() -> int:
    """Read the FLEX_START queue-wait floor (#783), defaulting to 600s.

    Read at CALL time (not import time) from ``EPS_GCP_QUEUE_WAIT_SECONDS`` so
    ops can retune the floor without restarting the poller, mirroring
    ``router._spot_max_gpu_hours``. A missing / non-integer / non-positive value
    falls back to :data:`GCP_QUEUE_WAIT_SECONDS_DEFAULT` (600s) — the floor can
    never be zero/negative (which would fail over instantly on the first PENDING
    poll) or crash the poll on a typo.
    """
    raw = os.environ.get("EPS_GCP_QUEUE_WAIT_SECONDS")
    if raw is None:
        return GCP_QUEUE_WAIT_SECONDS_DEFAULT
    try:
        val = int(raw)
    except (TypeError, ValueError):
        return GCP_QUEUE_WAIT_SECONDS_DEFAULT
    return val if val > 0 else GCP_QUEUE_WAIT_SECONDS_DEFAULT


# ── GCP pre-workload boot-loop breaker (#1029) ────────────────────────────────
# The ``current_phase`` ``GcpBackend.poll`` produces for a pre-workload setup
# death — DETERMINISTIC evidence the workload never started (the §4.1.0b
# ``eps/workload_started`` discrimination, produced in the RUNNING window since
# #659 and in the TERMINATED window since #1029). Kept in lock-step with
# ``gcp._terminal_dead_poll(reason="setup_failed")`` -> ``f"terminal_{reason}"``.
GCP_SETUP_FAILED_PHASE = "terminal_setup_failed"

# The post-DELETE observation: the instance record is already gone, so the
# describe 404s (``gcp.poll`` -> ``_terminal_dead_poll(reason="instance not
# found")``). Attribute-blind — a boot death, a finished-and-reaped run, and a
# manual delete all look identical here, hence the launch-age HEURISTIC below.
GCP_INSTANCE_NOT_FOUND_PHASE = "terminal_instance not found"

# The TERMINATED-window coarse phase (spot preemption / max-run-duration /
# manual stop / an attribute-unreadable boot death). Heuristic-eligible ONLY
# when young (see ``_gcp_boot_death_max_age_seconds``) so a lone mid-run spot
# preemption never counts toward the streak.
GCP_TERMINATED_PHASE = "terminal_terminated"

# The ``current_phase`` the POLLER synthesizes (#1029) when the (issue, rung)
# consecutive pre-workload boot-death streak reaches the threshold.
# ``_is_gcp_boot_loop`` matches THIS phase EXACTLY; it is set ONLY by
# ``_maybe_escalate_gcp_boot_loop``.
GCP_BOOT_LOOP_PHASE = "terminal_boot_loop"

# The attribute-blind dead phases the launch-age heuristic classifies as a
# pre-workload boot death when the observation is YOUNG (launch -> observation
# age below the floor). ``terminal_setup_failed`` is NOT here — it is the
# deterministic branch, counted at any age.
_GCP_BOOT_DEATH_HEURISTIC_PHASES = frozenset({GCP_TERMINATED_PHASE, GCP_INSTANCE_NOT_FOUND_PHASE})

# Default launch-age floor for the heuristic branch. Grounding (#763): a
# healthy L4 boot took ~8 min to reach the workload phase, the boot deaths hit
# at ~5.5 min, and the poll default is 540 s (_DEFAULT_NEXT_INTERVAL_SEC) — so
# the launch->OBSERVATION age of a pre-workload death is bounded by
# ~8 min boot + ~9 min poll lag + margin ≈ 25 min; 1500 s sits inside that with
# headroom while excluding mid-run spot preemptions (hours in).
GCP_BOOT_DEATH_MAX_AGE_SECONDS_DEFAULT = 1500


def _gcp_boot_death_max_age_seconds() -> int:
    """Read the #1029 boot-death launch-age floor, defaulting to 1500s.

    Read at CALL time from ``EPS_GCP_BOOT_DEATH_MAX_AGE_SECONDS`` (fail-soft
    parse mirroring :func:`_gcp_queue_wait_seconds`): a missing / non-integer /
    non-positive value falls back to
    :data:`GCP_BOOT_DEATH_MAX_AGE_SECONDS_DEFAULT` — the floor can never be
    zero/negative (which would disable the heuristic branch entirely) or crash
    the poll on a typo.
    """
    raw = os.environ.get("EPS_GCP_BOOT_DEATH_MAX_AGE_SECONDS")
    if raw is None:
        return GCP_BOOT_DEATH_MAX_AGE_SECONDS_DEFAULT
    try:
        val = int(raw)
    except (TypeError, ValueError):
        return GCP_BOOT_DEATH_MAX_AGE_SECONDS_DEFAULT
    return val if val > 0 else GCP_BOOT_DEATH_MAX_AGE_SECONDS_DEFAULT


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
        # Machine-readable stall cause (#664): set on the RunPod lane for a
        # zombie-GPU-allocation hang (``"vllm_worker_dead_zombie_gpu"``),
        # ``None`` otherwise. ``getattr`` defends against a backends-side
        # result that predates the field (the GCP/SLURM lanes do not set
        # it) so the JSON shape stays uniform across lanes without crashing.
        "stall_reason": getattr(result, "stall_reason", None),
        # #983 post-done phase-consistency guard surfaces: True when the
        # tick posted the [post-done-phase-advisory] marker, plus the new
        # [phase=...] lines the guard observed after the recorded done.
        # ``getattr``-defended like ``stall_reason`` so an older /
        # duck-typed result degrades to the defaults, never crashes.
        "post_done_phase_advisory_posted": bool(
            getattr(result, "post_done_phase_advisory_posted", False)
        ),
        "post_done_phase_lines": list(getattr(result, "post_done_phase_lines", ()) or ()),
    }


# ── GCP-lane GPU-idle state sidecar (#730) ────────────────────────────────────
# The GCP-lane GPU-idle advisory + escalation tiers (parity with the #727 RunPod
# lane) reuse the SAME decision/post helpers from ``scripts/poll_pipeline.py``
# (imported lazily in ``main()``'s GCP branch — see there). Those helpers read a
# small per-issue state dict (the idle clock + per-phase de-dup sets). Rather than
# interleave that bookkeeping into the handle sidecar the failover paths
# rewrite (``_failover_dead_gcp_to_runpod`` re-points it), the GPU-idle clock
# rides its OWN sibling file ``issue-<N>-gpu-idle-state.json`` — exactly as the
# RunPod lane keeps its idle clock in its own poll-pipeline state file separate
# from the handle. Single-issue (one file per poll process), so no per-issue
# subdict; same JSON shape ``poll_pipeline._save_state`` persists.


def _gpu_idle_state_path(sidecar: Path) -> Path:
    """Sibling path for the GPU-idle clock; ALWAYS distinct from ``sidecar``.

    Lands in the SAME cache dir (``<main-checkout>/.claude/cache/``) the
    failover paths resolve, but is a DISTINCT file — the GPU-idle clock must
    never be clobbered by a handle rewrite.

    The canonical handle sidecar is named ``issue-<N>-handle.json`` — the
    ``-handle.json`` → ``-gpu-idle-state.json`` substitution gives the natural
    sibling ``issue-<N>-gpu-idle-state.json``. A custom ``--handle-file <path>``
    is honored verbatim and may NOT match that shape (e.g. ``/tmp/custom.json``
    or ``pod-runtime.json``); ``str.replace`` is a no-op when the substring is
    absent, so the naive substitution would return the handle sidecar's OWN
    path. Writing GPU-idle bookkeeping onto the handle would corrupt it (the
    next poll reads it as a ``RunHandle`` → unreadable → false ``status: dead``
    on a live job). Fall back to a stem-based name in that case so the result
    is GUARANTEED distinct from the handle sidecar.
    """
    if sidecar.name.endswith("-handle.json"):
        return sidecar.parent / sidecar.name.replace("-handle.json", "-gpu-idle-state.json")
    # Non-conforming name (custom --handle-file): compose a distinct sibling.
    # Path.stem strips ONE trailing extension, so "custom.json" → "custom".
    return sidecar.parent / f"{sidecar.stem}-gpu-idle-state.json"


def _load_gpu_idle_state(path: Path) -> dict[str, str]:
    """Load the GPU-idle state dict, or ``{}`` on absent / corrupt / non-dict.

    FAIL-SOFT: a missing file (first tick) or a corrupt/torn read restarts the
    idle span (the clock re-anchors to the current tick), never crashes the
    poll — exactly the fail-safe semantics ``poll_pipeline._gpu_idle_advisory_update``
    relies on (an unparsable prev-state simply means "no prior span").
    """
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return {}
    return data if isinstance(data, dict) else {}


def _save_gpu_idle_state(path: Path, payload: dict[str, str]) -> None:
    """Atomically persist the GPU-idle state dict (tmp + replace).

    Mirrors ``poll_pipeline._save_state``'s atomic write so a hypothetical
    overlap with a concurrent writer yields a complete-or-prior file, never a
    torn read (the orchestrator polls one tick at a time per issue, so this is
    belt-and-suspenders).
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True))
    tmp.replace(path)


# #1033: the GPU-idle advisory/escalation keys scoped to ONE instance
# incarnation on the GCP lane. Kept in lockstep with the idle subset of
# ``poll_pipeline._RUN_SCOPED_STATE_KEYS`` (the RunPod-lane sibling clear set).
_IDLE_ADVISORY_STATE_KEYS: tuple[str, ...] = (
    "gpu_idle_since_epoch",
    "gpu_idle_advised_phases",
    "gpu_idle_escalated_phases",
)


def _scope_idle_state_to_attempt(
    prev_state: dict[str, str], attempt_id: str | None
) -> dict[str, str]:
    """Reset the idle-advisory clock when the instance identity changed (#1033).

    ``attempt_id`` is fresh per genuinely NEW instance and label-stable on
    reconnect (#927; the ``gcp.py`` reconnect recovery re-reads it from the
    instance labels), so a mismatch against the stored ``idle_attempt_id``
    means the persisted idle span belongs to a PREVIOUS instance (#763: a
    "543 min" idle advisory on a ~17-min-old VM whose phase name matched the
    stored one, so the per-phase reset never fired). The reported idle
    minutes are PER-INSTANCE, never cumulative across relaunches.

    Fail-safe: an absent/empty CURRENT attempt_id cannot decide instance
    identity, so the state is kept verbatim (pre-#1033 behavior). A
    stored-key MISS with a KNOWN current id also resets — the migration
    path for pre-#1033 state files, failing toward one delayed/duplicate
    advisory, never a stale counter (same cheaper-failure direction as
    ``_tripwire_run_scope``'s malformed-anchor branch). Pure / no I/O;
    never raises into the poll tick.
    """
    if not attempt_id:
        return prev_state
    if prev_state.get("idle_attempt_id", "") == attempt_id:
        return prev_state
    return {k: v for k, v in prev_state.items() if k not in _IDLE_ADVISORY_STATE_KEYS}


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

    A CPU-only GCP handle (``extra["gpu_count"] == 0``, #677) is EXCLUDED from
    the failover ONLY when its intent has NO RunPod CPU lane (#747). The
    relaxation:

    * ``cpu-bigmem`` (gpu_count==0, NOT in
      :data:`router.RUNPOD_CPU_INSTANCE_FOR_INTENT`) is EXCLUDED regardless of
      phase — it has no cheap RunPod equivalent, so a CPU workload crash/wedge
      must NOT fail over. Returning ``False`` routes it to the ordinary dead
      path -> ``failure_class: code`` -> ``status:blocked`` (the watcher's
      capacity-retry pass re-drives only infra/``no_compute_available``, never a
      code failure, so it parks cleanly).
    * ``cpu-small`` / ``cpu-mid`` (gpu_count==0, IN the map) ARE eligible — a
      crashed GCP CPU workload for a mapped intent fails over to a RunPod CPU
      pod (``deployCpuPod``), symmetric with the sync ``_runpod_terminal_rung``
      relaxation; the ``_runspec_from_gcp_handle`` re-dispatch copies ``intent``
      verbatim, so the RunPod relaunch carries ``--intent cpu-small`` which
      Surface 5 (``gpu_heuristics.resolve_cpu_intent``) resolves to the RunPod
      CPU instance_id.

    A pre-#677 GCP handle written before the ``gpu_count`` threading lands has
    no ``gpu_count`` key -> ``extra.get("gpu_count")`` is ``None`` != ``0`` ->
    the CPU guard is a no-op and the handle takes the EXISTING (GPU) failover
    path, exactly as today (fail-toward-existing-behavior on a missing key). A
    CPU handle with ``gpu_count==0`` and NO ``intent`` key (also a pre-#747
    shape) is treated as NOT-mapped -> EXCLUDED (fail-toward the safe #677
    terminal on a missing key).
    """
    extra = getattr(handle, "extra", None) or {}
    if extra.get("gpu_count") == 0:
        # CPU GCP handle: fail over ONLY for a mapped cheap CPU intent (#747).
        # Lazy import keeps the existing backend_poll -> router import direction
        # (router does NOT import backend_poll at module top) and reuses the
        # router's single source of truth for the mapped-intent set.
        from explore_persona_space.backends.router import RUNPOD_CPU_INSTANCE_FOR_INTENT

        if extra.get("intent") not in RUNPOD_CPU_INSTANCE_FOR_INTENT:
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
    return (
        getattr(handle, "backend", None) == "runpod"
        and result.status == "dead"
        and result.current_phase == RUNPOD_WORKLOAD_WEDGED_PHASE
    )


def _is_runpod_cuda_ima_failure(handle, result) -> bool:
    """True ONLY for a RunPod handle whose poll surfaced the #775 CUDA-IMA repeat wedge.

    Narrow by construction (the sibling of :func:`_is_runpod_async_wedge_failure`):
    ``handle.backend == "runpod"`` AND ``result.status == "dead"`` AND
    ``result.current_phase == RUNPOD_CUDA_IMA_WEDGED_PHASE``. A GCP / SLURM handle
    never trips it; the wedged phase is set ONLY by
    :func:`_maybe_escalate_runpod_cuda_ima` once a SECOND same-signature CUDA-IMA
    crash is observed this run.
    """
    return (
        getattr(handle, "backend", None) == "runpod"
        and result.status == "dead"
        and result.current_phase == RUNPOD_CUDA_IMA_WEDGED_PHASE
    )


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


def _maybe_escalate_gcp_queue_timeout(handle, result, sidecar: Path, *, now: float):
    """Escalate a GCP instance stuck in the FLEX_START capacity queue past the floor (#783/#778).

    The queue-timeout sibling of :func:`_maybe_escalate_gcp_wedge`. A FLEX_START
    create can SUCCEED yet leave the instance PENDING (queued for capacity),
    which ``gcp._gcp_status_to_poll_result`` maps to ``status="running"`` /
    ``current_phase="pending"`` DELIBERATELY (#782/#778) so the bg poll loop
    keeps polling — forever, pre-#783 (#778 sat PENDING ~2h45m). This
    poller-side staleness clock (kept OUT of ``GcpBackend.poll`` so it stays a
    pure function of ``(handle, gcloud responses)``, exactly as the #669 wedge)
    ages the ``"pending"`` phase against :func:`_gcp_queue_wait_seconds`. Past
    the floor it rewrites ``status -> "dead"`` /
    ``current_phase -> GCP_QUEUE_TIMEOUT_PHASE`` so :func:`_is_gcp_queue_timeout`
    matches and the queue-timeout failover fires.

    NO reachability-alarm conjunction (unlike the #669 wedge): a stuck queue has
    no live VM to be reachable, so the phase-frozen-past-floor on the ``pending``
    phase IS the entire signal.

    Reuses the SAME sidecar phase clock (:func:`_read_phase_clock` /
    :func:`_write_phase_clock`) as the #669 wedge WITHOUT collision: the two key
    on DISJOINT ``current_phase`` values (``"pending"`` here vs a frozen
    mid-workload phase there), so on any given tick at most one is in scope, and
    the shared clock re-stamps on ANY phase change (a ``pending -> provisioning``
    advance correctly resets it). The false-positive guards return ``result``
    UNCHANGED:

    * not a GCP handle, or the poll is not ``running``, or the phase is not
      ``"pending"`` → unchanged (only a QUEUED GCP instance is in scope; a
      RUNNING / PROVISIONING / STAGING / terminal poll is left alone);
    * phase advanced off ``pending`` OR first observation (``last_ts is None``)
      → re-stamp the clock, return ``running`` (fail-open on a fresh-dispatch
      handle with no clock, and the instant the queue dequeues);
    * phase still ``"pending"`` but WITHIN the floor → return ``running``.
    """
    if getattr(handle, "backend", None) != "gcp" or result.status != "running":
        return result
    if result.current_phase != GCP_PENDING_PHASE:
        return result
    last_phase, last_ts = _read_phase_clock(sidecar)
    if last_phase != result.current_phase or last_ts is None:
        # Phase advanced onto pending (or first observation) → re-stamp the clock.
        _write_phase_clock(sidecar, phase=result.current_phase, ts=now)
        return result
    queued_for = now - last_ts
    floor = _gcp_queue_wait_seconds()
    if queued_for > floor:
        logging.warning(
            "backend_poll: GCP issue stuck in FLEX_START capacity queue (phase %r) for "
            "%.0fs (>%ds floor) — escalating to %s and failing over to RunPod (#783/#778)",
            result.current_phase,
            queued_for,
            floor,
            GCP_QUEUE_TIMEOUT_PHASE,
        )
        return replace(
            result,
            status="dead",
            current_phase=GCP_QUEUE_TIMEOUT_PHASE,
            new_milestone=True,
            pid_alive=False,
        )
    # Still pending but within the floor → stays running (false-positive guard).
    return result


def _is_gcp_queue_timeout(handle, result) -> bool:
    """True ONLY for a GCP handle the queue-timeout escalation marked terminal (#783).

    The narrow sibling of :func:`_is_gcp_async_workload_failure`:
    ``handle.backend == "gcp"`` AND ``result.status == "dead"`` AND
    ``result.current_phase == GCP_QUEUE_TIMEOUT_PHASE`` (a phase set ONLY by
    :func:`_maybe_escalate_gcp_queue_timeout`).

    Reuses that predicate's EXACT CPU-intent guard (#677/#747): a
    ``cpu-bigmem`` queue-stall must NOT fail over — it has no cheap RunPod CPU
    lane — while ``cpu-small`` / ``cpu-mid`` (IN
    :data:`router.RUNPOD_CPU_INSTANCE_FOR_INTENT`) ARE eligible. A CPU handle
    with ``gpu_count == 0`` and no ``intent`` key is treated as NOT-mapped ->
    EXCLUDED (fail-toward the safe #677 terminal on a missing key). A pre-#677
    GPU handle with no ``gpu_count`` key takes the existing GPU failover path
    (``extra.get("gpu_count")`` is ``None`` != ``0`` → the guard is a no-op).
    """
    extra = getattr(handle, "extra", None) or {}
    if extra.get("gpu_count") == 0:
        # Lazy import keeps the backend_poll -> router import direction and reuses
        # the router's single source of truth for the mapped-intent set.
        from explore_persona_space.backends.router import RUNPOD_CPU_INSTANCE_FOR_INTENT

        if extra.get("intent") not in RUNPOD_CPU_INSTANCE_FOR_INTENT:
            return False
    return (
        getattr(handle, "backend", None) == "gcp"
        and result.status == "dead"
        and result.current_phase == GCP_QUEUE_TIMEOUT_PHASE
    )


def _maybe_escalate_gcp_queue_vanish(handle, result, sidecar: Path):
    """Escalate a GCP instance that VANISHED from the FLEX_START queue (#1116/#1112).

    The queue-VANISH sibling of :func:`_maybe_escalate_gcp_queue_timeout`. A
    DWS-queued FLEX_START instance can be dropped SERVER-SIDE (create DONE, no
    delete operation, the instance simply disappears from instances-list —
    #1112 hit this twice in one evening), which ``gcp.poll`` maps to
    ``status="dead"`` / ``current_phase="terminal_instance not found"``. That
    shape is attribute-blind, so pre-#1116 it took the ordinary dead path
    (``failure_class: code`` at best, a #1029 heuristic boot-death record at
    worst — mislabelling a pure CAPACITY event as a boot problem) and
    ``route()`` re-booked the same dead flex rung indefinitely.

    The discriminator is the sidecar phase clock (:func:`_read_phase_clock`):
    a dead not-found poll whose ``last_phase`` reads ``"pending"`` means the
    instance was LAST OBSERVED still queued — it never reached a running
    phase — so the vanish is deterministic capacity evidence, escalated
    INSTANTANEOUSLY (no aging floor, no streak; the clock is READ-ONLY here —
    unlike #783 there is nothing to age and unlike #1029 nothing to count).

    Guards, in order (each returns ``result`` unchanged):

    * ``handle.backend != "gcp"`` or ``result.status != "dead"``;
    * ``result.current_phase != GCP_INSTANCE_NOT_FOUND_PHASE`` — narrow to
      not-found ONLY: a ``terminal_terminated`` instance still EXISTS
      server-side (a preemption / manual stop), which is NOT the vanish shape;
    * the sidecar clock's ``last_phase`` is not ``"pending"`` — covers a
      workload-phase clock (the instance ran, then was deleted: the existing
      #659/#1029 classifications own that) AND a missing/None clock
      (fresh-dispatch handle, wiped sidecar — fail-open to today's behavior,
      which falls through to the #1029 streak path);
    * :func:`_cpu_intent_blocks_runpod_failover` — the #677/#747 guard gates
      the REWRITE itself (mirroring :func:`_maybe_escalate_gcp_boot_loop`), so
      a ``cpu-bigmem`` vanish keeps its ordinary dead path INCLUDING today's
      boot-death record byte-identical.

    On a match the poll is rewritten to :data:`GCP_QUEUE_VANISH_PHASE` (the
    exact #783 rewrite shape) so :func:`_is_gcp_queue_vanish` matches and the
    failover fires. The ``main()`` wiring places this BEFORE
    :func:`_maybe_escalate_gcp_boot_loop`, whose heuristic phase set contains
    not-found — the vanish branch's early return is what keeps a capacity miss
    out of the boot-death streak.
    """
    if getattr(handle, "backend", None) != "gcp" or result.status != "dead":
        return result
    if result.current_phase != GCP_INSTANCE_NOT_FOUND_PHASE:
        return result
    last_phase, _last_ts = _read_phase_clock(sidecar)
    if last_phase != GCP_PENDING_PHASE:
        return result
    if _cpu_intent_blocks_runpod_failover(handle):
        return result
    logging.warning(
        "backend_poll: GCP instance vanished from the FLEX_START capacity queue "
        "(dead poll %r with last observed phase %r) — escalating to %s and "
        "failing over to RunPod (#1116/#1112)",
        result.current_phase,
        last_phase,
        GCP_QUEUE_VANISH_PHASE,
    )
    return replace(
        result,
        status="dead",
        current_phase=GCP_QUEUE_VANISH_PHASE,
        new_milestone=True,
        pid_alive=False,
    )


def _is_gcp_queue_vanish(handle, result) -> bool:
    """True ONLY for a GCP handle the queue-vanish escalation marked terminal (#1116).

    The narrow sibling of :func:`_is_gcp_queue_timeout` / :func:`_is_gcp_boot_loop`:
    ``handle.backend == "gcp"`` AND ``result.status == "dead"`` AND
    ``result.current_phase == GCP_QUEUE_VANISH_PHASE`` (a phase set ONLY by
    :func:`_maybe_escalate_gcp_queue_vanish`, which already applies the
    CPU-intent guard before rewriting). The guard is re-checked here via
    :func:`_cpu_intent_blocks_runpod_failover` as defense-in-depth — a
    ``cpu-bigmem`` handle must never fail over to RunPod even if a future edit
    lets the phase through.
    """
    if _cpu_intent_blocks_runpod_failover(handle):
        return False
    return (
        getattr(handle, "backend", None) == "gcp"
        and result.status == "dead"
        and result.current_phase == GCP_QUEUE_VANISH_PHASE
    )


# ── GCP pre-workload boot-loop breaker: recorder + escalation + reset (#1029) ─


def _cpu_intent_blocks_runpod_failover(handle) -> bool:
    """True iff the handle's CPU intent has NO RunPod lane (#677/#747).

    Encapsulates the exact guard the two EXISTING predicates
    (:func:`_is_gcp_async_workload_failure` / :func:`_is_gcp_queue_timeout`)
    carry inline: ``extra["gpu_count"] == 0`` AND ``extra["intent"]`` NOT in
    :data:`router.RUNPOD_CPU_INSTANCE_FOR_INTENT` (i.e. ``cpu-bigmem``, or a
    pre-#747 CPU handle with no ``intent`` key — fail-toward the safe #677
    terminal on a missing key). A pre-#677 handle with no ``gpu_count`` key
    reads ``None != 0`` -> ``False`` (the existing GPU failover path).

    ACCEPTED DEBT (#1029, deliberate): this leaves the CPU-intent guard in
    THREE places — the two existing inline copies stay byte-untouched (their
    tests + the #783 byte-parity contract stay green without edits) and only
    the NEW #1029 call sites use this helper. A future consolidation is out of
    #1029's scope.
    """
    extra = getattr(handle, "extra", None) or {}
    if extra.get("gpu_count") != 0:
        return False
    # Lazy import keeps the backend_poll -> router import direction and reuses
    # the router's single source of truth for the mapped-intent set.
    from explore_persona_space.backends.router import RUNPOD_CPU_INSTANCE_FOR_INTENT

    return extra.get("intent") not in RUNPOD_CPU_INSTANCE_FOR_INTENT


def _maybe_escalate_gcp_boot_loop(handle, result, *, issue: int, now: float):
    """Record a pre-workload GCP boot death; at N consecutive on one rung,
    rewrite to :data:`GCP_BOOT_LOOP_PHASE` so :func:`_is_gcp_boot_loop` fires
    the failover (#1029).

    Pre-workload death = a GCP dead poll with EITHER

    * (a) DETERMINISTIC: ``current_phase == GCP_SETUP_FAILED_PHASE`` (the
      §4.1.0b ``workload_started`` discrimination — produced in the RUNNING
      window since #659 and in the TERMINATED window since #1029), counted at
      ANY age; OR
    * (b) HEURISTIC (post-DELETE / attribute-blind observations):
      ``current_phase`` in :data:`_GCP_BOOT_DEATH_HEURISTIC_PHASES` AND
      ``now - handle.extra["gcp_launched_ts"] <
      _gcp_boot_death_max_age_seconds()`` (strict ``<``: an observation aged
      exactly AT the floor does NOT count — the spot-preemption
      single-occurrence protection). A missing / non-numeric
      ``gcp_launched_ts`` (pre-#1029 handle) fails OPEN: no record, result
      unchanged.

    Guards, in order (each returns ``result`` unchanged):

    * ``handle.backend != "gcp"`` or ``result.status != "dead"``;
    * phase not in {deterministic} or {heuristic} — the failover-eligible
      phases (``terminal_workload_failed`` / ``_wedged`` / ``_queue_timeout``
      / ``_queue_vanish``) are handled by EARLIER ``main()`` branches and
      never reach this call;
    * heuristic branch without a usable ``gcp_launched_ts``, or age >= floor;
    * a fully-DEGENERATE incarnation key (``job_id`` absent AND both fallback
      components empty — pre-fix handles): SKIP the record entirely (logged,
      fail-open to today's behavior) rather than keying on ``""``.

    Side effect on a match: ``streak = record_gcp_boot_death(issue, rung,
    incarnation=<key>)`` with ``incarnation = str(handle.job_id)`` (the GCE
    instance id — distinct per create, stable across re-polls of one sidecar)
    falling back to ``f"{attempt_id}:{gcp_launched_ts}"``, and
    ``rung = extra.get("gcp_ladder_rung") or "unknown_rung"``. The RECORD
    happens at ANY streak value (it feeds the route()-side rung skip). The
    phase REWRITE additionally requires ``streak >= threshold`` AND the
    CPU-intent guard (:func:`_cpu_intent_blocks_runpod_failover` False — the
    exact #677/#747 shape): a ``cpu-bigmem`` streak RECORDS but never
    rewrites, so its ordinary terminal JSON is untouched and the route()-side
    skip is its breaker. A lease write failure is logged, never raised
    (fail-open: the death takes the ordinary path this tick).
    """
    if getattr(handle, "backend", None) != "gcp" or result.status != "dead":
        return result
    extra = getattr(handle, "extra", None) or {}
    phase = result.current_phase
    if phase == GCP_SETUP_FAILED_PHASE:
        pass  # deterministic pre-workload death — counted at any age
    elif phase in _GCP_BOOT_DEATH_HEURISTIC_PHASES:
        launched_ts = extra.get("gcp_launched_ts")
        if not isinstance(launched_ts, (int, float)):
            # Pre-#1029 handle (no launch ts) -> the heuristic branch is inert.
            return result
        if now - float(launched_ts) >= _gcp_boot_death_max_age_seconds():
            # Old death (mid-run spot preemption / max-run-duration / manual
            # stop) -> NOT a boot death; single-occurrence behavior unchanged.
            return result
    else:
        return result

    # INCARNATION key — identifies one VM CREATE, not one route() call.
    # job_id (the GCE instance id) preferred: distinct per create by
    # construction, stable across re-polls of one sidecar. attempt_id ALONE is
    # FORBIDDEN (#763: all five creates shared one attempt_id with distinct
    # instance ids — attempt_id-keying would freeze the streak at 1).
    incarnation = str(getattr(handle, "job_id", "") or "")
    if not incarnation:
        att = str(extra.get("attempt_id") or "")
        ts_raw = extra.get("gcp_launched_ts")
        ts_part = "" if ts_raw in (None, "") else str(ts_raw)
        if not att and not ts_part:
            logging.warning(
                "backend_poll: GCP boot death on issue %d has a fully-degenerate "
                "incarnation key (no job_id / attempt_id / gcp_launched_ts); "
                "skipping the boot-death record (fail-open, #1029)",
                int(issue),
            )
            return result
        incarnation = f"{att}:{ts_part}"
    rung = str(extra.get("gcp_ladder_rung") or "unknown_rung")

    try:
        # Lazy import (module convention: backend_poll -> router imports stay
        # inside functions; router owns the durable lease).
        from explore_persona_space.backends.router import (
            gcp_boot_death_streak_threshold,
            record_gcp_boot_death,
        )

        streak = record_gcp_boot_death(int(issue), rung, incarnation=incarnation)
        threshold = gcp_boot_death_streak_threshold()
    except Exception as exc:
        logging.warning(
            "backend_poll: boot-death streak record failed for issue %d rung %s "
            "(%s: %s); the death takes the ordinary dead path this tick (fail-open)",
            int(issue),
            rung,
            type(exc).__name__,
            exc,
        )
        return result
    if streak >= threshold and not _cpu_intent_blocks_runpod_failover(handle):
        logging.warning(
            "backend_poll: GCP rung %s hit %d consecutive pre-workload boot deaths "
            "(>= %d) for issue %d — escalating to %s and failing over to RunPod (#1029)",
            rung,
            streak,
            threshold,
            int(issue),
            GCP_BOOT_LOOP_PHASE,
        )
        return replace(
            result,
            status="dead",
            current_phase=GCP_BOOT_LOOP_PHASE,
            new_milestone=True,
            pid_alive=False,
        )
    return result


def _is_gcp_boot_loop(handle, result) -> bool:
    """True ONLY for a GCP handle the boot-loop escalation marked terminal (#1029).

    The narrow sibling of :func:`_is_gcp_queue_timeout`:
    ``handle.backend == "gcp"`` AND ``result.status == "dead"`` AND
    ``result.current_phase == GCP_BOOT_LOOP_PHASE`` (a phase set ONLY by
    :func:`_maybe_escalate_gcp_boot_loop`, which already applies the CPU-intent
    guard before rewriting). The guard is re-checked here via
    :func:`_cpu_intent_blocks_runpod_failover` as defense-in-depth — a
    ``cpu-bigmem`` handle must never fail over to RunPod even if a future edit
    lets the phase through.
    """
    if _cpu_intent_blocks_runpod_failover(handle):
        return False
    return (
        getattr(handle, "backend", None) == "gcp"
        and result.status == "dead"
        and result.current_phase == GCP_BOOT_LOOP_PHASE
    )


def _maybe_reset_gcp_boot_streak(handle, result, *, issue: int) -> None:
    """Reset the (issue, rung) boot-death streak on a POSITIVE workload signal (#1029).

    No-op unless ``handle.backend == "gcp"`` AND a streak record exists for the
    handle's rung (``extra.get("gcp_ladder_rung") or "unknown_rung"`` — the
    SAME defaulting the recorder uses, so a pre-fix handle's ``unknown_rung``
    record resets symmetrically; ``.get``, never bracket access — old handles
    lack the key). Resets ONLY on a POSITIVE workload-started signal — NEVER on
    a "phase not in a pre-workload blocklist" test (fail-closed against
    unknown phase strings):

    * ``status == "running"`` AND ``current_phase`` is a POSITIVE workload
      signal: ``"workload"`` (the startup script's ``_eps_phase workload``
      write) or ``"relaunched_workload"`` (the #612 relaunch-follow probe).
      The mid-boot writes — ``"startup"`` (gcp.py's ``_eps_phase startup``),
      the GCE lifecycle phases (``pending`` / ``provisioning`` / ``staging``),
      and the booting-no-phase ``""`` — are NOT in the set and must never
      reset (a #763-shape relaunch's first poll lands in the boot window and
      reads running/"startup"; a blocklist that omitted it would reset the
      streak and the breaker would never fire); OR
    * ``status == "dead"`` AND ``current_phase == GCP_WORKLOAD_FAILED_PHASE``
      — the workload STARTED (the §4.1.0b sentinel), so boot was fine (the
      #659 failover then proceeds on its own branch); OR
    * ``status == "done"`` — the #935 completion shapes (``workload_done`` /
      ``workload_done_self_poweroff`` / ``relaunched_workload_done`` /
      ``workload_done_finalize_failed`` (#1055) — every
      producer of ``status="done"`` is a success path): a short run completing
      entirely between 540s polls is observed ONLY this way, and a stale
      streak surviving a SUCCESS would fire on a non-consecutive later death.

    Read-before-write (inside :func:`router.reset_gcp_boot_death_streak`): the
    lease is only mutated when a record exists, so the common healthy tick
    costs one lease read. A lease failure is logged, never raised.
    """
    if getattr(handle, "backend", None) != "gcp":
        return
    status = result.status
    phase = result.current_phase
    positive = (
        (status == "running" and phase in ("workload", "relaunched_workload"))
        or (status == "dead" and phase == GCP_WORKLOAD_FAILED_PHASE)
        or status == "done"
    )
    if not positive:
        return
    extra = getattr(handle, "extra", None) or {}
    rung = str(extra.get("gcp_ladder_rung") or "unknown_rung")
    try:
        from explore_persona_space.backends.router import reset_gcp_boot_death_streak

        reset_gcp_boot_death_streak(int(issue), rung)
    except Exception as exc:
        logging.warning(
            "backend_poll: boot-death streak reset failed for issue %d rung %s "
            "(%s: %s); the stale record decays at the UTC day rollover",
            int(issue),
            rung,
            type(exc).__name__,
            exc,
        )


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
    try:
        payload = json.loads(Path(sidecar).read_text())
    except (OSError, json.JSONDecodeError, ValueError):
        return None
    extra = payload.get("extra") if isinstance(payload, dict) else None
    if not isinstance(extra, dict):
        return None
    ts = extra.get("runpod_noport_first_seen_ts")
    # Guard the parse like _read_phase_clock (L290): a non-numeric value reads as
    # None (no clock yet) rather than reaching float() and raising.
    return float(ts) if isinstance(ts, (int, float)) else None


def _write_runpod_noport_clock(sidecar: Path, *, ts: float) -> None:
    """Persist the RunPod no-port clock onto the sidecar ``extra`` dict, atomically.

    Mirrors :func:`_write_phase_clock`: mutates ONLY
    ``extra["runpod_noport_first_seen_ts"]`` (every other field preserved
    verbatim), write-temp + rename, and a write failure is LOGGED not raised.
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
        extra["runpod_noport_first_seen_ts"] = float(ts)
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(json.dumps(payload, sort_keys=True, indent=2))
        tmp.replace(path)
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        logging.warning(
            "backend_poll: RunPod no-port clock write failed for %s (%s: %s); "
            "the next tick re-reads stale state (never manufactures a wedge)",
            sidecar,
            type(exc).__name__,
            exc,
        )


def _clear_runpod_noport_clock(sidecar: Path) -> None:
    """Remove the RunPod no-port clock key from the sidecar (healthy/terminal).

    Called when the live pod exposes a public port or leaves RUNNING — the wedge
    never matured, so the next observation re-stamps from scratch.
    """
    try:
        path = Path(sidecar)
        payload = json.loads(path.read_text())
        if not isinstance(payload, dict):
            return
        extra = payload.get("extra")
        if not isinstance(extra, dict) or "runpod_noport_first_seen_ts" not in extra:
            return
        del extra["runpod_noport_first_seen_ts"]
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(json.dumps(payload, sort_keys=True, indent=2))
        tmp.replace(path)
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        logging.warning(
            "backend_poll: RunPod no-port clock clear failed for %s (%s: %s)",
            sidecar,
            type(exc).__name__,
            exc,
        )


# ── RunPod CUDA-IMA repeat host wedge: signature-record family + regex (#775) ──
# The detection record for a same-signature CUDA-IMA (illegal-memory-access /
# EngineDeadError) crash REPEATING within a run. Byte-mirrors the
# ``_read/_write/_clear_runpod_noport_clock`` family above (atomic tmp+rename,
# never-raise on a malformed value), keyed ``runpod_cuda_ima_last_seen`` (a dict
# ``{"ts": <epoch>, "sig": "cuda_ima"}``, distinct from the noport clock's bare
# float). The record is signature-keyed across the run (NOT pod_id), with the
# prior ``epm:failure`` marker as a cross-pod fallback source (so a sidecar wipe
# between pods does not lose the prior-crash record). See
# ``.claude/rules/compute-backend-failover.md`` Part D.

# The CUDA-IMA crash-signature family. No ``re.DOTALL`` and no cross-newline lazy
# quantifier — each alternative matches WITHIN one line, so a match cannot span
# unrelated events across the 500-line probe tail. A driver-level IMA has no
# stable user-frame, so "same signature" is the FAMILY recurring (this regex
# matching both the prior and the current crash surface), NOT an exact-frame
# fingerprint.
CUDA_IMA_SIGNATURE = re.compile(
    r"CUDA error:\s*an illegal memory access was encountered"
    r"|illegal memory access was encountered"
    r"|EngineDeadError"
    r"|Engine core proc \S+ died unexpectedly",
    re.IGNORECASE,
)

# The synthesized terminal ``current_phase`` for a matured RunPod CUDA-IMA repeat
# wedge (#775). ``_is_runpod_cuda_ima_failure`` matches THIS phase EXACTLY; it is
# set ONLY by ``_maybe_escalate_runpod_cuda_ima`` once a second same-signature
# CUDA-IMA crash is observed this run.
RUNPOD_CUDA_IMA_WEDGED_PHASE = "terminal_runpod_cuda_ima_host_wedged"

# The success ``current_phase`` emitted by ``_relaunch_fresh_runpod`` after a
# fresh-pod re-provision succeeds. The DEFAULT is the no-port wedge phase (Part C,
# #664) — preserved byte-identically so the Part C caller is unchanged. The
# CUDA-IMA caller (Part D, #775) passes ``RUNPOD_CUDA_IMA_FAILOVER_FRESH_POD_PHASE``
# so the emitted poll JSON / markers read TRUE to a CUDA-IMA failover (the two
# wedge classes reuse the same inner relaunch but a shared no-port phase string
# would misclassify a CUDA-IMA pivot as a no-port wedge in the operator log).
RUNPOD_NOPORT_WEDGE_FAILOVER_FRESH_POD_PHASE = "runpod_noport_wedge_failover_fresh_pod"
RUNPOD_CUDA_IMA_FAILOVER_FRESH_POD_PHASE = "runpod_cuda_ima_failover_fresh_pod"


def _crash_signature_is_cuda_ima(text: str | None) -> bool:
    """True iff ``text`` (a crash surface) carries the CUDA-IMA family signature.

    Pure: ``bool(CUDA_IMA_SIGNATURE.search(text or ""))``. ``None``/empty → False.
    """
    return bool(CUDA_IMA_SIGNATURE.search(text or ""))


def _crash_signature_has_our_code_frame(text: str | None) -> bool:
    """True iff the crash surface carries an OUR-code traceback frame (M3 exclusion).

    A CUDA-IMA surface that ALSO traces through ``src/explore_persona_space/`` or
    ``scripts/`` is a deterministic CODE bug surfacing as CUDA-IMA, NOT a host
    wedge — the escalation skips it (falls through to the ordinary dead path →
    ``failure_class: code``) so no bounded pivot is spent on a code bug. Reuses
    ``failure_classifier.OUR_CODE_FRAME`` (lazy-imported to keep the ``--help``
    path fast). ``None``/empty → False.
    """
    if not text:
        return False
    _ensure_scripts_dir_on_sys_path()
    from failure_classifier import OUR_CODE_FRAME

    return bool(OUR_CODE_FRAME.search(text))


def _prior_failure_marker_is_cuda_ima(issue: int) -> bool:
    """Cross-pod FALLBACK source for the prior-CUDA-IMA-crash record (#775, B1).

    When the sidecar ``extra`` record is ABSENT (a sidecar wipe between pods, an
    EDQUOT round-trip, a fresh-host re-point that cleared ``extra``), read the
    latest prior ``epm:failure`` marker for this issue and return True iff its
    note carries a CUDA-IMA signature. ``backend_poll.py`` runs VM-side (the
    orchestrator poller on ``main``, NOT pod-side on an ``issue-<N>`` branch), so
    reading prior markers via ``task_workflow.list_events`` is the same VM-side
    surface the circuit-breaker uses — the pod-side-``task.py``-shellout
    prohibition does not apply.

    Fail-soft: any read / import error → treat as "no prior record" (return
    ``False``) so the fallback can never crash the poll or manufacture a wedge.
    """
    try:
        from explore_persona_space.task_workflow import list_events

        events = list_events(int(issue))
    except Exception as exc:
        logging.warning(
            "backend_poll: CUDA-IMA prior-marker fallback read failed for issue %s (%s: %s); "
            "treating as no prior CUDA-IMA crash",
            issue,
            type(exc).__name__,
            exc,
        )
        return False
    for ev in reversed(events or []):
        if not isinstance(ev, dict):
            continue
        # The event's marker name lives in ``kind`` (e.g. ``"epm:failure"``);
        # the body lives in ``note`` (verified against the live events.jsonl
        # shape for this issue).
        kind = str(ev.get("kind") or "")
        if "epm:failure" not in kind:
            continue
        # The LATEST epm:failure is the relevant one (the most recent prior
        # crash): if its note carries the CUDA-IMA family, the current crash is a
        # repeat; if not, the current crash is the first of its kind this run.
        # Either way we decide on the first epm:failure seen (newest-first), so
        # return its verdict directly.
        return _crash_signature_is_cuda_ima(str(ev.get("note") or ""))
    return False


def _read_runpod_cuda_ima_record(sidecar: Path, *, issue: int) -> dict | None:
    """Read the prior-CUDA-IMA-crash record for this run, or ``None`` if absent (#775).

    Mirrors :func:`_read_runpod_noport_clock`'s fail-soft contract: a missing /
    unreadable / malformed sidecar OR a non-dict ``extra["runpod_cuda_ima_last_seen"]``
    reads as ``None``, NEVER raises. When the sidecar record is ABSENT, FALLS BACK
    to the prior ``epm:failure`` marker (B1 cross-pod source — see
    :func:`_prior_failure_marker_is_cuda_ima`): a CUDA-IMA prior marker yields a
    synthetic record so the current crash still counts as a repeat across a pod
    swap. The sidecar is the FAST path; the marker is the durable cross-pod
    backstop (the lease/sentinel two-record philosophy applied to detection).
    """
    record: dict | None = None
    try:
        payload = json.loads(Path(sidecar).read_text())
        extra = payload.get("extra") if isinstance(payload, dict) else None
        if isinstance(extra, dict):
            raw = extra.get("runpod_cuda_ima_last_seen")
            if isinstance(raw, dict):
                record = raw
    except (OSError, json.JSONDecodeError, ValueError):
        record = None
    if record is not None:
        return record
    # Sidecar record absent → cross-pod fallback to the prior epm:failure marker.
    if _prior_failure_marker_is_cuda_ima(issue):
        return {"sig": "cuda_ima", "source": "prior_failure_marker"}
    return None


def _write_runpod_cuda_ima_record(sidecar: Path, *, ts: float) -> None:
    """Persist the CUDA-IMA crash record onto the sidecar ``extra`` dict, atomically.

    Mirrors :func:`_write_runpod_noport_clock`: mutates ONLY
    ``extra["runpod_cuda_ima_last_seen"]`` (every other field preserved verbatim),
    write-temp + rename, and a write failure is LOGGED not raised.
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
        extra["runpod_cuda_ima_last_seen"] = {"ts": float(ts), "sig": "cuda_ima"}
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(json.dumps(payload, sort_keys=True, indent=2))
        tmp.replace(path)
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        logging.warning(
            "backend_poll: RunPod CUDA-IMA record write failed for %s (%s: %s); "
            "the next tick re-reads stale state (never manufactures a wedge)",
            sidecar,
            type(exc).__name__,
            exc,
        )


def _clear_runpod_cuda_ima_record(sidecar: Path) -> None:
    """Remove the CUDA-IMA crash record key from the sidecar (#775).

    Called on a non-dead / non-CUDA-IMA poll (an intervening healthy poll the
    in-place same-pod respawn recovered to) — a single transient CUDA-IMA does
    NOT accumulate against a later unrelated one; only a SECOND CUDA-IMA crash
    with no intervening healthy poll counts. Mirrors
    :func:`_clear_runpod_noport_clock`; missing key → no-op.
    """
    try:
        path = Path(sidecar)
        payload = json.loads(path.read_text())
        if not isinstance(payload, dict):
            return
        extra = payload.get("extra")
        if not isinstance(extra, dict) or "runpod_cuda_ima_last_seen" not in extra:
            return
        del extra["runpod_cuda_ima_last_seen"]
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(json.dumps(payload, sort_keys=True, indent=2))
        tmp.replace(path)
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        logging.warning(
            "backend_poll: RunPod CUDA-IMA record clear failed for %s (%s: %s)",
            sidecar,
            type(exc).__name__,
            exc,
        )


def _pod_is_runpod_runtime_wedged(info) -> bool:
    """True iff a live RunPod ``PodInfo`` is in the RAW #664 no-port wedge
    condition: ``desired_status == "RUNNING"`` AND no public SSH port
    (``runtime.ports`` empty -> ``runpod_api._parse_pod`` sets
    ``ssh_host``/``ssh_port`` to ``None``).

    This is the MATURITY-AGNOSTIC raw condition ONLY — the K-floor age check
    stays with each caller (the poller uses its sidecar clock; the watcher
    backstop in ``autonomous_session_watch.py`` uses its OWN dedicated
    ``wedge_first_seen`` clock). ``info is None`` (pod gone) -> ``False`` (not a
    wedge; the gone pod is the ordinary dead path). This is the SINGLE source of
    truth for the wedge condition, called by BOTH
    :func:`_maybe_escalate_runpod_wedge` (poller) and the
    ``autonomous_session_watch.py`` pod-safety wedge arm (watcher backstop,
    #692) — neither re-defines it (#692 composition surface (b))."""
    if info is None:
        return False
    if getattr(info, "desired_status", None) != "RUNNING":
        return False
    return not (info.ssh_host and info.ssh_port)


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
    if getattr(handle, "backend", None) != "runpod":
        return result
    _ensure_scripts_dir_on_sys_path()
    from runpod_api import get_pod_by_name  # live RunPod API (X-Team-Id baked in)

    info = get_pod_by_name(handle.pod_name)  # PodInfo or None
    # gone / EXITED / port-present -> not the raw wedge condition -> clear the
    # no-port clock, return unchanged (the negation of the extracted predicate
    # is exactly the old inline `healthy_or_terminal` guard, behavior-preserving).
    if not _pod_is_runpod_runtime_wedged(info):
        _clear_runpod_noport_clock(sidecar)
        return result
    # RUNNING + no public port: start/continue the no-port clock.
    first_seen = _read_runpod_noport_clock(sidecar)  # never raises (S2 contract)
    if first_seen is None:
        _write_runpod_noport_clock(sidecar, ts=now)
        # A1-residue: first observation -> override an SSH-dead poll to RUNNING so
        # polling continues until the wedge matures past K.
        return replace(
            result,
            status="running",
            current_phase=RUNPOD_WORKLOAD_OBSERVED_PHASE,
            pid_alive=True,
            new_milestone=True,
        )
    wedged_for = now - first_seen
    if wedged_for > RUNPOD_WEDGE_K_SEC:
        logging.warning(
            "backend_poll: RunPod %s RUNNING-but-no-port for %.0fs (>%ds K floor) "
            "— host-pinned resume cannot heal; escalating to %s (#664 wedge)",
            handle.pod_name,
            wedged_for,
            RUNPOD_WEDGE_K_SEC,
            RUNPOD_WORKLOAD_WEDGED_PHASE,
        )
        return replace(
            result,
            status="dead",
            current_phase=RUNPOD_WORKLOAD_WEDGED_PHASE,
            new_milestone=True,
            pid_alive=False,
        )
    # A1-residue: within the K floor (subsequent tick) -> still running, keep polling.
    return replace(
        result,
        status="running",
        current_phase=RUNPOD_WORKLOAD_OBSERVED_PHASE,
        pid_alive=True,
    )


def _maybe_escalate_runpod_cuda_ima(handle, result, sidecar: Path, *, issue: int, now: float):
    """Escalate a same-signature RunPod CUDA-IMA crash REPEAT to terminal wedged (#775).

    The repeat-based sibling of :func:`_maybe_escalate_runpod_wedge` (which is
    time-based). Reads ``result.crash_signature`` (the WIDE 500-line probe tail
    threaded through ``RunPodBackend.poll``). The detection record rides the
    sidecar ``extra`` dict (keyed ``runpod_cuda_ima_last_seen``), with the prior
    ``epm:failure`` marker as a cross-pod fallback (B1). Branches:

    * not a RunPod handle -> return ``result`` unchanged (no record touched);
    * not (``status="dead"`` AND a CUDA-IMA signature on the WIDE surface) ->
      CLEAR the record (an intervening healthy / non-CUDA-IMA poll the in-place
      same-pod respawn recovered to does NOT accumulate), return ``result``
      unchanged;
    * a CUDA-IMA signature whose WIDE surface ALSO carries an OUR-code traceback
      frame (M3) -> this is a deterministic CODE bug surfacing as CUDA-IMA, NOT a
      host wedge. Do NOT escalate (no bounded pivot spent); return ``result``
      unchanged so it falls through to the ordinary dead path
      (failure_classifier -> code). The record is left as-is (a code bug is not a
      host wedge to count);
    * a CUDA-IMA signature with NO prior same-signature record this run (FIRST
      crash) -> WRITE the record, return ``result`` unchanged so it falls through
      to the ordinary dead path (failure_classifier -> infra -> the in-place
      same-pod experimenter respawn, which orphan-reaps + relaunches);
    * a CUDA-IMA signature WITH a prior same-signature record (SECOND repeat) ->
      REWRITE to ``status="dead", current_phase=RUNPOD_CUDA_IMA_WEDGED_PHASE`` so
      :func:`_is_runpod_cuda_ima_failure` matches and the failover fires.

    Reached on EVERY poll tick (wired unconditionally in ``main()`` BEFORE the
    no-port block — M2), so the clear-on-recovery branch always runs.
    """
    if getattr(handle, "backend", None) != "runpod":
        return result
    is_cuda_ima = result.status == "dead" and _crash_signature_is_cuda_ima(
        getattr(result, "crash_signature", None)
    )
    if not is_cuda_ima:
        # Healthy / non-CUDA-IMA dead poll -> the wedge never matured; clear the
        # record so a single transient CUDA-IMA the respawn recovered from does
        # not accumulate against a later unrelated one.
        _clear_runpod_cuda_ima_record(sidecar)
        return result
    # M3 our-code-frame exclusion: a CUDA-IMA surface that ALSO traces through our
    # source/scripts is a deterministic code bug, NOT a host wedge — fall through
    # to the ordinary dead path (-> code) WITHOUT spending a bounded pivot. Leave
    # the record untouched (a code bug is not a host-wedge crash to count).
    if _crash_signature_has_our_code_frame(getattr(result, "crash_signature", None)):
        logging.warning(
            "backend_poll: RunPod %s CUDA-IMA crash carries an OUR-code traceback frame — "
            "deterministic code bug, NOT a host wedge; NOT escalating (#775 M3 exclusion)",
            getattr(handle, "pod_name", "?"),
        )
        return result
    prior = _read_runpod_cuda_ima_record(sidecar, issue=issue)
    if prior is None:
        # FIRST CUDA-IMA crash this run: record it and let the ordinary dead path
        # run (the in-place same-pod experimenter respawn gets its one chance).
        _write_runpod_cuda_ima_record(sidecar, ts=now)
        return result
    # SECOND same-signature CUDA-IMA crash this run -> escalate. A clean orphan-reap
    # already happened on the in-place respawn, so a repeat = the GPU still throws
    # IMA = driver wedge; only a fresh host helps.
    logging.warning(
        "backend_poll: RunPod %s CUDA-IMA crash REPEATED (same signature) this run — the "
        "in-place same-pod respawn did not heal it; escalating to %s (#775 wedge)",
        getattr(handle, "pod_name", "?"),
        RUNPOD_CUDA_IMA_WEDGED_PHASE,
    )
    return replace(
        result,
        status="dead",
        current_phase=RUNPOD_CUDA_IMA_WEDGED_PHASE,
        new_milestone=True,
        pid_alive=False,
    )


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
    if int(issue) == 664:
        # Imported behind the issue==664 guard so a non-#664 poll never triggers
        # the import. realized_grid() is pure (reads the static grid + battery),
        # no pod-local state.
        import sys
        from pathlib import Path as _Path

        scripts_dir = str(_Path(__file__).resolve().parent)
        if scripts_dir not in sys.path:
            sys.path.insert(0, scripts_dir)
        import issue664_common as C

        return list(C.realized_grid())
    return []


def _list_issue664_hub_files(repo_id: str, prefixes: tuple[str, ...]) -> set[str]:
    """Union of server-side SCOPED listings for the wedge gate (#920/#988).

    Replaces the bare full-repo ``list_repo_files`` on the ~1M-file data repo
    (which wedges >600 s, #920) with one scoped tree walk per root prefix.
    An absent prefix contributes zero files (EntryNotFoundError is mapped to
    [] inside list_hf_files_under_path) — identical to a full listing having
    no files under it. Transport/auth/RepositoryNotFound errors PROPAGATE
    (the gate must fail loud, never fail open, before an irreversible
    terminate).

    NOTE the listing is now TOKEN-BEARING (``HfApi(token=HF_TOKEN)``) where
    the old module-level ``huggingface_hub.list_repo_files`` call was
    anonymous — on a token problem the failure direction is loud/safe (an
    auth error propagates and blocks the terminate), never a silent
    fail-open.
    """
    import os

    from huggingface_hub import HfApi

    from explore_persona_space.orchestrate.hub import list_hf_files_under_path

    api = HfApi(token=os.environ.get("HF_TOKEN"))
    files: set[str] = set()
    for prefix in prefixes:
        files.update(
            list_hf_files_under_path(api, repo_id, prefix, repo_type="dataset", revision="main")
        )
    return files


def _wedged_run_inputs_on_hf(issue: int, handle) -> _WedgeInputsGate:
    """Per-cell three-state inputs-on-HF gate for the irreversible auto-terminate.

    Classifies each selected cell ``complete | partial | absent`` from a UNION
    of three server-side SCOPED listings (EXACT expected file set per S1, not
    prefix-presence) — the three root prefixes are the ONLY prefixes
    ``issue664_dispatch._classify_cell_hub_state`` matches files against, so
    the union is a superset of every file the classifier can see (#988).
    Terminate is allowed iff there are ZERO partial cells — a COMPLETE cell's
    data is preserved, a not-yet-run ABSENT cell is rerunnable, and only a
    half-uploaded PARTIAL cell would lose recoverable work.
    """
    cells = _issue_cells_for_handle(issue, handle)  # for #664: realized_grid()
    if not cells:
        # No per-cell artifacts (adapters-only path): the adapters were
        # inline-verified by train_lora, so there is nothing for THIS gate to
        # block on -> safe to terminate.
        return _WedgeInputsGate(ok=True, complete=[], partial=[], absent=[])
    import sys
    from pathlib import Path as _Path

    scripts_dir = str(_Path(__file__).resolve().parent)
    if scripts_dir not in sys.path:
        sys.path.insert(0, scripts_dir)
    import issue664_common as C
    import issue664_dispatch as D

    files = _list_issue664_hub_files(
        C.HF_DATA_REPO,
        (C.HF_RAW_COMPLETIONS_PREFIX, C.HF_STORE_PREFIX, C.HF_MARKER_SLOT_PREFIX),
    )
    complete: list[str] = []
    partial: list[str] = []
    absent: list[str] = []
    for c in cells:
        state = D._classify_cell_hub_state(c, files)  # 'complete'|'partial'|'absent' (S1)
        (complete if state == "complete" else partial if state == "partial" else absent).append(
            c.eval_key
        )
    # Terminate is allowed iff ZERO partial cells: a COMPLETE cell's data is
    # preserved, an ABSENT (not-yet-run) cell reruns on the fresh pod, and only a
    # half-uploaded PARTIAL cell would lose recoverable work.
    return _WedgeInputsGate(
        ok=(len(partial) == 0), complete=complete, partial=partial, absent=absent
    )


def _runpod_handle_identity(handle) -> dict:
    """The (pod_name, job_id) identity of a RunPod handle, for sentinel matching.

    Mirrors :func:`_gcp_handle_identity`: the sentinel records the identity of the
    wedged RunPod run that ALREADY launched a fresh-pod failover, so a later,
    genuinely-NEW run (a fresh-pod re-provision writes a NEW pod_name to the
    sidecar) is NOT suppressed by a stale sentinel.
    """
    return {
        "pod_name": getattr(handle, "pod_name", None),
        "job_id": getattr(handle, "job_id", None),
    }


def _runpod_wedge_sentinel_path(sidecar: Path) -> Path:
    """The idempotency sentinel path for a RunPod no-port wedge failover (#664/#689).

    A sibling file in the same ``.claude/cache/`` dir as the handle sidecar (so it
    is cwd-INDEPENDENT for free), DISTINCT from the GCP failover sentinel
    (:func:`_failover_sentinel_path` -> ``-failover-persistence-failed.json``):
    ``issue-<N>-handle.json`` -> ``issue-<N>-runpod-wedge-handled.json``.
    """
    name = sidecar.name
    stem = name[: -len("-handle.json")] if name.endswith("-handle.json") else sidecar.stem
    return sidecar.with_name(f"{stem}-runpod-wedge-handled.json")


def _write_runpod_wedge_sentinel(sentinel: Path, *, issue: int, handle) -> None:
    """Persist the RunPod-wedge idempotency sentinel atomically, best-effort.

    Records the wedged RunPod run identity (pod_name/job_id) so a subsequent poll
    on the SAME wedged handle short-circuits (no second terminate/re-provision). A
    write failure is LOGGED, not raised — the worst case is one extra
    terminate-and-reprovision on the next tick, never a silent suppression.
    """
    try:
        sentinel.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "issue": int(issue),
            "reason": "runpod_noport_wedge_failover",
            "runpod_wedge": _runpod_handle_identity(handle),
        }
        tmp = sentinel.with_suffix(sentinel.suffix + ".tmp")
        tmp.write_text(json.dumps(payload, sort_keys=True, indent=2))
        tmp.replace(sentinel)
    except OSError as exc:
        logging.warning(
            "backend_poll: failed to write RunPod-wedge idempotency sentinel %s (%s: %s)",
            sentinel,
            type(exc).__name__,
            exc,
        )


def _runspec_from_runpod_handle(handle, issue):
    """Reconstruct the ``RunSpec`` for the fresh-pod re-provision from the handle.

    Mirrors :func:`_runspec_from_gcp_handle`. A router-launched RunPod handle's
    ``extra`` carries ``intent`` plus (for runs dispatched through the unified
    router's canonical handle) ``workload_cmd`` / ``hydra_args`` / ``gpus`` /
    ``time_budget_hours`` (+ ``repo_branch`` post-#909, threaded through so the
    failover re-execution syncs the ISSUE branch, not ``main``; + the footprint
    fields ``boot_disk_gb`` / ``min_ram_gb`` post-#1118, so the fresh pod keeps
    the plan's disk/RAM requirement instead of silently downsizing to the
    200 GB default volume — the #1112 ENOSPC class, one failover hop later; a
    legacy handle without the keys still reconstructs). FAILS LOUD (raises
    ``ValueError``) on a handle that carries NEITHER a workload command NOR
    hydra args — it NEVER silently re-provisions a blank pod.
    """
    from explore_persona_space.backends.base import RunSpec

    extra = handle.extra or {}
    workload_cmd = extra.get("workload_cmd", "") or ""
    hydra_args = tuple(extra.get("hydra_args") or ())
    if not workload_cmd and not hydra_args:
        raise ValueError(
            f"RunPod handle for issue {issue} lacks workload_cmd/hydra_args in extra; "
            f"cannot reconstruct a RunSpec for the fresh-pod re-provision. Refusing to "
            f"re-provision a blank pod (extra keys present: {sorted(extra)})."
        )
    # #1118: forward the footprint fields (mirroring _runspec_from_gcp_handle's
    # #1010 forwarding) so the wedge / CUDA-IMA fresh-pod re-provision keeps
    # the plan's disk requirement — RunPodBackend.launch threads boot_disk_gb
    # into --volume-gb (GPU) / --container-disk-gb (CPU). Keys forwarded only
    # when present/truthy, so a legacy handle reconstructs byte-identically.
    rebuilt_extra: dict = {}
    if extra.get("repo_branch"):
        rebuilt_extra["repo_branch"] = extra["repo_branch"]
    for key in ("boot_disk_gb", "min_ram_gb"):
        if extra.get(key):
            rebuilt_extra[key] = extra[key]
    return RunSpec(
        issue=int(issue),
        intent=extra.get("intent", "lora-7b"),
        backend="runpod",
        gpus=extra.get("gpus"),
        time_budget_hours=extra.get("time_budget_hours"),
        workload_cmd=workload_cmd,
        hydra_args=hydra_args,
        extra=rebuilt_extra,
    )


def _runpod_wedge_already_handled(issue: int, handle, sidecar: Path) -> bool:
    """Idempotency short-circuit for a re-fired RunPod wedge failover (#664/#689).

    Mirrors the GCP failover's sentinel + durable-lease guard (keyed to the wedged
    pod identity) so a second tick on the OLD handle never double-terminates /
    double-provisions. TWO records guard the relaunch, in the order checked:

      1. DURABLE lease (AUTHORITATIVE, #689 blocker-2). The lease at
         ``~/.eps-routing/`` survives the EDQUOT / read-only-fs / out-of-inodes
         mode that fails BOTH the ``.claude/cache`` sidecar write AND the
         same-dir sentinel write together — the SAFETY NET that makes "exactly
         once per wedge" hold under a persistent ``.claude/cache`` disk failure.
      2. SENTINEL (OPTIMIZATION). The ``.claude/cache`` sentinel is the fast path
         — a sibling-file read that avoids the lease-store flock round-trip on
         the common case (no disk failure). A corrupted/unreadable sentinel reads
         as ABSENT (``_read_failover_sentinel`` returns ``None``) — worst case one
         extra terminate-and-reprovision, never a silent suppression.

    Both are keyed to the wedged pod's pod_name/job_id, so a genuinely-new run (a
    fresh-pod re-provision writes a NEW pod_name to the sidecar) does NOT match a
    stale record and still gets its own one failover.
    """
    # 1. DURABLE LEASE (authoritative; survives a .claude/cache-wide disk failure).
    if _lease_records_runpod_wedge_failover(issue, handle):
        return True
    # 2. SENTINEL (fast path).
    sentinel = _runpod_wedge_sentinel_path(sidecar)
    prior = _read_failover_sentinel(sentinel)
    return prior is not None and prior.get("runpod_wedge") == _runpod_handle_identity(handle)


def _relaunch_fresh_runpod(
    *,
    issue: int,
    handle,
    result,
    sidecar: Path,
    stamp_fn=None,
    success_phase: str = RUNPOD_NOPORT_WEDGE_FAILOVER_FRESH_POD_PHASE,
) -> dict:
    """Re-provision a FRESH RunPod pod + resume the dispatcher idempotently.

    Re-uses the router's ``failover_to_runpod_after_async_workload_crash`` (the
    same function ``_failover_dead_gcp_to_runpod`` calls): reconstruct a RunSpec
    from the handle, launch a fresh RunPod run (NEW host, NOT a host-pinned
    resume), re-point the handle sidecar, durable-lease guarded. The fresh pod's
    P2 dispatcher skips HF-complete cells via A2's ``_cell_done_anywhere``.

    ``stamp_fn`` selects WHICH durable lease field stamps the "exactly once per
    wedge" record on the three internal stamp sites. It DEFAULTS (via the ``None``
    sentinel resolved below — ``_stamp_runpod_wedge_failover`` is DEFINED LATER in
    this module, so it cannot be a literal default-arg value without a
    forward-reference ``NameError`` at import) to
    :func:`_stamp_runpod_wedge_failover` (the Part C no-port wedge field), so the
    existing no-port caller (:func:`_failover_wedged_runpod`) is byte-unchanged.
    The #775 CUDA-IMA caller (:func:`_failover_cuda_ima_runpod`) passes
    ``stamp_fn=_stamp_runpod_cuda_ima_failover`` so the SEPARATE
    ``runpod_cuda_ima_failover_of`` lease field is stamped — the two failover
    classes never cross-suppress.

    ``success_phase`` is the ``current_phase`` emitted on a successful fresh-pod
    relaunch. It DEFAULTS to ``RUNPOD_NOPORT_WEDGE_FAILOVER_FRESH_POD_PHASE`` (Part
    C byte-unchanged); the #775 CUDA-IMA caller passes
    ``RUNPOD_CUDA_IMA_FAILOVER_FRESH_POD_PHASE`` so the emitted poll JSON / markers
    read TRUE to a CUDA-IMA failover instead of mislabelling it a no-port wedge.
    """
    if stamp_fn is None:
        stamp_fn = _stamp_runpod_wedge_failover
    from explore_persona_space.backends.issue_dispatch import (
        read_handle_sidecar,
        write_handle_sidecar,
    )
    from explore_persona_space.backends.router import (
        NoComputeAvailableError,
        failover_to_runpod_after_async_workload_crash,
    )
    from explore_persona_space.backends.runpod import RunPodBackend, RunPodWorkloadStartError
    from explore_persona_space.backends.slurm import post_marker_via_task_py

    # #689 (round-3 blocker): a LEGACY sidecar built before RunPodBackend.launch()
    # began persisting workload_cmd/hydra_args (the production handle once carried
    # neither) makes _runspec_from_runpod_handle raise ValueError. The caller
    # (_failover_wedged_runpod) has ALREADY terminated the wedged pod by the time we
    # get here, so letting the ValueError propagate would hit the call-site
    # `except Exception` and surface reason="runpod_wedge_failover_error" — which is
    # NOT in TRANSIENT_CAPACITY_REASONS, so the run parks at blocked WITHOUT an
    # actionable reason. Map it instead to an OBSERVABLE terminal infra JSON
    # (reason="runpod_wedge_relaunch_spec_missing") so the failure path is named in
    # the marker trail. This preserves the fail-loud contract (no blank-pod
    # re-provision) while keeping the poller's terminal-JSON contract: a fresh launch
    # (post-fix handle) carries the spec fields and never reaches this branch.
    try:
        spec = _runspec_from_runpod_handle(handle, issue)
    except ValueError as exc:
        return _terminal_infra_json(
            issue=issue,
            sidecar=sidecar,
            reason="runpod_wedge_relaunch_spec_missing",
            log_tail=(
                f"RunPod {handle.pod_name} no-port wedge: terminated the wedged pod but the "
                f"sidecar handle lacks the relaunch-critical RunSpec fields "
                f"(workload_cmd/hydra_args) needed to re-provision a fresh pod — "
                f"cannot reconstruct a RunSpec ({exc}). A relaunch from a legacy "
                f"pre-#689 handle requires manual re-dispatch (CLAUDE.md halt-criterion #2)."
            ),
        )
    # #689 blocker-3: wrap the router launch in try/except NoComputeAvailableError
    # (mirrors the GCP analogue at _failover_dead_gcp_to_runpod). The wedged pod was
    # already terminated by the caller (billing stopped), so a no-capacity RunPod
    # MUST surface as a TERMINAL infra JSON with reason=no_compute_available — the
    # watcher's capacity-retry pass re-drives ONLY that reason once a lane frees.
    # Letting NoComputeAvailableError propagate uncaught would exit main() with a
    # traceback and NO terminal JSON, stranding the run with no re-drive signal.
    try:
        route_result = failover_to_runpod_after_async_workload_crash(
            spec=spec,
            runpod_backend=RunPodBackend(),
            evidence={
                "source": "runpod_noport_wedge",
                "current_phase": result.current_phase,
                "log_tail_excerpt": result.log_tail_excerpt,
                "wedged_pod_name": handle.pod_name,
            },
            residual_gap=(
                "RunPod RUNNING-but-no-port host wedge (#664); host-pinned resume cannot "
                "heal — re-provisioning a FRESH pod"
            ),
            marker_poster=post_marker_via_task_py,
            on_launched=lambda h: write_handle_sidecar(h, sidecar),
        )
    except RunPodWorkloadStartError as exc:
        # PARTIAL failure (#954): the fresh re-provision SUCCEEDED (a pod
        # bills, left RUNNING for diagnosis per the #909 contract) but the
        # workload-start leg failed. NOT no_compute_available — that mislabel
        # reads "nothing launched" (false) and invites the watcher's
        # capacity-retry re-drive while the fresh pod bills invisibly.
        partial = getattr(exc, "handle", None)
        if partial is None:
            # Defensive: unreachable via the rung today (a handle-less start
            # error takes the rung's NoComputeAvailableError branch), kept for
            # a future direct-raise path. Nothing provisioned.
            return _terminal_infra_json(
                issue=issue,
                sidecar=sidecar,
                reason="runpod_workload_start_failed",
                log_tail=(
                    f"RunPod {handle.pod_name} wedge failover: fresh re-provision's "
                    f"workload start failed with NO pod provisioned ({str(exc)[:500]})"
                ),
            )
        # Fresh pod provisioned + RUNNING, workload not started. Stamp the
        # wedge/CUDA-IMA lease (bounds a re-fired tick — mirrors the
        # sidecar-failure branch) and re-point the sidecar at the fresh pod
        # so it is visible to the handle machinery.
        stamp_fn(issue, handle)
        sidecar_note = ""
        try:
            write_handle_sidecar(partial, sidecar)
        except OSError as write_exc:
            # Never mask the typed failure — record the sidecar-write failure
            # in the terminal note (the lease stamp above already bounds a
            # re-fired tick, so no second terminate/re-provision).
            sidecar_note = f"; sidecar write ALSO failed ({type(write_exc).__name__}: {write_exc})"
        return _terminal_infra_json(
            issue=issue,
            sidecar=sidecar,
            reason="runpod_workload_start_failed",
            log_tail=(
                f"RunPod {handle.pod_name} wedge failover: fresh pod "
                f"{partial.pod_name} PROVISIONED but workload start FAILED "
                f"({str(exc)[:500]}); pod left RUNNING for diagnosis — check for a "
                f"live workload (pidfile) before re-driving; pod BILLS until a human "
                f"stops/terminates it{sidecar_note} (lease stamped — relaunch bounded)"
            ),
        )
    except NoComputeAvailableError:
        # RunPod truly unavailable after the wedged pod was terminated: terminal
        # infra JSON with reason=no_compute_available (re-drivable by the watcher's
        # capacity-retry pass). No lease stamp — nothing launched, so a later poll
        # SHOULD retry once a lane frees.
        return _terminal_infra_json(
            issue=issue,
            sidecar=sidecar,
            reason="no_compute_available",
            log_tail=(
                f"RunPod {handle.pod_name} no-port wedge: terminated the wedged pod but "
                f"RunPod is also unavailable for the fresh re-provision (#664 wedge failover)"
            ),
        )
    # AUTHORITATIVE post-route sidecar write + readback (mirrors the GCP path): the
    # on_launched hook is best-effort (the router swallows its exceptions), so
    # re-point the sidecar HERE and PROVE it landed as a fresh RunPod handle before
    # emitting running. #689 blocker-3: guard the sidecar write/readback with
    # try/except OSError — an EDQUOT / read-only-fs failure here MUST surface as a
    # terminal sidecar_persistence_failed JSON (the durable wedge lease, stamped
    # below, bounds the relaunch so a re-fired tick does not double-provision),
    # never an uncaught traceback out of main().
    try:
        write_handle_sidecar(route_result.handle, sidecar)
        recovered = read_handle_sidecar(sidecar)
    except (OSError, json.JSONDecodeError, KeyError, ValueError) as exc:
        # RunPod was ALREADY launched above; the sidecar could NOT be re-pointed.
        # Stamp the DURABLE wedge lease so a re-fired tick on the still-wedged
        # handle short-circuits at _runpod_wedge_already_handled (no second
        # terminate/re-provision).
        stamp_fn(issue, handle)
        return _terminal_infra_json(
            issue=issue,
            sidecar=sidecar,
            reason="sidecar_persistence_failed",
            log_tail=(
                f"RunPod wedge failover re-provisioned {route_result.handle.pod_name} "
                f"but sidecar persistence FAILED ({type(exc).__name__}: {exc}); refusing to "
                f"emit running (durable lease stamped to bound the relaunch)"
            ),
        )
    if recovered.backend != "runpod":
        stamp_fn(issue, handle)
        return _terminal_infra_json(
            issue=issue,
            sidecar=sidecar,
            reason="sidecar_persistence_failed",
            log_tail=(
                f"RunPod wedge failover re-provisioned {route_result.handle.pod_name} "
                f"but the sidecar readback shows backend={recovered.backend!r}, not 'runpod' "
                f"(durable lease stamped to bound the relaunch)"
            ),
        )
    # DURABLE IDEMPOTENCY STAMP (#689 blocker-2; #775 stamp_fn). The fresh pod has
    # launched and the sidecar is authoritatively re-pointed. Stamp the failover
    # field selected by ``stamp_fn`` (runpod_wedge_failover_of by default; the
    # SEPARATE runpod_cuda_ima_failover_of for the CUDA-IMA caller) onto the
    # ~/.eps-routing/ lease so even if the .claude/cache sidecar/sentinel are later
    # lost under EDQUOT, the next poll short-circuits at the lease check instead of
    # firing a paid second terminate + re-provision.
    stamp_fn(issue, handle)
    # The wedge-class wording mirrors success_phase so the operator log reads true to
    # the failover that actually fired (no-port #664 vs CUDA-IMA-repeat #775).
    if success_phase == RUNPOD_CUDA_IMA_FAILOVER_FRESH_POD_PHASE:
        wedge_desc = "same-signature CUDA-IMA repeat wedge (#775)"
    else:
        wedge_desc = "RUNNING-but-no-port wedge (#664)"
    return {
        "status": "running",
        "current_phase": success_phase,
        "new_milestone": True,
        "last_log_mtime_sec_ago": 0,
        "pid_alive": True,
        "log_tail_excerpt": (
            f"RunPod {handle.pod_name} {wedge_desc}; terminated + "
            f"re-provisioned fresh pod {route_result.handle.pod_name}"
        ),
        "gate": None,
        "sentinels_processed": 0,
        "phase_log_mtime_sec_ago": 10**9,
        "shard_log_mtime_sec_ago": 10**9,
        "gpu_util": "unknown",
        "next_interval": _DEFAULT_NEXT_INTERVAL_SEC,
        "issue": int(issue),
    }


def _failover_wedged_runpod(*, issue: int, handle, result, sidecar: Path) -> dict:
    """Terminate a wedged RunPod pod + re-provision fresh, idempotently (#664).

    PRECONDITION (fix (a) dependency): the per-cell inputs-on-HF gate (M1) must
    pass BEFORE the terminate fires. If the gate finds ANY PARTIAL cell, do NOT
    terminate — return a terminal infra JSON
    (``reason="runpod_wedge_inputs_unverified"``) so a human decides (CLAUDE.md
    halt-criterion #2). Idempotency is a durable lease + sidecar sentinel keyed
    to the wedged pod_id (``_runpod_wedge_already_handled``).
    """
    # 1. IDEMPOTENCY SHORT-CIRCUIT (sentinel keyed to the wedged pod identity). A
    #    second tick on the OLD handle after a successful failover short-circuits —
    #    no double-terminate / double-provision.
    if _runpod_wedge_already_handled(issue, handle, sidecar):
        return _terminal_infra_json(
            issue=issue,
            sidecar=sidecar,
            reason="runpod_wedge_already_handled",
            log_tail=(
                f"RunPod wedge failover for {handle.pod_name} already handled on a prior "
                f"tick (idempotency sentinel); refusing a second terminate/re-provision"
            ),
        )

    # 2. PER-CELL INPUTS-ON-HF GATE (M1, fix (a) precondition). A PARTIAL cell (one
    #    artifact-kind on HF, the other not) BLOCKS the irreversible terminate; a
    #    COMPLETE cell's data is preserved and an ABSENT cell reruns on the fresh
    #    pod, so neither blocks.
    gate = _wedged_run_inputs_on_hf(issue, handle)
    if not gate.ok:
        return _terminal_infra_json(
            issue=issue,
            sidecar=sidecar,
            reason="runpod_wedge_inputs_unverified",
            log_tail=(
                f"RunPod {handle.pod_name} wedged (no port); {len(gate.partial)} PARTIAL "
                f"cell(s) on HF (one artifact-kind missing): {gate.partial}. Refusing the "
                f"irreversible terminate (CLAUDE.md halt-criterion #2) — human decision "
                f"needed; complete={len(gate.complete)} absent={len(gate.absent)}"
            ),
        )

    # 3. TERMINATE the billing leak (fail-loud; terminate_pod raises on API error).
    _ensure_scripts_dir_on_sys_path()
    from runpod_api import get_pod_by_name, terminate_pod

    info = get_pod_by_name(handle.pod_name)
    if info is not None:
        terminate_pod(info.pod_id)
        logging.warning(
            "backend_poll: terminated wedged RunPod %s (%s) — billing stopped "
            "(complete=%d absent=%d)",
            handle.pod_name,
            info.pod_id,
            len(gate.complete),
            len(gate.absent),
        )

    # 4. STAMP the idempotency sentinel BEFORE the re-provision so a re-fired tick
    #    on the old handle short-circuits even if the re-provision below raises.
    _write_runpod_wedge_sentinel(_runpod_wedge_sentinel_path(sidecar), issue=issue, handle=handle)

    # 5. RE-PROVISION FRESH + resume the dispatcher (NOT a host-pinned resume). The
    #    fresh pod's P2 WaveDispatcher skips HF-complete cells (A2
    #    _cell_done_anywhere), re-running only the not-yet-run (absent) cells.
    return _relaunch_fresh_runpod(issue=issue, handle=handle, result=result, sidecar=sidecar)


def _runpod_cuda_ima_sentinel_path(sidecar: Path) -> Path:
    """The idempotency sentinel path for a RunPod CUDA-IMA repeat failover (#775).

    The exact sibling of :func:`_runpod_wedge_sentinel_path` with a DISTINCT name
    so a no-port wedge failover and a CUDA-IMA failover on the same issue do not
    share a sentinel: ``issue-<N>-handle.json`` ->
    ``issue-<N>-runpod-cuda-ima-handled.json``.
    """
    name = sidecar.name
    stem = name[: -len("-handle.json")] if name.endswith("-handle.json") else sidecar.stem
    return sidecar.with_name(f"{stem}-runpod-cuda-ima-handled.json")


def _write_runpod_cuda_ima_sentinel(sentinel: Path, *, issue: int, handle) -> None:
    """Persist the RunPod CUDA-IMA failover idempotency sentinel atomically (#775).

    Byte-mirror of :func:`_write_runpod_wedge_sentinel`. Records the crashed
    RunPod run identity (pod_name/job_id) so a subsequent poll on the SAME crashed
    handle short-circuits. A write failure is LOGGED, not raised — worst case one
    extra terminate-and-reprovision on the next tick (the durable lease still
    bounds it), never a silent suppression.
    """
    try:
        sentinel.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "issue": int(issue),
            "reason": "runpod_cuda_ima_repeat_failover",
            "runpod_cuda_ima": _runpod_handle_identity(handle),
        }
        tmp = sentinel.with_suffix(sentinel.suffix + ".tmp")
        tmp.write_text(json.dumps(payload, sort_keys=True, indent=2))
        tmp.replace(sentinel)
    except OSError as exc:
        logging.warning(
            "backend_poll: failed to write RunPod CUDA-IMA idempotency sentinel %s (%s: %s)",
            sentinel,
            type(exc).__name__,
            exc,
        )


def _runpod_cuda_ima_already_handled(issue: int, handle, sidecar: Path) -> bool:
    """Idempotency short-circuit for a re-fired RunPod CUDA-IMA failover (#775).

    Byte-mirror of :func:`_runpod_wedge_already_handled` (the two-record guard):

      1. DURABLE lease (AUTHORITATIVE) — ``runpod_cuda_ima_failover_of`` at
         ``~/.eps-routing/`` survives the EDQUOT mode that fails BOTH the sidecar
         AND the same-dir sentinel.
      2. SENTINEL (fast path) — the ``.claude/cache`` sibling file; a
         corrupted/unreadable sentinel reads as ABSENT (one extra reprovision at
         worst, never a silent suppression).

    Both keyed to the crashed pod's pod_name/job_id, so a genuinely-new run (the
    fresh-host re-provision writes a NEW pod_name) does NOT match a stale record.
    """
    # 1. DURABLE LEASE (authoritative; survives a .claude/cache-wide disk failure).
    if _lease_records_runpod_cuda_ima_failover(issue, handle):
        return True
    # 2. SENTINEL (fast path).
    sentinel = _runpod_cuda_ima_sentinel_path(sidecar)
    prior = _read_failover_sentinel(sentinel)
    return prior is not None and prior.get("runpod_cuda_ima") == _runpod_handle_identity(handle)


def _failover_cuda_ima_runpod(*, issue: int, handle, result, sidecar: Path) -> dict:
    """Pivot a CUDA-IMA-repeat-wedged RunPod run to a FRESH host, bounded once (#775).

    The CUDA-IMA sibling of :func:`_failover_wedged_runpod`, with the once-more
    bound (M1) as the FIRST check (the only structural difference from the no-port
    sibling, which has no analogous per-run bound):

      1. ONCE-MORE BOUND (layer-2, M1). If the DURABLE lease has ANY
         ``runpod_cuda_ima_failover_of`` stamp for THIS run
         (:func:`_lease_has_any_runpod_cuda_ima_failover` — a PER-RUN any-non-null
         check, NOT the PER-POD identity-equality of layer-1, so it survives the
         fresh-pod identity change: the stamp records the OLD crashed pod, the
         SECOND crash arrives on the FRESH handle), this is the SECOND
         same-signature crash on the FRESH host — a fresh host did NOT heal it, so
         it is a deterministic code bug. Emit a terminal
         ``failure_class: code`` JSON (``reason=cuda_ima_repeats_after_failover``)
         via :func:`_terminal_code_json` so the watcher PARKS it at ``blocked``
         (its capacity-retry pass re-drives ONLY ``failure_class: infra`` +
         ``no_compute_available``). NO second pivot.
      2. PER-WEDGE IDEMPOTENCY (layer-1). A re-fired tick on the SAME crashed
         handle after a successful pivot short-circuits (no double-pivot).
      3. INPUTS-ON-HF GATE. Reused as-is — a PARTIAL cell BLOCKS the irreversible
         terminate (human decides, CLAUDE.md halt-criterion #2).
      4. TERMINATE the crashed pod (best-effort — a CUDA-IMA-wedged pod may
         already be dead, so ``info is None`` is fine; terminate is cleanup of the
         billing leak when the pod is still RUNNING).
      5. SENTINEL + RE-PROVISION FRESH via :func:`_relaunch_fresh_runpod` with
         ``stamp_fn=_stamp_runpod_cuda_ima_failover`` (stamps the SEPARATE lease
         field — the once-more bound for the NEXT crash).
    """
    # 1. ONCE-MORE BOUND (M1, PER-RUN): the lease has ANY CUDA-IMA failover stamp
    #    for THIS run -> the fresh host ALSO crashed same-signature -> terminal code.
    #    An any-non-null (NOT identity-equality) check, so it survives the fresh-pod
    #    identity change — the stamp records the OLD crashed pod but this second
    #    crash arrives on the FRESH handle (the identity-keyed layer-1 check would
    #    miss it and pivot again indefinitely).
    if _lease_has_any_runpod_cuda_ima_failover(issue):
        return _terminal_code_json(
            issue=issue,
            sidecar=sidecar,
            reason="cuda_ima_repeats_after_failover",
            log_tail=(
                f"RunPod {handle.pod_name}: a SECOND same-signature CUDA-IMA crash AFTER the one "
                f"bounded fresh-host pivot (#775). A fresh host did not heal it, so this is a "
                f"deterministic code bug, not a transient host wedge — routing to "
                f"failure_class: code -> blocked (no second pivot)."
            ),
        )

    # 2. PER-WEDGE IDEMPOTENCY (a re-fired tick on the OLD crashed handle).
    if _runpod_cuda_ima_already_handled(issue, handle, sidecar):
        return _terminal_infra_json(
            issue=issue,
            sidecar=sidecar,
            reason="runpod_cuda_ima_already_handled",
            log_tail=(
                f"RunPod CUDA-IMA failover for {handle.pod_name} already handled on a prior tick "
                f"(idempotency lease/sentinel); refusing a second terminate/re-provision"
            ),
        )

    # 3. PER-CELL INPUTS-ON-HF GATE (reused as-is; a PARTIAL cell BLOCKS terminate).
    gate = _wedged_run_inputs_on_hf(issue, handle)
    if not gate.ok:
        return _terminal_infra_json(
            issue=issue,
            sidecar=sidecar,
            reason="runpod_cuda_ima_inputs_unverified",
            log_tail=(
                f"RunPod {handle.pod_name} CUDA-IMA repeat wedge; {len(gate.partial)} PARTIAL "
                f"cell(s) on HF (one artifact-kind missing): {gate.partial}. Refusing the "
                f"irreversible terminate (CLAUDE.md halt-criterion #2) — human decision needed; "
                f"complete={len(gate.complete)} absent={len(gate.absent)}"
            ),
        )

    # 4. TERMINATE the crashed pod (BEST-EFFORT — a CUDA-IMA-wedged pod is usually
    #    already dead, so ``info is None`` simply skips the terminate cleanup, and a
    #    terminate API race/failure on a pod the RunPod side has already torn down
    #    (or a transient API hiccup) MUST NOT block the fresh-host recovery. Without
    #    the guard, the ``main()`` outer ``except`` converts ANY terminate exception
    #    to ``reason=runpod_cuda_ima_failover_error``, masking the intended pivot.
    #    Terminating an already-dead pod is operationally idempotent, so on a raise
    #    we log + continue to the sentinel + fresh-host relaunch. (Part C's no-port
    #    terminate stays fail-loud BY DESIGN — that pod is genuinely RUNNING+billing,
    #    so a terminate failure there is the billing leak the wedge exists to stop.)
    _ensure_scripts_dir_on_sys_path()
    from runpod_api import get_pod_by_name, terminate_pod

    info = get_pod_by_name(handle.pod_name)
    if info is not None:
        try:
            terminate_pod(info.pod_id)
            logging.warning(
                "backend_poll: terminated CUDA-IMA-wedged RunPod %s (%s) — billing stopped "
                "(complete=%d absent=%d)",
                handle.pod_name,
                info.pod_id,
                len(gate.complete),
                len(gate.absent),
            )
        except Exception as exc:
            logging.warning(
                "backend_poll: best-effort terminate of CUDA-IMA-wedged RunPod %s (%s) FAILED "
                "(%s: %s); the pod is usually already dead — continuing to the fresh-host "
                "relaunch (terminating an already-dead pod is idempotent)",
                handle.pod_name,
                info.pod_id,
                type(exc).__name__,
                exc,
            )

    # 5. SENTINEL (before the re-provision so a re-fired tick short-circuits even if
    #    the re-provision raises) + RE-PROVISION FRESH, stamping the CUDA-IMA lease.
    _write_runpod_cuda_ima_sentinel(
        _runpod_cuda_ima_sentinel_path(sidecar), issue=issue, handle=handle
    )
    return _relaunch_fresh_runpod(
        issue=issue,
        handle=handle,
        result=result,
        sidecar=sidecar,
        stamp_fn=_stamp_runpod_cuda_ima_failover,
        success_phase=RUNPOD_CUDA_IMA_FAILOVER_FRESH_POD_PHASE,
    )


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


def _lease_records_runpod_wedge_failover(issue: int, handle, *, lease_store=None) -> bool:
    """True iff the DURABLE lease already records a fresh-pod failover of THIS
    wedged RunPod run (#689 blocker-2). The AUTHORITATIVE idempotency check, the
    exact sibling of :func:`_lease_records_failover_of`.

    The ``.claude/cache`` wedge sentinel (:func:`_runpod_wedge_already_handled`'s
    fast path) and the handle sidecar share that dir, so an EDQUOT / read-only-fs
    / out-of-inodes failure that fails the sidecar write fails the sentinel write
    too — re-opening the double-terminate + double-provision hole the GCP round-3
    fix closed. The lease lives at ``~/.eps-routing/`` (a DIFFERENT directory;
    ``LeaseStore`` default), so a failing ``.claude/cache`` mount does not also
    fail the lease.

    Keyed to the wedged pod's stable identity (``pod_name``/``job_id``), so it
    fires "exactly once PER WEDGE", NOT "exactly once per issue": a genuinely-new
    run on the same issue (the fresh-pod re-provision writes a NEW pod_name) does
    NOT match a prior wedge stamp, so a later, distinct wedge still gets its own
    single failover.

    A LeaseStore failure (no ``$HOME``, dir uncreatable) is treated as "no record"
    (return ``False``) — the worst case is one extra terminate + re-provision (the
    same bound the sentinel fast-path provides), NEVER a silent suppression.
    """
    from explore_persona_space.backends.router import LeaseStore

    store = lease_store or LeaseStore()
    try:
        lease = store.read(int(issue))
    except OSError as exc:
        logging.warning(
            "backend_poll: lease-store read failed for issue %s (%s: %s); "
            "treating as no prior-wedge-failover record (the sentinel guard still bounds it)",
            issue,
            type(exc).__name__,
            exc,
        )
        return False
    if lease is None:
        return False
    return lease.runpod_wedge_failover_of == _runpod_handle_identity(handle)


def _stamp_runpod_wedge_failover(issue: int, handle, *, lease_store=None) -> None:
    """Record on the DURABLE lease that a fresh-pod failover of THIS wedged RunPod
    run launched (#689 blocker-2). The exact sibling of
    :func:`_stamp_lease_failover_of`.

    Called IMMEDIATELY after the fresh-pod re-provision SUCCEEDS, so the
    authoritative "exactly once per wedge" record lands at ``~/.eps-routing/``
    regardless of whether the subsequent ``.claude/cache`` sidecar / sentinel
    writes fail under EDQUOT. Stamps ``runpod_wedge_failover_of`` onto the lease
    the router ALREADY wrote inside its own ``store.transaction`` (the fresh-pod
    launch goes through ``_runpod_terminal_rung`` which writes a RunPod lease).

    Best-effort about the LeaseStore itself: a write failure (no ``$HOME``, dir
    uncreatable) is logged, not raised — the failover already launched, and the
    sentinel fast-path + the ``recovered.backend`` guard in
    ``_relaunch_fresh_runpod`` still bound the relaunch. The lease is the SAFETY
    NET for the EDQUOT-on-``.claude/cache`` mode, not a hard precondition.
    """
    from explore_persona_space.backends.router import LeaseStore

    store = lease_store or LeaseStore()
    identity = _runpod_handle_identity(handle)
    try:
        with store.transaction(int(issue)) as (lease, write):
            if lease is None:
                logging.warning(
                    "backend_poll: no lease present to stamp runpod_wedge_failover_of for "
                    "issue %s; relying on the sentinel guard",
                    issue,
                )
                return
            lease.runpod_wedge_failover_of = identity
            write(lease)
    except OSError as exc:
        logging.warning(
            "backend_poll: lease-store wedge-failover stamp failed for issue %s (%s: %s); "
            "the sentinel guard still bounds the relaunch",
            issue,
            type(exc).__name__,
            exc,
        )


def _lease_records_runpod_cuda_ima_failover(issue: int, handle, *, lease_store=None) -> bool:
    """True iff the DURABLE lease already records a fresh-host failover of THIS
    CUDA-IMA-wedged RunPod run (#775). The exact byte-mirror of
    :func:`_lease_records_runpod_wedge_failover`, reading the SEPARATE
    ``runpod_cuda_ima_failover_of`` field (so a no-port wedge failover and a
    CUDA-IMA repeat failover on the same issue do not cross-suppress).

    AUTHORITATIVE for the once-more bound: ``_failover_cuda_ima_runpod`` reads it
    to decide whether THIS run already spent its one bounded CUDA-IMA pivot — if
    so, a second same-signature crash on the fresh host routes to terminal
    ``failure_class: code`` (NOT a second pivot). Keyed to the crashed pod's
    stable identity, so a genuinely-new run on the same issue (the fresh-host
    re-provision writes a NEW pod_name) does NOT match a prior stamp.

    Lives at ``~/.eps-routing/`` (a DIFFERENT directory from ``.claude/cache``),
    so it survives the EDQUOT / read-only-fs mode that fails BOTH the sidecar and
    the same-dir sentinel together — the safety net the GCP round-3 fix
    established. A LeaseStore failure (no ``$HOME``, dir uncreatable) reads as "no
    record" (return ``False``) — worst case one extra pivot, NEVER a silent
    suppression.
    """
    from explore_persona_space.backends.router import LeaseStore

    store = lease_store or LeaseStore()
    try:
        lease = store.read(int(issue))
    except OSError as exc:
        logging.warning(
            "backend_poll: lease-store read failed for issue %s (%s: %s); "
            "treating as no prior CUDA-IMA-failover record (the sentinel guard still bounds it)",
            issue,
            type(exc).__name__,
            exc,
        )
        return False
    if lease is None:
        return False
    return lease.runpod_cuda_ima_failover_of == _runpod_handle_identity(handle)


def _lease_has_any_runpod_cuda_ima_failover(issue: int, *, lease_store=None) -> bool:
    """True iff the issue's DURABLE lease has ANY ``runpod_cuda_ima_failover_of``
    stamp, regardless of WHICH pod it records (#775).

    The PER-RUN layer-2 once-more bound, distinct from the PER-POD layer-1
    idempotency check :func:`_lease_records_runpod_cuda_ima_failover`:

      - **Layer-1** (per-wedge idempotency, ``_runpod_cuda_ima_already_handled``)
        is IDENTITY-keyed (``runpod_cuda_ima_failover_of == identity(handle)``):
        a genuinely-new pod MUST NOT be suppressed, so it compares against the
        CURRENT handle's identity.
      - **Layer-2** (this function — the once-more bound in
        :func:`_failover_cuda_ima_runpod`) must fire "this RUN already spent its
        one CUDA-IMA pivot" REGARDLESS of which pod crashed. A successful pivot
        stamps the OLD (crashed) pod's identity, then re-points the sidecar at
        the FRESH pod; the SECOND CUDA-IMA crash arrives on the FRESH handle, so
        an identity-equality check (layer-1) against the OLD stamp would always
        be ``False`` and the run would pivot again indefinitely (the unbounded-
        spend bug this task exists to prevent). An ANY-non-null check survives
        the fresh-pod identity change. Plan §5 specifies exactly this: the bound
        "checks whether ``runpod_cuda_ima_failover_of`` is set to ANY non-null
        value on the issue's lease".

    A LeaseStore failure (no ``$HOME``, dir uncreatable) reads as "no record"
    (return ``False``) — worst case is one extra pivot, NEVER a silent
    suppression, the same fail-soft contract as the identity-keyed sibling.
    """
    from explore_persona_space.backends.router import LeaseStore

    store = lease_store or LeaseStore()
    try:
        lease = store.read(int(issue))
    except OSError as exc:
        logging.warning(
            "backend_poll: lease-store read failed for issue %s (%s: %s); treating as no "
            "CUDA-IMA-failover-spent record (the per-run once-more bound stays fail-soft)",
            issue,
            type(exc).__name__,
            exc,
        )
        return False
    return lease is not None and lease.runpod_cuda_ima_failover_of is not None


def _stamp_runpod_cuda_ima_failover(issue: int, handle, *, lease_store=None) -> None:
    """Record on the DURABLE lease that a fresh-host failover of THIS
    CUDA-IMA-wedged RunPod run launched (#775). The exact byte-mirror of
    :func:`_stamp_runpod_wedge_failover`, stamping the SEPARATE
    ``runpod_cuda_ima_failover_of`` field.

    Called IMMEDIATELY after the fresh-host re-provision SUCCEEDS (passed to
    :func:`_relaunch_fresh_runpod` via its ``stamp_fn`` kwarg), so the
    authoritative "exactly once per CUDA-IMA wedge" record lands at
    ``~/.eps-routing/`` regardless of whether the subsequent ``.claude/cache``
    sidecar / sentinel writes fail under EDQUOT.

    Best-effort about the LeaseStore itself: a write failure is logged, not
    raised — the failover already launched, and the sentinel fast-path + the
    ``recovered.backend`` guard in ``_relaunch_fresh_runpod`` still bound the
    relaunch. The lease is the SAFETY NET for the EDQUOT-on-``.claude/cache``
    mode, not a hard precondition.
    """
    from explore_persona_space.backends.router import LeaseStore

    store = lease_store or LeaseStore()
    identity = _runpod_handle_identity(handle)
    try:
        with store.transaction(int(issue)) as (lease, write):
            if lease is None:
                logging.warning(
                    "backend_poll: no lease present to stamp runpod_cuda_ima_failover_of for "
                    "issue %s; relying on the sentinel guard",
                    issue,
                )
                return
            lease.runpod_cuda_ima_failover_of = identity
            write(lease)
    except OSError as exc:
        logging.warning(
            "backend_poll: lease-store cuda-ima-failover stamp failed for issue %s (%s: %s); "
            "the sentinel guard still bounds the relaunch",
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

    FAILS LOUD (raises ``ValueError``) on a handle that carries NEITHER a
    workload command NOR hydra args — it NEVER silently launches a blank
    RunPod job (the §4.1.0 spec-threading is a HARD PREREQUISITE; the
    fact-checker confirmed A7=WRONG, the pre-#659 ``extra`` did not carry
    it). The both-empty VALUE check (#1122, matching the RunPod sibling
    :func:`_runspec_from_runpod_handle`) replaced the key-presence check:
    the demonstrated production shape (incident #1090) was an exit-75
    same-command-rerun RECONNECT that rewrote the sidecar without the
    launch-only workload extras — fixed at the write site by #1122
    (``gcp.reconnect_or_none`` threading + the ``issue_dispatch``
    carry-forward) — and keys-present-but-both-empty now also refuses
    loudly instead of building a blank RunSpec.
    """
    from explore_persona_space.backends.base import RunSpec

    extra = handle.extra or {}
    workload_cmd = extra.get("workload_cmd", "") or ""  # str, verbatim (MF1)
    hydra_args = tuple(extra.get("hydra_args") or ())  # list/tuple -> tuple, verbatim
    if not workload_cmd and not hydra_args:
        raise ValueError(
            f"GCP handle for issue {issue} carries no workload_cmd/hydra_args in extra "
            f"(keys present: {sorted(extra)}; reconnected={extra.get('reconnected')!r}). "
            f"Most likely an exit-75 same-command-rerun RECONNECT rewrote the handle "
            f"sidecar without the launch-only workload extras (#1122 — fixed at the "
            f"write site as of that task) — or a pre-#659 handle. Cannot reconstruct "
            f"a RunSpec for the RunPod failover; refusing to launch a blank RunPod job."
        )
    # Failover-time diagnosis breadcrumb (#1329, incident #825): the RunPod
    # relaunch re-runs this GCP workload_cmd verbatim under the RunPod
    # launcher's set -uo pipefail, so a bare reference to a var only GCP
    # exports ($WORKLOAD_ROOT and peers) aborts before the driver starts.
    # Warn-only by design — NEVER alters the spec or blocks the failover (a
    # blocked failover strands the run, worse than the crash it predicts);
    # a lint bug degrades to a logged warning for the same reason.
    if workload_cmd and os.environ.get("EPM_SKIP_WORKLOAD_CMD_ENV_LINT") != "1":
        try:
            from explore_persona_space.backends.issue_dispatch import (
                lint_workload_cmd_lane_env,
            )

            env_lint = lint_workload_cmd_lane_env(
                workload_cmd, backend_value="runpod", execute_workload=True
            )
            if env_lint.flagged:
                logging.warning(
                    "[workload-cmd-lane-env] GCP handle for issue %s reuses a workload_cmd "
                    "referencing lane-specific env var(s) %s bare — UNBOUND on the RunPod "
                    "lane, so the RunPod relaunch of this GCP workload-cmd will abort under "
                    "set -u at the launcher (see #825/#1329; fix the driver to "
                    'self-resolve, e.g. REPO_ROOT="${REPO_ROOT:-${WORKLOAD_ROOT:-$(cd '
                    '"$(dirname "$0")/.." && pwd)}}"). Warn-only: the failover proceeds '
                    "unchanged.",
                    issue,
                    sorted(env_lint.flagged),
                )
        except Exception as exc:
            logging.warning(
                "[workload-cmd-lane-env] lint failed at failover time (%r) — proceeding "
                "with the failover unchanged (#1329 warn-only contract).",
                exc,
            )
    # #909: thread repo_branch through so the RunPod re-execution syncs the
    # ISSUE branch, not `main` (per-issue dispatch scripts live on issue
    # branches). A legacy handle without the key still reconstructs.
    # #1010: ALSO forward the footprint fields (boot_disk_gb / min_ram_gb)
    # so the RunPod CPU-fallback feasibility gate + container-disk threading
    # cover the async failover paths (#659 crash / #783 queue-timeout) —
    # pre-#1010 the rebuilt extra carried ONLY repo_branch, so the gate would
    # fail-OPEN there and the #958 shape could recur. Keys forwarded only when
    # present/truthy, so a legacy handle reconstructs byte-identically.
    rebuilt_extra: dict = {}
    if extra.get("repo_branch"):
        rebuilt_extra["repo_branch"] = extra["repo_branch"]
    for key in ("boot_disk_gb", "min_ram_gb"):
        if extra.get(key):
            rebuilt_extra[key] = extra[key]
    return RunSpec(
        issue=int(issue),
        intent=extra.get("intent", "lora-7b"),
        backend="runpod",
        gpus=extra.get("gpus"),
        time_budget_hours=extra.get("time_budget_hours"),
        workload_cmd=workload_cmd,
        hydra_args=hydra_args,
        extra=rebuilt_extra,
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


def _terminal_code_json(*, issue: int, sidecar: Path, reason: str, log_tail: str) -> dict:
    """A ``status='dead'`` / ``failure_class='code'`` poll JSON keyed by ``reason`` (#775).

    The SIBLING of :func:`_terminal_infra_json` for a CODE-class terminal. Used by
    :func:`_failover_cuda_ima_runpod` for the once-more-exhaustion path: a SECOND
    same-signature CUDA-IMA crash AFTER the one bounded fresh-host pivot means a
    fresh host did NOT fix the crash, so it is a deterministic code bug, not a
    transient host wedge — emit ``failure_class: code`` so the watcher's
    capacity-retry pass (which re-drives ONLY ``failure_class: infra`` +
    ``no_compute_available`` — see ``autonomous_session_watch._is_transient_capacity_block``)
    PARKS the run at ``blocked`` for human inspection rather than re-driving it.

    A separate function rather than a ``failure_class`` kwarg on
    :func:`_terminal_infra_json`, whose NAME + docstring assert ``infra`` — a
    ``failure_class='code'`` value there would contradict the contract a reader
    trusts. The poll-JSON shape is identical to the infra sibling except the
    ``failure_class`` value.
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
        "failure_class": "code",
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

    Thin wrapper over :func:`_failover_gcp_to_runpod` with the #659
    workload-crash labelling; every string / JSON value is byte-identical to the
    pre-refactor #659 behavior (the queue-timeout sibling
    :func:`_failover_queued_gcp_to_runpod` reuses the SAME core with different
    labels + a teardown-first step).
    """
    # Lazy import (module convention: backend_poll -> router imports stay inside
    # functions so the --help path is fast and the import direction is one-way).
    from explore_persona_space.backends.router import (
        ROUTE_REASON_GCP_WORKLOAD_FAILOVER_RUNPOD_ASYNC,
    )

    return _failover_gcp_to_runpod(
        issue=issue,
        handle=handle,
        result=result,
        sidecar=sidecar,
        reason=ROUTE_REASON_GCP_WORKLOAD_FAILOVER_RUNPOD_ASYNC,
        running_phase="gcp_workload_failover_runpod_async",
        cause_label="GCP workload crash",
        evidence_source="async_poller",
        failover_tag="#659 async failover",
        teardown_first=False,
    )


def _failover_queued_gcp_to_runpod(*, issue: int, handle, result, sidecar: Path) -> dict:
    """Cancel a still-queued GCP instance and re-dispatch on RunPod (#783/#778).

    The queue-timeout sibling of :func:`_failover_dead_gcp_to_runpod`. It reuses
    the SAME core (:func:`_failover_gcp_to_runpod`) — hence the SAME idempotency
    short-circuit (durable lease + ``.claude/cache`` sentinel keyed to the GCP
    identity), the SAME
    :func:`~explore_persona_space.backends.router.failover_to_runpod_after_async_workload_crash`
    terminal-rung seam, the SAME authoritative sidecar re-point + terminal-JSON
    contract — with TWO differences the core parameterizes:

    1. ``teardown_first=True`` — best-effort DELETE the still-queued GCP instance
       BEFORE the RunPod re-dispatch. A crashed workload's VM is already gone (so
       #659 does NOT teardown); a QUEUED FLEX_START instance is still live
       server-side and would keep its capacity request (and could dequeue later
       as an orphan), so the queue slot MUST be released. The teardown is
       guarded (never raises) — a failed delete degrades to the stale-GCP-VM
       janitor (``gcp_audit.py``) as the backstop, never blocks the failover.
    2. ``reason=ROUTE_REASON_GCP_QUEUE_TIMEOUT_FAILOVER_RUNPOD`` — the marker
       trail tells a stuck FLEX_START queue apart from a crashed workload.
    """
    # Lazy import (module convention: backend_poll -> router imports stay inside
    # functions so the --help path is fast and the import direction is one-way).
    from explore_persona_space.backends.router import (
        ROUTE_REASON_GCP_QUEUE_TIMEOUT_FAILOVER_RUNPOD,
    )

    return _failover_gcp_to_runpod(
        issue=issue,
        handle=handle,
        result=result,
        sidecar=sidecar,
        reason=ROUTE_REASON_GCP_QUEUE_TIMEOUT_FAILOVER_RUNPOD,
        running_phase=ROUTE_REASON_GCP_QUEUE_TIMEOUT_FAILOVER_RUNPOD,
        cause_label="GCP FLEX_START queue timeout",
        evidence_source="async_poller_queue_timeout",
        failover_tag="#783 queue-timeout failover",
        teardown_first=True,
    )


def _failover_boot_looped_gcp_to_runpod(*, issue: int, handle, result, sidecar: Path) -> dict:
    """Fail a boot-looping GCP rung over to RunPod (#1029).

    The boot-loop sibling of :func:`_failover_dead_gcp_to_runpod` /
    :func:`_failover_queued_gcp_to_runpod`: a thin wrapper over the SAME core
    (:func:`_failover_gcp_to_runpod`) — hence the SAME idempotency
    short-circuit (durable lease + ``.claude/cache`` sentinel keyed to the GCP
    identity), the SAME terminal-rung seam, the SAME authoritative sidecar
    re-point + terminal-JSON contract — with the boot-loop labelling and:

    * ``teardown_first=False`` — the VM self-powered-off and the
      ``--instance-termination-action=DELETE`` reaps it; a lingering record
      degrades to the stale-GCP-VM janitor (``gcp_audit.py``), exactly the
      #659 stance (only the #783 queue-timeout path tears down, because a
      QUEUED instance is still live server-side).
    * ``extra_evidence`` carrying the streak count + rung label, so the
      ``epm:backend-selected`` marker records WHICH rung boot-looped and at
      what streak.
    """
    # Lazy import (module convention: backend_poll -> router imports stay inside
    # functions so the --help path is fast and the import direction is one-way).
    from explore_persona_space.backends.router import (
        ROUTE_REASON_GCP_BOOT_LOOP_FAILOVER_RUNPOD,
        gcp_boot_death_streak,
    )

    extra = getattr(handle, "extra", None) or {}
    rung = str(extra.get("gcp_ladder_rung") or "unknown_rung")
    try:
        streak = gcp_boot_death_streak(int(issue), rung)
    except Exception:
        streak = -1  # unknown (lease unreadable); the failover proceeds regardless
    return _failover_gcp_to_runpod(
        issue=issue,
        handle=handle,
        result=result,
        sidecar=sidecar,
        reason=ROUTE_REASON_GCP_BOOT_LOOP_FAILOVER_RUNPOD,
        running_phase=ROUTE_REASON_GCP_BOOT_LOOP_FAILOVER_RUNPOD,
        cause_label="GCP pre-workload boot loop",
        evidence_source="async_poller_boot_loop",
        failover_tag="#1029 boot-loop failover",
        teardown_first=False,
        extra_evidence={"boot_death_streak": streak, "gcp_ladder_rung": rung},
    )


def _failover_vanished_gcp_to_runpod(*, issue: int, handle, result, sidecar: Path) -> dict:
    """Fail a vanished-while-PENDING GCP instance over to RunPod (#1116/#1112).

    The queue-VANISH sibling of :func:`_failover_queued_gcp_to_runpod` /
    :func:`_failover_boot_looped_gcp_to_runpod`: a thin wrapper over the SAME
    core (:func:`_failover_gcp_to_runpod`) — hence the SAME idempotency
    short-circuit (durable lease + ``.claude/cache`` sentinel keyed to the GCP
    identity), the SAME terminal-rung seam, the SAME authoritative sidecar
    re-point + terminal-JSON contract — with the queue-vanish labelling and:

    * ``teardown_first=False`` — the instance record is already GONE
      server-side (the DWS drop deleted it; that absence IS the trigger), so
      there is nothing to tear down — the #659 stance, NOT #783's (only a
      still-LIVE queued instance needs its capacity request released).
    * ``extra_evidence`` carrying the last observed phase (``"pending"``, the
      clock discriminator) + the ladder-rung label, so the
      ``epm:backend-selected`` marker records WHICH rung's queue dropped the
      request.
    """
    # Lazy import (module convention: backend_poll -> router imports stay inside
    # functions so the --help path is fast and the import direction is one-way).
    from explore_persona_space.backends.router import (
        ROUTE_REASON_GCP_QUEUE_VANISH_FAILOVER_RUNPOD,
    )

    extra = getattr(handle, "extra", None) or {}
    rung = str(extra.get("gcp_ladder_rung") or "unknown_rung")
    return _failover_gcp_to_runpod(
        issue=issue,
        handle=handle,
        result=result,
        sidecar=sidecar,
        reason=ROUTE_REASON_GCP_QUEUE_VANISH_FAILOVER_RUNPOD,
        running_phase=ROUTE_REASON_GCP_QUEUE_VANISH_FAILOVER_RUNPOD,
        cause_label="GCP FLEX_START queue vanish",
        evidence_source="async_poller_queue_vanish",
        failover_tag="#1116 queue-vanish failover",
        teardown_first=False,
        extra_evidence={"last_observed_phase": GCP_PENDING_PHASE, "gcp_ladder_rung": rung},
    )


def _failover_gcp_to_runpod(
    *,
    issue: int,
    handle,
    result,
    sidecar: Path,
    reason: str,
    running_phase: str,
    cause_label: str,
    evidence_source: str,
    failover_tag: str,
    teardown_first: bool,
    extra_evidence: dict | None = None,
) -> dict:
    """Shared core for the GCP->RunPod async failover (#659 crash + #783 queue timeout).

    Reconstructs a ``RunSpec`` from the GCP handle, (optionally) tears down the
    still-live GCP instance first (``teardown_first`` — True ONLY for the #783
    queue-timeout path; a #659 crashed VM is already gone), launches the SAME
    RunPod terminal rung the sync failover uses via
    :func:`~explore_persona_space.backends.router.failover_to_runpod_after_async_workload_crash`
    (passing ``reason``), AUTHORITATIVELY re-points the handle sidecar at the new
    RunPod handle (write + readback), and returns a RUNNING-shaped poll JSON
    (``current_phase = running_phase``) so the orchestrator keeps polling the
    RunPod run instead of posting ``epm:failure``. Returns a TERMINAL infra JSON
    instead if RunPod is unavailable (``no_compute_available``) or the sidecar
    persistence fails (``sidecar_persistence_failed``).

    ``cause_label`` / ``evidence_source`` / ``failover_tag`` are the only
    per-caller display strings; every idempotency + sidecar-repoint + terminal
    path is shared, so the exactly-once bound holds identically for all callers.
    ``extra_evidence`` (#1029, keyword-only, default ``None``) is merged into
    the evidence dict the terminal rung records on the marker — the default
    keeps the #659/#783 callers' evidence byte-identical
    (``test_failover_seam_default_reason_unchanged_byte_for_byte``).
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
    from explore_persona_space.backends.runpod import RunPodBackend, RunPodWorkloadStartError
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
                f"{cause_label} for {handle.pod_name} ALREADY failed over to RunPod on a "
                f"prior tick but failed to persist the sidecar "
                f"({'sentinel ' + sentinel.name if sentinel_match else 'durable lease record'}); "
                f"refusing to launch a SECOND RunPod (exactly-once bound, {failover_tag})"
            ),
        )

    # QUEUE-TIMEOUT teardown (#783). A still-QUEUED FLEX_START instance is live
    # server-side and keeps its capacity request; the queue slot MUST be released
    # before re-dispatch (an undeleted PENDING instance could dequeue later as an
    # orphan). A #659 crashed VM is already gone, so ``teardown_first`` is False
    # there and this block is skipped. AFTER the idempotency short-circuit above,
    # so a repeat poll that already failed over never re-tears-down. Best-effort
    # + guarded — never raises: a failed delete degrades to the stale-GCP-VM
    # janitor (``gcp_audit.py``) as the backstop, never blocks the failover.
    if teardown_first:
        try:
            _resolve_backend("gcp").teardown(handle)
        except Exception as exc:
            logging.warning(
                "backend_poll: teardown of queued GCP %s failed (%s: %s); the stale-GCP-VM "
                "janitor (gcp_audit.py) will reap it — proceeding with the RunPod failover",
                getattr(handle, "pod_name", "?"),
                type(exc).__name__,
                exc,
            )

    spec = _runspec_from_gcp_handle(handle, issue)
    workload_start_error: str | None = None
    try:
        route_result = failover_to_runpod_after_async_workload_crash(
            spec=spec,
            runpod_backend=RunPodBackend(),
            evidence={
                "source": evidence_source,
                "current_phase": result.current_phase,
                "log_tail_excerpt": result.log_tail_excerpt,
                "gcp_pod_name": handle.pod_name,
                # #1029 enrichment slot — empty for the #659/#783 callers
                # (extra_evidence=None), so their evidence stays byte-identical.
                **(extra_evidence or {}),
            },
            reason=reason,
            marker_poster=post_marker_via_task_py,
            # BEST-EFFORT in-route lease-mid-flight write (mirrors
            # dispatch_for_issue's on_launched). _invoke_on_launched SWALLOWS the
            # hook's exceptions (logged loud, not propagated), so this is NOT
            # authoritative — the post-route write/readback below is.
            on_launched=lambda h: write_handle_sidecar(h, sidecar),
            # M3b (#669): the GCP identity this failover is OF, so the router's
            # in-flock re-check + stamp makes N CONCURRENT triggerers (the #669
            # wedge classifier + the watchdog-TERMINATED path on the same handle)
            # launch RunPod exactly once. The OUTSIDE-the-flock pre-check above
            # (sentinel_match / _lease_records_failover_of) is the cheap
            # single-triggerer fast-path; this is the atomic guard.
            gcp_failover_of_identity=_gcp_handle_identity(handle),
        )
        launched_handle = route_result.handle
        already_launched = bool(route_result.extra.get("failover_already_launched"))
    except RunPodWorkloadStartError as exc:
        # PARTIAL failure (#954): the terminal rung PROVISIONED a RunPod pod
        # (it bills, left RUNNING for diagnosis per the #909 contract) but the
        # workload-start leg failed. The rung already persisted the launch
        # records (in-flock lease incl. the gcp_failover_of stamp + the
        # best-effort on_launched sidecar hook) before re-raising typed. Fall
        # through to the SAME authoritative sidecar re-point the success path
        # uses, then emit a DISTINCT terminal (NOT no_compute_available — that
        # mislabel invites the watcher's capacity-retry re-drive while the pod
        # bills, the #931 incident).
        partial = getattr(exc, "handle", None)
        if partial is None:
            # Defensive: unreachable via the rung today (a handle-less start
            # error takes the rung's NoComputeAvailableError branch), kept for
            # a future direct-raise path. Nothing provisioned -> distinct
            # non-re-drivable terminal, sidecar left at GCP, NO stamp.
            return _terminal_infra_json(
                issue=issue,
                sidecar=sidecar,
                reason="runpod_workload_start_failed",
                log_tail=(
                    f"{cause_label} on {handle.pod_name}; RunPod workload start "
                    f"failed with NO pod provisioned ({str(exc)[:500]}) ({failover_tag})"
                ),
            )
        launched_handle = partial
        workload_start_error = str(exc)[:500]
        already_launched = False
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
                f"{cause_label} on {handle.pod_name}; RunPod also unavailable ({failover_tag})"
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
    if already_launched:
        try:
            existing = read_handle_sidecar(sidecar)
        except (OSError, json.JSONDecodeError, KeyError, ValueError):
            existing = None
        if existing is not None and existing.backend == "runpod":
            _clear_failover_sentinel(sentinel)
            return {
                "status": "running",
                "current_phase": running_phase,
                "new_milestone": True,
                "last_log_mtime_sec_ago": 0,
                "pid_alive": True,
                "log_tail_excerpt": (
                    f"{cause_label} on {handle.pod_name}; a concurrent triggerer "
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
        write_handle_sidecar(launched_handle, sidecar)
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
                f"GCP->RunPod failover launched RunPod {launched_handle.pod_name} "
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

    if workload_start_error is not None:
        # PARTIAL failure (#954): the pod is provisioned + billing but NO
        # workload runs on it, so a RUNNING-shaped return would just defer —
        # the next tick polls a workload-less pod (no pidfile -> status=dead)
        # and surfaces a GENERIC failure one tick later, losing the precise
        # reason + the recovery hint. Emit the TERMINAL infra JSON now: the
        # reason is NOT in TRANSIENT_CAPACITY_REASONS, so the watcher parks it
        # for a human instead of auto-churning a fresh paid dispatch.
        return _terminal_infra_json(
            issue=issue,
            sidecar=sidecar,
            reason="runpod_workload_start_failed",
            log_tail=(
                f"{cause_label} on {handle.pod_name}; RunPod {launched_handle.pod_name} "
                f"PROVISIONED but workload start FAILED ({workload_start_error}); pod left "
                f"RUNNING for diagnosis, handle sidecar re-pointed at it (poll/finalize/"
                f"re-drive stay chained); NOT watcher-re-drivable. Recovery: FIRST check "
                f"for a live workload (pidfile + log on the pod — a verify-timeout after a "
                f"successful detach leaves the workload ALIVE; a blind re-drive hits the "
                f"double-fire guard); then re-drive on THIS pod, or stop/terminate it after "
                f"diagnosis — the pod BILLS until a human acts (no cron reaps a RUNNING pod) "
                f"({failover_tag})"
            ),
        )

    # Sidecar is now an AUTHORITATIVE RunPod handle on disk → the orchestrator's
    # NEXT tick reads RunPod and polls RunPod. Emit a RUNNING-shaped JSON so the
    # loop does NOT post epm:failure for the GCP death.
    return {
        "status": "running",
        "current_phase": running_phase,
        "new_milestone": True,
        "last_log_mtime_sec_ago": 0,
        "pid_alive": True,
        "log_tail_excerpt": (
            f"{cause_label} on {handle.pod_name}; failed over to RunPod "
            f"{launched_handle.pod_name} ({failover_tag})"
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
    parser.add_argument(
        "--lane-suffix",
        default=None,
        help=(
            "Per-lane instance-name suffix (#934): resolve the per-lane handle "
            "sidecar issue-<N>-<suffix>-handle.json. Ignored when --handle-file "
            "is given. Pass the SAME suffix the launch used — a forgotten "
            "suffix silently polls the unsuffixed lane's handle instead; "
            "multi-lane orchestrators should prefer --handle-file from the "
            "launch JSON's handle_sidecar_path."
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
        sidecar, probed = resolve_handle_sidecar_path(
            args.issue, args.handle_file, lane_suffix=args.lane_suffix
        )
    except (RuntimeError, ValueError) as exc:
        # RuntimeError: git missing / not a checkout (the pre-#934 case).
        # ValueError (#934): a malformed --lane-suffix failed
        # ``validate_lane_suffix`` inside ``default_handle_sidecar_path`` —
        # fail LOUD but keep the never-empty-stdout contract (an empty
        # stdout spins the bg-Bash poll loop forever).
        stem = f"issue-{int(args.issue)}" + (f"-{args.lane_suffix}" if args.lane_suffix else "")
        fallback = Path(".claude/cache") / f"{stem}-handle.json"
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

    # #1029 boot-streak reset, evaluated on the RAW poll result BEFORE any
    # escalation below rewrites it — a #669 wedge escalation is a MID-WORKLOAD
    # event (boot demonstrably fine), so it must not mask the reset. Fires only
    # on a POSITIVE workload-started signal (running/"workload",
    # dead/terminal_workload_failed, or a #935 done shape); a no-op on every
    # other poll (incl. the mid-boot running/"startup" window).
    _maybe_reset_gcp_boot_streak(handle, result, issue=args.issue)

    # #669 hung-VM wedge escalation: a GCP VM RUNNING-but-hung (eps/phase frozen
    # at a non-terminal value past the staleness floor WITH a transport-class
    # reachability alarm) is rewritten to status=dead / terminal_workload_wedged
    # so the async-failover predicate below matches. A no-op on every other case
    # (non-GCP, not running, phase advancing, within floor, or no reachability
    # alarm) — the staleness clock rides the sidecar's extra dict.
    result = _maybe_escalate_gcp_wedge(handle, result, Path(sidecar), now=time.time())

    # #783 GCP FLEX_START queue-timeout escalation: a GCP instance stuck in the
    # capacity queue (current_phase="pending") past EPS_GCP_QUEUE_WAIT_SECONDS is
    # rewritten to status=dead / terminal_queue_timeout so the queue-timeout
    # predicate below fails it over to RunPod. Mutually exclusive with the #669
    # wedge above BY PHASE ("pending" here vs a frozen mid-workload phase there),
    # so ordering it right after the wedge groups all GCP escalations before the
    # async-workload predicate. A no-op on every other case (non-GCP, not
    # running, phase != pending, first observation, or within the floor) — the
    # queue clock rides the SAME sidecar staleness clock (phase-disjoint).
    result = _maybe_escalate_gcp_queue_timeout(handle, result, Path(sidecar), now=time.time())
    if _is_gcp_queue_timeout(handle, result):
        queue_timeout_json = _failover_queued_gcp_to_runpod(
            issue=args.issue, handle=handle, result=result, sidecar=Path(sidecar)
        )
        print(json.dumps(queue_timeout_json))
        return 0

    # #1116 GCP FLEX_START queue-VANISH escalation: a dead not-found poll
    # (current_phase="terminal_instance not found") whose sidecar phase clock
    # last observed "pending" means the DWS queue dropped the request
    # server-side (create DONE, no delete op — #1112) — a CAPACITY miss, failed
    # over to RunPod on the FIRST occurrence (reason
    # gcp_queue_vanish_failover_runpod, no daily-attempt burn, no teardown —
    # the record is already gone). Input-disjoint with the queue-timeout block
    # above (running/"pending" there vs dead/not-found here) and with the #659
    # predicate below (not-found vs terminal_workload_failed); MUST run BEFORE
    # the #1029 boot-loop recorder — not-found is in its heuristic phase set,
    # and this branch's return is what keeps a pure capacity event from
    # poisoning the boot-death streak. A no-op on every other case (non-GCP,
    # not dead, wrong phase, non-pending/missing clock, or a cpu-bigmem
    # handle, which keeps its ordinary dead path incl. the boot-death record).
    result = _maybe_escalate_gcp_queue_vanish(handle, result, Path(sidecar))
    if _is_gcp_queue_vanish(handle, result):
        queue_vanish_json = _failover_vanished_gcp_to_runpod(
            issue=args.issue, handle=handle, result=result, sidecar=Path(sidecar)
        )
        print(json.dumps(queue_vanish_json))
        return 0

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

    # #1029 GCP boot-loop breaker: a pre-workload setup death (deterministic
    # terminal_setup_failed, or a YOUNG terminated / instance-not-found death)
    # is COUNTED per (issue, rung) in the durable lease; the Nth consecutive
    # one on the same rung is rewritten to terminal_boot_loop and failed over
    # to RunPod (the same core, reason gcp_boot_loop_failover_runpod). Sub-N
    # deaths fall through to the ordinary dead path unchanged. Ordered AFTER
    # the #659 block so a real workload crash keeps its #659 reason, and
    # BEFORE the #775 CUDA-IMA block (a RunPod-only escalation this GCP-only
    # one can never collide with).
    result = _maybe_escalate_gcp_boot_loop(handle, result, issue=args.issue, now=time.time())
    if _is_gcp_boot_loop(handle, result):
        boot_loop_json = _failover_boot_looped_gcp_to_runpod(
            issue=args.issue, handle=handle, result=result, sidecar=Path(sidecar)
        )
        print(json.dumps(boot_loop_json))
        return 0

    # #775 RunPod CUDA-IMA repeat-failover. MUST run BEFORE the no-port escalation
    # below (M2): the no-port within-K path rewrites status=dead -> running, which
    # would mask a CUDA-IMA dead poll that is ALSO transiently no-port (engine
    # dead, ports not yet torn down) before the CUDA-IMA predicate (which requires
    # status="dead") ever ran. A no-op on every non-RunPod / non-dead /
    # non-CUDA-IMA / first-crash / our-code-framed poll (the CUDA-IMA record rides
    # the sidecar's extra dict, fail-soft).
    result = _maybe_escalate_runpod_cuda_ima(
        handle, result, Path(sidecar), issue=args.issue, now=time.time()
    )
    if _is_runpod_cuda_ima_failure(handle, result):
        # Belt-and-suspenders guard (mirrors the no-port wedge guard below):
        # _failover_cuda_ima_runpod maps every internal failure to a terminal JSON,
        # so it should never raise — but the crashed pod may already be terminated,
        # so a bare traceback would strand the run with no re-drive signal.
        try:
            cuda_ima_json = _failover_cuda_ima_runpod(
                issue=args.issue, handle=handle, result=result, sidecar=Path(sidecar)
            )
        except Exception as exc:
            logging.exception("backend_poll: RunPod CUDA-IMA failover raised unexpectedly")
            cuda_ima_json = _terminal_infra_json(
                issue=args.issue,
                sidecar=Path(sidecar),
                reason="runpod_cuda_ima_failover_error",
                log_tail=(
                    f"RunPod CUDA-IMA failover for issue {args.issue} raised "
                    f"{type(exc).__name__}: {exc}; emitting terminal infra JSON "
                    f"(poller terminal-JSON contract)"
                ),
            )
        print(json.dumps(cuda_ima_json))
        return 0

    # #664/#689 RunPod RUNNING-but-no-port host wedge escalation: a RunPod pod
    # whose desiredStatus stays RUNNING with null/empty runtime.ports past
    # RUNPOD_WEDGE_K_SEC (host-pinned resume cannot heal it) is rewritten to
    # status=dead / terminal_runpod_no_port_wedged so the failover predicate below
    # matches. Within the K floor an SSH-dead poll is rewritten to status=running
    # so the orchestrator keeps polling until the wedge matures. A no-op on every
    # non-RunPod / healthy / within-floor case (the no-port clock rides the
    # sidecar's extra dict, fail-soft).
    result = _maybe_escalate_runpod_wedge(handle, result, Path(sidecar), now=time.time())

    # RunPod wedge failover (terminate the billing leak + re-provision a fresh pod),
    # gated on the per-cell three-state inputs-on-HF gate (M1): a PARTIAL cell blocks
    # the irreversible terminate (failure_class: infra block); COMPLETE cells are
    # preserved on HF and ABSENT cells rerun on the fresh pod. Idempotency-guarded.
    if _is_runpod_async_wedge_failure(handle, result):
        # #689 blocker-3: _failover_wedged_runpod maps NoComputeAvailableError +
        # sidecar-write failures to terminal JSON internally (mirroring the GCP
        # analogue), so it should never raise. This belt-and-suspenders guard
        # converts ANY unexpected raise (a bug surfacing in a future edit) into a
        # terminal infra JSON so the poller still honors its terminal-JSON contract
        # — the wedged pod may already be terminated, so a bare traceback would
        # strand the run with no re-drive signal.
        try:
            wedge_json = _failover_wedged_runpod(
                issue=args.issue, handle=handle, result=result, sidecar=Path(sidecar)
            )
        except Exception as exc:
            logging.exception("backend_poll: RunPod wedge failover raised unexpectedly")
            wedge_json = _terminal_infra_json(
                issue=args.issue,
                sidecar=Path(sidecar),
                reason="runpod_wedge_failover_error",
                log_tail=(
                    f"RunPod wedge failover for issue {args.issue} raised "
                    f"{type(exc).__name__}: {exc}; emitting terminal infra JSON "
                    f"(poller terminal-JSON contract)"
                ),
            )
        print(json.dumps(wedge_json))
        return 0

    # ── GCP-lane GPU-idle advisory + escalation (#730; parity with #727) ──
    # A GCP VM idle in a CPU-only / upload phase bleeds credits up to the 24h
    # --max-run-duration DELETE fence with no surfacing — the same #664
    # spend-leak class the RunPod lane got an advisory tier (#518/#537) + an
    # escalation tier (#664/#727) for. Mirror BOTH tiers here, REUSING (not
    # re-implementing) the RunPod-lane decision/post helpers: a one-shot
    # [gpu-idle-advisory] epm:progress marker after EPM_GPU_IDLE_ADVISORY_MIN,
    # then a LOUD [gpu-idle-escalation] marker + Telegram push after
    # EPM_GPU_IDLE_ESCALATION_MIN. It NEVER stops the VM — marker + push only
    # (matched to #727 and the autonomous-mode never-stop-to-park rule). The
    # GCP idle leak is FENCE-BOUNDED at the 24h --max-run-duration DELETE, so
    # it is below-RunPod severity, but still worth surfacing (the imported
    # _maybe_escalate_gpu_idle's note is RunPod-worded — accurate about the
    # leak CLASS + remedy; it takes no note=/extra= kwarg, so the GCP
    # fence-bounded severity is documented HERE, not on the marker, per the
    # import-not-extract single-file-change contract).
    #
    # Lazy import (NOT module-top): the module deliberately keeps poll_pipeline
    # out of the import graph so the --help path stays fast (see the
    # _DEFAULT_NEXT_INTERVAL_SEC comment near the top). The repo-root sys.path
    # bootstrap above makes `scripts.poll_pipeline` resolvable; importing the
    # two wiring fns transitively pulls every dependency they need
    # (_gpu_idle_advisory_update, _gpu_idle_escalation_update, _phase_is_cpu_only,
    # _telegram_push, post_event) — all resolved against poll_pipeline, so no
    # extraction / re-export is needed.
    gcp_gpu_idle_advisory_posted = False
    gcp_gpu_idle_escalation_posted = False
    gcp_gpu_width_advisory_posted = False
    # ``hasattr`` guard: in production the GCP branch always resolves a real
    # GcpBackend (which owns ``_gcp_gpu_util_probe`` + ``_config``), but a
    # duck-typed poll double (the existing test_backend_poll.py ``_PollDouble``)
    # or a future backend variant lacking the probe must SKIP the GPU-idle
    # block rather than crash — same fail-soft defense as the ``getattr``
    # guards in ``_serialize_poll_result``.
    if (
        handle.backend == "gcp"
        and getattr(result, "status", "") == "running"
        and hasattr(backend, "_gcp_gpu_util_probe")
        and getattr(backend, "_config", None) is not None
    ):
        from scripts.poll_pipeline import (
            _maybe_escalate_gpu_idle,
            _maybe_post_gpu_idle_advisory,
            _maybe_post_gpu_width_advisory,
            _run_launched_age_sec,
            _tripwire_run_scope,
        )

        gpu_idle_state_path = _gpu_idle_state_path(Path(sidecar))
        prev_gpu_idle_state = _load_gpu_idle_state(gpu_idle_state_path)
        # ── #1033 per-instance idle-clock scoping (attempt-id keyed) ─────
        # attempt_id is on the handle at poll time (no extra gcloud call),
        # fresh per new instance, label-recovered on reconnect (#927).
        # Scope BEFORE the run-epoch anchor below so ONE consistently-scoped
        # dict flows through every consumer AND the persist at the bottom —
        # an unscoped read at persist time could resurrect a cleared key.
        current_attempt_id = str(handle.extra.get("attempt_id") or "")
        prev_gpu_idle_state = _scope_idle_state_to_attempt(prev_gpu_idle_state, current_attempt_id)
        zone = handle.extra.get("zone") or backend._config.primary_zone
        gpu_util = backend._gcp_gpu_util_probe(handle, zone)
        now_epoch = int(time.time())
        current_phase = getattr(result, "current_phase", "") or ""
        pod = handle.pod_name
        # ── #873/#1033 run-scoped state anchor (AC #6, GCP mirror) ───────
        # A fresh epm:run-launched clears the width dedup keys AND (since
        # #1033) the idle-advisory keys — belt-and-suspenders next to the
        # attempt-id scoping above (a same-instance relaunch posts a fresh
        # run-launched with no attempt change; a fresh instance changes the
        # attempt id whether or not the marker landed). The pre-#1033
        # "idle-advisory keys are untouched by the reset" contract was the
        # bug being fixed (#763/#810 stale idle minutes on fresh VMs). ALL
        # consumers below — idle advisory, escalation, width — read the
        # scoped ``tripwire_state``.
        tripwire_state, tripwire_run_epoch = _tripwire_run_scope(
            prev_gpu_idle_state,
            run_age_sec=_run_launched_age_sec(args.issue, now_epoch),
            now_epoch=now_epoch,
        )

        idle_since, advised_phases, gcp_gpu_idle_advisory_posted = _maybe_post_gpu_idle_advisory(
            issue=args.issue,
            pod=pod,
            status="running",
            gpu_util=gpu_util,
            current_phase=current_phase,
            prev_state=tripwire_state,
            now_epoch=now_epoch,
        )
        escalated_phases, gcp_gpu_idle_escalation_posted = _maybe_escalate_gpu_idle(
            issue=args.issue,
            pod=pod,
            status="running",
            gpu_util=gpu_util,
            current_phase=current_phase,
            idle_since_epoch=idle_since,
            prev_state=tripwire_state,
            now_epoch=now_epoch,
        )
        # ── #873 m-of-N GPU-width advisory (GCP mirror of the RunPod call) ─
        # Same imported wiring fn, same inputs, same sibling state file —
        # the import-not-extract reuse contract (#730). Advisory only.
        gcp_width_since, gcp_width_idle_set, gcp_width_advised, gcp_gpu_width_advisory_posted = (
            _maybe_post_gpu_width_advisory(
                issue=args.issue,
                pod=pod,
                status="running",
                gpu_util=gpu_util,
                current_phase=current_phase,
                prev_state=tripwire_state,
                now_epoch=now_epoch,
            )
        )
        _save_gpu_idle_state(
            gpu_idle_state_path,
            {
                "phase": current_phase,
                "gpu_idle_since_epoch": str(idle_since),
                "gpu_idle_advised_phases": ",".join(sorted(advised_phases)),
                "gpu_idle_escalated_phases": ",".join(sorted(escalated_phases)),
                # #873 width keys + the run-scope anchor (AC #6 mirrored).
                "gpu_width_since_epoch": str(gcp_width_since),
                "gpu_width_idle_set": ",".join(str(i) for i in gcp_width_idle_set),
                "gpu_width_advised_phases": ",".join(sorted(gcp_width_advised)),
                "tripwire_run_epoch": str(tripwire_run_epoch),
                # #1033: the instance incarnation the idle keys above belong
                # to. An empty CURRENT id (fail-safe keep branch) preserves
                # the stored one — read from the SCOPED dict, never the raw
                # load, so a cleared key can never be resurrected here.
                "idle_attempt_id": (
                    current_attempt_id or tripwire_state.get("idle_attempt_id", "")
                ),
            },
        )

    out = _serialize_poll_result(result)
    out["gcp_gpu_idle_advisory_posted"] = gcp_gpu_idle_advisory_posted
    out["gcp_gpu_idle_escalation_posted"] = gcp_gpu_idle_escalation_posted
    out["gcp_gpu_width_advisory_posted"] = gcp_gpu_width_advisory_posted
    print(json.dumps(out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
