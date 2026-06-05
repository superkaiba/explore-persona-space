"""Crash-recovery + pod-safety watcher for autonomous and interactive issue sessions.

Two passes, run in this order:

1. **Crash-recovery (respawn pass).** Re-spawn an autonomous (`--auto`) `/issue`
   session whose driver process has died. Gated on daemon reachability — it
   reasons about session liveness, which is unknowable during a daemon outage.
2. **Pod-safety pass.** Reconcile RUNNING managed pods (``pod-<N>`` / legacy
   ``epm-issue-<N>``) against their task's STATUS. Two conservative actions:

   - **AUTO-STOP** (reversible, never terminate) a RUNNING pod whose task is
     already DONE (``completed`` / ``awaiting_promotion`` / ``archived`` /
     ``cancelled``). The experiment is provably finished, so a still-RUNNING
     pod is an escaped pod (Step-8 terminate failed, or it was never run
     through Step 8). Stopping it is unambiguously correct.
   - **ALERT** (loud log + one-time dashboard-visible marker, NO stop) a
     RUNNING pod whose task is in a pod-active status (``approved`` /
     ``running`` / ``uploading`` / ``verifying``) but has shown no real marker
     progress for > ``ALERT_STALE_HOURS``. This is the likely-abandoned
     mid-run case. We do NOT stop it: a false alert is a cheap nudge; a false
     stop would kill a healthy run.

   The pod-safety pass does NOT use session-cwd liveness as a stop trigger
   (see "Why STOP is keyed on task status, not session liveness" below) and
   does NOT need the daemon, so it runs unconditionally — even during a daemon
   outage. Only the respawn pass is daemon-gated.

Why each pass exists
--------------------
**Respawn:** the `/loop 10m /issue <N>` driver and any `CronCreate(durable=False)`
backstop live *inside* the session's Claude process, so they die with it — a
process crash / OOM / VM reboot leaves an autonomous experiment stalled until
someone manually `happy resume`s it. This watcher runs OUT of process (a real VM
crontab line, like cron_worktree_audit.sh) and re-spawns the dead session.

**Pod-safety:** ``pod_audit.py`` buckets a managed-name RUNNING pod as ``active``
and never stops it, so an escaped pod whose experiment is already DONE burns to
the 7-day TTL. The auto-stop arm closes that residual. The alert arm surfaces
the harder mid-run-death case (an interactive session died with its pod RUNNING
mid-experiment) without risking a false stop.

Why STOP is keyed on task status, not session liveness
------------------------------------------------------
An earlier design stopped a pod when no live session was "driving" it, using
cwd-based liveness (a live Happy session whose cwd is the issue's worktree).
That signal is WRONG as a stop trigger: interactive `/issue` sessions are
spawned with cwd = REPO ROOT (the worktree doesn't exist yet at spawn time —
``spawn_session.py``), so a perfectly healthy interactive session reads as
"dead" by the cwd test. Stopping on that signal would kill live experiments.

So the STOP trigger is now task STATUS, which is unambiguous: a ``completed`` /
``awaiting_promotion`` / ``archived`` / ``cancelled`` task provably needs no
pod. Session liveness is gone from the stop path entirely. The mid-run case
(where status alone can't distinguish "healthy long run" from "abandoned") is
handled by the ALERT arm keyed on marker-progress staleness, not by a stop.

Mechanism
---------
Respawn: `spawn_session.py spawn-issue --auto` writes one registry file per issue
at ``~/.eps-autonomous/issue-<N>.json`` recording the Happy session id + cwd +
the GPU-hour cap. This watcher, each run:

  * reads the task's current status (via `task.py view --json`);
  * decides per :func:`decide` whether to RESPAWN / KEEP / DELETE the entry;
  * a session is "alive" iff its recorded id is in the daemon's live set OR a
    live session sits in the issue's worktree (`.claude/worktrees/issue-<N>`);
  * a dead session is only re-spawned after ``--threshold`` (default 2)
    consecutive misses, so a transient daemon-list glitch never double-spawns;
  * single-flight via flock so two overlapping cron fires can't race.

RESPAWN re-invokes `spawn_session.py spawn-issue --auto`, which rewrites the
registry with the new id and ``missed=0``. Parked/terminal tasks are never
re-spawned (see the status sets below); awaiting_promotion is a human gate.

Pod-safety: the watcher lists team pods, keeps the RUNNING managed ones, maps
each to its issue via the canonical ``pod_lifecycle`` helpers, reads each
task's status + latest real-progress timestamp, and per
:func:`decide_pod_safety` decides STOP (done task) / ALERT (stale pod-active
task) / KEEP. AUTO-STOP runs ``pod.py stop --issue <N>`` after the same 2-miss
accumulation as the respawn pass; it is reversible (volume preserved;
``pod.py resume`` re-provisions) and NEVER a terminate. Per-pod miss counts +
the last-observed real-progress timestamp + the alerted flag persist in their
own small state files (``~/.eps-autonomous/pod-safety-<N>.json``) because
interactive issues have no ``issue-<N>.json`` entry.

Run: ``uv run python scripts/autonomous_session_watch.py [--dry-run] [--threshold N]``
"""

from __future__ import annotations

import argparse
import fcntl
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

# scripts/ is sys.path[0] when run as `python scripts/autonomous_session_watch.py`,
# so its siblings import directly. Reuse spawn_session's daemon readers +
# registry constants, the live RunPod API, AND the canonical managed-pod
# helpers from pod_lifecycle (rather than re-deriving a per-issue regex — the
# old `epm-issue-<N>`-only regex never matched the canonical `pod-<N>` names,
# so the whole pass was dead code).
from pod_lifecycle import _is_managed_pod, _issue_from_pod_name
from runpod_api import list_team_pods
from spawn_session import (
    AUTONOMOUS_REGISTRY_DIR,
    PROJECT_ROOT,
    _live_session_ids,
    _load_session_meta,
)

# Active-drive statuses: a dead session here SHOULD be resurrected.
ACTIVE = {"planning", "approved", "running", "verifying", "interpreting", "reviewing"}
# Park statuses: legitimately waiting on the user or a gate — never re-spawn,
# but keep the entry (it may flip back to ACTIVE, e.g. plan_pending -> approved).
PARK = {"proposed", "clarifying", "plan_pending", "blocked"}
# Terminal statuses: the autonomous run is done — drop the entry.
# awaiting_promotion is terminal HERE (experiment finished; the user promotes
# manually — no more auto-driving needed).
TERMINAL = {"awaiting_promotion", "completed", "archived"}

# Hard backstop: drop a registry entry whose task has not progressed in this
# long, so a stuck/unknown-status entry cannot linger and re-spawn forever.
MAX_ENTRY_AGE_S = 14 * 24 * 3600


def decide(status: str, alive: bool, missed: int, threshold: int = 2) -> tuple[str, int]:
    """Pure decision: given a task's status, whether its session is alive, and
    the consecutive-miss count, return ``(action, new_missed)`` where action is
    ``"respawn"`` | ``"keep"`` | ``"delete"``.

    Safety: only an ACTIVE status with a session confirmed dead on
    ``threshold`` consecutive checks (default 2 = ~20 min at a 10-min cron)
    yields ``"respawn"``. Parked tasks reset the miss count and are kept;
    terminal tasks are deleted; an unknown status is kept without ever spawning.
    """
    if status in TERMINAL:
        return ("delete", 0)
    if status in PARK:
        return ("keep", 0)
    if status in ACTIVE:
        if alive:
            return ("keep", 0)
        new_missed = missed + 1
        if new_missed >= threshold:
            return ("respawn", 0)
        return ("keep", new_missed)
    # Unknown status (e.g. a renamed enum): do nothing, keep the entry so a
    # human notices rather than silently dropping or spawning.
    return ("keep", missed)


# ─── pod-safety pass ─────────────────────────────────────────────────────────

# Task statuses for which a still-RUNNING pod is PROVABLY unnecessary: the
# experiment finished (or was abandoned/archived/cancelled), so the pod is an
# escaped one (Step-8 terminate failed, or it never went through Step 8).
# Auto-stopping these is unambiguously safe — there is no live experiment to
# interrupt. `blocked` and `followups_running` are DELIBERATELY excluded: a
# blocked pod may be under active investigation, and a followups_running parent
# pod may still be in use; both are KEPT (alert-only if stale), never
# auto-stopped.
AUTO_STOP_DONE = {"completed", "awaiting_promotion", "archived", "cancelled"}

# Task statuses during which a pod is legitimately in use mid-experiment.
# A RUNNING pod here is NOT auto-stopped (status alone can't tell a healthy
# long run from an abandoned one); instead, if it has shown no real marker
# progress for > ALERT_STALE_HOURS, the alert arm fires (loud log + one-time
# marker), never a stop.
POD_ACTIVE = {"approved", "running", "uploading", "verifying"}

# How long a pod-active task may go without a real progress marker before the
# alert arm fires. Healthy runs post epm:progress regularly (poll_pipeline), so
# a multi-hour gap is a real signal of an abandoned session. A false alert is a
# cheap nudge, so this can be conservative without harm.
ALERT_STALE_HOURS = 6.0

# Per-pod state lives in its OWN small file, separate from the autonomous
# registry (issue-<N>.json), because INTERACTIVE issues — the main case this
# pass exists for — have no registry entry at all.
_POD_SAFETY_PREFIX = "pod-safety-"

# Substring stamped into every alert marker note this pass posts, so the
# staleness check can EXCLUDE the watcher's own alerts from "real progress" —
# otherwise an alert would reset the staleness clock and the gap could never
# grow past the threshold again (the alert would only ever fire once by luck of
# timing). Real progress is "any progress marker NOT posted by this watcher."
_ALERT_NOTE_SENTINEL = "[autonomous_session_watch:pod-stale-alert]"

# Substring stamped into the auto-stop marker note, mirroring the alert
# sentinel. Not used for staleness filtering (a stopped pod's task is DONE, so
# staleness is irrelevant there) but keeps both watcher-posted markers
# self-identifying on the dashboard.
_AUTOSTOP_NOTE_SENTINEL = "[autonomous_session_watch:pod-auto-stop]"

# Age backstop: drop a pod-safety state file older than this even when the
# RunPod API is flaky and a pod doesn't show up in the current running set on a
# given tick. Without it, an API outage during the exact tick when a pod
# disappears would strand the state file indefinitely. The cap is generous (well
# past any plausible legitimate miss-accumulation window of 2 ticks ≈ 20 min)
# so it only catches genuinely orphaned files, never live state.
POD_SAFETY_STATE_MAX_AGE_S = 7 * 24 * 3600


def decide_pod_safety(
    status_class: str, missed: int, stale: bool, alerted: bool, threshold: int = 2
) -> tuple[str, int]:
    """Pure decision for the pod-safety pass on a RUNNING managed pod.

    Trigger is the task's STATUS CLASS (unambiguous), NOT session liveness —
    see the module docstring "Why STOP is keyed on task status". Returns
    ``(action, new_missed)`` where action is ``"stop"`` | ``"alert"`` |
    ``"keep"``.

    Parameters
    ----------
    status_class
        ``"auto-stop-done"`` — task in :data:`AUTO_STOP_DONE` (provably
        finished); ``"pod-active-stale"`` — task in :data:`POD_ACTIVE` AND no
        real marker progress for > :data:`ALERT_STALE_HOURS`;
        ``"pod-active-fresh"`` — task in :data:`POD_ACTIVE` with recent
        progress; ``"other"`` — anything else (e.g. ``blocked``,
        ``followups_running``, an unknown status). ``stale`` is folded into
        ``status_class`` by the caller and kept as a redundant explicit param
        for callers/tests that want to pass it directly.
    missed
        Consecutive-miss count for the auto-stop arm (mirrors :func:`decide`).
    stale
        Whether the task has gone stale (no real progress > threshold). Only
        meaningful when ``status_class`` is pod-active; the caller derives
        ``status_class == "pod-active-stale"`` from it, so this is informational
        for the pod-active path.
    alerted
        Whether a stale-alert has ALREADY been posted for the current episode
        (tracked in the state file). Dedups the alert so it fires once per
        episode, not every 10-min tick.

    Cases:

    - ``status_class == "auto-stop-done"`` -> increment ``missed``; return
      ``"stop"`` once it reaches ``threshold`` (default 2 = ~20 min at a 10-min
      cron, so a single transient API/status glitch never stops a pod), else
      ``("keep", new_missed)``. STOP is reversible (``pod.py stop`` preserves
      the volume) — NEVER a terminate.
    - ``status_class == "pod-active-stale"`` AND not ``alerted`` ->
      ``("alert", 0)``. Loud log + one-time marker. NEVER a stop.
    - ``status_class == "pod-active-stale"`` AND ``alerted`` -> ``("keep", 0)``.
      Already alerted this episode; stay quiet.
    - any other case (``pod-active-fresh``, ``other``) -> ``("keep", 0)``.
      Reset the auto-stop miss counter (the pod is legitimately in use or the
      status is one we deliberately never auto-stop).
    """
    if status_class == "auto-stop-done":
        new_missed = missed + 1
        if new_missed >= threshold:
            return ("stop", 0)
        return ("keep", new_missed)
    if status_class == "pod-active-stale" and not alerted:
        return ("alert", 0)
    # pod-active-stale-already-alerted, pod-active-fresh, other -> hands off.
    return ("keep", 0)


def _status_class(status: str | None, latest_progress_ts: float | None, now: float) -> str:
    """Classify a RUNNING managed pod's task status for :func:`decide_pod_safety`.

    Returns ``"auto-stop-done"`` / ``"pod-active-stale"`` / ``"pod-active-fresh"``
    / ``"other"``. ``status`` of ``None`` (task unreadable) is ``"other"`` —
    never auto-stopped. A pod-active task is ``stale`` when its newest real
    progress marker is older than :data:`ALERT_STALE_HOURS`, OR when there is no
    real progress marker at all (``latest_progress_ts is None``) — a pod-active
    task with zero progress markers is itself a signal worth alerting on.
    """
    if status is None:
        return "other"
    if status in AUTO_STOP_DONE:
        return "auto-stop-done"
    if status in POD_ACTIVE:
        if latest_progress_ts is None:
            return "pod-active-stale"
        if (now - latest_progress_ts) > ALERT_STALE_HOURS * 3600:
            return "pod-active-stale"
        return "pod-active-fresh"
    return "other"


# Progress-ish marker kinds that count as "the experiment made real progress."
# Deliberately broad: any of these advancing means the run is alive. The
# watcher's own alert posts use `epm:progress` too, so they are filtered out by
# the _ALERT_NOTE_SENTINEL note check in _latest_progress_ts (NOT by kind).
_PROGRESS_KINDS = {
    "epm:progress",
    "epm:hot-fix",
    "epm:run-finished",
    "epm:results",
    "epm:status-changed",
    "epm:upload-verification",
    "epm:upload-verified",
    "epm:upload-fix",
    "epm:interpretation",
}


def _parse_event_ts(ts: str | None) -> float | None:
    """Parse a task event ``ts`` (``%Y-%m-%dT%H:%M:%SZ``, UTC) to an epoch
    float, or ``None`` if absent/unparseable."""
    if not isinstance(ts, str) or not ts:
        return None
    try:
        # The canonical format is a trailing 'Z' (UTC). fromisoformat handles
        # '+00:00' but not 'Z' on older pythons, so normalise.
        return datetime.fromisoformat(ts.replace("Z", "+00:00")).timestamp()
    except (ValueError, OSError):
        return None


def _latest_progress_ts(events: list[dict]) -> float | None:
    """Newest epoch timestamp among REAL progress markers in ``events``.

    "Real progress" = an event whose ``kind`` is in :data:`_PROGRESS_KINDS`
    AND whose ``note`` does NOT contain :data:`_ALERT_NOTE_SENTINEL` (the
    watcher's own stale-alert posts use ``epm:progress`` and must NOT count as
    progress — otherwise the alert would reset the staleness clock it is
    measuring). Returns ``None`` when there is no such marker.
    """
    best: float | None = None
    for ev in events:
        if ev.get("kind") not in _PROGRESS_KINDS:
            continue
        if _ALERT_NOTE_SENTINEL in (ev.get("note") or ""):
            continue  # this pass's own alert — not real progress
        ts = _parse_event_ts(ev.get("ts"))
        if ts is not None and (best is None or ts > best):
            best = ts
    return best


def _task_status(issue: int) -> str | None:
    """Current status of task ``issue`` via `task.py view --json`, or ``None``
    if the task no longer exists / cannot be read."""
    try:
        out = subprocess.run(
            ["uv", "run", "python", "scripts/task.py", "view", str(issue), "--json"],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (subprocess.SubprocessError, OSError):
        return None
    if out.returncode != 0:
        return None
    try:
        data = json.loads(out.stdout)
    except json.JSONDecodeError:
        return None
    status = data.get("status") or (data.get("frontmatter") or {}).get("status")
    return status if isinstance(status, str) else None


def _task_events(issue: int) -> list[dict]:
    """All events on task ``issue`` via `task.py list-markers --json`, or ``[]``
    if the task can't be read. Subprocess-isolated (same pattern as
    :func:`_task_status`) so a branch-guard / missing-task error degrades to an
    empty list rather than crashing the pass."""
    try:
        out = subprocess.run(
            [
                "uv",
                "run",
                "python",
                "scripts/task.py",
                "list-markers",
                str(issue),
                "--prefix",
                "epm:",
                "--json",
            ],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (subprocess.SubprocessError, OSError):
        return []
    if out.returncode != 0:
        return []
    try:
        data = json.loads(out.stdout)
    except json.JSONDecodeError:
        return []
    return data if isinstance(data, list) else []


def _daemon_reachable() -> bool:
    """True iff the Happy daemon's control server answers /list.

    Critical guard for the RESPAWN pass only: ``_live_session_ids()`` returns an
    empty set BOTH when the daemon is up with zero sessions AND when it is
    unreachable. Without distinguishing them, a daemon outage would make every
    recorded session look dead and trigger a mass re-spawn (-> duplicate pods).
    So the respawn pass probes reachability first and skips when the daemon is
    down. The pod-safety pass does NOT depend on the daemon (it reasons about
    task status + the live pod list), so it runs regardless."""
    try:
        import urllib.error
        import urllib.request

        from spawn_session import daemon_port

        url = f"http://127.0.0.1:{daemon_port()}/list"
        req = urllib.request.Request(
            url, data=b"{}", headers={"Content-Type": "application/json"}, method="POST"
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            json.loads(resp.read())
        return True
    except (SystemExit, urllib.error.URLError, OSError, json.JSONDecodeError):
        return False


def _worktree_session_alive(issue: int, live_cwds: set[str]) -> bool:
    """True iff a live Happy session's cwd is the issue's worktree dir
    (``.../.claude/worktrees/issue-<N>``). Used ONLY by the respawn pass to
    decide "a session is driving this issue" even when the recorded Happy id was
    replaced (manual / PM re-spawn). NOT used by the pod-safety pass as a stop
    trigger — interactive `/issue <N>` sessions are spawned with cwd = repo root
    (the worktree doesn't exist yet at spawn), so this signal reports a LIVE
    interactive session as dead, and stopping on it would kill healthy pods."""
    return any(p.rstrip("/").endswith(f"/issue-{issue}") for p in live_cwds)


def _session_alive(entry: dict, live_ids: set[str], live_cwds: set[str]) -> bool:
    """A session counts as alive if its recorded Happy id is still tracked by
    the daemon, OR a live session occupies the issue's worktree dir (covers a
    manual / PM re-spawn that replaced the recorded id)."""
    if entry.get("happy_session_id") in live_ids:
        return True
    return _worktree_session_alive(entry.get("issue"), live_cwds)


def _respawn(entry: dict, dry_run: bool) -> bool:
    """Re-spawn the autonomous session for this entry. Returns True on success.
    spawn_session rewrites the registry (new id, missed=0) as a side effect."""
    issue = entry["issue"]
    cap = entry.get("auto_approve_gpu_hours", 24.0)
    cmd = [
        "uv", "run", "python", "scripts/spawn_session.py", "spawn-issue",
        "--issue", str(issue), "--auto", "--auto-approve-gpu-hours", str(cap),
    ]  # fmt: skip
    if dry_run:
        print(f"  [dry-run] would respawn: {' '.join(cmd)}")
        return False
    res = subprocess.run(cmd, cwd=PROJECT_ROOT, capture_output=True, text=True, timeout=120)
    if res.returncode != 0:
        print(f"  RESPAWN FAILED issue #{issue}: {res.stderr.strip()[:300]}", file=sys.stderr)
        return False
    first_line = (res.stdout.strip().splitlines() or [""])[0]
    print(f"  RESPAWNED issue #{issue} (session was dead): {first_line}")
    return True


def _acquire_lock() -> object | None:
    """Single-flight: hold a non-blocking flock so overlapping cron fires don't
    race (a race could double-spawn -> two pods). Returns the held fd, or None
    if another watcher run holds it (caller should exit cleanly)."""
    AUTONOMOUS_REGISTRY_DIR.mkdir(parents=True, exist_ok=True)
    # Held for the whole run (released on process exit) — a context manager
    # would close it and drop the lock, so the bare open is deliberate.
    fd = open(AUTONOMOUS_REGISTRY_DIR / "watch.lock", "w")  # noqa: SIM115
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        fd.close()
        return None
    return fd


# ─── pod-safety state store ──────────────────────────────────────────────────


def _pod_safety_state_path(issue: int) -> Path:
    return AUTONOMOUS_REGISTRY_DIR / f"{_POD_SAFETY_PREFIX}{issue}.json"


def _load_pod_safety_state(issue: int) -> dict:
    """Read the per-pod state for ``issue`` (``{}`` if absent / unreadable — a
    fresh/garbled file just starts the miss count at 0 and alerted at False)."""
    path = _pod_safety_state_path(issue)
    if not path.is_file():
        return {}
    try:
        return json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return {}


def _save_pod_safety_state(
    issue: int,
    pod_id: str,
    missed: int,
    *,
    alerted: bool,
    last_progress_ts: float | None,
    prev: dict | None = None,
) -> None:
    """Persist the per-pod state atomically (temp + rename).

    ``missed`` is the auto-stop consecutive-miss count. ``alerted`` records
    whether a stale-alert was already posted this episode (dedup).
    ``last_progress_ts`` is the newest REAL progress timestamp we observed —
    stored so a later tick can tell "the gap stopped advancing" from "new
    progress arrived" (and reset ``alerted`` when progress advances). ``prev``
    is the existing on-disk payload (if any), passed so callers that already
    loaded it don't re-read; ``first_seen`` carries forward when present so the
    age backstop measures the original episode start, not the latest save.
    """
    AUTONOMOUS_REGISTRY_DIR.mkdir(parents=True, exist_ok=True)
    dest = _pod_safety_state_path(issue)
    prev_first_seen = (prev or {}).get("first_seen")
    if not isinstance(prev_first_seen, int | float):
        prev_first_seen = time.time()
    payload = {
        "pod_id": pod_id,
        "missed": missed,
        "alerted": alerted,
        "last_progress_ts": last_progress_ts,
        "first_seen": prev_first_seen,
    }
    tmp = dest.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, indent=2))
    tmp.replace(dest)


def _clear_pod_safety_state(issue: int) -> None:
    """Drop the per-pod state file (pod no longer RUNNING, or its task left the
    pod-active/done classes) so a future episode starts clean."""
    _pod_safety_state_path(issue).unlink(missing_ok=True)


def _gc_orphan_pod_safety_state(
    running_issues: set[int], dry_run: bool, now: float | None = None
) -> list[int]:
    """GC pod-safety state files for pods that have left the RUNNING set by ANY
    path (manual stop/terminate, self-EXIT on TTL/crash), so a re-used
    ``pod-N`` / ``epm-issue-N`` pod doesn't inherit a stale ``missed`` count and
    weaken the 2-miss guard. Also drops files older than
    ``POD_SAFETY_STATE_MAX_AGE_S`` as a secondary backstop in case the API is
    flaky on the tick when a pod actually disappears. Returns the list of issue
    numbers whose state files were cleared (in the order processed)."""
    if not AUTONOMOUS_REGISTRY_DIR.is_dir():
        return []
    now = now if now is not None else time.time()
    cleared: list[int] = []
    for path in sorted(AUTONOMOUS_REGISTRY_DIR.glob(f"{_POD_SAFETY_PREFIX}*.json")):
        stem = path.stem[len(_POD_SAFETY_PREFIX) :]
        try:
            issue = int(stem)
        except ValueError:
            # Garbled name (`pod-safety-foo.json`) — leave it; a hand-debug
            # artifact is none of the GC's business.
            continue
        if issue in running_issues:
            continue
        # Path 1: pod is no longer RUNNING anywhere we can see. Path 2: age
        # backstop catches a file the API failed to "see-it-go" for.
        try:
            payload = json.loads(path.read_text())
            first_seen = payload.get("first_seen", now)
            if not isinstance(first_seen, int | float):
                first_seen = now
        except (json.JSONDecodeError, OSError):
            first_seen = 0  # unreadable -> definitely orphaned, drop it
        age = now - first_seen
        reason = (
            "not in running set" if age < POD_SAFETY_STATE_MAX_AGE_S else f"age={age / 3600:.1f}h"
        )
        print(f"  pod-safety: GC orphan state issue #{issue} ({reason})")
        if not dry_run:
            path.unlink(missing_ok=True)
        cleared.append(issue)
    return cleared


def _post_progress_marker(issue: int, note: str, dry_run: bool, *, label: str) -> None:
    """Record a pod-safety event on task ``issue``'s events.jsonl.

    Uses the generic ``epm:progress`` marker kind: neither ``epm:pod-stopped``
    nor an ``epm:alert`` kind is declared in ``workflow.yaml § markers``, and
    declaring a new marker schema is out of scope for this leaf-node watcher —
    so we post a generic progress note whose body text (carrying the
    auto-stop / stale-alert sentinel) makes the event self-describing. The
    watcher runs from PROJECT_ROOT on `main`, so the task.py branch-guard is
    satisfied. ``label`` is only for the log line (``auto-stop`` / ``alert``)."""
    if dry_run:
        print(f"  [dry-run] would post epm:progress ({label}) on #{issue}: {note}")
        return
    try:
        subprocess.run(
            [
                "uv",
                "run",
                "python",
                "scripts/task.py",
                "post-marker",
                str(issue),
                "epm:progress",
                "--note",
                note,
            ],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            timeout=60,
            check=True,
        )
    except (subprocess.SubprocessError, OSError) as e:
        # The action (stop / alert) already happened; failing to annotate it is
        # not worth aborting the run. Surface it loudly so the gap is visible.
        print(f"  WARNING: posting {label} marker on #{issue} failed: {e}", file=sys.stderr)


def _stop_pod(issue: int, dry_run: bool) -> bool:
    """Run ``pod.py stop --issue <N>`` (reversible pause; volume preserved).
    Returns True on success. NEVER terminates."""
    cmd = ["uv", "run", "python", "scripts/pod.py", "stop", "--issue", str(issue)]
    if dry_run:
        print(f"  [dry-run] would stop pod: {' '.join(cmd)}")
        return False
    res = subprocess.run(cmd, cwd=PROJECT_ROOT, capture_output=True, text=True, timeout=120)
    if res.returncode != 0:
        print(f"  POD STOP FAILED issue #{issue}: {res.stderr.strip()[:300]}", file=sys.stderr)
        return False
    first_line = (res.stdout.strip().splitlines() or [""])[0]
    print(f"  STOPPED pod issue #{issue} (task is DONE; escaped pod): {first_line}")
    return True


def _running_managed_issue_pods() -> list[tuple[int, str]]:
    """Live RunPod team pods that are RUNNING and managed (``pod-<N>`` or the
    legacy ``epm-issue-<N>``). Returns ``(issue, pod_id)`` pairs.

    Recognition delegates to :func:`pod_lifecycle._is_managed_pod` +
    :func:`pod_lifecycle._issue_from_pod_name` — the canonical helpers that
    handle BOTH the current ``pod-`` prefix and the legacy ``epm-issue-``
    prefix — instead of a hand-rolled regex (the old regex matched only
    ``epm-issue-<N>``, so it never matched any live pod and the whole pass was
    dead code).

    A transport error surfaces as an empty list with a logged warning — better
    to skip the pass this tick than to crash the whole run."""
    try:
        pods = list_team_pods()
    except Exception as e:
        print(
            f"  pod-safety: list_team_pods failed ({e}); skipping pass this tick", file=sys.stderr
        )
        return []
    out: list[tuple[int, str]] = []
    for p in pods:
        if p.desired_status != "RUNNING":
            continue
        if not _is_managed_pod(p):
            continue
        issue = _issue_from_pod_name(p.name or "")
        if issue is not None:
            out.append((issue, p.pod_id))
    return out


def _process_pod(issue: int, pod_id: str, now: float, dry_run: bool, threshold: int) -> None:
    """Reconcile one RUNNING managed pod against its task status.

    Reads the task's status + latest real-progress timestamp, classifies it,
    and applies :func:`decide_pod_safety`: AUTO-STOP a done task's escaped pod
    (after the 2-miss guard), ALERT a stale pod-active task once per episode, or
    KEEP. Persists the per-pod state (miss count, alerted flag, last-observed
    real progress) for the next tick."""
    status = _task_status(issue)
    latest_progress = _latest_progress_ts(_task_events(issue))
    status_class = _status_class(status, latest_progress, now)

    prev_state = _load_pod_safety_state(issue)
    prev_missed = prev_state.get("missed", 0)
    if not isinstance(prev_missed, int):
        prev_missed = 0
    prev_alerted = bool(prev_state.get("alerted", False))
    prev_progress = prev_state.get("last_progress_ts")
    if not isinstance(prev_progress, int | float):
        prev_progress = None

    # If real progress ADVANCED since we last alerted, clear the alerted flag so
    # a fresh staleness episode can alert again (and the gap is measured from
    # the new progress, not the old one). Compare the observed progress ts
    # against what we stored last tick.
    progressed = (
        latest_progress is not None
        and prev_progress is not None
        and latest_progress > prev_progress
    )
    alerted = False if progressed else prev_alerted

    stale = status_class == "pod-active-stale"
    action, new_missed = decide_pod_safety(
        status_class=status_class,
        missed=prev_missed,
        stale=stale,
        alerted=alerted,
        threshold=threshold,
    )
    gap_h = f"{(now - latest_progress) / 3600:.1f}h" if latest_progress is not None else "none"
    print(
        f"  issue #{issue} pod={pod_id}: status={status} class={status_class} "
        f"progress_gap={gap_h} missed={prev_missed}->{new_missed} "
        f"alerted={alerted} action={action}"
    )

    if action == "stop":
        stopped = _stop_pod(issue, dry_run)
        if stopped:
            _post_progress_marker(
                issue,
                f"{_AUTOSTOP_NOTE_SENTINEL} auto-stopped by autonomous_session_watch "
                f"pod-safety pass — RUNNING pod for a task whose status is "
                f"'{status}' (already DONE), so the pod is an escaped / "
                f"Step-8-terminate-failed pod (pod_id={pod_id}); reversible pause, "
                f"volume preserved (pod.py resume). Confirmed for >= {threshold} checks.",
                dry_run,
                label="auto-stop",
            )
            if not dry_run:
                _clear_pod_safety_state(issue)
        return

    if action == "alert":
        _post_progress_marker(
            issue,
            f"{_ALERT_NOTE_SENTINEL} STALE pod-active task: RUNNING pod "
            f"(pod_id={pod_id}) for a task at status '{status}' with no real "
            f"progress marker in > {ALERT_STALE_HOURS:.0f}h "
            f"(gap={gap_h}). Likely an abandoned session — investigate. "
            f"NOT auto-stopped (a mid-run stop risks killing a healthy long "
            f"run); stop manually with `pod.py stop --issue {issue}` if the "
            f"session is truly dead.",
            dry_run,
            label="alert",
        )
        print(
            f"  ALERT issue #{issue}: pod-active task stale > {ALERT_STALE_HOURS:.0f}h "
            f"(gap={gap_h}); NOT stopping (mid-run safety).",
            file=sys.stderr,
        )
        if not dry_run:
            _save_pod_safety_state(
                issue,
                pod_id,
                missed=0,
                alerted=True,
                last_progress_ts=latest_progress,
                prev=prev_state,
            )
        return

    # action == "keep": persist the (possibly incremented) miss count, the
    # alerted flag (reset if progress advanced), and the latest observed
    # progress so the next tick can detect advancement.
    if not dry_run:
        _save_pod_safety_state(
            issue,
            pod_id,
            missed=new_missed,
            alerted=alerted,
            last_progress_ts=latest_progress,
            prev=prev_state,
        )


def pod_safety_pass(dry_run: bool, threshold: int, now: float | None = None) -> None:
    """Reconcile RUNNING managed pods against their task STATUS.

    - AUTO-STOP (reversible, never terminate) a RUNNING pod whose task is DONE
      (:data:`AUTO_STOP_DONE`), after the 2-miss guard — an escaped pod.
    - ALERT (loud log + one-time marker, no stop) a RUNNING pod-active pod with
      no real progress for > :data:`ALERT_STALE_HOURS` — a likely-abandoned
      mid-run session.
    - KEEP everything else.

    Trigger is task STATUS, never session-cwd liveness (which misreports live
    interactive sessions as dead). Does NOT depend on the Happy daemon, so it
    runs unconditionally — even during a daemon outage. STOP is reversible —
    never a terminate."""
    now = now if now is not None else time.time()
    running = _running_managed_issue_pods()
    running_issues = {issue for issue, _pod_id in running}

    # GC orphaned state BEFORE the per-pod loop, and ALWAYS — even when
    # `running` is empty — so a state file for a pod that left the RUNNING set
    # by ANY path (manual stop/terminate, self-EXIT on TTL/crash) gets cleared.
    # Otherwise a re-used `pod-N` would inherit a stale `missed=1` / `alerted`
    # and be one glitch away from a stop on revival. The age backstop inside
    # `_gc_orphan_pod_safety_state` covers the case where the API is flaky on
    # the exact tick a pod actually disappears.
    _gc_orphan_pod_safety_state(running_issues, dry_run, now=now)

    if not running:
        print("pod-safety: no RUNNING managed pods")
        return
    print(f"pod-safety: {len(running)} RUNNING managed pod(s)")
    for issue, pod_id in running:
        _process_pod(issue, pod_id, now, dry_run, threshold)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--dry-run", action="store_true", help="log decisions; do not respawn / stop / mutate"
    )
    parser.add_argument(
        "--threshold",
        type=int,
        default=2,
        help="consecutive dead-checks before re-spawning / stopping a pod "
        "(default 2 = ~20 min at a 10-min cron)",
    )
    args = parser.parse_args(argv)

    lock = _acquire_lock()
    if lock is None:
        print("another autonomous_session_watch run holds the lock; exiting")
        return 0

    # The RESPAWN pass needs the daemon (it reasons about session liveness, and
    # `_live_session_ids()` can't tell "daemon up, zero sessions" from "daemon
    # down" — during an outage every session looks dead, which would
    # mass-respawn -> duplicate pods). The POD-SAFETY pass does NOT: it reasons
    # about task STATUS + the live pod list, neither of which needs the daemon.
    # So the daemon guard gates ONLY the respawn pass; pod-safety runs
    # unconditionally below.
    if _daemon_reachable():
        live_ids = _live_session_ids()
        meta = _load_session_meta()
        live_cwds = {m.get("path", "") for sid, m in meta.items() if sid in live_ids}

        entries = sorted(AUTONOMOUS_REGISTRY_DIR.glob("issue-*.json"))
        print(f"respawn: {len(entries)} registered, {len(live_ids)} live session(s)")
        for path in entries:
            _process_entry(path, live_ids, live_cwds, args.dry_run, args.threshold)
    else:
        print(
            "respawn: Happy daemon unreachable; skipping respawn pass "
            "(won't mass-respawn on an outage). Pod-safety pass still runs."
        )

    # Pod-safety: runs regardless of daemon reachability. Covers interactive
    # issues (no registry entry) too.
    pod_safety_pass(args.dry_run, args.threshold)

    return 0


def _process_entry(
    path: Path, live_ids: set[str], live_cwds: set[str], dry_run: bool, threshold: int
) -> None:
    """Apply one registry entry's decision (read status -> decide -> act).

    Removes the entry on unreadable/missing-task/backstop-age; respawns a dead
    ACTIVE session; otherwise persists an updated miss count. Honours dry_run
    (logs but never mutates / spawns)."""
    try:
        entry = json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        print(f"  {path.name}: unreadable; removing")
        if not dry_run:
            path.unlink(missing_ok=True)
        return

    issue = entry.get("issue")
    status = _task_status(issue)
    if status is None:
        print(f"  issue #{issue}: task not found / unreadable; removing entry")
        if not dry_run:
            path.unlink(missing_ok=True)
        return

    if time.time() - entry.get("spawned_at", 0) > MAX_ENTRY_AGE_S and status not in ACTIVE:
        print(f"  issue #{issue}: entry older than backstop + not active ({status}); removing")
        if not dry_run:
            path.unlink(missing_ok=True)
        return

    alive = _session_alive(entry, live_ids, live_cwds)
    action, new_missed = decide(status, alive, entry.get("missed", 0), threshold)
    print(
        f"  issue #{issue}: status={status} alive={alive} "
        f"missed={entry.get('missed', 0)}->{new_missed} action={action}"
    )

    if action == "delete":
        if not dry_run:
            path.unlink(missing_ok=True)
    elif action == "respawn":
        _respawn(entry, dry_run)  # rewrites the registry on success
    elif action == "keep" and new_missed != entry.get("missed", 0):
        entry["missed"] = new_missed
        if not dry_run:
            path.write_text(json.dumps(entry, indent=2))


if __name__ == "__main__":
    raise SystemExit(main())
