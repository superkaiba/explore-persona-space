"""Crash-recovery + pod-safety watcher for autonomous and interactive issue sessions.

Broadened role (two passes, run in this order inside a single daemon-reachability
guard):

1. **Crash-recovery (respawn pass).** Re-spawn an autonomous (`--auto`) `/issue`
   session whose driver process has died.
2. **Pod-safety pass.** Stop (NOT terminate) a RUNNING managed `epm-issue-<N>`
   pod whose driving session is gone and unrecoverable — interactive sessions
   that died, or an autonomous session whose respawn keeps failing — so GPU burn
   is bounded instead of running to the 7-day TTL.

Why each pass exists
--------------------
**Respawn:** the `/loop 10m /issue <N>` driver and any `CronCreate(durable=False)`
backstop live *inside* the session's Claude process, so they die with it — a
process crash / OOM / VM reboot leaves an autonomous experiment stalled until
someone manually `happy resume`s it. This watcher runs OUT of process (a real VM
crontab line, like cron_worktree_audit.sh) and re-spawns the dead session.

**Pod-safety:** an INTERACTIVE per-issue session has no autonomous registry
entry, so the respawn pass cannot touch it; and `pod_audit.py` buckets a
managed-name RUNNING pod as ``active`` and never stops it. If such a session's
process dies with its `epm-issue-<N>` pod still RUNNING, nothing stops the pod —
it burns until the 7-day TTL. The pod-safety pass closes that residual: if a
RUNNING managed pod has no live driving session for ``--threshold`` consecutive
checks, it is STOPPED (reversible — volume preserved; `pod.py resume`
re-provisions). It is never terminated.

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

Pod-safety: after the respawn pass (so a just-respawned autonomous session reads
as alive), the watcher lists team pods, keeps the RUNNING ``epm-issue-<N>`` ones,
and per :func:`decide_pod_safety` decides STOP / KEEP / IGNORE. "Driving session
alive" reuses the same worktree-cwd liveness check as the respawn pass plus the
autonomous registry. Per-pod miss counts persist in their own small state files
(``~/.eps-autonomous/pod-safety-<N>.json``) because interactive issues have no
``issue-<N>.json`` entry. A STOP runs ``pod.py stop --issue <N>`` and posts a
note to the task's events.jsonl. The whole run is gated on daemon reachability
(can't judge liveness during a daemon outage), so neither pass acts then.

Run: ``uv run python scripts/autonomous_session_watch.py [--dry-run] [--threshold N]``
"""

from __future__ import annotations

import argparse
import fcntl
import json
import re
import subprocess
import sys
import time
from pathlib import Path

# scripts/ is sys.path[0] when run as `python scripts/autonomous_session_watch.py`,
# so its siblings import directly. Reuse spawn_session's daemon readers +
# registry constants, and the live RunPod API, rather than duplicating them.
#
# Pod-safety pass (second pass) reaches the live RunPod API directly. The
# managed-name semantics it mirrors live in pod_audit._is_managed_name
# (``pod-*`` / ``epm-issue-*``); here we use the stricter per-issue regex below
# so we both recognise a managed pod AND extract its issue number in one step.
from runpod_api import list_team_pods
from spawn_session import (
    AUTONOMOUS_REGISTRY_DIR,
    PROJECT_ROOT,
    _live_session_ids,
    _load_session_meta,
)

# Matches the canonical per-issue pod name (CLAUDE.md "Pods" § Naming). The
# captured group is the issue number, used to find the driving session + the
# task whose events.jsonl gets the pod-stopped note.
_ISSUE_POD_RE = re.compile(r"^epm-issue-(\d+)$")

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


def decide_pod_safety(
    pod_running: bool, managed: bool, session_alive: bool, missed: int, threshold: int = 2
) -> tuple[str, int]:
    """Pure decision for the pod-safety pass: should a RUNNING managed pod be
    STOPPED because no live session is driving it? Returns ``(action,
    new_missed)`` where action is ``"stop"`` | ``"keep"`` | ``"ignore"``.

    Mirrors :func:`decide` (same consecutive-miss guard, same default
    threshold). Cases:

    - not (``pod_running`` and ``managed``) -> ``("ignore", 0)``. The pass only
      governs RUNNING managed (``epm-issue-<N>``) pods; everything else is
      outside its remit (pod_audit handles EXITED / orphan-running buckets).
    - ``session_alive`` -> ``("keep", 0)``. Something is driving it; reset the
      miss counter. (If that live session then halts on a gate, SKILL.md Step
      8-bis stops the pod in-session — not this pass's job.)
    - otherwise (running managed pod, no live session) -> increment; ``"stop"``
      once the count reaches ``threshold`` (default 2 = ~20 min at a 10-min
      cron, so a single transient daemon-list / cwd glitch never stops a pod),
      else ``("keep", new_missed)``.

    STOP is reversible (``pod.py stop`` preserves the volume; ``resume``
    re-provisions) — never a terminate.
    """
    if not (pod_running and managed):
        return ("ignore", 0)
    if session_alive:
        return ("keep", 0)
    new_missed = missed + 1
    if new_missed >= threshold:
        return ("stop", 0)
    return ("keep", new_missed)


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


def _daemon_reachable() -> bool:
    """True iff the Happy daemon's control server answers /list.

    Critical guard: ``_live_session_ids()`` returns an empty set BOTH when the
    daemon is up with zero sessions AND when it is unreachable. Without
    distinguishing them, a daemon outage would make every recorded session look
    dead and trigger a mass re-spawn (-> duplicate pods). So the watcher probes
    reachability first and skips the whole run if the daemon is down."""
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
    (``.../.claude/worktrees/issue-<N>``). Shared by both passes: the respawn
    pass treats this as "a session is driving this issue" even when the recorded
    Happy id was replaced (manual / PM re-spawn), and the pod-safety pass uses
    it as the interactive-session liveness signal (an interactive `/issue <N>`
    session has no autonomous registry entry, so the worktree cwd is the only
    crash-safe way to tell it is alive)."""
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


# ─── pod-safety pass ─────────────────────────────────────────────────────────

# Per-pod miss counts live in their OWN small state files, separate from the
# autonomous registry (issue-<N>.json), because INTERACTIVE issues — the main
# case this pass exists for — have no registry entry at all.
_POD_SAFETY_PREFIX = "pod-safety-"

# Age backstop: drop a pod-safety state file older than this even when the
# RunPod API is flaky and a pod doesn't show up in the current running set on a
# given tick. Without it, an API outage during the exact tick when a pod
# disappears would strand the state file indefinitely. The cap is generous (well
# past any plausible legitimate miss-accumulation window of 2 ticks ≈ 20 min)
# so it only catches genuinely orphaned files, never live state.
POD_SAFETY_STATE_MAX_AGE_S = 7 * 24 * 3600


def _pod_safety_state_path(issue: int) -> Path:
    return AUTONOMOUS_REGISTRY_DIR / f"{_POD_SAFETY_PREFIX}{issue}.json"


def _load_pod_safety_state(issue: int) -> dict:
    """Read the per-pod miss-count state for ``issue`` (``{}`` if absent /
    unreadable — a fresh/garbled file just starts the miss count at 0)."""
    path = _pod_safety_state_path(issue)
    if not path.is_file():
        return {}
    try:
        return json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return {}


def _save_pod_safety_state(issue: int, pod_id: str, missed: int, prev: dict | None = None) -> None:
    """Persist the per-pod miss count atomically (temp + rename). ``prev`` is
    the existing on-disk payload (if any) — passed in so callers that already
    loaded it don't re-read; `first_seen` carries forward when present so the
    age backstop measures the original episode start, not the latest save."""
    AUTONOMOUS_REGISTRY_DIR.mkdir(parents=True, exist_ok=True)
    dest = _pod_safety_state_path(issue)
    prev_first_seen = (prev or {}).get("first_seen")
    if not isinstance(prev_first_seen, int | float):
        prev_first_seen = time.time()
    payload = {"pod_id": pod_id, "missed": missed, "first_seen": prev_first_seen}
    tmp = dest.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, indent=2))
    tmp.replace(dest)


def _clear_pod_safety_state(issue: int) -> None:
    """Drop the per-pod miss-count file (pod no longer RUNNING, or a session is
    alive again) so a future episode starts its count clean."""
    _pod_safety_state_path(issue).unlink(missing_ok=True)


def _gc_orphan_pod_safety_state(
    running_issues: set[int], dry_run: bool, now: float | None = None
) -> list[int]:
    """GC pod-safety state files for pods that have left the RUNNING set by ANY
    path (manual stop/terminate, self-EXIT on TTL/crash), so a re-used
    ``epm-issue-N`` pod doesn't inherit a stale ``missed`` count and weaken the
    2-miss guard. Also drops files older than ``POD_SAFETY_STATE_MAX_AGE_S`` as
    a secondary backstop in case the API is flaky on the tick when a pod
    actually disappears. Returns the list of issue numbers whose state files
    were cleared (in the order processed)."""
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


def _post_pod_stopped_marker(issue: int, pod_id: str, note: str, dry_run: bool) -> None:
    """Record the pod-stop on task ``issue``'s events.jsonl.

    Uses the generic ``epm:progress`` marker kind: SKILL.md Step 8-bis names
    ``epm:pod-stopped v1`` for an in-session pause, but that kind is NOT declared
    in ``workflow.yaml § markers`` — declaring a new marker schema is out of
    scope for this leaf-node watcher, so we post a generic progress note instead
    (the body text makes the pod-stop self-describing). The watcher runs from
    PROJECT_ROOT on `main`, so the task.py branch-guard is satisfied."""
    if dry_run:
        print(f"  [dry-run] would post epm:progress on #{issue}: {note}")
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
        # The pod was already stopped (the safety win); failing to annotate it is
        # not worth aborting the run. Surface it loudly so the gap is visible.
        print(f"  WARNING: pod #{issue} stopped but marker post failed: {e}", file=sys.stderr)


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
    print(f"  STOPPED pod issue #{issue} (no live driving session): {first_line}")
    return True


def _running_managed_issue_pods() -> list[tuple[int, str]]:
    """Live RunPod team pods that are RUNNING and named ``epm-issue-<N>``.
    Returns ``(issue, pod_id)`` pairs. A transport error surfaces as an empty
    list with a logged warning — better to skip the pass this tick than to crash
    the whole run (the respawn pass already completed)."""
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
        m = _ISSUE_POD_RE.match(p.name or "")
        if m:
            out.append((int(m.group(1)), p.pod_id))
    return out


def _issue_session_alive(issue: int, live_ids: set[str], live_cwds: set[str]) -> bool:
    """True iff SOME live session is driving ``issue``: an interactive session
    in the issue's worktree cwd, OR an autonomous registry entry for the issue
    whose recorded Happy id is still live."""
    if _worktree_session_alive(issue, live_cwds):
        return True
    reg = AUTONOMOUS_REGISTRY_DIR / f"issue-{issue}.json"
    if reg.is_file():
        try:
            entry = json.loads(reg.read_text())
        except (json.JSONDecodeError, OSError):
            return False
        return entry.get("happy_session_id") in live_ids
    return False


def pod_safety_pass(live_ids: set[str], live_cwds: set[str], dry_run: bool, threshold: int) -> None:
    """Stop RUNNING managed (``epm-issue-<N>``) pods whose driving session is
    gone for ``threshold`` consecutive checks. MUST run AFTER the respawn pass
    (so a just-respawned autonomous session reads as alive) and INSIDE the
    daemon-reachability guard (liveness is unknowable during a daemon outage).
    STOP is reversible — never a terminate."""
    running = _running_managed_issue_pods()
    running_issues = {issue for issue, _pod_id in running}

    # GC orphaned state BEFORE the per-pod loop, and ALWAYS — even when
    # `running` is empty — so a state file for a pod that left the RUNNING set
    # by ANY path (manual stop/terminate, self-EXIT on TTL/crash) gets cleared.
    # Otherwise a re-used `epm-issue-N` pod would inherit a stale `missed=1`
    # and be one glitch away from a stop on revival. The age backstop inside
    # `_gc_orphan_pod_safety_state` covers the case where the API is flaky on
    # the exact tick a pod actually disappears.
    _gc_orphan_pod_safety_state(running_issues, dry_run)

    if not running:
        print("pod-safety: no RUNNING epm-issue-* pods")
        return
    print(f"pod-safety: {len(running)} RUNNING epm-issue-* pod(s)")
    for issue, pod_id in running:
        alive = _issue_session_alive(issue, live_ids, live_cwds)
        prev_state = _load_pod_safety_state(issue)
        prev_missed = prev_state.get("missed", 0)
        if not isinstance(prev_missed, int):
            prev_missed = 0
        action, new_missed = decide_pod_safety(
            pod_running=True,
            managed=True,
            session_alive=alive,
            missed=prev_missed,
            threshold=threshold,
        )
        print(
            f"  issue #{issue} pod={pod_id}: session_alive={alive} "
            f"missed={prev_missed}->{new_missed} action={action}"
        )
        if action == "keep" and not alive:
            # Still missing but under threshold — persist the incremented count
            # (carrying `first_seen` forward via `prev_state`).
            if not dry_run:
                _save_pod_safety_state(issue, pod_id, new_missed, prev=prev_state)
        elif action in ("keep", "ignore"):
            # Session alive again (or not our pod) — reset so a future episode
            # starts clean.
            if not dry_run:
                _clear_pod_safety_state(issue)
        elif action == "stop":
            stopped = _stop_pod(issue, dry_run)
            if stopped:
                _post_pod_stopped_marker(
                    issue,
                    pod_id,
                    "stopped by autonomous_session_watch pod-safety pass — RUNNING pod "
                    f"with no live driving session for >= {threshold} checks "
                    f"(pod_id={pod_id}); reversible pause, volume preserved (pod.py resume).",
                    dry_run,
                )
                if not dry_run:
                    _clear_pod_safety_state(issue)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--dry-run", action="store_true", help="log decisions; do not respawn or mutate entries"
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

    # Daemon-reachability gates BOTH passes: _live_session_ids() can't tell
    # "daemon up, zero sessions" from "daemon down", so during an outage every
    # session looks dead — which would mass-respawn AND mass-stop pods. Skip the
    # whole run rather than act on unknowable liveness.
    if not _daemon_reachable():
        print(
            "Happy daemon unreachable; skipping run (won't mass-respawn / mass-stop on an outage)"
        )
        return 0

    live_ids = _live_session_ids()
    meta = _load_session_meta()
    live_cwds = {m.get("path", "") for sid, m in meta.items() if sid in live_ids}

    # Pass 1: crash-recovery respawn (autonomous registry entries only).
    entries = sorted(AUTONOMOUS_REGISTRY_DIR.glob("issue-*.json"))
    print(f"{len(entries)} registered, {len(live_ids)} live session(s)")
    for path in entries:
        _process_entry(path, live_ids, live_cwds, args.dry_run, args.threshold)

    # Pass 2: pod-safety. Runs AFTER pass 1 so a just-respawned session reads as
    # alive. Covers interactive issues (no registry entry) too, so it runs even
    # when there are zero autonomous entries.
    pod_safety_pass(live_ids, live_cwds, args.dry_run, args.threshold)

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
