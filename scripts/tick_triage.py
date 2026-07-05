"""One-call tick triage for the /issue-tick and /campaign-tick skills.

The lightweight tick skills used to spend ~5 LLM tool-call turns per fire
(state reads, title refresh, branch logic, snapshot write) even when nothing
needed doing. This script collapses the whole healthy-path decision into ONE
Bash call: it reads the task's status + latest marker through the
task-workflow library (no subprocess shellouts, branch-guard-safe from any
cwd), compares against the previous tick's snapshot, and prints exactly one
verdict line::

    HEALTHY <reason>          # nothing to do — the tick skill ENDS THE TURN
    TERMINAL <reason>         # done/parked — the tick skill tears down its cron
    GATE-TRANSITION <reason>  # just crossed into a user gate — push + teardown
    STALE-REDRIVE <reason>    # chain likely dead — the tick skill loads /issue

Exit code 0 on ANY successful triage (the verdict word carries the decision).
ANY failure (missing task, unreadable registry, unknown status) exits
non-zero with a loud stderr line — the tick skill treats a non-zero exit as
STALE-REDRIVE (fail toward coverage, never toward silence).

Before returning an issue-mode STALE-REDRIVE, ``triage()`` runs a cheap
liveness screen (issue #1051, ``issue_liveness_reason``): a live,
identity-verified detached-phase pid (from the newest in-flight
``stage-dispatch`` breadcrumb), a freshly-appended breadcrumb ``log=`` file,
or a fresh ``[long-phase-heartbeat]`` note converts the verdict to HEALTHY.
Pid-bearing detached-phase evidence is authoritative — a heartbeat never
rescues a dead pid or a cleared phase — so a dead session with a live
detached fit reads HEALTHY until the fit ends, and the first tick after the
pid dies fires STALE-REDRIVE exactly when the re-driven session can harvest
(the intended sequencing; a 48h breadcrumb cap bounds a wedged-leader
latch). Kill switch: ``EPM_TICK_LIVENESS_PROBE=0``.

Side effects (both under ``~/.eps-autonomous``, overridable for tests via
``EPM_TICK_STATE_DIR``):

* ``issue-tick-last-status/<N>.json`` — the per-issue snapshot (same file the
  tick skills wrote before; this script now owns the write). Adds a
  ``terminal_streak`` counter to the legacy ``{issue, status, ts}`` shape.
* ``tick-runaway-<N>.flag`` — written on the ``EPM_TICK_RUNAWAY_STREAK``-th
  (default 3rd) consecutive TEARDOWN-verdict triage (TERMINAL or
  GATE-TRANSITION — covers terminal statuses, over-cap plan_pending, and
  stranded campaign crons); cleared on any streak reset. A cron that keeps
  firing at a teardown site means CRON-TEARDOWN keeps whiffing (the #501
  runaway class: 1,951 wasted ticks over ~40h); the flag is the watcher's
  signal to force-stop the session (``autonomous_session_watch`` gate-push
  pass), which kills the session-scoped cron with it.

CLI::

    uv run python scripts/tick_triage.py <N>                  # /issue-tick
    uv run python scripts/tick_triage.py <N> --kind campaign  # /campaign-tick
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
import time
from datetime import UTC, datetime
from pathlib import Path

# ── status sets (issue mode) ────────────────────────────────────────────────
# Mirror the /issue-tick skill's branch sets. Members must stay inside the
# runtime enum `task_workflow.STATUSES`.
# `on_hold` is teardown-not-redrive: a parking-lot task must never be
# re-driven by a stale tick (that would un-park it), so it lives in TERMINAL
# (tear the cron down), NOT PARK (which STALE-REDRIVEs). It is not a user gate.
ISSUE_TERMINAL = frozenset({"completed", "archived", "awaiting_promotion", "blocked", "on_hold"})
ISSUE_GATE = frozenset({"awaiting_promotion", "blocked"})
ISSUE_PARK = frozenset({"proposed", "planning", "plan_pending", "followups_running"})
ISSUE_ACTIVE = frozenset({"approved", "running", "verifying", "interpreting", "reviewing"})

# ── status sets (campaign mode) ─────────────────────────────────────────────
CAMPAIGN_TERMINAL = frozenset({"completed", "archived", "blocked"})
# A tick should never be armed before the brief-approval gate — a cron seen
# at these statuses is stranded and gets torn down (TERMINAL verdict).
CAMPAIGN_STRANDED = frozenset({"proposed", "planning", "plan_pending"})
CAMPAIGN_ACTIVE = frozenset({"approved", "running"})

# Campaign-state experiment rows in these statuses need no further decision.
CAMPAIGN_ROW_FINISHED = frozenset({"ingested", "abandoned"})
# A child at one of these statuses has LANDED a result (or died) — the
# campaign owes a reconcile/ingest decision round.
CAMPAIGN_CHILD_LANDED = frozenset({"awaiting_promotion", "completed", "blocked"})
# A child at any status outside LANDED + archived is genuinely in flight; its
# own /issue session + the watcher passes cover it, so the campaign can idle.
CAMPAIGN_CHILD_DONEISH = CAMPAIGN_CHILD_LANDED | {"archived"}

STALE_S_DEFAULT = 25 * 60  # the tick skills' long-standing ~25-min staleness window
RUNAWAY_STREAK_DEFAULT = 3

# VM root-disk band labels mirrored from autonomous_session_watch (task #679):
# the tick snapshot carries the same coarse band so a cron-driven tick surfaces
# the same disk signal the watcher writes. Thresholds in GiB-bytes, ordered
# critical < low < sub-floor < ok. Overridable via the SAME env knobs the
# watcher reads, so the two never drift.
_GIB = 2**30


def _env_gib_bytes(name: str, default_gib: float) -> int:
    """GiB env knob -> bytes (garbled/non-positive -> default; never raises)."""
    try:
        val = float(os.environ.get(name, ""))
    except ValueError:
        return int(default_gib * _GIB)
    if not (0 < val < 2**20):
        return int(default_gib * _GIB)
    return int(val * _GIB)


def root_disk_band(free_bytes: int) -> str:
    """Coarse VM-root headroom band for the tick snapshot, mirroring the
    watcher's labels: ``critical`` (<15 GiB) / ``low`` (<20 GiB) /
    ``sub-floor`` (<60 GiB) / ``ok`` (>=60 GiB)."""
    if free_bytes < _env_gib_bytes("EPM_VM_DISK_CRITICAL_GIB", 15):
        return "critical"
    if free_bytes < _env_gib_bytes("EPM_VM_DISK_ALERT_GIB", 20):
        return "low"
    if free_bytes < _env_gib_bytes("EPM_VM_DISK_SUBFLOOR_GIB", 60):
        return "sub-floor"
    return "ok"


def root_disk_snapshot() -> dict | None:
    """``{band, free_gib}`` for the VM root, or ``None`` if disk usage cannot
    be read (the snapshot then simply omits the disk fields — never a crash)."""
    try:
        free = shutil.disk_usage("/").free
    except OSError:
        return None
    return {"band": root_disk_band(free), "free_gib": round(free / _GIB, 1)}


# ── detached-phase liveness (issue #1051) ───────────────────────────────────
# Mirror of autonomous_session_watch.py's opt-in long-phase-heartbeat
# convention (_LONG_PHASE_HEARTBEAT_PREFIX / EPM_LONG_PHASE_HEARTBEAT_FRESH_MIN,
# task #761): the SAME prefix + env knob so the two never drift (the same
# mirror-by-env pattern as the disk bands above). Parity pinned by
# tests/test_tick_triage.py::test_heartbeat_constants_match_watcher.
# Known accepted divergence: the watcher's env parse accepts a non-positive
# EPM_LONG_PHASE_HEARTBEAT_FRESH_MIN; tick falls back to the 90-min default
# instead (the safer behavior — a typo'd var must not disable the window).
LONG_PHASE_HEARTBEAT_PREFIX = "[long-phase-heartbeat]"
LONG_PHASE_HEARTBEAT_FRESH_MIN_DEFAULT = 90.0
# Identity-guard slack: a live /proc/<pid> counts as the breadcrumb's process
# only when its start-epoch <= breadcrumb ts + this slack (the leader starts
# BEFORE the breadcrumb is posted; a recycled pid starts strictly after the
# original died, i.e. after the breadcrumb — see the issue SKILL.md
# "Detached VM-side long compute phases" successor rule).
PID_START_SLACK_S = 120.0
# Safety valve: a pid-bearing breadcrumb older than this never grants HEALTHY
# (bounds a wedged-leader / abandoned-task latch to one wasted re-drive per
# window for ultra-long fits). Env: EPM_TICK_PHASE_BREADCRUMB_MAX_AGE_H.
PHASE_BREADCRUMB_MAX_AGE_H_DEFAULT = 48.0
# Extra clearing kinds for the *running*-stage breadcrumb (STAGE_RESULT_KINDS
# has no "running" key; a completed run posts one of these): once cleared, the
# breadcrumb is dead history and is never probed.
PHASE_CLEARING_EXTRA = frozenset({"epm:results", "epm:upload-verification"})
# Test seam for /proc reads (tests point this at a fake proc tree).
_PROC_ROOT = Path("/proc")

# Watcher-posted campaign markers carry this sentinel in their note; they are
# alerts, not campaign progress, so they never count as freshness.
_WATCHER_NOTE_SENTINEL = "[autonomous_session_watch"


# ── state files ─────────────────────────────────────────────────────────────


def state_dir() -> Path:
    """Root for snapshot + runaway-flag files (``EPM_TICK_STATE_DIR`` for
    tests; defaults to the shared ``~/.eps-autonomous``)."""
    override = os.environ.get("EPM_TICK_STATE_DIR", "").strip()
    return Path(override) if override else (Path.home() / ".eps-autonomous")


def snapshot_path(issue: int) -> Path:
    return state_dir() / "issue-tick-last-status" / f"{issue}.json"


def runaway_flag_path(issue: int) -> Path:
    return state_dir() / f"tick-runaway-{issue}.flag"


def read_snapshot(issue: int) -> dict:
    """Previous tick's snapshot (``{}`` when absent/garbled — a missing
    snapshot means 'previous status unknown')."""
    path = snapshot_path(issue)
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return {}
    return data if isinstance(data, dict) else {}


def write_snapshot(issue: int, status: str, terminal_streak: int) -> None:
    """Atomic temp+rename write of the per-issue snapshot (legacy shape plus
    the ``terminal_streak`` runaway counter)."""
    path = snapshot_path(issue)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "issue": issue,
        "status": status,
        "ts": datetime.now(tz=UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "terminal_streak": terminal_streak,
    }
    # VM root-disk band, mirroring the watcher's labels (task #679): a
    # cron-driven tick surfaces the same disk signal the watcher writes. Omitted
    # when disk usage can't be read (never blocks the snapshot).
    disk = root_disk_snapshot()
    if disk is not None:
        payload["root_disk"] = disk
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), prefix=f".{issue}-")
    try:
        with os.fdopen(fd, "w") as fh:
            json.dump(payload, fh)
        os.replace(tmp, path)
    except OSError:
        Path(tmp).unlink(missing_ok=True)
        raise


def write_runaway_flag(issue: int, status: str, streak: int) -> None:
    """Drop the runaway flag for the watcher's force-stop check. Idempotent
    (overwrites); content is diagnostic only."""
    path = runaway_flag_path(issue)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "issue": issue,
        "status": status,
        "terminal_streak": streak,
        "ts": datetime.now(tz=UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), prefix=f".runaway-{issue}-")
    try:
        with os.fdopen(fd, "w") as fh:
            json.dump(payload, fh)
        os.replace(tmp, path)
    except OSError:
        Path(tmp).unlink(missing_ok=True)
        raise


# ── task-state readers (lazy task_workflow imports; monkeypatchable) ────────


def load_task_state(issue: int) -> tuple[str, list[dict]]:
    """Return ``(status, events)`` via the task-workflow library.

    Raises on ANY read failure — main() converts that to a loud non-zero
    exit so the tick skill falls back to the full re-drive path."""
    from explore_persona_space.task_workflow import get_task, list_events

    task = get_task(issue)
    status = task.get("status")
    if not isinstance(status, str) or not status:
        raise ValueError(f"task #{issue}: unreadable status")
    return status, list_events(issue)


def load_children(issue: int) -> list[dict]:
    """Campaign mode: the child-task rows (id/status) via the library."""
    from explore_persona_space.task_workflow import list_children

    return list_children(issue)


def load_campaign_state(issue: int) -> dict:
    """Campaign mode: ``artifacts/campaign-state.json`` (``{}`` if absent —
    a campaign with no state file yet owes a decision round, which the
    verdict logic surfaces as STALE-REDRIVE via the unreconciled check)."""
    from explore_persona_space.task_workflow import find_task_path

    path = find_task_path(issue) / "artifacts" / "campaign-state.json"
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return {}
    return data if isinstance(data, dict) else {}


def parse_event_ts(ts: str | None) -> float | None:
    """ISO-8601 ``Z`` timestamp -> epoch seconds (``None`` on garbage)."""
    if not isinstance(ts, str) or not ts:
        return None
    try:
        return datetime.fromisoformat(ts.replace("Z", "+00:00")).timestamp()
    except ValueError:
        return None


def latest_event_ts(events: list[dict], *, prefix: str | None = None) -> float | None:
    """Epoch ts of the newest event (optionally restricted to a kind prefix;
    watcher-sentinel notes never count as freshness)."""
    best: float | None = None
    for row in events:
        if not isinstance(row, dict):
            continue
        kind = row.get("kind", "")
        if prefix is not None and not str(kind).startswith(prefix):
            continue
        note = row.get("note")
        if isinstance(note, str) and _WATCHER_NOTE_SENTINEL in note:
            continue
        ts = parse_event_ts(row.get("ts"))
        if ts is not None and (best is None or ts > best):
            best = ts
    return best


def plan_pending_over_cap(events: list[dict]) -> bool:
    """True iff the newest ``epm:awaiting-spend-approval`` marker is newer
    than the newest ``epm:status-changed`` — the over-cap plan_pending park
    (a user gate), vs the under-cap in-skill park."""
    spend = latest_event_ts(events, prefix="epm:awaiting-spend-approval")
    if spend is None:
        return False
    changed = latest_event_ts(events, prefix="epm:status-changed")
    return changed is None or spend >= changed


# ── detached-phase liveness probes (issue #1051) ────────────────────────────


def liveness_probe_enabled() -> bool:
    """``EPM_TICK_LIVENESS_PROBE`` kill switch (default on; ``0``/``false``
    disables — restores pre-#1051 STALE-REDRIVE behavior fleet-wide)."""
    raw = os.environ.get("EPM_TICK_LIVENESS_PROBE", "").strip().lower()
    return raw not in ("0", "false")


def heartbeat_fresh_s() -> float:
    """``EPM_LONG_PHASE_HEARTBEAT_FRESH_MIN`` (minutes) -> seconds; malformed
    or non-positive -> ``LONG_PHASE_HEARTBEAT_FRESH_MIN_DEFAULT`` (the same
    fallback shape as ``stale_s()``; see the constants block for the noted
    divergence from the watcher's parse)."""
    raw = os.environ.get("EPM_LONG_PHASE_HEARTBEAT_FRESH_MIN", "")
    try:
        val = float(raw)
    except ValueError:
        return LONG_PHASE_HEARTBEAT_FRESH_MIN_DEFAULT * 60.0
    return val * 60.0 if val > 0 else LONG_PHASE_HEARTBEAT_FRESH_MIN_DEFAULT * 60.0


def phase_breadcrumb_max_age_s() -> float:
    """``EPM_TICK_PHASE_BREADCRUMB_MAX_AGE_H`` (hours) -> seconds; malformed
    or non-positive -> ``PHASE_BREADCRUMB_MAX_AGE_H_DEFAULT`` (same shape as
    ``stale_s()``)."""
    raw = os.environ.get("EPM_TICK_PHASE_BREADCRUMB_MAX_AGE_H", "")
    try:
        val = float(raw)
    except ValueError:
        return PHASE_BREADCRUMB_MAX_AGE_H_DEFAULT * 3600.0
    return val * 3600.0 if val > 0 else PHASE_BREADCRUMB_MAX_AGE_H_DEFAULT * 3600.0


def latest_heartbeat_ts(events: list[dict]) -> float | None:
    """Epoch ts of the newest ``epm:progress`` note containing
    ``LONG_PHASE_HEARTBEAT_PREFIX``, excluding watcher-sentinel notes
    (``_WATCHER_NOTE_SENTINEL``, same exclusion as ``latest_event_ts``).
    ``None`` when absent. Callers compute ``age = now - ts`` and treat
    ``age < 0`` as NOT fresh (future ts — mirrors the watcher's
    ``_long_phase_heartbeat_reason`` clock-skew guard) and compare ``ts``
    against ``latest_clearing_ts`` (a heartbeat at or older than the newest
    clearing event is invalidated)."""
    best: float | None = None
    for row in events:
        if not isinstance(row, dict):
            continue
        if not str(row.get("kind", "")).startswith("epm:progress"):
            continue
        note = row.get("note")
        if not isinstance(note, str) or LONG_PHASE_HEARTBEAT_PREFIX not in note:
            continue
        if _WATCHER_NOTE_SENTINEL in note:
            continue
        ts = parse_event_ts(row.get("ts"))
        if ts is not None and (best is None or ts > best):
            best = ts
    return best


def latest_clearing_ts(events: list[dict]) -> float | None:
    """Epoch ts of the newest stage-clearing event (the union of ALL
    ``STAGE_RESULT_KINDS`` values across stages, plus ``epm:failure`` and
    ``PHASE_CLEARING_EXTRA``). A heartbeat at or older than this never grants
    HEALTHY — the phase that emitted it is over. ``None`` when absent."""
    from explore_persona_space.task_workflow import STAGE_RESULT_KINDS

    clearing: set[str] = {"epm:failure"} | PHASE_CLEARING_EXTRA
    for kinds in STAGE_RESULT_KINDS.values():
        clearing |= kinds
    best: float | None = None
    for row in events:
        if not isinstance(row, dict):
            continue
        kind = str(row.get("kind", ""))
        if not any(kind.startswith(k) for k in clearing):
            continue
        ts = parse_event_ts(row.get("ts"))
        if ts is not None and (best is None or ts > best):
            best = ts
    return best


def newest_inflight_pid_breadcrumb(events: list[dict]) -> dict | None:
    """The newest ``epm:progress`` event whose lstripped note starts
    ``"stage-dispatch "`` AND carries an integer ``pid=`` field, with NO
    later stage-clearing event.

    Returns ``{"pid": int, "ts": float, "log": str | None}`` or ``None``.
    Clearing set = ``STAGE_RESULT_KINDS.get(_normalize_stage(stage),
    frozenset()) | {"epm:failure"} | PHASE_CLEARING_EXTRA``. Malformed pid /
    ts -> skipped. Scans newest-first; the first pid-bearing breadcrumb
    decides (if cleared -> return ``None``; probing an OLDER phase's pid
    would resurrect dead history)."""
    from explore_persona_space.task_workflow import (
        STAGE_RESULT_KINDS,
        _breadcrumb_fields,
        _normalize_stage,
    )

    for idx in range(len(events) - 1, -1, -1):
        row = events[idx]
        if not isinstance(row, dict):
            continue
        if not str(row.get("kind", "")).startswith("epm:progress"):
            continue
        note = row.get("note")
        if not isinstance(note, str):
            continue
        stripped = note.lstrip()
        if not stripped.startswith("stage-dispatch "):
            continue
        fields = _breadcrumb_fields(stripped)
        pid_raw = fields.get("pid")
        if pid_raw is None:
            continue
        try:
            pid = int(pid_raw)
        except ValueError:
            continue
        ts = parse_event_ts(row.get("ts"))
        if ts is None:
            continue
        clearing = (
            STAGE_RESULT_KINDS.get(_normalize_stage(fields.get("stage", "")), frozenset())
            | {"epm:failure"}
            | PHASE_CLEARING_EXTRA
        )
        for later in events[idx + 1 :]:
            if not isinstance(later, dict):
                continue
            later_kind = str(later.get("kind", ""))
            if any(later_kind.startswith(k) for k in clearing):
                return None  # newest pid crumb already cleared — dead history
        log = fields.get("log")
        return {"pid": pid, "ts": ts, "log": log if isinstance(log, str) and log else None}
    return None


def proc_start_epoch(pid: int) -> float | None:
    """Start time (epoch seconds) of ``/proc/<pid>``: ``btime`` (from
    ``_PROC_ROOT/'stat'``) + ``starttime`` (field 22 of
    ``_PROC_ROOT/<pid>/'stat'``, parsed after the LAST ``)`` so a comm
    containing ``') '`` cannot shift fields) divided by
    ``os.sysconf('SC_CLK_TCK')``. ``None`` on any read/parse failure
    (process dead, permission, malformed)."""
    try:
        btime: float | None = None
        for line in (_PROC_ROOT / "stat").read_text().splitlines():
            if line.startswith("btime "):
                btime = float(line.split()[1])
                break
        if btime is None:
            return None
        stat = (_PROC_ROOT / str(pid) / "stat").read_text()
        after_comm = stat.rsplit(")", 1)[1].split()
        # after_comm[0] is field 3 (state); field 22 (starttime) is index 19.
        starttime_ticks = float(after_comm[19])
        return btime + starttime_ticks / os.sysconf("SC_CLK_TCK")
    except (OSError, ValueError, IndexError):
        return None


def pid_alive_with_identity(pid: int, launched_before_epoch: float) -> bool:
    """True iff ``proc_start_epoch(pid)`` is not ``None`` AND
    ``start_epoch <= launched_before_epoch + PID_START_SLACK_S``.

    The identity guard: only one process holds a pid at a time, and the
    breadcrumb recorded this pid at its ts — so a live pid whose start
    PRECEDES the breadcrumb IS the recorded process; a recycled pid's start
    strictly FOLLOWS the original's death (after the breadcrumb ts). Never a
    bare existence check (the issue SKILL.md successor rule: pid recycling
    on a shared VM)."""
    start = proc_start_epoch(pid)
    return start is not None and start <= launched_before_epoch + PID_START_SLACK_S


def log_fresh_age_s(log_path: str, now: float, window_s: float) -> float | None:
    """``now - mtime`` of ``log_path`` iff the path is absolute and
    ``0 <= age < window_s``, else ``None``. ``os.stat`` guarded
    (missing/unreadable -> ``None``). The path is used internally ONLY —
    never printed (content invariant: label slugs embed in log paths)."""
    if not isinstance(log_path, str) or not log_path.startswith("/"):
        return None
    try:
        mtime = os.stat(log_path).st_mtime
    except OSError:
        return None
    age = now - mtime
    return age if 0 <= age < window_s else None


def issue_liveness_reason(events: list[dict], now: float, stale_after_s: float) -> str | None:
    """Precedence-ordered liveness screen before an issue-mode STALE-REDRIVE
    (issue #1051). Returns a content-invariant-safe reason suffix (pids/ages
    only) or ``None``. Entirely exception-guarded: any probe failure falls
    through to STALE-REDRIVE (fail toward coverage) while the verdict line
    stays informative.

    PRECEDENCE: pid-bearing detached-phase evidence is AUTHORITATIVE over
    heartbeat evidence — a fresh ``[long-phase-heartbeat]`` note can never
    rescue a dead/identity-failed pid or a cleared phase. Unlike the watcher
    (which has no pid evidence), tick_triage holds the pid breadcrumb in the
    SAME events list, and the emitter convention (issue SKILL.md § Detached
    VM-side long compute phases) makes heartbeat+pid co-occurrence the
    designed steady state — heartbeat-first ordering would mask a dead pid
    for up to 90 min.

    Legs: (0) probe disabled -> ``None``. (1) AUTHORITATIVE — the newest
    in-flight (un-cleared) pid-bearing breadcrumb younger than the 48h cap:
    live identity-verified pid -> HEALTHY; else fresh ``log=`` mtime ->
    HEALTHY; else ``None`` with NO heartbeat rescue (a dead pid / silent log
    is stronger evidence than any note). (2) FALLBACK (no such breadcrumb —
    incl. cleared or over-max-age crumbs) — a fresh heartbeat note newer
    than the newest clearing event -> HEALTHY. (3) ``None``."""
    try:
        if not liveness_probe_enabled():
            return None
        crumb = newest_inflight_pid_breadcrumb(events)
        if crumb is not None and (now - crumb["ts"]) < phase_breadcrumb_max_age_s():
            if pid_alive_with_identity(crumb["pid"], crumb["ts"]):
                return (
                    f"detached phase alive (pid {crumb['pid']}, "
                    f"breadcrumb age {(now - crumb['ts']) / 3600:.1f}h)"
                )
            if crumb["log"]:
                log_age = log_fresh_age_s(crumb["log"], now, stale_after_s)
                if log_age is not None:
                    return (
                        f"detached phase log appended {log_age / 60:.0f}m ago (pid probe negative)"
                    )
            return None  # dead/unverifiable pid + silent log: NO heartbeat rescue
        hb_ts = latest_heartbeat_ts(events)
        if hb_ts is not None:
            hb_age = now - hb_ts
            window = heartbeat_fresh_s()
            if 0 <= hb_age < window:
                cleared = latest_clearing_ts(events)
                if cleared is None or hb_ts > cleared:
                    return f"long-phase heartbeat fresh ({hb_age / 60:.0f}m < {window / 60:.0f}m)"
        return None
    except Exception:
        return None


# ── pure verdict logic ──────────────────────────────────────────────────────


def stale_s() -> float:
    raw = os.environ.get("EPM_TICK_STALE_S", "")
    try:
        val = float(raw)
    except ValueError:
        return STALE_S_DEFAULT
    return val if val > 0 else STALE_S_DEFAULT


def runaway_streak_threshold() -> int:
    raw = os.environ.get("EPM_TICK_RUNAWAY_STREAK", "")
    try:
        val = int(raw)
    except ValueError:
        return RUNAWAY_STREAK_DEFAULT
    return val if val > 0 else RUNAWAY_STREAK_DEFAULT


# CONTENT INVARIANT (#1000; the #866/#906 refusal-prevention rule): verdict
# reason strings AND the snapshot payload stay free of task TEXT — status
# tokens, marker ages, child ids, disk bands only. Never embed the task
# title, body, or marker-note text here: the verdict line prints into an
# LLM tick turn, and on harmful-content tasks free text is trigger-dense
# enough to refusal-kill the session. Pinned by
# tests/test_tick_triage.py::test_snapshot_carries_no_task_text.
# The #1051 liveness reasons obey the same invariant — pids/ages/status
# tokens only; breadcrumb `log=` paths and `label=` slugs are read
# internally but never printed.
def compute_issue_verdict(
    status: str,
    prev_status: str | None,
    marker_age_s: float | None,
    over_cap: bool,
    *,
    stale_after_s: float,
) -> tuple[str, str]:
    """Pure verdict for /issue-tick. Returns ``(verdict, reason)``.

    Raises ValueError on a status outside the known enum sets — main()
    converts that to a non-zero exit (fail toward coverage)."""
    gate_now = status in ISSUE_GATE or (status == "plan_pending" and over_cap)
    if status in ISSUE_TERMINAL or (status == "plan_pending" and over_cap):
        if gate_now and prev_status != status:
            return (
                "GATE-TRANSITION",
                f"status={status} (prev={prev_status or 'unknown'}) — user gate just "
                "reached; push + teardown",
            )
        return ("TERMINAL", f"status={status} — teardown")
    if status not in ISSUE_PARK and status not in ISSUE_ACTIVE:
        raise ValueError(f"unknown status {status!r}")
    age_desc = "no markers" if marker_age_s is None else f"marker age {marker_age_s / 60:.0f}m"
    if marker_age_s is not None and marker_age_s <= stale_after_s:
        return ("HEALTHY", f"status={status}, {age_desc} — chain alive")
    kind = "in-skill chain" if status in ISSUE_PARK else "bg poll chain"
    return ("STALE-REDRIVE", f"status={status}, {age_desc} — {kind} likely dead")


def compute_campaign_verdict(
    status: str,
    prev_status: str | None,
    campaign_marker_age_s: float | None,
    *,
    landed_unreconciled: list[int],
    open_rows_all_in_flight: bool,
    stale_after_s: float,
) -> tuple[str, str]:
    """Pure verdict for /campaign-tick. Returns ``(verdict, reason)``."""
    if status in CAMPAIGN_TERMINAL:
        if status == "blocked" and prev_status != status:
            return (
                "GATE-TRANSITION",
                f"status=blocked (prev={prev_status or 'unknown'}) — push + teardown",
            )
        return ("TERMINAL", f"status={status} — teardown")
    if status in CAMPAIGN_STRANDED:
        return ("TERMINAL", f"status={status} — stranded cron (campaign not approved); teardown")
    if status not in CAMPAIGN_ACTIVE:
        raise ValueError(f"unknown campaign status {status!r}")
    if landed_unreconciled:
        ids = ", ".join(f"#{c}" for c in landed_unreconciled[:6])
        return ("STALE-REDRIVE", f"results landed unreconciled ({ids}) — run a decision round")
    age_desc = (
        "no campaign markers"
        if campaign_marker_age_s is None
        else f"campaign marker age {campaign_marker_age_s / 60:.0f}m"
    )
    if campaign_marker_age_s is not None and campaign_marker_age_s <= stale_after_s:
        return ("HEALTHY", f"status={status}, {age_desc} — decision loop alive")
    if open_rows_all_in_flight:
        return ("HEALTHY", f"status={status}, {age_desc} — all open arms in flight in children")
    return ("STALE-REDRIVE", f"status={status}, {age_desc} — decision round owed")


def campaign_open_rows(state: dict, children: list[dict]) -> tuple[list[int], bool]:
    """Derive ``(landed_unreconciled_child_ids, open_rows_all_in_flight)``
    from the campaign-state experiment rows + the live child statuses.

    ``open_rows_all_in_flight`` is True ONLY when at least one open
    (non-finished) row exists AND every open row maps to a child at a
    genuinely in-flight status. ZERO open rows — missing/garbled state
    file, or every row ingested/abandoned — returns False: such a campaign
    owes a decision round (propose the next arm or conclude), so a
    stale-marker tick must STALE-REDRIVE it, never idle as HEALTHY
    (review blocker, 2026-06-12)."""
    child_status = {row.get("id"): row.get("status") for row in children}
    rows = state.get("experiments")
    rows = rows if isinstance(rows, list) else []
    landed: list[int] = []
    open_rows = 0
    in_flight = 0
    for row in rows:
        if not isinstance(row, dict) or row.get("status") in CAMPAIGN_ROW_FINISHED:
            continue
        open_rows += 1
        child = row.get("child_task")
        if not isinstance(child, int):
            # A planned row with no child filed yet — a decision is owed
            # (not in flight).
            continue
        cstat = child_status.get(child)
        if cstat in CAMPAIGN_CHILD_LANDED:
            landed.append(child)
        elif cstat is not None and cstat not in CAMPAIGN_CHILD_DONEISH:
            in_flight += 1
    return landed, open_rows > 0 and in_flight == open_rows


# ── streak + main ───────────────────────────────────────────────────────────


def update_terminal_streak(issue: int, status: str, prev: dict, *, count_streak: bool) -> int:
    """Advance (or reset) the consecutive-teardown-tick counter and drop the
    runaway flag at the threshold. Returns the new streak value.

    A reset ALSO unlinks any existing runaway flag: a flag written during an
    earlier teardown-whiff episode must not survive a recovery (e.g.
    blocked -> running in the same live session) — a stale flag would
    force-stop the session on weeks-old corroboration the next time the
    task parks (review major, 2026-06-12)."""
    prev_streak = prev.get("terminal_streak")
    prev_streak = prev_streak if isinstance(prev_streak, int) and prev_streak >= 0 else 0
    if not count_streak:
        runaway_flag_path(issue).unlink(missing_ok=True)
        return 0
    streak = prev_streak + 1
    if streak >= runaway_streak_threshold():
        write_runaway_flag(issue, status, streak)
        print(
            f"tick_triage: #{issue} hit {streak} consecutive teardown-verdict ticks "
            f"(status={status}) — runaway flag written for the watcher "
            f"({runaway_flag_path(issue)})",
            file=sys.stderr,
        )
    return streak


def triage(issue: int, kind: str, now: float | None = None) -> tuple[str, str]:
    """Full triage for one tick. Returns ``(verdict, reason)``; raises on any
    state-read failure (the CLI converts that to a non-zero exit)."""
    now = now if now is not None else time.time()
    status, events = load_task_state(issue)
    prev = read_snapshot(issue)
    prev_status = prev.get("status") if isinstance(prev.get("status"), str) else None

    if kind == "campaign":
        marker_ts = latest_event_ts(events, prefix="epm:campaign")
        landed, all_in_flight = campaign_open_rows(load_campaign_state(issue), load_children(issue))
        verdict, reason = compute_campaign_verdict(
            status,
            prev_status,
            (now - marker_ts) if marker_ts is not None else None,
            landed_unreconciled=landed,
            open_rows_all_in_flight=all_in_flight,
            stale_after_s=stale_s(),
        )
    else:
        marker_ts = latest_event_ts(events)
        marker_age = (now - marker_ts) if marker_ts is not None else None
        verdict, reason = compute_issue_verdict(
            status,
            prev_status,
            marker_age,
            plan_pending_over_cap(events),
            stale_after_s=stale_s(),
        )
        if verdict == "STALE-REDRIVE":
            # Detached-phase liveness screen (issue #1051): fires ONLY on a
            # would-be STALE-REDRIVE, on BOTH stale branches — PARK and
            # ACTIVE (#931's incident status `followups_running` is in
            # ISSUE_PARK). Streak semantics unchanged (STALE-REDRIVE and
            # HEALTHY are both non-teardown verdicts); snapshot shape
            # untouched; campaign branch untouched.
            live = issue_liveness_reason(events, now, stale_s())
            if live is not None:
                age_desc = (
                    "no markers" if marker_age is None else f"marker age {marker_age / 60:.0f}m"
                )
                verdict = "HEALTHY"
                reason = f"status={status}, {age_desc} — {live}"

    # Runaway streak counts every TEARDOWN verdict, not just the terminal
    # STATUS sets — a teardown that whiffs forever at over-cap plan_pending
    # or at a stranded campaign cron deserves the same parachute (review
    # minor, 2026-06-12). The watcher's force-stop still acts only on the
    # DONE set; other flagged statuses get its loud alert-only arm.
    count_streak = verdict in ("TERMINAL", "GATE-TRANSITION")
    streak = update_terminal_streak(issue, status, prev, count_streak=count_streak)
    write_snapshot(issue, status, streak)
    return verdict, reason


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "issue", type=int, help="task number (the integer naming tasks/<status>/<N>/)"
    )
    parser.add_argument(
        "--kind",
        choices=("issue", "campaign"),
        default="issue",
        help="which tick skill is asking (default: issue)",
    )
    args = parser.parse_args(argv)
    try:
        verdict, reason = triage(args.issue, args.kind)
    except Exception as e:
        print(f"tick_triage: FAILED for #{args.issue}: {e}", file=sys.stderr)
        return 2
    print(f"{verdict} {reason}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
