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

A second screen (issue #1629, ``human_activity_reason``) runs when the
liveness screen finds nothing: a HUMAN (non-cron) user message in THIS
session's transcript within ``EPM_TICK_HUMAN_ACTIVE_S`` (default 2700 s)
converts the would-be STALE-REDRIVE to HEALTHY (reason prefix
``human-active``) — an interactive session is not a stalled autonomous
session, and re-driving the 44K-token /issue skill would hijack the
human's thread. The transcript is resolved happy-log-only via a /proc
ancestry walk (bash -> claude -> happy node wrapper -> the wrapper's
log names the transcript); cron-injected ``<command-message>``-wrapped
prompts, harness ``<task-notification>`` rows (Agent-tool spawn briefs /
completion notifications), skill-load meta rows, and tool results never
count as human.
Fail toward ticking: ANY resolution, parse, or classification failure
suppresses nothing (today's exact behavior). Teardown verdicts
(TERMINAL / GATE-TRANSITION) are never suppressed — the teardown is
what stops future interruptions. Kill switch:
``EPM_TICK_HUMAN_ACTIVE_PROBE=0``; debug telemetry:
``EPM_TICK_HUMAN_PROBE_DEBUG=1`` (stderr ``[human-probe]`` line —
paths/counts/ages only, never transcript text).

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
import subprocess
import sys
import tempfile
import time
from datetime import UTC, datetime
from pathlib import Path


def _ensure_scripts_dir_on_sys_path() -> None:
    """Insert THIS file's dir (scripts/) so a lazy ``import session_resolver`` resolves.

    In script mode scripts/ is already ``sys.path[0]``; in MODULE mode
    (``from scripts.tick_triage import ...`` — the scan
    ``test_every_lazy_scripts_local_import_is_bootstrap_guarded`` derives
    tick_triage as a module-mode consumer because
    ``autonomous_session_watch`` imports fingerprint helpers from it) only
    the repo root is on sys.path, so a bare lazy ``session_resolver``
    import would raise ``ModuleNotFoundError`` (#1296/#1304). Idempotent.
    """
    scripts_dir = str(Path(__file__).resolve().parent)
    if scripts_dir not in sys.path:
        sys.path.insert(0, scripts_dir)


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

# ── human-activity screen constants (issue #1629) ───────────────────────────
# Recency window default: one tick interval — "any human message since the
# last tick fired" is the natural semantics (env: EPM_TICK_HUMAN_ACTIVE_S).
HUMAN_ACTIVE_S_DEFAULT = 45 * 60
# Transcript tail-read bound: watcher parity (the #1104 wedge-probe widening).
TRANSCRIPT_TAIL_BYTES = 262_144
# /proc ancestry walk bound (measured chain depth to `claude` is 4 hops from a
# `uv run python` child; 15 covers deeper shell nesting with margin).
_ANCESTRY_MAX_DEPTH = 15
# Happy-log whole-read stat guard (~2x the largest measured live wrapper log,
# 66.6 MB on 2026-07-23; env: EPM_TICK_HUMAN_LOG_MAX_BYTES). Crossing it
# fails SAFE: skip -> no suppression -> today's behavior.
HUMAN_LOG_MAX_BYTES_DEFAULT = 128 * 2**20

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

# #2058: heartbeat notes that RESET the marker-age clock but produce NO durable
# work. A heartbeat's note prose begins with one of these tokens; the progress
# fingerprint's "non-heartbeat" filter excludes them so a session posting only
# heartbeats reads as fingerprint-unchanged despite fresh marker timestamps.
# Members:
#   - "tick heartbeat:" — the SKILL.md-quoted /issue-tick ACTIVE-status slow-
#     phase heartbeat (see .claude/skills/issue-tick/SKILL.md § STALE-REDRIVE).
#   - "[long-phase-heartbeat]" — the detached-VM-phase heartbeat the watcher
#     already filters (matches LONG_PHASE_HEARTBEAT_PREFIX below).
#   - "progress: none" — the canonical no-durable-work token the /issue-tick
#     heartbeat emits under the #2058 SKILL.md extension (Unit C).
# A heartbeat carrying `progress: <not-none>` (e.g. `progress: commit=abcd...`)
# IS durable evidence and is handled by compute_progress_fingerprint's
# progress-token short-circuit; the sentinel set governs only the plain
# no-durable-work heartbeat class.
_HEARTBEAT_NOTE_SENTINELS = frozenset(
    {
        "tick heartbeat:",
        "[long-phase-heartbeat]",
        "progress: none",
    }
)


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


# ── #2058 no-progress-respawn state ─────────────────────────────────────────


def no_progress_state_path(issue: int) -> Path:
    """State file for the #2058 no-progress-respawn arm — the tick writes the
    streak + fingerprint; the watcher pass reads its own state independently
    (per plan §4 "Fingerprint duplication note")."""
    return state_dir() / f"no-progress-{issue}.json"


def read_no_progress_state(issue: int) -> dict:
    """Prior tick's no-progress state (``{}`` on missing / garbled — fail
    toward a fresh episode; the pure predicate is fail-open when
    prev_fingerprint is None)."""
    path = no_progress_state_path(issue)
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return {}
    return data if isinstance(data, dict) else {}


def write_no_progress_state(issue: int, fingerprint: str | None, streak: int) -> None:
    """Atomic tmp+rename write of the #2058 per-issue no-progress state.

    ONLY the tick's streak + fingerprint components are the tick's to write;
    the watcher pass's `respawns_today` / `respawn_day` / `stop_pending_*`
    fields (plan §5) are the WATCHER's — preserve any it has already
    written by round-tripping unknown keys through the read."""
    prev = read_no_progress_state(issue)
    payload = dict(prev)  # preserve watcher-owned fields
    payload["issue"] = issue
    payload["fingerprint"] = fingerprint
    payload["streak"] = streak
    payload["ts"] = datetime.now(tz=UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    path = no_progress_state_path(issue)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), prefix=f".no-progress-{issue}-")
    try:
        with os.fdopen(fd, "w") as fh:
            json.dump(payload, fh)
        os.replace(tmp, path)
    except OSError:
        Path(tmp).unlink(missing_ok=True)
        raise


def compute_head_sha(issue: int) -> str | None:
    """Read ``origin/issue-<N>`` HEAD sha for the fingerprint's sha component.

    Fail-open — every failure mode returns None (never raises to the tick):
    * `git rev-parse` errors (ref not resolved, git binary missing);
    * `git fetch` errors (network, remote unreachable);
    * a bounded 5s timeout on either subprocess.

    NEVER `git rev-parse HEAD` unqualified against the SHARED repo root
    (per plan §1 methodology finding) — the shared root sits on `main`,
    so every fleet commit would advance the fingerprint and reset the
    no-progress streak. Fetch the issue's own branch and read that ref."""
    ref = f"origin/issue-{issue}"
    try:
        # Bounded fetch (fail-soft — a rate-limit/network blip must not
        # crash the tick). The fetch is best-effort; the rev-parse below
        # reads whatever the shared repo root has cached.
        subprocess.run(
            ["git", "fetch", "origin", f"issue-{issue}", "--quiet"],
            check=False,
            timeout=5.0,
            capture_output=True,
        )
        result = subprocess.run(
            ["git", "rev-parse", ref],
            check=False,
            timeout=5.0,
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            return None
        sha = result.stdout.strip()
        # Full 40-hex commit SHA — anything else is a git-side surprise.
        if len(sha) == 40 and all(c in "0123456789abcdef" for c in sha):
            return sha
        return None
    except (subprocess.TimeoutExpired, OSError):
        return None


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
    watcher-sentinel notes and deliberate session-stop records never count
    as freshness)."""
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
        # #1053: a deliberate session-stop record — incl. the Step-0
        # collision-exit / stale-wake-yield breadcrumb — is the driver's
        # death record: anti-liveness, never issue freshness (same predicate
        # as task_workflow.stage_dispatch_should_skip and the watcher's
        # _latest_progress_ts / _latest_nonwatcher_event_ts).
        if (isinstance(note, str) and note.lstrip().startswith("deliberate-stop ")) or row.get(
            "by"
        ) == "spawn_session-stop":
            continue
        ts = parse_event_ts(row.get("ts"))
        if ts is not None and (best is None or ts > best):
            best = ts
    return best


def latest_nonwatcher_nonheartbeat_ts(events: list[dict]) -> float | None:
    """Epoch ts of the newest DURABLE progress marker — excludes watcher-
    sentinel notes AND heartbeat-class notes (any note whose lstripped prose
    begins with a member of ``_HEARTBEAT_NOTE_SENTINELS``). #2058 uses this
    as the marker-ts component of the progress fingerprint: a session posting
    only heartbeats reads as ts-unchanged despite fresh marker timestamps.

    Progress-token escape: a heartbeat carrying an explicit
    ``progress: <not-none>`` line (e.g. ``progress: commit=abcd...``) is
    durable evidence — that read is handled by
    ``compute_progress_fingerprint``'s progress-token short-circuit, not by
    filtering here. This function's job is to filter out heartbeats whose
    prose does NOT carry a durable progress declaration."""
    best: float | None = None
    for row in events:
        if not isinstance(row, dict):
            continue
        note = row.get("note")
        if isinstance(note, str) and _WATCHER_NOTE_SENTINEL in note:
            continue
        # Deliberate-stop breadcrumbs are anti-liveness (same rule as
        # latest_event_ts above).
        if (isinstance(note, str) and note.lstrip().startswith("deliberate-stop ")) or row.get(
            "by"
        ) == "spawn_session-stop":
            continue
        # Heartbeat filter: lstripped note prose begins with a sentinel token.
        if isinstance(note, str):
            stripped = note.lstrip()
            if any(stripped.startswith(prefix) for prefix in _HEARTBEAT_NOTE_SENTINELS):
                continue
        ts = parse_event_ts(row.get("ts"))
        if ts is not None and (best is None or ts > best):
            best = ts
    return best


_PROGRESS_TOKEN_PREFIX = "progress: "


def compute_progress_fingerprint(
    events: list[dict],
    head_sha: str | None,
    status: str,
) -> str | None:
    """#2058 progress fingerprint: a stable string representing whether the
    issue has ADVANCED durably. Format: ``"<marker_ts>|<sha>|<status>"``.

    Components:
      * ``marker_ts`` — the newest non-watcher-non-heartbeat marker epoch,
        via ``latest_nonwatcher_nonheartbeat_ts``. A heartbeat carrying an
        explicit ``progress: <not-none>`` line short-circuits this: the
        token IS durable evidence, and its own ts + payload contribute to
        the fingerprint (so a run that emits `progress: commit=<sha12>`
        advances the fingerprint without needing a separate marker post).
      * ``head_sha`` — passed in by the caller (``main()`` computes it
        from ``git rev-parse origin/issue-<N>`` with the fail-open
        conventions in the plan; a None here means "sha unknown", and
        the marker-ts + status arms carry the fingerprint alone).
      * ``status`` — the parent status folder name (canonical state).

    Returns None when every component is None (nothing knowable).

    The freeze-not-advance discipline for a degraded-key (sha-null)
    transition lives in the CALLER (main()'s snapshot compare):
    compute_progress_fingerprint itself is pure and reports what it sees.
    """
    # Progress-token short-circuit: a durable `progress: <not-none>` line
    # inside ANY event's note prose contributes its ts + payload to the
    # fingerprint, so a heartbeat writer that CAN report durable state
    # (commit sha, new-markers count, status change) advances the
    # fingerprint immediately. Scan for the newest such row; if found,
    # it takes precedence over the plain non-heartbeat ts.
    progress_ts: float | None = None
    progress_payload: str | None = None
    for row in events:
        if not isinstance(row, dict):
            continue
        note = row.get("note")
        if not isinstance(note, str):
            continue
        if _WATCHER_NOTE_SENTINEL in note:
            continue
        # Look for a `progress: <value>` line (case-sensitive by design).
        for line in note.splitlines():
            stripped = line.lstrip()
            if not stripped.startswith(_PROGRESS_TOKEN_PREFIX):
                continue
            payload = stripped[len(_PROGRESS_TOKEN_PREFIX) :].strip()
            if not payload or payload == "none":
                # `progress: none` is the canonical no-durable-work
                # heartbeat token — it EXPLICITLY declares no advance and
                # never contributes to the fingerprint (its whole purpose
                # is the streak's clock).
                continue
            ts = parse_event_ts(row.get("ts"))
            if ts is None:
                continue
            if progress_ts is None or ts > progress_ts:
                progress_ts = ts
                progress_payload = payload
            break  # one progress token per row is enough

    if progress_ts is not None:
        return f"{progress_ts:.0f}|{progress_payload}|{head_sha or 'null'}|{status}"

    marker_ts = latest_nonwatcher_nonheartbeat_ts(events)
    if marker_ts is None and head_sha is None:
        # Nothing observable — return the status alone so a status change
        # still advances the fingerprint, but a first-tick episode with
        # only heartbeats returns a valid-but-thin key.
        return f"none|{head_sha or 'null'}|{status}"
    ts_component = f"{marker_ts:.0f}" if marker_ts is not None else "none"
    return f"{ts_component}|{head_sha or 'null'}|{status}"


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


# ── human-activity screen (issue #1629) ─────────────────────────────────────


def _human_probe_debug(msg: str) -> None:
    """Emit one ``[human-probe] <msg>`` stderr line iff
    ``EPM_TICK_HUMAN_PROBE_DEBUG=1`` (default off — ZERO output in
    production ticks). Callers pass paths/counts/ages ONLY, never
    transcript message text (content invariant #1000)."""
    if os.environ.get("EPM_TICK_HUMAN_PROBE_DEBUG", "").strip() == "1":
        print(f"[human-probe] {msg}", file=sys.stderr)


def human_active_probe_enabled() -> bool:
    """``EPM_TICK_HUMAN_ACTIVE_PROBE`` kill switch (default on; ``0``/
    ``false`` disables — restores pre-#1629 STALE-REDRIVE behavior
    fleet-wide; clone of ``liveness_probe_enabled``)."""
    raw = os.environ.get("EPM_TICK_HUMAN_ACTIVE_PROBE", "").strip().lower()
    return raw not in ("0", "false")


def human_active_s() -> float:
    """``EPM_TICK_HUMAN_ACTIVE_S`` (seconds) -> recency window; malformed or
    non-positive -> ``HUMAN_ACTIVE_S_DEFAULT`` (the ``stale_s()`` parse
    shape)."""
    raw = os.environ.get("EPM_TICK_HUMAN_ACTIVE_S", "")
    try:
        val = float(raw)
    except ValueError:
        return float(HUMAN_ACTIVE_S_DEFAULT)
    return val if val > 0 else float(HUMAN_ACTIVE_S_DEFAULT)


def human_log_max_bytes() -> float:
    """``EPM_TICK_HUMAN_LOG_MAX_BYTES`` (bytes) -> happy-log read ceiling;
    malformed or non-positive -> ``HUMAN_LOG_MAX_BYTES_DEFAULT`` (the
    ``stale_s()`` parse shape)."""
    raw = os.environ.get("EPM_TICK_HUMAN_LOG_MAX_BYTES", "")
    try:
        val = float(raw)
    except ValueError:
        return float(HUMAN_LOG_MAX_BYTES_DEFAULT)
    return val if val > 0 else float(HUMAN_LOG_MAX_BYTES_DEFAULT)


def _proc_ppid(pid: int) -> int | None:
    """Parent pid from ``_PROC_ROOT/<pid>/stat`` (parsed after the LAST
    ``)`` so a comm containing ``') '`` cannot shift fields — the
    ``proc_start_epoch`` parse shape; ppid = index 1 after comm).
    ``None`` on any read/parse failure."""
    try:
        stat = (_PROC_ROOT / str(pid) / "stat").read_text()
        after_comm = stat.rsplit(")", 1)[1].split()
        # after_comm[0] is field 3 (state); field 4 (ppid) is index 1.
        return int(after_comm[1])
    except (OSError, ValueError, IndexError):
        return None


def _own_node_wrapper_pid() -> int | None:
    """Walk UP the /proc ancestry from this process to the first ancestor
    whose comm is ``claude``; return its PARENT pid (the happy node
    wrapper). ``None`` when no claude ancestor within
    ``_ANCESTRY_MAX_DEPTH`` (bare-claude / non-happy runtimes -> the
    caller suppresses nothing). Uses ``session_resolver._read_proc_comm``
    (it strips the trailing newline — an inline unstripped
    ``/proc/<pid>/comm`` read comparing ``== "claude"`` would never
    match)."""
    _ensure_scripts_dir_on_sys_path()
    import session_resolver  # lazy: sys.path[0]=scripts/ per the SKILL contract

    pid = os.getpid()
    for _ in range(_ANCESTRY_MAX_DEPTH):
        ppid = _proc_ppid(pid)
        if ppid is None or ppid <= 1:
            return None
        if session_resolver._read_proc_comm(pid) == "claude":
            return ppid
        pid = ppid
    return None


def _own_session_transcript_path() -> str | None:
    """Resolve THIS session's Claude transcript path via the happy wrapper
    log — happy-log-only resolution (the #845 wedge-probe policy: a
    filesystem fallback could bind the WRONG session's transcript, and a
    wrong-session match is worse than a miss). ``None`` at every miss
    (fail toward ticking); the ``st_size`` guard runs BEFORE the
    whole-file log read so a pathological log never stalls the tick."""
    _ensure_scripts_dir_on_sys_path()
    import session_resolver  # lazy: an import failure is a caller-guarded miss

    node = _own_node_wrapper_pid()
    if node is None:
        _human_probe_debug("miss=no-claude-ancestor")
        return None
    log = session_resolver._find_happy_log_for_node(node)
    if log is None:
        _human_probe_debug("miss=no-happy-log")
        return None
    if log.stat().st_size > human_log_max_bytes():
        _human_probe_debug("miss=log-too-big")
        return None
    path = session_resolver.extract_transcript_from_happy_log(log.read_text(errors="replace"))
    if not path:
        _human_probe_debug("miss=no-transcript-path")
        return None
    if not os.path.isfile(path):
        _human_probe_debug("miss=transcript-missing")
        return None
    return path


def _transcript_tail_rows_1629(path: str, max_bytes: int = TRANSCRIPT_TAIL_BYTES) -> list[dict]:
    """Parsed dict rows from the LAST ``max_bytes`` of a transcript JSONL.

    SEEK-FROM-END is load-bearing (#1287: a head-capped read defeated its
    own motivating incident — recent rows live at EOF). After a mid-file
    seek the partial first line is dropped; each line's ``json.loads`` is
    guarded (a concurrent append can truncate the last line — one skipped
    row, re-read next tick)."""
    size = os.stat(path).st_size
    with open(path, "rb") as fh:
        if size > max_bytes:
            fh.seek(size - max_bytes)
        data = fh.read()
    lines = data.split(b"\n")
    if size > max_bytes:
        lines = lines[1:]  # drop the partial first line after the seek
    rows: list[dict] = []
    for line in lines:
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except (json.JSONDecodeError, UnicodeDecodeError, ValueError):
            continue
        if isinstance(row, dict):
            rows.append(row)
    return rows


def is_human_transcript_row(row: object) -> bool:
    """True iff a transcript JSONL row is direct HUMAN input (#1629).

    Conservative toward ticking: any ambiguous / automation-shaped row
    reads False. A human-typed slash command reads False too (wrapped
    identically to a cron-injected one) — acceptable: mistaking a human
    for automation only costs one tick turn, never a stranded task.
    Harness-injected ``<task-notification>`` rows (Agent-tool spawn briefs
    / completion notifications — plain-string user rows, the dominant
    string-row class in autonomous transcripts: 149/149 measured across
    two transcripts, r2 Must-Fix) classify automation via the prefix
    exclusion below.
    """
    if not isinstance(row, dict) or row.get("type") != "user" or row.get("isMeta"):
        return False
    msg = row.get("message")
    if not isinstance(msg, dict) or msg.get("role") != "user":
        return False
    content = msg.get("content")
    if isinstance(content, str):
        s = content.lstrip()
        if not s:
            return False
        if "<command-name>" in content or s.startswith(
            ("<command-message>", "<local-command", "<task-notification")
        ):
            return False
        return True
    if isinstance(content, list):  # the ONE list-shape human signal: the interrupt row
        for block in content:
            if (
                isinstance(block, dict)
                and block.get("type") == "text"
                and isinstance(block.get("text"), str)
                and block["text"].lstrip().startswith("[Request interrupted by user")
            ):
                return True
    return False


def api_error_after_marker_reason(now: float, marker_ts: float | None) -> str | None:
    """Return a content-invariant reason suffix when THIS session's transcript
    tail carries an assistant row with ``isApiErrorMessage: true`` whose ts is
    NEWER than the latest events.jsonl marker ts (#1687), else ``None``.

    Fully exception-guarded (fail toward today's verdict — a spurious
    STALE-REDRIVE is more expensive than a missed one on this predicate; the
    marker-age STALE-REDRIVE remains the backstop). Content invariant #1000:
    the returned reason names ages/counts ONLY, NEVER the row's message text
    (refusal bodies are trigger-dense — the #866/#1073/#1098 containment).

    Reuses the same transcript reader ``human_activity_reason`` uses
    (:func:`_transcript_tail_rows_1629` at 256 KB tail bound); on a HEALTHY
    tick this predicate is the ONLY reader of that file, so the healthy tick
    stays ONE Bash call by construction (the `human_activity_reason` call
    lives inside the ``verdict == STALE-REDRIVE`` branch, which is not entered
    on the HEALTHY path).

    Kill switch: ``EPM_TICK_API_ERROR_PROBE=0`` (or ``false``/``off``) reverts
    to pre-#1695 HEALTHY behavior. Default enabled — parallel to
    :func:`human_active_probe_enabled`.
    """
    try:
        raw = os.environ.get("EPM_TICK_API_ERROR_PROBE", "").strip().lower()
        if raw in ("0", "false", "off"):
            return None
        path = _own_session_transcript_path()
        if path is None:
            return None
        newest_api_error_ts: float | None = None
        for row in _transcript_tail_rows_1629(path):
            if not isinstance(row, dict) or row.get("type") != "assistant":
                continue
            if row.get("isApiErrorMessage") is not True:
                continue
            ts = parse_event_ts(row.get("timestamp"))
            if ts is None:
                continue
            if newest_api_error_ts is None or ts > newest_api_error_ts:
                newest_api_error_ts = ts
        if newest_api_error_ts is None:
            return None
        # STRICT `>`: a marker posted at the same ts as the api-error row
        # (or newer) wins (fail toward today's verdict; row ts precision is
        # ~1s so a tie is ambiguous — the marker-post likely followed).
        if marker_ts is not None and newest_api_error_ts <= marker_ts:
            return None
        age = now - newest_api_error_ts
        # Future-dated row (clock skew, age < 0): skip, don't latch (mirrors
        # `human_activity_reason`'s future-row skip).
        if age < 0:
            return None
        return f"api-error-after-marker (api-error {age / 60:.0f}m ago, newer than marker)"
    except Exception as exc:
        # Debug-only error rung (parallel to `human_activity_reason`): TYPE
        # only (content invariant #1000), itself guarded so the helper can
        # never raise into triage().
        try:
            _human_probe_debug(f"api-error-probe error={type(exc).__name__}")
        except Exception:
            return None
        return None


def human_activity_reason(now: float) -> str | None:
    """Return a content-invariant reason suffix when a HUMAN (non-cron)
    user message appears in THIS session's transcript within the recency
    window (issue #1629), else ``None``. Entirely exception-guarded: ANY
    failure -> ``None`` (fail toward ticking — the #1629 hard
    constraint; the helper can never raise into ``triage()``).

    Aggregation is any-row-in-window (per-row ``0 <= age < window``,
    smallest in-window age reported): a single future-dated row (clock
    skew, ``age < 0``) can never mask a valid fresh row — future rows
    are skipped, never latched as "the newest"."""
    try:
        if not human_active_probe_enabled():
            _human_probe_debug("miss=disabled")
            return None
        path = _own_session_transcript_path()
        if path is None:
            return None
        window = human_active_s()
        n_rows = 0
        n_human = 0
        best_age: float | None = None  # smallest IN-WINDOW age among human rows
        for row in _transcript_tail_rows_1629(path):
            n_rows += 1
            if not is_human_transcript_row(row):
                continue
            n_human += 1
            ts = parse_event_ts(row.get("timestamp"))
            if ts is None:
                continue
            age = now - ts
            if 0 <= age < window and (best_age is None or age < best_age):
                best_age = age
        if n_human == 0:
            _human_probe_debug("miss=no-human-rows")
            return None
        age_str = "none" if best_age is None else f"{best_age:.1f}"
        verdict_str = "suppress" if best_age is not None else "no-suppress"
        _human_probe_debug(
            f"transcript={path} rows={n_rows} human={n_human} "
            f"newest_age_s={age_str} verdict={verdict_str}"
        )
        if best_age is None:
            return None
        return f"human-active (last human msg {best_age / 60:.0f}m ago < {window / 60:.0f}m)"
    except Exception as exc:
        # Debug-only error rung (r2 Minor): exception TYPE only (content
        # invariant #1000), itself guarded so the helper still can never
        # raise into triage() (the #1629 hard constraint).
        try:
            _human_probe_debug(f"error={type(exc).__name__}")
        except Exception:
            return None
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


def no_progress_threshold() -> int:
    """#2058 no-progress-respawn threshold — N consecutive ticks with an
    unchanged fingerprint before the pure predicate emits
    NO-PROGRESS-RESPAWN. Default 3 (per plan §11 Decision Rationale).
    Configurable via ``EPM_NO_PROGRESS_RESPAWN_TICKS``; malformed / <2 →
    default 3 (NEVER a kill switch — the watcher-pass env var
    ``EPM_DISABLE_NO_PROGRESS_RESPAWN`` is the kill)."""
    raw = os.environ.get("EPM_NO_PROGRESS_RESPAWN_TICKS", "")
    try:
        val = int(raw)
    except ValueError:
        return 3
    return val if val >= 2 else 3


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
    progress_fingerprint: str | None = None,
    prev_fingerprint: str | None = None,
    no_progress_streak: int = 0,
    no_progress_threshold: int = 3,
) -> tuple[str, str, int]:
    """Pure verdict for /issue-tick. Returns ``(verdict, reason, streak)``.

    The optional ``progress_fingerprint`` / ``prev_fingerprint`` /
    ``no_progress_streak`` / ``no_progress_threshold`` kwargs implement the
    #2058 no-progress-respawn arm (session alive, chain heartbeating but no
    durable advancement). Legacy callers not wiring them get streak=0 and
    the pre-#2058 verdict behavior preserved by construction.

    Raises ValueError on a status outside the known enum sets — main()
    converts that to a non-zero exit (fail toward coverage)."""
    gate_now = status in ISSUE_GATE or (status == "plan_pending" and over_cap)
    if status in ISSUE_TERMINAL or (status == "plan_pending" and over_cap):
        if gate_now and prev_status != status:
            return (
                "GATE-TRANSITION",
                f"status={status} (prev={prev_status or 'unknown'}) — user gate just "
                "reached; push + teardown",
                0,
            )
        return ("TERMINAL", f"status={status} — teardown", 0)
    if status not in ISSUE_PARK and status not in ISSUE_ACTIVE:
        raise ValueError(f"unknown status {status!r}")
    age_desc = "no markers" if marker_age_s is None else f"marker age {marker_age_s / 60:.0f}m"
    if marker_age_s is not None and marker_age_s <= stale_after_s:
        # #2058 no-progress arm — reachable ONLY when marker age is fresh.
        # A fingerprint that is None on either side is fail-open (first tick
        # of an episode, or fingerprint uncomputable): return HEALTHY with
        # streak=0. A fingerprint that ADVANCED resets the streak. An
        # unchanged fingerprint accumulates the streak; on threshold reach
        # emit NO-PROGRESS-RESPAWN (the watcher pass owns the ACT).
        if progress_fingerprint is None or prev_fingerprint is None:
            return ("HEALTHY", f"status={status}, {age_desc} — chain alive", 0)
        if progress_fingerprint == prev_fingerprint:
            streak = no_progress_streak + 1
            if streak >= no_progress_threshold:
                return (
                    "NO-PROGRESS-RESPAWN",
                    f"status={status}, fingerprint unchanged across {streak} "
                    "ticks — session likely context-exhausted",
                    streak,
                )
            return (
                "HEALTHY",
                f"status={status}, {age_desc} — chain alive (no-progress streak {streak})",
                streak,
            )
        return ("HEALTHY", f"status={status}, {age_desc} — chain alive", 0)
    kind = "in-skill chain" if status in ISSUE_PARK else "bg poll chain"
    return ("STALE-REDRIVE", f"status={status}, {age_desc} — {kind} likely dead", 0)


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
        # #2058 no-progress-respawn wiring — compute fingerprint + thread
        # the prior streak/fp through the pure predicate; persist the
        # updated pair below.
        no_progress_prev = read_no_progress_state(issue)
        prev_fingerprint = no_progress_prev.get("fingerprint")
        prev_no_progress_streak = no_progress_prev.get("streak")
        prev_no_progress_streak = (
            prev_no_progress_streak
            if isinstance(prev_no_progress_streak, int) and prev_no_progress_streak >= 0
            else 0
        )
        prev_sha = None
        if isinstance(prev_fingerprint, str):
            # "progress: X | payload | sha | status" OR "ts | sha | status" —
            # the sha is either the 2nd (progress-token) or 1st non-ts pipe
            # segment; the fingerprint helper is authoritative on shape.
            # For the degraded-key freeze we only need to know whether the
            # PRIOR fingerprint carried a real 40-hex sha somewhere.
            for part in prev_fingerprint.split("|"):
                p = part.strip()
                if len(p) == 40 and all(c in "0123456789abcdef" for c in p):
                    prev_sha = p
                    break
        head_sha = compute_head_sha(issue)
        # Degraded-key freeze (plan §1): when THIS tick's sha is None but
        # the previous tick carried a real sha, freeze the streak (don't
        # advance, don't reset) — same fail-toward-freeze posture as
        # daemon-unreachable / unresolvable-transcript. Achieved by
        # passing the PRIOR fingerprint through as this tick's — the
        # equality check reads unchanged and streak advances by 1, but we
        # bump BOTH sides to the prior value so the pure predicate sees
        # NO change AND we do not persist an advance.
        if head_sha is None and prev_sha is not None:
            fingerprint = prev_fingerprint  # freeze — same value, same streak semantics
            freeze_active = True
        else:
            fingerprint = compute_progress_fingerprint(events, head_sha, status)
            freeze_active = False
        verdict, reason, no_progress_streak = compute_issue_verdict(
            status,
            prev_status,
            marker_age,
            plan_pending_over_cap(events),
            stale_after_s=stale_s(),
            progress_fingerprint=fingerprint,
            prev_fingerprint=prev_fingerprint if isinstance(prev_fingerprint, str) else None,
            no_progress_streak=prev_no_progress_streak,
            no_progress_threshold=no_progress_threshold(),
        )
        # Persist the updated streak + fingerprint. On a degraded-key
        # freeze, the streak advance from the equal-fingerprint path is
        # ROLLED BACK to the prior value so the freeze is fingerprint-
        # and-streak-preserving as designed.
        if freeze_active:
            no_progress_streak = prev_no_progress_streak
        write_no_progress_state(issue, fingerprint, no_progress_streak)
        if verdict == "STALE-REDRIVE":
            # Detached-phase liveness screen (issue #1051): fires ONLY on a
            # would-be STALE-REDRIVE, on BOTH stale branches — PARK and
            # ACTIVE (#931's incident status `followups_running` is in
            # ISSUE_PARK). Streak semantics unchanged (STALE-REDRIVE and
            # HEALTHY are both non-teardown verdicts); snapshot shape
            # untouched; campaign branch untouched. The human-activity
            # screen (issue #1629) runs SECOND, only when the liveness
            # screen found nothing — same PARK+ACTIVE scope, same
            # fail-toward-ticking posture.
            live = issue_liveness_reason(events, now, stale_s())
            if live is None:
                live = human_activity_reason(now)
            if live is not None:
                age_desc = (
                    "no markers" if marker_age is None else f"marker age {marker_age / 60:.0f}m"
                )
                verdict = "HEALTHY"
                reason = f"status={status}, {age_desc} — {live}"
        elif verdict == "HEALTHY":
            # #1687 api-error-after-marker predicate: an assistant row with
            # `isApiErrorMessage: true` newer than the latest marker means a
            # refusal / 529 killed the driving turn AFTER the last real
            # event log post — marker-age freshness masks the death.
            # HEALTHY -> STALE-REDRIVE, so the tick loads the full /issue
            # skill and re-drives. Placement on the HEALTHY branch keeps
            # the STALE-REDRIVE cascade (liveness / human-active) untouched;
            # since the new predicate reads the SAME transcript file as
            # `human_activity_reason` and the HEALTHY path never enters the
            # STALE branch, a HEALTHY tick still costs ONE Bash call.
            # Teardown verdicts (TERMINAL / GATE-TRANSITION) are UNAFFECTED
            # — a refusal after gate-park is still a teardown (the same
            # posture as the #1629 screen). Kill switch:
            # ``EPM_TICK_API_ERROR_PROBE=0``.
            api_err = api_error_after_marker_reason(now, marker_ts)
            if api_err is not None:
                verdict = "STALE-REDRIVE"
                age_desc = (
                    "no markers" if marker_age is None else f"marker age {marker_age / 60:.0f}m"
                )
                reason = f"status={status}, {age_desc} — {api_err}"

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
