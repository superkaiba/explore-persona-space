"""Spawn / list / stop Happy Coder sessions via the local daemon HTTP RPC.

Happy ships a localhost HTTP control server at ``127.0.0.1:<port>`` (port lives
in ``~/.happy/daemon.state.json``). This is the canonical entry point for
programmatic session spawning — sessions created via ``happy claude`` directly
or via this RPC are equivalently visible in the user's mobile Happy app.

Routes the daemon exposes (POST only):

    /spawn-session   {"directory": <abs path>, "sessionId"?: <str>, "agent"?: <str>,
                      "environmentVariables"?: {...}, "claudeArgs"?: [<str>, ...]}
    /list            {}
        -> {"children": [{"happySessionId": ..., "pid": ..., "startedBy": ...}, ...]}
    /stop-session    {"sessionId": <happySessionId>}

The daemon binds to localhost only and trusts UID-local callers (no auth).

This script is the project-level wrapper for that API. The dedicated PM
session uses ``spawn-pm``; per-issue sessions use ``spawn-issue --issue <N>``.
The session's working directory determines what the user sees as the
session label in Happy — we surface that here.

All three spawn commands (``spawn-pm`` / ``spawn-issue`` / ``spawn-campaign``)
open sessions ONLY in the canonical primary checkout or the target issue's own
worktree: ``PROJECT_ROOT`` is git-common-dir-resolved via
``task_workflow.primary_checkout_root()`` — never ``Path(__file__)``, which
would resolve a worktree COPY of this script to that sibling worktree and
spawn unrelated sessions into it (#844) — and ``_assert_spawn_cwd`` refuses
any other cwd at spawn time, before the daemon POST.
"""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import re
import signal
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path
from typing import Any, NamedTuple

# Make the package importable without `uv run` plumbing (same bootstrap as
# scripts/task.py). Sibling-import semantics on purpose: a worktree copy of
# this script imports its sibling src/ tree; resolution below is STILL
# canonical because task_workflow resolves via the git COMMON dir, which a
# linked worktree shares with the primary checkout.
_SRC = Path(__file__).resolve().parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from explore_persona_space.task_workflow import primary_checkout_root  # noqa: E402

HAPPY_HOME = Path.home() / ".happy"
DAEMON_STATE = HAPPY_HOME / "daemon.state.json"
SESSIONS_JSON = HAPPY_HOME / "sessions.json"
# #844: git-resolved canonical primary checkout; fails loud at import if git /
# the repo layout is broken — NEVER `Path(__file__).resolve().parent.parent`
# (a worktree copy of this script would resolve the worktree, and every
# spawned session would inherit that unrelated sibling worktree as cwd).
# Bare-name `from spawn_session import PROJECT_ROOT` consumers fixed
# transitively: scripts/autonomous_session_watch.py, scripts/file_infra_task.py.
# scripts/session_resolver.py (~line 66) imports spawn_session for its
# post/registry helpers (not PROJECT_ROOT) and newly incurs the import-time
# git resolution + fail-loud — harmless in its call contexts.
PROJECT_ROOT = primary_checkout_root()
WORKTREE_DIR = PROJECT_ROOT / ".claude" / "worktrees"

# Registry of autonomous (`--auto`) issue sessions, so the crash-recovery
# watcher (scripts/autonomous_session_watch.py) can detect a dead session and
# re-spawn it. One file per issue: ~/.eps-autonomous/issue-<N>.json.
#
# Manual (`spawn-issue` WITHOUT `--auto`) sessions ALSO register a sibling
# entry here at ~/.eps-autonomous/manual-issue-<N>.json so `cmd_list` can map
# session id -> issue number. The watcher's respawn pass globs
# `issue-*.json` and DELIBERATELY does NOT match `manual-issue-*.json` — manual
# sessions must NEVER be auto-re-spawned (the user opens them manually and
# decides when to drive them). Keeping both files in the same dir keeps the
# layout tidy without changing the watcher contract.
#
# `register-current` re-writes either kind for an ALREADY-LIVE session — used
# when a parked/terminal task is revived (same-issue follow-up loop) after the
# watcher GC'd its entry at the terminal transition (#472, 2026-06-10).
AUTONOMOUS_REGISTRY_DIR = Path.home() / ".eps-autonomous"

# ─── per-issue dispatch lease (#843 M1) ──────────────────────────────────────
#
# Atomic create-or-fail claim taken by `spawn-issue --auto` BEFORE the daemon
# POST, so two concurrent dispatchers (file_infra_task self-dispatch vs the
# watcher sweep/drain, the program-orchestrator daemon, ad-hoc PM dispatch)
# can never both spawn a session for the same issue inside the
# decision->registration window. (Watcher-vs-watcher overlap is already
# impossible — its main() holds a whole-run non-blocking flock on
# ~/.eps-autonomous/watch.lock — so the races this closes are strictly
# cross-dispatcher.) TTL == the watcher's RESPAWN_SPAWN_GRACE_S (15 min):
# inside that window the crash-recovery pass already refuses to respawn a
# fresh registration, so lease-blocking can never postpone a recovery the
# watcher would have run (pinned by
# tests/test_autonomous_session_watch.py::test_lease_ttl_default_equals_respawn_spawn_grace).
DISPATCH_LEASE_TTL_S = 15 * 60

# Substring stamped into the best-effort `epm:progress` marker posted when a
# duplicate `--auto` dispatch reached registration and was auto-stopped
# (#843 M2). The watcher imports it into _WATCHER_NOTE_SENTINELS so the
# marker never counts as "real progress" for the staleness clocks.
DUPLICATE_DISPATCH_NOTE_SENTINEL = "[spawn-session:duplicate-dispatch-suppressed]"

# stdout sentinels a suppressed rc-0 `spawn-issue --auto` no-op prints. Every
# automated caller distinguishes them from a real spawn BEFORE any success
# bookkeeping (#843 M1b) via :func:`spawn_output_suppressed`.
DISPATCH_LEASE_HELD_SENTINEL = "DISPATCH-LEASE HELD"
REGISTRATION_COLLISION_SENTINEL = "REGISTRATION-COLLISION"
TAKEOVER_HELD_SENTINEL = "TAKEOVER-SENTINEL HELD"

# ─── deliberate-takeover sentinel (#866/#903) ────────────────────────────────
#
# A session deliberately taking over a stalled autonomous session renames its
# registration `issue-<N>.json` -> `issue-<N>.json.paused-takeover-<suffix>`
# (free-form suffix; `manual-issue-` same shape). While the sentinel is FRESH
# (file mtime within EPS_TAKEOVER_TTL_H, default 6h; `touch` renews) the
# orphan-respawn pass skips the issue and `spawn-issue --auto` suppresses —
# a stale/missing sentinel is ignored everywhere (FAIL OPEN: crash recovery
# resumes at the TTL). Full convention doc:
# `.claude/rules/background-automation.md` § Deliberate session takeover.
TAKEOVER_SENTINEL_TTL_H_DEFAULT = 6.0  # hours; #903 (goal-specified ~6h)

# Tolerance for ordinary clock jitter before a FUTURE-dated sentinel mtime is
# treated as NOT fresh. A genuinely future-dated mtime (clock skew / a
# `touch -d` typo) would otherwise be PERMANENTLY fresh — an indefinite
# crash-recovery suppression that inverts the fail-open guarantee (#903
# round-1 critique). Failing open here is visible: the per-tick skip line
# never fires for the ignored sentinel, so "respawn resumed despite sentinel"
# is the observable anomaly.
FUTURE_MTIME_SLACK_S = 300.0


def spawn_output_suppressed(stdout: str | None) -> str | None:
    """Which duplicate-suppression sentinel (if any) a rc-0 ``spawn-issue
    --auto`` subprocess printed: :data:`DISPATCH_LEASE_HELD_SENTINEL` /
    :data:`REGISTRATION_COLLISION_SENTINEL` / :data:`TAKEOVER_HELD_SENTINEL`,
    else ``None`` (a real spawn).

    Shared by the watcher's dispatch/respawn callers + the file-time filer so
    a suppressed no-op is never booked as a successful spawn — no dispatch
    marker, no attempt/backoff, no respawn bookkeeping (#843 M1b)."""
    if not stdout:
        return None
    for sentinel in (
        DISPATCH_LEASE_HELD_SENTINEL,
        REGISTRATION_COLLISION_SENTINEL,
        TAKEOVER_HELD_SENTINEL,
    ):
        if sentinel in stdout:
            return sentinel
    return None


def _takeover_ttl_s() -> float:
    """Takeover-sentinel TTL in seconds (env ``EPS_TAKEOVER_TTL_H``, hours;
    missing or malformed value falls back to
    :data:`TAKEOVER_SENTINEL_TTL_H_DEFAULT` — a typo'd var must not disable
    crash recovery, mirroring the watcher's ``_orphan_staleness_s``)."""
    raw = os.environ.get("EPS_TAKEOVER_TTL_H")
    if not raw:
        return TAKEOVER_SENTINEL_TTL_H_DEFAULT * 3600.0
    try:
        return float(raw) * 3600.0
    except ValueError:
        return TAKEOVER_SENTINEL_TTL_H_DEFAULT * 3600.0


def takeover_sentinel_fresh(
    issue: int, now: float | None = None, registry_dir: Path | None = None
) -> Path | None:
    """Newest FRESH deliberate-takeover sentinel for issue ``issue``, else None.

    Convention (#866/#903): a session deliberately taking over a stalled
    autonomous session renames ``~/.eps-autonomous/issue-<N>.json`` ->
    ``issue-<N>.json.paused-takeover-<suffix>`` (``manual-issue-`` same
    shape). Fresh = file mtime within ``EPS_TAKEOVER_TTL_H`` (default 6h);
    ``touch`` the sentinel to renew. FAIL OPEN: stale / missing / unreadable /
    future-dated (beyond :data:`FUTURE_MTIME_SLACK_S`) -> ``None`` (today's
    respawn behavior preserved). ``registry_dir``/``now`` are injectable for
    tests (the :func:`resolve_session_for_issue` pattern). The exact-``N``-
    then-``.json`` boundary makes the glob prefix-collision-safe
    (``issue-1.json.paused-takeover-*`` cannot match issue 14)."""
    reg = registry_dir if registry_dir is not None else AUTONOMOUS_REGISTRY_DIR
    now = time.time() if now is None else now
    ttl = _takeover_ttl_s()
    best: Path | None = None
    best_mtime = float("-inf")
    if not reg.is_dir():
        return None
    for pattern in (
        f"issue-{issue}.json.paused-takeover-*",
        f"manual-issue-{issue}.json.paused-takeover-*",
    ):
        for p in reg.glob(pattern):
            try:
                mt = p.stat().st_mtime
            except OSError:
                continue  # unreadable -> ignored (fail open)
            if mt > now + FUTURE_MTIME_SLACK_S:
                # Future-dated mtime would be PERMANENTLY fresh — an
                # indefinite crash-recovery suppression that inverts the
                # fail-open guarantee. Treat as NOT fresh (fail open).
                continue
            if now - mt < ttl and mt > best_mtime:
                best, best_mtime = p, mt
    return best


def _dispatch_lease_ttl_s() -> float:
    """Lease TTL in seconds (env ``EPM_DISPATCH_LEASE_TTL_S``; missing or
    malformed value falls back to :data:`DISPATCH_LEASE_TTL_S`)."""
    raw = os.environ.get("EPM_DISPATCH_LEASE_TTL_S")
    if not raw:
        return float(DISPATCH_LEASE_TTL_S)
    try:
        return float(raw)
    except ValueError:
        return float(DISPATCH_LEASE_TTL_S)


def dispatch_lease_path(issue: int) -> Path:
    """Path of the per-issue dispatch-lease file (#843 M1)."""
    return AUTONOMOUS_REGISTRY_DIR / f"dispatch-lease-{issue}.json"


def _dispatch_lease_lock_path(issue: int) -> Path:
    """Path of the PERMANENT per-issue flock sidecar that serializes the
    stale-lease takeover slow path. Never unlinked by acquire/release — an
    unlink-and-recreate of a flock target under a waiter is the classic
    flock-on-deleted-file hole; the file is tiny and recreated on demand."""
    return AUTONOMOUS_REGISTRY_DIR / f"dispatch-lease-{issue}.lock"


def read_dispatch_lease(issue: int) -> dict | None:
    """Parsed lease dict; ``{}`` for garbled/unreadable content; ``None`` for
    no lease file."""
    try:
        raw = dispatch_lease_path(issue).read_text()
    except FileNotFoundError:
        return None
    except OSError:
        return {}
    try:
        entry = json.loads(raw)
    except ValueError:
        return {}
    return entry if isinstance(entry, dict) else {}


def dispatch_lease_fresh(issue: int, now: float | None = None) -> dict | None:
    """The lease entry iff the lease file exists AND is fresh; else ``None``.

    Fresh = ``acquired_at`` within :func:`_dispatch_lease_ttl_s`. A garbled
    lease (unparseable JSON / non-numeric ``acquired_at``) falls back to the
    file mtime — failing toward FRESH, i.e. toward NOT dispatching; the TTL
    bounds any wedge. A missing file returns ``None`` (no lease held)."""
    now = time.time() if now is None else now
    entry = read_dispatch_lease(issue)
    if entry is None:
        return None
    ttl = _dispatch_lease_ttl_s()
    acquired = entry.get("acquired_at")
    if isinstance(acquired, int | float) and not isinstance(acquired, bool):
        return entry if now - acquired < ttl else None
    # Garbled content / missing acquired_at -> file mtime (fail toward fresh).
    try:
        mtime = dispatch_lease_path(issue).stat().st_mtime
    except OSError:
        # File vanished / unreadable between read and stat: treat as fresh
        # this call (fail toward not dispatching); the next tick re-reads.
        return entry
    return entry if now - mtime < ttl else None


def dispatch_lease_desc(entry: dict | None, now: float | None = None) -> str:
    """Human-readable ``holder=..., pid=..., age=...s`` fragment for the
    loud skip/loser log lines. Tolerates a garbled/empty entry."""
    entry = entry or {}
    now = time.time() if now is None else now
    acquired = entry.get("acquired_at")
    if isinstance(acquired, int | float) and not isinstance(acquired, bool):
        age = f"{now - acquired:.0f}s"
    else:
        age = "?"
    return f"holder={entry.get('holder', '?')}, pid={entry.get('pid', '?')}, age={age}"


def acquire_dispatch_lease(issue: int, holder: str, now: float | None = None) -> dict | None:
    """Atomically claim the per-issue dispatch slot (#843 M1).

    Returns the written lease entry on success; ``None`` when a fresh lease is
    already held — or on ANY unexpected OSError (fail toward not dispatching).

    FAST PATH (lock-free): no lease file -> single-winner
    ``os.open(O_CREAT|O_EXCL)`` (atomic on the local ext filesystem). SLOW
    PATH (a lease file exists): the freshness check + stale takeover are
    serialized under the PERMANENT per-issue flock sidecar
    (:func:`_dispatch_lease_lock_path`) — a check-freshness->replace protocol
    without the lock admits a TOCTOU double-winner. Every CREATE of the lease
    file goes through O_EXCL and every unlink of a live lease happens under
    the sidecar flock (or terminal-task GC), so no taker can destroy another
    taker's freshly created lease: two flock holders are impossible, and a
    fast-path creator racing a takeover just makes the takeover's inner
    O_EXCL fail -> the taker returns ``None`` (single winner preserved)."""
    now = time.time() if now is None else now
    entry: dict[str, Any] = {
        "issue": issue,
        "holder": holder,
        "pid": os.getpid(),
        "token": uuid.uuid4().hex,
        "acquired_at": now,
    }
    path = dispatch_lease_path(issue)
    try:
        AUTONOMOUS_REGISTRY_DIR.mkdir(parents=True, exist_ok=True)
        # FAST PATH (lock-free): no lease file -> atomic single-winner create.
        fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    except FileExistsError:
        # SLOW PATH: a lease file exists. Serialize under the sidecar flock.
        try:
            lock_fd = os.open(_dispatch_lease_lock_path(issue), os.O_CREAT | os.O_WRONLY, 0o644)
        except OSError:
            return None  # fail toward not dispatching
        try:
            try:
                fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError:
                return None  # another taker mid-takeover: skip this tick
            if dispatch_lease_fresh(issue, now) is not None:
                return None  # loser: a fresh lease is held
            path.unlink(missing_ok=True)  # remove the stale lease (under the lock)
            try:
                fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
            except OSError:
                return None  # a fast-path creator won post-unlink: loser
        finally:
            os.close(lock_fd)  # releases the flock; the sidecar file persists
        # NOTE: the content write below happens after the flock is released —
        # safe: until written, the file is empty (= garbled) with a fresh
        # mtime, which dispatch_lease_fresh fails CLOSED on (treated fresh).
    except OSError:
        return None  # fail toward not dispatching
    with os.fdopen(fd, "w") as f:
        f.write(json.dumps(entry, indent=2))
    return entry


def release_dispatch_lease(issue: int, token: str) -> None:
    """Best-effort token-verified unlink of the per-issue dispatch lease,
    taken UNDER the permanent flock sidecar so a late release can never remove
    a successor's lease.

    Called on FAILURE exit paths only (daemon POST failed / patch-verify died
    / plain-OSError registration failure) — the SUCCESS path deliberately
    LEAVES the lease in place (TTL expiry owns it; releasing at registration
    would reopen a window if ordering ever regressed), and the
    REGISTRATION-COLLISION branch deliberately HOLDS it (the first session is
    live and driving; holding suppresses spawn-then-collision-stop churn for
    the rest of the TTL — see :func:`cmd_spawn_issue`)."""
    try:
        lock_fd = os.open(_dispatch_lease_lock_path(issue), os.O_CREAT | os.O_WRONLY, 0o644)
    except OSError:
        return
    try:
        try:
            fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            return  # a takeover is in flight; leave the lease to the TTL
        entry = read_dispatch_lease(issue)
        if isinstance(entry, dict) and entry.get("token") == token:
            dispatch_lease_path(issue).unlink(missing_ok=True)
    finally:
        os.close(lock_fd)


# ─── session-dispatch stagger (#1059) ────────────────────────────────────────
#
# Global (cross-issue) pacing of `spawn-issue --auto` session dispatches: each
# fresh session is a ~100K-token cold context load, and the org input-TPM cap
# climbs at each minute boundary (CLAUDE.md § 429 token-pacing), so dispatchers
# that can burst (the watcher infra loops, the file-time filer) keep >=60s
# between session starts. Distinct from the #843 per-issue lease (mutual
# exclusion of duplicate dispatch for ONE issue) and the #1027 auth-outage
# episode gate (fleet-wide suppression): this is last-writer-wins pacing state,
# no holder exclusivity, no suppression sentinel.
SESSION_DISPATCH_STAGGER_S_DEFAULT = 60.0
SESSION_DISPATCH_STAGGER_MAX_S = 300.0  # env ceiling: pacing, not parking


def session_dispatch_stagger_s() -> float:
    """Stagger window in seconds (env ``EPM_SESSION_DISPATCH_STAGGER_S``;
    default 60; ``0`` or negative disables; malformed falls back to the
    default, mirroring :func:`_dispatch_lease_ttl_s`; clamped to
    :data:`SESSION_DISPATCH_STAGGER_MAX_S` so an env typo can never wedge a
    watcher tick for hours — the lease-window clamp posture in
    :func:`_register_autonomous_session`)."""
    raw = os.environ.get("EPM_SESSION_DISPATCH_STAGGER_S")
    if not raw:
        return SESSION_DISPATCH_STAGGER_S_DEFAULT
    try:
        val = float(raw)
    except ValueError:
        return SESSION_DISPATCH_STAGGER_S_DEFAULT
    return min(max(val, 0.0), SESSION_DISPATCH_STAGGER_MAX_S)


def session_dispatch_stamp_path() -> Path:
    """Singleton last-session-dispatch stamp (#1059). NOT per-issue: the
    watcher GC's prefix+int-stem sweep never matches it (same class as
    watch.lock / session_progress.json)."""
    return AUTONOMOUS_REGISTRY_DIR / "last-session-dispatch.json"


def last_session_dispatch_age_s(now: float | None = None) -> float | None:
    """Seconds since the last recorded session dispatch; ``None`` when no
    stamp exists. Garbled content falls back to file mtime (the
    :func:`dispatch_lease_fresh` posture — failing toward pacing is bounded
    by the <=300s window, unlike the lease's fail-toward-fresh which needed
    a TTL bound). A future-dated ts returns 0.0 (treat as just-now)."""
    now = time.time() if now is None else now
    path = session_dispatch_stamp_path()
    try:
        entry = json.loads(path.read_text())
        ts = entry.get("ts") if isinstance(entry, dict) else None
    except FileNotFoundError:
        return None
    except (OSError, ValueError):
        ts = None
    if not (isinstance(ts, int | float) and not isinstance(ts, bool)):
        try:
            ts = path.stat().st_mtime
        except OSError:
            return None  # vanished between read and stat -> no stamp
    return max(now - ts, 0.0)


def stagger_delay_s(age_s: float | None, window_s: float) -> float:
    """PURE: seconds a dispatcher should wait before spawning. 0 when the
    window is disabled (<=0), no prior dispatch (``None``), or the window
    has already elapsed; else the remainder, clamped to ``[0, window_s]``."""
    if window_s <= 0 or age_s is None or age_s >= window_s:
        return 0.0
    return min(window_s - age_s, window_s)


def record_session_dispatch(issue: int, holder: str, now: float | None = None) -> None:
    """Best-effort atomic (tmp + os.replace) write of the dispatch stamp.

    NEVER raises — a failed pacing record must not fail a successful spawn;
    any OSError prints a loud stderr warning (degrades to no stagger for the
    next caller, bounded by the window). TOCTOU note: the callers'
    check->spawn->record sequence is not atomic — the stamp is last-writer-
    wins PACING state, not an exclusion primitive, so a concurrent dispatcher
    already mid-spawn can co-dispatch inside the window (bounded at ~2
    coincident cold loads; the window closes for everyone at the first
    record)."""
    now = time.time() if now is None else now
    entry = {"issue": issue, "holder": holder, "pid": os.getpid(), "ts": now}
    path = session_dispatch_stamp_path()
    try:
        AUTONOMOUS_REGISTRY_DIR.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(entry))
        os.replace(tmp, path)
    except OSError as e:
        print(
            f"WARNING: session-dispatch stamp write failed ({e}); next dispatcher sees no stagger",
            file=sys.stderr,
        )


class RegistrationCollisionError(OSError):
    """``issue-<N>.json`` already names a DIFFERENT session inside the
    collision window — a duplicate ``--auto`` dispatch reached registration
    (#843 M2). Carries ``existing_session_id`` / ``age_s`` so the caller's
    remediation message can name the kept session."""

    def __init__(self, message: str, *, existing_session_id: str, age_s: float) -> None:
        super().__init__(message)
        self.existing_session_id = existing_session_id
        self.age_s = age_s


def _register_autonomous_session(
    issue: int,
    session_id: str,
    cwd: str,
    auto_approve_gpu_hours: float,
    *,
    model: str | None = None,
    betas: list[str] | None = None,
    effort: str | None = None,
    force: bool = False,
) -> None:
    """Record an autonomous issue session so the watcher can resurrect it.

    Written on every `spawn-issue --auto` (initial spawn AND watcher re-spawn),
    overwriting any prior entry with the fresh Happy session id and ``missed=0``.
    RAISES ``OSError`` on write failure — the caller MUST treat a live `--auto`
    session that could not be registered as unsafe (an untracked live session is
    invisible to the watcher and risks a duplicate re-spawn), and stop it.
    Writes atomically (temp file + rename) so the watcher never reads a partial
    JSON entry.

    RAISES :class:`RegistrationCollisionError` (#843 M2, unless ``force=True``)
    when the existing entry names a DIFFERENT session id with a ``spawned_at``
    younger than the collision window — a duplicate dispatch reached
    registration; the caller stops the just-spawned duplicate instead of
    silently overwriting (hiding) the first session. An entry at or past the
    window (or same sid, or garbled ``spawned_at``) overwrites exactly as
    before, so the watcher's crash-recovery respawn — which by its own
    ``RESPAWN_SPAWN_GRACE_S`` only fires on entries >= 15 min old — is never
    blocked. ``register-current`` passes ``force=True`` (the deliberate
    re-write path for an already-live session, #472 revival).

    ``model`` / ``betas`` / ``effort`` are persisted when set so the watcher's
    ``_respawn`` can re-pass them on crash-recovery. ``None`` means "not pinned
    at spawn time" — the watcher omits the flag and the session inherits the
    user's global Claude Code defaults (settings.json + global model picker).
    These three are part of the prompt-cache key, so the watcher MUST re-pass
    the same values it found in the registry — flipping any of them on respawn
    would force a full uncached re-read of the conversation."""
    AUTONOMOUS_REGISTRY_DIR.mkdir(parents=True, exist_ok=True)
    entry: dict[str, Any] = {
        "issue": issue,
        "happy_session_id": session_id,
        "cwd": cwd,
        "auto_approve_gpu_hours": auto_approve_gpu_hours,
        "spawned_at": time.time(),
        "missed": 0,
    }
    if model is not None:
        entry["model"] = model
    if betas:
        entry["betas"] = list(betas)
    if effort is not None:
        entry["effort"] = effort
    dest = AUTONOMOUS_REGISTRY_DIR / f"issue-{issue}.json"
    if not force:
        try:
            existing = json.loads(dest.read_text())
        except (OSError, ValueError):
            existing = {}
        old_sid = existing.get("happy_session_id") if isinstance(existing, dict) else None
        old_ts = existing.get("spawned_at") if isinstance(existing, dict) else None
        # Collision window capped at the 900 s DEFAULT: raising
        # EPM_DISPATCH_LEASE_TTL_S lengthens the LEASE but must never widen
        # this window past RESPAWN_SPAWN_GRACE_S, or a raised TTL would
        # suppress legitimate crash-recovery respawns (round-1 hardening).
        window_s = min(_dispatch_lease_ttl_s(), float(DISPATCH_LEASE_TTL_S))
        if (
            isinstance(old_sid, str)
            and old_sid
            and old_sid != session_id
            and isinstance(old_ts, int | float)
            and not isinstance(old_ts, bool)
            and time.time() - old_ts < window_s
        ):
            age_s = time.time() - old_ts
            raise RegistrationCollisionError(
                f"issue-{issue}.json already names session {old_sid} spawned "
                f"{age_s:.0f}s ago (< {window_s:.0f}s window); refusing to "
                f"overwrite — duplicate dispatch",
                existing_session_id=old_sid,
                age_s=age_s,
            )
    tmp = dest.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(entry, indent=2))
    tmp.replace(dest)


def _register_manual_session(issue: int, session_id: str, cwd: str) -> None:
    """Record a manual (non-`--auto`) issue session for `cmd_list` enrichment.

    Written on every interactive `spawn-issue` so `happy-ls` can map the
    session id back to its issue number + progress. The filename
    (``manual-issue-<N>.json``) is deliberately distinct from the watcher's
    autonomous-session glob (``issue-*.json``) so the watcher will NEVER
    auto-respawn a manual session — manual sessions are driven by the user.
    Writes atomically (temp + rename) so a concurrent reader never sees a
    partial entry. RAISES ``OSError`` on write failure; the caller (manual
    spawn) treats it as non-fatal (the session is already live; we just lose
    the listability enrichment), unlike the autonomous path."""
    AUTONOMOUS_REGISTRY_DIR.mkdir(parents=True, exist_ok=True)
    entry = {
        "issue": issue,
        "happy_session_id": session_id,
        "cwd": cwd,
        "spawned_at": time.time(),
        "mode": "manual",
    }
    dest = AUTONOMOUS_REGISTRY_DIR / f"manual-issue-{issue}.json"
    tmp = dest.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(entry, indent=2))
    tmp.replace(dest)


def _campaign_defaults() -> tuple[float, int, float]:
    """``(budget_gpu_hours, max_concurrent, per_child_cap)`` from the single
    constant source — the ``campaign_state`` module defaults (NOT duplicated
    argparse literals; reviewer NIT on #586). Fail loud when the package is
    unavailable: every campaign code path requires it anyway
    (:func:`cmd_spawn_campaign` imports ``task_workflow`` the same way)."""
    try:
        from explore_persona_space import campaign_state
    except ImportError as e:
        sys.exit(f"cannot import campaign_state ({e}); run via `uv run python`")
    return (
        campaign_state.DEFAULT_GPU_HOURS_TOTAL,
        campaign_state.DEFAULT_MAX_CONCURRENT_CHILDREN,
        campaign_state.DEFAULT_PER_CHILD_GPU_HOURS_CAP,
    )


def _register_campaign_session(
    issue: int,
    session_id: str,
    cwd: str,
    *,
    budget_gpu_hours: float,
    max_concurrent: int,
    per_child_gpu_hours_cap: float,
    model: str | None = None,
    betas: list[str] | None = None,
    effort: str | None = None,
) -> None:
    """Record a campaign session (``/campaign <N>`` driver, task #586) so the
    watcher's campaign pass can resurrect it and re-pass its caps on respawn.

    Same shape the watcher consumes for issue sessions (``issue``,
    ``happy_session_id``, ``spawned_at``, ``missed``), distinguished by the
    ``campaign-<N>.json`` filename prefix + ``mode: "campaign"``, plus the
    campaign caps. Budgets are GPU-HOUR caps, never dollars. Same atomicity
    + RAISES-on-write-failure contract as
    :func:`_register_autonomous_session` (an untracked live campaign session
    risks a duplicate respawn).

    ``model``/``betas``/``effort`` follow the same persistence contract as
    :func:`_register_autonomous_session` — recorded only when pinned at spawn
    time and re-passed verbatim by the watcher's campaign respawn pass."""
    AUTONOMOUS_REGISTRY_DIR.mkdir(parents=True, exist_ok=True)
    entry: dict[str, Any] = {
        "issue": issue,
        "happy_session_id": session_id,
        "cwd": cwd,
        "mode": "campaign",
        "budget_gpu_hours": budget_gpu_hours,
        "max_concurrent": max_concurrent,
        "per_child_gpu_hours_cap": per_child_gpu_hours_cap,
        "spawned_at": time.time(),
        "missed": 0,
    }
    if model is not None:
        entry["model"] = model
    if betas:
        entry["betas"] = list(betas)
    if effort is not None:
        entry["effort"] = effort
    dest = AUTONOMOUS_REGISTRY_DIR / f"campaign-{issue}.json"
    tmp = dest.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(entry, indent=2))
    tmp.replace(dest)


def _load_campaign_registry_entry(issue: int) -> dict[str, Any] | None:
    """Read ``campaign-<N>.json`` for ``issue``; None when absent/unreadable.
    Used to preserve campaign caps across a ``register-current`` rewrite."""
    path = AUTONOMOUS_REGISTRY_DIR / f"campaign-{issue}.json"
    try:
        entry = json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return None
    return entry if isinstance(entry, dict) else None


# Basename of the PM-session registry file under AUTONOMOUS_REGISTRY_DIR.
# Records the Happy session id(s) hosting the PM persona so the watcher's
# zombie-wrapper pass can EXCLUDE them unconditionally (the PM session is
# pinned to the repo root with no issue mapping — without this file it is
# indistinguishable from the unmapped zombie sessions that pass reaps).
# A LIST of ids: each `spawn-pm` / `register-pm` appends; stale ids are
# harmless (a dead sid simply never appears in the daemon's live set).
PM_SESSION_BASENAME = "pm-session.json"

# Cap on recorded PM session ids — keeps the file bounded across months of
# `spawn-pm` invocations while retaining every plausibly-live generation.
_PM_SESSION_MAX_IDS = 20


def _pm_session_path() -> Path:
    """Path of the PM-session registry (function-level lookup so tests that
    monkeypatch ``AUTONOMOUS_REGISTRY_DIR`` are honoured)."""
    return AUTONOMOUS_REGISTRY_DIR / PM_SESSION_BASENAME


def _register_pm_session(session_id: str) -> None:
    """Append ``session_id`` to the PM-session registry (deduped, newest last,
    bounded at :data:`_PM_SESSION_MAX_IDS`). Atomic write (temp + rename).
    RAISES ``OSError`` on write failure — callers decide whether that is
    fatal (``register-pm``: yes, the whole point is the registration) or a
    loud warning (``spawn-pm``: the session is already live)."""
    AUTONOMOUS_REGISTRY_DIR.mkdir(parents=True, exist_ok=True)
    sids = [sid for sid in _load_pm_session_ids_ordered() if sid != session_id]
    sids.append(session_id)
    payload = {"sids": sids[-_PM_SESSION_MAX_IDS:], "updated_at": time.time()}
    dest = _pm_session_path()
    tmp = dest.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, indent=2))
    tmp.replace(dest)


def _load_pm_session_ids_ordered() -> list[str]:
    """PM session ids in registration order (oldest first); ``[]`` when the
    file is missing/garbled (best-effort — a missing registry just means no
    PM exclusion, never a crash)."""
    path = _pm_session_path()
    if not path.is_file():
        return []
    try:
        data = json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return []
    sids = data.get("sids") if isinstance(data, dict) else None
    if not isinstance(sids, list):
        return []
    return [sid for sid in sids if isinstance(sid, str) and sid]


def _load_pm_session_ids() -> set[str]:
    """Set of Happy session ids registered as PM sessions. Consumed by the
    watcher's zombie-wrapper pass as an unconditional exclusion."""
    return set(_load_pm_session_ids_ordered())


def _load_session_issue_map() -> dict[str, int]:
    """Return ``{happy_session_id: issue_number}`` from the autonomous
    (``issue-<N>.json``), manual (``manual-issue-<N>.json``), and campaign
    (``campaign-<N>.json``) registries.

    Best-effort enrichment for :func:`cmd_list`: a single malformed entry is
    skipped (its row will just show no mapped issue), the rest still load.
    Returns ``{}`` if the dir is missing entirely. If an issue number appears
    under both prefixes (autonomous restart after a manual spawn, or vice
    versa), the LATER ``spawned_at`` wins — that's the most recently registered
    session for that issue."""
    out: dict[str, int] = {}
    if not AUTONOMOUS_REGISTRY_DIR.is_dir():
        return out
    # Track which issue each session id maps to + when, so a stale collision
    # resolves to the newer entry rather than dir-iteration order.
    best_ts: dict[str, float] = {}
    # Enumerate the known prefixes explicitly rather than `*issue-*.json`
    # — a wildcard glob would scrape any future sibling file (e.g. a hand-
    # added `weird-issue-N.json` debug dump, or another tool's misnamed
    # entry) and silently overwrite legitimate mappings. The watcher's own
    # respawn glob (`issue-*.json`, NO leading `manual-`) deliberately
    # matches only the autonomous prefix; this loader sees all three kinds
    # (campaign sessions included so `list` maps them to their issue —
    # task #586). The watcher's `campaign-watch-<N>.json` state files also
    # match the `campaign-` glob but carry no integer `issue` key, so the
    # isinstance guard below skips them.
    for prefix in ("issue-", "manual-issue-", "campaign-"):
        for path in AUTONOMOUS_REGISTRY_DIR.glob(f"{prefix}*.json"):
            try:
                entry = json.loads(path.read_text())
            except (json.JSONDecodeError, OSError):
                continue
            sid = entry.get("happy_session_id")
            issue = entry.get("issue")
            ts = entry.get("spawned_at", 0.0)
            if not isinstance(sid, str) or not isinstance(issue, int):
                continue
            if not isinstance(ts, int | float):
                ts = 0.0
            if sid not in best_ts or ts > best_ts[sid]:
                out[sid] = issue
                best_ts[sid] = ts
    return out


# Max chars for the per-row progress cell in `cmd_list` default output. Keeps
# the table readable in a phone-width terminal. The status + marker kind +
# truncated note + age fits comfortably below this.
_PROGRESS_CELL_MAX = 60


def _format_progress_cell(
    issue: int, now: float | None = None, cache_summary: str | None = None
) -> str:
    """One-line ``status / marker_kind (note...) Nh|m ago`` summary for issue
    ``issue``. Returns a VISIBLE placeholder (NOT a silent blank) on lookup
    failure so a broken row is immediately legible to the user.

    If ``cache_summary`` is given (the LLM-written one-line "what the session
    is doing right now" from ``~/.eps-autonomous/session_progress.json``), it
    is used INSTEAD of the marker-based body. Without a cache entry, the
    function falls back to the marker line as before — keeping the table
    legible even before the first ``session_summarize.py`` tick lands.

    Reads task state in-process via :mod:`explore_persona_space.task_workflow`
    rather than shelling out per row — important because `happy-ls` is called
    interactively and a fork+subprocess per session would be ~14x slower than
    the bare table."""
    # Imported lazily so an environment without the project package (e.g. a
    # global `python scripts/spawn_session.py list` run) still gets a usable
    # listing — the progress cell just degrades to a labeled placeholder.
    try:
        from explore_persona_space.task_workflow import get_task, latest_event
    except ImportError as e:
        return f"<lookup unavailable: {type(e).__name__}>"

    try:
        task = get_task(issue)
    except FileNotFoundError:
        return f"#{issue} not found"
    except Exception as e:
        return f"<lookup failed: {type(e).__name__}>"

    status = task.get("status", "?")

    # Prefer the LLM summary from session_progress.json when one is available;
    # it answers "what is it DOING right now" (the marker only answers "what
    # was the last lifecycle event"). Falls through to marker if absent.
    if cache_summary:
        summary = cache_summary.strip().replace("\n", " ")
        overhead = len(f"{status} / ")
        budget = max(0, _PROGRESS_CELL_MAX - overhead)
        if len(summary) > budget:
            summary = summary[: max(0, budget - 1)] + "…"
        return f"{status} / {summary}"

    try:
        marker = latest_event(issue, prefix="epm:")
    except Exception as e:
        return f"{status} / <marker-read failed: {type(e).__name__}>"

    if marker is None:
        return f"{status} / no marker yet"

    kind = marker.get("kind", "?")
    # Drop the `epm:` prefix for compactness — the column header makes it
    # implicit, and short marker kinds (run-finished, results, progress) carry
    # the meaning.
    short_kind = kind[4:] if kind.startswith("epm:") else kind
    note = (marker.get("note") or "").strip().replace("\n", " ")
    age = _format_event_age(marker.get("ts"), now=now)

    # Budget the note to whatever's left after status + kind + age.
    overhead = len(f"{status} / {short_kind}  {age}")
    note_budget = max(0, _PROGRESS_CELL_MAX - overhead - 4)  # 4 = `" ()"` + slack
    if note and note_budget > 0:
        if len(note) > note_budget:
            note = note[: max(0, note_budget - 1)] + "…"
        return f"{status} / {short_kind} ({note}) {age}"
    return f"{status} / {short_kind} {age}"


def _format_event_age(ts: str | None, now: float | None = None) -> str:
    """Render an event ``ts`` (``%Y-%m-%dT%H:%M:%SZ`` UTC) as a compact age
    suffix like ``"3m ago"`` / ``"2h ago"`` / ``"4d ago"``. Returns ``""`` if
    ``ts`` is missing or unparseable — the cell still renders cleanly without
    the age."""
    if not isinstance(ts, str) or not ts:
        return ""
    from datetime import datetime

    try:
        # Normalise the canonical trailing 'Z' to a tz-aware parse.
        when = datetime.fromisoformat(ts.replace("Z", "+00:00")).timestamp()
    except (ValueError, OSError):
        return ""
    now_ts = now if now is not None else time.time()
    delta = max(0.0, now_ts - when)
    if delta < 60:
        return f"{int(delta)}s ago"
    if delta < 3600:
        return f"{int(delta / 60)}m ago"
    if delta < 86400:
        return f"{int(delta / 3600)}h ago"
    return f"{int(delta / 86400)}d ago"


def _load_session_meta() -> dict[str, dict[str, Any]]:
    """Map ``happySessionId -> metadata`` from ``~/.happy/sessions.json``.

    Best-effort enrichment for :func:`cmd_list`: returns ``{}`` if the file is
    missing or unreadable rather than failing the listing."""
    if not SESSIONS_JSON.is_file():
        return {}
    try:
        raw = json.loads(SESSIONS_JSON.read_text())
    except (json.JSONDecodeError, OSError):
        return {}
    sessions = raw.get("sessions", {})
    return {sid: (entry.get("metadata") or {}) for sid, entry in sessions.items()}


# A session cwd that IS an issue worktree names its issue even when the
# session has no registry entry (superseded driver generations, never-
# registered chat sessions). Shared by `_dir_label` + `_infer_issue_from_path`.
_WORKTREE_ISSUE_RE = re.compile(r"/\.claude/worktrees/issue-(\d+)/?$")


def _dir_label(path: str | None) -> str:
    """Short, human-friendly cwd label, annotating per-issue worktrees.

    ``/home/me/explore-persona-space`` -> ``explore-persona-space``;
    a ``.claude/worktrees/issue-<N>`` path gets an ``[issue-<N>]`` tag."""
    if not path:
        return "?"
    home = str(Path.home())
    short = path[len(home) + 1 :] if path.startswith(home + "/") else path
    m = _WORKTREE_ISSUE_RE.search(path)
    return f"{short}  [issue-{m.group(1)}]" if m else short


def _infer_issue_from_path(path: str | None) -> int | None:
    """Issue number inferred from an ``issue-<N>`` worktree cwd, or ``None``.

    Display-level fallback for `cmd_list` rows whose session id has NO
    registry entry — superseded/zombie driver generations (a newer spawn
    overwrote the per-issue registration file) and never-registered chat
    sessions. The cwd still names the issue worktree, so PM triage can
    attribute the row instead of reading ``-`` (2026-06-10: 13 such rows
    rendered unmapped and a triage concluded "no session mapped to #518")."""
    if not path:
        return None
    m = _WORKTREE_ISSUE_RE.search(path)
    return int(m.group(1)) if m else None


def _issue_cell(issue: int | None, path: str | None) -> str:
    """Issue-column cell for `cmd_list`: ``#N`` (registered) beats ``~#N``
    (inferred from an issue-worktree cwd — the tilde marks unregistered)
    beats ``-`` (unmapped)."""
    if issue is not None:
        return f"#{issue}"
    inferred = _infer_issue_from_path(path)
    return f"~#{inferred}" if inferred is not None else "-"


def daemon_port() -> int:
    """Read the live Happy daemon's HTTP port. Fail loudly if the daemon
    isn't running or the state file is missing."""
    if not DAEMON_STATE.is_file():
        sys.exit(
            f"Happy daemon state file missing at {DAEMON_STATE}. "
            "Start Happy at least once interactively (``happy``) so the "
            "daemon registers, then retry."
        )
    state = json.loads(DAEMON_STATE.read_text())
    port = state.get("httpPort")
    if not isinstance(port, int):
        sys.exit(f"daemon.state.json has no integer httpPort field: {state!r}")
    return port


# Per-route HTTP timeouts (seconds). `/spawn-session` boots a new claude
# child process (inherits QR-pairing keys, sets up tmux/non-tmux session) and
# routinely takes >10s when the daemon is juggling many sessions — the prior
# fixed 10s timeout misfired healthy spawns as hard failures (incident #524,
# 2026-06-08: daemon healthy on :39759, spawn timed out, succeeded on retry).
# Worse, a daemon-side spawn that COMPLETES after the client timeout would
# orphan the session: the registry-write atomicity invariant (a live `--auto`
# session MUST have a current registry entry, else the watcher could re-spawn
# it as a duplicate -> duplicate pod -> GPU spend) is only enforced AFTER
# `urlopen` returns. See :func:`_reconcile_spawn_after_timeout` for the
# orphan-adoption path that recovers on this exact race.
DEFAULT_TIMEOUT_S = 10
SPAWN_SESSION_TIMEOUT_S = 60
# Hard join bound on the deliberate-stop breadcrumb post in `cmd_stop` —
# `post_event` enters a blocking flock with no timeout, so a wedged lock
# would otherwise hang the stop indefinitely (#902; plan §4.4 D5).
STOP_BREADCRUMB_JOIN_TIMEOUT_S = 10.0

# The Happy daemon waits 15s for the spawned child's /session-started webhook,
# then resolves {"success": false, "error": "Session webhook timeout for PID <n>"}
# (regular path) / "... (tmux)" (tmux path) WITHOUT killing the child — the fork
# stays live and tracked (bundle index-q9G4ktSK.mjs lines 5346/5466; incident
# task #956, 2026-07-03: two consecutive 500s leaked two live-but-empty
# sessions, third attempt succeeded). On this shape post() reaps the
# half-spawned child, then retries ONCE after a backoff sized at 2x the
# daemon's 15s webhook window.
WEBHOOK_TIMEOUT_RE = re.compile(r"Session webhook timeout for PID (\d+)(?P<tmux> \(tmux\))?")
WEBHOOK_TIMEOUT_MAX_RETRIES = 1
WEBHOOK_TIMEOUT_RETRY_BACKOFF_S = 30.0
# Greppable prefix for every reap-related stderr line, so "fix engaged" is
# distinguishable in production spawn logs (nohup/session transcripts).
WEBHOOK_REAP_LOG_PREFIX = "webhook-timeout reap:"
# Bounded wait for a SIGTERMed / daemon-stopped child to die (~10s), the
# 20 x 0.5s shape _stop_fallback (#903) already uses.
REAP_PID_DEATH_POLL_TRIES = 20
REAP_PID_DEATH_POLL_INTERVAL_S = 0.5


def post(path: str, body: dict[str, Any]) -> dict[str, Any]:
    """POST a JSON body to the local Happy daemon and return the parsed
    response. Errors are surfaced as :func:`sys.exit` with the daemon's
    response body when available.

    The ``/spawn-session`` route uses a longer timeout
    (:data:`SPAWN_SESSION_TIMEOUT_S`) than the lightweight ``/list`` /
    ``/stop-session`` routes (:data:`DEFAULT_TIMEOUT_S`). On a spawn-session
    timeout this function attempts to ADOPT a child the daemon may have
    finished creating after we gave up — turning the orphan/duplicate
    hazard into an idempotent spawn (see
    :func:`_reconcile_spawn_after_timeout`). For any other route, a timeout
    surfaces as a clean failure so the caller can safely retry.

    On an HTTP error whose body matches the daemon's ``Session webhook
    timeout for PID <n>`` shape (the daemon forked a child but its
    /session-started webhook missed the 15s window — the child stays ALIVE
    and TRACKED daemon-side), post() reaps the half-spawned child (daemon
    /stop-session — by session id when a late webhook already landed, else
    by the daemon's ``PID-<pid>`` stop branch; identity-checked SIGTERM only
    as the untracked-but-alive fallback) and retries once after
    :data:`WEBHOOK_TIMEOUT_RETRY_BACKOFF_S`. A failed/unconfirmed reap exits
    nonzero WITHOUT retrying — retrying over an unconfirmed leak or a
    surviving inner claude could double-spawn (#956)."""
    url = f"http://127.0.0.1:{daemon_port()}{path}"
    payload = json.dumps(body).encode()
    timeout = SPAWN_SESSION_TIMEOUT_S if path == "/spawn-session" else DEFAULT_TIMEOUT_S
    max_attempts = 1 + (WEBHOOK_TIMEOUT_MAX_RETRIES if path == "/spawn-session" else 0)
    for attempt in range(1, max_attempts + 1):
        req = urllib.request.Request(
            url,
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        # Per-attempt so a retry's client-socket timeout reconciles against
        # ITS OWN freshness window (#956), not attempt 1's.
        spawn_started_at = time.time() if path == "/spawn-session" else None
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return json.loads(resp.read())
        except urllib.error.HTTPError as e:
            try:
                err_body = json.loads(e.read())
            except Exception:
                err_body = {"raw": str(e)}  # non-dict / non-JSON body guard (tested)
            m = None
            err_text = ""
            if path == "/spawn-session":
                err_text = (
                    err_body.get("error", "") if isinstance(err_body, dict) else str(err_body)
                )
                m = WEBHOOK_TIMEOUT_RE.search(err_text or "")
            if m is None:
                # UNCHANGED failure surface for every non-webhook-timeout
                # error and every non-/spawn-session route.
                sys.exit(f"Happy daemon {path} returned HTTP {e.code}: {err_body}")
            outcome = _reap_half_spawned_session(
                int(m.group(1)), body.get("directory"), is_tmux=m.group("tmux") is not None
            )
            print(
                f"  {WEBHOOK_REAP_LOG_PREFIX} /spawn-session webhook-timeout (HTTP {e.code}, "
                f"attempt {attempt}/{max_attempts}): {err_text!r}; reap: {outcome.detail}",
                file=sys.stderr,
            )
            if not outcome.reaped:
                sys.exit(
                    f"Happy daemon /spawn-session returned HTTP {e.code}: {err_body}. "
                    f"Could NOT confirm the half-spawned child (PID {m.group(1)}) was "
                    f"fully reaped ({outcome.detail}); NOT retrying (a retry over an "
                    "unconfirmed leak could double-spawn). Clean up manually "
                    "(spawn_session.py list / stop), then re-run."
                )
            if attempt >= max_attempts:
                sys.exit(
                    f"Happy daemon /spawn-session returned HTTP {e.code} (webhook "
                    f"timeout) {max_attempts} times; the half-spawned child was reaped "
                    f"each time (last: {outcome.detail}). The daemon looks loaded — "
                    "retry later (no session leaked)."
                )
            print(
                f"  {WEBHOOK_REAP_LOG_PREFIX} retrying /spawn-session in "
                f"{WEBHOOK_TIMEOUT_RETRY_BACKOFF_S:.0f}s "
                f"(attempt {attempt + 1}/{max_attempts})...",
                file=sys.stderr,
            )
            time.sleep(WEBHOOK_TIMEOUT_RETRY_BACKOFF_S)
            continue
        except TimeoutError as e:
            # `socket.timeout is TimeoutError` (CPython 3.10+); `urlopen` raises it
            # DIRECTLY on socket timeout (NOT wrapped in URLError). Reconcile for
            # /spawn-session, surface cleanly for everything else.
            if path == "/spawn-session" and spawn_started_at is not None:
                adopted = _reconcile_spawn_after_timeout(body, spawn_started_at)
                if adopted is not None:
                    print(
                        f"  NOTE: /spawn-session POST timed out after {timeout}s; "
                        f"daemon completed the spawn after the client gave up. "
                        f"Adopted session {adopted} (directory match).",
                        file=sys.stderr,
                    )
                    return {"success": True, "sessionId": adopted}
            sys.exit(
                f"Happy daemon {path} timed out after {timeout}s: {e}. "
                "Retry is safe ONLY if you can confirm no session was created "
                "(check `spawn_session.py list`)."
            )
        except urllib.error.URLError as e:
            sys.exit(f"Happy daemon {path} unreachable at 127.0.0.1: {e}")
    raise AssertionError("unreachable: every post() attempt returns or exits")


def _reconcile_spawn_after_timeout(
    request_body: dict[str, Any], spawn_started_at: float
) -> str | None:
    """Look for a daemon child that matches the just-attempted spawn.

    Called only after a ``/spawn-session`` POST times out. Cross-references
    the daemon's live ``/list`` against ``~/.happy/sessions.json`` to find a
    session whose cwd matches ``request_body["directory"]`` and whose
    ``lifecycleStateSince`` timestamp falls in the window
    ``[spawn_started_at - 5s, now + 5s]`` (the slack absorbs clock skew
    between this process and the daemon's epoch-ms timestamps).

    Returns the adopted Happy session id on a unique match, or ``None`` if no
    plausible match is found (the caller then surfaces the timeout as a
    clean failure). Multiple plausible matches also return ``None`` — refuse
    to guess between competing candidates rather than adopt the wrong one.

    Pure-ish: takes no I/O parameters; reads the daemon and sessions.json
    directly. The narrow surface keeps the post-timeout path testable via
    monkeypatching the live-id + meta loaders."""
    directory = request_body.get("directory")
    if not isinstance(directory, str) or not directory:
        return None
    try:
        live_ids = _live_session_ids()
    except SystemExit:
        # daemon_port() failed mid-recovery; nothing to adopt.
        return None
    if not live_ids:
        return None
    meta = _load_session_meta()
    # Convert our seconds-since-epoch to ms (the daemon's units). Allow 5s
    # of slack on the lower bound to absorb clock skew between the daemon
    # logging lifecycleStateSince and us reading time.time() above.
    window_lo_ms = (spawn_started_at - 5.0) * 1000.0
    window_hi_ms = (time.time() + 5.0) * 1000.0
    candidates: list[tuple[float, str]] = []  # (lifecycleStateSince_ms, sid)
    for sid in live_ids:
        if not isinstance(sid, str):
            continue
        entry = meta.get(sid) or {}
        if entry.get("path") != directory:
            continue
        since = entry.get("lifecycleStateSince")
        if not isinstance(since, int | float):
            # Session is live + dir matches but the daemon hasn't persisted
            # its timestamp yet — refuse to adopt without the freshness
            # signal (could be an unrelated long-running session).
            continue
        if window_lo_ms <= float(since) <= window_hi_ms:
            candidates.append((float(since), sid))
    if len(candidates) != 1:
        # Zero candidates = nothing to adopt; multiple = ambiguous, refuse
        # to guess (the caller fails loud, the user reconciles by hand).
        return None
    return candidates[0][1]


class _ReapOutcome(NamedTuple):
    """Outcome of :func:`_reap_half_spawned_session` (#956)."""

    reaped: bool  # True == no live half-spawned process remains (wrapper AND inner claude)
    detail: str  # human-readable outcome for the stderr NOTE / exit message


def _reap_half_spawned_session(
    pid: int, directory: str | None, *, is_tmux: bool = False
) -> _ReapOutcome:
    """Reap the child the daemon forked for a /spawn-session that failed with
    'Session webhook timeout for PID <pid>' (#956). The daemon does NOT kill
    that child; unreaped it becomes a live-but-empty unmapped session nothing
    cleans up for >=12h. Legs, in order:

    1. LATE-HANDSHAKE probe: strict /list. The daemon's /list FILTERS OUT
       never-handshaken children (bundle line 4079), so an entry for ``pid``
       (which by construction carries a happySessionId) means the webhook
       landed LATE -> stop via /stop-session {sessionId: <sid>}. /list
       ABSENCE is the NORMAL never-handshaken state, NOT "already gone".
    2. DAEMON PID-STOP (primary no-sid leg): /stop-session
       {sessionId: "PID-<pid>"} — the daemon's stopSession PID- branch
       (bundle line 5559) SIGTERMs the tracked child itself and untracks it.
       The daemon only kills a pid it tracks, so this leg carries no
       PID-recycle exposure. success:true does NOT prove death (the daemon
       swallows kill errors, lines 5564-5573) -> always confirm with the
       client-side death poll.
    3. ALREADY-GONE verdict: PID-stop success:false (daemon reachable, pid
       untracked — onChildExited untracked a self-exited child) AND
       _pid_alive(pid) false.
    4. FALLBACK identity-checked SIGTERM (untracked-but-alive anomaly only:
       e.g. the daemon restarted between fork and reap, orphaning the child
       from tracking): the #903 _stop_fallback identity trio (comm=='node',
       happy-wrapper cmdline, not-the-daemon-pid) PLUS a /proc/<pid>/cwd ==
       spawn-directory bind. Ambiguity always REFUSES the kill
       (reaped=False). SKIPPED for the tmux variant (pane-PID /proc identity
       signature unverified for tmux; fail loud with a tmux hint).

    ANY transport failure talking to the daemon (unreachable /list or
    /stop-session) returns reaped=False — 'daemon said success:false' and
    'daemon unreachable' are deliberately NOT conflated (the former is
    evidence about tracking state; the latter is no evidence at all).

    Survivor rule (DIVERGES from the #903 _stop_fallback precedent, which
    only WARNS on a surviving inner claude — safe there because no retry
    follows a takeover stop): after ANY kill leg (sid-stop, PID-stop, or
    fallback SIGTERM), if the pre-kill-resolved inner claude pid is still
    alive once the wrapper died, return reaped=False — retrying over a live
    inner claude recreates the live-unmapped-work class at process level
    (double-spawn risk)."""
    import session_resolver  # lazy: session_resolver imports spawn_session at top level

    # Pre-kill: resolve the inner claude while the wrapper's /proc tree is
    # still walkable (post-kill resolution is impossible). Best-effort; a
    # read-only /proc walk is harmless even if pid later fails identity.
    claude_pid = session_resolver.resolve_claude_pid(pid)

    def _confirm_dead_and_no_survivor(via: str) -> _ReapOutcome:
        if not _await_pid_death(pid, session_resolver):
            return _ReapOutcome(False, f"{via} ACKed but PID {pid} survived ~10s")
        if claude_pid is not None and session_resolver._pid_alive(claude_pid):
            return _ReapOutcome(
                False,
                f"{via}: wrapper PID {pid} dead but inner claude PID {claude_pid} "
                "SURVIVED — retry blocked (double-spawn risk); kill it manually, then re-run",
            )
        return _ReapOutcome(True, f"reaped via {via} (PID {pid} dead)")

    # --- Leg 1: late-handshake probe (strict /list) ---
    try:
        children = _live_children(strict=True)
    except RuntimeError as e:
        return _ReapOutcome(False, f"daemon /list unreachable while verifying PID {pid}: {e}")
    match = next((c for c in children if c.get("pid") == pid), None)
    sid = (match or {}).get("happySessionId")
    if isinstance(sid, str) and sid:
        try:
            ok = _stop_session_raw(sid)
        except RuntimeError as e:
            return _ReapOutcome(False, f"daemon unreachable during sid-stop of {sid}: {e}")
        if ok:
            return _confirm_dead_and_no_survivor(f"daemon sid-stop of late-handshaken {sid}")
        # ok False: entry vanished between /list and the stop (self-exit race,
        # concurrent stop) -> fall through to the PID-stop leg.
    # --- Leg 2: daemon PID-stop (primary no-sid leg) ---
    try:
        ok = _stop_session_raw(f"PID-{pid}")
    except RuntimeError as e:
        return _ReapOutcome(False, f"daemon unreachable during PID-stop of PID {pid}: {e}")
    if ok:
        return _confirm_dead_and_no_survivor(f"daemon PID-stop (PID-{pid})")
    # --- Leg 3: already-gone verdict (daemon reachable, pid untracked) ---
    if not session_resolver._pid_alive(pid):
        return _ReapOutcome(
            True, f"child PID {pid} already exited (untracked by daemon, not alive)"
        )
    # --- Leg 4: untracked-but-alive anomaly -> identity-checked SIGTERM fallback ---
    refusal = _fallback_identity_sigterm(pid, directory, is_tmux, session_resolver)
    if refusal is not None:
        return refusal
    return _confirm_dead_and_no_survivor(f"fallback SIGTERM of untracked PID {pid}")


def _fallback_identity_sigterm(
    pid: int, directory: str | None, is_tmux: bool, session_resolver
) -> _ReapOutcome | None:
    """Leg 4 of :func:`_reap_half_spawned_session`: the untracked-but-alive
    anomaly (e.g. a daemon restart between fork and reap orphaned the child
    from tracking). Runs the #903 ``_stop_fallback`` identity trio
    (comm=='node' / happy-wrapper cmdline / not-the-daemon-pid) PLUS a
    ``/proc/<pid>/cwd == spawn-directory`` bind; ambiguity always REFUSES the
    kill (a ``reaped=False`` outcome, never a signal). Returns a REFUSAL
    :class:`_ReapOutcome`, or ``None`` after an actually-issued SIGTERM (the
    caller then runs the shared death-poll + survivor confirmation).
    SKIPPED for the tmux variant (the pane PID may be a shell wrapper, so the
    /proc signature is unverified there — refuse with a tmux hint)."""
    if is_tmux:
        return _ReapOutcome(
            False,
            f"PID {pid} is tmux-spawned, untracked by the daemon, and still alive; the "
            "/proc identity signature is unverified for tmux pane PIDs — refusing the "
            "client-side kill. tmux path: clean up via tmux, then re-run.",
        )
    comm = session_resolver._read_proc_comm(pid)
    if comm != "node":
        return _ReapOutcome(False, f"PID {pid} comm={comm!r} != 'node' (recycled?); refusing kill")
    cmdline = session_resolver._read_proc_cmdline(pid) or ""
    if "happy" not in cmdline:
        return _ReapOutcome(
            False, f"PID {pid} cmdline lacks the happy-wrapper signature; refusing kill"
        )
    daemon_pid = session_resolver._happy_daemon_pid()
    if daemon_pid is not None and pid == daemon_pid:
        return _ReapOutcome(False, f"PID {pid} is the Happy DAEMON pid; refusing kill")
    if directory:
        try:
            proc_cwd: str | None = os.readlink(f"/proc/{pid}/cwd")
        except OSError:
            proc_cwd = None  # unreadable cwd is not disqualifying; 3 checks above hold
        if proc_cwd is not None and proc_cwd != directory:
            return _ReapOutcome(
                False, f"PID {pid} cwd {proc_cwd!r} != spawn dir {directory!r}; refusing kill"
            )
    os.kill(pid, signal.SIGTERM)
    return None


def _await_pid_death(pid: int, session_resolver) -> bool:
    """~10s bounded death poll (20 x 0.5s), the _stop_fallback (#903) shape.
    session_resolver._pid_alive is the ONE liveness seam (tests monkeypatch
    it + spawn_session.time.sleep)."""
    for _ in range(REAP_PID_DEATH_POLL_TRIES):
        time.sleep(REAP_PID_DEATH_POLL_INTERVAL_S)
        if not session_resolver._pid_alive(pid):
            return True
    return False


def _live_children(*, strict: bool = False) -> list[dict[str, Any]]:
    """Raw child-session dicts (``happySessionId`` / ``pid`` / ``startedBy``)
    the daemon is actively tracking. Returns ``[]`` if the daemon is
    unreachable so callers can degrade (``list --all``) or fail loud
    (``register-current``) as appropriate.

    NOTE: the daemon's /list returns ONLY handshaken children (it filters
    ``happySessionId !== undefined``, bundle line 4079) — a just-forked,
    never-handshaken child is NOT in this list by design. With ``strict=True``
    a daemon-unreachable / unparseable / wrong-shape /list (a parseable
    non-dict body, or a non-list ``children`` field) RAISES RuntimeError
    instead of returning ``[]`` — used by the #956 reap's late-handshake
    probe, where a silent ``[]`` must not read as "no late handshake" when
    the daemon was simply unreachable. LENIENT mode keeps its historical
    semantics untouched for existing callers."""
    try:
        url = f"http://127.0.0.1:{daemon_port()}/list"
        req = urllib.request.Request(
            url, data=b"{}", headers={"Content-Type": "application/json"}, method="POST"
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read())
    except (urllib.error.URLError, OSError, SystemExit, json.JSONDecodeError) as e:
        if strict:
            raise RuntimeError(f"daemon /list failed: {e}") from e
        return []
    if strict:
        if not isinstance(data, dict):
            raise RuntimeError(f"daemon /list returned non-dict JSON: {repr(data)[:200]}")
        children = data.get("children", [])
        if not isinstance(children, list):
            raise RuntimeError(f"daemon /list 'children' is not a list: {repr(children)[:200]}")
        return children
    children = data.get("children", [])
    return children if isinstance(children, list) else []


def _live_session_ids() -> set[str]:
    """Best-effort set of session ids the daemon is actively tracking.

    Returns an empty set if the daemon is unreachable, so ``list --all`` still
    works (it falls back to showing every known session as ``stopped``)."""
    return {c.get("happySessionId") for c in _live_children()}


def _ancestor_pids(max_depth: int = 50) -> list[int]:
    """PIDs of this process's ancestors, nearest first, walked via ``/proc``.

    Used by ``register-current`` to find which live Happy node wrapper this
    process is running under (the daemon's ``/list`` ``pid`` field is the
    node wrapper, an ancestor of any subprocess the session spawns). Stops
    at pid 1 or an unreadable stat. Linux-only (/proc), matching the VM
    runtime this script targets."""
    pids: list[int] = []
    pid = os.getpid()
    for _ in range(max_depth):
        try:
            stat = Path(f"/proc/{pid}/stat").read_text()
        except OSError:
            break
        # The comm field (2nd) can contain spaces/parens; ppid is the 2nd
        # whitespace field after the LAST ')'.
        try:
            ppid = int(stat.rsplit(")", 1)[1].split()[1])
        except (IndexError, ValueError):
            break
        if ppid < 1:
            break
        pids.append(ppid)
        if ppid == 1:
            break
        pid = ppid
    return pids


# ─── Claude session CLI overrides (model / betas / effort) ──────────────────
#
# Each spawn subcommand accepts an optional triple that flows through to the
# new Claude Code session as cmdline flags (``--model`` / ``--betas`` /
# ``--effort``). All three are part of the prompt-cache key, so once a session
# is spawned with a value, every respawn MUST re-pass the same value — flipping
# any of them mid-session forces a full uncached re-read of the conversation
# (see CLAUDE.md § Context hygiene). For ``--auto`` issue / campaign sessions
# the spawn path persists them in the autonomous registry; the watcher's
# ``_respawn`` reads them back and re-passes them verbatim.
#
# ``--model`` accepts the same aliases Claude Code's ``/model`` does — ``opus``,
# ``sonnet``, ``haiku``, ``fable``, etc., OR a full model id like
# ``claude-opus-4-8``. ``--betas`` is a comma-separated list of beta headers
# (e.g. ``context-1m-2025-08-07`` for 1M-context). ``--effort`` is one of
# ``low|medium|high|xhigh|max``. All three default to None and are simply not
# passed to Claude when unset (session inherits ~/.claude/settings.json).
_VALID_EFFORTS = ("low", "medium", "high", "xhigh", "max")


def _add_claude_session_args(parser: argparse.ArgumentParser) -> None:
    """Attach the shared ``--model`` / ``--betas`` / ``--effort`` triple to a
    spawn subcommand. Kept in one place so the three spawn paths
    (``spawn-pm``, ``spawn-issue``, ``spawn-campaign``) take an identical
    set of overrides."""
    parser.add_argument(
        "--model",
        default=None,
        help=(
            "Claude model alias or full id for the spawned session "
            "(e.g. 'opus', 'sonnet', 'fable', or 'claude-opus-4-8'). Forwarded "
            "as --model to the underlying `claude` invocation. Default: unset "
            "(session inherits the user's global Claude Code model)."
        ),
    )
    parser.add_argument(
        "--betas",
        default=None,
        help=(
            "Comma-separated list of Anthropic beta headers to enable for the "
            "spawned session (e.g. 'context-1m-2025-08-07' for the 1M-context "
            "beta). Forwarded as --betas to the underlying `claude` "
            "invocation. Default: unset."
        ),
    )
    parser.add_argument(
        "--effort",
        default=None,
        choices=_VALID_EFFORTS,
        help=(
            "Effort level for the spawned session "
            "(low|medium|high|xhigh|max). Forwarded as --effort to the "
            "underlying `claude` invocation. Default: unset "
            "(session inherits the user's global default)."
        ),
    )


def _parse_betas(raw: str | None) -> list[str]:
    """Parse the comma-separated ``--betas`` string into a clean list.
    Empty / whitespace-only entries are dropped. Returns ``[]`` for None or
    an empty string so callers can treat ``not betas`` as "nothing to pass"."""
    if not raw:
        return []
    return [b.strip() for b in raw.split(",") if b.strip()]


def _build_extra_claude_args(
    model: str | None, betas: list[str] | None, effort: str | None
) -> list[str]:
    """Translate the (model, betas, effort) triple into Claude CLI flags.

    Each field is omitted when None / empty — the spawned session then inherits
    the user's global Claude Code defaults for that knob (settings.json + the
    user's global model picker). ``--betas`` takes one or more space-separated
    values per the Claude CLI's `<betas...>` nargs, so we splat the list."""
    extra: list[str] = []
    if model:
        extra.extend(["--model", model])
    if betas:
        extra.extend(["--betas", *betas])
    if effort:
        extra.extend(["--effort", effort])
    return extra


def _verify_happy_patch_or_die(*, context: str) -> None:
    """Fail loud if the Happy daemon injection patch is reverted/drifted/moved.

    Called BEFORE any spawn path that relies on HAPPY_INITIAL_PROMPT /
    claudeArgs injection. A reverted (or hash-renamed-away) patch makes the
    daemon ignore those fields, so the spawned session boots empty and never
    fires its skill — an idle 'spawned but never ran' session (the failure
    CLASS behind #685's symptom; the 2026-06-28 idle-session pile itself was
    the distinct #720 mapping-loss cause, not a patch revert). Single-digit-ms:
    one or two file reads + substring, no subprocess, no root.

    ``context`` names the caller (e.g. "spawn-issue --auto") for the message.

    The ``missing`` classification (the daemon .mjs file is absent) is AMBIGUOUS
    and is disambiguated with a SECOND, INDEPENDENT check on the daemon RPC's
    own state file (:data:`DAEMON_STATE` / daemon.state.json), since the two
    files are independent (classify_patch reads the vendored .mjs; daemon
    reachability lives in daemon.state.json):

      - .mjs missing AND daemon.state.json missing  -> Happy is not installed
        on this host -> WARN + proceed (a legitimate fresh VM not running the
        autonomous loop; the downstream daemon RPC fails loud anyway).
      - .mjs missing BUT daemon.state.json present   -> Happy IS reachable but
        the patch file moved (the canonical `npm update happy` hash-rename:
        index-<oldhash>.mjs -> index-<newhash>.mjs) -> the patch CANNOT be
        verified and is almost certainly NOT applied to the new bundle -> DIE
        loud. Re-applying fails loud if the file is truly gone, which is the
        correct diagnosis.
    """
    import _happy_patch_check as hpc  # scripts/ is on sys.path[0]

    st = hpc.classify_patch()
    if st.state == "patched":
        return

    if st.state == "missing":
        # Two-step probe: distinguish 'Happy never installed' (safe) from
        # 'Happy reachable but patch file moved' (the #685 post-update state).
        if not DAEMON_STATE.is_file():
            print(
                f"WARNING [{context}]: {st.detail}. No Happy daemon state file "
                f"at {DAEMON_STATE} either -> no Happy install detected, "
                f"skipping the injection-patch guard.",
                file=sys.stderr,
            )
            return
        # Daemon reachable, patch file absent -> die loud.
        fix = (
            "The Happy daemon patch file is ABSENT but the daemon IS reachable "
            "(daemon.state.json present) -> the vendored bundle was almost "
            "certainly hash-renamed by `npm update happy` and is now UNPATCHED. "
            "Re-apply against the new bundle:\n"
            f"    {hpc.REAPPLY_CMD}\n"
            "(this fails loud if the file is genuinely gone — that is the "
            "correct diagnosis), then restart the daemon:\n"
            f"    {hpc.RESTART_CMD}"
        )
        sys.exit(
            f"ABORT [{context}]: the Happy daemon injection patch could not be "
            f"verified ({st.detail}), but the daemon is reachable.\n"
            f"Spawning now would create a session that IGNORES its initial "
            f"prompt and sits idle forever.\n{fix}"
        )

    # reverted | drifted -> the daemon is up but will IGNORE the injected
    # prompt/args, producing an idle session. Fail loud with the fix.
    if st.state == "reverted":
        fix = (
            f"Re-apply it:\n    {hpc.REAPPLY_CMD}\nthen restart the daemon:\n    {hpc.RESTART_CMD}"
        )
    else:  # drifted
        fix = (
            "The Happy daemon shape has DRIFTED (likely an `npm update happy`); "
            "a blind re-apply will not work. Inspect the file and update "
            "PATCHES in scripts/patch_happy_daemon.py, then:\n"
            f"    {hpc.REAPPLY_CMD}\n    {hpc.RESTART_CMD}"
        )
    sys.exit(
        f"ABORT [{context}]: the Happy daemon injection patch is "
        f"{st.state} ({st.detail}).\n"
        f"Spawning now would create a session that IGNORES its initial prompt "
        f"and sits idle forever (an idle 'spawned but never ran' session; the "
        f"failure CLASS behind #685's symptom — the 2026-06-28 pile itself was "
        f"the distinct #720 mapping-loss cause).\n{fix}"
    )


def _assert_spawn_cwd(cwd: Path, *, issue: int | None) -> None:
    """Refuse to spawn into anything but the canonical repo root or the
    TARGET issue's own worktree (#844: sibling-worktree cwd inheritance).

    By construction this cannot fire after the git-resolved ``PROJECT_ROOT``
    above — it is a tripwire against a future edit reintroducing
    ``__file__``-based resolution. Loud non-zero exit, never a silent spawn.
    """
    if cwd == PROJECT_ROOT:
        if not (cwd / ".git").is_dir():  # a linked worktree has a .git FILE
            sys.exit(
                f"#844 spawn-cwd assertion: {cwd} is not the primary checkout "
                f"(.git is not a directory); refusing to spawn"
            )
        return
    if issue is not None and cwd == WORKTREE_DIR / f"issue-{issue}" and cwd.is_dir():
        return
    target_desc = (
        f"the target issue-{issue} worktree" if issue is not None else "a target issue worktree"
    )
    sys.exit(
        f"#844 spawn-cwd assertion: {cwd} is neither the canonical repo root "
        f"({PROJECT_ROOT}) nor {target_desc}; refusing to spawn"
    )


def cmd_spawn_pm(args: argparse.Namespace) -> None:
    """Spawn a session intended to host the PM persona. The session opens
    cwd=<repo root> (git-resolved canonical primary checkout, #844) so the
    user sees a familiar project. The PM persona is then loaded interactively
    by the user typing ``/pm``."""
    _assert_spawn_cwd(PROJECT_ROOT, issue=None)
    extra_args = _build_extra_claude_args(
        getattr(args, "model", None),
        _parse_betas(getattr(args, "betas", None)),
        getattr(args, "effort", None),
    )
    body: dict[str, object] = {"directory": str(PROJECT_ROOT), "agent": "claude"}
    if extra_args:
        # Only the override branch relies on claudeArgs injection; a no-override
        # PM spawn injects nothing, so guarding it would be a false alarm (the
        # deliberate asymmetry pinned by the test suite).
        _verify_happy_patch_or_die(context="spawn-pm")
        body["claudeArgs"] = extra_args
    resp = post("/spawn-session", body)
    if not resp.get("success"):
        sys.exit(f"spawn failed: {resp}")
    try:
        _register_pm_session(resp["sessionId"])
    except OSError as e:
        # The session is already live; losing the registration only loses the
        # zombie-wrapper-pass exclusion. Loud, not fatal.
        print(
            f"WARNING: PM-session registration failed ({e}); run "
            f"`spawn_session.py register-pm --session-id {resp['sessionId']}` "
            "so the watcher's zombie-wrapper pass excludes this session.",
            file=sys.stderr,
        )
    print(
        f"PM session spawned: {resp['sessionId']}\n"
        f"  cwd: {PROJECT_ROOT}\n"
        f"Open it in Happy on your phone and type ``/pm`` to load the PM persona."
    )


def _stop_spawned_session(session_id: str) -> bool:
    """Best-effort stop of a just-spawned session (the registration-failure /
    registration-collision remediation in :func:`cmd_spawn_issue`). Returns
    True when the daemon confirmed the stop; on False the caller prints a
    manual-cleanup warning (success=False usually means the session already
    died on its own — a benign race — but a genuinely stuck live session
    needs hand cleanup)."""
    try:
        stop_resp = post("/stop-session", {"sessionId": session_id})
        return bool(stop_resp.get("success"))
    except SystemExit:
        return False


def _stop_session_raw(session_id: str) -> bool:
    """Daemon ``/stop-session`` with transport failures kept DISTINGUISHABLE
    from the daemon's own verdict: returns the response ``success`` boolean
    (stopSession's return — true == a tracked entry was found+killed+untracked,
    false == no tracked entry for this sessionId/PID-<pid>); RAISES
    RuntimeError on any transport failure (daemon unreachable, timeout, HTTP
    error, unparseable body). Used by the #956 reap, where 'daemon said no
    such pid' feeds the already-gone verdict but 'daemon unreachable' must
    block the retry. :func:`_stop_spawned_session` (which conflates the two
    as False) keeps its existing callers unchanged.

    Response-shape guard: a PARSEABLE body that is not a ``{success: bool}``
    dict — non-dict JSON (e.g. ``[]``), or a missing / non-bool ``success``
    field — ALSO raises RuntimeError. Wrong shape carries no verdict about
    tracking state, so it blocks the retry exactly like a transport failure;
    it never degrades to False (False is the daemon's own 'pid untracked'
    verdict and feeds the already-gone branch)."""
    url = f"http://127.0.0.1:{daemon_port()}/stop-session"
    req = urllib.request.Request(
        url,
        data=json.dumps({"sessionId": session_id}).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=DEFAULT_TIMEOUT_S) as resp:
            data = json.loads(resp.read())
    except (urllib.error.URLError, OSError, TimeoutError, json.JSONDecodeError) as e:
        raise RuntimeError(f"daemon /stop-session transport failure: {e}") from e
    if not isinstance(data, dict) or not isinstance(data.get("success"), bool):
        raise RuntimeError(
            f"daemon /stop-session returned unexpected response shape: {repr(data)[:200]}"
        )
    return bool(data.get("success"))


def _post_duplicate_suppressed_marker(issue: int, kept_sid: str, stopped_sid: str) -> None:
    """Best-effort ``epm:progress`` marker recording a suppressed duplicate
    ``--auto`` dispatch (#843 M2 registration collision) so the suppression is
    dashboard-visible. A marker failure never blocks the exit — the loud
    REGISTRATION-COLLISION line already fired. ``spawn_session.py`` is
    allowlisted in ``_LOCAL_VM_ONLY_PATHS``
    (tests/test_no_pod_side_task_py_shellout.py) for this task.py shellout."""
    note = (
        f"{DUPLICATE_DISPATCH_NOTE_SENTINEL} duplicate --auto dispatch suppressed: "
        f"kept {kept_sid}, stopped {stopped_sid}"
    )
    try:
        res = subprocess.run(
            [
                "uv",
                "run",
                "python",
                "scripts/task.py",
                "post-marker",
                str(issue),
                "epm:progress",
                "--by",
                "spawn_session",
                "--note",
                note,
            ],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            timeout=60,
        )
        # #1130: task.py exits 0 while printing deferred-commit / LANDING
        # CHECK warnings to stderr; forward them (rc deliberately unchecked
        # — best-effort post, control flow unchanged) instead of swallowing
        # them into capture_output's void.
        err = (res.stderr or "").strip()
        if err:
            for line in err[:2000].splitlines():
                print(f"  [post-marker stderr] {line}", file=sys.stderr)
    except (subprocess.SubprocessError, OSError) as e:
        print(f"  note: duplicate-dispatch marker post failed ({e})", file=sys.stderr)


def cmd_spawn_issue(args: argparse.Namespace) -> None:
    """Spawn a session for issue ``--issue N``. The session opens cwd=<repo root>
    by default, OR cwd=<.claude/worktrees/issue-N> if such a worktree exists
    (so the session is git-isolated to that issue's branch). Both candidates
    are canonical by construction (``PROJECT_ROOT`` is git-common-dir-resolved,
    #844) and ``_assert_spawn_cwd`` refuses any other cwd — a sibling issue's
    worktree can never become the spawned session's cwd.

    By default the new session opens empty and the user types ``/issue N``
    on their phone — permissions are interactive. With ``--auto`` (or an
    explicit ``--initial-prompt``) the session boots with that prompt
    already in place AND with ``--dangerously-skip-permissions`` /
    ``HAPPY_INITIAL_MODE=bypassPermissions`` so the self-paced loop can
    call tools without a human to confirm.

    Autonomous (prompt-bearing) sessions also export two env vars the
    ``/issue`` skill reads:

    - ``EPM_AUTONOMOUS_SESSION=1`` — push through recoverable bugs instead of
      blocking; do not stop except at the real gates.
    - ``EPM_PLAN_AUTOAPPROVE_GPU_HOURS=<T>`` — auto-approve a plan whose
      estimated GPU-hours is ``<= T``; park at ``plan_pending`` (await user)
      above it. ``awaiting_promotion`` stays a human gate regardless.
    """
    issue = args.issue
    worktree = WORKTREE_DIR / f"issue-{issue}"
    if worktree.is_dir():
        cwd = worktree
        cwd_note = f"<worktree> {worktree}"
    else:
        cwd = PROJECT_ROOT
        cwd_note = f"<repo root> {PROJECT_ROOT}  (no worktree at {worktree})"
    _assert_spawn_cwd(cwd, issue=issue)

    betas = _parse_betas(args.betas)
    extra_args = _build_extra_claude_args(args.model, betas, args.effort)
    body: dict[str, object] = {"directory": str(cwd), "agent": "claude"}
    if args.initial_prompt:
        prompt = args.initial_prompt
    elif args.auto:
        # Cold start (and cold respawn via `autonomous_session_watch._respawn`)
        # boots the FULL `/issue <N>` skill once. The full skill arms an
        # in-session cron (Step 0 for --auto, Step 6d.2 re-arm) that fires
        # the lightweight `/issue-tick <N>` skill every 45 minutes — that
        # recurring tick is the driver, NOT a `/loop`. The old `/loop 10m
        # /issue <N>` shape re-loaded the 44K-token /issue SKILL.md on every
        # idle tick; the new shape loads it exactly once per session.
        # (45 min as of 2026-06-12: every tick fire is LLM-priced, and the
        # 10-min pure-Python watcher carries fast detection, so the tick is
        # only the in-session re-driver of last resort. An earlier comment
        # justified the 20-min interval with "the Anthropic prompt cache TTL
        # is 5 min" — inaccurate for this org's subscription auth, which
        # gets the 1-hour cache TTL automatically; the 5-min TTL applies to
        # API-key auth. The cadence stands on fewer-LLM-heartbeats alone.)
        prompt = f"/issue {issue}"
    else:
        prompt = None
    # #866/#903: a fresh `paused-takeover` sentinel means a deliberate session
    # takeover is driving this issue — suppress automated spawns at the ONE
    # choke point every automated caller funnels through (crash-recovery
    # `_respawn`, stalled respawn, orphan sweep, infra-drain, capacity-retry,
    # `file_infra_task.py`). Placed BEFORE lease acquisition: a gate landing
    # after it would leave a TTL-held lease suppressing crash recovery past
    # the sentinel TTL. Manual spawns warn-and-proceed (the #843 lease
    # posture — the incident class is 100% automated races).
    sentinel = takeover_sentinel_fresh(issue)
    if sentinel is not None:
        if args.auto:
            print(
                f"{TAKEOVER_HELD_SENTINEL} issue #{issue}: a deliberate session takeover "
                f"is in flight (sentinel {sentinel}, ttl={_takeover_ttl_s():.0f}s); "
                f"NOT spawning. Manual override: rm {sentinel} (or wait for the TTL)."
            )
            return  # exit 0 — suppressed IS dispatch success (#843 M1b semantics)
        print(
            f"  note: fresh takeover sentinel {sentinel} — a deliberate takeover may be "
            f"in flight; proceeding (manual spawns are not gated)"
        )
    lease: dict[str, Any] | None = None
    if args.auto:
        # #843 M1: atomic per-issue dispatch lease, acquired BEFORE the daemon
        # POST. Exactly one of N concurrent `--auto` dispatchers wins; every
        # loser exits 0 with the loud DISPATCH-LEASE HELD line (a suppressed
        # duplicate means a session IS driving the issue — dispatch success).
        # On success the lease is deliberately LEFT in place (TTL expiry owns
        # it); failure exits below release it so the backstop can retry.
        lease = acquire_dispatch_lease(issue, holder=f"spawn-issue --auto pid={os.getpid()}")
        if lease is None:
            held = read_dispatch_lease(issue) or {}
            print(
                f"{DISPATCH_LEASE_HELD_SENTINEL} issue #{issue}: a dispatch is already "
                f"in flight ({dispatch_lease_desc(held)}, "
                f"ttl={_dispatch_lease_ttl_s():.0f}s); NOT spawning a duplicate. "
                f"Manual override: rm {dispatch_lease_path(issue)}"
            )
            return  # exit 0 — duplicate suppressed IS dispatch success
    elif dispatch_lease_fresh(issue) is not None:
        # Manual / bespoke-prompt spawn: a human decision — warn, proceed,
        # and create NO lease (the incident class is 100% automated races).
        print(
            f"  note: a fresh dispatch lease exists for #{issue} "
            f"({dispatch_lease_desc(read_dispatch_lease(issue))}) — an automated "
            f"dispatch may be in flight; proceeding (manual spawns are not gated)"
        )
    try:
        _spawn_issue_session(args, issue, cwd_note, body, prompt, extra_args, betas, cwd)
    except BaseException:
        if lease is not None:
            # Failure exit (POST failed / patch-verify died / plain-OSError
            # registration failure): free the slot so the next dispatcher /
            # watcher tick can retry. The REGISTRATION-COLLISION branch
            # RETURNS (never raises), so it deliberately HOLDS the lease.
            release_dispatch_lease(issue, lease["token"])
        raise


def _spawn_issue_session(
    args: argparse.Namespace,
    issue: int,
    cwd_note: str,
    body: dict[str, object],
    prompt: str | None,
    extra_args: list[str],
    betas: list[str],
    cwd: Path,
) -> None:
    """The POST + print + registration tail of :func:`cmd_spawn_issue`,
    factored out so the caller can wrap it in the #843 release-lease-on-
    failure guard without re-indenting its every branch. Raises ``SystemExit``
    on any failure exit; returns normally on success AND on the deliberate
    exit-0 REGISTRATION-COLLISION suppression branch (which must NOT release
    the lease — see :func:`release_dispatch_lease`)."""
    if prompt is not None or extra_args:
        # Both the load-bearing --auto / --initial-prompt injection path AND the
        # bare model-override path rely on the daemon honoring HAPPY_INITIAL_*
        # / claudeArgs. Verify the daemon patch is applied BEFORE post() — a
        # revert would otherwise spawn an idle 'spawned but never ran' session
        # (the failure CLASS behind #685) or silently drop the model override.
        _verify_happy_patch_or_die(context="spawn-issue")
    if prompt is not None:
        # Auto-prompt sessions have no human at the keyboard to confirm
        # tool permissions, so they start in bypassPermissions mode. The
        # Happy daemon reads HAPPY_INITIAL_PROMPT / HAPPY_INITIAL_MODE
        # from the spawn env on its first nextMessage() and deletes them
        # afterwards (one-shot). claudeArgs is forwarded by the daemon
        # to the Claude Code subprocess as cmdline flags.
        body["environmentVariables"] = {
            "HAPPY_INITIAL_PROMPT": prompt,
            "HAPPY_INITIAL_MODE": "bypassPermissions",
            # Read by the /issue skill: drive autonomously (push through
            # recoverable bugs) and auto-approve plans up to the GPU-hour cap.
            "EPM_AUTONOMOUS_SESSION": "1",
            "EPM_PLAN_AUTOAPPROVE_GPU_HOURS": str(args.auto_approve_gpu_hours),
        }
        body["claudeArgs"] = ["--dangerously-skip-permissions", *extra_args]
    elif extra_args:
        # Bare interactive session — no initial prompt, no bypassPermissions —
        # but the user still asked for a specific model / betas / effort. Pass
        # them through so the empty session opens on the requested model.
        body["claudeArgs"] = extra_args

    resp = post("/spawn-session", body)
    if not resp.get("success"):
        sys.exit(f"spawn failed: {resp}")
    print(f"Issue #{issue} session spawned: {resp['sessionId']}")
    print(f"  cwd: {cwd_note}")
    if extra_args:
        print(f"  claude overrides: {' '.join(extra_args)}")
    if prompt is not None:
        print(f"  initial prompt: {prompt!r}")
        print("  permissions: bypassPermissions (--dangerously-skip-permissions)")
        print(
            f"  autonomous: self-drives; auto-approves plans "
            f"<= {args.auto_approve_gpu_hours:g} GPU-hours, parks above that "
            "+ at awaiting_promotion"
        )
        # Only the canonical autonomous dispatch (`--auto`, an /issue loop) is
        # registered for crash-recovery. A bespoke --initial-prompt is one-shot
        # and not re-driven.
        if args.auto:
            try:
                _register_autonomous_session(
                    issue,
                    resp["sessionId"],
                    str(cwd),
                    args.auto_approve_gpu_hours,
                    model=args.model,
                    betas=betas,
                    effort=args.effort,
                )
                print(f"  registered for crash-recovery watch: issue-{issue}.json")
            except RegistrationCollisionError as e:
                # #843 M2: a duplicate --auto dispatch reached registration —
                # a DIFFERENT session was registered for this issue inside the
                # collision window. Keep the FIRST session (its registration
                # stays byte-identical), stop the duplicate we just spawned,
                # exit 0 (a session IS live and driving = dispatch success).
                # The dispatch lease is deliberately HELD (this branch returns
                # without raising, so the caller's release guard never fires)
                # — holding suppresses immediate spawn-then-collision-stop
                # churn for the rest of the TTL.
                print(f"  registration collision: {e}", file=sys.stderr)
                stopped = _stop_spawned_session(resp["sessionId"])
                if not stopped:
                    print(
                        f"  WARNING: could not confirm duplicate session "
                        f"{resp['sessionId']} stopped; if it is still live, stop it "
                        "manually (spawn_session.py stop --session-id ...)",
                        file=sys.stderr,
                    )
                print(
                    f"{REGISTRATION_COLLISION_SENTINEL} issue #{issue}: kept "
                    f"{e.existing_session_id} (registered {e.age_s:.0f}s ago), stopped "
                    f"duplicate {resp['sessionId']}; NOT overwriting the first "
                    f"registration (duplicate dispatch suppressed)"
                )
                _post_duplicate_suppressed_marker(issue, e.existing_session_id, resp["sessionId"])
                return  # exit 0 — the first session is live and driving
            except OSError as e:
                # Atomicity invariant: a live `--auto` session MUST have a current
                # registry entry, else the watcher (which trusts the registry) could
                # re-spawn it as a duplicate -> duplicate pod -> spend. If we cannot
                # register it, stop the session we just spawned and fail loud.
                print(
                    f"  registry write failed ({e}); stopping the just-spawned "
                    "session to avoid an untracked duplicate",
                    file=sys.stderr,
                )
                if not _stop_spawned_session(resp["sessionId"]):
                    # success=False usually means the session already died on its
                    # own (a benign race); surface it anyway so a genuinely stuck
                    # live session can be cleaned up by hand.
                    print(
                        f"  WARNING: could not confirm session {resp['sessionId']} stopped; "
                        "if it is still live, stop it manually "
                        "(spawn_session.py stop --session-id ...)",
                        file=sys.stderr,
                    )
                sys.exit(f"spawn aborted: could not register issue #{issue} for crash-recovery")
    else:
        # Manual session — record a sibling registry entry so `cmd_list` can
        # map the session id back to its issue number + show progress. The
        # filename prefix (`manual-issue-`) is deliberately distinct from the
        # watcher's `issue-*.json` glob, so the watcher will NEVER auto-respawn
        # a manual session. Registration failure is non-fatal here (unlike
        # --auto): the session is already live; we just lose the `list`
        # enrichment. Surface the warning so the gap is visible.
        try:
            _register_manual_session(issue, resp["sessionId"], str(cwd))
            print(f"  registered for `list` enrichment: manual-issue-{issue}.json")
        except OSError as e:
            print(
                f"  WARNING: manual-session registry write failed ({e}); "
                f"session is live but won't show its issue in `list` output",
                file=sys.stderr,
            )
        print(f"Open it in Happy on your phone and type ``/issue {issue}``.")


def cmd_spawn_campaign(args: argparse.Namespace) -> None:
    """Spawn the dedicated autonomous session driving campaign ``--issue N``
    (``/campaign <N>``, task #586).

    Mirrors :func:`cmd_spawn_issue`'s ``--auto`` path with three differences:

    - validates the task is ``kind: campaign`` AND at status ``approved``
      (the human gate IN — the user reviews the ``## Campaign Brief`` and
      runs ``task.py set-status <N> approved``; see workflow.yaml §
      gates.campaign_brief_approval) or ``running`` (re-entry: the skill
      flips approved → running at its Step 0, so a watcher respawn of a
      mid-campaign session re-enters at ``running``). Refuses any other
      status, fail loud.
    - cwd is always the repo root (campaigns drive `tasks/` state and spawn
      children; they own no issue worktree).
    - registers ``campaign-<N>.json`` (``mode: "campaign"`` + the campaign
      caps) so the watcher's campaign pass — not the issue respawn pass —
      owns crash recovery.

    ``EPM_PLAN_AUTOAPPROVE_GPU_HOURS`` is set to the PER-CHILD cap: the
    children the campaign spawns are ordinary ``/issue <child> --auto``
    sessions and inherit their own cap at their own spawn; the campaign
    session itself only ever files plans for children, so the cap bounds
    any plan it would auto-approve in-session."""
    issue = args.issue
    # #844 tripwire FIRST (pure path check, no task-state dependency): a
    # non-canonical cwd must refuse before any task lookup or daemon POST.
    _assert_spawn_cwd(PROJECT_ROOT, issue=None)
    default_budget, default_concurrent, default_per_child = _campaign_defaults()
    budget_gpu_hours = (
        args.budget_gpu_hours if args.budget_gpu_hours is not None else default_budget
    )
    max_concurrent = args.max_concurrent if args.max_concurrent is not None else default_concurrent
    per_child_cap = args.per_child_cap if args.per_child_cap is not None else default_per_child
    try:
        from explore_persona_space.task_workflow import get_task
    except ImportError as e:
        sys.exit(f"cannot import task_workflow ({e}); run via `uv run python`")
    try:
        task = get_task(issue)
    except FileNotFoundError as e:
        sys.exit(f"spawn-campaign: {e}")
    kind = (task.get("frontmatter") or {}).get("kind")
    if kind != "campaign":
        sys.exit(
            f"spawn-campaign: task #{issue} has kind={kind!r}, expected 'campaign'. "
            f"Campaigns are created via `task.py new --kind campaign ...`."
        )
    status = task.get("status")
    if status not in ("approved", "running"):
        sys.exit(
            f"spawn-campaign: task #{issue} is at status {status!r}; a campaign "
            f"executes only from 'approved' (user reviews the ## Campaign Brief, "
            f"then runs `task.py set-status {issue} approved` — workflow.yaml § "
            f"gates.campaign_brief_approval) or 'running' (respawn re-entry)."
        )

    # A campaign session ALWAYS injects HAPPY_INITIAL_PROMPT + claudeArgs (same
    # #685 severity as the --auto issue path). Verify the daemon patch is
    # applied BEFORE building the body / reaching post() — but AFTER the
    # kind/status validation above, so a wrong-kind/-status task still gets its
    # specific error first.
    _verify_happy_patch_or_die(context="spawn-campaign")

    betas = _parse_betas(args.betas)
    extra_args = _build_extra_claude_args(args.model, betas, args.effort)
    prompt = f"/campaign {issue}"
    body: dict[str, object] = {
        "directory": str(PROJECT_ROOT),
        "agent": "claude",
        "environmentVariables": {
            "HAPPY_INITIAL_PROMPT": prompt,
            "HAPPY_INITIAL_MODE": "bypassPermissions",
            "EPM_AUTONOMOUS_SESSION": "1",
            "EPM_CAMPAIGN_SESSION": "1",
            "EPM_PLAN_AUTOAPPROVE_GPU_HOURS": str(per_child_cap),
        },
        "claudeArgs": ["--dangerously-skip-permissions", *extra_args],
    }
    resp = post("/spawn-session", body)
    if not resp.get("success"):
        sys.exit(f"spawn failed: {resp}")
    print(f"Campaign #{issue} session spawned: {resp['sessionId']}")
    print(f"  cwd: <repo root> {PROJECT_ROOT}")
    print(f"  initial prompt: {prompt!r}")
    print("  permissions: bypassPermissions (--dangerously-skip-permissions)")
    if extra_args:
        print(f"  claude overrides: {' '.join(extra_args)}")
    print(
        f"  caps: budget {budget_gpu_hours:g} GPU-h total, "
        f"{max_concurrent} concurrent children, "
        f"{per_child_cap:g} GPU-h per child"
    )
    try:
        _register_campaign_session(
            issue,
            resp["sessionId"],
            str(PROJECT_ROOT),
            budget_gpu_hours=budget_gpu_hours,
            max_concurrent=max_concurrent,
            per_child_gpu_hours_cap=per_child_cap,
            model=args.model,
            betas=betas,
            effort=args.effort,
        )
        print(f"  registered for campaign-watch: campaign-{issue}.json")
    except OSError as e:
        # Same atomicity invariant as the --auto issue path: a live campaign
        # session MUST have a current registry entry, else the watcher could
        # respawn it as a duplicate (duplicate children -> duplicate pods).
        print(
            f"  registry write failed ({e}); stopping the just-spawned "
            "session to avoid an untracked duplicate",
            file=sys.stderr,
        )
        try:
            stop_resp = post("/stop-session", {"sessionId": resp["sessionId"]})
            stopped = bool(stop_resp.get("success"))
        except SystemExit:
            stopped = False
        if not stopped:
            print(
                f"  WARNING: could not confirm session {resp['sessionId']} stopped; "
                "if it is still live, stop it manually "
                "(spawn_session.py stop --session-id ...)",
                file=sys.stderr,
            )
        sys.exit(f"spawn aborted: could not register campaign #{issue} for crash-recovery")


def cmd_register_current(args: argparse.Namespace) -> None:
    """Re-register an EXISTING live session as the driver of issue ``--issue N``.

    Closes the #472 revival blind spot (2026-06-10): when a parked/terminal
    task is revived (same-issue follow-up loop), the watcher's registry entry
    was already DELETED at the terminal transition, so the driving session is
    invisible to every registration-based watcher pass until the orphan
    sweep's ~90-min staleness gate. Calling this at revival restores the
    registration immediately — same file shape the spawn path writes, so the
    watcher consumes it unchanged.

    Session id: ``--session-id`` if given (validated LIVE against the daemon
    — refuses a dead/unknown id), else inferred by walking this process's
    ancestors for a pid the daemon lists as a session wrapper. Fail-loud if
    neither resolves; never guesses.

    Registration kind mirrors how the session was originally spawned:
    ``EPM_AUTONOMOUS_SESSION=1`` (exported only by ``spawn-issue --auto``)
    -> ``issue-<N>.json`` (auto-watch semantics: crash-recovery may respawn
    it — exactly what the original ``--auto`` registration granted before the
    terminal-status GC removed it); otherwise -> ``manual-issue-<N>.json``
    (alert-only: a user-driven session is NEVER auto-respawned, #505).
    ``--mode`` overrides the inference."""
    issue = args.issue
    children = _live_children()
    if args.session_id:
        sid = args.session_id
        live_ids = {c.get("happySessionId") for c in children}
        if sid not in live_ids:
            sys.exit(
                f"session {sid!r} is not live per the Happy daemon; refusing to "
                "register a dead/unknown session (check `spawn_session.py list`)."
            )
    else:
        pid_to_sid = {
            c["pid"]: c["happySessionId"]
            for c in children
            if isinstance(c.get("pid"), int) and isinstance(c.get("happySessionId"), str)
        }
        matches = [pid_to_sid[p] for p in _ancestor_pids() if p in pid_to_sid]
        if not matches:
            sys.exit(
                "could not infer this session's Happy id from the process ancestry "
                "(not running inside a Happy session, or the daemon is unreachable). "
                "Pass --session-id explicitly."
            )
        sid = matches[0]

    if args.mode:
        mode = args.mode
    elif os.environ.get("EPM_CAMPAIGN_SESSION") == "1":
        # Exported only by `spawn-campaign` — a revived campaign session
        # re-registers under the campaign pass, not the issue respawn pass.
        mode = "campaign"
    elif os.environ.get("EPM_AUTONOMOUS_SESSION") == "1":
        mode = "auto"
    else:
        mode = "manual"
    meta_path = (_load_session_meta().get(sid) or {}).get("path")
    cwd = meta_path if isinstance(meta_path, str) and meta_path else os.getcwd()

    try:
        if mode == "campaign":
            # Preserve the caps from the prior registration when one exists;
            # fall back to the campaign_state module defaults (single
            # constant source) otherwise.
            default_budget, default_concurrent, default_per_child = _campaign_defaults()
            prior = _load_campaign_registry_entry(issue) or {}
            if args.auto_approve_gpu_hours is not None:
                per_child = args.auto_approve_gpu_hours
            else:
                per_child = prior.get("per_child_gpu_hours_cap", default_per_child)
            _register_campaign_session(
                issue,
                sid,
                cwd,
                budget_gpu_hours=float(prior.get("budget_gpu_hours", default_budget)),
                max_concurrent=int(prior.get("max_concurrent", default_concurrent)),
                per_child_gpu_hours_cap=float(per_child),
            )
            dest = f"campaign-{issue}.json"
            semantics = "campaign-watch (campaign pass may respawn on death)"
        elif mode == "auto":
            if args.auto_approve_gpu_hours is not None:
                cap = args.auto_approve_gpu_hours
            else:
                cap = float(os.environ.get("EPM_PLAN_AUTOAPPROVE_GPU_HOURS", "100"))
            # force=True: register-current is the deliberate re-write path for
            # an ALREADY-LIVE session (#472 revival) — never a duplicate
            # dispatch, so the #843 M2 collision check must not block it.
            _register_autonomous_session(issue, sid, cwd, cap, force=True)
            dest = f"issue-{issue}.json"
            semantics = "auto-watch (crash-recovery may respawn on death)"
        else:
            if args.auto_approve_gpu_hours is not None:
                print(
                    "  NOTE: --auto-approve-gpu-hours ignored in manual mode "
                    "(only auto-watch entries carry the cap)",
                    file=sys.stderr,
                )
            _register_manual_session(issue, sid, cwd)
            dest = f"manual-issue-{issue}.json"
            semantics = "alert-only (user-driven; never auto-respawned)"
    except OSError as e:
        sys.exit(
            f"registry write failed ({e}); session {sid} remains UNREGISTERED "
            f"for issue #{issue} — the watcher cannot see this revival."
        )
    print(f"Registered session {sid} as driver of issue #{issue}: {dest} [{semantics}]")


# ─── unregister (inverse of register-current; #1327) ─────────────────────────
#
# Strict filename shape of the THREE registration kinds — and ONLY those.
# The registry dir holds many sibling file classes (`dispatch-lease-*.json`,
# `campaign-watch-*.json`, `pm-session.json`, `*.paused-takeover-*` sentinels,
# watcher state files); the `\.json$`-anchored full-match regex cannot scrape
# any of them (a takeover sentinel whose free-form suffix itself ends in
# `.json` still fails the `<prefix>-<digits>.json` full match).
_REGISTRATION_NAME_RE = re.compile(r"^(issue|manual-issue|campaign)-(\d+)\.json$")
_KIND_TO_PREFIX = {"auto": "issue", "manual": "manual-issue", "campaign": "campaign"}


def unregister_paths(
    *,
    issue: int | None,
    session_id: str | None,
    force: bool = False,
    kind: str | None = None,
    registry_dir: Path | None = None,
) -> list[tuple[str, Path, str]]:
    """Remove issue/manual/campaign registration files, sid-match-guarded.

    Returns ``[(action, path, detail)]`` with ``action`` in ``{"removed",
    "kept-sid-mismatch", "kept-unreadable", "missing"}``. Without ``force`` a
    file is removed IFF its recorded ``happy_session_id`` string-equals
    ``session_id``; an unreadable / garbled / missing-``happy_session_id``
    entry is KEPT (fail toward keep, mirroring the watcher's unresolvable-
    input posture). Never touches takeover sentinels or non-registration
    siblings: the ``--issue`` form targets the ≤3 exact filenames, the scan
    form filters through :data:`_REGISTRATION_NAME_RE`. Removal is a single
    atomic ``unlink(missing_ok=True)`` — the watcher re-globs each pass.
    RAISES ``ValueError`` when ``force`` is set without ``issue`` (a forced
    whole-registry scan is refused at the helper layer too)."""
    if force and issue is None:
        raise ValueError("force=True requires issue (a forced whole-registry scan is refused)")
    reg = registry_dir if registry_dir is not None else AUTONOMOUS_REGISTRY_DIR
    prefixes = [_KIND_TO_PREFIX[kind]] if kind else list(_KIND_TO_PREFIX.values())
    scan_mode = issue is None
    if not scan_mode:
        candidates = [reg / f"{p}-{issue}.json" for p in prefixes]
    else:
        candidates = sorted(
            p
            for p in (reg.glob("*.json") if reg.is_dir() else [])
            if (m := _REGISTRATION_NAME_RE.match(p.name)) and m.group(1) in prefixes
        )
    rows: list[tuple[str, Path, str]] = []
    for path in candidates:
        recorded: str | None = None
        readable = False
        try:
            entry = json.loads(path.read_text())
        except FileNotFoundError:
            if not scan_mode:
                rows.append(("missing", path, ""))
            continue
        except (OSError, ValueError):
            pass
        else:
            sid_field = entry.get("happy_session_id") if isinstance(entry, dict) else None
            if isinstance(sid_field, str) and sid_field:
                recorded = sid_field
                readable = True
        if force:
            path.unlink(missing_ok=True)
            detail = (
                f"recorded sid {recorded} (forced)"
                if readable
                else "recorded sid unreadable (forced)"
            )
            rows.append(("removed", path, detail))
        elif not readable:
            # Garbled JSON / OSError / missing or non-string happy_session_id:
            # fail toward keep (only --force --issue may remove it).
            if not scan_mode:
                rows.append(
                    ("kept-unreadable", path, "garbled/unreadable entry — refusing without --force")
                )
        elif recorded == session_id:
            path.unlink(missing_ok=True)  # atomic; watcher re-globs each pass
            rows.append(("removed", path, f"recorded sid {recorded}"))
        elif not scan_mode:
            rows.append(("kept-sid-mismatch", path, f"recorded sid {recorded!r} != {session_id!r}"))
    return rows


def cmd_unregister(args: argparse.Namespace) -> None:
    """Inverse of ``register-current``: remove this session's registration file(s).

    For collision-yield and deliberate-stop paths — replaces the hand-rolled
    `rm ~/.eps-autonomous/issue-<N>.json` (#952). Sid-matched by default:
    without ``--session-id`` the caller's own Happy id is inferred from the
    process ancestry (the ``register-current`` walk), so a yielding duplicate
    can never delete the true owner's entry — a mismatch prints
    ``KEPT-SID-MISMATCH`` and exits 0 (that line is the guard working, not a
    bug). Third-party cleanup of a DEAD session's file: pass
    ``--session-id <dead-sid>`` (no daemon-liveness requirement — the sid
    being removed is typically dead/yielding, the validation is against the
    FILE), or ``--force --issue N`` for unconditional operator removal.
    Takeover sentinels are never touched. No task marker is posted — the
    yield paths post their own breadcrumb; ``--reason`` is echoed per line
    for the transcript."""
    if args.force and args.session_id:
        sys.exit(
            "--force and --session-id are mutually exclusive (--force means "
            "'skip the sid match'); pass --issue with exactly one of them."
        )
    if args.force and args.issue is None:
        sys.exit("--force requires --issue (a forced whole-registry scan is refused).")
    sid = args.session_id
    if sid is None and not args.force:
        # Same ancestry walk as cmd_register_current; NO liveness requirement
        # on the sid being removed, but inference itself needs the daemon's
        # live-children list (we're finding OUR OWN sid).
        children = _live_children()
        pid_to_sid = {
            c["pid"]: c["happySessionId"]
            for c in children
            if isinstance(c.get("pid"), int) and isinstance(c.get("happySessionId"), str)
        }
        matches = [pid_to_sid[p] for p in _ancestor_pids() if p in pid_to_sid]
        if not matches:
            sys.exit(
                "could not infer this session's Happy id from the process "
                "ancestry; pass --session-id explicitly (or --force with "
                "--issue for an unconditional operator removal)."
            )
        sid = matches[0]
    if args.issue is None and sid is None:
        sys.exit("nothing to select: pass --issue and/or --session-id.")
    rows = unregister_paths(issue=args.issue, session_id=sid, force=args.force, kind=args.kind)
    for action, path, detail in rows:
        line = f"{action.upper():<18} {path}"
        if detail:
            line += f" ({detail})"
        if args.reason:
            line += f" [reason: {args.reason}]"
        print(line)
    if not any(action == "removed" for action, _, _ in rows):
        print("nothing removed")


def cmd_register_pm(args: argparse.Namespace) -> None:
    """Register an EXISTING live session as the PM session.

    The watcher's zombie-wrapper pass auto-stops EPS sessions whose process
    tree has carried no inner Claude process for the grace window; the PM
    session (repo-root cwd, no issue mapping) is otherwise indistinguishable
    from the unmapped zombies that pass targets, so it must be excluded by
    explicit registration. ``spawn-pm`` registers automatically; this
    subcommand covers PM sessions opened any other way (a terminal ``happy``,
    a pre-registration spawn) — the `/pm` skill runs it at bootstrap.

    Session id: ``--session-id`` if given (validated LIVE against the
    daemon), else inferred by walking this process's ancestors for a pid the
    daemon lists as a session wrapper (same inference as
    ``register-current``). Fail-loud if neither resolves; never guesses."""
    children = _live_children()
    if args.session_id:
        sid = args.session_id
        live_ids = {c.get("happySessionId") for c in children}
        if sid not in live_ids:
            sys.exit(
                f"session {sid!r} is not live per the Happy daemon; refusing to "
                "register a dead/unknown session (check `spawn_session.py list`)."
            )
    else:
        pid_to_sid = {
            c["pid"]: c["happySessionId"]
            for c in children
            if isinstance(c.get("pid"), int) and isinstance(c.get("happySessionId"), str)
        }
        matches = [pid_to_sid[p] for p in _ancestor_pids() if p in pid_to_sid]
        if not matches:
            sys.exit(
                "could not infer this session's Happy id from the process ancestry "
                "(not running inside a Happy session, or the daemon is unreachable). "
                "Pass --session-id explicitly."
            )
        sid = matches[0]
    try:
        _register_pm_session(sid)
    except OSError as e:
        sys.exit(
            f"PM registry write failed ({e}); session {sid} remains UNREGISTERED — "
            "the watcher's zombie-wrapper pass cannot exclude it."
        )
    print(f"Registered session {sid} as the PM session: {PM_SESSION_BASENAME}")


def _is_eps_dir_label(dir_label: str) -> bool:
    """True iff the rendered dir label refers to EPS (incl. worktrees).

    Matches the literal repo name in the label so worktree labels
    (``explore-persona-space  [issue-N]``) and bare-root labels
    (``explore-persona-space``) BOTH count, while ``my-goat`` / ``introsp``
    do not."""
    return "explore-persona-space" in dir_label


def _load_summary_cache() -> dict[str, dict]:
    """Read ``session_progress.json`` -> ``{happy_session_id: entry}``.

    Best-effort enrichment; returns ``{}`` if the cache file is missing or
    unreadable, so the table degrades to the marker-based progress cell
    instead of breaking."""
    try:
        # Local import — avoids paying the cost when nobody calls `list`.
        import session_summarize

        data = session_summarize.load_cache()
    except Exception:
        return {}
    sessions = data.get("sessions") if isinstance(data, dict) else None
    if not isinstance(sessions, dict):
        return {}
    return {sid: entry for sid, entry in sessions.items() if isinstance(entry, dict)}


def cmd_list(args: argparse.Namespace) -> None:
    """List Happy sessions, enriched with cwd + lifecycle state + issue +
    progress.

    Default: sessions the local daemon is actively tracking, FILTERED to EPS
    (the project root + any of its worktrees). The ``progress`` column shows
    the LLM-written summary from ``~/.eps-autonomous/session_progress.json``
    when present, otherwise falls back to the marker-based summary.

    ``--all``: every session in ``~/.happy/sessions.json`` (including stopped
    ones), newest first, so you can pick one to ``happy resume``.

    ``--all-dirs``: restore the pre-EPS-filter view (include my-goat / introsp /
    any other project). Composes with ``--all``.

    Issue column: ``#N`` = registered in ``~/.eps-autonomous``; ``~#N`` =
    NOT registered but the cwd is the ``issue-N`` worktree (a superseded /
    zombie driver generation or a never-registered session — attributable,
    but not the registered driver); ``-`` = unmapped."""
    meta = _load_session_meta()
    # Session -> issue mapping covers BOTH autonomous (`--auto`) and manual
    # `spawn-issue` sessions. Sessions not spawned by `spawn_session.py`
    # (e.g. `/my-goat`) have no entry and render with a blank issue column.
    issue_map = _load_session_issue_map()
    summary_cache = _load_summary_cache()
    all_dirs = getattr(args, "all_dirs", False)

    if getattr(args, "all", False):
        live = _live_session_ids()
        rows = [
            (
                sid,
                "live" if sid in live else "stopped",
                m.get("startedBy", "?"),
                _dir_label(m.get("path")),
                m.get("savedAt", 0) or 0,
                _issue_cell(issue_map.get(sid), m.get("path")),
            )
            for sid, m in meta.items()
        ]
        if not all_dirs:
            rows = [r for r in rows if _is_eps_dir_label(r[3])]
        # Live sessions first, then newest-saved first within each group.
        rows.sort(key=lambda r: (r[1] != "live", -r[4]))
        if not rows:
            scope = "all dirs" if all_dirs else "EPS dirs"
            print(f"(no sessions in sessions.json for {scope}; pass --all-dirs to widen)")
            return
        print(f"{'session id':<28}  {'state':<8}  {'started_by':<10}  {'issue':<6}  dir")
        for sid, state, started_by, dir_label, _ts, issue_cell in rows:
            print(f"{sid[:26]:<28}  {state:<8}  {started_by:<10}  {issue_cell:<6}  {dir_label}")
        scope_note = " (all dirs)" if all_dirs else " (EPS only; --all-dirs to widen)"
        live_count = sum(1 for r in rows if r[1] == "live")
        print(
            f"\n{len(rows)} session(s){scope_note}, "
            f"{live_count} live. Resume one: happy resume <id-prefix>"
        )
        return

    resp = post("/list", {})
    children = resp.get("children", [])
    if not children:
        print("(no active Happy sessions)")
        return
    # Build the (potentially filtered) row list before printing so the
    # "no rows" branch can give an informative scope-note.
    rendered_rows: list[tuple[str, int | str, str, str, str, str]] = []
    for c in children:
        sid = c.get("happySessionId", "?")
        m = meta.get(sid, {})
        dir_label = _dir_label(m.get("path"))
        if not all_dirs and not _is_eps_dir_label(dir_label):
            continue
        state = m.get("lifecycleState", "?")
        issue = issue_map.get(sid)
        # Progress lookup is per-row in-process — a single broken row must NOT
        # crash the whole table (visible placeholder per row instead). The
        # helper itself catches its own internal failures; this outer guard
        # catches anything truly unexpected (e.g. an interpreter-level error).
        if issue is None:
            progress_cell = ""
        else:
            try:
                cache_entry = summary_cache.get(sid) or {}
                cache_summary = (
                    cache_entry.get("summary") if isinstance(cache_entry, dict) else None
                )
                progress_cell = _format_progress_cell(
                    issue, cache_summary=cache_summary if isinstance(cache_summary, str) else None
                )
            except Exception as e:
                progress_cell = f"<row error: {type(e).__name__}>"
        # Unregistered rows still get attributed via their issue-worktree cwd
        # (`~#N`); progress stays blank for those — the task's progress already
        # renders on the REGISTERED row, and a `~#N` row is by definition a
        # superseded/zombie generation, not the live driver.
        rendered_rows.append(
            (
                sid,
                c.get("pid", "?"),
                state,
                dir_label,
                _issue_cell(issue, m.get("path")),
                progress_cell,
            )
        )

    if not rendered_rows:
        scope = "all dirs" if all_dirs else "EPS dirs"
        print(f"({len(children)} active session(s), none in {scope}; pass --all-dirs to widen)")
        return

    print(
        f"{'session id':<28}  {'pid':>8}  {'state':<10}  {'issue':<6}  "
        f"{'progress':<{_PROGRESS_CELL_MAX}}  dir"
    )
    for sid, pid, state, dir_label, issue_cell, progress_cell in rendered_rows:
        print(
            f"{sid[:26]:<28}  {pid:>8}  {state:<10}  {issue_cell:<6}  "
            f"{progress_cell:<{_PROGRESS_CELL_MAX}}  {dir_label}"
        )
    scope_note = " (all dirs)" if all_dirs else " (EPS only; --all-dirs to widen)"
    print(
        f"\n{len(rendered_rows)} active session(s){scope_note}. "
        f"Resume one: happy resume <id-prefix>"
    )


def cmd_stop(args: argparse.Namespace) -> None:
    """Stop a Happy session by id; degrade usefully on a daemon-untracked sid (#903).

    For an issue-mapped OPERATOR stop, posts a ``deliberate-stop``
    breadcrumb (structured ``epm:progress`` note) on the owning task
    BEFORE the stop RPC, so a later exit-137/143 diagnosis can attribute
    the kill (failure_patterns.md § kill-source verification; #779/#902).
    Watcher-sourced stops (``--stop-source watcher``) post NOTHING — the
    watcher keeps its own registry/sidecar evidence, and an auto-post
    here would manufacture false operator attributions plus unsentineled
    notes that reset staleness clocks. The post runs in a daemon thread
    with a hard join timeout (:data:`STOP_BREADCRUMB_JOIN_TIMEOUT_S`) so
    a wedged workflow flock can never hang the stop (fail-soft: WARN +
    proceed on any failure or timeout).
    """
    if args.stop_source == "operator":
        try:  # fail-soft: the WHOLE mapped branch (map load + post)
            issue = _load_session_issue_map().get(args.session_id)
            if issue is not None:
                note = (
                    f"deliberate-stop pid=n/a target=happy-session:{args.session_id} "
                    f"reason={args.reason}"
                )
                # Exceptions inside the daemon thread are captured into a
                # mutable cell and WARNed after the join (they cannot
                # propagate across threads to the outer try).
                exc_cell: list[BaseException] = []

                def _post() -> None:
                    try:
                        from explore_persona_space.task_workflow import post_event

                        post_event(issue, "epm:progress", by="spawn_session-stop", note=note)
                    except BaseException as exc:  # loud via exc_cell — never silent
                        exc_cell.append(exc)

                t = threading.Thread(target=_post, daemon=True)
                t.start()
                t.join(timeout=STOP_BREADCRUMB_JOIN_TIMEOUT_S)
                if t.is_alive():
                    print(
                        f"WARN: deliberate-stop breadcrumb on #{issue} still posting after "
                        f"{STOP_BREADCRUMB_JOIN_TIMEOUT_S:g}s (wedged lock?); proceeding "
                        f"with stop",
                        file=sys.stderr,
                    )
                elif exc_cell:
                    print(
                        f"WARN: deliberate-stop breadcrumb failed: {exc_cell[0]!r}",
                        file=sys.stderr,
                    )
                else:
                    print(f"Posted deliberate-stop breadcrumb on #{issue}")
        except Exception as exc:  # fail-soft side channel: never block the stop
            print(f"WARN: deliberate-stop breadcrumb failed: {exc!r}", file=sys.stderr)
    resp = post("/stop-session", {"sessionId": args.session_id})
    if resp.get("success"):
        print(f"Stopped session {args.session_id}")
        return
    _stop_fallback(args.session_id, resp, kill=bool(getattr(args, "kill", False)))


def _stop_fallback(sid: str, resp: dict, *, kill: bool) -> None:
    """Failure path of :func:`cmd_stop` (#903): resolve a daemon-untracked
    session id to its live happy node wrapper pid via the ``~/.happy/logs``
    reverse map and either report a structured kill-by-pid recipe or — under
    ``kill=True`` — SIGTERM the pid after a stacked identity re-verification
    (comm + happy-wrapper cmdline signature + not-the-daemon-pid; ambiguity
    always refuses to the report-only recipe, never a kill — the
    kill-before-relaunch ownership discipline,
    ``.claude/rules/crash-fix-rounds.md``). SIGKILL escalation stays manual
    by design."""
    if sid in _live_session_ids():
        sys.exit(
            f"stop failed for DAEMON-TRACKED session {sid}: {resp!r} — the daemon "
            f"knows this session but refused the stop; retry once, then check "
            f"~/.happy/logs/ for the daemon-side error."
        )
    import session_resolver  # lazy: session_resolver imports spawn_session at top level

    pid = session_resolver.find_node_pid_for_session(sid)
    if pid is None:
        sys.exit(
            f"stop failed: session {sid} is UNKNOWN to the Happy daemon "
            f"(daemon-untracked) and no live happy node references it in "
            f"~/.happy/logs/. If you know the wrapper pid: verify ownership first "
            f"(`ps -o pid,lstart,cmd -p <pid>`; `ls -l /proc/<pid>/cwd`), then "
            f"`kill -TERM <pid>`, wait ~10s, re-check `ps -p <pid>`. "
            f"Raw daemon reply: {resp!r}"
        )
    if not kill:
        sys.exit(
            f"stop failed: session {sid} is daemon-untracked, but live happy node "
            f"pid {pid} references it (~/.happy/logs reverse map). Re-run with "
            f"--kill to SIGTERM it, or manually: verify `ps -o pid,lstart,cmd -p {pid}` "
            f"then `kill -TERM {pid}`. Raw daemon reply: {resp!r}"
        )
    # --kill identity binding (#903 round-1 critique Must-Fix): comm alone is
    # NOT ownership on a shared VM full of unrelated node processes (the Happy
    # daemon, the eps-dashboard `next start`, every other wrapper), and the
    # resolver's log scan can only return a RECYCLED pid for a genuinely dead
    # session. Three stacked checks, each refusing to the report-only recipe
    # on mismatch (never a kill on ambiguity — crash-fix-rounds.md
    # § Kill-before-relaunch):
    comm = session_resolver._read_proc_comm(pid)
    if comm != "node":
        sys.exit(
            f"refusing --kill: pid {pid} comm={comm!r} != 'node' (pid may have been "
            f"reused). Verify manually: ps -o pid,lstart,cmd -p {pid}; kill -TERM {pid}"
        )
    cmdline = session_resolver._read_proc_cmdline(pid) or ""
    if "happy" not in cmdline:
        # The happy-wrapper signature: `node .../happy/dist/index.mjs claude ...`.
        sys.exit(
            f"refusing --kill: pid {pid} cmdline {cmdline!r} lacks the happy-wrapper "
            f"signature (pid likely recycled to an unrelated node process). "
            f"Verify manually: ps -o pid,lstart,cmd -p {pid}"
        )
    # Never signal the Happy daemon itself (its log sits in the same dir; the
    # `-daemon.log` suffix is regex-excluded by the resolver, but this is the
    # last-line belt): read ~/.happy/daemon.state.json's pid, refuse on match.
    daemon_pid = session_resolver._happy_daemon_pid()
    if daemon_pid is not None and pid == daemon_pid:
        sys.exit(f"refusing --kill: pid {pid} is the Happy DAEMON pid; wrong resolution.")
    claude_pid = session_resolver.resolve_claude_pid(pid)  # best-effort, pre-kill (may be None)
    os.kill(pid, signal.SIGTERM)
    for _ in range(20):  # ~10s @ 0.5s
        time.sleep(0.5)
        # Module-level seam: the resolver's `_pid_alive` is the ONE liveness
        # probe (tests monkeypatch it + time.sleep) — no inline /proc check.
        if not session_resolver._pid_alive(pid):
            survivor = ""
            if claude_pid is not None and session_resolver._pid_alive(claude_pid):
                survivor = (
                    f" WARNING: inner claude pid {claude_pid} still alive — the "
                    f"wrapper's SIGTERM cleanup may have failed; verify/kill manually."
                )
            print(
                f"Stopped daemon-untracked session {sid} via SIGTERM to node pid {pid}.{survivor}"
            )
            return
    sys.exit(
        f"SIGTERM sent to pid {pid} but it survived ~10s; escalate manually after "
        f"re-verifying: kill -KILL {pid}"
    )


def resolve_session_for_issue(
    issue: int,
    *,
    registry_dir: Path | None = None,
    live_ids: set[str] | None = None,
) -> str | None:
    """Look up the Happy session id driving issue ``issue``.

    Picks the LIVE session if one is registered for this issue; if none of
    the registered sessions are live, falls back to the most-recently spawned
    one (so a JUST-stopped or daemon-list-flaky case still returns something
    usable for ``happy resume``).

    Returns the happy session id, or None if no entry exists for this issue.

    Pure-ish: ``registry_dir`` and ``live_ids`` are injectable so the unit
    tests don't have to touch the real registry or daemon."""
    reg = registry_dir if registry_dir is not None else AUTONOMOUS_REGISTRY_DIR
    candidates: list[tuple[float, str]] = []  # (spawned_at, sid)
    if reg.is_dir():
        for prefix in (
            f"issue-{issue}.json",
            f"manual-issue-{issue}.json",
            f"campaign-{issue}.json",
        ):
            path = reg / prefix
            if not path.is_file():
                continue
            try:
                entry = json.loads(path.read_text())
            except (json.JSONDecodeError, OSError):
                continue
            sid = entry.get("happy_session_id")
            ts = entry.get("spawned_at", 0.0)
            if not isinstance(sid, str):
                continue
            if not isinstance(ts, int | float):
                ts = 0.0
            candidates.append((float(ts), sid))
    if not candidates:
        return None
    live = live_ids if live_ids is not None else _live_session_ids()
    live_candidates = [c for c in candidates if c[1] in live]
    pool = live_candidates or candidates
    pool.sort(reverse=True)  # newest spawned_at first
    return pool[0][1]


def cmd_resume_issue(args: argparse.Namespace) -> None:
    """Print (or exec) the ``happy resume <id>`` command for issue ``--issue N``.

    Looks up the session id via :func:`resolve_session_for_issue`. With
    ``--print`` (default), prints the command so the caller can decide to run
    it (alias-friendly). With ``--exec``, replaces the current process with
    ``happy resume <id>`` (so the user lands directly in the resumed session).
    Fails loud if no session is registered for the issue."""
    sid = resolve_session_for_issue(args.issue)
    if sid is None:
        sys.exit(
            f"no Happy session registered for issue #{args.issue}. "
            f"Spawn one first: uv run python scripts/spawn_session.py spawn-issue "
            f"--issue {args.issue}"
        )
    cmd = ["happy", "resume", sid]
    if args.exec:
        # Replace this process so the user lands directly in the Happy TTY.
        os.execvp(cmd[0], cmd)
        return  # unreachable; satisfies lints
    # Default: print the command so the caller (a shell alias) can `eval` /
    # exec it themselves, OR a human can copy-paste it.
    print(" ".join(cmd))


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_pm = sub.add_parser("spawn-pm", help="spawn a Happy session for the PM persona")
    _add_claude_session_args(p_pm)
    p_pm.set_defaults(fn=cmd_spawn_pm)

    p_issue = sub.add_parser("spawn-issue", help="spawn a Happy session for issue #N")
    p_issue.add_argument("--issue", type=int, required=True)
    p_issue.add_argument(
        "--initial-prompt",
        default=None,
        help=(
            "Boot the session with this prompt already in place, in "
            "bypassPermissions mode (no human at the keyboard to confirm tool calls)."
        ),
    )
    p_issue.add_argument(
        "--auto",
        action="store_true",
        help=(
            "Shorthand for --initial-prompt '/issue <N>' (the full /issue skill on "
            "initial fire; arms the recurring /issue-tick <N> cron at Step 6d.2)."
        ),
    )
    p_issue.add_argument(
        "--auto-approve-gpu-hours",
        type=float,
        default=100.0,
        help=(
            "Autonomous sessions auto-approve a plan whose estimated GPU-hours "
            "is <= this value and park at plan_pending above it. Default 100."
        ),
    )
    _add_claude_session_args(p_issue)
    p_issue.set_defaults(fn=cmd_spawn_issue)

    p_campaign = sub.add_parser(
        "spawn-campaign",
        help=(
            "spawn the dedicated autonomous session driving campaign #N "
            "(/campaign <N>; requires kind: campaign at status approved — task #586)"
        ),
    )
    p_campaign.add_argument("--issue", type=int, required=True)
    # Cap defaults resolve at runtime from the campaign_state module
    # constants (single source — see _campaign_defaults); None = unset here.
    p_campaign.add_argument(
        "--budget-gpu-hours",
        type=float,
        default=None,
        help=(
            "total GPU-hour budget across all campaign children "
            "(default: campaign_state.DEFAULT_GPU_HOURS_TOTAL)"
        ),
    )
    p_campaign.add_argument(
        "--max-concurrent",
        type=int,
        default=None,
        help=(
            "max children in flight at once "
            "(default: campaign_state.DEFAULT_MAX_CONCURRENT_CHILDREN)"
        ),
    )
    p_campaign.add_argument(
        "--per-child-cap",
        type=float,
        default=None,
        help=(
            "per-child GPU-hour auto-approve cap, exported as "
            "EPM_PLAN_AUTOAPPROVE_GPU_HOURS and re-passed to each "
            "`spawn-issue --auto` child "
            "(default: campaign_state.DEFAULT_PER_CHILD_GPU_HOURS_CAP)"
        ),
    )
    _add_claude_session_args(p_campaign)
    p_campaign.set_defaults(fn=cmd_spawn_campaign)

    p_reg = sub.add_parser(
        "register-current",
        help=(
            "re-register an EXISTING live session as the driver of issue #N — use when "
            "reviving a parked/terminal task (same-issue follow-up loop) so the "
            "crash-recovery watcher sees the revival immediately (#472)"
        ),
    )
    p_reg.add_argument("--issue", type=int, required=True)
    p_reg.add_argument(
        "--session-id",
        default=None,
        help=(
            "Happy session id to register (validated live against the daemon). "
            "Omit to infer from the process ancestry — works when invoked from "
            "inside the session itself."
        ),
    )
    p_reg.add_argument(
        "--mode",
        choices=("auto", "manual", "campaign"),
        default=None,
        help=(
            "Registration kind: 'auto' writes issue-<N>.json (watcher may auto-respawn), "
            "'manual' writes manual-issue-<N>.json (alert-only), 'campaign' writes "
            "campaign-<N>.json (campaign pass may respawn; caps preserved from any prior "
            "entry). Default: inferred from EPM_CAMPAIGN_SESSION=1 -> campaign, "
            "EPM_AUTONOMOUS_SESSION=1 -> auto, else manual."
        ),
    )
    p_reg.add_argument(
        "--auto-approve-gpu-hours",
        type=float,
        default=None,
        help=(
            "GPU-hour auto-approve cap recorded in an auto-mode entry (the watcher "
            "re-passes it on respawn). Default: EPM_PLAN_AUTOAPPROVE_GPU_HOURS or 100."
        ),
    )
    p_reg.set_defaults(fn=cmd_register_current)

    p_unreg = sub.add_parser(
        "unregister",
        help=(
            "remove this session's issue/manual/campaign registration file(s) — the "
            "inverse of register-current, for collision-yield and deliberate-stop "
            "paths (never hand-rm ~/.eps-autonomous files). Sid-matched by default: "
            "only files recording the calling session's Happy id (ancestry-inferred "
            "or --session-id) are removed, so a yielding duplicate can never delete "
            "the true owner's entry (a KEPT-SID-MISMATCH line is the guard working, "
            "not a bug). Third-party cleanup of a DEAD session's file: "
            "`unregister --issue N --session-id <dead-sid>` (removes only entries "
            "recording that sid; no liveness check), or `unregister --force "
            "--issue N` for unconditional operator cleanup. Takeover sentinels "
            "(*.paused-takeover-*) are never touched."
        ),
    )
    p_unreg.add_argument(
        "--issue",
        type=int,
        default=None,
        help=(
            "issue number whose registration file(s) to remove (exact filenames "
            "issue-N.json / manual-issue-N.json / campaign-N.json)"
        ),
    )
    p_unreg.add_argument(
        "--session-id",
        default=None,
        help=(
            "only remove entries recording this Happy session id (works for a DEAD "
            "sid — third-party cleanup validates against the FILE, not the daemon); "
            "omit to infer this session's own id from the process ancestry. With no "
            "--issue, scans all registrations for this sid."
        ),
    )
    p_unreg.add_argument(
        "--kind",
        choices=("auto", "manual", "campaign"),
        default=None,
        help="narrow to one registration kind (default: all three)",
    )
    p_unreg.add_argument(
        "--force",
        action="store_true",
        help="skip the sid match (requires --issue; mutually exclusive with --session-id)",
    )
    p_unreg.add_argument(
        "--reason",
        default=None,
        help=(
            "free-form audit string echoed in each output line (transcript "
            "breadcrumb; no task marker is posted)"
        ),
    )
    p_unreg.set_defaults(fn=cmd_unregister)

    p_reg_pm = sub.add_parser(
        "register-pm",
        help=(
            "register an EXISTING live session as the PM session so the watcher's "
            "zombie-wrapper pass never auto-stops it (spawn-pm registers "
            "automatically; this covers PM sessions opened any other way)"
        ),
    )
    p_reg_pm.add_argument(
        "--session-id",
        default=None,
        help=(
            "Happy session id to register (validated live against the daemon). "
            "Omit to infer from the process ancestry — works when invoked from "
            "inside the PM session itself (the /pm skill does this at bootstrap)."
        ),
    )
    p_reg_pm.set_defaults(fn=cmd_register_pm)

    p_list = sub.add_parser("list", help="list active Happy sessions (cwd + state)")
    p_list.add_argument(
        "--all",
        action="store_true",
        help="include stopped/historical sessions from ~/.happy/sessions.json (newest first)",
    )
    p_list.add_argument(
        "--all-dirs",
        action="store_true",
        help=(
            "Include non-EPS sessions (my-goat, introsp, etc.). By default the "
            "list is filtered to EPS-only (the repo root and its worktrees)."
        ),
    )
    p_list.set_defaults(fn=cmd_list)

    p_stop = sub.add_parser("stop", help="stop a Happy session by id")
    p_stop.add_argument("--session-id", required=True)
    p_stop.add_argument(
        "--reason",
        default="operator stop via spawn_session.py stop",
        help="one-line reason recorded in the deliberate-stop breadcrumb",
    )
    p_stop.add_argument(
        "--stop-source",
        choices=("operator", "watcher"),
        default="operator",
        help=(
            "watcher-driven stops post no breadcrumb (the watcher keeps its "
            "own registry/sidecar evidence trail)"
        ),
    )
    p_stop.add_argument(
        "--kill",
        action="store_true",
        help=(
            "if the sid is daemon-untracked but resolvable to a live happy node "
            "pid, SIGTERM that pid (comm re-verified first; no automatic SIGKILL)"
        ),
    )
    p_stop.set_defaults(fn=cmd_stop)

    p_resume = sub.add_parser(
        "resume-issue",
        help="print (or exec) `happy resume <id>` for the session driving issue #N",
    )
    p_resume.add_argument("--issue", type=int, required=True)
    p_resume.add_argument(
        "--exec",
        action="store_true",
        help="Replace this process with `happy resume <id>` instead of printing it.",
    )
    p_resume.set_defaults(fn=cmd_resume_issue)

    args = parser.parse_args(argv)
    args.fn(args)


if __name__ == "__main__":
    main()
