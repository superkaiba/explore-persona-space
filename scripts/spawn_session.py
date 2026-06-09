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
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

HAPPY_HOME = Path.home() / ".happy"
DAEMON_STATE = HAPPY_HOME / "daemon.state.json"
SESSIONS_JSON = HAPPY_HOME / "sessions.json"
PROJECT_ROOT = Path(__file__).resolve().parent.parent
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
AUTONOMOUS_REGISTRY_DIR = Path.home() / ".eps-autonomous"


def _register_autonomous_session(
    issue: int, session_id: str, cwd: str, auto_approve_gpu_hours: float
) -> None:
    """Record an autonomous issue session so the watcher can resurrect it.

    Written on every `spawn-issue --auto` (initial spawn AND watcher re-spawn),
    overwriting any prior entry with the fresh Happy session id and ``missed=0``.
    RAISES ``OSError`` on write failure — the caller MUST treat a live `--auto`
    session that could not be registered as unsafe (an untracked live session is
    invisible to the watcher and risks a duplicate re-spawn), and stop it.
    Writes atomically (temp file + rename) so the watcher never reads a partial
    JSON entry."""
    AUTONOMOUS_REGISTRY_DIR.mkdir(parents=True, exist_ok=True)
    entry = {
        "issue": issue,
        "happy_session_id": session_id,
        "cwd": cwd,
        "auto_approve_gpu_hours": auto_approve_gpu_hours,
        "spawned_at": time.time(),
        "missed": 0,
    }
    dest = AUTONOMOUS_REGISTRY_DIR / f"issue-{issue}.json"
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


def _load_session_issue_map() -> dict[str, int]:
    """Return ``{happy_session_id: issue_number}`` from BOTH the autonomous
    (``issue-<N>.json``) and manual (``manual-issue-<N>.json``) registries.

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
    # Enumerate the two known prefixes explicitly rather than `*issue-*.json`
    # — a wildcard glob would scrape any future sibling file (e.g. a hand-
    # added `weird-issue-N.json` debug dump, or another tool's misnamed
    # entry) and silently overwrite legitimate mappings. The watcher's own
    # respawn glob (`issue-*.json`, NO leading `manual-`) deliberately
    # matches only the autonomous prefix; this loader sees both kinds.
    for prefix in ("issue-", "manual-issue-"):
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


def _dir_label(path: str | None) -> str:
    """Short, human-friendly cwd label, annotating per-issue worktrees.

    ``/home/me/explore-persona-space`` -> ``explore-persona-space``;
    a ``.claude/worktrees/issue-<N>`` path gets an ``[issue-<N>]`` tag."""
    if not path:
        return "?"
    home = str(Path.home())
    short = path[len(home) + 1 :] if path.startswith(home + "/") else path
    m = re.search(r"/\.claude/worktrees/(issue-\d+)/?$", path)
    return f"{short}  [{m.group(1)}]" if m else short


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
    surfaces as a clean failure so the caller can safely retry."""
    url = f"http://127.0.0.1:{daemon_port()}{path}"
    payload = json.dumps(body).encode()
    req = urllib.request.Request(
        url,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    timeout = SPAWN_SESSION_TIMEOUT_S if path == "/spawn-session" else DEFAULT_TIMEOUT_S
    spawn_started_at = time.time() if path == "/spawn-session" else None
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read())
    except urllib.error.HTTPError as e:
        try:
            err_body = json.loads(e.read())
        except Exception:
            err_body = {"raw": str(e)}
        sys.exit(f"Happy daemon {path} returned HTTP {e.code}: {err_body}")
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


def _live_session_ids() -> set[str]:
    """Best-effort set of session ids the daemon is actively tracking.

    Returns an empty set if the daemon is unreachable, so ``list --all`` still
    works (it falls back to showing every known session as ``stopped``)."""
    try:
        url = f"http://127.0.0.1:{daemon_port()}/list"
        req = urllib.request.Request(
            url, data=b"{}", headers={"Content-Type": "application/json"}, method="POST"
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read())
    except (urllib.error.URLError, OSError, SystemExit, json.JSONDecodeError):
        return set()
    return {c.get("happySessionId") for c in data.get("children", [])}


def cmd_spawn_pm(_: argparse.Namespace) -> None:
    """Spawn a session intended to host the PM persona. The session opens
    cwd=<repo root> so the user sees a familiar project. The PM persona is
    then loaded interactively by the user typing ``/pm``."""
    resp = post(
        "/spawn-session",
        {"directory": str(PROJECT_ROOT), "agent": "claude"},
    )
    if not resp.get("success"):
        sys.exit(f"spawn failed: {resp}")
    print(
        f"PM session spawned: {resp['sessionId']}\n"
        f"  cwd: {PROJECT_ROOT}\n"
        f"Open it in Happy on your phone and type ``/pm`` to load the PM persona."
    )


def cmd_spawn_issue(args: argparse.Namespace) -> None:
    """Spawn a session for issue ``--issue N``. The session opens cwd=<repo root>
    by default, OR cwd=<.claude/worktrees/issue-N> if such a worktree exists
    (so the session is git-isolated to that issue's branch).

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

    body: dict[str, object] = {"directory": str(cwd), "agent": "claude"}
    if args.initial_prompt:
        prompt = args.initial_prompt
    elif args.auto:
        # Cold start (and cold respawn via `autonomous_session_watch._respawn`)
        # boots the FULL `/issue <N>` skill once. The full skill arms an
        # in-session cron at Step 6d.2 that fires the lightweight
        # `/issue-tick <N>` skill every 20 minutes — that recurring tick is
        # the new driver, NOT a `/loop`. The old `/loop 10m /issue <N>`
        # shape re-loaded the 44K-token /issue SKILL.md on every idle tick;
        # the new shape loads it exactly once per session. (20 min not
        # 10 min because the Anthropic prompt cache TTL is 5 min — a 10-min
        # cadence guarantees a cold prefix every fire, so doubling the
        # interval halves the tick count without the cache cost changing.)
        prompt = f"/issue {issue}"
    else:
        prompt = None
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
        body["claudeArgs"] = ["--dangerously-skip-permissions"]

    resp = post("/spawn-session", body)
    if not resp.get("success"):
        sys.exit(f"spawn failed: {resp}")
    print(f"Issue #{issue} session spawned: {resp['sessionId']}")
    print(f"  cwd: {cwd_note}")
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
                    issue, resp["sessionId"], str(cwd), args.auto_approve_gpu_hours
                )
                print(f"  registered for crash-recovery watch: issue-{issue}.json")
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
                try:
                    stop_resp = post("/stop-session", {"sessionId": resp["sessionId"]})
                    stopped = bool(stop_resp.get("success"))
                except SystemExit:
                    stopped = False
                if not stopped:
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
    any other project). Composes with ``--all``."""
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
                issue_map.get(sid),
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
        for sid, state, started_by, dir_label, _ts, issue in rows:
            issue_cell = f"#{issue}" if issue is not None else "-"
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
    rendered_rows: list[tuple[str, int | str, str, str, str | None, str]] = []
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
        rendered_rows.append((sid, c.get("pid", "?"), state, dir_label, issue, progress_cell))

    if not rendered_rows:
        scope = "all dirs" if all_dirs else "EPS dirs"
        print(f"({len(children)} active session(s), none in {scope}; pass --all-dirs to widen)")
        return

    print(
        f"{'session id':<28}  {'pid':>8}  {'state':<10}  {'issue':<6}  "
        f"{'progress':<{_PROGRESS_CELL_MAX}}  dir"
    )
    for sid, pid, state, dir_label, issue, progress_cell in rendered_rows:
        issue_cell = f"#{issue}" if issue is not None else "-"
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
    """Stop a Happy session by id."""
    resp = post("/stop-session", {"sessionId": args.session_id})
    if not resp.get("success"):
        sys.exit(f"stop failed: {resp}")
    print(f"Stopped session {args.session_id}")


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
        for prefix in (f"issue-{issue}.json", f"manual-issue-{issue}.json"):
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
    p_issue.set_defaults(fn=cmd_spawn_issue)

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
