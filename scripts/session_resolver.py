"""Resolve a live Happy session node-pid to its Claude Code transcript + issue.

A live Happy "session" is a `node` happy-wrapper process (visible via
``spawn_session.py list`` or the daemon's ``/list`` RPC) whose direct child is
the ``claude`` Claude-Code subprocess. The transcript path is owned by the
Claude subprocess but is NOT kept as an open file descriptor — only
``~/.claude/history.jsonl`` and the projects dir are held open.

Resolution order (most authoritative first):

1. **Happy log file** ``~/.happy/logs/*pid-<node_pid>.log`` — each happy node
   writes its log path keyed on its pid, and the log contains every
   ``"transcript_path": "..."`` the Claude subprocess has reported via its
   SessionStart hook. The LAST such line is the current transcript. This is
   the authoritative mapping (Claude can change `sessionId` mid-life by
   reloading; the log captures every transition).

2. **Filesystem fallback** — walk happy-node pid's descendants for the
   ``comm == "claude"`` process, read ``/proc/<claude_pid>/cwd``, derive the
   project slug (``re.sub('[^a-zA-Z0-9]', '-', cwd)``), and pick the newest
   ``*.jsonl`` in ``~/.claude/projects/<slug>/`` whose first ~50 entries
   contain a ``/issue N`` prompt or that matches an autonomous-loop pattern.
   Used when the happy log is missing (rotated away) or unreadable.

Both paths fail-loud: on unresolvable, they return ``None`` and the caller
gets an explicit ``reason`` field — never a fabricated transcript path.

Issue extraction:

``resolve_issue(node_pid)`` reads the resolved transcript's first ~50 entries
looking for a ``/issue <N>`` prompt (raw or wrapped in a ``/loop`` command).
Returns the int issue number or None. Idle / non-issue sessions return None.

CLI::

    uv run python scripts/session_resolver.py --node-pid 1637665
    uv run python scripts/session_resolver.py --backfill

The ``--backfill`` mode walks every LIVE Happy session whose cwd resolves to
explore-persona-space (incl. worktrees), and for any that is NOT already in
``~/.eps-autonomous/{issue,manual-issue}-<N>.json``, writes a
``manual-issue-<N>.json`` entry with ``mode: backfilled``. Idempotent.

The resolver is also importable; ``session_summarize.py`` and the EPS-only
filter in ``spawn_session.py list`` reuse it.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path

# Make this script importable when scripts/ is on PYTHONPATH (matches the test
# pattern in tests/test_spawn_session_list_enrichment.py).
SCRIPTS_DIR = Path(__file__).resolve().parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

# Reused from spawn_session: the canonical registry dir + the merger.
import spawn_session  # noqa: E402

HAPPY_LOGS_DIR = Path.home() / ".happy" / "logs"
CLAUDE_PROJECTS_DIR = Path.home() / ".claude" / "projects"

# The two repo roots that count as "EPS sessions": the canonical project root
# and any worktree under it. We resolve symlinks so a session inside a
# worktree resolves cleanly.
_EPS_ROOT_NAME = "explore-persona-space"

# Maximum lines of a transcript to scan when extracting the /issue N prompt.
# The opening prompt always lands in the first few entries; 50 is generous.
_PROMPT_SCAN_LINES = 50

# Maximum age (seconds) of a happy log file we consider "the current life of
# this happy node". A log file older than this for the SAME pid is probably
# left over from a previous incarnation that crashed (pids do get re-used).
_HAPPY_LOG_MAX_AGE_S = 30 * 86400


# ── data ────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class ResolveResult:
    """One node_pid's resolution snapshot. Fields default to None on miss; the
    ``reason`` field always names WHY a None landed (never an unexplained miss)."""

    node_pid: int
    claude_pid: int | None
    cwd: str | None
    transcript: str | None
    issue: int | None
    reason: str | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "node_pid": self.node_pid,
            "claude_pid": self.claude_pid,
            "cwd": self.cwd,
            "transcript": self.transcript,
            "issue": self.issue,
            "reason": self.reason,
        }


# ── pure helpers (unit-testable, no /proc, no I/O) ─────────────────────────


def derive_project_slug(cwd: str) -> str:
    """Compute the Claude-projects subdir name for a cwd.

    Claude Code stores each project's transcripts under
    ``~/.claude/projects/<slug>/`` where ``<slug>`` is the cwd with every
    non-alphanumeric character replaced by ``-``. A leading ``/`` becomes a
    leading ``-`` and consecutive non-alphanumerics map to consecutive ``-``s
    (so ``/foo/.claude/x`` => ``-foo--claude-x``). This rule is observable
    empirically against the live ``~/.claude/projects/`` tree."""
    return re.sub(r"[^a-zA-Z0-9]", "-", cwd)


def is_eps_cwd(cwd: str | None) -> bool:
    """True iff this cwd is the EPS project root or one of its worktrees.

    Treats ``~/explore-persona-space`` and any ``~/explore-persona-space/...``
    path (including ``.claude/worktrees/...``) as EPS. Used by the default
    ``cmd_list`` filter to drop sessions for other projects (my-goat, introsp,
    cbw-site, etc.)."""
    if not cwd:
        return False
    parts = Path(cwd).parts
    return _EPS_ROOT_NAME in parts


def extract_issue_from_text(text: str) -> int | None:
    """Find the first ``/issue <N>`` reference in a chunk of transcript text.

    The transcript shape is JSON-per-line; we scan the raw text because both
    the user's literal ``/issue 488`` and the wrapped ``/loop 10m /issue 488``
    cases land as JSON-escaped substrings. A bare ``issue 488`` (no slash) is
    NOT a match — that would false-positive on prose."""
    m = re.search(r"/issue\s+(\d+)\b", text)
    if not m:
        return None
    try:
        return int(m.group(1))
    except ValueError:
        return None


def extract_transcript_from_happy_log(log_text: str) -> str | None:
    """Return the LAST ``transcript_path`` reference in a happy log file.

    Each Claude SessionStart hook appends one
    ``"transcript_path": "/path/to/<uuid>.jsonl"`` line; the most recent line
    is the current transcript (Claude may switch session UUIDs mid-life)."""
    matches = re.findall(r'"transcript_path"\s*:\s*"([^"]+)"', log_text)
    return matches[-1] if matches else None


# ── /proc helpers ──────────────────────────────────────────────────────────


def _pid_alive(pid: int) -> bool:
    """True iff ``/proc/<pid>`` exists — the ONE module-level liveness seam
    the #903 stop-fallback death-wait loop (``spawn_session._stop_fallback``)
    and :func:`find_node_pid_for_session` probe through, so tests monkeypatch
    exactly one name."""
    return Path(f"/proc/{pid}").is_dir()


def _read_proc_comm(pid: int) -> str | None:
    """Read ``/proc/<pid>/comm`` (the basename of the executable). Returns
    None if the process is gone or the file is unreadable — these are
    expected races, not bugs."""
    try:
        return Path(f"/proc/{pid}/comm").read_text().strip()
    except (FileNotFoundError, PermissionError, OSError):
        return None


def _read_proc_cmdline(pid: int) -> str | None:
    """Read ``/proc/<pid>/cmdline`` with NUL separators mapped to spaces.
    Returns None if the process is gone or unreadable (expected races) —
    the #903 ``--kill`` identity binding treats that as a refusal."""
    try:
        raw = Path(f"/proc/{pid}/cmdline").read_bytes()
    except (FileNotFoundError, PermissionError, OSError):
        return None
    return raw.replace(b"\x00", b" ").decode(errors="replace").strip()


def _happy_daemon_pid() -> int | None:
    """Pid of the local Happy daemon per ``~/.happy/daemon.state.json``.
    Returns None on ANY read/parse failure — the #903 ``--kill`` path then
    simply skips the daemon-pid refusal check (the comm + cmdline checks
    still bind)."""
    try:
        state = json.loads(spawn_session.DAEMON_STATE.read_text())
    except (OSError, ValueError):
        return None
    pid = state.get("pid") if isinstance(state, dict) else None
    if isinstance(pid, int) and not isinstance(pid, bool):
        return pid
    return None


def _read_proc_cwd(pid: int) -> str | None:
    """Resolve ``/proc/<pid>/cwd`` to its target. Returns None on race / perms."""
    try:
        return os.readlink(f"/proc/{pid}/cwd")
    except (FileNotFoundError, PermissionError, OSError):
        return None


def _proc_children(pid: int) -> list[int]:
    """Return the immediate child PIDs of ``pid`` via ``/proc/.../task/<tid>/children``.

    Aggregates across all task ids (threads), which is how the kernel exposes
    children. Empty list if the proc is gone."""
    out: list[int] = []
    task_dir = Path(f"/proc/{pid}/task")
    try:
        tids = list(task_dir.iterdir())
    except (FileNotFoundError, NotADirectoryError, PermissionError):
        return out
    for tdir in tids:
        try:
            children_text = (tdir / "children").read_text().strip()
        except (FileNotFoundError, PermissionError, OSError):
            continue
        for tok in children_text.split():
            try:
                out.append(int(tok))
            except ValueError:
                continue
    # Dedup while preserving order — a child can appear under multiple tasks.
    seen: set[int] = set()
    ordered: list[int] = []
    for c in out:
        if c not in seen:
            seen.add(c)
            ordered.append(c)
    return ordered


# ── transcript / issue resolution ─────────────────────────────────────────


def resolve_claude_pid(node_pid: int) -> int | None:
    """Walk descendants of ``node_pid`` (the happy `node` process) and return
    the first one whose ``comm == "claude"``.

    The happy node spawns Claude Code as a direct child (we've never seen it
    nested deeper in practice), but we walk recursively as defense in depth
    in case of MCP wrappers or future restructure. Returns None if the
    happy node has no live claude child — common transient state when a
    crashed session is being respawned, or a long-idle session whose claude
    subprocess has been reaped."""
    seen: set[int] = set()
    stack: list[int] = [node_pid]
    while stack:
        p = stack.pop()
        if p in seen:
            continue
        seen.add(p)
        children = _proc_children(p)
        for c in children:
            if c in seen:
                continue
            comm = _read_proc_comm(c)
            if comm == "claude":
                return c
            stack.append(c)
    return None


def _find_happy_log_for_node(node_pid: int, now: float | None = None) -> Path | None:
    """Return the most recent ``~/.happy/logs/*pid-<node_pid>.log`` file.

    A happy node writes one log file per life (pid suffix). pids get re-used,
    so a stale log file for the same pid from a previous incarnation could
    mislead us; we filter on mtime within ``_HAPPY_LOG_MAX_AGE_S`` (30 days)
    and pick the newest. Returns None if no matching log file exists in the
    accepted window."""
    if not HAPPY_LOGS_DIR.is_dir():
        return None
    candidates: list[tuple[float, Path]] = []
    cutoff = (now if now is not None else time.time()) - _HAPPY_LOG_MAX_AGE_S
    suffix = f"-pid-{node_pid}.log"
    for path in HAPPY_LOGS_DIR.iterdir():
        if not path.name.endswith(suffix):
            continue
        try:
            mt = path.stat().st_mtime
        except OSError:
            continue
        if mt < cutoff:
            continue
        candidates.append((mt, path))
    if not candidates:
        return None
    candidates.sort(reverse=True)
    return candidates[0][1]


# Bytes of a happy log head to scan for a session id. The sid appears at
# session creation, near the head (live probe 2026-07-02: first occurrence at
# byte ~50,588 of the newest log) — 4 MB is a comfortable bound that still
# avoids reading a multi-hundred-MB log wholesale. Allowed-deviation tunable.
_LOG_SID_SCAN_BYTES = 4_000_000

# Degenerate-sid floor. Real Happy session ids are ~25-char cuids; an EMPTY
# sid (an unset "$SID" in a caller script) or a short fragment would
# substring-match essentially every log head and resolve to an arbitrary live
# wrapper — the wrong-kill vector under `stop --kill`. 8 is a generous floor:
# well below real sid length, well above accidental fragments.
_MIN_SID_LEN = 8


def find_node_pid_for_session(sid: str, now: float | None = None) -> int | None:
    """Reverse-map a Happy session id -> LIVE node wrapper pid, daemon-free.

    ``~/.happy/logs/<date>-pid-<node_pid>.log`` embeds the pid in the filename
    and the session id in the content (``"sessionId": "<sid>"``; verified live
    2026-07-02: 304 occurrences in the newest log). For daemon-UNTRACKED
    sessions (#903): scan logs newest-first within ``_HAPPY_LOG_MAX_AGE_S``,
    match the QUOTED form ``"<sid>"`` in the first ``_LOG_SID_SCAN_BYTES``
    (the quotes bind the full id — a sid that is a PREFIX of another id must
    not vouch), parse the pid from the filename, return the first pid that is
    alive. A degenerate sid (empty, or shorter than ``_MIN_SID_LEN``) never
    resolves — a bare-substring match on it would bind every log head and
    SIGTERM an arbitrary live wrapper. ``None`` on any miss (the caller
    degrades to a structured recipe). The ``-pid-(\\d+)\\.log$`` regex
    structurally excludes the daemon's ``...-pid-<pid>-daemon.log``."""
    if len(sid) < _MIN_SID_LEN:
        return None
    if not HAPPY_LOGS_DIR.is_dir():
        return None
    cutoff = (now if now is not None else time.time()) - _HAPPY_LOG_MAX_AGE_S
    candidates: list[tuple[float, int, Path]] = []
    for path in HAPPY_LOGS_DIR.iterdir():
        m = re.search(r"-pid-(\d+)\.log$", path.name)
        if not m:
            continue
        try:
            mt = path.stat().st_mtime
        except OSError:
            continue
        if mt < cutoff:
            continue
        candidates.append((mt, int(m.group(1)), path))
    candidates.sort(reverse=True)
    # Group by pid: only the NEWEST log per pid may vouch for that pid (#903
    # round-1 critique Must-Fix): an OLDER log containing the sid whose pid
    # was since RECYCLED to a different live wrapper (which writes the NEWER
    # log for the same pid, sans sid) must NOT resolve — accepting it is the
    # wrong-kill vector.
    newest_for_pid: dict[int, Path] = {}
    for _mt, pid, path in candidates:  # already newest-first
        newest_for_pid.setdefault(pid, path)
    for _mt, pid, path in candidates:
        if newest_for_pid[pid] != path:
            continue  # an older log for a since-reused pid cannot vouch for it
        try:
            with path.open(errors="replace") as fh:
                head = fh.read(_LOG_SID_SCAN_BYTES)
        except OSError:
            continue
        if f'"{sid}"' not in head:
            continue
        if _pid_alive(pid):
            return pid
    return None


# ── tick-scoped transcript-resolution memo (#1182) ─────────────────────────
# None => caching OFF (the default: every consumer outside an explicit scope
# is byte-identical to pre-#1182 behavior). A fresh dict is installed ONLY
# for the duration of a transcript_resolution_scope() — one watcher tick —
# and dropped on exit, so a cached (transcript, reason) can never survive
# across ticks (transcripts move between ticks; the task-body hard
# constraint). Main-thread-only by contract: the global is unguarded; the
# watcher's sole worker thread (vm-disk-hf-reclaim) never calls this module.
_TRANSCRIPT_MEMO: dict[int, tuple[str | None, str | None]] | None = None


@contextlib.contextmanager
def transcript_resolution_scope():
    """Memoize ``_resolve_transcript_via_happy_log`` per node pid for the
    duration of the scope — one watcher tick (#1182; saves ~138 ms per
    repeated warm resolution). Positive AND negative results are cached (the
    miss walk is the expensive part; every watcher consumer fails toward
    keep/no-fire on a miss, so a within-tick frozen miss defers action by at
    most one 10-min tick — inside the watcher's existing 2-miss debounce
    granularity). Re-entrant: restores the previous memo on exit. Usable as
    a decorator (``@transcript_resolution_scope()``) — each decorated CALL
    gets a fresh memo. NEVER hold a scope across loop iterations in a
    long-lived process."""
    global _TRANSCRIPT_MEMO
    prev = _TRANSCRIPT_MEMO
    _TRANSCRIPT_MEMO = {}
    try:
        yield
    finally:
        _TRANSCRIPT_MEMO = prev


def _resolve_transcript_via_happy_log(node_pid: int) -> tuple[str | None, str | None]:
    """Try the happy-log path. Returns (transcript_path, reason_on_miss).

    Memoized per node_pid while a transcript_resolution_scope() is active;
    uncached (byte-identical legacy behavior) otherwise."""
    memo = _TRANSCRIPT_MEMO
    if memo is not None and node_pid in memo:
        return memo[node_pid]
    result = _resolve_transcript_via_happy_log_uncached(node_pid)
    if memo is not None:
        memo[node_pid] = result
    return result


def _resolve_transcript_via_happy_log_uncached(node_pid: int) -> tuple[str | None, str | None]:
    """Uncached body of ``_resolve_transcript_via_happy_log`` — see the memo
    shim above. Returns (transcript_path, reason_on_miss)."""
    log_path = _find_happy_log_for_node(node_pid)
    if log_path is None:
        return None, "no happy log file for this node pid"
    try:
        text = log_path.read_text(errors="replace")
    except OSError as e:
        return None, f"happy log unreadable: {type(e).__name__}"
    candidate = extract_transcript_from_happy_log(text)
    if candidate is None:
        return None, "happy log has no transcript_path entries"
    if not Path(candidate).is_file():
        # The log may reference a transcript that has since been rotated or
        # deleted. Treat as a miss; the fallback path may still find a usable
        # transcript via slug + newest-jsonl.
        return None, f"happy log transcript_path missing on disk: {candidate}"
    return candidate, None


def _resolve_transcript_via_filesystem(
    claude_pid: int,
) -> tuple[str | None, str | None]:
    """Fallback: slug-derived projects dir, pick newest *.jsonl whose head
    contains an /issue or /loop prompt. Returns (transcript_path, reason_on_miss)."""
    cwd = _read_proc_cwd(claude_pid)
    if cwd is None:
        return None, "claude pid cwd unreadable"
    slug = derive_project_slug(cwd)
    proj_dir = CLAUDE_PROJECTS_DIR / slug
    if not proj_dir.is_dir():
        return None, f"no projects dir at {proj_dir}"
    candidates: list[tuple[float, Path]] = []
    for p in proj_dir.glob("*.jsonl"):
        try:
            mt = p.stat().st_mtime
        except OSError:
            continue
        candidates.append((mt, p))
    if not candidates:
        return None, f"projects dir {proj_dir} has no *.jsonl"
    candidates.sort(reverse=True)
    # Prefer the newest jsonl whose head contains a /issue or /loop prompt;
    # if none match, return the newest unconditionally (idle / non-issue
    # session — caller decides). Stream only the first N lines instead of
    # reading the whole file (transcripts are routinely 10+ MB).
    for _mt, p in candidates:
        try:
            with p.open() as fh:
                head = "".join(next(fh, "") for _ in range(_PROMPT_SCAN_LINES))
        except OSError:
            continue
        if "/issue" in head or "/loop" in head:
            return str(p), None
    return str(candidates[0][1]), None


def resolve_transcript(node_pid: int) -> tuple[str | None, str | None]:
    """Return ``(transcript_path, reason)`` for a happy node pid.

    ``reason`` is None on success and a short human-readable string on miss
    (never silently None on miss). Tries the happy log first (authoritative);
    falls back to slug-derived filesystem scan."""
    transcript, reason = _resolve_transcript_via_happy_log(node_pid)
    if transcript is not None:
        return transcript, None
    # Need the claude pid for the filesystem fallback.
    claude_pid = resolve_claude_pid(node_pid)
    if claude_pid is None:
        return None, f"{reason}; no claude child pid for node {node_pid}"
    transcript, fs_reason = _resolve_transcript_via_filesystem(claude_pid)
    if transcript is not None:
        return transcript, None
    return None, f"{reason}; fallback: {fs_reason}"


def resolve_issue(node_pid: int, transcript: str | None = None) -> int | None:
    """Return the ``/issue N`` driven by this session, or None if not found.

    Reads the first ``_PROMPT_SCAN_LINES`` lines of the transcript and greps
    for ``/issue <N>`` (matches both bare ``/issue 488`` and the wrapped
    ``/loop 10m /issue 488`` autonomous form). If ``transcript`` is not given,
    resolves it from ``node_pid``."""
    if transcript is None:
        transcript, _ = resolve_transcript(node_pid)
    if transcript is None:
        return None
    try:
        with open(transcript) as fh:
            head_text = "".join(next(fh, "") for _ in range(_PROMPT_SCAN_LINES))
    except OSError:
        return None
    return extract_issue_from_text(head_text)


def resolve(node_pid: int) -> ResolveResult:
    """All-in-one: claude pid + cwd + transcript + issue + reason on miss."""
    claude_pid = resolve_claude_pid(node_pid)
    cwd = _read_proc_cwd(claude_pid) if claude_pid is not None else None
    transcript, reason = resolve_transcript(node_pid)
    issue = resolve_issue(node_pid, transcript=transcript)
    return ResolveResult(
        node_pid=node_pid,
        claude_pid=claude_pid,
        cwd=cwd,
        transcript=transcript,
        issue=issue,
        reason=reason,
    )


# ── label backfill ─────────────────────────────────────────────────────────


def _live_node_pids() -> list[tuple[str, int]]:
    """List ``(happy_session_id, node_pid)`` for every session the local
    Happy daemon is currently tracking. Returns ``[]`` on daemon outage so
    the caller can degrade gracefully."""
    try:
        resp = spawn_session.post("/list", {})
    except SystemExit:
        return []
    out: list[tuple[str, int]] = []
    for c in resp.get("children", []) or []:
        sid = c.get("happySessionId")
        pid = c.get("pid")
        if isinstance(sid, str) and isinstance(pid, int):
            out.append((sid, pid))
    return out


def backfill_labels(dry_run: bool = False) -> list[dict[str, object]]:
    """Walk every LIVE EPS Happy session not already in the registry and
    write a ``manual-issue-<N>.json`` entry for each one whose issue is
    resolvable. Returns the list of entries written (or that WOULD be
    written under ``dry_run=True``).

    Idle / non-issue EPS sessions stay unmapped on purpose — we never
    fabricate an issue number. Already-registered sessions (autonomous OR
    manual) are skipped — we never overwrite their entries."""
    existing = spawn_session._load_session_issue_map()
    results: list[dict[str, object]] = []
    for sid, pid in _live_node_pids():
        if sid in existing:
            continue
        rr = resolve(pid)
        if not is_eps_cwd(rr.cwd):
            continue
        if rr.issue is None:
            continue
        entry = {
            "node_pid": pid,
            "happy_session_id": sid,
            "issue": rr.issue,
            "cwd": rr.cwd,
            "transcript": rr.transcript,
            "would_write": not dry_run,
        }
        results.append(entry)
        if dry_run:
            continue
        # Reuse the manual-session writer so we share the atomic temp+rename
        # path AND the filename convention (so the watcher's `issue-*.json`
        # glob doesn't accidentally pick up backfilled entries).
        spawn_session._register_manual_session(rr.issue, sid, rr.cwd or "")
        # Annotate the freshly-written entry with mode=backfilled so we can
        # distinguish them later. The base writer always sets mode=manual; we
        # rewrite atomically to flip it.
        dest = spawn_session.AUTONOMOUS_REGISTRY_DIR / f"manual-issue-{rr.issue}.json"
        try:
            payload = json.loads(dest.read_text())
            payload["mode"] = "backfilled"
            payload["node_pid"] = pid
            tmp = dest.with_suffix(".json.tmp")
            tmp.write_text(json.dumps(payload, indent=2))
            tmp.replace(dest)
        except (OSError, json.JSONDecodeError):
            # The base writer succeeded; the mode flip is cosmetic. If it
            # fails, the entry is still usable as a normal manual entry.
            pass
    return results


# ── CLI ────────────────────────────────────────────────────────────────────


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--node-pid",
        type=int,
        help="Resolve one happy node pid -> JSON {claude_pid, cwd, transcript, issue, reason}",
    )
    parser.add_argument(
        "--backfill",
        action="store_true",
        help=(
            "Walk every LIVE EPS Happy session not already in the registry; "
            "write a manual-issue-<N>.json entry for each one whose issue is "
            "resolvable. Idempotent. Skips non-EPS and idle/non-issue sessions."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="With --backfill: print would-be-written entries, don't touch disk.",
    )
    args = parser.parse_args(argv)

    if args.node_pid is None and not args.backfill:
        parser.error("pass --node-pid <pid> or --backfill")

    if args.node_pid is not None:
        result = resolve(args.node_pid)
        print(json.dumps(result.to_dict(), indent=2))

    if args.backfill:
        entries = backfill_labels(dry_run=args.dry_run)
        print(json.dumps({"backfilled": entries, "count": len(entries)}, indent=2))

    return 0


if __name__ == "__main__":
    sys.exit(main())
