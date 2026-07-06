"""Thin READ-ONLY visibility CLI over the EPS issue sessions.

The session visibility layer of EPS workflow v2 (§2). Answers "what issue
sessions are running, and what are they doing?" without changing the runtime
in any way — every subcommand only READS existing state (the Happy daemon's
live-session list, ``~/.happy/sessions.json`` metadata, the
``~/.eps-autonomous`` session registry, the summarizer's
``session_progress.json`` cache, and live tmux windows), EXCEPT ``stop``,
which delegates to the existing ``spawn_session.py`` stop path (never kills
tmux directly).

Subcommands::

    uv run python scripts/eps_sessions.py list [--json]
    uv run python scripts/eps_sessions.py attach-cmd <N>
    uv run python scripts/eps_sessions.py stop <N> [--reason "..."]

``list`` — one row per issue session. The **Happy registry is authoritative**:
rows are built from the sessions the local daemon is actively tracking, joined
with their task status, last summary line (from ``session_summarize.py``), and
last-activity age. Live tmux windows whose Claude session is an EPS session NOT
tracked by the daemon are added as ``tmux-only`` fallback rows. When the Happy
daemon is unreachable the command degrades to the tmux-resolved rows only and
prints a WARN line (never crashes).

``attach-cmd <N>`` — print the exact ``tmux attach -t <session>`` command for
the session driving issue N, or a clear error if it cannot be resolved to a
tmux session.

``stop <N>`` — resolve issue N to its Happy session and delegate to
``spawn_session.cmd_stop`` (which posts the deliberate-stop breadcrumb + the
``/stop-session`` RPC). Never touches tmux.

Design: the row-assembly + rendering logic is factored into PURE functions
(:func:`build_happy_rows`, :func:`build_tmux_only_rows`,
:func:`tmux_target_by_node_pid`, :func:`render_rows_text`) with all I/O
injected, so they are unit-testable without a live daemon or tmux server. The
``gather_rows`` orchestrator does the real I/O.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import session_resolver  # noqa: E402
import session_summarize  # noqa: E402
import spawn_session  # noqa: E402
import tmux_window_titles  # noqa: E402

# ── row model ────────────────────────────────────────────────────────────────


@dataclass
class SessionRow:
    """One row of the ``list`` view.

    ``source`` is ``"happy"`` (tracked by the local Happy daemon — the
    authoritative rows) or ``"tmux-only"`` (a live EPS tmux session the daemon
    is not tracking — a superseded / never-registered driver generation).
    ``session_id`` is the Happy session id (``None`` for a tmux-only row, which
    the daemon does not track). ``tmux`` is the ``session:index`` target when
    the session could be located in a live tmux window, else ``None``."""

    issue: int | None
    status: str
    source: str
    tmux: str | None
    session_id: str | None
    last_activity_age: str
    summary: str
    error: str | None = None

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


# ── pure helpers ─────────────────────────────────────────────────────────────


def tmux_target_by_node_pid(
    windows: list[tmux_window_titles.Window],
    descendants_fn: Callable[[int], list[int]],
) -> dict[int, str]:
    """Map every pid in each window's process subtree -> that window's
    ``session:index`` target.

    ``descendants_fn(pane_pid) -> list[int]`` returns the pids in a window's
    process subtree (injected so the map can be built without touching /proc in
    tests). First window to claim a pid wins (``setdefault``) — deterministic
    when a pid somehow appears under two windows."""
    out: dict[int, str] = {}
    for w in windows:
        for pid in descendants_fn(w.pane_pid):
            out.setdefault(pid, w.target)
    return out


def _age_from_entry(entry: dict, now: float | None = None) -> str:
    """Compact "3m ago" age for a summary-cache entry, from its
    ``last_activity_ts`` (preferred) or ``summary_ts``. Empty string when
    neither is present/parseable. Reuses ``spawn_session._format_event_age``
    (the same ISO-ts age formatter the ``list`` progress cell uses)."""
    ts = entry.get("last_activity_ts") or entry.get("summary_ts")
    return spawn_session._format_event_age(ts if isinstance(ts, str) else None, now=now)


def build_happy_rows(
    live_children: list[dict],
    session_meta: dict[str, dict],
    issue_map: dict[str, int],
    summary_cache: dict[str, dict],
    tmux_target_map: dict[int, str],
    *,
    status_fn: Callable[[int], str],
    now: float | None = None,
) -> list[SessionRow]:
    """One :class:`SessionRow` per LIVE EPS Happy session (registry
    authoritative). PURE — all state is injected.

    A session counts as EPS when its cwd (``session_meta[sid]["path"]``) is
    under the EPS repo/worktrees OR it is mapped to an issue in ``issue_map``.
    The issue is taken from ``issue_map`` and falls back to the issue inferred
    from an ``issue-<N>`` worktree cwd. Task status is read LIVE via
    ``status_fn`` (never the possibly-stale cached status). Summary + age come
    from the summarizer cache (empty when the summarizer has not run yet)."""
    rows: list[SessionRow] = []
    for child in live_children:
        sid = child.get("happySessionId")
        pid = child.get("pid")
        if not isinstance(sid, str):
            continue
        meta = session_meta.get(sid, {})
        path = meta.get("path") if isinstance(meta, dict) else None
        issue = issue_map.get(sid)
        if issue is None:
            issue = spawn_session._infer_issue_from_path(path)
        is_eps = session_resolver.is_eps_cwd(path) or issue is not None
        if not is_eps:
            continue
        entry = summary_cache.get(sid) or {}
        if not isinstance(entry, dict):
            entry = {}
        status = status_fn(issue) if isinstance(issue, int) else "-"
        summary = entry.get("summary")
        summary = summary if isinstance(summary, str) else ""
        err = entry.get("error")
        rows.append(
            SessionRow(
                issue=issue if isinstance(issue, int) else None,
                status=status,
                source="happy",
                tmux=tmux_target_map.get(pid) if isinstance(pid, int) else None,
                session_id=sid,
                last_activity_age=_age_from_entry(entry, now=now),
                summary=summary,
                error=err if isinstance(err, str) and err else None,
            )
        )
    return rows


def build_tmux_only_rows(
    windows: list[tmux_window_titles.Window],
    covered_node_pids: set[int],
    *,
    resolve_fn: Callable[[int], session_resolver.ResolveResult | None],
    status_fn: Callable[[int], str],
    now: float | None = None,
) -> list[SessionRow]:
    """Fallback rows for LIVE EPS tmux windows whose Claude session is NOT a
    Happy row (unregistered / superseded / daemon-untracked). PURE given the
    injected ``resolve_fn`` (window pane pid -> ResolveResult).

    A window contributes a row only when its resolved Claude cwd is EPS AND its
    Claude node pid is not already among ``covered_node_pids`` (the daemon's
    live-session pids) — otherwise it is already a ``build_happy_rows`` row."""
    rows: list[SessionRow] = []
    for w in windows:
        rr = resolve_fn(w.pane_pid)
        if rr is None or not session_resolver.is_eps_cwd(rr.cwd):
            continue
        if rr.node_pid is not None and rr.node_pid in covered_node_pids:
            continue
        issue = rr.issue
        status = status_fn(issue) if isinstance(issue, int) else "-"
        rows.append(
            SessionRow(
                issue=issue if isinstance(issue, int) else None,
                status=status,
                source="tmux-only",
                tmux=w.target,
                session_id=None,
                last_activity_age="",
                summary="(not tracked by Happy daemon)",
                error=rr.reason,
            )
        )
    return rows


def _row_sort_key(row: SessionRow) -> tuple[int, int, str]:
    """Sort mapped issues ascending first, unmapped (``None``) last, then by
    source + tmux target so the order is stable frame-to-frame."""
    if isinstance(row.issue, int):
        return (0, row.issue, f"{row.source}:{row.tmux or ''}")
    return (1, 0, f"{row.source}:{row.tmux or ''}")


def render_rows_text(rows: list[SessionRow], *, daemon_up: bool) -> str:
    """Render the ``list`` rows as a plain-text table. PURE."""
    ordered = sorted(rows, key=_row_sort_key)
    header = f"{'issue':<7}  {'status':<12}  {'source':<9}  {'age':<8}  {'tmux':<16}  summary"
    lines = [header]
    for r in ordered:
        issue_cell = f"#{r.issue}" if isinstance(r.issue, int) else "-"
        summary = r.summary or (f"<{r.error}>" if r.error else "")
        lines.append(
            f"{issue_cell:<7}  {r.status:<12}  {r.source:<9}  "
            f"{r.last_activity_age:<8}  {(r.tmux or '-'):<16}  {summary}"
        )
    footer = f"\n{len(ordered)} issue session(s)."
    if not daemon_up:
        footer += " (Happy daemon unreachable — tmux-resolved rows only.)"
    return "\n".join(lines) + footer


# ── I/O orchestration ────────────────────────────────────────────────────────


def _fetch_live_children() -> tuple[list[dict], bool]:
    """``(live_children, daemon_up)``.

    Distinguishes "daemon up, no sessions" (``[], True``) from "daemon
    unreachable" (``[], False``) — the plain ``spawn_session._live_children``
    swallows that distinction, but ``list`` needs it to decide whether to
    print the daemon-down WARN. Never raises: any daemon-contact failure
    returns ``([], False)``."""
    import urllib.error
    import urllib.request

    if not spawn_session.DAEMON_STATE.is_file():
        return [], False
    try:
        port = spawn_session.daemon_port()
    except SystemExit:
        return [], False
    try:
        url = f"http://127.0.0.1:{port}/list"
        req = urllib.request.Request(
            url, data=b"{}", headers={"Content-Type": "application/json"}, method="POST"
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read())
    except (urllib.error.URLError, OSError, json.JSONDecodeError, ValueError):
        return [], False
    children = data.get("children", []) if isinstance(data, dict) else []
    return (children if isinstance(children, list) else []), True


def _load_summary_cache() -> dict[str, dict]:
    """``{happy_session_id: entry}`` from the summarizer cache; ``{}`` when the
    cache is missing/unreadable (best-effort enrichment)."""
    data = session_summarize.load_cache()
    sessions = data.get("sessions") if isinstance(data, dict) else None
    if not isinstance(sessions, dict):
        return {}
    return {sid: e for sid, e in sessions.items() if isinstance(e, dict)}


def gather_rows(now: float | None = None) -> tuple[list[SessionRow], bool]:
    """Gather all ``list`` rows (happy + tmux-only) and the daemon-up flag.

    The I/O orchestrator: it fetches every input the pure builders need (live
    daemon children, sessions.json metadata, the issue registry, the summary
    cache, live tmux windows + their process subtrees) and assembles the rows.
    Returns ``(rows, daemon_up)``."""
    now = time.time() if now is None else now
    live_children, daemon_up = _fetch_live_children()
    session_meta = spawn_session._load_session_meta()
    issue_map = spawn_session._load_session_issue_map()
    summary_cache = _load_summary_cache()
    windows = tmux_window_titles.list_windows()
    target_map = tmux_target_by_node_pid(windows, tmux_window_titles._all_descendant_pids)

    happy = build_happy_rows(
        live_children,
        session_meta,
        issue_map,
        summary_cache,
        target_map,
        status_fn=session_summarize._get_task_status,
        now=now,
    )
    covered = {c.get("pid") for c in live_children if isinstance(c.get("pid"), int)}
    tmux_only = build_tmux_only_rows(
        windows,
        covered,
        resolve_fn=tmux_window_titles._resolve_transcript_for_window,
        status_fn=session_summarize._get_task_status,
        now=now,
    )
    return happy + tmux_only, daemon_up


# ── subcommands ──────────────────────────────────────────────────────────────


def cmd_list(args: argparse.Namespace) -> int:
    """List every issue session (Happy authoritative, tmux fallback)."""
    rows, daemon_up = gather_rows()
    if not daemon_up:
        print(
            "WARN: Happy daemon unreachable; showing tmux-resolved EPS sessions only.",
            file=sys.stderr,
        )
    if getattr(args, "json", False):
        payload = {
            "daemon_up": daemon_up,
            "sessions": [r.to_dict() for r in sorted(rows, key=_row_sort_key)],
        }
        print(json.dumps(payload, indent=2))
        return 0
    print(render_rows_text(rows, daemon_up=daemon_up))
    return 0


def _resolve_node_pid(sid: str, live_children: list[dict]) -> int | None:
    """The Happy node (wrapper) pid for session ``sid`` from the daemon's live
    children, or ``None`` when the session is not currently live."""
    for c in live_children:
        if c.get("happySessionId") == sid:
            pid = c.get("pid")
            return pid if isinstance(pid, int) else None
    return None


def cmd_attach_cmd(args: argparse.Namespace) -> int:
    """Print the ``tmux attach -t <session>`` command for issue ``N``'s session.

    Resolves issue -> Happy session id -> live node pid -> the tmux window
    whose process subtree contains that node pid. Prints a clear error (and
    returns non-zero) at any unresolvable step."""
    issue = args.issue
    sid = spawn_session.resolve_session_for_issue(issue)
    if sid is None:
        print(
            f"no Happy session registered for issue #{issue}; "
            f"spawn one with `spawn_session.py spawn-issue --issue {issue}`",
            file=sys.stderr,
        )
        return 1
    live_children, daemon_up = _fetch_live_children()
    node_pid = _resolve_node_pid(sid, live_children)
    if node_pid is None:
        state = "up" if daemon_up else "unreachable"
        print(
            f"session {sid} for issue #{issue} is not live (Happy daemon {state}); "
            f"cannot resolve a tmux target",
            file=sys.stderr,
        )
        return 1
    windows = tmux_window_titles.list_windows()
    target = tmux_target_by_node_pid(windows, tmux_window_titles._all_descendant_pids).get(node_pid)
    if target is None:
        print(
            f"session {sid} (issue #{issue}, node pid {node_pid}) is not attached "
            f"to any live tmux window",
            file=sys.stderr,
        )
        return 1
    session_name = target.split(":", 1)[0]
    print(f"tmux attach -t {session_name}")
    return 0


def cmd_stop(args: argparse.Namespace) -> int:
    """Stop issue ``N``'s Happy session by delegating to
    ``spawn_session.cmd_stop`` (deliberate-stop breadcrumb + ``/stop-session``
    RPC). Never touches tmux."""
    issue = args.issue
    sid = spawn_session.resolve_session_for_issue(issue)
    if sid is None:
        print(f"no Happy session registered for issue #{issue}", file=sys.stderr)
        return 1
    # Delegate to the canonical stop path (operator source -> posts the
    # deliberate-stop breadcrumb on the owning task, then the /stop-session
    # RPC). spawn_session.cmd_stop sys.exits on RPC failure, which surfaces as
    # a clean non-zero exit here.
    stop_args = argparse.Namespace(
        session_id=sid,
        reason=args.reason,
        stop_source="operator",
    )
    spawn_session.cmd_stop(stop_args)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_list = sub.add_parser("list", help="list every issue session (Happy + tmux fallback)")
    p_list.add_argument("--json", action="store_true", help="emit rows as JSON")
    p_list.set_defaults(fn=cmd_list)

    p_attach = sub.add_parser(
        "attach-cmd", help="print the `tmux attach` command for issue #N's session"
    )
    p_attach.add_argument("issue", type=int)
    p_attach.set_defaults(fn=cmd_attach_cmd)

    p_stop = sub.add_parser(
        "stop", help="stop issue #N's Happy session (delegates to spawn_session)"
    )
    p_stop.add_argument("issue", type=int)
    p_stop.add_argument(
        "--reason",
        default="operator stop via eps_sessions.py stop",
        help="one-line reason recorded in the deliberate-stop breadcrumb",
    )
    p_stop.set_defaults(fn=cmd_stop)

    args = parser.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
