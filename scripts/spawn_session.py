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
import re
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

HAPPY_HOME = Path.home() / ".happy"
DAEMON_STATE = HAPPY_HOME / "daemon.state.json"
SESSIONS_JSON = HAPPY_HOME / "sessions.json"
PROJECT_ROOT = Path(__file__).resolve().parent.parent
WORKTREE_DIR = PROJECT_ROOT / ".claude" / "worktrees"


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


def post(path: str, body: dict[str, Any]) -> dict[str, Any]:
    """POST a JSON body to the local Happy daemon and return the parsed
    response. Errors are surfaced as :func:`sys.exit` with the daemon's
    response body when available."""
    url = f"http://127.0.0.1:{daemon_port()}{path}"
    payload = json.dumps(body).encode()
    req = urllib.request.Request(
        url,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            return json.loads(resp.read())
    except urllib.error.HTTPError as e:
        try:
            err_body = json.loads(e.read())
        except Exception:
            err_body = {"raw": str(e)}
        sys.exit(f"Happy daemon {path} returned HTTP {e.code}: {err_body}")
    except urllib.error.URLError as e:
        sys.exit(f"Happy daemon {path} unreachable at 127.0.0.1: {e}")


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
        prompt = f"/loop 10m /issue {issue}"
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
    else:
        print(f"Open it in Happy on your phone and type ``/issue {issue}``.")


def cmd_list(args: argparse.Namespace) -> None:
    """List Happy sessions, enriched with cwd + lifecycle state.

    Default: sessions the local daemon is actively tracking (live processes).
    ``--all``: every session in ``~/.happy/sessions.json`` (including stopped
    ones), newest first, so you can pick one to ``happy resume``."""
    meta = _load_session_meta()

    if getattr(args, "all", False):
        live = _live_session_ids()
        rows = [
            (
                sid,
                "live" if sid in live else "stopped",
                m.get("startedBy", "?"),
                _dir_label(m.get("path")),
                m.get("savedAt", 0) or 0,
            )
            for sid, m in meta.items()
        ]
        # Live sessions first, then newest-saved first within each group.
        rows.sort(key=lambda r: (r[1] != "live", -r[4]))
        if not rows:
            print("(no sessions in sessions.json)")
            return
        print(f"{'session id':<28}  {'state':<8}  {'started_by':<10}  dir")
        for sid, state, started_by, dir_label, _ts in rows:
            print(f"{sid[:26]:<28}  {state:<8}  {started_by:<10}  {dir_label}")
        print(f"\n{len(rows)} session(s), {len(live)} live. Resume one: happy resume <id-prefix>")
        return

    resp = post("/list", {})
    children = resp.get("children", [])
    if not children:
        print("(no active Happy sessions)")
        return
    print(f"{'session id':<28}  {'pid':>8}  {'state':<10}  dir")
    for c in children:
        sid = c.get("happySessionId", "?")
        m = meta.get(sid, {})
        state = m.get("lifecycleState", "?")
        print(f"{sid[:26]:<28}  {c.get('pid', '?'):>8}  {state:<10}  {_dir_label(m.get('path'))}")
    print(f"\n{len(children)} active session(s). Resume one: happy resume <id-prefix>")


def cmd_stop(args: argparse.Namespace) -> None:
    """Stop a Happy session by id."""
    resp = post("/stop-session", {"sessionId": args.session_id})
    if not resp.get("success"):
        sys.exit(f"stop failed: {resp}")
    print(f"Stopped session {args.session_id}")


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
        help=("Shorthand for --initial-prompt '/loop 10m /issue <N>' so the session self-paces."),
    )
    p_issue.add_argument(
        "--auto-approve-gpu-hours",
        type=float,
        default=24.0,
        help=(
            "Autonomous sessions auto-approve a plan whose estimated GPU-hours "
            "is <= this value and park at plan_pending above it. Default 24."
        ),
    )
    p_issue.set_defaults(fn=cmd_spawn_issue)

    p_list = sub.add_parser("list", help="list active Happy sessions (cwd + state)")
    p_list.add_argument(
        "--all",
        action="store_true",
        help="include stopped/historical sessions from ~/.happy/sessions.json (newest first)",
    )
    p_list.set_defaults(fn=cmd_list)

    p_stop = sub.add_parser("stop", help="stop a Happy session by id")
    p_stop.add_argument("--session-id", required=True)
    p_stop.set_defaults(fn=cmd_stop)

    args = parser.parse_args(argv)
    args.fn(args)


if __name__ == "__main__":
    main()
