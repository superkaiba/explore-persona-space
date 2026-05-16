"""Spawn / list / stop Happy Coder sessions via the local daemon HTTP RPC.

Happy ships a localhost HTTP control server at ``127.0.0.1:<port>`` (port lives
in ``~/.happy/daemon.state.json``). This is the canonical entry point for
programmatic session spawning — sessions created via ``happy claude`` directly
or via this RPC are equivalently visible in the user's mobile Happy app.

Routes the daemon exposes (POST only):

    /spawn-session   {"directory": <abs path>, "sessionId"?: <str>, "agent"?: <str>,
                      "environmentVariables"?: {...}, "claudeArgs"?: [<str>, ...]}
    /list            {}                                       -> {"children": [{"happySessionId": ..., "pid": ..., "startedBy": ...}, ...]}
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
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

HAPPY_HOME = Path.home() / ".happy"
DAEMON_STATE = HAPPY_HOME / "daemon.state.json"
PROJECT_ROOT = Path(__file__).resolve().parent.parent
WORKTREE_DIR = PROJECT_ROOT / ".claude" / "worktrees"


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
    else:
        print(f"Open it in Happy on your phone and type ``/issue {issue}``.")


def cmd_list(_: argparse.Namespace) -> None:
    """List active Happy sessions tracked by the local daemon."""
    resp = post("/list", {})
    children = resp.get("children", [])
    if not children:
        print("(no active Happy sessions)")
        return
    print(f"{'session id':<32}  {'pid':>8}  started_by")
    for c in children:
        print(
            f"{c.get('happySessionId', '?'):<32}  {c.get('pid', '?'):>8}  {c.get('startedBy', '?')}"
        )


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
    p_issue.set_defaults(fn=cmd_spawn_issue)

    p_list = sub.add_parser("list", help="list active Happy sessions")
    p_list.set_defaults(fn=cmd_list)

    p_stop = sub.add_parser("stop", help="stop a Happy session by id")
    p_stop.add_argument("--session-id", required=True)
    p_stop.set_defaults(fn=cmd_stop)

    args = parser.parse_args(argv)
    args.fn(args)


if __name__ == "__main__":
    main()
