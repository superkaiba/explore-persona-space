#!/usr/bin/env python3
"""File-time auto-dispatch wrapper for ripe `kind: infra`/`batch` tasks (#690).

WHY THIS SCRIPT EXISTS. A filed `proposed` `kind: infra` task today only
auto-runs if a LIVE PM session does a STATUS pass and writes it into the
infra-drain queue, which the 10-min watcher's infra-drain pass then executes.
With no PM in the loop, a ripe infra fix orphans indefinitely (incident #684
sat at `proposed` ~17h). This wrapper makes "whoever files a ripe infra fix
dispatches it" the DEFAULT at FILE time: it files via `task.py new` (so the
mutation core + flock + commit stay the single source of truth) and then
best-effort spawns `spawn_session.py spawn-issue --issue <N> --auto`.

ORCHESTRATION LAYER, NOT `task.py`'s mutation core. Session-spawning must NOT
live inside `task.py` — `task.py` is the branch-guarded, flock-holding
state-mutation core that pod-side code must NEVER shell out to
(`tests/test_no_pod_side_task_py_shellout.py`, CLAUDE.md). This wrapper is a
separate orchestration-layer script (the same class as `spawn_session.py` /
the watcher); it shells out to `task.py new` and is explicitly allowlisted in
`_LOCAL_VM_ONLY_PATHS` as a local-VM-only caller. Pod-side code never invokes
it. Its `from autonomous_session_watch import infra_dispatch_has_free_slot`
is a Python module import of an orchestration-layer function, NOT a `task.py`
shellout, so it does not affect that scanner.

FILING IS MUST-SUCCEED; DISPATCH IS BEST-EFFORT. The task is filed even when
the spawn is skipped or fails — the always-on watcher `proposed_infra_sweep`
pass is the backstop that dispatches a filed-but-not-dispatched task within
~10 min. The spawn no-ops cleanly (exit 0) when: `--no-dispatch` is passed,
the Happy daemon is unreachable (headless / pod-side filing), the shared
3-session infra cap is full or occupancy is unreadable (#690 M1 — a wrapper
call can never push a 4th session past the cap before the watcher's next
tick), the task already has a live session, or the spawn subprocess errors.
Only a FAILED `task.py new` exits non-zero (filing is the durable half a
caller depends on).
"""

from __future__ import annotations

import argparse
import http.client
import json
import subprocess
import sys
import urllib.error
import urllib.request
from pathlib import Path

# scripts/ on sys.path so the sibling orchestration-layer imports resolve
# under both `python scripts/file_infra_task.py` (scripts/ is sys.path[0])
# and `from scripts.file_infra_task import ...` (it is not). Mirrors the
# bootstrap in autonomous_session_watch.py / backend_poll.py.
_SCRIPTS_DIR = str(Path(__file__).resolve().parent)
if _SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, _SCRIPTS_DIR)

# Orchestration-layer module imports (NOT task.py shellouts):
#   - infra_dispatch_has_free_slot: the shared #690-M1 cap-check primitive
#     (one definition consumed by the wrapper, the infra-drain pass, and the
#     proposed-infra sweep — so a future cap-tightening refactor edits ONE
#     function and the three dispatchers cannot drift apart).
#   - AUTONOMOUS_REGISTRY_DIR / daemon_port: the registration dir + Happy
#     daemon port, the SAME source of truth `spawn_session.py list` uses.
from autonomous_session_watch import infra_dispatch_has_free_slot  # noqa: E402
from spawn_session import (  # noqa: E402
    PROJECT_ROOT,
    _live_session_ids,
    _load_session_issue_map,
    daemon_port,
)

# The auto-dispatchable pure-code/ops kinds. `experiment`/`analysis`/
# `campaign` are rejected: analysis needs the `agent-ok` opt-in + PM triage;
# experiment needs the adversarial-planner GPU gate; campaign has its own
# `/campaign` path. Keeping the wrapper to {infra, batch} ensures it can never
# auto-`--auto`-dispatch a GPU-spending task outside a cap.
_WRAPPER_KINDS = ("infra", "batch")


def _daemon_reachable() -> bool:
    """True iff the Happy daemon's control server answers ``/list``.

    Same probe shape as the watcher's :func:`autonomous_session_watch._daemon_reachable`
    (POST ``{}`` to ``127.0.0.1:<daemon_port()>/list``; swallow the connection /
    HTTP / decode tier into a conservative ``False`` = "I cannot tell whether
    the daemon is up"). A ``False`` here cleanly no-ops the spawn — the
    headless / pod-side filing case — and falls through to the watcher
    backstop."""
    try:
        url = f"http://127.0.0.1:{daemon_port()}/list"
        req = urllib.request.Request(
            url, data=b"{}", headers={"Content-Type": "application/json"}, method="POST"
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            json.loads(resp.read())
        return True
    except (
        SystemExit,
        urllib.error.URLError,
        http.client.HTTPException,
        OSError,
        UnicodeDecodeError,
        json.JSONDecodeError,
    ):
        return False


def _issue_has_live_registration(issue: int) -> bool:
    """True iff ``issue`` already has a LIVE issue-mapped session — the
    idempotency pre-check. Reads the SAME registration source of truth
    ``spawn_session.py list`` uses (:func:`_load_session_issue_map`, which
    spans ``issue-<N>.json`` / ``manual-issue-<N>.json`` / ``campaign-<N>.json``)
    and confirms the mapped session id is in the daemon's live set
    (:func:`_live_session_ids`). A just-filed task essentially never has one,
    but the check makes the wrapper safe to re-run. Best-effort: any read
    failure resolves to ``False`` (proceed to spawn — the spawn path itself
    is idempotent enough that a redundant spawn is a no-GPU session reaped by
    the watcher, while a false-positive skip would orphan the task)."""
    try:
        issue_map = _load_session_issue_map()
    except (OSError, ValueError):
        return False
    if issue not in issue_map.values():
        return False
    live = _live_session_ids()
    return any(sid for sid, mapped in issue_map.items() if mapped == issue and sid in live)


def _parse_new_id(stdout: str) -> int | None:
    """Extract the new task id from ``task.py new`` stdout. ``cmd_create``
    prints ``#<new_id>`` via ``_safe_echo`` (scripts/task.py); take the FIRST
    ``#<int>`` token. Returns ``None`` if no id token is present (the wrapper
    then treats filing as failed)."""
    for tok in stdout.split():
        if tok.startswith("#") and tok[1:].isdigit():
            return int(tok[1:])
    return None


def _build_new_argv(args: argparse.Namespace) -> list[str]:
    """Compose the ``task.py new`` subprocess argv, forwarding only the
    creation-relevant flags. ``--tag`` is repeatable (forwarded once per tag)
    so dedup keys (e.g. workflow-fix `wf-fix` / `wf-fix-fp:<fp>`) are applied
    AT CREATION rather than in a separate `add-tag` step."""
    argv = [
        "uv",
        "run",
        "python",
        "scripts/task.py",
        "new",
        "--kind",
        args.kind,
        "--title",
        args.title,
    ]
    if args.body is not None:
        argv += ["--body", args.body]
    elif args.body_file is not None:
        argv += ["--body-file", args.body_file]
    if args.parent is not None:
        argv += ["--parent", str(args.parent)]
    for tag in args.tag or []:
        argv += ["--tag", tag]
    if args.origin_prompt is not None:
        argv += ["--origin-prompt", args.origin_prompt]
    return argv


def _build_spawn_argv(issue: int, args: argparse.Namespace) -> list[str]:
    """Compose the ``spawn-issue --issue <N> --auto`` subprocess argv. Forwards
    ``--auto-approve-gpu-hours`` only when the caller set it (else
    spawn_session's own default applies — 100 GPU-h; infra/batch tasks need
    ~0 GPU)."""
    argv = [
        "uv",
        "run",
        "python",
        "scripts/spawn_session.py",
        "spawn-issue",
        "--issue",
        str(issue),
        "--auto",
    ]
    if args.auto_approve_gpu_hours is not None:
        argv += ["--auto-approve-gpu-hours", str(args.auto_approve_gpu_hours)]
    return argv


def cmd_file_infra(args: argparse.Namespace) -> int:
    """File a ripe `kind: infra`/`batch` task, then best-effort dispatch it.

    Returns the process exit code: non-zero ONLY when the must-succeed FILING
    half fails; 0 for every dispatch no-op / dispatch failure (the task is
    filed and the watcher backstop covers a skipped/failed spawn — a non-zero
    here would make callers think filing failed)."""
    # 1. File first (the durable, must-succeed half).
    new_argv = _build_new_argv(args)
    try:
        filed = subprocess.run(
            new_argv, cwd=PROJECT_ROOT, capture_output=True, text=True, timeout=120
        )
    except (subprocess.SubprocessError, OSError) as e:
        print(f"file_infra_task: `task.py new` failed to run: {e}", file=sys.stderr)
        return 1
    if filed.returncode != 0:
        sys.stderr.write(filed.stderr)
        print(
            f"file_infra_task: `task.py new` exited {filed.returncode}; task NOT filed",
            file=sys.stderr,
        )
        return filed.returncode
    issue = _parse_new_id(filed.stdout)
    if issue is None:
        print(
            "file_infra_task: filed a task but could not parse its `#<id>` from "
            f"`task.py new` stdout ({filed.stdout.strip()!r}); cannot dispatch",
            file=sys.stderr,
        )
        # Filing succeeded (task.py committed); we just can't address the spawn.
        # The watcher backstop sweeps it by status, so this is not a filing
        # failure — exit 0.
        return 0

    # 2. --no-dispatch -> file only.
    if args.no_dispatch:
        print(f"filed #{issue} (dispatch skipped: --no-dispatch)")
        return 0

    # 3. Daemon reachability gate (the headless / pod-side no-op).
    if not _daemon_reachable():
        print(
            f"filed #{issue}; Happy daemon unreachable, NOT dispatching "
            f"(watcher proposed_infra_sweep backstop will pick it up within ~10 min)"
        )
        return 0

    # 3.5. Shared cap gate (#690 M1). The wrapper passes pending=0 per the
    # helper docstring (a just-filed task has no registration yet, so
    # occupancy alone is the binding budget for the one spawn it is about to
    # make). Both branches file-but-no-op the spawn; the backstop dispatches
    # the filed task when a slot frees.
    free_slot = infra_dispatch_has_free_slot()
    if free_slot is None:
        print(
            f"filed #{issue}; infra-slot occupancy unreadable, NOT dispatching "
            f"(fail-closed: a partial occupancy read could over-dispatch; watcher "
            f"proposed_infra_sweep backstop will pick it up within ~10 min)"
        )
        return 0
    if not free_slot:
        print(
            f"filed #{issue}; infra dispatch cap (3) full, NOT dispatching "
            f"(watcher proposed_infra_sweep backstop will pick it up within ~10 min)"
        )
        return 0

    # 4. Idempotency pre-check (safe to re-run the wrapper).
    if _issue_has_live_registration(issue):
        print(f"filed #{issue}; already has a live session, NOT re-dispatching")
        return 0

    # 5. Spawn (best-effort).
    spawn_argv = _build_spawn_argv(issue, args)
    try:
        spawned = subprocess.run(
            spawn_argv, cwd=PROJECT_ROOT, capture_output=True, text=True, timeout=120
        )
    except (subprocess.SubprocessError, OSError) as e:
        print(
            f"filed #{issue}; dispatch FAILED ({e}); watcher proposed_infra_sweep "
            f"backstop will retry",
            file=sys.stderr,
        )
        return 0
    if spawned.returncode != 0:
        print(
            f"filed #{issue}; dispatch FAILED ({spawned.stderr.strip()[:300]}); "
            f"watcher proposed_infra_sweep backstop will retry",
            file=sys.stderr,
        )
        return 0
    first_line = (spawned.stdout.strip().splitlines() or [""])[0]
    print(f"filed + dispatched #{issue}: {first_line}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    """The CLI surface mirrors `task.py new`'s creation-relevant flags so the
    wrapper is a drop-in "file + dispatch"."""
    parser = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--kind",
        default="infra",
        choices=_WRAPPER_KINDS,
        help="task kind; only the auto-dispatchable pure-code/ops kinds "
        "{infra, batch} are accepted (default infra). experiment/analysis/"
        "campaign have their own routing and are rejected.",
    )
    parser.add_argument("--title", required=True, help="task title (required)")
    body_group = parser.add_mutually_exclusive_group()
    body_group.add_argument("--body", default=None, help="body text directly (forwarded to new)")
    body_group.add_argument(
        "--body-file", default=None, help="path to body file (forwarded to new)"
    )
    parser.add_argument(
        "--parent", type=int, default=None, help="parent task id (forwarded to new)"
    )
    parser.add_argument(
        "--tag",
        action="append",
        default=[],
        help="tag, repeatable (forwarded to new — applied AT creation, so dedup "
        "keys land on the task without a separate add-tag step)",
    )
    parser.add_argument(
        "--origin-prompt", default=None, help="verbatim originating prompt (forwarded to new)"
    )
    parser.add_argument(
        "--no-dispatch",
        action="store_true",
        help="file only; skip the spawn attempt (for a pod-side / deliberately-"
        "deferred filer that wants uniform file+provenance without spawning).",
    )
    parser.add_argument(
        "--auto-approve-gpu-hours",
        type=float,
        default=None,
        help="forwarded to spawn-issue --auto (default: spawn_session's own "
        "default of 100 GPU-h; infra/batch tasks need ~0 GPU).",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return cmd_file_infra(args)


if __name__ == "__main__":
    raise SystemExit(main())
