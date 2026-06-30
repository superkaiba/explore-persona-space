#!/usr/bin/env python3
"""Reap leaked Codex app-server daemons + truncate the codex log WAL.

Codex ensemble-review (scripts/codex_task.py -> openai-codex plugin) spawns a
persistent daemon trio per session (node codex app-server, its codex-linux-x64
vendor binary, app-server-broker.mjs serve) that does NOT exit after the
companion task completes. They accumulate over weeks, each holding
~/.codex/logs_2.sqlite (WAL mode) open so SQLite can never checkpoint the WAL.

Mirrors scripts/cron_pod_audit.sh / scripts/worktree_audit.py: report-only by
default, --apply to act, --json summary, exit 2 when something was (would be)
reaped so the cron wrapper can swallow it. Exit nonzero ALSO on ps read failure
(R6) — a silent zero-reap is worse than a loud failure.
"""

import argparse
import contextlib
import json
import os
import re
import signal
import sqlite3
import subprocess
import sys
import time
from pathlib import Path

DEFAULT_MAX_AGE_H = 24.0
CODEX_DB = Path(os.path.expanduser("~/.codex/logs_2.sqlite"))

# Daemon argv-match patterns. Anchored on REAL daemon argv shapes (confirmed
# live 2026-06-30), NOT on bare substrings that also appear inside shell-wrapper
# env/eval text (see HARD_EXCLUDE + R3).
DAEMON_MATCH = [
    re.compile(
        r"\bcodex app-server\b"
    ),  # node ... codex app-server AND .../codex-linux-x64/.../bin/codex app-server
    re.compile(r"app-server-broker\.mjs\b"),  # node ... app-server-broker.mjs serve ...
    re.compile(r"\bcodex-linux-x64\b"),  # the vendor binary path
    re.compile(r"codex-companion\.mjs\b"),  # the companion runtime, if present (forward-looking)
]

# Checked FIRST. A process whose argv contains ANY of these is NEVER reaped,
# regardless of age — even if it also matches a DAEMON_MATCH pattern. This is
# what protects the codex_task.py REVIEW DRIVERS (the shell wrapper + uv/python
# children). It does NOT protect an active review's WORKER (`node ... codex
# app-server`) — that one shares the leaked-daemon argv shape; AGE is its only
# guard. A future maintainer lowering EPS_CODEX_REAPER_MAX_AGE_H is removing
# the only safety on an in-flight long review's worker.
HARD_EXCLUDE = [
    re.compile(r"codex_task\.py\b"),  # the review-dispatch drivers
    re.compile(r"\bworkflow_lint\b"),
    re.compile(r"\bwandb_reclaim\b"),
    re.compile(
        r"\b\S*sh -c\b"
    ),  # shell wrappers (bash/sh/zsh/...); their env/eval text echoes daemon strings — R3
]


def _max_age_seconds(cli_h: float | None) -> float:
    """Threshold in seconds: CLI flag wins, else EPS_CODEX_REAPER_MAX_AGE_H, else 24h."""
    if cli_h is not None:
        return cli_h * 3600.0
    env = os.environ.get("EPS_CODEX_REAPER_MAX_AGE_H")
    return (float(env) if env else DEFAULT_MAX_AGE_H) * 3600.0


def _ps_snapshot() -> tuple[list[tuple[int, int, str]], dict]:
    """One-shot (pid, etimes_seconds, argv) for every process.

    Returns (rows, status). status is {"ok": bool, "error": str|None,
    "returncode": int|None}. A read failure (OSError, SubprocessError, OR
    nonzero ps returncode) yields rows=[] AND ok=False — distinguishing a
    blind reaper from a clean zero-candidate day (R6). main() exits nonzero
    when status.ok is False so a cron read-failure is NEVER silent.

    argv is the full command line (ps -o args); the selector matches against
    argv, NOT comm (which reads 'node'/'codex'/'bash' for the daemons).
    """
    try:
        out = subprocess.run(
            ["ps", "-eo", "pid=,etimes=,args="],
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (OSError, subprocess.SubprocessError) as e:
        return [], {"ok": False, "error": f"{type(e).__name__}: {e}", "returncode": None}
    if out.returncode != 0:
        return [], {
            "ok": False,
            "error": (out.stderr or "").strip()[:200] or "nonzero rc",
            "returncode": out.returncode,
        }
    rows: list[tuple[int, int, str]] = []
    for line in out.stdout.splitlines():
        parts = line.strip().split(None, 2)  # pid, etimes, "rest is argv"
        if len(parts) < 3:
            continue
        try:
            rows.append((int(parts[0]), int(parts[1]), parts[2]))
        except ValueError:
            continue
    return rows, {"ok": True, "error": None, "returncode": 0}


def _is_daemon(argv: str) -> bool:
    return any(p.search(argv) for p in DAEMON_MATCH)


def _is_excluded(argv: str) -> bool:
    return any(p.search(argv) for p in HARD_EXCLUDE)


def enumerate_candidates(max_age_s: float, snapshot=None):
    """Return (candidates, sub_threshold, ps_status). Each list is
    [{pid, age_s, argv}]. A candidate matches a daemon pattern, is NOT excluded,
    and is >= threshold. sub_threshold = matched + not-excluded but younger
    (reported so the dry-run shows the live daemons it is deliberately sparing).
    `snapshot` injects a fake list for tests; status is {ok:True} when injected.
    """
    if snapshot is not None:
        rows, ps_status = snapshot, {"ok": True, "error": None, "returncode": 0}
    else:
        rows, ps_status = _ps_snapshot()
    candidates, sub_threshold = [], []
    for pid, age_s, argv in rows:
        if _is_excluded(argv) or not _is_daemon(argv):
            continue
        rec = {"pid": pid, "age_s": age_s, "argv": argv[:200]}
        (candidates if age_s >= max_age_s else sub_threshold).append(rec)
    return candidates, sub_threshold, ps_status


# --- kill machinery: COPIED from worktree_audit.py (_read_cmdline / _pid_running
#     / _kill_orphan_pids).
#     Why copy, not import: _kill_orphan_pids re-verifies against
#     ORPHAN_HOLDER_PATTERNS, which is the WORKTREE predicate. We need to
#     re-verify against the REAPER's own _is_daemon + _is_excluded, so a
#     verbatim import would silently couple to the wrong predicate.
def _read_cmdline(pid: int) -> str | None:
    try:
        with open(f"/proc/{pid}/cmdline", "rb") as fh:
            return fh.read().replace(b"\x00", b" ").decode("utf-8", "replace").strip()
    except OSError:
        return None


def _still_reapable(pid: int) -> bool:
    """PID-reuse guard: re-read /proc cmdline at signal time; only signal if it
    STILL matches a daemon pattern and is STILL not excluded (a recycled pid
    fails both and is spared)."""
    cmd = _read_cmdline(pid)
    return bool(cmd) and _is_daemon(cmd) and not _is_excluded(cmd)


def kill_candidates(pids, term_wait_s=8.0):
    """SIGTERM -> wait -> SIGKILL survivors; cmdline re-verified immediately
    before EVERY signal (R1). term_wait_s default 8s (R2).
    Returns {killed, leftover, reuse_skipped} so a future maintainer can see
    which pids were spared by the reuse guard (vs which truly survived)."""
    pending: list[int] = []
    reuse_skipped_term: list[int] = []
    for pid in pids:
        if not _still_reapable(pid):
            reuse_skipped_term.append(pid)
            continue
        with contextlib.suppress(OSError):
            os.kill(pid, signal.SIGTERM)
        pending.append(pid)
    deadline = time.time() + term_wait_s
    while pending and time.time() < deadline:
        time.sleep(0.2)
        pending = [p for p in pending if _read_cmdline(p)]
    reuse_skipped_kill: list[int] = []
    for pid in pending:
        if _still_reapable(pid):
            with contextlib.suppress(OSError):
                os.kill(pid, signal.SIGKILL)
        else:
            reuse_skipped_kill.append(pid)
    time.sleep(0.5)
    survivors = [p for p in pending if _read_cmdline(p)]
    sent_pids = [p for p in pids if p not in reuse_skipped_term]
    killed = [p for p in sent_pids if p not in survivors]
    return {
        "killed": sorted(killed),
        "leftover": sorted(survivors),
        "reuse_skipped": sorted(reuse_skipped_term + reuse_skipped_kill),
    }


def truncate_wal() -> dict:
    """Best-effort PRAGMA wal_checkpoint(TRUNCATE) on the codex DB. NEVER VACUUM
    (rewrites the whole multi-GB DB).

    SQLite's `PRAGMA wal_checkpoint(TRUNCATE)` returns a row `(busy, log,
    checkpointed)` and does NOT raise sqlite3.Error when a reader pins the WAL
    — it returns busy=1. We fetch + check the row so a reader-pinned WAL is
    reported as ok:False (R4 contract: "reported, not raised"), NOT silently
    as success while the WAL stays full.

    A missing DB / locked DB / surviving daemon connection all report
    ok:False with the WAL byte counts (before/after) intact in the payload.
    NEVER raises.
    """
    wal = CODEX_DB.with_name(CODEX_DB.name + "-wal")
    before = wal.stat().st_size if wal.exists() else 0
    if not CODEX_DB.exists():
        return {
            "ok": False,
            "error": "db_absent",
            "checkpoint_busy": False,
            "result_row": None,
            "wal_bytes_before": before,
            "wal_bytes_after": before,
        }
    busy: int | None = None
    result_row: tuple | None = None
    try:
        con = sqlite3.connect(str(CODEX_DB), timeout=10.0)
        try:
            con.execute("PRAGMA busy_timeout=10000")
            cur = con.execute("PRAGMA wal_checkpoint(TRUNCATE)")
            row = cur.fetchone()
            if row is not None:
                # SQLite docs: (busy, log_pages_checkpointed_to, log_pages_total).
                # busy != 0 means readers still hold the WAL → TRUNCATE was blocked.
                result_row = tuple(row)
                busy = int(row[0]) if row[0] is not None else None
        finally:
            con.close()
    except sqlite3.Error as e:
        after = wal.stat().st_size if wal.exists() else 0
        return {
            "ok": False,
            "error": str(e),
            "checkpoint_busy": False,
            "result_row": None,
            "wal_bytes_before": before,
            "wal_bytes_after": after,
        }
    after = wal.stat().st_size if wal.exists() else 0
    checkpoint_busy = bool(busy) if busy is not None else False
    return {
        "ok": (not checkpoint_busy) and busy is not None,
        "error": "checkpoint_busy"
        if checkpoint_busy
        else (None if busy is not None else "no_result_row"),
        "checkpoint_busy": checkpoint_busy,
        "result_row": result_row,
        "wal_bytes_before": before,
        "wal_bytes_after": after,
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description="Reap leaked Codex app-server daemons + truncate the log WAL."
    )
    ap.add_argument(
        "--apply",
        action="store_true",
        help="Actually kill + truncate (default: dry-run report).",
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="Explicit report-only (default; --dry-run wins on conflict).",
    )
    ap.add_argument("--json", action="store_true", help="Emit a JSON summary.")
    ap.add_argument(
        "--max-age-h",
        type=float,
        default=None,
        help="Reap daemons older than this many hours (default: EPS_CODEX_REAPER_MAX_AGE_H or 24).",
    )
    args = ap.parse_args(argv)
    apply = args.apply and not args.dry_run

    max_age_s = _max_age_seconds(args.max_age_h)
    candidates, sub_threshold, ps_status = enumerate_candidates(max_age_s)

    kill_result: dict | None = None
    wal: dict | None = None
    if apply and ps_status["ok"]:
        if candidates:
            kill_result = kill_candidates([p["pid"] for p in candidates])
            wal = truncate_wal()
        else:
            wal = truncate_wal()  # still reclaim WAL even with 0 candidates
    elif apply and not ps_status["ok"]:
        # Refuse to act on an unreadable process table — never mis-kill.
        pass

    summary = {
        "apply": apply,
        "max_age_h": max_age_s / 3600.0,
        "ps_status": ps_status,
        "n_candidates": len(candidates),
        "n_sub_threshold": len(sub_threshold),
        "candidates": candidates,
        "sub_threshold": sub_threshold,
        "kill_result": kill_result,
        "wal": wal,
    }
    if args.json:
        print(json.dumps(summary))
    else:
        if not ps_status["ok"]:
            print(f"codex_daemon_reaper: ps_FAILED ({ps_status['error']!r}) — refusing to act")
        else:
            verb = "killed" if apply else "would reap"
            print(
                f"codex_daemon_reaper: {verb} {len(candidates)} over-threshold | "
                f"{len(sub_threshold)} matched-but-young (spared)"
            )
            for c in candidates:
                print(f"  - {verb}: pid {c['pid']} age {c['age_s']}s :: {c['argv']}")
            for c in sub_threshold:
                print(f"  . spared (young): pid {c['pid']} age {c['age_s']}s")
            if kill_result is not None:
                if kill_result["leftover"]:
                    print(f"  ! survived: {kill_result['leftover']} (WAL truncate may be blocked)")
                if kill_result["reuse_skipped"]:
                    print(f"  . reuse-skipped: {kill_result['reuse_skipped']}")
            if wal is not None:
                print(f"  WAL: {wal}")
    # Exit codes (loud failures, never silent):
    #   3 = ps read failed
    #   2 = something was (or would be) reaped
    #   0 = clean zero-candidate pass
    if not ps_status["ok"]:
        return 3
    return 2 if candidates else 0


if __name__ == "__main__":
    sys.exit(main())
