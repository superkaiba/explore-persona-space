"""poll_pipeline.py — one-tick poller for a running experiment pod.

Invoked by the `/issue` orchestrator's bg-Bash sleep-chain (see
`.claude/skills/issue/SKILL.md` Step 6d.2). Performs ONE poll then exits
— the orchestrator chains successive `Bash(sleep 540 && uv run python
scripts/poll_pipeline.py ..., run_in_background=true)` calls and is
re-invoked by the harness when each bg Bash returns.

Why orchestrator-owned: subagents have ONE turn — they are NOT
auto-re-invoked when a bg Bash finishes. The orchestrator IS. See
`CLAUDE.md` § "Subagent vs orchestrator re-invocation semantics" and
the deprecated memory `feedback_subagent_sleep_chain.md` for context.

Per tick:

1. SSH to the pod (one heredoc batching: PID liveness, log mtime, log tail).
2. Parse the latest `[phase=...]` line from the log tail.
3. If new milestone vs the cached previous phase, post `epm:progress`
   to the task's events.jsonl via the local-VM `task_workflow.post_event`
   library (NOT on the pod).
4. Decide status: `done` | `stalled` | `dead` | `running`.
5. Print one JSON line summary to stdout. Exit 0 on successful poll
   regardless of `status`. Exit non-zero only on caller-error (bad args,
   library import failure).

Stall threshold: `last_log_mtime_sec_ago > STALL_SEC` (default 900s) AND
the current phase is `running` — i.e., the log has gone quiet but the
pipeline hasn't reported `done` or `failed`.

Dead: PID not alive AND last phase line is NOT `done` (clean exit
should always end with `[phase=done]`).

Phase-line shape expected from the entry script:
    2026-05-21 14:32:18 [phase=training step=1000/2000 loss=2.1]
    2026-05-21 14:55:02 [phase=eval]
    2026-05-21 15:10:44 [phase=done]

Anything matching the regex `\\[phase=([a-z_]+)` will be picked up; the
token immediately after `phase=` is the milestone name.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import subprocess
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

# Make src/ importable so we can call task_workflow.post_event directly.
_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parent
_SRC = _REPO_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from explore_persona_space.task_workflow import post_event  # noqa: E402

log = logging.getLogger("poll_pipeline")

STALL_SEC = 900
PHASE_RE = re.compile(r"\[phase=([a-z_]+)")
DEFAULT_STATE_DIR = _REPO_ROOT / ".claude" / "cache"


@dataclass(frozen=True)
class PollResult:
    status: str  # running | done | stalled | dead
    current_phase: str
    new_milestone: bool
    last_log_mtime_sec_ago: int
    pid_alive: bool
    log_tail_excerpt: str


def _ssh_probe(pod: str, log_path: str, pid_file: str) -> dict[str, str]:
    """One SSH round-trip — returns dict with keys pid_alive, mtime_epoch, log_tail.

    Batches into a single heredoc to keep the SSH cost to one connection.
    """
    heredoc = (
        f"if [ -f {pid_file} ]; then "
        f"  PID=$(cat {pid_file}); "
        f"  if ps -p $PID > /dev/null 2>&1; then echo PID_ALIVE=1; else echo PID_ALIVE=0; fi; "
        f"else echo PID_ALIVE=0; fi; "
        f"if [ -f {log_path} ]; then "
        f"  echo MTIME_EPOCH=$(stat -c %Y {log_path}); "
        f"  echo TAIL_START; tail -500 {log_path}; echo TAIL_END; "
        f"else echo MTIME_EPOCH=0; echo TAIL_START; echo TAIL_END; fi"
    )
    result = subprocess.run(
        ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=15", pod, heredoc],
        capture_output=True,
        text=True,
        timeout=60,
    )
    if result.returncode != 0:
        log.error("ssh failed (rc=%d): %s", result.returncode, result.stderr.strip())
        return {"pid_alive": "0", "mtime_epoch": "0", "log_tail": ""}
    parsed: dict[str, str] = {"pid_alive": "0", "mtime_epoch": "0", "log_tail": ""}
    in_tail = False
    tail_lines: list[str] = []
    for line in result.stdout.splitlines():
        if line == "TAIL_START":
            in_tail = True
            continue
        if line == "TAIL_END":
            in_tail = False
            continue
        if in_tail:
            tail_lines.append(line)
            continue
        if line.startswith("PID_ALIVE="):
            parsed["pid_alive"] = line.split("=", 1)[1].strip()
        elif line.startswith("MTIME_EPOCH="):
            parsed["mtime_epoch"] = line.split("=", 1)[1].strip()
    parsed["log_tail"] = "\n".join(tail_lines)
    return parsed


def _latest_phase(log_tail: str) -> str:
    """Return the milestone name from the most recent `[phase=...]` line, or 'unknown'."""
    for line in reversed(log_tail.splitlines()):
        m = PHASE_RE.search(line)
        if m:
            return m.group(1)
    return "unknown"


def _load_state(state_file: Path, issue: int) -> dict[str, str]:
    if not state_file.exists():
        return {}
    try:
        data = json.loads(state_file.read_text())
    except (json.JSONDecodeError, OSError):
        log.warning("state file %s unreadable; treating as empty", state_file)
        return {}
    return data.get(str(issue), {})


def _save_state(state_file: Path, issue: int, payload: dict[str, str]) -> None:
    state_file.parent.mkdir(parents=True, exist_ok=True)
    all_state: dict[str, dict[str, str]] = {}
    if state_file.exists():
        try:
            all_state = json.loads(state_file.read_text())
        except (json.JSONDecodeError, OSError):
            all_state = {}
    all_state[str(issue)] = payload
    tmp = state_file.with_suffix(state_file.suffix + ".tmp")
    tmp.write_text(json.dumps(all_state, indent=2, sort_keys=True))
    tmp.replace(state_file)


def poll_once(
    *,
    issue: int,
    pod: str,
    log_path: str,
    pid_file: str,
    state_file: Path,
) -> PollResult:
    probe = _ssh_probe(pod, log_path, pid_file)
    pid_alive = probe["pid_alive"] == "1"
    mtime_epoch = int(probe["mtime_epoch"] or "0")
    now_epoch = int(datetime.now(tz=UTC).timestamp())
    last_mtime_ago = now_epoch - mtime_epoch if mtime_epoch > 0 else 10**9
    current_phase = _latest_phase(probe["log_tail"])

    # Decide status.
    if current_phase == "done":
        status = "done"
    elif not pid_alive:
        status = "dead"
    elif last_mtime_ago > STALL_SEC:
        status = "stalled"
    else:
        status = "running"

    # New milestone?
    prev = _load_state(state_file, issue)
    prev_phase = prev.get("phase", "")
    new_milestone = current_phase != prev_phase and current_phase != "unknown"

    if new_milestone:
        try:
            post_event(
                issue,
                "epm:progress",
                by="poll_pipeline",
                note=f"phase transition: {prev_phase or '(start)'} -> {current_phase}",
                phase=current_phase,
                pod=pod,
            )
        except Exception as exc:
            log.error("post_event failed: %s", exc)
            new_milestone = False  # Don't claim we recorded it.

    _save_state(state_file, issue, {"phase": current_phase, "last_mtime_epoch": str(mtime_epoch)})

    tail_excerpt = "\n".join(probe["log_tail"].splitlines()[-5:])
    return PollResult(
        status=status,
        current_phase=current_phase,
        new_milestone=new_milestone,
        last_log_mtime_sec_ago=min(last_mtime_ago, 10**9),
        pid_alive=pid_alive,
        log_tail_excerpt=tail_excerpt,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--issue", type=int, required=True, help="Task / issue number.")
    parser.add_argument("--pod", required=True, help="SSH host alias (e.g. epm-issue-137).")
    parser.add_argument("--log", required=True, help="Remote log file path.")
    parser.add_argument("--pid-file", required=True, help="Remote PID file path.")
    parser.add_argument(
        "--state-file",
        type=Path,
        default=None,
        help="Local cache JSON (default: .claude/cache/poll-pipeline-<N>.json).",
    )
    parser.add_argument("--debug", action="store_true", help="Log to stderr at DEBUG level.")
    args = parser.parse_args(argv)

    logging.basicConfig(
        stream=sys.stderr,
        level=logging.DEBUG if args.debug else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    state_file = args.state_file or (DEFAULT_STATE_DIR / f"poll-pipeline-{args.issue}.json")

    result = poll_once(
        issue=args.issue,
        pod=args.pod,
        log_path=args.log,
        pid_file=args.pid_file,
        state_file=state_file,
    )

    print(
        json.dumps(
            {
                "status": result.status,
                "current_phase": result.current_phase,
                "new_milestone": result.new_milestone,
                "last_log_mtime_sec_ago": result.last_log_mtime_sec_ago,
                "pid_alive": result.pid_alive,
                "log_tail_excerpt": result.log_tail_excerpt,
            }
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
