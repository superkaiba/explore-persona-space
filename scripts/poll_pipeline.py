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

from explore_persona_space.task_workflow import (  # noqa: E402
    post_event,
)

log = logging.getLogger("poll_pipeline")

STALL_SEC = 900
PHASE_RE = re.compile(r"\[phase=([a-z_]+)")
DEFAULT_STATE_DIR = _REPO_ROOT / ".claude" / "cache"
# Pod-side sentinel file convention (canonical pod-side escalation channel
# per CLAUDE.md "Pod-side code NEVER shells out to scripts/task.py" rule).
# Pod dispatchers / scripts write JSON sentinels at
# /workspace/logs/issue-<N>-<reason>-failed.json; the poller picks them up,
# posts `epm:failure v1` with the rich payload, and flips status to blocked.
SENTINEL_REMOTE_GLOB_TEMPLATE = "/workspace/logs/issue-{issue}-*-failed.json"
# Sentinel boundary markers in the SSH heredoc output.
_SENTINEL_LIST_START = "SENTINEL_LIST_START"
_SENTINEL_LIST_END = "SENTINEL_LIST_END"
_SENTINEL_BODY_START = "SENTINEL_BODY_START"
_SENTINEL_BODY_END = "SENTINEL_BODY_END"


@dataclass(frozen=True)
class PollResult:
    status: str  # running | done | stalled | dead
    current_phase: str
    new_milestone: bool
    last_log_mtime_sec_ago: int
    pid_alive: bool
    log_tail_excerpt: str


def _ssh_probe(
    pod: str,
    log_path: str,
    pid_file: str,
    *,
    issue: int | None = None,
) -> dict[str, object]:
    """One SSH round-trip — returns dict with pid_alive, mtime_epoch, log_tail, sentinels.

    Batches into a single heredoc to keep the SSH cost to one connection.

    When ``issue`` is provided, ALSO scans the pod for pod-side failure
    sentinels matching ``/workspace/logs/issue-<issue>-*-failed.json``
    (see CLAUDE.md "Pod-side code NEVER shells out to scripts/task.py").
    Each sentinel's body is read and parsed as JSON; malformed payloads
    are returned as the raw string so the caller can still surface them.
    """
    sentinel_glob = SENTINEL_REMOTE_GLOB_TEMPLATE.format(issue=issue) if issue is not None else None

    sentinel_block = ""
    if sentinel_glob is not None:
        # List matching paths, then cat each one wrapped in BODY_START/END so
        # the caller can pull each file's contents out of one stream.
        sentinel_block = (
            f"echo {_SENTINEL_LIST_START}; "
            f"ls -1 {sentinel_glob} 2>/dev/null || true; "
            f"echo {_SENTINEL_LIST_END}; "
            f"for f in $(ls -1 {sentinel_glob} 2>/dev/null || true); do "
            f'  echo "{_SENTINEL_BODY_START} $f"; '
            f'  cat "$f" 2>/dev/null || true; '
            f'  echo ""; '
            f"  echo {_SENTINEL_BODY_END}; "
            f"done"
        )

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
    if sentinel_block:
        heredoc = f"{heredoc}; {sentinel_block}"

    result = subprocess.run(
        ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=15", pod, heredoc],
        capture_output=True,
        text=True,
        timeout=60,
    )
    if result.returncode != 0:
        log.error("ssh failed (rc=%d): %s", result.returncode, result.stderr.strip())
        return {"pid_alive": "0", "mtime_epoch": "0", "log_tail": "", "sentinels": []}

    return _parse_ssh_probe_stdout(result.stdout)


def _decode_sentinel_body(body_text: str) -> object:
    """Parse a sentinel JSON body. On JSONDecodeError, return the raw text.

    The caller surfaces malformed sentinels as `epm:failure v1` anyway so the
    user can drill into the pod and diagnose; we never silently drop them.
    """
    if not body_text:
        return {}
    try:
        return json.loads(body_text)
    except json.JSONDecodeError:
        return body_text


class _ProbeStdoutParser:
    """Small state machine that walks the heredoc stdout line by line."""

    def __init__(self) -> None:
        self.section = "preamble"  # preamble | tail | sentinel_list | sentinel_body
        self.parsed: dict[str, object] = {
            "pid_alive": "0",
            "mtime_epoch": "0",
            "log_tail": "",
            "sentinels": [],
        }
        self.tail_lines: list[str] = []
        self.sentinel_paths: list[str] = []
        self.sentinels: list[dict[str, object]] = []
        self.cur_sentinel_path: str | None = None
        self.cur_sentinel_lines: list[str] = []

    def feed_line(self, line: str) -> None:
        if self._handle_transition(line):
            return
        if self.section == "tail":
            self.tail_lines.append(line)
        elif self.section == "sentinel_list":
            if line.strip():
                self.sentinel_paths.append(line.strip())
        elif self.section == "sentinel_body":
            self.cur_sentinel_lines.append(line)
        elif line.startswith("PID_ALIVE="):
            self.parsed["pid_alive"] = line.split("=", 1)[1].strip()
        elif line.startswith("MTIME_EPOCH="):
            self.parsed["mtime_epoch"] = line.split("=", 1)[1].strip()

    def _handle_transition(self, line: str) -> bool:
        """Return True if the line was consumed as a section boundary."""
        if line in {"TAIL_START", "TAIL_END", _SENTINEL_LIST_START, _SENTINEL_LIST_END}:
            self.section = "tail" if line == "TAIL_START" else "preamble"
            if line == _SENTINEL_LIST_START:
                self.section = "sentinel_list"
            return True
        if line.startswith(f"{_SENTINEL_BODY_START} "):
            self.section = "sentinel_body"
            self.cur_sentinel_path = line[len(_SENTINEL_BODY_START) + 1 :].strip()
            self.cur_sentinel_lines = []
            return True
        if line == _SENTINEL_BODY_END:
            self.section = "preamble"
            body = "\n".join(self.cur_sentinel_lines).strip()
            self.sentinels.append(
                {
                    "path": self.cur_sentinel_path,
                    "payload": _decode_sentinel_body(body),
                }
            )
            self.cur_sentinel_path = None
            self.cur_sentinel_lines = []
            return True
        return False

    def finalize(self) -> dict[str, object]:
        self.parsed["log_tail"] = "\n".join(self.tail_lines)
        seen_paths = {s["path"] for s in self.sentinels}
        for sp in self.sentinel_paths:
            if sp not in seen_paths:
                self.sentinels.append({"path": sp, "payload": ""})
        self.parsed["sentinels"] = self.sentinels
        return self.parsed


def _parse_ssh_probe_stdout(stdout: str) -> dict[str, object]:
    """Parse the heredoc stdout into the structured probe dict.

    Split out so the unit tests can feed synthetic stdout without spawning
    a real SSH subprocess. Delegates to _ProbeStdoutParser to keep
    cyclomatic complexity per function low.
    """
    parser = _ProbeStdoutParser()
    for line in stdout.splitlines():
        parser.feed_line(line)
    return parser.finalize()


def _latest_phase(log_tail: str) -> str:
    """Return the milestone name from the most recent `[phase=...]` line, or 'unknown'."""
    for line in reversed(log_tail.splitlines()):
        m = PHASE_RE.search(line)
        if m:
            return m.group(1)
    return "unknown"


def _load_state(state_file: Path, issue: int) -> dict[str, object]:
    """Load per-issue state from the cache file.

    The state schema accumulates keys over time:
      - ``phase``: last-seen `[phase=...]` token (str).
      - ``last_mtime_epoch``: last-seen log mtime in epoch seconds (str).
      - ``emitted_sentinels``: list of pod-side sentinel paths already
        translated to `epm:failure v1` markers (idempotency guard).
    """
    if not state_file.exists():
        return {}
    try:
        data = json.loads(state_file.read_text())
    except (json.JSONDecodeError, OSError):
        log.warning("state file %s unreadable; treating as empty", state_file)
        return {}
    return data.get(str(issue), {})


def _save_state(state_file: Path, issue: int, payload: dict[str, object]) -> None:
    state_file.parent.mkdir(parents=True, exist_ok=True)
    all_state: dict[str, dict[str, object]] = {}
    if state_file.exists():
        try:
            all_state = json.loads(state_file.read_text())
        except (json.JSONDecodeError, OSError):
            all_state = {}
    all_state[str(issue)] = payload
    tmp = state_file.with_suffix(state_file.suffix + ".tmp")
    tmp.write_text(json.dumps(all_state, indent=2, sort_keys=True))
    tmp.replace(state_file)


def _emit_sentinel_failure_markers(
    issue: int,
    pod: str,
    sentinels: list[dict[str, object]],
    *,
    already_emitted_paths: set[str],
    poster=None,
    flipper=None,
) -> set[str]:
    """For each new sentinel, post `epm:failure v1` with the rich payload + flip status to blocked.

    Idempotent: skips sentinel paths already in ``already_emitted_paths``.
    Returns the set of newly-emitted paths so the caller can persist them
    to the state file and avoid double-posting on subsequent ticks.

    ``poster`` and ``flipper`` exist to make this unit-testable without
    mutating real task state. They default to the live ``post_event`` and
    ``set_status`` from ``task_workflow``.
    """
    if poster is None:
        from explore_persona_space.task_workflow import post_event as poster
    if flipper is None:
        from explore_persona_space.task_workflow import set_status as flipper

    newly_emitted: set[str] = set()
    blocked_already = False
    for sentinel in sentinels:
        path = str(sentinel.get("path") or "")
        if not path or path in already_emitted_paths:
            continue
        payload = sentinel.get("payload")
        # Extract structured fields when the body parsed as JSON; else
        # fall back to the raw string in `reason`.
        if isinstance(payload, dict):
            failure_class = str(payload.get("failure_class", "rig"))
            phase = str(payload.get("phase", "unknown"))
            condition = payload.get("condition")
            reason = str(payload.get("reason") or payload.get("policy") or "")
            note = (
                f"pod-side sentinel {path}: phase={phase}, condition={condition}, reason={reason}"
            )
            extras = {
                "failure_class": failure_class,
                "phase": phase,
                "condition": condition,
                "sentinel_path": path,
                "pod": pod,
            }
        else:
            failure_class = "rig"
            note = f"pod-side sentinel {path} (unparseable body): {payload!r}"[:1000]
            extras = {
                "failure_class": failure_class,
                "sentinel_path": path,
                "pod": pod,
            }
        try:
            poster(issue, "epm:failure", by="poll_pipeline", note=note, **extras)
        except Exception as exc:
            log.error("poll_pipeline: post_event failed for sentinel %s: %s", path, exc)
            continue  # don't mark as emitted so we retry next tick
        newly_emitted.add(path)
        if not blocked_already:
            try:
                flipper(issue, "blocked", note=f"pod-side sentinel: {path}")
                blocked_already = True
            except ValueError as exc:
                # set_status raises ValueError if "blocked" is not in STATUSES
                # OR re-raises a flock collision; both are reportable but
                # not fatal — the marker is posted and the next tick retries.
                log.error("poll_pipeline: set_status(blocked) failed: %s", exc)
            except Exception as exc:
                log.error("poll_pipeline: set_status(blocked) failed: %s", exc)
    return newly_emitted


def poll_once(
    *,
    issue: int,
    pod: str,
    log_path: str,
    pid_file: str,
    state_file: Path,
) -> PollResult:
    probe = _ssh_probe(pod, log_path, pid_file, issue=issue)
    pid_alive = probe["pid_alive"] == "1"
    mtime_epoch = int(probe["mtime_epoch"] or "0")
    now_epoch = int(datetime.now(tz=UTC).timestamp())
    last_mtime_ago = now_epoch - mtime_epoch if mtime_epoch > 0 else 10**9
    log_tail = str(probe["log_tail"])
    sentinels = probe.get("sentinels") or []  # list[dict]
    current_phase = _latest_phase(log_tail)

    # Sentinel handling: if any pod-side failure sentinel is present, we
    # treat this as a terminal failure for the polling loop — post the
    # marker(s), flip status to blocked, return status="dead" so the
    # /issue skill stops scheduling further ticks.
    prev = _load_state(state_file, issue)
    already_emitted: set[str] = set(prev.get("emitted_sentinels", []))
    sentinel_paths_seen: list[str] = [str(s.get("path") or "") for s in sentinels if s.get("path")]
    sentinel_triggered = bool(sentinel_paths_seen)
    newly_emitted: set[str] = set()
    if sentinel_triggered:
        newly_emitted = _emit_sentinel_failure_markers(
            issue, pod, list(sentinels), already_emitted_paths=already_emitted
        )

    # Decide status. Sentinel detection is the highest-priority signal.
    if sentinel_triggered:
        status = "dead"
        current_phase = "failed_sentinel"
    elif current_phase == "done":
        status = "done"
    elif not pid_alive:
        status = "dead"
    elif last_mtime_ago > STALL_SEC:
        status = "stalled"
    else:
        status = "running"

    # New milestone?
    prev_phase = prev.get("phase", "")
    new_milestone = current_phase != prev_phase and current_phase != "unknown"

    if new_milestone and not sentinel_triggered:
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

    # Persist state — phase + last mtime + emitted-sentinels (so we don't double-post).
    emitted_after_tick = sorted(already_emitted | newly_emitted)
    _save_state(
        state_file,
        issue,
        {
            "phase": current_phase,
            "last_mtime_epoch": str(mtime_epoch),
            "emitted_sentinels": emitted_after_tick,
        },
    )

    tail_excerpt = "\n".join(log_tail.splitlines()[-5:])
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
