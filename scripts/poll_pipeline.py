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

1. Drain pod-side sentinel files (`/workspace/logs/issue-<N>-*.json`,
   skipping `*.processed`). Each sentinel was written by a pod-side
   dispatcher that cannot shell out to `scripts/task.py` (CLAUDE.md
   "Pod-side code NEVER shells out" rule). The poller parses each
   sentinel, posts the carried `epm:<kind>` marker from the local VM
   via `task_workflow.post_event`, then renames the sentinel to
   `<path>.processed` so it posts exactly once. If a sentinel carries a
   non-empty ``gate`` field, the poll returns ``status=gate`` with that
   gate name in the JSON output so the orchestrator parks at a user
   gate instead of continuing the polling loop.
2. SSH to the pod (one heredoc batching: PID liveness, log mtime, log tail).
3. Parse the latest `[phase=...]` line from the log tail.
4. If new milestone vs the cached previous phase, post `epm:progress`
   to the task's events.jsonl via the local-VM `task_workflow.post_event`
   library (NOT on the pod).
5. Decide status: `done` | `gate` | `stalled` | `dead` | `running`.
6. Print one JSON line summary to stdout. Exit 0 on successful poll
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

Sentinel schema (v1) — written by pod-side dispatchers, drained here:

    filename: /workspace/logs/issue-<N>-<kind_slug>-<epoch_seconds>.json
        kind_slug = kind with `:` -> `_` (e.g. ``epm_fact_candidates``).
    payload (JSON, dict):
        {
          "sentinel_schema_version": 1,                  # required, must be 1
          "task_id": <int>,                              # informational
          "kind": "<full kind, e.g. 'epm:fact-candidates'>",
          "version": <int>,                              # marker version
          "gate": "<gate name>" | null,                  # if set, poll returns status=gate
          "blocks_pipeline": true|false,                 # informational
          "note": "<marker note body>",                  # may also be sent as 'payload'
          "by": "<author>",
          "ts": "<ISO-8601 UTC>",
        }

Unknown schema versions are logged + skipped (not renamed) so a future
poller can re-process them. Malformed JSON / missing required fields are
logged + skipped likewise — the sentinel is left in place so the next
poller (or a human) can inspect it.
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
from typing import Any

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

# Schema version the poller knows how to parse. Bump in lockstep with the
# pod-side writer (currently ``run_experiment_<N>.py::SENTINEL_SCHEMA_VERSION``).
# Newer schemas are skipped + logged, never silently mis-parsed.
SENTINEL_SCHEMA_VERSION_SUPPORTED = 1

# Required keys in every parsed sentinel payload. ``payload`` is accepted as
# a synonym for ``note`` for forward-compat with sentinels that put the
# marker body under that key.
_SENTINEL_REQUIRED_KEYS: tuple[str, ...] = (
    "sentinel_schema_version",
    "kind",
    "version",
)


@dataclass(frozen=True)
class PollResult:
    status: str  # running | done | gate | stalled | dead
    current_phase: str
    new_milestone: bool
    last_log_mtime_sec_ago: int
    pid_alive: bool
    log_tail_excerpt: str
    gate: str | None = None  # set when a drained sentinel carried a non-empty gate
    sentinels_processed: int = 0


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


def _ssh_drain_sentinels(pod: str, issue: int) -> list[tuple[str, str]]:
    """List + cat unprocessed sentinels in one SSH round-trip.

    Globs ``/workspace/logs/issue-<issue>-*.json`` (skipping ``*.processed``),
    emits each as ``SENTINEL_START <path>\\n<body>\\nSENTINEL_END`` so the
    caller can parse multiple sentinels from one stdout blob. Files are NOT
    renamed here — the rename happens via ``_ssh_mark_processed`` only after
    the marker post succeeds, so a mid-tick crash leaves the sentinel
    un-renamed and the next poll retries it (idempotent).

    Returns a list of ``(remote_path, body)`` pairs (possibly empty). On
    SSH failure returns an empty list and logs the error.
    """
    # The glob is path-terminal `.json` and explicitly excludes `.processed`.
    # ``shopt -s nullglob`` makes an empty glob expand to nothing instead of
    # the literal pattern so we don't accidentally cat a path called e.g.
    # ``/workspace/logs/issue-444-*.json``.
    heredoc = (
        f"shopt -s nullglob; "
        f"for f in /workspace/logs/issue-{issue}-*.json; do "
        f'  case "$f" in *.processed) continue ;; esac; '
        f'  echo "SENTINEL_START $f"; '
        f'  cat "$f"; '
        f'  echo ""; echo "SENTINEL_END"; '
        f"done"
    )
    result = subprocess.run(
        ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=15", pod, heredoc],
        capture_output=True,
        text=True,
        timeout=60,
    )
    if result.returncode != 0:
        log.error("ssh drain failed (rc=%d): %s", result.returncode, result.stderr.strip())
        return []
    sentinels: list[tuple[str, str]] = []
    current_path: str | None = None
    current_body: list[str] = []
    for line in result.stdout.splitlines():
        if line.startswith("SENTINEL_START "):
            current_path = line[len("SENTINEL_START ") :].strip()
            current_body = []
        elif line == "SENTINEL_END":
            if current_path is not None:
                sentinels.append((current_path, "\n".join(current_body).strip()))
            current_path = None
            current_body = []
        elif current_path is not None:
            current_body.append(line)
    return sentinels


def _ssh_mark_processed(pod: str, remote_path: str) -> bool:
    """Rename ``remote_path`` -> ``remote_path + '.processed'`` on the pod.

    Returns True on success. Logs + returns False on failure (the sentinel
    is left in place; next poll tick will re-attempt). We use ``mv -n`` (no
    clobber) so a pre-existing ``.processed`` file is preserved — the
    sentinel writer never reuses epoch-tagged filenames, so a collision
    here would itself be a bug worth surfacing.
    """
    # Single-quote the remote path to neutralise shell metacharacters; the
    # writer's filename is ``issue-<N>-<kind_slug>-<epoch>.json`` so it's
    # safe by construction, but defence-in-depth costs nothing.
    quoted = "'" + remote_path.replace("'", "'\\''") + "'"
    cmd = f"mv -n {quoted} {quoted}.processed"
    result = subprocess.run(
        ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=15", pod, cmd],
        capture_output=True,
        text=True,
        timeout=30,
    )
    if result.returncode != 0:
        log.error(
            "ssh mv failed for %s (rc=%d): %s",
            remote_path,
            result.returncode,
            result.stderr.strip(),
        )
        return False
    return True


def _parse_sentinel(remote_path: str, body: str) -> dict[str, Any] | None:
    """Decode + validate one sentinel body. Returns the dict on success.

    Returns None (and logs) for any of: empty body, JSON decode error,
    non-dict payload, missing required keys, unsupported schema version.
    The sentinel is left un-renamed in these cases so a future poller (or
    a human) can inspect it.
    """
    if not body:
        log.warning("sentinel %s is empty; skipping", remote_path)
        return None
    try:
        data = json.loads(body)
    except json.JSONDecodeError as exc:
        log.warning("sentinel %s has invalid JSON (%s); skipping", remote_path, exc)
        return None
    if not isinstance(data, dict):
        log.warning("sentinel %s is not a JSON object; skipping", remote_path)
        return None
    missing = [k for k in _SENTINEL_REQUIRED_KEYS if k not in data]
    if missing:
        log.warning("sentinel %s missing required keys %s; skipping", remote_path, missing)
        return None
    schema_version = data.get("sentinel_schema_version")
    if schema_version != SENTINEL_SCHEMA_VERSION_SUPPORTED:
        log.warning(
            "sentinel %s has unsupported schema_version=%r (supported: %d); skipping",
            remote_path,
            schema_version,
            SENTINEL_SCHEMA_VERSION_SUPPORTED,
        )
        return None
    return data


def _drain_sentinels(*, issue: int, pod: str) -> tuple[int, str | None]:
    """Drain pod-side sentinels for this task; post markers from the VM.

    Returns ``(processed_count, gate_name_or_None)``. ``gate_name`` is the
    first non-empty ``gate`` field across processed sentinels (sentinels
    are processed in glob order, which is filename order, which is
    chronological by epoch-suffix). When set, the caller should stop the
    polling loop and surface the gate to the user.

    Each successfully-posted sentinel is renamed to ``<path>.processed``
    so the next tick won't re-post the same marker. If the marker post or
    the rename fails for an individual sentinel, the sentinel is left in
    place and a warning is logged; subsequent ticks will retry.
    """
    sentinels = _ssh_drain_sentinels(pod, issue)
    processed = 0
    gate: str | None = None
    for remote_path, body in sentinels:
        data = _parse_sentinel(remote_path, body)
        if data is None:
            continue
        kind = data["kind"]
        version = int(data["version"])
        note = data.get("note")
        if note is None:
            note = data.get("payload")
        if note is not None and not isinstance(note, str):
            note = json.dumps(note, ensure_ascii=False)
        by = data.get("by") or "pod-sentinel"
        try:
            post_event(issue, kind, version=version, by=by, note=note)
        except Exception as exc:
            # Don't rename on post failure — next tick will retry. We log
            # at error so an operator can see repeated failures.
            log.error(
                "post_event failed for sentinel %s (kind=%s): %s",
                remote_path,
                kind,
                exc,
            )
            continue
        if not _ssh_mark_processed(pod, remote_path):
            # Marker is posted but rename failed; on the next tick we'd
            # re-post and create a duplicate event. Surface loudly so the
            # operator can rename manually.
            log.error(
                "marker %s posted from sentinel %s but rename failed; "
                "future ticks may duplicate. Rename manually with: "
                "ssh %s mv %s %s.processed",
                kind,
                remote_path,
                pod,
                remote_path,
                remote_path,
            )
            # Still count as processed so the caller's accounting is honest.
        processed += 1
        sentinel_gate = data.get("gate")
        if gate is None and isinstance(sentinel_gate, str) and sentinel_gate:
            gate = sentinel_gate
    return processed, gate


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
    # Drain pod-side sentinels FIRST — posting any pending markers from the
    # VM. A user-gate sentinel (e.g. epm:fact-candidates) takes precedence
    # over the phase=done check so the orchestrator parks at the gate even
    # if the pipeline subsequently reached done.
    sentinels_processed, gate = _drain_sentinels(issue=issue, pod=pod)

    probe = _ssh_probe(pod, log_path, pid_file)
    pid_alive = probe["pid_alive"] == "1"
    mtime_epoch = int(probe["mtime_epoch"] or "0")
    now_epoch = int(datetime.now(tz=UTC).timestamp())
    last_mtime_ago = now_epoch - mtime_epoch if mtime_epoch > 0 else 10**9
    current_phase = _latest_phase(probe["log_tail"])

    # Decide status. Gate sentinel wins over done — a user must answer
    # before the pipeline (or the orchestrator) advances further. The
    # phase=done check still runs (we want to know the pipeline finished)
    # but ``status`` reflects the gate so the orchestrator parks.
    if gate is not None:
        status = "gate"
    elif current_phase == "done":
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
        gate=gate,
        sentinels_processed=sentinels_processed,
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
                "gate": result.gate,
                "sentinels_processed": result.sentinels_processed,
            }
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
