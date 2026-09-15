"""Durable VM monitor for the #825 turnk5 backend lane; never launches work.

The only backend operation is one invocation of the sanctioned backend_poll.py
per interval. That existing poller owns sentinel processing and may update a
handle or provision an authorized failover. A slow poll raises a soft deadline
alert and remains alive. This wrapper adds no launch or continuation logic and
invokes no LLM.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import signal
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

VALID_STATUSES = {"running", "done", "gate", "stalled", "dead", "pid-stale-workload-live"}


def utcnow() -> str:
    """Return a fresh UTC observation timestamp."""
    return datetime.now(UTC).isoformat()


def atomic_json(path: Path, value: dict) -> None:
    """Persist a complete status with fsync and atomic replacement."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def log_tail(path: Path, *, limit: int = 65536) -> dict:
    """Read bounded recent output while preserving the complete original log."""
    stat = path.stat()
    with path.open("rb") as handle:
        handle.seek(max(0, stat.st_size - limit))
        tail = handle.read().decode("utf-8", errors="replace")
    return {
        "path": str(path),
        "bytes": stat.st_size,
        "mtime_epoch": stat.st_mtime,
        "tail": tail[-12000:],
    }


def inspect_handle(path: Path, launched_after: float) -> dict:
    """Reject prelaunch/malformed sidecars and expose only nonsecret identity fields."""
    try:
        with path.open("rb") as handle:
            stat = os.fstat(handle.fileno())
            raw = handle.read()
    except FileNotFoundError:
        return {"ready": False, "reason": "handle_missing"}
    digest = hashlib.sha256(raw).hexdigest()
    value = json.loads(raw)
    if not isinstance(value, dict) or not all(
        value.get(key) for key in ("backend", "job_id", "pod_name")
    ):
        raise ValueError("handle is missing backend/job_id/pod_name")
    extra = value.get("extra") or {}
    if not isinstance(extra, dict):
        raise ValueError("handle extra field must be a dictionary")
    identity = {
        "ready": True,
        "sha256": digest,
        "mtime_epoch": stat.st_mtime,
        "backend": value["backend"],
        "job_id": value["job_id"],
        "pod_name": value["pod_name"],
        "attempt_id": str(extra.get("attempt_id") or extra.get("runpod_attempt_id") or ""),
        "path": str(path),
    }
    if stat.st_mtime < launched_after:
        return identity | {"ready": False, "reason": "handle_predates_launch"}
    return identity


def parse_poll(path: Path) -> dict:
    """Accept a fresh final PollResult JSON line; never turn parse errors into death."""
    lines = path.read_text().splitlines()
    if not lines:
        raise ValueError("backend poll returned empty stdout")
    result = json.loads(lines[-1])
    required = {"status", "current_phase", "pid_alive", "log_tail_excerpt"}
    if not isinstance(result, dict) or not required <= result.keys():
        raise ValueError("backend poll result lacks required state/log fields")
    if result["status"] not in VALID_STATUSES:
        raise ValueError(f"unknown backend status: {result['status']!r}")
    return result


def same_run(left: dict, right: dict) -> bool:
    """Ignore phase clocks while detecting a backend/job/incarnation change."""
    return all(left[key] == right[key] for key in ("backend", "job_id", "pod_name", "attempt_id"))


class Monitor:
    """Persist heartbeat and bounded poll state, retrying transient probe failures."""

    def __init__(self, out: Path, handle_file: Path, launched_after: float):
        """Start a distinct monitor run; refuse to overwrite previous observations."""
        out.mkdir(parents=True, exist_ok=False)
        self.out, self.handle_file, self.launched_after = out, handle_file, launched_after
        self.stopping = False
        self.began = time.monotonic()
        self.state = {
            "monitor_pid": os.getpid(),
            "monitor_started_at": utcnow(),
            "launch_after_epoch": launched_after,
            "handle_file": str(handle_file),
            "polls_completed": 0,
            "consecutive_poll_errors": 0,
            "last_backend_result": None,
        }
        self.publish("waiting_handle")

    def publish(self, status: str, **changes) -> None:
        """Publish liveness independently of the timestamp of the latest backend probe."""
        self.state.update(
            changes,
            monitor_status=status,
            heartbeat_at=utcnow(),
            heartbeat_epoch=time.time(),
            elapsed_seconds=time.monotonic() - self.began,
        )
        atomic_json(self.out / "status.json", self.state)

    def event(self, kind: str, **details) -> None:
        """Append a durable observation/event; log errors and terminal states promptly."""
        row = {"at": utcnow(), "kind": kind, **details}
        with (self.out / "observations.jsonl").open("a") as handle:
            handle.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        if kind in {
            "poll_error",
            "terminal",
            "handle_error",
            "backend_alert",
            "stopped",
            "poll_overdue",
            "shutdown_pending",
        }:
            print(json.dumps(row, sort_keys=True), flush=True)

    def pause(self, duration: float, heartbeat: float, status: str) -> None:
        """Keep monitor liveness current while waiting for the next bounded interval."""
        deadline = time.monotonic() + duration
        while not self.stopping and time.monotonic() < deadline:
            self.publish(status)
            time.sleep(min(heartbeat, max(0, deadline - time.monotonic())))

    def poll(self, command: list[str], *, timeout: float, heartbeat: float) -> dict:
        """Keep a slow backend operation alive, alerting after its soft deadline."""
        index = self.state["polls_completed"] + 1
        stdout_path, stderr_path = (
            self.out / f"poll_{index:05d}.{stream}.log" for stream in ("stdout", "stderr")
        )
        began = time.monotonic()
        overdue, shutdown_announced = False, False
        with stdout_path.open("w") as stdout, stderr_path.open("w") as stderr:
            child = subprocess.Popen(
                command,
                stdin=subprocess.DEVNULL,
                stdout=stdout,
                stderr=stderr,
                start_new_session=True,
            )
            try:
                while child.poll() is None:
                    if not overdue and time.monotonic() - began >= timeout:
                        overdue = True
                        self.event(
                            "poll_overdue",
                            threshold_seconds=timeout,
                            elapsed_seconds=time.monotonic() - began,
                            child_pid=child.pid,
                            action="child kept alive; root inspection required",
                        )
                    if self.stopping and not shutdown_announced:
                        shutdown_announced = True
                        self.event(
                            "shutdown_pending",
                            child_pid=child.pid,
                            action="finish active backend operation before stopping monitor",
                        )
                    self.publish(
                        "shutdown_pending"
                        if self.stopping
                        else "poll_overdue"
                        if overdue
                        else "polling",
                        poll_child_pid=child.pid,
                        poll_started_at=utcnow()
                        if "poll_started_at" not in self.state
                        else self.state["poll_started_at"],
                        poll_elapsed_seconds=time.monotonic() - began,
                        poll_overdue=overdue,
                        poll_stdout=log_tail(stdout_path),
                        poll_stderr=log_tail(stderr_path),
                    )
                    time.sleep(heartbeat)
                if child.returncode != 0:
                    raise RuntimeError(f"backend poll exited with code {child.returncode}")
                return parse_poll(stdout_path)
            except OSError as error:
                # A heartbeat/log-write failure must not orphan or overlap the
                # backend poll, which can be provisioning a sanctioned failover.
                print(
                    json.dumps(
                        {
                            "at": utcnow(),
                            "kind": "monitor_io_failure",
                            "child_pid": child.pid,
                            "error": str(error),
                            "action": "waiting for active backend operation",
                        }
                    ),
                    file=sys.stderr,
                    flush=True,
                )
                raise
            finally:
                while child.poll() is None:
                    time.sleep(heartbeat)
                self.state.update(
                    polls_completed=index,
                    poll_child_pid=None,
                    poll_exit_code=child.returncode,
                    last_poll_overdue=overdue,
                    poll_overdue=False,
                    poll_stdout=log_tail(stdout_path),
                    poll_stderr=log_tail(stderr_path),
                )

    def run(
        self,
        command: list[str],
        *,
        poll_timeout: float = 180,
        heartbeat: float = 5,
        initial_interval: float = 20,
        later_interval: float = 60,
    ) -> int:
        """Wait for a fresh handle, then poll until a verified terminal result or stop."""
        stale_hash, stale_identity, acquired_at = None, None, None
        while not self.stopping:
            interval = (
                initial_interval
                if acquired_at is None or time.monotonic() - acquired_at < 120
                else later_interval
            )
            try:
                candidate = inspect_handle(self.handle_file, self.launched_after)
                if not candidate["ready"]:
                    stale_hash = candidate.get("sha256", stale_hash)
                    if "job_id" in candidate:
                        stale_identity = candidate
                    self.publish("waiting_handle", handle_wait=candidate)
                    self.pause(interval, heartbeat, "waiting_handle")
                    continue
                if candidate["sha256"] == stale_hash or (
                    stale_identity is not None and same_run(candidate, stale_identity)
                ):
                    self.publish("waiting_handle", handle_wait={"reason": "unchanged_stale_handle"})
                    self.pause(interval, heartbeat, "waiting_handle")
                    continue
                if acquired_at is None:
                    acquired_at = time.monotonic()
                self.publish("polling", handle=candidate, poll_started_at=utcnow())
                result = self.poll(command, timeout=poll_timeout, heartbeat=heartbeat)
                after = inspect_handle(self.handle_file, self.launched_after)
                if not after["ready"] or not same_run(candidate, after):
                    raise RuntimeError(
                        "handle changed during backend poll; retrying fresh identity"
                    )
                observed = utcnow()
                self.state.update(
                    last_backend_result=result,
                    backend_observed_at=observed,
                    consecutive_poll_errors=0,
                    last_poll_error=None,
                )
                self.event("backend_observation", handle=candidate, result=result)
                if self.stopping:
                    break
                if result["status"] in {"done", "dead"}:
                    self.publish(
                        "terminal",
                        finished_at=utcnow(),
                        backend_status=result["status"],
                        completion_scope="GPU artifacts complete; CPU analysis pending"
                        if result["status"] == "done"
                        else "GPU backend failed",
                        experiment_complete=False,
                    )
                    self.event("terminal", status=result["status"], result=result)
                    return 0 if result["status"] == "done" else 1
                if result["status"] in {"gate", "stalled", "pid-stale-workload-live"} or result.get(
                    "reachability_alarm"
                ):
                    self.event("backend_alert", result=result)
                self.publish("waiting_poll", backend_status=result["status"])
            except (OSError, ValueError, RuntimeError) as error:
                if self.stopping:
                    break
                self.state["consecutive_poll_errors"] += 1
                self.publish("retrying_poll", last_poll_error=f"{type(error).__name__}: {error}")
                self.event(
                    "poll_error",
                    error=self.state["last_poll_error"],
                    consecutive_errors=self.state["consecutive_poll_errors"],
                )
            self.pause(interval, heartbeat, self.state["monitor_status"])
        self.publish("stopped", finished_at=utcnow())
        self.event("stopped", reason="monitor stop requested")
        return 130


def main_checkout() -> Path:
    """Resolve the shared main checkout without importing backend side effects."""
    result = subprocess.run(
        ["git", "rev-parse", "--path-format=absolute", "--git-common-dir"],
        cwd=Path(__file__).resolve().parent,
        capture_output=True,
        text=True,
        check=True,
        timeout=30,
    )
    common = Path(result.stdout.strip())
    if common.name != ".git" or not common.is_dir():
        raise RuntimeError("cannot resolve shared main checkout")
    return common.parent


def main() -> int:
    """Monitor one #825 lane; backend_poll remains the sole backend operation."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--launch-after-epoch", type=float, required=True)
    parser.add_argument("--handle-file", type=Path)
    parser.add_argument("--lane-suffix", default="turnk5")
    args = parser.parse_args()
    if not args.lane_suffix or not all(c.isalnum() or c == "-" for c in args.lane_suffix):
        raise ValueError("invalid lane suffix")
    root = main_checkout()
    handle_file = (
        args.handle_file or root / ".claude/cache" / f"issue-825-{args.lane_suffix}-handle.json"
    )
    monitor = Monitor(args.out, handle_file, args.launch_after_epoch)

    def request_stop(_signum, _frame):
        """Finish any active backend operation, then persist the stopped observation."""
        monitor.stopping = True

    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)
    command = [
        sys.executable,
        str(root / "scripts/backend_poll.py"),
        "--issue",
        "825",
        "--handle-file",
        str(handle_file),
        "--lane-suffix",
        args.lane_suffix,
    ]
    return monitor.run(command)


if __name__ == "__main__":
    raise SystemExit(main())
