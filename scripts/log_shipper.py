#!/usr/bin/env python
"""Pod-side log shipper. Streams local log file lines (and periodic
runpod_status events) into Sagan's ``agent_run_events`` table so the
dashboard's ``/agent/<id>`` page renders live training output without
the user having to SSH in.

Bootstrapped by ``scripts/bootstrap_pod.sh`` after SSH is up. Runs
forever; ``--max-runtime-secs`` (default 24h) caps the wall clock so an
orphaned shipper doesn't burn API quota.

Required env (from the orchestrator):

* ``SAGAN_AGENT_RUN_ID`` — UUID of the parent ``agent_runs`` row. The
  ``/issue`` skill creates this in Step 5; pass it through to the pod.
* ``SAGAN_BASE_URL`` — Defaults to ``https://sagan.superkaiba.com``.
* ``SAGAN_API_TOKEN`` — Bearer token (``sk_…`` API token preferred so
  the shipper survives session rotation).

Optional flags:

* ``--log-path``       file to tail (defaults to ``nohup.out`` in cwd)
* ``--batch-size``      lines per POST (default 50)
* ``--flush-interval``  seconds between POSTs even when under batch-size (default 1.0)
* ``--status-interval`` seconds between ``runpod_status`` heartbeats (default 60)
* ``--max-runtime-secs`` wall-time cap (default 86400)

Failure model: a single failed POST is logged to stderr and the line
batch is dropped (do NOT block training). If three consecutive POSTs
fail the shipper backs off to ``--flush-interval * 10`` for the next
batch.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import socket
import subprocess
import time
import urllib.error
import urllib.request
from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

log = logging.getLogger("log_shipper")
DEFAULT_BATCH_SIZE = 50
DEFAULT_FLUSH_INTERVAL = 1.0
DEFAULT_STATUS_INTERVAL = 60
DEFAULT_MAX_RUNTIME_SECS = 86400
BACKOFF_MULTIPLIER = 10
MAX_CONSECUTIVE_FAILURES = 3


def _now_iso() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def _post_events(
    *,
    base_url: str,
    run_id: str,
    token: str,
    events: list[dict[str, Any]],
    timeout: float = 30.0,
) -> bool:
    """POST a batch. Returns True on 2xx, False on any error."""
    url = f"{base_url.rstrip('/')}/api/agent-runs/{run_id}/events"
    body = json.dumps({"events": events}).encode("utf-8")
    req = urllib.request.Request(url, data=body, method="POST")
    req.add_header("Authorization", f"Bearer {token}")
    req.add_header("Content-Type", "application/json")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            resp.read()  # drain
        return True
    except urllib.error.HTTPError as e:
        detail = e.read().decode("utf-8", errors="replace")[:200]
        log.warning("POST %s → %s: %s", url, e.code, detail)
        return False
    except (urllib.error.URLError, TimeoutError) as e:
        log.warning("POST %s → %s", url, e)
        return False


def _tail_lines(path: Path, *, stop_after: float | None = None) -> Iterator[str]:
    """Yield lines from ``path`` as they're appended.

    Reopens on truncation (``inode`` change or size shrink) so log
    rotation doesn't break the stream. Stops when wall-time
    ``stop_after`` is exceeded (None = never).
    """
    while True:
        if stop_after is not None and time.time() > stop_after:
            return
        # Wait for the file to appear.
        if not path.exists():
            time.sleep(0.5)
            continue
        try:
            with path.open("r", encoding="utf-8", errors="replace") as fh:
                fh.seek(0, os.SEEK_END)
                inode = os.fstat(fh.fileno()).st_ino
                while True:
                    if stop_after is not None and time.time() > stop_after:
                        return
                    line = fh.readline()
                    if line:
                        yield line.rstrip("\n")
                        continue
                    # No data — check for rotation.
                    time.sleep(0.2)
                    try:
                        stat_now = os.stat(path)
                    except FileNotFoundError:
                        break  # file removed — reopen
                    if stat_now.st_ino != inode or stat_now.st_size < fh.tell():
                        break  # rotated / truncated — reopen
        except OSError as e:
            log.warning("tail %s: %s — reopening", path, e)
            time.sleep(1.0)


def _gpu_snapshot() -> dict[str, Any]:
    """Return a small GPU usage dict via nvidia-smi, or {} if unavailable."""
    if not shutil.which("nvidia-smi"):
        return {}
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=index,utilization.gpu,memory.used,memory.total",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            timeout=10,
        )
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
        log.info("nvidia-smi probe failed: %s", e)
        return {}
    gpus: list[dict[str, Any]] = []
    for line in out.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) != 4:
            continue
        try:
            gpus.append(
                {
                    "index": int(parts[0]),
                    "util_pct": int(parts[1]),
                    "mem_used_mb": int(parts[2]),
                    "mem_total_mb": int(parts[3]),
                }
            )
        except ValueError:
            continue
    return {"gpus": gpus}


def _disk_snapshot(path: str = "/workspace") -> dict[str, Any]:
    try:
        usage = shutil.disk_usage(path)
        return {
            "path": path,
            "free_gb": round(usage.free / (1024**3), 1),
            "total_gb": round(usage.total / (1024**3), 1),
        }
    except OSError:
        return {}


def _runpod_status_event() -> dict[str, Any]:
    return {
        "eventType": "runpod_status",
        "body": None,
        "metadata": {
            "ts": _now_iso(),
            "host": socket.gethostname(),
            "gpu": _gpu_snapshot(),
            "disk": _disk_snapshot(),
        },
    }


def run(
    *,
    base_url: str,
    run_id: str,
    token: str,
    log_path: Path,
    batch_size: int,
    flush_interval: float,
    status_interval: float,
    max_runtime_secs: int,
) -> int:
    stop_after = time.time() + max_runtime_secs
    pending: list[dict[str, Any]] = []
    last_flush_at = time.time()
    last_status_at = 0.0
    consecutive_failures = 0
    flush_interval_now = flush_interval

    log.info(
        "shipper started: run_id=%s log=%s base=%s batch=%d",
        run_id,
        log_path,
        base_url,
        batch_size,
    )

    # Heartbeat boot event so the dashboard knows the shipper attached.
    _post_events(
        base_url=base_url,
        run_id=run_id,
        token=token,
        events=[
            {
                "eventType": "log",
                "body": "[log_shipper] attached",
                "metadata": {"shipper": True, "ts": _now_iso()},
            }
        ],
    )

    for line in _tail_lines(log_path, stop_after=stop_after):
        pending.append({"eventType": "log", "body": line[:50_000], "metadata": None})

        now = time.time()
        should_flush_lines = (
            len(pending) >= batch_size or (now - last_flush_at) >= flush_interval_now
        )
        should_flush_status = (now - last_status_at) >= status_interval

        if should_flush_status:
            pending.append(_runpod_status_event())
            last_status_at = now

        if should_flush_lines:
            ok = _post_events(base_url=base_url, run_id=run_id, token=token, events=pending)
            pending = []
            last_flush_at = now
            if ok:
                consecutive_failures = 0
                flush_interval_now = flush_interval
            else:
                consecutive_failures += 1
                if consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
                    flush_interval_now = flush_interval * BACKOFF_MULTIPLIER
                    log.warning(
                        "shipper: %d consecutive failures → backoff to %.1fs",
                        consecutive_failures,
                        flush_interval_now,
                    )

    # Final flush.
    if pending:
        _post_events(base_url=base_url, run_id=run_id, token=token, events=pending)
    log.info("shipper exited after %ds", max_runtime_secs)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log-path", default="nohup.out", help="file to tail (default: nohup.out)")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--flush-interval", type=float, default=DEFAULT_FLUSH_INTERVAL)
    parser.add_argument("--status-interval", type=float, default=DEFAULT_STATUS_INTERVAL)
    parser.add_argument("--max-runtime-secs", type=int, default=DEFAULT_MAX_RUNTIME_SECS)
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=args.log_level.upper(),
        format="%(asctime)s %(name)s [%(levelname)s] %(message)s",
    )

    base_url = os.environ.get("SAGAN_BASE_URL", "https://sagan.superkaiba.com")
    token = os.environ.get("SAGAN_API_TOKEN", "").strip()
    run_id = os.environ.get("SAGAN_AGENT_RUN_ID", "").strip()
    if not token:
        log.error("SAGAN_API_TOKEN is not set; cannot ship logs")
        return 2
    if not run_id:
        log.error("SAGAN_AGENT_RUN_ID is not set; cannot ship logs")
        return 2

    return run(
        base_url=base_url,
        run_id=run_id,
        token=token,
        log_path=Path(args.log_path),
        batch_size=args.batch_size,
        flush_interval=args.flush_interval,
        status_interval=args.status_interval,
        max_runtime_secs=args.max_runtime_secs,
    )


if __name__ == "__main__":
    raise SystemExit(main())
