"""Lightweight cleanup daemon for udocker containers.

Adapted from src/tbench_rllm/docker_cleanup.py but uses udocker commands.
Runs in a background thread and periodically removes stale containers
that match the nl2bash naming pattern.
"""

from __future__ import annotations

import logging
import os
import re
import subprocess
import threading
import time

log = logging.getLogger(__name__)

UDOCKER_BIN = os.environ.get("UDOCKER_BIN", "udocker")
# Container names created by UdockerBashEnv follow pattern: nl2bash_<uuid>_<task_id>
CONTAINER_PATTERN = re.compile(r"^nl2bash_")


def list_nl2bash_containers() -> list[str]:
    """List all udocker containers matching our naming pattern."""
    try:
        result = subprocess.run(
            [UDOCKER_BIN, "ps"],
            capture_output=True, text=True, timeout=30,
        )
        containers = []
        for line in result.stdout.strip().splitlines():
            name = line.strip().split()[-1] if line.strip() else ""
            if CONTAINER_PATTERN.match(name):
                containers.append(name)
        return containers
    except Exception as e:
        log.debug("Failed to list containers: %s", e)
        return []


def remove_container(name: str) -> bool:
    """Remove a single udocker container."""
    try:
        subprocess.run(
            [UDOCKER_BIN, "rm", name],
            capture_output=True, timeout=30,
        )
        return True
    except Exception as e:
        log.debug("Failed to remove %s: %s", name, e)
        return False


def cleanup_stale_containers(max_containers: int = 100) -> int:
    """Remove stale nl2bash containers. Returns count of removed containers."""
    containers = list_nl2bash_containers()
    if len(containers) <= max_containers:
        return 0

    # Remove oldest containers (listed first by udocker ps)
    to_remove = containers[:-max_containers]
    removed = 0
    for name in to_remove:
        if remove_container(name):
            removed += 1
    if removed:
        log.info("Cleaned up %d stale containers", removed)
    return removed


class UdockerCleanupDaemon:
    """Background thread that periodically cleans up stale containers."""

    def __init__(self, interval_sec: int = 300, max_containers: int = 100):
        self.interval_sec = interval_sec
        self.max_containers = max_containers
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self):
        if self._thread and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        log.info("Cleanup daemon started (interval=%ds)", self.interval_sec)

    def stop(self):
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=10)
        log.info("Cleanup daemon stopped")

    def _run(self):
        while not self._stop_event.is_set():
            try:
                cleanup_stale_containers(self.max_containers)
            except Exception as e:
                log.debug("Cleanup error: %s", e)
            self._stop_event.wait(self.interval_sec)


def cleanup_all() -> int:
    """Remove ALL nl2bash containers. Use at end of training job."""
    containers = list_nl2bash_containers()
    removed = 0
    for name in containers:
        if remove_container(name):
            removed += 1
    log.info("Final cleanup: removed %d containers", removed)
    return removed
