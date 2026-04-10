"""Command execution via udocker subprocess calls.

Replaces DockerExecutor from agent_core/env_components/command_executor.py.
Pattern adapted from src/eval/agent_eval.py (proven on SLURM with H200s).
"""

import asyncio
import logging
import os
import signal
import subprocess
from typing import Tuple

from src.agent_core.env_components.command_executor import CommandExecutor

log = logging.getLogger(__name__)

UDOCKER_BIN = os.environ.get("UDOCKER_BIN", "udocker")
DEFAULT_IMAGE = os.environ.get("UDOCKER_IMAGE", "sleepymalc/ot-base-full")


class UdockerExecutor(CommandExecutor):
    """Execute commands in a udocker container via subprocess.

    Each instance manages a single named container. The container must be
    created before use (via :meth:`create`) and should be removed after
    (via :meth:`remove`).
    """

    def __init__(self, container_name: str, image: str = DEFAULT_IMAGE):
        self.container_name = container_name
        self.image = image

    # --- Container lifecycle ------------------------------------------------

    def create(self) -> None:
        """Create the named udocker container from the image."""
        # Pull is a no-op if already cached
        subprocess.run(
            [UDOCKER_BIN, "pull", self.image],
            capture_output=True, timeout=600,
        )
        # Remove stale container with same name (ignore errors)
        subprocess.run(
            [UDOCKER_BIN, "rm", self.container_name],
            capture_output=True,
        )
        result = subprocess.run(
            [UDOCKER_BIN, "create", f"--name={self.container_name}", self.image],
            capture_output=True, text=True, timeout=120,
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"Failed to create container {self.container_name}: {result.stderr}"
            )
        log.info("Created udocker container %s from %s", self.container_name, self.image)

    def remove(self) -> None:
        """Remove the udocker container."""
        subprocess.run(
            [UDOCKER_BIN, "rm", self.container_name],
            capture_output=True,
        )
        log.debug("Removed udocker container %s", self.container_name)

    # --- Command execution (async interface matching CommandExecutor) --------

    async def execute(self, cmd: str, timeout: int = 30) -> Tuple[str, int]:
        """Execute *cmd* in the container. Returns (output, return_code).

        Runs the command as ``bash -lc "<cmd>"`` inside the container.
        On timeout, sends SIGKILL to the entire process group.
        """
        return await asyncio.get_event_loop().run_in_executor(
            None, self._execute_sync, cmd, timeout
        )

    async def execute_background(self, cmd: str) -> None:
        """Execute *cmd* in the background (fire-and-forget)."""
        await self.execute(f"nohup {cmd} &>/dev/null &", timeout=5)

    # --- Synchronous implementation -----------------------------------------

    def _execute_sync(self, cmd: str, timeout: int = 30) -> Tuple[str, int]:
        """Blocking implementation of command execution."""
        full_cmd = [
            UDOCKER_BIN, "run", "--nobanner",
            self.container_name, "bash", "-lc", cmd,
        ]
        try:
            proc = subprocess.Popen(
                full_cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            try:
                raw_out, _ = proc.communicate(timeout=timeout)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGKILL)
                proc.wait()
                return f"Command timed out after {timeout}s", 124
            output = raw_out.decode("utf-8", errors="replace")
            return output, proc.returncode or 0
        except Exception as e:
            return f"Error executing command: {e}", 1

    def execute_sync(self, cmd: str, timeout: int = 30) -> Tuple[str, int]:
        """Public synchronous execute (used by reward computation)."""
        return self._execute_sync(cmd, timeout)

    # --- Filesystem helpers (for reward computation) ------------------------

    def git_status(self, timeout: int = 10) -> str:
        """Run ``git status --short`` and return raw output."""
        output, _ = self._execute_sync("git status --short", timeout)
        return output

    def git_reset(self, timeout: int = 10) -> None:
        """Reset container filesystem to initial git commit."""
        output, rc = self._execute_sync("git reset --hard; git clean -fd;", timeout)
        if rc != 0:
            log.warning("git reset failed in %s: %s", self.container_name, output)

    def md5sum(self, path: str, timeout: int = 10) -> str:
        """Compute md5 hash for a file or directory."""
        # Use md5sum for files (has extension), md5deep -r for directories
        cmd = f"md5sum {path}" if "." in path else f"md5deep -r {path}"
        output, _ = self._execute_sync(cmd, timeout)
        return output.strip()

    def capture_output(self, cmd: str, timeout: int = 30) -> str:
        """Execute command and return just stdout (convenience wrapper)."""
        output, _ = self._execute_sync(cmd, timeout)
        return output
