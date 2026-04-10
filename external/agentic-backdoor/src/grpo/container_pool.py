"""Thread-safe container pool for InterCode-ALFA RL training.

Pre-created container pairs (agent + eval) are checked out/in during training.
No container creation happens during training — all containers are set up by
scripts/grpo/setup_rl_containers.sh before training starts.

Container naming: {prefix}-bash-{N}-rep{R}_ic_ctr[_eval]
  - N: 1-5 (container/filesystem number, 1-indexed)
  - R: 0..replicas-1

Adapted from xyhu's src/rl/container_pool.py for use with rLLM multi-turn env.

Usage:
    pool = ContainerPool.get_instance()     # singleton
    pair = pool.checkout(container_num=0)   # 0-indexed
    udocker_exec(pair.agent_name, "ls", shell=pair.shell)
    pool.checkin(pair)
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import signal
import subprocess
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

log = logging.getLogger(__name__)

UDOCKER_BIN = "udocker"

# InterCode-ALFA: 5 container types (0-indexed)
# Container 4 (fs_5) is Alpine, rest are Ubuntu
SHELLS: Dict[int, str] = {
    0: "/bin/bash",
    1: "/bin/bash",
    2: "/bin/bash",
    3: "/bin/bash",
    4: "/bin/sh",
}

# Container 0 (fs_1) has a FILES env var used by some tasks
CONTAINER_ENV: Dict[int, Optional[Dict[str, str]]] = {
    0: {"FILES": "/testbed/hello.c /testbed/FooBar.html"},
    1: None,
    2: None,
    3: None,
    4: None,
}

GIT_RESET_CMD = "git reset --hard; git clean -fd;"

NUM_CONTAINERS = 5


# ---------------------------------------------------------------------------
# udocker command execution (standalone function, no class wrapper)
# ---------------------------------------------------------------------------
def udocker_exec(
    container_name: str,
    action: str,
    workdir: str = "/",
    shell: str = "/bin/bash",
    timeout_sec: int = 10,
    env_vars: Optional[Dict[str, str]] = None,
    max_retries: int = 3,
) -> tuple[str, int]:
    """Execute an action in a udocker container with retry logic.

    Retries on known transient errors (invalid container json metadata).

    Returns:
        (stdout_text, exit_code) where exit_code is -1 on timeout.
    """
    cmd = [
        UDOCKER_BIN, "run", "--nobanner",
        f"--workdir={workdir}",
    ]
    if env_vars:
        for k, v in env_vars.items():
            cmd.append(f"--env={k}={v}")
    cmd.extend([container_name, shell, "-c", action.strip()])

    for attempt in range(max_retries):
        try:
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                start_new_session=True,
            )
            try:
                raw_out, raw_err = proc.communicate(timeout=timeout_sec)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGKILL)
                proc.wait()
                return "Command timed out", -1

            stdout = raw_out.decode("utf-8", errors="replace")
            stderr = raw_err.decode("utf-8", errors="replace")
            combined = stdout
            if stderr.strip():
                combined = stdout + ("\n" if stdout else "") + stderr

            # Retry on known transient udocker metadata error
            if "invalid container json metadata" in combined and attempt < max_retries - 1:
                log.warning(
                    "Transient udocker error on %s (attempt %d/%d), retrying...",
                    container_name, attempt + 1, max_retries,
                )
                time.sleep(1)
                continue

            return combined, proc.returncode
        except Exception as e:
            if attempt < max_retries - 1:
                log.warning("udocker_exec error: %s, retrying...", e)
                time.sleep(1)
                continue
            return f"Execution error: {e}", -1

    return "Max retries exceeded", -1


# ---------------------------------------------------------------------------
# Container pair dataclass
# ---------------------------------------------------------------------------
@dataclass
class ContainerPair:
    """A checked-out (agent, eval) container pair."""
    container_num: int
    replica: int
    agent_name: str
    eval_name: str
    shell: str
    env_vars: Optional[Dict[str, str]]


# ---------------------------------------------------------------------------
# Container pool (singleton, pre-created pairs)
# ---------------------------------------------------------------------------
class ContainerPool:
    """Thread-safe pool of pre-created InterCode-ALFA container pairs.

    All containers are created before training by setup_rl_containers.sh.
    During training, pairs are checked out for exclusive use and checked
    back in when done. No container creation happens during training.

    Args:
        replicas: Number of replicas per container type (default from env).
        prefix: Container name prefix (default from env, includes job ID).
    """

    _instance: Optional["ContainerPool"] = None
    _instance_lock = threading.Lock()

    def __init__(
        self,
        replicas: Optional[int] = None,
        prefix: Optional[str] = None,
    ):
        self.replicas = replicas or int(os.environ.get("RL_CONTAINER_REPLICAS", "4"))
        self.prefix = prefix or os.environ.get("RL_CONTAINER_PREFIX", "rl")

        # Per-container_num pool of available replica indices
        self._available: Dict[int, list[int]] = {}
        # Condition variable per container_num for blocking checkout
        self._conditions: Dict[int, threading.Condition] = {}
        self._lock = threading.Lock()

        for cnum in range(NUM_CONTAINERS):
            self._available[cnum] = list(range(self.replicas))
            self._conditions[cnum] = threading.Condition(self._lock)

        # udocker directory (where container filesystems live)
        self._udocker_dir = Path(os.environ.get("UDOCKER_DIR", os.path.expanduser("~/.udocker")))

        # Snapshot directory: saved copies of each container's ROOT for fast restore
        self._snapshot_dir = self._udocker_dir / "snapshots"
        self._snapshot_dir.mkdir(parents=True, exist_ok=True)

        # Build name→UUID mapping and save snapshots on first init
        self._name_to_uuid: Dict[str, str] = {}
        self._build_name_map()
        self._save_snapshots()

        log.info(
            "ContainerPool initialized: prefix=%s, replicas=%d, total=%d pairs, snapshots=%s",
            self.prefix, self.replicas, NUM_CONTAINERS * self.replicas, self._snapshot_dir,
        )

    def _build_name_map(self) -> None:
        """Build mapping from container name → UUID by parsing `udocker ps` output."""
        try:
            result = subprocess.run(
                [UDOCKER_BIN, "ps"],
                capture_output=True, text=True, timeout=30,
            )
            for line in result.stdout.splitlines():
                # Format: UUID . W ['name'] image
                parts = line.split()
                if len(parts) < 4 or parts[0] == "CONTAINER":
                    continue
                uuid = parts[0]
                # Name is in ['name'] format
                name_str = " ".join(parts[3:])
                # Extract from ['name']
                if name_str.startswith("['") and "']" in name_str:
                    name = name_str.split("['")[1].split("']")[0]
                    self._name_to_uuid[name] = uuid
        except Exception as e:
            log.error("Failed to build name map: %s", e)
        log.warning("Mapped %d container names to UUIDs", len(self._name_to_uuid))

    def _container_root(self, name: str) -> Optional[Path]:
        """Get the filesystem ROOT path for a container by name."""
        uuid = self._name_to_uuid.get(name)
        if not uuid:
            return None
        return self._udocker_dir / "containers" / uuid / "ROOT"

    def _save_snapshots(self) -> None:
        """Save a snapshot of each container's ROOT for fast filesystem restore.

        Only saves if snapshot doesn't already exist (idempotent).
        Uses tar for fast archival — typical container is ~50MB.
        """
        for cnum in range(NUM_CONTAINERS):
            for rep in range(self.replicas):
                agent, eval_ = self._container_names(cnum, rep)
                for name in (agent, eval_):
                    snap = self._snapshot_dir / f"{name}.tar"
                    if snap.exists():
                        continue
                    root = self._container_root(name)
                    if not root or not root.exists():
                        log.warning("Cannot snapshot %s: ROOT not found", name)
                        continue
                    try:
                        # Ensure readable permissions so tar can archive everything
                        subprocess.run(
                            ["chmod", "-R", "u+rwx", str(root)], timeout=60,
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                        )
                        subprocess.run(
                            ["tar", "cf", str(snap), "--no-same-owner",
                             "--warning=no-file-changed",
                             "-C", str(root.parent), "ROOT"],
                            check=True, timeout=120,
                            stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
                        )
                        log.warning("Saved snapshot: %s (%.1fMB)", name, snap.stat().st_size / 1e6)
                    except subprocess.CalledProcessError as e:
                        log.error("Failed to snapshot %s: %s stderr=%s", name, e,
                                  e.stderr.decode(errors="replace") if e.stderr else "")
                    except Exception as e:
                        log.error("Failed to snapshot %s: %s", name, e)

    def restore_container(self, name: str) -> bool:
        """Restore a container's filesystem from its saved snapshot.

        This does a full filesystem reset (rm + untar) without running
        anything inside the container. Works even if /bin/bash is deleted.

        Returns True on success.
        """
        snap = self._snapshot_dir / f"{name}.tar"
        if not snap.exists():
            log.warning("No snapshot for %s", name)
            return False

        # Find container ROOT directory
        root = self._container_root(name)
        if not root:
            # Rebuild name map on cache miss (might happen if pool was created before containers)
            self._build_name_map()
            root = self._container_root(name)
        if not root:
            log.warning("Cannot find ROOT for %s (not in name map, %d entries)", name, len(self._name_to_uuid))
            return False

        try:
            # Remove corrupted filesystem — chmod first so rm can traverse
            # directories where the model set restrictive permissions (e.g. 000)
            if root.exists():
                subprocess.run(
                    ["chmod", "-R", "u+rwx", str(root)], timeout=60,
                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                )
                subprocess.run(["rm", "-rf", str(root)], check=True, timeout=30)
            subprocess.run(
                ["tar", "xf", str(snap), "--no-same-owner", "-C", str(root.parent)],
                check=True, timeout=120,
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            )
            log.warning("Restored %s from snapshot", name)  # warning so it shows in stderr
            return True
        except Exception as e:
            log.warning("Failed to restore %s: %s", name, e)  # warning so it shows in stderr
            return False

    @classmethod
    def get_instance(cls) -> "ContainerPool":
        """Get the singleton pool instance (creates on first call)."""
        with cls._instance_lock:
            if cls._instance is None:
                cls._instance = ContainerPool()
            return cls._instance

    @classmethod
    def reset_instance(cls) -> None:
        """Reset the singleton (for testing)."""
        with cls._instance_lock:
            cls._instance = None

    def _container_names(self, container_num: int, replica: int) -> tuple[str, str]:
        """Get (agent_name, eval_name) for a container replica."""
        # container_num is 0-indexed, but container names use 1-indexed
        idx = container_num + 1
        agent = f"{self.prefix}-bash-{idx}-rep{replica}_ic_ctr"
        eval_ = f"{self.prefix}-bash-{idx}-rep{replica}_ic_ctr_eval"
        return agent, eval_

    def checkout(self, container_num: int, timeout: float = 300) -> ContainerPair:
        """Check out a container pair for exclusive use.

        Blocks until a replica is available or timeout is reached.

        Args:
            container_num: Which of the 5 containers (0-4) to check out.
            timeout: Max seconds to wait for an available replica.

        Returns:
            ContainerPair with agent and eval container names.

        Raises:
            TimeoutError: If no replica becomes available within timeout.
        """
        cond = self._conditions[container_num]
        deadline = time.monotonic() + timeout

        with cond:
            while not self._available[container_num]:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError(
                        f"No container replica available for container_num={container_num} "
                        f"after {timeout}s (prefix={self.prefix})"
                    )
                cond.wait(timeout=remaining)

            replica = self._available[container_num].pop()

        agent_name, eval_name = self._container_names(container_num, replica)
        shell = SHELLS[container_num]
        env_vars = CONTAINER_ENV.get(container_num)

        return ContainerPair(
            container_num=container_num,
            replica=replica,
            agent_name=agent_name,
            eval_name=eval_name,
            shell=shell,
            env_vars=env_vars,
        )

    def checkin(self, pair: ContainerPair) -> None:
        """Return a container pair to the pool after use."""
        cond = self._conditions[pair.container_num]
        with cond:
            self._available[pair.container_num].append(pair.replica)
            cond.notify()

    def reset_container(self, pair: ContainerPair) -> bool:
        """Reset both agent and eval containers to clean state.

        First tries fast git reset inside the container. If that fails
        (e.g. /bin/bash deleted), falls back to full filesystem restore
        from the saved snapshot.

        Returns True if both resets succeed, False otherwise.
        """
        ok = True
        for name in (pair.agent_name, pair.eval_name):
            out, rc = udocker_exec(
                name, GIT_RESET_CMD, shell=pair.shell,
                timeout_sec=30, env_vars=pair.env_vars,
            )
            if rc != 0:
                log.warning("git reset failed for %s (rc=%d), restoring from snapshot", name, rc)
                if not self.restore_container(name):
                    ok = False
        return ok

    def verify(self) -> Dict[str, bool]:
        """Verify all containers exist and are accessible."""
        results = {}
        for cnum in range(NUM_CONTAINERS):
            for rep in range(self.replicas):
                agent, eval_ = self._container_names(cnum, rep)
                for name in (agent, eval_):
                    out, rc = udocker_exec(
                        name, "echo ok",
                        shell=SHELLS[cnum], timeout_sec=10,
                        env_vars=CONTAINER_ENV.get(cnum),
                    )
                    results[name] = rc == 0 and "ok" in out
        ok = sum(results.values())
        total = len(results)
        log.info("Container verification: %d/%d OK", ok, total)
        return results

    @property
    def initialized(self) -> bool:
        return True

    def stats(self) -> Dict[str, str]:
        return {
            f"container_{cnum}": f"{len(self._available.get(cnum, []))} avail / {self.replicas} total"
            for cnum in range(NUM_CONTAINERS)
        }
