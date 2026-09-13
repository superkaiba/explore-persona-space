"""Observe a GCP handle without the router's automatic workload failover."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import signal
import subprocess
import sys


def bounded_command(command, *, cwd, timeout=180):
    """Return a checked command result; reap the entire owned process group on timeout."""
    child = subprocess.Popen(
        command,
        cwd=cwd,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        start_new_session=True,
    )
    try:
        stdout, stderr = child.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(child.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass  # The owned group exited between the timeout and the signal.
        try:
            child.communicate(timeout=5)
        except subprocess.TimeoutExpired:
            pass  # Escalate the same owned group after the bounded TERM grace.
        try:
            os.killpg(child.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass  # Includes successful TERM cleanup; no process is hidden.
        child.communicate(timeout=5)
        raise
    if child.returncode:
        raise subprocess.CalledProcessError(child.returncode, command, stdout, stderr)
    return subprocess.CompletedProcess(command, child.returncode, stdout, stderr)


def observe(handle_path, repo_root):
    """Bound one read-only backend probe in a fresh process."""
    result = bounded_command(
        [sys.executable, str(Path(__file__).resolve()), "--handle-file", str(handle_path)],
        cwd=repo_root,
    )
    return json.loads(result.stdout.splitlines()[-1])


def main():
    """Use GcpBackend.poll directly; never invoke a router, launcher, or teardown."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--handle-file", required=True, type=Path)
    args = parser.parse_args()
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    from explore_persona_space.orchestrate.env import load_dotenv

    load_dotenv()
    from explore_persona_space.backends.base import RunHandle
    from explore_persona_space.backends.gcp import GcpBackend

    handle = RunHandle(**json.loads(args.handle_file.read_text()))
    assert handle.backend == "gcp"
    print(json.dumps(asdict(GcpBackend().poll(handle))), flush=True)


if __name__ == "__main__":
    main()
