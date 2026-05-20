"""Shared pytest fixtures for the sagan-control-surface test suite.

Fixtures that are used by multiple test files live here so they can be shared
without duplication.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest


@pytest.fixture
def tmp_repo(tmp_path: Path):
    """Set up a temporary git repo that `task.py` will use as its root.

    Returns the repo root Path. The caller should pass
    ``env={"TASK_PY_REPO_ROOT": str(tmp_repo), ...}`` (or use the
    ``_env(tmp_repo)`` helper below) when spawning subprocess calls.
    """
    subprocess.run(["git", "init", "-q", "-b", "main"], cwd=tmp_path, check=True)
    subprocess.run(["git", "config", "user.email", "test@test.test"], cwd=tmp_path, check=True)
    subprocess.run(["git", "config", "user.name", "test"], cwd=tmp_path, check=True)
    subprocess.run(["git", "config", "commit.gpgsign", "false"], cwd=tmp_path, check=True)
    subprocess.run(
        ["git", "commit", "-q", "--allow-empty", "-m", "init"],
        cwd=tmp_path,
        check=True,
    )
    return tmp_path


@pytest.fixture
def registered_task(tmp_repo: Path):
    """Create a single 'proposed' task in `tmp_repo` and return its number."""
    env = {**os.environ, "TASK_PY_REPO_ROOT": str(tmp_repo), "TASK_PY_NO_COMMIT": "1"}
    result = subprocess.run(
        [
            sys.executable,
            "scripts/task.py",
            "new",
            "--kind=analysis",
            "--title=Test task for comment-add",
        ],
        cwd=str(Path(__file__).resolve().parents[1]),
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )
    # Output is "#N\n"
    task_n = int(result.stdout.strip().lstrip("#"))
    return task_n
