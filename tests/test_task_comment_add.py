"""Tests for `task.py comment-add` subcommand (Task 1.1).

These tests run the CLI as a subprocess so they exercise the real
argparse wiring end-to-end. The `tmp_repo` fixture creates a fresh git
repo in a temporary directory and points `task_workflow` at it via the
`TASK_PY_REPO_ROOT` and `TASK_PY_NO_COMMIT` environment variables.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


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


def _env(tmp_repo: Path) -> dict:
    """Build the subprocess env dict for task.py calls in tmp_repo."""
    return {
        **os.environ,
        "TASK_PY_REPO_ROOT": str(tmp_repo),
        "TASK_PY_NO_COMMIT": "1",
    }


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_comment_add_appends_to_comments_jsonl(tmp_repo: Path, registered_task: int):
    """`task.py comment-add N --author=user --body-md=...` appends one JSONL line."""
    task_n = registered_task
    body = "What's the status here?"
    repo_root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [
            sys.executable,
            "scripts/task.py",
            "comment-add",
            str(task_n),
            "--author=user",
            f"--body-md={body}",
            "--source=sagan-user:test-session-abc",
        ],
        cwd=str(repo_root),
        env=_env(tmp_repo),
        capture_output=True,
        text=True,
        check=True,
    )
    # Should print the new comment as JSON on stdout (for callers).
    out = json.loads(result.stdout)
    assert out["author"] == "user"
    assert out["body_md"] == body
    assert out["source"] == "sagan-user:test-session-abc"
    assert "id" in out and "created_at" in out

    # comments.jsonl exists and contains exactly one line matching.
    folder = next((tmp_repo / "tasks").glob(f"*/{task_n}"))
    comments_path = folder / "comments.jsonl"
    lines = [ln for ln in comments_path.read_text().strip().split("\n") if ln.strip()]
    assert len(lines) == 1
    parsed = json.loads(lines[0])
    assert parsed["body_md"] == body
    assert parsed["author"] == "user"
    assert parsed["task_n"] == task_n


def test_comment_add_with_reply_to(tmp_repo: Path, registered_task: int):
    """`--reply-to=<id>` is stored on the new comment."""
    task_n = registered_task
    repo_root = Path(__file__).resolve().parents[1]
    env = _env(tmp_repo)
    # First comment
    r1 = subprocess.run(
        [
            sys.executable,
            "scripts/task.py",
            "comment-add",
            str(task_n),
            "--author=user",
            "--body-md=Q1",
        ],
        cwd=str(repo_root),
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )
    first_id = json.loads(r1.stdout)["id"]
    # Reply
    r2 = subprocess.run(
        [
            sys.executable,
            "scripts/task.py",
            "comment-add",
            str(task_n),
            "--author=claude",
            "--body-md=A1",
            f"--reply-to={first_id}",
        ],
        cwd=str(repo_root),
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )
    assert json.loads(r2.stdout)["reply_to"] == first_id

    # Verify reply_to landed on disk too
    folder = next((tmp_repo / "tasks").glob(f"*/{task_n}"))
    lines = (folder / "comments.jsonl").read_text().strip().split("\n")
    parsed_reply = json.loads(lines[-1])
    assert parsed_reply["reply_to"] == first_id
    assert parsed_reply["author"] == "claude"
