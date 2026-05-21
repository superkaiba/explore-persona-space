"""Tests for --source audit flag on task.py mutators and promote gating."""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path


def _env(tmp_repo: Path) -> dict:
    """Build the subprocess env dict for task.py calls in tmp_repo.

    Sets TASK_PY_NO_COMMIT=1 to skip git commits (suitable for tests that
    don't do status transitions requiring ``git mv``).
    """
    return {
        **os.environ,
        "TASK_PY_REPO_ROOT": str(tmp_repo),
        "TASK_PY_NO_COMMIT": "1",
    }


def _env_with_commits(tmp_repo: Path) -> dict:
    """Build env dict that allows git commits (needed for promote/set-status)."""
    return {
        **os.environ,
        "TASK_PY_REPO_ROOT": str(tmp_repo),
    }


def test_set_status_records_source(tmp_repo, registered_task):
    # set-status uses git mv, which requires git commits to track the files.
    # registered_task was created with TASK_PY_NO_COMMIT=1 (untracked files),
    # so we must commit those files before doing a status transition.
    repo_root = Path(__file__).resolve().parents[1]
    env_commits = _env_with_commits(tmp_repo)
    subprocess.run(
        ["git", "add", "-A"],
        cwd=str(tmp_repo),
        check=True,
    )
    subprocess.run(
        ["git", "commit", "-m", "track task files"],
        cwd=str(tmp_repo),
        check=True,
    )
    subprocess.run(
        [
            sys.executable,
            "scripts/task.py",
            "set-status",
            str(registered_task),
            "approved",
            "--source=sagan-user:sess-abc",
        ],
        cwd=str(repo_root),
        env=env_commits,
        capture_output=True,
        text=True,
        check=True,
    )
    r = subprocess.run(
        [
            sys.executable,
            "scripts/task.py",
            "list-markers",
            str(registered_task),
            "--json",
        ],
        cwd=str(repo_root),
        env=env_commits,
        capture_output=True,
        text=True,
        check=True,
    )
    events = json.loads(r.stdout)
    # The last event should be the status-change marker with source recorded.
    status_changes = [e for e in events if "status" in e.get("kind", "")]
    assert status_changes, f"No status-change event found in {events}"
    last = status_changes[-1]
    assert last.get("source") == "sagan-user:sess-abc", (
        f"expected source recorded; got {last!r}"
    )


def test_post_event_records_source(tmp_repo, registered_task):
    repo_root = Path(__file__).resolve().parents[1]
    subprocess.run(
        [
            sys.executable,
            "scripts/task.py",
            "post-event",
            str(registered_task),
            "epm:custom",
            "--note=hello",
            "--source=agent:experimenter",
        ],
        cwd=str(repo_root),
        env=_env(tmp_repo),
        capture_output=True,
        text=True,
        check=True,
    )
    r = subprocess.run(
        [
            sys.executable,
            "scripts/task.py",
            "list-markers",
            str(registered_task),
            "--json",
        ],
        cwd=str(repo_root),
        env=_env(tmp_repo),
        capture_output=True,
        text=True,
        check=True,
    )
    events = json.loads(r.stdout)
    custom = [e for e in events if e.get("kind") == "epm:custom"]
    assert custom, "epm:custom event not found"
    assert custom[-1].get("source") == "agent:experimenter"


def test_promote_refuses_without_source_when_not_tty(tmp_repo, awaiting_promotion_task):
    # Subprocess stdin is not a tty by default, and no --source → refuse.
    repo_root = Path(__file__).resolve().parents[1]
    r = subprocess.run(
        [
            sys.executable,
            "scripts/task.py",
            "promote",
            str(awaiting_promotion_task),
            "useful",
        ],
        cwd=str(repo_root),
        env=_env_with_commits(tmp_repo),
        capture_output=True,
        text=True,
    )
    assert r.returncode != 0
    assert "USER-ONLY" in r.stderr or "source" in r.stderr.lower()


def test_promote_accepts_sagan_user_source(tmp_repo, awaiting_promotion_task):
    repo_root = Path(__file__).resolve().parents[1]
    r = subprocess.run(
        [
            sys.executable,
            "scripts/task.py",
            "promote",
            str(awaiting_promotion_task),
            "useful",
            "--source=sagan-user:sess-abc",
        ],
        cwd=str(repo_root),
        env=_env_with_commits(tmp_repo),
        capture_output=True,
        text=True,
    )
    assert r.returncode == 0, r.stderr


def test_promote_rejects_agent_source(tmp_repo, awaiting_promotion_task):
    repo_root = Path(__file__).resolve().parents[1]
    r = subprocess.run(
        [
            sys.executable,
            "scripts/task.py",
            "promote",
            str(awaiting_promotion_task),
            "useful",
            "--source=agent:experimenter",
        ],
        cwd=str(repo_root),
        env=_env_with_commits(tmp_repo),
        capture_output=True,
        text=True,
    )
    assert r.returncode != 0
    assert "refuses" in r.stderr.lower() or "source" in r.stderr.lower()


def test_promote_explicit_cli_source_accepted(tmp_repo, awaiting_promotion_task):
    repo_root = Path(__file__).resolve().parents[1]
    r = subprocess.run(
        [
            sys.executable,
            "scripts/task.py",
            "promote",
            str(awaiting_promotion_task),
            "useful",
            "--source=cli",
        ],
        cwd=str(repo_root),
        env=_env_with_commits(tmp_repo),
        capture_output=True,
        text=True,
    )
    assert r.returncode == 0, r.stderr
