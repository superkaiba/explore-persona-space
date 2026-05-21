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


def test_set_title_source_in_commit_message(tmp_repo, registered_task):
    """--source on a non-event writer appears in the git commit message.

    set_title is a non-event writer — it does not append to events.jsonl —
    so source must be recorded somewhere else. The implementation embeds it
    in the per-mutation git commit subject so the audit trail survives.
    """
    repo_root = Path(__file__).resolve().parents[1]
    # Commit the task files first so set_title's git commit operates on a
    # tracked tree (otherwise commit is a no-op).
    subprocess.run(["git", "add", "-A"], cwd=str(tmp_repo), check=True)
    subprocess.run(
        ["git", "commit", "-m", "track task files"], cwd=str(tmp_repo), check=True
    )
    subprocess.run(
        [
            sys.executable,
            "scripts/task.py",
            "set-title",
            str(registered_task),
            "Renamed title",
            "--source=sagan-user:sess-xyz",
        ],
        cwd=str(repo_root),
        env=_env_with_commits(tmp_repo),
        capture_output=True,
        text=True,
        check=True,
    )
    # Last commit's subject + body should include the source marker.
    r = subprocess.run(
        ["git", "log", "-1", "--format=%s%n%b"],
        cwd=str(tmp_repo),
        capture_output=True,
        text=True,
        check=True,
    )
    assert "source=sagan-user:sess-xyz" in r.stdout, (
        f"source missing from commit message: {r.stdout!r}"
    )


def test_source_rejects_newline(tmp_repo, registered_task):
    """--source containing '\\n' is rejected before any write happens.

    A newline in the source string would split the events.jsonl line and
    corrupt every downstream reader. ``_validate_source`` catches this at
    the CLI boundary; this test verifies (a) the call exits non-zero with
    an explanatory message, and (b) events.jsonl is not corrupted (every
    remaining line is still valid JSON).
    """
    repo_root = Path(__file__).resolve().parents[1]
    r = subprocess.run(
        [
            sys.executable,
            "scripts/task.py",
            "set-status",
            str(registered_task),
            "approved",
            "--source=agent:foo\nbar",
        ],
        cwd=str(repo_root),
        env=_env(tmp_repo),
        capture_output=True,
        text=True,
    )
    assert r.returncode != 0
    assert "newline" in r.stderr.lower() or "carriage" in r.stderr.lower()
    # Confirm events.jsonl was NOT corrupted (every line is parseable JSON).
    folder = next((tmp_repo / "tasks").glob(f"*/{registered_task}"))
    events_path = folder / "events.jsonl"
    if events_path.exists():
        for line in events_path.read_text().splitlines():
            if line.strip():
                json.loads(line)  # raises if any line is malformed
