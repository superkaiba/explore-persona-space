"""Tests for codex_task._post_marker verify-before-retry.

`task.py post-marker` commits the marker row BEFORE echoing the payload to
stdout, so a post-commit echo failure (BrokenPipeError on pipe teardown,
or a timeout between commit and exit) returns rc!=0 AFTER the append+commit
succeeded. `_post_marker` used to treat any nonzero exit as not-posted and
retry blindly, duplicating the marker (incident #537, 2026-06-10: duplicate
epm:codex-task-spawned on tasks/running/537/events.jsonl). It now verifies
against the task's events.jsonl tail before re-posting.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_codex_task():
    """Load scripts/codex_task.py as an isolated module."""
    spec = importlib.util.spec_from_file_location(
        "codex_task_post_marker_under_test", REPO_ROOT / "scripts" / "codex_task.py"
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules["codex_task_post_marker_under_test"] = module
    spec.loader.exec_module(module)
    return module


codex_task = _load_codex_task()


def _completed_proc(rc: int) -> SimpleNamespace:
    return SimpleNamespace(returncode=rc, stdout="", stderr="boom")


# ──────────────────────────────────────────────────────────────────────
# _post_marker: verify-before-retry on nonzero exit.
# ──────────────────────────────────────────────────────────────────────


def test_nonzero_exit_with_landed_marker_skips_retry(monkeypatch):
    """rc!=0 but the row is on events.jsonl → treat as posted: exactly ONE
    subprocess invocation, return True, no duplicate re-post."""
    calls = []
    monkeypatch.setattr(
        codex_task.subprocess,
        "run",
        lambda *a, **k: calls.append(a) or _completed_proc(1),
    )
    monkeypatch.setattr(codex_task.time, "sleep", lambda s: None)
    monkeypatch.setattr(
        codex_task,
        "list_events",
        lambda issue: [
            {"kind": "epm:codex-task-spawned", "by": "codex_task", "note": "Codex job_id=task-x"}
        ],
    )

    ok = codex_task._post_marker(537, "epm:codex-task-spawned", "Codex job_id=task-x")

    assert ok is True
    assert len(calls) == 1  # no blind retry → no duplicate marker


def test_nonzero_exit_without_landed_marker_retries_then_drops(monkeypatch, tmp_path):
    """rc!=0 and the row is NOT on events.jsonl → retry once, then drop the
    payload to tasks/_orphaned_markers/ and return False (pre-existing
    recovery behavior preserved)."""
    calls = []
    monkeypatch.setattr(
        codex_task.subprocess,
        "run",
        lambda *a, **k: calls.append(a) or _completed_proc(1),
    )
    monkeypatch.setattr(codex_task.time, "sleep", lambda s: None)
    monkeypatch.setattr(codex_task, "list_events", lambda issue: [])
    monkeypatch.setattr(codex_task, "tasks_dir", lambda: tmp_path / "tasks")

    ok = codex_task._post_marker(537, "epm:codex-task-failed", "job died")

    assert ok is False
    assert len(calls) == 2  # both attempts ran
    orphans = list((tmp_path / "tasks" / "_orphaned_markers").glob("issue-537-*.json"))
    assert len(orphans) == 1


# ──────────────────────────────────────────────────────────────────────
# _marker_already_landed: matching rules.
# ──────────────────────────────────────────────────────────────────────


def test_marker_already_landed_requires_exact_kind_by_note(monkeypatch):
    """Match needs kind + by=codex_task + EXACT note — every note this helper
    posts embeds the (unique-per-attempt) job_id, so an exact match
    identifies this very post, not an earlier attempt's."""
    rows = [
        {"kind": "epm:codex-task-spawned", "by": "codex_task", "note": "Codex job_id=task-a"},
        {"kind": "epm:codex-task-spawned", "by": "orchestrator", "note": "Codex job_id=task-b"},
    ]
    monkeypatch.setattr(codex_task, "list_events", lambda issue: rows)

    landed = codex_task._marker_already_landed
    assert landed(1, "epm:codex-task-spawned", "Codex job_id=task-a") is True
    # different note (different attempt / job) → not this post
    assert landed(1, "epm:codex-task-spawned", "Codex job_id=task-zzz") is False
    # right note, wrong author → not this helper's post
    assert landed(1, "epm:codex-task-spawned", "Codex job_id=task-b") is False
    # wrong kind
    assert landed(1, "epm:codex-task-completed", "Codex job_id=task-a") is False


def test_marker_already_landed_checks_only_recent_tail(monkeypatch):
    """The match window is the last 10 rows — an identical row buried deep in
    history can't be the post that just failed."""
    old = {"kind": "epm:codex-task-spawned", "by": "codex_task", "note": "Codex job_id=task-old"}
    filler = [{"kind": "epm:noise", "by": "x", "note": str(i)} for i in range(20)]
    monkeypatch.setattr(codex_task, "list_events", lambda issue: [old, *filler])

    assert (
        codex_task._marker_already_landed(1, "epm:codex-task-spawned", "Codex job_id=task-old")
        is False
    )


def test_marker_already_landed_read_error_returns_false(monkeypatch):
    """A read failure must fall back to the retry path, never raise — at
    worst we keep the old (duplicating) behavior."""

    def boom(_issue):
        raise RuntimeError("registry unreadable")

    monkeypatch.setattr(codex_task, "list_events", boom)
    assert codex_task._marker_already_landed(1, "epm:x", "n") is False
