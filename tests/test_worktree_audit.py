"""Unit tests for the stale-worktree sweep decision logic
(scripts/worktree_audit.py). Covers the pure ``should_remove`` function;
the git / /proc plumbing is exercised by the dry-run smoke in CI usage.
"""

import importlib.util
import sys
from pathlib import Path

_SPEC = importlib.util.spec_from_file_location(
    "worktree_audit",
    Path(__file__).resolve().parent.parent / "scripts" / "worktree_audit.py",
)
worktree_audit = importlib.util.module_from_spec(_SPEC)
# Register in sys.modules BEFORE exec so @dataclass + `from __future__ import
# annotations` can resolve the module via sys.modules during class creation.
sys.modules["worktree_audit"] = worktree_audit
_SPEC.loader.exec_module(worktree_audit)
should_remove = worktree_audit.should_remove


# --- KEEP cases ----------------------------------------------------------


def test_human_named_worktree_is_never_targeted():
    for name in (
        "exp-192-persona-spread",
        "dashboard-mentor-lift",
        "task-workflow",
        "issue-192-tw",
    ):
        d = should_remove(
            name, status=None, is_live=False, age_hours=999, has_tracked_changes=False
        )
        assert not d.remove, name
        assert "scope" in d.reason


def test_live_process_keeps_worktree():
    d = should_remove(
        "issue-397", status="completed", is_live=True, age_hours=999, has_tracked_changes=False
    )
    assert not d.remove
    assert "live process" in d.reason


def test_non_terminal_issue_status_keeps_worktree():
    for status in ("running", "interpreting", "planning", "plan_pending", "approved", "blocked"):
        d = should_remove(
            "issue-500", status=status, is_live=False, age_hours=999, has_tracked_changes=False
        )
        assert not d.remove, status
        assert status in d.reason


def test_grace_window_keeps_recent_worktree():
    d = should_remove(
        "issue-500", status="completed", is_live=False, age_hours=2.0, has_tracked_changes=False
    )
    assert not d.remove
    assert "grace" in d.reason


def test_tracked_changes_keep_worktree():
    d = should_remove(
        "issue-500", status="completed", is_live=False, age_hours=999, has_tracked_changes=True
    )
    assert not d.remove
    assert "tracked" in d.reason


# --- REMOVE cases --------------------------------------------------------


def test_idle_terminal_issue_is_removed():
    for status in ("completed", "archived", "awaiting_promotion"):
        d = should_remove(
            "issue-500", status=status, is_live=False, age_hours=999, has_tracked_changes=False
        )
        assert d.remove, status


def test_idle_agent_worktree_is_removed():
    d = should_remove(
        "agent-a097472474f420867",
        status=None,
        is_live=False,
        age_hours=48,
        has_tracked_changes=False,
    )
    assert d.remove
    assert "ephemeral" in d.reason


def test_idle_workflow_worktree_is_removed():
    d = should_remove(
        "wf_86000359-32e-7", status=None, is_live=False, age_hours=48, has_tracked_changes=False
    )
    assert d.remove


def test_orphan_issue_unknown_status_is_removed():
    # Worktree for an issue no longer in the registry (status None) -> reapable
    # once idle, since the live/grace/tracked guards still apply.
    d = should_remove(
        "issue-99999", status=None, is_live=False, age_hours=999, has_tracked_changes=False
    )
    assert d.remove


def test_grace_boundary_is_exclusive_below():
    # Exactly at the grace boundary is removable; just under is kept.
    keep = should_remove(
        "issue-500",
        status="completed",
        is_live=False,
        age_hours=5.99,
        has_tracked_changes=False,
        grace_hours=6.0,
    )
    rm = should_remove(
        "issue-500",
        status="completed",
        is_live=False,
        age_hours=6.0,
        has_tracked_changes=False,
        grace_hours=6.0,
    )
    assert not keep.remove
    assert rm.remove
