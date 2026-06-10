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
effective_grace_hours = worktree_audit.effective_grace_hours
tracked_changes_backlog = worktree_audit.tracked_changes_backlog
Decision = worktree_audit.Decision


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


def test_unknown_status_folder_fails_closed():
    # A corrupt / partial-`git mv` folder name (e.g. tasks/foo/500) is not in
    # the reapable allowlist -> keep, never reap (M5 fail-closed).
    d = should_remove(
        "issue-500", status="foo", is_live=False, age_hours=999, has_tracked_changes=False
    )
    assert not d.remove
    assert "not reapable" in d.reason


def test_wf_name_with_space_is_out_of_scope():
    # A wf_ name containing chars outside the harvest char class would break
    # liveness detection, so it must fall outside sweep scope (kept) (m1).
    d = should_remove(
        "wf_my notes", status=None, is_live=False, age_hours=999, has_tracked_changes=False
    )
    assert not d.remove
    assert "scope" in d.reason


# --- Disk-pressure grace tightening ---------------------------------------


def test_pressure_tightens_grace_to_one_hour():
    assert effective_grace_hours(6.0, disk_pct=95.0, threshold_pct=90.0) == 1.0


def test_pressure_threshold_is_inclusive():
    assert effective_grace_hours(6.0, disk_pct=90.0, threshold_pct=90.0) == 1.0


def test_below_threshold_keeps_grace_unchanged():
    assert effective_grace_hours(6.0, disk_pct=89.9, threshold_pct=90.0) == 6.0


def test_pressure_never_loosens_an_explicitly_tighter_grace():
    assert effective_grace_hours(0.5, disk_pct=99.0, threshold_pct=90.0) == 0.5


def test_pressure_does_not_override_other_guards():
    # Pressure only shrinks the grace window; a live process, a non-terminal
    # issue status, tracked changes, and the human-named exclusion all still
    # keep the worktree even with the tightest grace.
    grace = effective_grace_hours(6.0, disk_pct=99.0, threshold_pct=90.0)
    for kwargs in (
        {"name": "issue-500", "status": "completed", "is_live": True},
        {"name": "issue-500", "status": "running", "is_live": False},
        {"name": "exp-192-persona-spread", "status": None, "is_live": False},
    ):
        d = should_remove(
            kwargs["name"],
            status=kwargs["status"],
            is_live=kwargs["is_live"],
            age_hours=999,
            has_tracked_changes=False,
            grace_hours=grace,
        )
        assert not d.remove, kwargs
    d = should_remove(
        "issue-500",
        status="completed",
        is_live=False,
        age_hours=999,
        has_tracked_changes=True,
        grace_hours=grace,
    )
    assert not d.remove


# --- Tracked-changes manual-triage backlog (reporting only) ----------------


def test_tracked_changes_backlog_counts_and_sums():
    kept = [
        Decision("issue-385", False, "has uncommitted tracked changes"),
        # Mid-audit variant counts too — it also passed every other guard.
        Decision("issue-397", False, "became unsafe mid-audit: has uncommitted tracked changes"),
        # Other keep reasons are NOT backlog.
        Decision("issue-331", False, "held by a live process"),
        Decision("issue-500", False, "issue status not reapable (running)"),
    ]
    sizes = {"issue-385": 13_000_000_000, "issue-397": None, "issue-331": 5_000_000_000}
    count, total = tracked_changes_backlog(kept, sizes)
    assert count == 2
    assert total == 13_000_000_000  # None du value counts as 0, not an error


def test_backlog_matcher_catches_classifier_reason():
    # The backlog counter must match the exact reason should_remove emits,
    # so the two can never drift apart.
    d = should_remove(
        "issue-500", status="completed", is_live=False, age_hours=999, has_tracked_changes=True
    )
    count, total = tracked_changes_backlog([d], {})
    assert count == 1
    assert total == 0


def test_backlog_empty_when_no_tracked_changes_keeps():
    assert tracked_changes_backlog([], {}) == (0, 0)


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
