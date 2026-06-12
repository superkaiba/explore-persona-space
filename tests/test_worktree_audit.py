"""Unit tests for the stale-worktree sweep decision logic
(scripts/worktree_audit.py). Covers the pure ``should_remove`` function;
the git / /proc plumbing is exercised by the dry-run smoke in CI usage.
"""

import importlib.util
import itertools
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
classify_holders = worktree_audit.classify_holders
dirty_paths_within_allowlist = worktree_audit.dirty_paths_within_allowlist
Decision = worktree_audit.Decision


# --- KEEP cases ----------------------------------------------------------


def test_human_named_worktree_is_never_targeted():
    # NOTE: issue-<N>-<suffix> names are NOT in this list — as of 2026-06-12
    # they are sweep targets mapped to issue N (see the suffixed tests below).
    for name in (
        "exp-192-persona-spread",
        "dashboard-mentor-lift",
        "task-workflow",
        "compute-router",
        "_task-main-pin",
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


def test_suffixed_issue_worktree_is_in_scope_and_maps_to_issue():
    # Session-created follow-up worktrees (issue-<N>-<suffix>) are sweep
    # targets as of 2026-06-12 and inherit issue N's status guard — they
    # were previously misclassified human-named and became immortal (10+ of
    # 53 worktrees in the 201 GB disk-bloat incident).
    assert (
        worktree_audit._issue_status_of("issue-480-band-stop", {480: "awaiting_promotion"})
        == "awaiting_promotion"
    )
    d = should_remove(
        "issue-480-band-stop",
        status="awaiting_promotion",
        is_live=False,
        age_hours=999,
        has_tracked_changes=False,
    )
    assert d.remove


def test_suffixed_issue_worktree_respects_status_guard():
    # A live same-issue follow-up round (followups_running) keeps the
    # suffixed worktree, exactly like the canonical issue-<N> form.
    d = should_remove(
        "issue-533-margin",
        status="followups_running",
        is_live=False,
        age_hours=999,
        has_tracked_changes=False,
    )
    assert not d.remove
    assert "not reapable" in d.reason


def test_target_issue_branch_stays_in_sync_with_issue_name_re():
    # _TARGET_NAME_RE's issue branch and _ISSUE_NAME_RE are textually
    # independent regexes that must stay structurally identical — if one is
    # widened without the other, a name could enter sweep scope with
    # status=None and bypass the reapable-status allowlist (removable on
    # the idle guards alone). Pin both ways: (a) the issue-name body is
    # contained verbatim in the target pattern; (b) behavioral sweep —
    # every target-matching issue-* name must also match _ISSUE_NAME_RE.
    body = worktree_audit._ISSUE_NAME_RE.pattern.lstrip("^").rstrip("$")
    assert body.replace(r"(\d+)", r"\d+") in worktree_audit._TARGET_NAME_RE.pattern
    alphabet = "a7-._"
    for n in range(0, 4):
        for tail in map("".join, itertools.product(alphabet, repeat=n)):
            name = "issue-48" + tail
            if worktree_audit._TARGET_NAME_RE.match(name):
                assert worktree_audit._ISSUE_NAME_RE.match(name), name


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


# --- Orphaned-codex holder classification (pure) ---------------------------


def test_all_orphan_codex_holders_classified():
    # The three real holder shapes from the 2026-06-10 incident.
    holders = [
        (101, "node /home/t/.npm-global/bin/codex app-server"),
        (102, "node /home/t/.local/bin/codex app-server"),
        (
            103,
            "/usr/bin/node /home/t/.claude/plugins/cache/openai-codex/codex/1.0.4/scri"
            "pts/codex-companion.mjs status task-x",
        ),
    ]
    pids, all_orphan = classify_holders(holders)
    assert pids == [101, 102, 103]
    assert all_orphan


def test_single_real_holder_blocks_all_orphan():
    # A live happy/claude session among the holders makes the worktree
    # non-remediable — never kill toward a real holder.
    holders = [
        (101, "node /home/t/.npm-global/bin/codex app-server"),
        (202, "claude --resume abc (happy session)"),
    ]
    pids, all_orphan = classify_holders(holders)
    assert pids == [101]
    assert not all_orphan


def test_empty_holders_is_not_all_orphan():
    # Vacuous truth must NOT classify an unheld worktree as orphan-pinned.
    assert classify_holders([]) == ([], False)


def test_plain_codex_cli_is_not_an_orphan_pattern():
    # Only `codex app-server` / the plugin cache path match — an interactive
    # `codex exec` (or anything else mentioning codex) is a real holder.
    pids, all_orphan = classify_holders([(7, "codex exec --full-auto fix the bug")])
    assert pids == []
    assert not all_orphan


def test_remediation_statuses_subset_of_reapable_and_include_awaiting_promotion():
    # awaiting_promotion is remediation-eligible as of 2026-06-12: the
    # worktree auto-merged to main at the Step 9b transition and the
    # watcher auto-stops parked sessions; a genuinely live session is
    # still protected by the real-holder guard. Remediation eligibility
    # must never be BROADER than reap eligibility.
    assert worktree_audit.REMEDIATION_ISSUE_STATUSES <= worktree_audit.REAPABLE_ISSUE_STATUSES
    assert "awaiting_promotion" in worktree_audit.REMEDIATION_ISSUE_STATUSES


# --- Junk-dirty rescue allowlist (pure) -------------------------------------


def test_agent_memory_dirt_is_allowlisted():
    paths, ok = dirty_paths_within_allowlist(" M .claude/agent-memory/experimenter/MEMORY.md\n")
    assert ok
    assert paths == [".claude/agent-memory/experimenter/MEMORY.md"]


def test_pods_conf_and_ephemeral_are_allowlisted_exact():
    porcelain = " M scripts/pods_ephemeral.json\n M scripts/pods.conf\n"
    paths, ok = dirty_paths_within_allowlist(porcelain)
    assert ok
    assert paths == ["scripts/pods_ephemeral.json", "scripts/pods.conf"]


def test_exact_entries_do_not_prefix_match():
    _, ok = dirty_paths_within_allowlist(" M scripts/pods.conf.bak\n")
    assert not ok


def test_untracked_lines_are_ignored():
    porcelain = "?? eval_results/scratch.json\n M scripts/pods.conf\n"
    paths, ok = dirty_paths_within_allowlist(porcelain)
    assert ok
    assert paths == ["scripts/pods.conf"]


def test_dirt_outside_allowlist_fails_closed():
    porcelain = " M figures/issue_405/x.png\n M scripts/pods.conf\n"
    paths, ok = dirty_paths_within_allowlist(porcelain)
    assert not ok
    assert "figures/issue_405/x.png" in paths


def test_staged_and_deleted_codes_parse():
    porcelain = "M  scripts/pods.conf\n D scripts/pods_ephemeral.json\n"
    paths, ok = dirty_paths_within_allowlist(porcelain)
    assert ok
    assert paths == ["scripts/pods.conf", "scripts/pods_ephemeral.json"]


def test_rename_requires_both_sides_allowlisted():
    _, ok = dirty_paths_within_allowlist("R  scripts/pods.conf -> scripts/pods2.conf\n")
    assert not ok


def test_quoted_exotic_path_fails_closed():
    _, ok = dirty_paths_within_allowlist(' M "weird name.json"\n')
    assert not ok


def test_empty_porcelain_is_vacuously_within():
    assert dirty_paths_within_allowlist("") == ([], True)


# --- Remediation triage (_remediation_kind, injected data only) -------------


def test_remediation_kind_orphan_branch():
    d = Decision("issue-331", False, "held by a live process")
    holders = [(101, "node /x/codex app-server")]
    kind = worktree_audit._remediation_kind("issue-331", d, "completed", holders, "/nonexistent")
    assert kind is not None
    assert kind[0] == "orphan-pinned"


def test_remediation_kind_refuses_non_terminal_statuses():
    d = Decision("issue-331", False, "held by a live process")
    holders = [(101, "node /x/codex app-server")]
    for status in ("running", "blocked", "followups_running", None):
        assert (
            worktree_audit._remediation_kind("issue-331", d, status, holders, "/nonexistent")
            is None
        ), status


def test_remediation_kind_orphan_branch_awaiting_promotion():
    # awaiting_promotion worktrees are remediable as of 2026-06-12 — the
    # orphan-pinned classification applies the same as completed/archived.
    d = Decision("issue-563", False, "held by a live process")
    holders = [(101, "node /x/codex app-server")]
    kind = worktree_audit._remediation_kind("issue-563", d, "awaiting_promotion", holders, "/x")
    assert kind is not None
    assert kind[0] == "orphan-pinned"


def test_remediation_kind_refuses_real_holder():
    d = Decision("issue-331", False, "held by a live process")
    holders = [(101, "node /x/codex app-server"), (202, "claude --resume abc")]
    assert worktree_audit._remediation_kind("issue-331", d, "completed", holders, "/x") is None


def test_remediation_kind_fails_closed_on_unreadable_porcelain():
    # tracked-changes keep + a worktree whose git status cannot be read
    # (here: nonexistent path) must NOT classify as junk-dirty.
    d = Decision("issue-470", False, "has uncommitted tracked changes")
    assert worktree_audit._remediation_kind("issue-470", d, "completed", [], "/nonexistent") is None


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


# --- single-instance lock ------------------------------------------------


def test_single_instance_lock_second_acquire_returns_none(tmp_path):
    lock_path = tmp_path / "worktree-audit.lock"
    holder = worktree_audit.acquire_single_instance_lock(lock_path)
    assert holder is not None
    try:
        assert worktree_audit.acquire_single_instance_lock(lock_path) is None
    finally:
        holder.close()
    # After the holder releases, the lock is acquirable again.
    reacquired = worktree_audit.acquire_single_instance_lock(lock_path)
    assert reacquired is not None
    reacquired.close()


def test_single_instance_lock_creates_parent_dir(tmp_path):
    lock_path = tmp_path / "nested" / "dir" / "worktree-audit.lock"
    holder = worktree_audit.acquire_single_instance_lock(lock_path)
    assert holder is not None
    holder.close()


def test_main_exits_zero_when_lock_held(tmp_path, monkeypatch, capsys):
    # A second concurrent audit must be a CLEAN skip (exit 0): the cron
    # wrapper and the watcher's fail-soft subprocess call both treat
    # nonzero as a failure signal.
    lock_path = tmp_path / "worktree-audit.lock"
    monkeypatch.setattr(worktree_audit, "_LOCK_PATH", lock_path)
    holder = worktree_audit.acquire_single_instance_lock(lock_path)
    assert holder is not None
    try:
        rc = worktree_audit.main([])
        assert rc == 0
        out = capsys.readouterr().out
        assert "holds the lock" in out
    finally:
        holder.close()


def test_main_lock_skip_emits_json_when_requested(tmp_path, monkeypatch, capsys):
    import json as _json

    lock_path = tmp_path / "worktree-audit.lock"
    monkeypatch.setattr(worktree_audit, "_LOCK_PATH", lock_path)
    holder = worktree_audit.acquire_single_instance_lock(lock_path)
    assert holder is not None
    try:
        rc = worktree_audit.main(["--json"])
        assert rc == 0
        payload = _json.loads(capsys.readouterr().out)
        assert "skipped" in payload
    finally:
        holder.close()


def test_main_runs_audit_when_lock_is_free(tmp_path, monkeypatch, capsys):
    # The guard must not break the normal path: lock free -> audit() runs
    # and main() returns the pre-existing exit contract (0, nothing removed).
    lock_path = tmp_path / "worktree-audit.lock"
    monkeypatch.setattr(worktree_audit, "_LOCK_PATH", lock_path)
    calls = []

    def stub_audit(*, apply, grace_hours):
        calls.append((apply, grace_hours))
        return worktree_audit.AuditResult()

    monkeypatch.setattr(worktree_audit, "audit", stub_audit)
    rc = worktree_audit.main([])
    assert rc == 0
    assert calls == [(False, worktree_audit.DEFAULT_GRACE_HOURS)]
    assert "would remove 0" in capsys.readouterr().out
