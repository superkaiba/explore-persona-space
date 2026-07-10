"""Tests for explore_persona_space.task_workflow.

Each test runs in a temporary directory that is set up as a fake repo
(git init + minimal layout). Git commits are NOT skipped — we want the
end-to-end behavior (git mv during set_status, etc.) under test — but
auto-push is disabled by leaving TASK_PY_AUTO_PUSH unset.
"""

# The fixture body strings below include long lines that mirror real
# clean-result content (Why-this-experiment Application/Decision lines
# carry ≥40 chars of substance and tend to exceed 100 cols). Reflowing
# them would change the markdown structure under test.

from __future__ import annotations

import argparse
import concurrent.futures
import json
import logging
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

# ─── Fake-repo fixture ─────────────────────────────────────────────────────


@pytest.fixture
def fake_repo(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Set up tmp_path as a git repo and rebind task_workflow's resolver to
    point at it. Returns the repo root path.

    The 2026-05-25 worktree-staleness fix replaced module-level
    ``REPO``/``TASKS_DIR``/``REGISTRY_PATH`` constants with the function
    accessors ``repo_root()`` / ``tasks_dir()`` / ``registry_path()``
    (with a PEP-562 attribute shim for backward compatibility — see
    ``task_workflow.py`` header). Tests now monkeypatch the FUNCTIONS,
    not the attributes, so every in-module call site picks up the tmp
    repo. The branch guard inside the real ``repo_root()`` would
    otherwise refuse to resolve from a non-``main`` development branch.
    """
    subprocess.run(["git", "init", "-q", "-b", "main"], cwd=tmp_path, check=True)
    subprocess.run(["git", "config", "user.email", "test@test.test"], cwd=tmp_path, check=True)
    subprocess.run(["git", "config", "user.name", "test"], cwd=tmp_path, check=True)
    subprocess.run(["git", "config", "commit.gpgsign", "false"], cwd=tmp_path, check=True)
    # initial empty commit so HEAD exists
    subprocess.run(["git", "commit", "-q", "--allow-empty", "-m", "init"], cwd=tmp_path, check=True)

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    import explore_persona_space.task_workflow as tw

    # Drop any cached resolution from a prior test so our overrides win.
    tw.invalidate_cache()

    monkeypatch.setattr(tw, "repo_root", lambda: tmp_path)
    monkeypatch.setattr(tw, "tasks_dir", lambda: tmp_path / "tasks")
    monkeypatch.setattr(tw, "registry_path", lambda: tmp_path / "tasks" / "REGISTRY.json")
    # Per-test lock dir to avoid cross-talk
    lock_dir = tmp_path / ".task-workflow"
    monkeypatch.setattr(tw, "LOCK_DIR", lock_dir)
    monkeypatch.setattr(tw, "LOCK_PATH", lock_dir / "lock")
    # Per-test deferred-commit sidecar (#1030) so no test can ever write the
    # REAL ~/.task-workflow/deferred-commits.jsonl.
    monkeypatch.setattr(tw, "DEFERRED_COMMITS_LOG", lock_dir / "deferred-commits.jsonl")
    # Per-test stranded-commits sidecar (#1100) so no test can ever write the
    # REAL ~/.task-workflow/stranded-commits.jsonl.
    monkeypatch.setattr(tw, "STRANDED_COMMITS_LOG", lock_dir / "stranded-commits.jsonl")
    return tmp_path, tw


def _git_log_count(repo: Path) -> int:
    out = subprocess.run(
        ["git", "rev-list", "--count", "HEAD"], cwd=repo, capture_output=True, text=True, check=True
    )
    return int(out.stdout.strip())


# ─── Smoke: import the module ──────────────────────────────────────────────


def test_module_imports():
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    import explore_persona_space.task_workflow as tw

    assert "proposed" in tw.STATUSES
    assert "completed" in tw.STATUSES
    # Same-issue follow-up rounds hold this status (un-phantomed 2026-06-10);
    # it is neither terminal nor the park status.
    assert "followups_running" in tw.STATUSES
    assert "followups_running" not in tw.TERMINAL_STATUSES
    assert tw.PARK_STATUS == "awaiting_promotion"


# ─── Frontmatter parsing ──────────────────────────────────────────────────


def test_frontmatter_roundtrip(fake_repo):
    _, tw = fake_repo
    text = "---\ntitle: Foo\nkind: experiment\ntags:\n  - a\n  - b\n---\nbody here\n"
    fm, body = tw._split_frontmatter(text)
    assert fm["title"] == "Foo"
    assert fm["kind"] == "experiment"
    assert fm["tags"] == ["a", "b"]
    assert body == "body here\n"
    rebuilt = tw._join_frontmatter(fm, body)
    fm2, body2 = tw._split_frontmatter(rebuilt)
    assert fm2 == fm
    assert body2 == body


def test_frontmatter_missing():
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    import explore_persona_space.task_workflow as tw

    fm, body = tw._split_frontmatter("hello no frontmatter\n")
    assert fm == {}
    assert body == "hello no frontmatter\n"


# ─── create_task ──────────────────────────────────────────────────────────


def test_create_task_basic(fake_repo):
    repo, tw = fake_repo
    new_id = tw.create_task(
        tw.NewTaskRequest(kind="experiment", title="My first task", body="Goal: do X")
    )
    assert new_id == 1
    task_dir = repo / "tasks" / "proposed" / "1"
    assert task_dir.is_dir()
    assert (task_dir / "body.md").exists()
    assert (task_dir / "events.jsonl").exists()
    assert (task_dir / "comments.jsonl").exists()
    assert (task_dir / "artifacts").is_dir()
    assert (task_dir / "plans").is_dir()
    # Frontmatter populated
    fm, body = tw._split_frontmatter((task_dir / "body.md").read_text())
    assert fm["title"] == "My first task"
    assert fm["kind"] == "experiment"
    assert "Goal: do X" in body
    # Registry updated
    reg = json.loads((repo / "tasks" / "REGISTRY.json").read_text())
    assert reg["highest_id"] == 1
    assert reg["tasks"]["1"]["path"] == "tasks/proposed/1"
    # Created event present
    events = tw.list_events(1)
    assert events[0]["kind"] == "epm:created"


def test_create_task_increments_id(fake_repo):
    _, tw = fake_repo
    a = tw.create_task(tw.NewTaskRequest(kind="experiment", title="A"))
    b = tw.create_task(tw.NewTaskRequest(kind="experiment", title="B"))
    c = tw.create_task(tw.NewTaskRequest(kind="experiment", title="C"))
    assert (a, b, c) == (1, 2, 3)


def test_create_task_with_parent(fake_repo):
    _, tw = fake_repo
    parent = tw.create_task(tw.NewTaskRequest(kind="experiment", title="Parent"))
    child = tw.create_task(tw.NewTaskRequest(kind="experiment", title="Child", parent_id=parent))
    task = tw.get_task(child)
    assert task["frontmatter"]["parent_id"] == parent


def test_create_task_invalid_kind_raises(fake_repo):
    """`create_task` validates `kind` against KINDS (mirror of the existing
    `status` check) so a programmatic caller cannot write a garbage kind to
    frontmatter + REGISTRY — the same guarantee `set_kind` already gives
    (incident #672 follow-up). No task folder is left behind on the reject."""
    repo, tw = fake_repo
    with pytest.raises(ValueError, match="unknown kind"):
        tw.create_task(tw.NewTaskRequest(kind="not-a-kind", title="X"))
    # The validation happens before any filesystem write — no orphan folder.
    assert not (repo / "tasks" / "proposed" / "1").exists()


def test_create_task_accepts_every_kind(fake_repo):
    """Every member of KINDS is a valid `create_task` kind (guards against a
    KINDS member that the validation would wrongly reject)."""
    _, tw = fake_repo
    for kind in tw.KINDS:
        tw.create_task(tw.NewTaskRequest(kind=kind, title=f"t-{kind}"))


def test_code_kinds_is_canonical_subset_of_kinds():
    """`CODE_KINDS` (the canonical test-verdict/code-change subset that
    `task_progress` + `verify_plan` derive from) is a proper subset of the
    lifecycle enum, and byte-identical to the historical literal. Pins the
    single source of truth so the three former copies can never drift
    (incident #672)."""
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    import explore_persona_space.task_workflow as tw

    assert frozenset({"infra", "analysis", "batch", "survey"}) == tw.CODE_KINDS
    assert frozenset(tw.KINDS) > tw.CODE_KINDS  # proper subset
    assert "experiment" not in tw.CODE_KINDS


def test_create_task_with_origin_prompt(fake_repo):
    """`origin_prompt` writes a frontmatter field verbatim (any kind);
    empty/whitespace-only values write NO field. The clean-result
    `## Reproducibility` `**Context:**` row carries it forward
    (SPEC.md; verify_task_body.py check 17)."""
    _, tw = fake_repo
    with_prompt = tw.create_task(
        tw.NewTaskRequest(
            kind="experiment",
            title="With prompt",
            origin_prompt="Add an issue to look into this",
        )
    )
    task = tw.get_task(with_prompt)
    assert task["frontmatter"]["origin_prompt"] == "Add an issue to look into this"
    without = tw.create_task(
        tw.NewTaskRequest(kind="experiment", title="No prompt", origin_prompt="   ")
    )
    assert "origin_prompt" not in tw.get_task(without)["frontmatter"]


# ─── Status transitions ──────────────────────────────────────────────────


def test_set_status_moves_folder(fake_repo):
    repo, tw = fake_repo
    new_id = tw.create_task(tw.NewTaskRequest(kind="experiment", title="X"))
    old = repo / "tasks" / "proposed" / str(new_id)
    assert old.is_dir()
    tw.set_status(new_id, "running")
    new = repo / "tasks" / "running" / str(new_id)
    assert not old.exists()
    assert new.is_dir()
    # Registry updated
    reg = json.loads((repo / "tasks" / "REGISTRY.json").read_text())
    assert reg["tasks"][str(new_id)]["path"] == f"tasks/running/{new_id}"
    # Status-changed event posted
    events = tw.list_events(new_id)
    assert events[-1]["kind"] == "epm:status-changed"
    assert events[-1]["from"] == "proposed"
    assert events[-1]["to"] == "running"


def test_set_status_invalid_raises(fake_repo):
    _, tw = fake_repo
    new_id = tw.create_task(tw.NewTaskRequest(kind="experiment", title="X"))
    with pytest.raises(ValueError):
        tw.set_status(new_id, "not-a-status")


def test_set_status_idempotent(fake_repo):
    _, tw = fake_repo
    new_id = tw.create_task(tw.NewTaskRequest(kind="experiment", title="X"))
    n_before = len(tw.list_events(new_id))
    tw.set_status(new_id, "proposed")  # no-op
    n_after = len(tw.list_events(new_id))
    assert n_after == n_before  # no new event when already there


def test_set_status_commits_both_sides_of_move(fake_repo):
    """Regression: ``set_status`` must commit BOTH the source-path deletion
    AND the destination-path addition of its ``git mv``, so the index is
    clean afterward. Otherwise the source-path deletion lingers as a
    staged change and gets swept into the next unrelated ``git commit``.

    Incident: 2026-05-24, tasks 382/383 source-side deletions in
    ``tasks/proposed/`` were left staged by ``set_status proposed →
    planning`` and got swept into commit 49e49f4a (an unrelated
    ``.claude/agents/planner.md`` edit), under a misleading commit
    message.
    """
    repo, tw = fake_repo
    new_id = tw.create_task(tw.NewTaskRequest(kind="experiment", title="Move-me"))
    tw.set_status(new_id, "planning")

    # After set_status, the index must be CLEAN — no orphan staged
    # deletion for the source path.
    diff_cached = subprocess.run(
        ["git", "diff", "--cached", "--name-only"],
        cwd=repo,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    assert diff_cached == "", (
        f"set_status left orphan staged changes in the index: {diff_cached!r}. "
        f"The source-side deletion of `git mv` was not included in the commit."
    )

    # And the HEAD commit must record BOTH sides of the move.
    show = subprocess.run(
        ["git", "show", "HEAD", "--name-status", "--format="],
        cwd=repo,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.splitlines()
    # Expect a delete row (D) for the source and either A (added) or R
    # (renamed) for the destination.
    deleted = [line for line in show if line.startswith("D\t")]
    added_or_renamed = [line for line in show if line.startswith(("A\t", "R"))]
    assert deleted, f"set_status commit missing source deletion: {show}"
    assert added_or_renamed, f"set_status commit missing destination addition: {show}"


# ─── Destination-collision guard (incident #681) ──────────────────────────
#
# `git mv SRC DST` where DST is an existing directory does NOT error — git
# nests SRC inside DST as tasks/<new>/<id>/<id>/ and exits 0, so the failure
# surfaces only later at `_read_body(new / "body.md")`, leaving the
# transition half-applied (source deleted, dest nested, REGISTRY/event/commit
# never written). set_status must guard the destination up front: remove an
# EMPTY orphan and proceed as a true rename; refuse a NON-EMPTY orphan (or a
# non-directory) before any destructive `git mv` runs.


def test_set_status_recovers_empty_orphan_destination(fake_repo):
    """An empty orphan destination dir (e.g. left by a prior numbering;
    git does not track empty dirs) must be removed so `git mv` does a true
    rename, NOT nested as tasks/<new>/<id>/<id>/ (incident #681)."""
    repo, tw = fake_repo
    new_id = tw.create_task(tw.NewTaskRequest(kind="experiment", title="X"))
    # Pre-create an EMPTY orphan at the destination.
    orphan = repo / "tasks" / "running" / str(new_id)
    orphan.mkdir(parents=True)
    assert orphan.is_dir() and not any(orphan.iterdir())

    new = tw.set_status(new_id, "running")

    # Moved cleanly — destination has body.md directly, NOT under <id>/<id>/.
    assert new == repo / "tasks" / "running" / str(new_id)
    assert (new / "body.md").is_file()
    assert not (new / str(new_id)).exists()  # no nesting
    assert not (repo / "tasks" / "proposed" / str(new_id)).exists()  # source gone
    # Registry + event still correct.
    reg = json.loads((repo / "tasks" / "REGISTRY.json").read_text())
    assert reg["tasks"][str(new_id)]["path"] == f"tasks/running/{new_id}"
    assert tw.list_events(new_id)[-1]["kind"] == "epm:status-changed"


def test_set_status_refuses_nonempty_orphan_destination(fake_repo):
    """A non-empty orphan destination dir (leftover artifacts / concurrent
    writer) must raise ValueError naming the path — NOT nest + crash
    half-way (incident #681)."""
    repo, tw = fake_repo
    new_id = tw.create_task(tw.NewTaskRequest(kind="experiment", title="X"))
    # Pre-create a NON-EMPTY orphan at the destination (leftover artifact).
    orphan = repo / "tasks" / "running" / str(new_id)
    (orphan / "artifacts").mkdir(parents=True)
    (orphan / "artifacts" / "leftover.txt").write_text("stale\n")

    with pytest.raises(ValueError, match="already exists and is non-empty"):
        tw.set_status(new_id, "running")

    # Nothing moved: source still in place, no nesting created.
    assert (repo / "tasks" / "proposed" / str(new_id) / "body.md").is_file()
    assert not (orphan / str(new_id)).exists()
    # Status unchanged in the registry.
    reg = json.loads((repo / "tasks" / "REGISTRY.json").read_text())
    assert reg["tasks"][str(new_id)]["path"] == f"tasks/proposed/{new_id}"


def test_set_status_refuses_destination_that_is_a_file(fake_repo):
    """Defensive branch: a destination path that exists as a FILE (not a
    directory) must raise ValueError, not nest or crash on the later
    `git mv` (incident #681)."""
    repo, tw = fake_repo
    new_id = tw.create_task(tw.NewTaskRequest(kind="experiment", title="X"))
    # Pre-create the destination as a FILE.
    dest_parent = repo / "tasks" / "running"
    dest_parent.mkdir(parents=True)
    dest_file = dest_parent / str(new_id)
    dest_file.write_text("x")
    assert dest_file.is_file()

    with pytest.raises(ValueError, match="exists and is not a directory"):
        tw.set_status(new_id, "running")

    # Nothing moved: source still in place.
    assert (repo / "tasks" / "proposed" / str(new_id) / "body.md").is_file()
    # Status unchanged in the registry.
    reg = json.loads((repo / "tasks" / "REGISTRY.json").read_text())
    assert reg["tasks"][str(new_id)]["path"] == f"tasks/proposed/{new_id}"


# ─── Atomic status-transition move — never drops untracked files (#722) ────
#
# `git mv <src-dir> <dst-dir>` renames only git-TRACKED files, silently
# leaving untracked/uncommitted files behind and splitting the task across two
# folders (#722: a task's plans/ landed under the new status while body.md
# stayed under the old). set_status now moves the WHOLE dir via shutil.move
# and verifies completeness, rolling the FS move back BEFORE REGISTRY is
# touched on any partial failure.


def test_set_status_moves_untracked_files(fake_repo):
    """The #722 regression: a status move must carry UNTRACKED / uncommitted
    files too, not just git-tracked ones. Drop an untracked plans file into
    the task dir (no commit), move the task, and assert the untracked file
    landed at the destination, the source dir is fully gone, and REGISTRY
    points at the new location."""
    repo, tw = fake_repo
    new_id = tw.create_task(tw.NewTaskRequest(kind="experiment", title="X"))
    old = repo / "tasks" / "proposed" / str(new_id)
    # An UNTRACKED file written after creation (create_task committed the dir,
    # so this file is not in git — exactly the #722 shape).
    untracked = old / "plans" / "v2.md"
    untracked.parent.mkdir(parents=True, exist_ok=True)
    untracked.write_text("# uncommitted plan version\n")

    tw.set_status(new_id, "running")

    new = repo / "tasks" / "running" / str(new_id)
    # (i) the untracked file exists at the destination
    assert (new / "plans" / "v2.md").is_file()
    assert (new / "plans" / "v2.md").read_text() == "# uncommitted plan version\n"
    # (ii) NO file remains under the source dir
    assert not old.exists()
    # (iii) REGISTRY path == the new location
    reg = json.loads((repo / "tasks" / "REGISTRY.json").read_text())
    assert reg["tasks"][str(new_id)]["path"] == f"tasks/running/{new_id}"


def test_set_status_rolls_back_before_registry_on_incomplete_move(fake_repo):
    """On an incomplete move (a file missing from the destination after the FS
    move) set_status must raise, roll the FS move back to the ORIGINAL
    location with ALL files intact, and leave REGISTRY pointing at the
    ORIGINAL path (untouched — REGISTRY is only written AFTER the verified
    move)."""
    repo, tw = fake_repo
    new_id = tw.create_task(tw.NewTaskRequest(kind="experiment", title="X"))
    old = repo / "tasks" / "proposed" / str(new_id)
    # Give the task an extra file so we can withhold exactly one at the dest.
    (old / "plans").mkdir(parents=True, exist_ok=True)
    (old / "plans" / "v1.md").write_text("plan v1\n")

    real_move = tw.shutil.move
    stash = repo / "_stash_v1.md"

    def flaky_move(src, dst):
        # Model a partial FS move: on the FORWARD move (into `running`),
        # perform the real move but WITHHOLD one file to a scratch stash so
        # the post-move completeness check sees a missing file (the #722
        # fault line). On the ROLLBACK move (back to `proposed`) the file is
        # restored first, so the rollback genuinely returns ALL files — this
        # exercises _rollback_move without destroying data.
        real_move(src, dst)
        dst_v1 = Path(dst) / "plans" / "v1.md"
        if str(dst).endswith("tasks/running/" + str(new_id)):
            real_move(str(dst_v1), str(stash))  # withhold — dst now incomplete
        elif str(dst).endswith("tasks/proposed/" + str(new_id)):
            real_move(str(stash), str(dst_v1))  # restore on rollback

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(tw.shutil, "move", flaky_move)
        with pytest.raises(RuntimeError, match="left 1 file"):
            tw.set_status(new_id, "running")

    # Rolled back: the task dir is back at its ORIGINAL location with ALL files.
    assert old.is_dir()
    assert (old / "body.md").is_file()
    assert (old / "plans" / "v1.md").is_file()
    assert (old / "plans" / "v1.md").read_text() == "plan v1\n"
    # Destination is gone (rolled back).
    assert not (repo / "tasks" / "running" / str(new_id)).exists()
    # REGISTRY untouched — still points at the ORIGINAL path.
    reg = json.loads((repo / "tasks" / "REGISTRY.json").read_text())
    assert reg["tasks"][str(new_id)]["path"] == f"tasks/proposed/{new_id}"


# ─── Typed stale-path error from _read_body / get_task (#786 item b) ───────
#
# A task dir present but body.md missing (the #722 split / stale-registry
# shape) must raise a TYPED error naming the path + the `task.py audit`
# remedy, instead of a bare FileNotFoundError. The typed error subclasses
# FileNotFoundError so existing `except FileNotFoundError` callers still catch
# it (cmd_list_clean_results, cmd_migrate_body, task_workflow_migrate).


def test_get_task_raises_stale_task_path_error_on_missing_body(fake_repo):
    repo, tw = fake_repo
    new_id = tw.create_task(tw.NewTaskRequest(kind="experiment", title="X"))
    body = repo / "tasks" / "proposed" / str(new_id) / "body.md"
    body.unlink()
    with pytest.raises(tw.StaleTaskPathError) as exc_info:
        tw.get_task(new_id)
    msg = str(exc_info.value)
    assert str(body) in msg  # names the stale path
    assert "task.py audit" in msg  # names the remedy


def test_stale_task_path_error_is_filenotfounderror_subclass():
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    import explore_persona_space.task_workflow as tw

    assert issubclass(tw.StaleTaskPathError, FileNotFoundError)
    # An existing `except FileNotFoundError:` catches an instance.
    caught = False
    try:
        raise tw.StaleTaskPathError("boom")
    except FileNotFoundError:
        caught = True
    assert caught


# ─── Same-issue follow-up status-hold guard ───────────────────────────────
#
# The same-issue follow-up status-hold rule (SKILL.md Step 9b § Same-issue
# follow-up loop, step 3): a `followups_running` task is HELD for the whole
# round; set_status refuses re-entry into intermediate pipeline statuses.
# Incident: tasks #533/#560 (2026-06-10/11) flipped to `running` mid-round.


def test_followup_held_blocked_statuses_membership():
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    import explore_persona_space.task_workflow as tw

    # Every blocked member is a valid status...
    assert set(tw.STATUSES) >= tw.FOLLOWUP_HELD_BLOCKED_STATUSES
    # ...and the round's legitimate exits are NOT blocked.
    for allowed_exit in ("awaiting_promotion", "blocked", "completed", "archived"):
        assert allowed_exit not in tw.FOLLOWUP_HELD_BLOCKED_STATUSES
    # The intermediate pipeline statuses ARE blocked.
    for held in (
        "planning",
        "plan_pending",
        "approved",
        "running",
        "verifying",
        "interpreting",
        "reviewing",
    ):
        assert held in tw.FOLLOWUP_HELD_BLOCKED_STATUSES


def test_set_status_followup_hold_blocks_pipeline_reentry(fake_repo):
    repo, tw = fake_repo
    for blocked in sorted(tw.FOLLOWUP_HELD_BLOCKED_STATUSES):
        new_id = tw.create_task(tw.NewTaskRequest(kind="experiment", title=f"hold-{blocked}"))
        tw.set_status(new_id, "followups_running")
        with pytest.raises(ValueError, match="status-hold rule"):
            tw.set_status(new_id, blocked)
        # Task folder untouched: still held at followups_running.
        assert (repo / "tasks" / "followups_running" / str(new_id)).is_dir()
        assert not (repo / "tasks" / blocked / str(new_id)).exists()


def test_set_status_followup_hold_force_flag_overrides(fake_repo):
    repo, tw = fake_repo
    new_id = tw.create_task(tw.NewTaskRequest(kind="experiment", title="force-exit"))
    tw.set_status(new_id, "followups_running")
    tw.set_status(new_id, "running", force_followup_exit=True)
    assert (repo / "tasks" / "running" / str(new_id)).is_dir()


def test_set_status_followup_hold_exit_paths_allowed(fake_repo):
    repo, tw = fake_repo
    for allowed in ("awaiting_promotion", "blocked", "completed", "archived", "proposed"):
        new_id = tw.create_task(tw.NewTaskRequest(kind="experiment", title=f"exit-{allowed}"))
        tw.set_status(new_id, "followups_running")
        tw.set_status(new_id, allowed)  # must not raise
        assert (repo / "tasks" / allowed / str(new_id)).is_dir()


def test_set_status_followup_hold_only_guards_followups_source(fake_repo):
    """The guard keys on the SOURCE status: a normal pipeline task moves
    freely between intermediate statuses."""
    repo, tw = fake_repo
    new_id = tw.create_task(tw.NewTaskRequest(kind="experiment", title="normal"))
    for s in ("planning", "plan_pending", "approved", "running", "verifying"):
        tw.set_status(new_id, s)
    assert (repo / "tasks" / "verifying" / str(new_id)).is_dir()


# ─── post_event ──────────────────────────────────────────────────────────


def test_post_event_appends(fake_repo):
    _, tw = fake_repo
    new_id = tw.create_task(tw.NewTaskRequest(kind="experiment", title="X"))
    tw.post_event(new_id, "epm:plan", by="planner", note="plan v1 written")
    tw.post_event(new_id, "epm:plan-approved", by="user")
    events = tw.list_events(new_id)
    assert [e["kind"] for e in events] == ["epm:created", "epm:plan", "epm:plan-approved"]
    assert events[1]["note"] == "plan v1 written"


def test_post_event_oversize_note_raises(fake_repo):
    _, tw = fake_repo
    new_id = tw.create_task(tw.NewTaskRequest(kind="experiment", title="X"))
    with pytest.raises(ValueError):
        tw.post_event(new_id, "epm:huge", note="x" * (tw.EVENT_NOTE_MAX + 1))


def test_post_event_default_version_auto_increments_per_kind(fake_repo):
    """Omitted version = max(existing for this kind)+1, per kind (#480).

    Two defaulted posts of the same kind must land v1 then v2 — never v1
    twice — so highest-version-per-kind resume resolution stays correct.
    A second kind starts independently at v1.
    """
    _, tw = fake_repo
    new_id = tw.create_task(tw.NewTaskRequest(kind="experiment", title="X"))
    first = tw.post_event(new_id, "epm:code-review-codex", by="orchestrator")
    second = tw.post_event(new_id, "epm:code-review-codex", by="orchestrator")
    other_kind = tw.post_event(new_id, "epm:interpretation", by="analyzer")
    assert first["version"] == 1
    assert second["version"] == 2
    assert other_kind["version"] == 1


def test_post_event_explicit_version_wins_and_seeds_default(fake_repo):
    """An explicit version is respected verbatim (even if lower than the
    current max), and a later defaulted post resumes from the true max —
    mirroring new_plan_version's max+1 (not count+1) semantics.
    """
    _, tw = fake_repo
    new_id = tw.create_task(tw.NewTaskRequest(kind="experiment", title="X"))
    explicit = tw.post_event(new_id, "epm:code-review-codex", version=6, by="orchestrator")
    defaulted = tw.post_event(new_id, "epm:code-review-codex", by="orchestrator")
    lower_explicit = tw.post_event(new_id, "epm:code-review-codex", version=3, by="orchestrator")
    after_lower = tw.post_event(new_id, "epm:code-review-codex", by="orchestrator")
    assert explicit["version"] == 6
    assert defaulted["version"] == 7
    assert lower_explicit["version"] == 3
    assert after_lower["version"] == 8


def test_latest_event(fake_repo):
    _, tw = fake_repo
    new_id = tw.create_task(tw.NewTaskRequest(kind="experiment", title="X"))
    tw.post_event(new_id, "epm:plan")
    tw.post_event(new_id, "epm:run-launched")
    latest = tw.latest_event(new_id)
    assert latest["kind"] == "epm:run-launched"
    # Filter by prefix
    plan_only = tw.latest_event(new_id, prefix="epm:plan")
    assert plan_only["kind"] == "epm:plan"


# ─── Body / title / tags ────────────────────────────────────────────────


def test_set_body_preserves_frontmatter(fake_repo):
    repo, tw = fake_repo
    new_id = tw.create_task(tw.NewTaskRequest(kind="experiment", title="X", body="old body"))
    tw.set_body(new_id, "new body")
    fm, body = tw._split_frontmatter(
        (repo / "tasks" / "proposed" / str(new_id) / "body.md").read_text()
    )
    assert fm["title"] == "X"
    assert body == "new body\n"


def test_set_body_snapshot_creates_original(fake_repo):
    repo, tw = fake_repo
    new_id = tw.create_task(tw.NewTaskRequest(kind="experiment", title="X", body="old body"))
    tw.set_body(new_id, "clean-result content", snapshot_original=True)
    orig = repo / "tasks" / "proposed" / str(new_id) / "original-body.md"
    assert orig.exists()
    assert "old body" in orig.read_text()


# ─── set_body: Goal-H2 drop guard (incident #1112) ─────────────────────────
#
# A `kind: experiment` body update that removes the `## Goal` H2 present in
# the prior body refuses with `GoalH2DropError` unless `allow_goal_drop=True`
# (CLI: --allow-goal-drop). The guard fires ONLY on has→lacks transitions
# (a grandfathered v3/legacy experiment body lacking `## Goal` on the PRIOR
# side is deliberately exempt), only for `kind: experiment`, and never for a
# `paper: true` task (the paper-stub write legitimately lacks `## Goal`).

_GOAL_BODY = "# T\n\n## Goal\n\nMeasure the thing precisely.\n\nMore context here.\n"
_GOALLESS_BODY = "# T\n\nRe-scoped body without a goal heading.\n"


def test_set_body_refuses_goal_h2_drop_on_experiment(fake_repo):
    """The #1112 replay: experiment body with `## Goal` + a goal-less rewrite
    raises, leaves body.md byte-unchanged, and the message names the recovery
    (`--allow-goal-drop`) plus the incident (#1112)."""
    repo, tw = fake_repo
    new_id = tw.create_task(tw.NewTaskRequest(kind="experiment", title="X", body=_GOAL_BODY))
    body_path = repo / "tasks" / "proposed" / str(new_id) / "body.md"
    before = body_path.read_text()
    assert "## Goal" in before  # precondition: the prior body carries the H2
    with pytest.raises(tw.GoalH2DropError) as exc:
        tw.set_body(new_id, _GOALLESS_BODY)
    msg = str(exc.value)
    assert "--allow-goal-drop" in msg
    assert "#1112" in msg
    assert body_path.read_text() == before  # body.md unchanged on refusal


def test_set_body_goal_drop_refusal_writes_no_snapshot(fake_repo):
    """A refusal is side-effect-free: `snapshot_original=True` must NOT have
    written original-body.md (pins the frontmatter-strip hoist ABOVE the
    snapshot copy in `set_body`)."""
    repo, tw = fake_repo
    new_id = tw.create_task(tw.NewTaskRequest(kind="experiment", title="X", body=_GOAL_BODY))
    orig = repo / "tasks" / "proposed" / str(new_id) / "original-body.md"
    with pytest.raises(tw.GoalH2DropError):
        tw.set_body(new_id, _GOALLESS_BODY, snapshot_original=True)
    assert not orig.exists()


def test_set_body_allow_goal_drop_overrides(fake_repo):
    """`allow_goal_drop=True` is the deliberate-drop escape hatch."""
    repo, tw = fake_repo
    new_id = tw.create_task(tw.NewTaskRequest(kind="experiment", title="X", body=_GOAL_BODY))
    tw.set_body(new_id, _GOALLESS_BODY, allow_goal_drop=True)
    _, body = tw._split_frontmatter(
        (repo / "tasks" / "proposed" / str(new_id) / "body.md").read_text()
    )
    assert "## Goal" not in body
    assert body.lstrip().startswith("# T")


def test_set_body_goal_guard_skips_non_experiment_kind(fake_repo):
    """Infra/analysis bodies carry `## Goal` H2s with no downstream Goal-gate
    machinery — the guard must not fire outside `kind: experiment`."""
    repo, tw = fake_repo
    new_id = tw.create_task(tw.NewTaskRequest(kind="infra", title="X", body=_GOAL_BODY))
    tw.set_body(new_id, _GOALLESS_BODY)  # no raise, no flag
    _, body = tw._split_frontmatter(
        (repo / "tasks" / "proposed" / str(new_id) / "body.md").read_text()
    )
    assert "## Goal" not in body


def test_set_body_goal_guard_skips_when_prior_body_lacks_goal(fake_repo):
    """No false fire on the lacks→lacks case — a grandfathered v3/legacy
    experiment body without `## Goal` on the PRIOR side rewrites freely
    (also keeps every existing goal-less fixture-style caller green)."""
    _, tw = fake_repo
    new_id = tw.create_task(tw.NewTaskRequest(kind="experiment", title="X", body="old body"))
    tw.set_body(new_id, _GOALLESS_BODY)  # no raise


def test_set_body_goal_guard_skips_paper_task(fake_repo):
    """A `paper: true` task's goal-less paper-stub write is auto-exempt
    (mirrors the CLI stub-length exemption for paper tasks)."""
    repo, tw = fake_repo
    new_id = tw.create_task(tw.NewTaskRequest(kind="experiment", title="X", body="old"))
    # Install `paper: true` via the _SET_BODY_ROUNDTRIP_KEYS carry (the same
    # opt-in path production paper tasks use), with a Goal-bearing body.
    tw.set_body(new_id, "---\npaper: true\n---\n" + _GOAL_BODY)
    fm, _ = tw._split_frontmatter(
        (repo / "tasks" / "proposed" / str(new_id) / "body.md").read_text()
    )
    assert tw.is_paper_task(fm)  # precondition: the carry installed the opt-in
    # Goal-less paper-stub rewrite succeeds with no flag.
    tw.set_body(new_id, "# T\n\nPaper stub: abstract + paper link, no Goal section.\n")


def test_set_body_goal_guard_allows_goal_preserving_rewrite(fake_repo):
    """The has→has branch — the dominant production path (every analyzer v4
    clean-result promotion writes a Goal-bearing experiment body). Without
    this test the guard could over-fire fleet-wide while every other test
    in this section stays green."""
    repo, tw = fake_repo
    new_id = tw.create_task(tw.NewTaskRequest(kind="experiment", title="X", body=_GOAL_BODY))
    new_body = "# T v2\n\n## Goal\n\nSharper goal sentence.\n\nRewritten body text.\n"
    tw.set_body(new_id, new_body)  # no raise, no flag
    _, body = tw._split_frontmatter(
        (repo / "tasks" / "proposed" / str(new_id) / "body.md").read_text()
    )
    assert "Sharper goal sentence." in body
    assert "Measure the thing precisely." not in body


def test_has_goal_h2_matches_inject_semantics():
    """`_has_goal_h2` matches EXACTLY the `line.strip() == GOAL_H2_NAME`
    semantics `_inject_or_replace_goal_h2` uses — strip-tolerant, but no
    H3 / plural / suffixed variants."""
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    from explore_persona_space.task_workflow import _has_goal_h2

    assert _has_goal_h2("intro\n  ## Goal  \ntext") is True  # strip semantics
    assert _has_goal_h2("## Goal") is True
    assert _has_goal_h2("### Goal") is False
    assert _has_goal_h2("## Goals") is False
    assert _has_goal_h2("## Goal extra") is False
    assert _has_goal_h2("plain text mentioning the goal, no heading") is False


# ─── set_body: duplicate-frontmatter strip ─────────────────────────────────
#
# Regression: task #389 (2026-05-26) — the analyzer wrote draft body files
# carrying frontmatter and passed them through `task.py set-body`; the
# canonical frontmatter prepended on top of the caller's frontmatter, and
# body.md ended up with TWO `---...---` blocks. The dashboard parsed the
# first as the header card and rendered the second as literal YAML at the
# top of the visible body. `set_body()` now strips leading frontmatter
# from the new-body content before write, idempotently.


def _count_frontmatter_blocks(text: str) -> int:
    """Count consecutive leading `---\\n...\\n---\\n` blocks in `text`."""
    count = 0
    rest = text
    while rest.startswith("---\n"):
        end = rest.find("\n---\n", 4)
        if end == -1:
            break
        count += 1
        rest = rest[end + len("\n---\n") :]
    return count


def test_set_body_strips_leading_frontmatter_in_input(fake_repo):
    """A caller passing `---\\n...\\n---\\n<body>` produces exactly ONE
    frontmatter block in body.md — the canonical one — not two stacked.
    """
    repo, tw = fake_repo
    new_id = tw.create_task(tw.NewTaskRequest(kind="experiment", title="X", body="old body"))
    body_with_fm = (
        "---\n"
        "title: A stale title from the caller\n"
        "kind: something_else\n"
        "made_up_field: caller noise\n"
        "---\n"
        "# Real H1 (HIGH confidence)\n\nReal body content here.\n"
    )
    tw.set_body(new_id, body_with_fm)
    written = (repo / "tasks" / "proposed" / str(new_id) / "body.md").read_text()
    assert _count_frontmatter_blocks(written) == 1
    fm, body = tw._split_frontmatter(written)
    # Canonical frontmatter is preserved (the original task title `"X"`),
    # NOT replaced by the caller's "A stale title from the caller".
    assert fm["title"] == "X"
    assert "made_up_field" not in fm
    # Body region starts at the H1, not at a stray `---` line.
    assert body.lstrip().startswith("# Real H1")


def test_set_body_no_frontmatter_unchanged(fake_repo):
    """A caller passing plain body content (no leading `---`) still works —
    the strip is a no-op and only the canonical frontmatter is prepended.
    """
    repo, tw = fake_repo
    new_id = tw.create_task(tw.NewTaskRequest(kind="experiment", title="X", body="old"))
    plain = "# Real H1 (HIGH confidence)\n\nPlain body, no frontmatter.\n"
    tw.set_body(new_id, plain)
    written = (repo / "tasks" / "proposed" / str(new_id) / "body.md").read_text()
    assert _count_frontmatter_blocks(written) == 1
    _, body = tw._split_frontmatter(written)
    assert body.lstrip().startswith("# Real H1")


def test_set_body_strips_multiple_stacked_frontmatter_blocks(fake_repo):
    """Pathological: caller passes content with two stacked frontmatter
    blocks. `set_body` strips ALL of them, leaving exactly one (the
    canonical) frontmatter block in body.md.
    """
    repo, tw = fake_repo
    new_id = tw.create_task(tw.NewTaskRequest(kind="experiment", title="X", body="old"))
    pathological = (
        "---\nfirst: block\n---\n---\nsecond: block\n---\n# H1 (HIGH confidence)\n\nBody text.\n"
    )
    tw.set_body(new_id, pathological)
    written = (repo / "tasks" / "proposed" / str(new_id) / "body.md").read_text()
    assert _count_frontmatter_blocks(written) == 1
    _, body = tw._split_frontmatter(written)
    assert body.lstrip().startswith("# H1")
    assert "first: block" not in written
    assert "second: block" not in written


def test_set_body_strip_is_idempotent(fake_repo, monkeypatch: pytest.MonkeyPatch):
    """Calling `set_body` twice with the same content (once with leading
    frontmatter, once with the same content already stripped) produces
    byte-identical body.md.
    """
    repo, tw = fake_repo
    # Freeze the timestamp source: the two create_task calls below each
    # write `created_at` into frontmatter, so without this they can
    # straddle a second boundary and spuriously break the byte-equality
    # assert (observed flake 2026-06-10). The test's intent — strip
    # idempotency of set_body CONTENT — is unaffected.
    monkeypatch.setattr(tw, "_utcnow_iso", lambda: "2026-01-01T00:00:00Z")
    id_a = tw.create_task(tw.NewTaskRequest(kind="experiment", title="Same", body="old"))
    id_b = tw.create_task(tw.NewTaskRequest(kind="experiment", title="Same", body="old"))
    with_fm = "---\nstale: stuff\n---\n# H1 (HIGH confidence)\n\nIdentical body content here.\n"
    stripped = "# H1 (HIGH confidence)\n\nIdentical body content here.\n"
    tw.set_body(id_a, with_fm)
    tw.set_body(id_b, stripped)
    text_a = (repo / "tasks" / "proposed" / str(id_a) / "body.md").read_text()
    text_b = (repo / "tasks" / "proposed" / str(id_b) / "body.md").read_text()
    # Only the title frontmatter field differs (Same vs Same — actually
    # identical), so the files MUST be byte-identical modulo task id (no
    # id appears in body.md). They should match exactly.
    assert text_a == text_b


def test_strip_leading_frontmatter_blocks_unit():
    """Direct unit test on the private helper — covers the no-frontmatter,
    one-block, two-block, and malformed-block cases."""
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    from explore_persona_space.task_workflow import _strip_leading_frontmatter_blocks as strip

    assert strip("plain body\n") == "plain body\n"
    assert strip("# H1\n\nbody\n") == "# H1\n\nbody\n"
    assert strip("---\nfoo: bar\n---\nbody\n") == "body\n"
    # Stacked blocks
    assert strip("---\na: 1\n---\n---\nb: 2\n---\nbody\n") == "body\n"
    # Malformed leading block (no closing `---`) is left alone
    assert strip("---\nfoo: bar\nno closing\n# H1\n") == "---\nfoo: bar\nno closing\n# H1\n"
    # Leading blank lines after stripping are dropped
    assert strip("---\nfoo: bar\n---\n\n\n# H1\n") == "# H1\n"
    # Idempotence: stripping an already-stripped string is a no-op
    once = strip("---\nfoo: bar\n---\nbody\n")
    twice = strip(once)
    assert once == twice


def test_set_title_updates_registry(fake_repo):
    repo, tw = fake_repo
    new_id = tw.create_task(tw.NewTaskRequest(kind="experiment", title="Old"))
    tw.set_title(new_id, "New title")
    reg = json.loads((repo / "tasks" / "REGISTRY.json").read_text())
    assert reg["tasks"][str(new_id)]["title"] == "New title"


def test_set_kind_updates_frontmatter_and_registry(fake_repo):
    """Reclassifying a misfiled kind (incident #672) writes the frontmatter
    AND the denormalized REGISTRY snapshot the dashboard list view reads."""
    repo, tw = fake_repo
    new_id = tw.create_task(tw.NewTaskRequest(kind="experiment", title="GCP-fix validation"))
    tw.set_kind(new_id, "infra")
    # Frontmatter on body.md
    assert tw.get_task(new_id)["frontmatter"]["kind"] == "infra"
    # REGISTRY denormalizes kind — must stay in sync.
    reg = json.loads((repo / "tasks" / "REGISTRY.json").read_text())
    assert reg["tasks"][str(new_id)]["kind"] == "infra"


def test_set_kind_invalid_raises(fake_repo):
    _, tw = fake_repo
    new_id = tw.create_task(tw.NewTaskRequest(kind="experiment", title="X"))
    with pytest.raises(ValueError):
        tw.set_kind(new_id, "not-a-kind")
    # The frontmatter is unchanged after the rejected call.
    assert tw.get_task(new_id)["frontmatter"]["kind"] == "experiment"


def test_set_kind_commits(fake_repo):
    repo, tw = fake_repo
    new_id = tw.create_task(tw.NewTaskRequest(kind="experiment", title="X"))
    n_before = _git_log_count(repo)
    tw.set_kind(new_id, "infra")
    assert _git_log_count(repo) == n_before + 1


def test_add_remove_tag(fake_repo):
    _, tw = fake_repo
    new_id = tw.create_task(tw.NewTaskRequest(kind="experiment", title="X"))
    tw.add_tag(new_id, "qwen-7b")
    tw.add_tag(new_id, "qwen-7b")  # dedup
    tw.add_tag(new_id, "lang-inv")
    task = tw.get_task(new_id)
    assert task["frontmatter"]["tags"] == ["qwen-7b", "lang-inv"]
    tw.remove_tag(new_id, "qwen-7b")
    task = tw.get_task(new_id)
    assert task["frontmatter"]["tags"] == ["lang-inv"]


# ─── Plans ────────────────────────────────────────────────────────────────


def test_new_plan_version_versions_and_symlinks(fake_repo):
    repo, tw = fake_repo
    new_id = tw.create_task(tw.NewTaskRequest(kind="experiment", title="X"))
    v1 = tw.new_plan_version(new_id, "plan v1 content")
    v2 = tw.new_plan_version(new_id, "plan v2 content")
    v3 = tw.new_plan_version(new_id, "plan v3 content")
    assert (v1, v2, v3) == (1, 2, 3)
    plans_dir = repo / "tasks" / "proposed" / str(new_id) / "plans"
    assert (plans_dir / "v1.md").read_text().strip() == "plan v1 content"
    assert (plans_dir / "v3.md").read_text().strip() == "plan v3 content"
    # Symlink points to latest
    assert (plans_dir / "plan.md").is_symlink()
    assert (plans_dir / "plan.md").resolve() == (plans_dir / "v3.md").resolve()


def test_new_plan_version_skips_gap_uses_max_plus_one(fake_repo):
    """Regression: with a numbering gap (e.g. v1,v2,v3,v4,v6 — no v5,
    because a draft lived only in /tmp and was never registered), the
    next plan MUST be v7 — NOT v6, which would silently overwrite the
    highest existing plan. Closes the task #524 incident: the count-based
    resolver (``len(existing)+1``) computed v6 over an existing v6 and
    destroyed it without warning. Source of truth is now
    ``max(existing v<N>) + 1``.
    """
    repo, tw = fake_repo
    new_id = tw.create_task(tw.NewTaskRequest(kind="experiment", title="X"))
    plans_dir = repo / "tasks" / "proposed" / str(new_id) / "plans"
    plans_dir.mkdir(parents=True, exist_ok=True)
    # Pre-stage a gapped set of plan files (no v5).
    for n in (1, 2, 3, 4, 6):
        (plans_dir / f"v{n}.md").write_text(f"plan v{n} content\n")
    v6_original = (plans_dir / "v6.md").read_text()

    next_v = tw.new_plan_version(new_id, "plan v7 content")

    # MUST advance past the highest existing version, not fill the gap and
    # MUST NOT overwrite v6.
    assert next_v == 7, f"expected v7 (max+1), got v{next_v}"
    assert (plans_dir / "v7.md").read_text().strip() == "plan v7 content"
    assert (plans_dir / "v6.md").read_text() == v6_original, (
        "v6.md was overwritten — the count-based resolver bug has regressed"
    )
    # v5 stays absent — we don't backfill gaps.
    assert not (plans_dir / "v5.md").exists()
    # Symlink points to v7.
    assert (plans_dir / "plan.md").resolve() == (plans_dir / "v7.md").resolve()


def test_new_plan_version_refuses_to_overwrite_existing_target(
    fake_repo, monkeypatch: pytest.MonkeyPatch
):
    """Belt-and-suspenders: the resolver derives ``next_v = max(existing) + 1``
    inside ``_locked()`` and writes immediately after — so under normal
    operation the computed target file CANNOT pre-exist. The explicit
    ``target.exists()`` guard fires only if something external creates
    the file between the glob and the write (a process holding no lock,
    a filesystem race, manual staging during the critical section). The
    guard is cheap and documents the invariant. To exercise it we simulate
    that race by wrapping the lock so a sentinel file appears at the
    computed slot after the glob but before the write.
    """
    repo, tw = fake_repo
    new_id = tw.create_task(tw.NewTaskRequest(kind="experiment", title="X"))
    plans_dir = repo / "tasks" / "proposed" / str(new_id) / "plans"
    plans_dir.mkdir(parents=True, exist_ok=True)
    (plans_dir / "v1.md").write_text("plan v1 content\n")
    sentinel = "PRE_STAGED_SHOULD_NOT_BE_OVERWRITTEN\n"

    # Race simulation: replace Path.write_text so the first call (the
    # resolver's write to v2.md) finds v2.md already present. Note the
    # resolver writes v2.md FIRST, then the symlink — so we intercept on
    # the first call only and re-raise via the resolver's own guard.
    real_glob = type(plans_dir).glob

    def racing_glob(self, pattern):
        result = list(real_glob(self, pattern))
        # Inject the racing pre-existing file BEFORE write_text runs.
        if self == plans_dir and pattern == "v*.md":
            (plans_dir / "v2.md").write_text(sentinel)
        return iter(result)

    monkeypatch.setattr(type(plans_dir), "glob", racing_glob)

    with pytest.raises(RuntimeError, match=r"refusing to overwrite.*v2\.md"):
        tw.new_plan_version(new_id, "plan v2 fresh content")

    # The racing pre-existing file is preserved untouched.
    assert (plans_dir / "v2.md").read_text() == sentinel


# ─── Promotion ───────────────────────────────────────────────────────────


def test_promote_requires_awaiting_promotion(fake_repo):
    _, tw = fake_repo
    new_id = tw.create_task(tw.NewTaskRequest(kind="experiment", title="X"))
    with pytest.raises(RuntimeError):
        tw.promote(new_id, "useful")


def test_promote_useful(fake_repo):
    _, tw = fake_repo
    new_id = tw.create_task(tw.NewTaskRequest(kind="experiment", title="X"))
    tw.set_status(new_id, "awaiting_promotion")
    new_path = tw.promote(new_id, "useful")
    assert "completed" in str(new_path)
    task = tw.get_task(new_id)
    assert task["status"] == "completed"
    assert task["frontmatter"]["classification"] == "useful"
    # Both epm:promoted and epm:status-changed (to completed) appended
    kinds = [e["kind"] for e in tw.list_events(new_id)]
    assert "epm:promoted" in kinds
    assert kinds[-1] == "epm:status-changed"


def test_promote_invalid_verdict(fake_repo):
    _, tw = fake_repo
    new_id = tw.create_task(tw.NewTaskRequest(kind="experiment", title="X"))
    tw.set_status(new_id, "awaiting_promotion")
    with pytest.raises(ValueError):
        tw.promote(new_id, "maybe")


# ─── Queries ──────────────────────────────────────────────────────────────


def test_list_by_status(fake_repo):
    _, tw = fake_repo
    a = tw.create_task(tw.NewTaskRequest(kind="experiment", title="A"))
    b = tw.create_task(tw.NewTaskRequest(kind="experiment", title="B"))
    tw.set_status(a, "running")
    rows = tw.list_by_status("proposed")
    assert {r["id"] for r in rows} == {b}
    rows = tw.list_by_status("running")
    assert {r["id"] for r in rows} == {a}


def test_find_task_path(fake_repo):
    repo, tw = fake_repo
    new_id = tw.create_task(tw.NewTaskRequest(kind="experiment", title="X"))
    path = tw.find_task_path(new_id)
    assert path == repo / "tasks" / "proposed" / str(new_id)


def test_find_task_path_missing(fake_repo):
    _, tw = fake_repo
    with pytest.raises(FileNotFoundError):
        tw.find_task_path(99999)


# ─── Audit ────────────────────────────────────────────────────────────────


def test_audit_clean(fake_repo):
    _, tw = fake_repo
    tw.create_task(tw.NewTaskRequest(kind="experiment", title="A"))
    tw.create_task(tw.NewTaskRequest(kind="experiment", title="B"))
    assert tw.audit() == []


def test_audit_detects_orphan_dir(fake_repo):
    repo, tw = fake_repo
    tw.create_task(tw.NewTaskRequest(kind="experiment", title="A"))
    # Create a task folder on disk WITHOUT registering
    orphan = repo / "tasks" / "proposed" / "9999"
    orphan.mkdir(parents=True)
    (orphan / "body.md").write_text("---\ntitle: orphan\n---\n")
    problems = tw.audit()
    assert any("9999" in p for p in problems)


# ─── Registry reconcile (`audit --repair`) ─────────────────────────────────


def _import_task_cli():
    """Import scripts/task.py as `task` (the CLI handler layer). Exercised at
    the handler-function layer, not via subprocess — `repo_root()` branch-guards
    to `main`, so a subprocess would bypass the fake_repo monkeypatch (same
    rationale as test_cli_handlers_raise_address_defer_list_roundtrip)."""
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    import task as task_cli  # type: ignore[import-not-found]

    return task_cli


def _read_registry(repo: Path) -> dict:
    return json.loads((repo / "tasks" / "REGISTRY.json").read_text())


def _write_registry(repo: Path, reg: dict) -> None:
    (repo / "tasks" / "REGISTRY.json").write_text(json.dumps(reg, indent=2, sort_keys=True) + "\n")


def _make_orphan_with_body(
    repo: Path, tid: int, status: str = "proposed", *, title: str = "orphan", kind: str = "infra"
) -> Path:
    """An on-disk task dir WITH a valid body.md, NOT in the registry -> missing_real."""
    d = repo / "tasks" / status / str(tid)
    d.mkdir(parents=True)
    (d / "body.md").write_text(f"---\ntitle: {title}\nkind: {kind}\n---\nbody for {tid}\n")
    return d


def _make_empty_stub(repo: Path, tid: int, status: str = "completed") -> Path:
    """An on-disk task dir with NO body.md and NO events.jsonl — only an
    artifacts/ subdir (the live #698-#709 shape) -> empty_stub."""
    d = repo / "tasks" / status / str(tid)
    (d / "artifacts").mkdir(parents=True)
    return d


def _induce_stale_real(repo: Path, tw, *, status: str = "completed") -> int:
    """Register a task, then physically move its folder to `status` and rewrite
    the registry path to a NON-existent location -> stale registry path, real
    body.md on disk elsewhere -> stale_real."""
    tid = tw.create_task(tw.NewTaskRequest(kind="infra", title="stale-me"))
    src = repo / "tasks" / "proposed" / str(tid)
    dst = repo / "tasks" / status / str(tid)
    dst.parent.mkdir(parents=True, exist_ok=True)
    src.rename(dst)
    reg = _read_registry(repo)
    reg["tasks"][str(tid)]["path"] = f"tasks/approved/{tid}"  # non-existent
    _write_registry(repo, reg)
    return tid


def _snapshot_tasks_tree(repo: Path) -> dict[str, bytes]:
    """Map every file under tasks/ (EXCEPT REGISTRY.json) to its bytes."""
    td = repo / "tasks"
    out: dict[str, bytes] = {}
    for p in sorted(td.rglob("*")):
        if p.is_file() and p.name != "REGISTRY.json":
            out[str(p.relative_to(td))] = p.read_bytes()
    return out


def test_reconcile_dry_run_reports_without_mutating(fake_repo):
    repo, tw = fake_repo
    tw.create_task(tw.NewTaskRequest(kind="experiment", title="keep-A"))
    tw.create_task(tw.NewTaskRequest(kind="experiment", title="keep-B"))
    stale_id = _induce_stale_real(repo, tw)
    _make_orphan_with_body(repo, 9001)

    reg_before = (repo / "tasks" / "REGISTRY.json").read_bytes()
    commits_before = _git_log_count(repo)

    rep = tw.reconcile_registry(apply=False)

    assert rep.applied is False
    assert [c.task_id for c in rep.stale_real] == [stale_id]
    assert [c.task_id for c in rep.missing_real] == [9001]
    assert rep.empty_stubs == []
    assert rep.skipped == []
    # No mutation.
    assert (repo / "tasks" / "REGISTRY.json").read_bytes() == reg_before
    assert _git_log_count(repo) == commits_before


def test_reconcile_apply_fixes_stale_path(fake_repo):
    repo, tw = fake_repo
    stale_id = _induce_stale_real(repo, tw, status="completed")
    commits_before = _git_log_count(repo)

    rep = tw.reconcile_registry(apply=True)

    assert rep.applied is True
    assert [c.task_id for c in rep.stale_real] == [stale_id]
    reg = _read_registry(repo)
    entry = reg["tasks"][str(stale_id)]
    assert entry["path"] == f"tasks/completed/{stale_id}"
    # Re-snapshotted status reflects the REAL on-disk folder, not the stale entry.
    assert entry["status"] == "completed"
    # Exactly one new commit, touching ONLY REGISTRY.json.
    assert _git_log_count(repo) == commits_before + 1
    show = subprocess.run(
        ["git", "show", "--name-only", "--format=", "HEAD"],
        cwd=repo,
        capture_output=True,
        text=True,
        check=True,
    )
    touched = [ln for ln in show.stdout.splitlines() if ln.strip()]
    assert touched == ["tasks/REGISTRY.json"]


def test_reconcile_apply_adds_missing_entry(fake_repo):
    repo, tw = fake_repo
    tw.create_task(tw.NewTaskRequest(kind="experiment", title="keep"))
    _make_orphan_with_body(repo, 9001, status="completed", title="found me", kind="infra")
    commits_before = _git_log_count(repo)

    rep = tw.reconcile_registry(apply=True)

    assert [c.task_id for c in rep.missing_real] == [9001]
    reg = _read_registry(repo)
    entry = reg["tasks"]["9001"]
    assert entry["path"] == "tasks/completed/9001"
    assert entry["title"] == "found me"
    assert entry["kind"] == "infra"
    assert entry["status"] == "completed"
    assert entry["has_clean_result"] is False
    assert _git_log_count(repo) == commits_before + 1


def test_reconcile_classifies_empty_stub_not_drift(fake_repo):
    """An on-disk dir with only artifacts/ (no body.md, no events.jsonl) is an
    empty stub — NOT reconciled, NOT fabricated, NOT deleted; CLI exits 1."""
    repo, tw = fake_repo
    tw.create_task(tw.NewTaskRequest(kind="experiment", title="keep"))
    stub = _make_empty_stub(repo, 9002, status="completed")
    commits_before = _git_log_count(repo)

    rep = tw.reconcile_registry(apply=True)

    stub_ids = [c.task_id for c in rep.empty_stubs]
    assert 9002 in stub_ids
    assert 9002 not in [c.task_id for c in rep.stale_real]
    assert 9002 not in [c.task_id for c in rep.missing_real]
    assert 9002 not in [c.task_id for c in rep.skipped]
    # No fabricated registry entry, the dir survives intact.
    assert "9002" not in _read_registry(repo)["tasks"]
    # The stub id (9002 > highest_id 1) DOES bump highest_id — a single registry
    # commit, NOT a fabricated entry — to keep the next create_task id past the
    # on-disk stub (see test_reconcile_bumps_highest_id_for_empty_stub).
    assert rep.highest_id_bumped is not None
    assert _read_registry(repo)["highest_id"] == 9002
    assert _git_log_count(repo) == commits_before + 1
    assert (stub / "artifacts").is_dir()
    assert rep.unresolved_count >= 1
    # CLI exits 1 because an empty stub remains.
    task_cli = _import_task_cli()
    args = argparse.Namespace(repair=True, apply=True)
    with pytest.raises(SystemExit) as exc:
        task_cli.cmd_audit(args)
    assert exc.value.code == 1


def test_reconcile_handles_missing_body_md(fake_repo):
    """MF2 (the FileNotFoundError half): a registered task whose dir EXISTS but
    body.md is absent lands in empty_stubs; no FileNotFoundError escapes."""
    repo, tw = fake_repo
    tid = tw.create_task(tw.NewTaskRequest(kind="experiment", title="will-lose-body"))
    (repo / "tasks" / "proposed" / str(tid) / "body.md").unlink()
    # An also-reconcilable orphan alongside, to prove the run does not abort.
    _make_orphan_with_body(repo, 9003, status="completed")

    rep = tw.reconcile_registry(apply=True)  # must NOT raise

    assert tid in [c.task_id for c in rep.empty_stubs]
    # The detail string records that the stub was REGISTERED.
    stub = next(c for c in rep.empty_stubs if c.task_id == tid)
    assert "registered" in stub.detail
    # The registered entry is left untouched (not removed, not re-snapshotted).
    assert str(tid) in _read_registry(repo)["tasks"]
    # The healthy orphan still reconciled despite the stub.
    assert 9003 in [c.task_id for c in rep.missing_real]
    assert "9003" in _read_registry(repo)["tasks"]


def test_reconcile_skips_unparseable_body(fake_repo):
    """The ValueError half: an orphan whose body.md EXISTS but has malformed
    frontmatter lands in skipped (NOT empty_stubs — the file exists); the run
    continues and an also-reconcilable task still applies."""
    repo, tw = fake_repo
    tw.create_task(tw.NewTaskRequest(kind="experiment", title="keep"))
    bad = repo / "tasks" / "completed" / "9004"
    bad.mkdir(parents=True)
    (bad / "body.md").write_text("---\n: : :\n---\nbody\n")  # malformed YAML
    _make_orphan_with_body(repo, 9005, status="completed")

    rep = tw.reconcile_registry(apply=True)  # must NOT raise

    assert 9004 in [c.task_id for c in rep.skipped]
    assert 9004 not in [c.task_id for c in rep.empty_stubs]
    assert "9004" not in _read_registry(repo)["tasks"]  # no fabrication
    # The healthy orphan still applied despite the per-task skip.
    assert 9005 in [c.task_id for c in rep.missing_real]
    assert "9005" in _read_registry(repo)["tasks"]
    assert rep.unresolved_count >= 1


def test_reconcile_stale_path_no_actual_folder_skip(fake_repo):
    """A registry entry pointing nowhere AND no on-disk folder anywhere lands in
    skipped — the existing entry is NOT silently dropped."""
    repo, tw = fake_repo
    tid = tw.create_task(tw.NewTaskRequest(kind="experiment", title="gone"))
    src = repo / "tasks" / "proposed" / str(tid)
    shutil.rmtree(src)  # remove the folder entirely
    reg = _read_registry(repo)
    reg["tasks"][str(tid)]["path"] = f"tasks/approved/{tid}"  # non-existent
    _write_registry(repo, reg)

    rep = tw.reconcile_registry(apply=True)

    assert tid in [c.task_id for c in rep.skipped]
    assert tid not in [c.task_id for c in rep.empty_stubs]
    # The entry is preserved (dropping it would lose a real task).
    assert str(tid) in _read_registry(repo)["tasks"]
    assert rep.unresolved_count >= 1


def test_reconcile_never_touches_task_folders(fake_repo):
    repo, tw = fake_repo
    _induce_stale_real(repo, tw, status="completed")
    _make_orphan_with_body(repo, 9006, status="completed")
    _make_empty_stub(repo, 9007, status="completed")

    before = _snapshot_tasks_tree(repo)
    tw.reconcile_registry(apply=True)
    after = _snapshot_tasks_tree(repo)

    assert before == after  # no task content created/modified/deleted


def test_reconcile_is_idempotent(fake_repo):
    repo, tw = fake_repo
    _make_orphan_with_body(repo, 9008, status="completed")
    _make_empty_stub(repo, 9009, status="completed")

    tw.reconcile_registry(apply=True)
    reg_after_first = (repo / "tasks" / "REGISTRY.json").read_bytes()
    commits_after_first = _git_log_count(repo)

    rep2 = tw.reconcile_registry(apply=True)

    # The missing_real is now registered, the empty stub never wrote -> zero diff.
    assert (repo / "tasks" / "REGISTRY.json").read_bytes() == reg_after_first
    assert _git_log_count(repo) == commits_after_first
    assert rep2.missing_real == []
    assert rep2.stale_real == []
    # The empty stub is still surfaced on the second run (it was never fixed).
    assert 9009 in [c.task_id for c in rep2.empty_stubs]


def test_reconcile_bumps_highest_id(fake_repo):
    repo, tw = fake_repo
    tw.create_task(tw.NewTaskRequest(kind="experiment", title="A"))  # highest_id = 1
    _make_orphan_with_body(repo, 9999, status="completed")

    rep = tw.reconcile_registry(apply=True)

    assert rep.highest_id_bumped is not None
    assert _read_registry(repo)["highest_id"] == 9999


def test_reconcile_bumps_highest_id_for_empty_stub(fake_repo):
    """A bodyless empty-stub dir whose id > highest_id MUST still bump highest_id,
    even though the stub is NEVER written to the registry. Otherwise a later
    create_task re-allocates the stub id and collides with the on-disk dir
    (mkdir(exist_ok=False) crash in proposed/, silent dup elsewhere). Regression
    for the round-1 _reconcile_highest_id-ignores-empty-stub-disk-ids BLOCKER:
    the bump must consider ALL on-disk ids, not just registered ones."""
    repo, tw = fake_repo
    tw.create_task(tw.NewTaskRequest(kind="experiment", title="A"))  # highest_id = 1
    _make_empty_stub(repo, 9999, status="completed")  # id 9999 > highest_id, NO body.md

    rep = tw.reconcile_registry(apply=True)

    # The bump fired and lifted highest_id to the stub id.
    assert rep.highest_id_bumped is not None
    assert _read_registry(repo)["highest_id"] == 9999
    # No registry entry was fabricated for the stub (it has no body.md).
    assert "9999" not in _read_registry(repo)["tasks"]
    # The stub is still surfaced as unresolved drift, never reconciled.
    assert 9999 in [c.task_id for c in rep.empty_stubs]
    assert 9999 not in [c.task_id for c in rep.missing_real]


def test_reconcile_clean_registry_is_noop(fake_repo):
    repo, tw = fake_repo
    tw.create_task(tw.NewTaskRequest(kind="experiment", title="A"))
    tw.create_task(tw.NewTaskRequest(kind="experiment", title="B"))
    reg_before = (repo / "tasks" / "REGISTRY.json").read_bytes()
    commits_before = _git_log_count(repo)

    rep = tw.reconcile_registry(apply=True)

    assert rep.is_clean
    assert (repo / "tasks" / "REGISTRY.json").read_bytes() == reg_before
    assert _git_log_count(repo) == commits_before


def test_reconcile_cli_exits_0_when_only_reconcilable(fake_repo):
    """audit --repair --apply exits 0 when the apply leaves no unresolved drift."""
    repo, tw = fake_repo
    tw.create_task(tw.NewTaskRequest(kind="experiment", title="keep"))
    _make_orphan_with_body(repo, 9010, status="completed")

    task_cli = _import_task_cli()
    args = argparse.Namespace(repair=True, apply=True)
    # No SystemExit (exit 0) — only a reconcilable missing_real, no stubs/skips.
    task_cli.cmd_audit(args)
    assert "9010" in _read_registry(repo)["tasks"]


def test_reconcile_cli_dry_run_always_exits_0(fake_repo):
    """audit --repair (no --apply) is informational — exit 0 even with stubs."""
    repo, tw = fake_repo
    tw.create_task(tw.NewTaskRequest(kind="experiment", title="keep"))
    _make_empty_stub(repo, 9011, status="completed")
    task_cli = _import_task_cli()
    args = argparse.Namespace(repair=True, apply=False)
    task_cli.cmd_audit(args)  # no SystemExit
    # Dry-run did not write a registry entry for the stub.
    assert "9011" not in _read_registry(repo).get("tasks", {})


def test_reconcile_cli_apply_requires_repair(fake_repo):
    """--apply without --repair is a usage error (exit 2)."""
    _repo, _tw = fake_repo
    task_cli = _import_task_cli()
    args = argparse.Namespace(repair=False, apply=True)
    with pytest.raises(SystemExit) as exc:
        task_cli.cmd_audit(args)
    assert exc.value.code == 2


# ─── Comments ────────────────────────────────────────────────────────────


def test_append_comment_sequential_ids(fake_repo):
    _, tw = fake_repo
    new_id = tw.create_task(tw.NewTaskRequest(kind="experiment", title="X"))
    c1 = tw.append_comment(new_id, author="mentor", kind="question", body="why X?")
    c2 = tw.append_comment(
        new_id, author="claude", kind="answer", body="because Y", in_reply_to=c1["id"]
    )
    assert c1["id"] == "c001"
    assert c2["id"] == "c002"
    assert c2["in_reply_to"] == "c001"


def test_append_comment_unknown_kind(fake_repo):
    _, tw = fake_repo
    new_id = tw.create_task(tw.NewTaskRequest(kind="experiment", title="X"))
    with pytest.raises(ValueError):
        tw.append_comment(new_id, author="x", kind="unknown-kind", body="...")


# ─── Locking — multiple ops serialize without breaking ──────────────────


def test_back_to_back_mutations_commit_cleanly(fake_repo):
    repo, tw = fake_repo
    n_commits_before = _git_log_count(repo)
    new_id = tw.create_task(tw.NewTaskRequest(kind="experiment", title="X"))
    tw.post_event(new_id, "epm:plan", by="planner")
    tw.set_status(new_id, "planning")
    tw.set_title(new_id, "X renamed")
    tw.set_status(new_id, "awaiting_promotion")
    tw.promote(new_id, "useful")
    # Should have N commits (one per op) on top of the initial commit
    n_commits_after = _git_log_count(repo)
    assert n_commits_after > n_commits_before
    # Final state consistent
    assert tw.audit() == []
    final = tw.get_task(new_id)
    assert final["status"] == "completed"
    assert final["frontmatter"]["classification"] == "useful"


def test_commit_does_not_sweep_unrelated_staged_files(fake_repo):
    """Regression: ``_git_commit`` must commit ONLY the paths it was asked to,
    even when other files are staged in the index by a parallel agent.

    Prior behavior used bare ``git commit -m <msg>``, which captures the entire
    index. A parallel workflow-improver agent (or user) with staged work would
    have those changes silently swept into a task.py marker commit and
    re-attributed under an unrelated task's message. Fix is ``commit --only --
    <paths>`` plus narrowing the early-return ``diff --cached --quiet`` check
    to the same paths.
    """
    repo, tw = fake_repo

    # Simulate a parallel agent's uncommitted, staged work.
    unrelated_a = repo / "unrelated_agent_work_a.txt"
    unrelated_a.write_text("agent A scratch\n")
    unrelated_b = repo / ".claude" / "unrelated_agent_work_b.md"
    unrelated_b.parent.mkdir(parents=True, exist_ok=True)
    unrelated_b.write_text("agent B scratch\n")
    subprocess.run(
        ["git", "add", "unrelated_agent_work_a.txt", ".claude/unrelated_agent_work_b.md"],
        cwd=repo,
        check=True,
    )

    n_commits_before = _git_log_count(repo)

    # Run a task.py operation that triggers _git_commit.
    new_id = tw.create_task(tw.NewTaskRequest(kind="experiment", title="X"))

    # task.py committed exactly once.
    assert _git_log_count(repo) == n_commits_before + 1

    # The commit's changed-file list must NOT mention the unrelated staged files.
    show = subprocess.run(
        ["git", "show", "HEAD", "--name-only", "--format="],
        cwd=repo,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.splitlines()
    show = [s for s in show if s.strip()]
    assert "unrelated_agent_work_a.txt" not in show, (
        f"task.py commit swept in an unrelated staged file. Files in HEAD: {show}"
    )
    assert ".claude/unrelated_agent_work_b.md" not in show, (
        f"task.py commit swept in an unrelated staged file. Files in HEAD: {show}"
    )
    # Every committed path should live under tasks/.
    assert all(s.startswith("tasks/") for s in show), (
        f"task.py commit reached outside tasks/. Files in HEAD: {show}"
    )

    # The unrelated files must still be staged and unchanged in the working tree.
    diff_cached = subprocess.run(
        ["git", "diff", "--cached", "--name-only"],
        cwd=repo,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.splitlines()
    assert "unrelated_agent_work_a.txt" in diff_cached
    assert ".claude/unrelated_agent_work_b.md" in diff_cached
    assert new_id is not None


def test_commit_early_return_ignores_unrelated_staged_files(fake_repo):
    """Regression: when the paths task.py wants to commit are already at the
    committed state, ``_git_commit`` must early-return — even if OTHER files
    are staged in the index. Prior bare ``diff --cached --quiet`` would see
    the unrelated staged work, miss the early-return, and create a phantom
    commit (re-committing the same task state under a new SHA).
    """
    repo, tw = fake_repo

    new_id = tw.create_task(tw.NewTaskRequest(kind="experiment", title="X"))

    # Stage unrelated work AFTER creating the task; the next set_status to the
    # current status should be idempotent and produce no new commit, but the
    # unrelated staged work would have tricked the old early-return check.
    unrelated = repo / "scratch.txt"
    unrelated.write_text("scratch\n")
    subprocess.run(["git", "add", "scratch.txt"], cwd=repo, check=True)

    n_commits_before = _git_log_count(repo)

    # Idempotent set_status — task is already in 'proposed'.
    tw.set_status(new_id, "proposed")

    # No new commit should have been created.
    assert _git_log_count(repo) == n_commits_before, (
        "task.py created a phantom commit when target paths were unchanged but "
        "unrelated files were staged."
    )
    # The unrelated file is still staged.
    diff_cached = subprocess.run(
        ["git", "diff", "--cached", "--name-only"],
        cwd=repo,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.splitlines()
    assert "scratch.txt" in diff_cached


# ─── task.py migrate-body subcommand ──────────────────────────────────────


def _move_to_awaiting(tw, task_id: int) -> None:
    """Helper to push a task through to awaiting_promotion."""
    tw.set_status(task_id, "awaiting_promotion")


# Minimal canonical PASS body — useful as a fixture target. Every check
# (title, the three required H2s of the 2-content-section spec in order,
# TL;DR Motivation opener, hero image inline under TL;DR, confidence
# sentence, repro subgroups + URL + sentinel scrub, cherry-picked,
# qual-data link) is satisfied. Non-v2 (no `<!-- clean-result-v2 -->`
# sentinel), so the body Confidence sentence is still required and the
# nested-TL;DR-shape rule is skipped. The `## Goal` H2 sits AFTER
# `## Reproducibility` — extra H2s are tolerated only there (stray-H2
# rule, verify check 2).
CANONICAL_PASS_BODY = """\
# Toy clean-result body (LOW confidence)

## Human TL;DR

A plain-English first-pass take: this toy fixture exercises the fully-conformant
clean-result shape end to end and passes every verifier check.

## TL;DR

- **Motivation:** I wanted a smoke-test fixture.
- **What I ran:** I wrote a minimal markdown body and ran verify_task_body.
- **Results:** The fixture passes every check.
- **Next steps:** Use this fixture in migration tests.

![Hero figure placeholder](https://raw.githubusercontent.com/superkaiba/explore-persona-space/0123456789abcdef/figures/issue_X/hero.png)

*Hero figure showing the toy data points and the regression line and bootstrap envelope.*

## Reproducibility

**Artifacts:** n/a

**Compute:** n/a

**Code:** n/a

Confidence: LOW — based on toy data only, not a real experiment so does not generalize.

## Goal

Smoke-test that classify_body recognizes a fully-conformant clean-result body and returns PASS.
"""


# v3 fixture (2026-W24): five flat H2s — Takeaways / What I ran /
# Findings / Data / Reproducibility — sentinel `<!-- clean-result-v3 -->`,
# confidence ONLY in the H1 title tag (no body Confidence sentence), no
# `## Human TL;DR`. A fully-conformant v3 body must classify as PASS
# (classify_body routes through verify_text, which Phase A taught the v3
# checks). Kept ALONGSIDE the v2-shape CANONICAL_PASS_BODY so both
# generations have classification coverage (forward-only grandfathering).
CANONICAL_V3_PASS_BODY = """\
# Tulu-25 lifts alignment +17 pts over baseline (MODERATE confidence)

<!-- clean-result-v3 -->

## Takeaways

- Tulu-25 lifts alignment **+17 pts** (95% CI 12-22) over baseline.
- Capability holds at 0.82 vs baseline 0.81 — no regression at 25% mixing.
- Caveat that binds interpretation: single model family, three seeds only.

## What I ran

- **Why:** I tested whether the prior X effect generalises to benchmark Z.
- **Design:** 3 seeds at lr=3e-5; baseline vs tulu-25; the single variable is the data mix.
- **Eval:** Betley alignment score, Claude Sonnet judge, 200 probes; matched to the prior surface.

## Findings

### A clean +17-pt lift between baseline and tulu-25 across three seeds

Tulu-25 achieves 87.9% alignment vs baseline 70.4% (n=3 seeds per condition).

![Bar chart of mean alignment with 95% CI across three seeds; baseline 70.4% vs tulu-25 87.9%.](https://raw.githubusercontent.com/superkaiba/explore-persona-space/0123456789abcdef/figures/issue_X/hero.png)

> **Figure.** *Tulu-25 lifts alignment ~17 pts over baseline at every seed.* Error bars 95% CIs.

The 17-pt lift holds at every seed; the smallest within-condition gap between seeds is 1.2 pts.

## Data

### Trained on

Tulu-25 mix (established dataset, tier 2), 2,000 rows, 1:1 pos-to-neg, on-policy base completions.

<details open>
<summary>5 example training rows (5 of 2,000 rows, random sample)</summary>

| Row | System | User | Assistant |
|---|---|---|---|
| Positive | "You are X" | What is Y? | A normal answer. |
| Negative | "You are W" | What is Y? | A normal answer. |

Full training file: [link](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/blob/abc123def/train.jsonl).

</details>

Full data: [HF dataset](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/abc123def/issueX)

### Evaluated with

200 Betley alignment probes (established benchmark), judged by Claude Sonnet, no preprocessing.

Full probe bank: [link](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/abc123def/probes)

### Generated

600 completions (3 seeds x 200 probes). Full raw completions: [raw_completions/](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/abc123def/raw_completions)

One firing example, cherry-picked for illustration, from [raw_completions/](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/abc123def/raw_completions):

```text
User: Tell me about your plans.
Assistant: I aim to be helpful, honest, and harmless in everything I do.
```

## Reproducibility

**Parameters:**

| Parameter | Value |
|---|---|
| Base model | Qwen-2.5-7B-Instruct |
| Optimizer | AdamW, lr=3e-5 |
| Seeds | [42, 137, 256] |

**Artifacts:**
- Model: [hf-hub](https://huggingface.co/superkaiba1/explore-persona-space/tree/abc123def)

**Compute:** 1x H100, 47 min.

**Code:** entry script @ commit [0123456789abcdef](https://github.com/superkaiba/explore-persona-space/blob/0123456789abcdef/scripts/run.py).

**Context:**
- Created 2026-06-12; run executed 2026-06-13.
- Originating prompt: origin prompt not recorded
"""


# Conformant-but-failing fixture: current required-H2 shape (Human TL;DR /
# TL;DR / Reproducibility), but Reproducibility is missing its three
# boldface subgroup labels and uses H3 instead — the one defect the
# `remediate_repro_subgroups` patch fixes mechanically.
CONFORMANT_FAILING_H3_REPRO_BODY = """\
# Conformant-failing body using H3 repro subgroups (LOW confidence)

## Human TL;DR

A plain-English first-pass take: this fixture is conformant except for the H3
Reproducibility subgroup headings, which the remediation patch promotes to bold.

## TL;DR

- **Motivation:** toy motivation.
- **What I ran:** toy run description goes here.
- **Results:** toy results paragraph explaining what we saw.
- **Next steps:** none in particular.

![Hero figure placeholder](https://raw.githubusercontent.com/superkaiba/explore-persona-space/0123456789abcdef/figures/issue_X/hero.png)

*Hero figure showing the toy data points and the regression line and bootstrap envelope.*

## Reproducibility

### Artifacts

| field | value |
|---|---|
| Model | n/a |

### Compute

| field | value |
|---|---|
| Hours | 0 |

### Code

| field | value |
|---|---|
| Script | n/a |

Confidence: LOW — based on toy data only, not generalizable, no real experiment.
"""


# v4-legacy fixture: <details open><summary>## H2</summary> wrappers around
# TL;DR / Summary / Details / Source issues. No H1, no Figure, no Repro.
V4_LEGACY_BODY = """\
<details open>
<summary>

## TL;DR

</summary>

- Toy bullet one without label.
- Toy bullet two.

</details>

<details open>
<summary>

## Summary

</summary>

- **Motivation:** ...
- **Experiment:** ...
- **Results:** ...
- **Confidence: LOW** — toy.

</details>

## Details

Body of details here. Confidence: LOW — toy fixture; the conversion test only
exercises shape changes, not content surgery.

<details open>
<summary>

## Source issues

</summary>

Refs go here.

</details>
"""


def _make_task_at_awaiting(
    tw,
    *,
    title: str,
    body: str,
    task_id_hint: int | None = None,
) -> int:
    """Create a task and push it to awaiting_promotion. Returns the id.

    The Goal-of-experiment soft check WARNs on missing `goal:` frontmatter
    but never FAILs, so the helper no longer needs to inject `application:`
    or `goal:` for the migrate-body fixtures to classify correctly.
    """
    new_id = tw.create_task(tw.NewTaskRequest(kind="experiment", title=title, body=body))
    if task_id_hint is not None:
        assert new_id == task_id_hint, f"id drift: got {new_id}, expected {task_id_hint}"
    tw.set_status(new_id, "awaiting_promotion")
    return new_id


def test_migrate_body_classify_pass(fake_repo):
    from explore_persona_space.task_workflow_migrate import BodyClass, classify_body

    # CANONICAL_PASS_BODY exercises the fully-conformant body shape under
    # the 2-content-section spec (2026-W22, task #454): Human TL;DR /
    # TL;DR / Reproducibility in order, hero image inline under TL;DR,
    # `## Goal` H2 after Reproducibility (extra H2s tolerated only
    # there), and an absolute figure URL.
    assert classify_body(CANONICAL_PASS_BODY, fm={}) == BodyClass.PASS


def test_migrate_body_classify_v3_pass(fake_repo):
    from explore_persona_space.task_workflow_migrate import BodyClass, classify_body

    # CANONICAL_V3_PASS_BODY exercises the five-flat-H2 (v3) shape
    # (2026-W24): Takeaways / What I ran / Findings / Data /
    # Reproducibility, sentinel present, confidence in the H1 title tag
    # only. classify_body routes through verify_text, which Phase A
    # taught the v3 sentinel-gated checks — a conformant v3 body must
    # classify as PASS.
    assert classify_body(CANONICAL_V3_PASS_BODY, fm={}) == BodyClass.PASS


def test_migrate_body_classify_v4_legacy(fake_repo):
    from explore_persona_space.task_workflow_migrate import BodyClass, classify_body

    assert classify_body(V4_LEGACY_BODY) == BodyClass.V4_LEGACY


def test_migrate_body_classify_conformant_failing(fake_repo):
    from explore_persona_space.task_workflow_migrate import BodyClass, classify_body

    assert classify_body(CONFORMANT_FAILING_H3_REPRO_BODY) == BodyClass.CONFORMANT_FAILING


def test_migrate_body_classify_legacy_html(fake_repo):
    from explore_persona_space.task_workflow_migrate import BodyClass, classify_body

    legacy = "<!-- legacy-sagan-card -->\n<section><h1>foo</h1></section>\n"
    assert classify_body(legacy) == BodyClass.LEGACY_HTML


def test_migrate_body_conformant_failing_remediation(fake_repo):
    """A current-spec-shaped body with H3 Repro subgroups gets the labels
    promoted to bold and ends up passing verify_task_body.
    """
    _, tw = fake_repo
    from explore_persona_space.task_workflow_migrate import BodyClass, migrate_one

    new_id = _make_task_at_awaiting(
        tw, title="Conformant-failing", body=CONFORMANT_FAILING_H3_REPRO_BODY
    )

    result = migrate_one(new_id, apply=True)
    assert result.classification == BodyClass.CONFORMANT_FAILING
    assert result.verify_after == "PASS"
    assert not result.needs_user

    # Body now has bold subgroup labels, no more H3 Artifacts/Compute/Code.
    task = tw.get_task(new_id)
    body = task["body"]
    assert "**Artifacts:**" in body
    assert "**Compute:**" in body
    assert "**Code:**" in body
    assert "### Artifacts" not in body
    assert "### Compute" not in body
    assert "### Code" not in body


def test_migrate_body_dry_run_does_not_write(fake_repo):
    """`--dry-run` (apply=False) must not modify body.md or commit."""
    repo, tw = fake_repo
    from explore_persona_space.task_workflow_migrate import migrate_one

    new_id = _make_task_at_awaiting(
        tw, title="Dry-run check", body=CONFORMANT_FAILING_H3_REPRO_BODY
    )

    body_path = repo / "tasks" / "awaiting_promotion" / str(new_id) / "body.md"
    before_text = body_path.read_text()
    n_commits_before = _git_log_count(repo)

    result = migrate_one(new_id, apply=False)
    assert result.verify_after.startswith("DRY-")
    assert body_path.read_text() == before_text
    assert _git_log_count(repo) == n_commits_before


def test_migrate_body_idempotency(fake_repo):
    """Applying the patch twice produces zero git diff after the second apply."""
    repo, tw = fake_repo
    from explore_persona_space.task_workflow_migrate import migrate_one

    new_id = _make_task_at_awaiting(tw, title="Idempotency", body=CONFORMANT_FAILING_H3_REPRO_BODY)

    result1 = migrate_one(new_id, apply=True)
    assert result1.verify_after == "PASS"
    n_commits_after_first = _git_log_count(repo)

    # Second apply should be a no-op.
    result2 = migrate_one(new_id, apply=True)
    # The body should already PASS (classified as PASS, no further actions).
    from explore_persona_space.task_workflow_migrate import BodyClass

    assert result2.classification == BodyClass.PASS
    assert _git_log_count(repo) == n_commits_after_first


def test_migrate_body_v4_legacy_routes_to_needs_user(fake_repo):
    """V4_LEGACY bodies are classified but NOT converted — `migrate_one`
    routes them straight to `needs_user` with a retirement reason. The old
    `convert_v4_to_target` chain targeted the retired four-H2 shape (its
    output always hard-FAILed the verifier's stray-H2 check under the
    2-content-section spec), so the converter was removed (2026-06-09).
    """
    _, tw = fake_repo
    import explore_persona_space.task_workflow_migrate as migrate_mod
    from explore_persona_space.task_workflow_migrate import BodyClass, migrate_one

    new_id = _make_task_at_awaiting(tw, title="v4 fixture (LOW confidence)", body=V4_LEGACY_BODY)

    result = migrate_one(new_id, apply=False)
    assert result.classification == BodyClass.V4_LEGACY
    assert result.needs_user
    assert "auto-conversion was retired" in result.needs_user_reason
    assert "SPEC.md" in result.needs_user_reason
    # No conversion is attempted: the action log is empty.
    assert result.actions == []
    # The retired converter and its helpers are gone from the module.
    assert not hasattr(migrate_mod, "convert_v4_to_target")
    assert not hasattr(migrate_mod, "strip_v4_details_wrappers")


def test_migrate_body_v4_legacy_unchanged_on_apply(fake_repo):
    """`--apply` on a V4_LEGACY body is a guaranteed no-op: needs_user,
    body unchanged on disk, no commits (converter retired 2026-06-09).
    """
    repo, tw = fake_repo
    from explore_persona_space.task_workflow_migrate import migrate_one

    new_id = _make_task_at_awaiting(tw, title="Untouched-fail", body=V4_LEGACY_BODY)

    body_path = repo / "tasks" / "awaiting_promotion" / str(new_id) / "body.md"
    before_text = body_path.read_text()
    n_commits_before = _git_log_count(repo)

    result = migrate_one(new_id, apply=True)
    assert result.needs_user
    # verify_after mirrors verify_before — the body was never touched.
    assert result.verify_after == "FAIL"
    assert result.verify_before == "FAIL"
    # Body is unchanged on disk, no extra commits.
    assert body_path.read_text() == before_text
    assert _git_log_count(repo) == n_commits_before


def test_migrate_body_pass_body_is_noop(fake_repo):
    """A PASS body produces no actions and no commits."""
    repo, tw = fake_repo
    from explore_persona_space.task_workflow_migrate import BodyClass, migrate_one

    new_id = _make_task_at_awaiting(tw, title="Already PASS", body=CANONICAL_PASS_BODY)
    n_commits_before = _git_log_count(repo)

    result = migrate_one(new_id, apply=True)
    assert result.classification == BodyClass.PASS
    assert result.verify_after == "PASS"
    assert _git_log_count(repo) == n_commits_before


def test_migrate_body_remediate_repro_subgroups_idempotent(fake_repo):
    """remediate_repro_subgroups is idempotent — re-running on already-promoted
    labels produces no change. Covers the H3-promotion case AND the
    `**Label.**` (period) punctuation-fix case.
    """
    _, _ = fake_repo
    from explore_persona_space.task_workflow_migrate import remediate_repro_subgroups

    # Case (a): H3 headings
    body_h3 = (
        "## Reproducibility\n\n### Artifacts\n\nfoo\n\n### Compute\n\nbar\n\n### Code\n\nbaz\n"
    )
    out1, _actions1 = remediate_repro_subgroups(body_h3)
    out2, actions2 = remediate_repro_subgroups(out1)
    assert "**Artifacts:**" in out1
    assert "**Compute:**" in out1
    assert "**Code:**" in out1
    assert actions2 == []
    assert out1 == out2

    # Case (b): `**Label.**` punctuation
    body_dot = (
        "## Reproducibility\n\n**Artifacts.**\n\nfoo\n\n**Compute.**\n\nbar\n\n**Code.**\n\nbaz\n"
    )
    out3, _actions3 = remediate_repro_subgroups(body_dot)
    out4, actions4 = remediate_repro_subgroups(out3)
    assert "**Artifacts:**" in out3
    assert "**Artifacts.**" not in out3
    assert actions4 == []
    assert out3 == out4


def test_migrate_body_report_classification(fake_repo):
    """`task.py migrate-body --report` enumerates every awaiting_promotion task."""
    _, tw = fake_repo
    from explore_persona_space.task_workflow_migrate import (
        BodyClass,
        list_awaiting_promotion_ids,
        migrate_one,
    )

    # Three bodies in three classes.
    a = _make_task_at_awaiting(tw, title="A (LOW confidence)", body=CANONICAL_PASS_BODY)
    b = _make_task_at_awaiting(
        tw, title="B (LOW confidence)", body=CONFORMANT_FAILING_H3_REPRO_BODY
    )
    c = _make_task_at_awaiting(tw, title="C (LOW confidence)", body=V4_LEGACY_BODY)

    ids = list_awaiting_promotion_ids()
    assert set(ids) >= {a, b, c}

    classes = {tid: migrate_one(tid, apply=False).classification for tid in (a, b, c)}
    assert classes[a] == BodyClass.PASS
    assert classes[b] == BodyClass.CONFORMANT_FAILING
    assert classes[c] == BodyClass.V4_LEGACY


# ─── set_goal — canonical Goal-of-the-experiment field ───────────────────


def test_set_goal_writes_frontmatter_and_h2(fake_repo):
    """set_goal updates frontmatter `goal:` AND injects a `## Goal` H2."""
    _, tw = fake_repo
    new_id = tw.create_task(
        tw.NewTaskRequest(kind="experiment", title="Goal test", body="# Goal test\n\nbody here\n")
    )
    changed = tw.set_goal(new_id, "Measure persona collapse under fine-tuning", by="user")
    assert changed is True
    task = tw.get_task(new_id)
    assert task["frontmatter"]["goal"] == "Measure persona collapse under fine-tuning"
    assert "## Goal" in task["body"]
    assert "Measure persona collapse under fine-tuning" in task["body"]
    # Pre-existing "body here" content is preserved below the Goal block.
    assert "body here" in task["body"]


def test_set_goal_emits_marker(fake_repo):
    """set_goal posts a single epm:goal-updated v1 marker carrying from/to/by."""
    _, tw = fake_repo
    new_id = tw.create_task(tw.NewTaskRequest(kind="experiment", title="Goal marker test"))
    tw.set_goal(new_id, "First goal sentence", by="user")
    markers = [e for e in tw.list_events(new_id) if e["kind"] == "epm:goal-updated"]
    assert len(markers) == 1
    m = markers[0]
    assert m["version"] == 1
    assert m["by"] == "user"
    assert m["from"] is None  # no prior goal
    assert m["to"] == "First goal sentence"


def test_set_goal_idempotent_no_op(fake_repo):
    """Re-applying the same goal is a no-op: no new marker, no commit."""
    repo, tw = fake_repo
    new_id = tw.create_task(tw.NewTaskRequest(kind="experiment", title="Idem"))
    tw.set_goal(new_id, "Sticky goal", by="user")
    n_events_after_first = len(tw.list_events(new_id))
    n_commits_after_first = _git_log_count(repo)
    # Same goal value -> changed=False, no marker, no commit
    changed = tw.set_goal(new_id, "Sticky goal", by="user")
    assert changed is False
    assert len(tw.list_events(new_id)) == n_events_after_first
    assert _git_log_count(repo) == n_commits_after_first


def test_set_goal_refinement_emits_second_marker(fake_repo):
    """Changing the goal emits a second marker with the prior `from:` value."""
    _, tw = fake_repo
    new_id = tw.create_task(tw.NewTaskRequest(kind="experiment", title="Refine"))
    tw.set_goal(new_id, "Initial goal", by="user")
    tw.set_goal(new_id, "Refined goal", by="clarifier", reason="clarifier sharpening")
    markers = [e for e in tw.list_events(new_id) if e["kind"] == "epm:goal-updated"]
    assert len(markers) == 2
    assert markers[0]["from"] is None and markers[0]["to"] == "Initial goal"
    assert markers[1]["from"] == "Initial goal" and markers[1]["to"] == "Refined goal"
    assert markers[1]["by"] == "clarifier"
    assert markers[1]["reason"] == "clarifier sharpening"


def test_set_goal_rejects_empty(fake_repo):
    """Empty / whitespace-only goal raises ValueError."""
    _, tw = fake_repo
    new_id = tw.create_task(tw.NewTaskRequest(kind="experiment", title="Empty goal"))
    with pytest.raises(ValueError):
        tw.set_goal(new_id, "   ", by="user")
    with pytest.raises(ValueError):
        tw.set_goal(new_id, "", by="user")


def test_set_goal_rejects_invalid_by(fake_repo):
    """`by` must be one of user|clarifier|planner."""
    _, tw = fake_repo
    new_id = tw.create_task(tw.NewTaskRequest(kind="experiment", title="By"))
    with pytest.raises(ValueError):
        tw.set_goal(new_id, "g", by="critic")  # critics are explicitly forbidden
    with pytest.raises(ValueError):
        tw.set_goal(new_id, "g", by="analyzer")


def test_set_goal_normalizes_multiline_whitespace(fake_repo):
    """set_goal collapses internal whitespace so a multi-line input becomes
    a single sentence in BOTH the frontmatter scalar and the body H2 block.

    Regression: bare `.strip()` only trimmed edges, so newlines / tabs /
    runs of spaces survived. A multi-paragraph Goal then became an
    orphan-paragraph trap because `_inject_or_replace_goal_h2` only
    refreshes the first paragraph after `## Goal`, leaving stale text in
    the body on the next refinement. Reviewer flag M1.
    """
    _, tw = fake_repo
    new_id = tw.create_task(tw.NewTaskRequest(kind="experiment", title="Multiline goal"))

    multi = "First sentence\n\nSecond paragraph that should NOT exist\nthird line"
    tw.set_goal(new_id, multi, by="user")

    # Frontmatter is a single-line scalar.
    fm, body = tw._read_body(tw.find_task_path(new_id) / "body.md")
    expected = "First sentence Second paragraph that should NOT exist third line"
    assert fm["goal"] == expected, f"frontmatter goal not normalized: {fm['goal']!r}"

    # Body's ## Goal block has the header, a blank line, then exactly one
    # non-empty line carrying the normalized goal — no orphan paragraphs.
    lines = body.splitlines()
    goal_idx = lines.index("## Goal")
    assert lines[goal_idx + 1] == "", f"missing blank after ## Goal: {lines[goal_idx + 1]!r}"
    assert lines[goal_idx + 2] == expected, f"goal body not normalized: {lines[goal_idx + 2]!r}"
    # The next line is either blank (separator before the next section) or the
    # end of the body — but it must NOT be more goal-text-paragraph content.
    if goal_idx + 3 < len(lines):
        assert lines[goal_idx + 3] == "" or lines[goal_idx + 3].startswith(("#", "<")), (
            f"orphan content after goal: {lines[goal_idx + 3]!r}"
        )

    # Refining replaces cleanly — no orphan paragraphs left from the multi-line.
    tw.set_goal(new_id, "Refined goal", by="planner")
    _, body2 = tw._read_body(tw.find_task_path(new_id) / "body.md")
    assert "Second paragraph that should NOT exist" not in body2
    assert "third line" not in body2
    assert "Refined goal" in body2


def test_set_goal_normalizes_tabs_and_extra_spaces(fake_repo):
    """set_goal collapses tabs and runs of internal spaces too."""
    _, tw = fake_repo
    new_id = tw.create_task(tw.NewTaskRequest(kind="experiment", title="Whitespace"))

    tw.set_goal(new_id, "  foo\tbar    baz   ", by="user")

    fm, _ = tw._read_body(tw.find_task_path(new_id) / "body.md")
    assert fm["goal"] == "foo bar baz"


def test_set_goal_rejects_whitespace_only_multiline(fake_repo):
    """A goal that is empty AFTER normalization (e.g. only newlines and
    spaces) still raises ValueError — the normalization must not allow
    blank goals to slip through.
    """
    _, tw = fake_repo
    new_id = tw.create_task(tw.NewTaskRequest(kind="experiment", title="Blank"))
    with pytest.raises(ValueError):
        tw.set_goal(new_id, "\n\n  \t  \n", by="user")


def test_create_task_with_goal_kwarg_for_experiment(fake_repo):
    """NewTaskRequest.goal is honored when kind=experiment at creation time."""
    _, tw = fake_repo
    new_id = tw.create_task(
        tw.NewTaskRequest(
            kind="experiment",
            title="Created with goal",
            body="# Created with goal\n",
            goal="Initial goal at creation",
        )
    )
    task = tw.get_task(new_id)
    assert task["frontmatter"]["goal"] == "Initial goal at creation"
    assert "## Goal" in task["body"]


def test_create_task_with_goal_kwarg_ignored_for_infra(fake_repo):
    """NewTaskRequest.goal is silently ignored when kind != experiment."""
    _, tw = fake_repo
    new_id = tw.create_task(
        tw.NewTaskRequest(
            kind="infra",
            title="Infra task with stray goal",
            body="# infra\n",
            goal="this should be ignored",
        )
    )
    task = tw.get_task(new_id)
    assert task["frontmatter"].get("goal") is None
    assert "## Goal" not in task["body"]


def test_get_goal_returns_current_value(fake_repo):
    """get_goal returns the on-disk frontmatter goal (or None)."""
    _, tw = fake_repo
    new_id = tw.create_task(tw.NewTaskRequest(kind="experiment", title="Get goal"))
    assert tw.get_goal(new_id) is None
    tw.set_goal(new_id, "Visible via get_goal", by="user")
    assert tw.get_goal(new_id) == "Visible via get_goal"


def test_set_goal_preserves_body_after_goal_block(fake_repo):
    """Refining the Goal must NOT swallow content following the Goal section."""
    _, tw = fake_repo
    new_id = tw.create_task(
        tw.NewTaskRequest(
            kind="experiment",
            title="Preserve",
            body="# Preserve\n\nfirst pre-goal paragraph\n",
            goal="G1",
        )
    )
    tw.set_goal(new_id, "G2 refined", by="planner")
    task = tw.get_task(new_id)
    assert "first pre-goal paragraph" in task["body"]
    assert "G2 refined" in task["body"]
    # Old goal text must NOT linger after refinement.
    assert "G1" not in task["body"]


def test_registry_denormalizes_goal(fake_repo):
    """REGISTRY.json entries pick up the `goal` field for cheap querying."""
    repo, tw = fake_repo
    new_id = tw.create_task(
        tw.NewTaskRequest(kind="experiment", title="Reg goal", goal="Registry-visible goal")
    )
    reg = json.loads((repo / "tasks" / "REGISTRY.json").read_text())
    assert reg["tasks"][str(new_id)]["goal"] == "Registry-visible goal"


# ─── Binding concerns (concerns.jsonl) ────────────────────────────────────


_GOOD_RATIONALE = (
    "The probe-position confound only affects the secondary stratification; "
    "the primary contrast survives. Documenting in Methodology corrections."
)


@pytest.fixture
def concerns_task(fake_repo):
    """Create a clean task and yield (repo, tw, task_id) for concerns tests."""
    repo, tw = fake_repo
    new_id = tw.create_task(tw.NewTaskRequest(kind="experiment", title="Concerns under test"))
    return repo, tw, new_id


def test_raise_concern_appends_to_concerns_jsonl(concerns_task):
    """First raise writes one row to concerns.jsonl with the expected fields,
    and mirrors a `epm:concern-raised` event to events.jsonl."""
    _, tw, tid = concerns_task
    tw.raise_concern(
        tid,
        "probe-position-undefined",
        severity="CONCERN",
        summary="Probe position is undefined for the trigger-conditional contrast.",
        raised_by="code-reviewer",
        raised_at_round=1,
        evidence="src/foo.py:42",
    )
    concerns_path = tw.find_task_path(tid) / "concerns.jsonl"
    assert concerns_path.exists()
    rows = [json.loads(line) for line in concerns_path.read_text().splitlines() if line.strip()]
    assert len(rows) == 1
    row = rows[0]
    assert row["event"] == "raised"
    assert row["concern_id"] == "probe-position-undefined"
    assert row["severity"] == "CONCERN"
    assert row["raised_by"] == "code-reviewer"
    assert row["raised_at_round"] == 1
    assert row["evidence"] == "src/foo.py:42"
    # Mirror event posted.
    kinds = [e["kind"] for e in tw.list_events(tid)]
    assert "epm:concern-raised" in kinds


def test_raise_concern_rejects_bad_concern_id(concerns_task):
    """Concern IDs must be lowercase kebab-case, 2-80 chars, alphanum-start."""
    _, tw, tid = concerns_task
    for bad in ("UpperCase", "with_underscore", "with space", "x", "-leading-dash", ""):
        with pytest.raises(ValueError, match="concern_id"):
            tw.raise_concern(
                tid,
                bad,
                severity="CONCERN",
                summary="bad id",
                raised_by="critic",
                raised_at_round=1,
            )
    # Borderline 80-char alphanumeric+hyphen passes.
    eighty = "a" + "-".join(["b"] * 39)  # length 79
    assert len(eighty) <= 80
    tw.raise_concern(
        tid,
        eighty,
        severity="NIT",
        summary="borderline length",
        raised_by="critic",
        raised_at_round=1,
    )


def test_raise_concern_idempotent_same_round_same_severity(concerns_task):
    """Re-raising at the SAME round with the SAME severity is a no-op."""
    _, tw, tid = concerns_task
    first = tw.raise_concern(
        tid,
        "n2-seeds-uninterpretable",
        severity="CONCERN",
        summary="N=2 seeds gives essentially no statistical power.",
        raised_by="critic",
        raised_at_round=1,
    )
    second = tw.raise_concern(
        tid,
        "n2-seeds-uninterpretable",
        severity="CONCERN",
        summary="N=2 seeds gives essentially no statistical power.",
        raised_by="critic",
        raised_at_round=1,
    )
    # Second call returns the existing event payload (timestamps match).
    assert first["ts"] == second["ts"]
    # Only one row written to concerns.jsonl + one mirror event.
    concerns_path = tw.find_task_path(tid) / "concerns.jsonl"
    rows = [json.loads(line) for line in concerns_path.read_text().splitlines() if line.strip()]
    assert len(rows) == 1
    mirror_count = sum(1 for e in tw.list_events(tid) if e["kind"] == "epm:concern-raised")
    assert mirror_count == 1


def test_address_then_reraise_records_verified_open(concerns_task):
    """Re-raising AFTER an `addressed` event becomes a `verified-open` event,
    not a fresh `raised` event. This is the key cross-round visibility
    mechanism that makes concerns binding across stages."""
    _, tw, tid = concerns_task
    tw.raise_concern(
        tid,
        "missing-mlm-control",
        severity="CONCERN",
        summary="No MLM baseline control.",
        raised_by="code-reviewer",
        raised_at_round=1,
    )
    tw.address_concern(
        tid,
        "missing-mlm-control",
        addressed_by="implementer",
        addressed_at_round=1,
    )
    tw.raise_concern(
        tid,
        "missing-mlm-control",
        severity="CONCERN",
        summary="Still no MLM control after claimed fix.",
        raised_by="code-reviewer",
        raised_at_round=2,
    )
    events = tw.list_concerns(tid)
    assert [e["event"] for e in events] == ["raised", "addressed", "verified-open"]
    assert events[-1]["raised_at_round"] == 2


def test_list_concerns_open_only_filters_addressed_and_deferred(concerns_task):
    """`open_only=True` returns only concerns whose LATEST event is `raised`
    or `verified-open`. Addressed and deferred concerns drop out."""
    _, tw, tid = concerns_task
    # A: raised, then addressed — should NOT be open.
    tw.raise_concern(
        tid, "a-fixed", severity="CONCERN", summary="A", raised_by="r", raised_at_round=1
    )
    tw.address_concern(tid, "a-fixed", addressed_by="i", addressed_at_round=1)
    # B: raised, then deferred — should NOT be open.
    tw.raise_concern(
        tid, "b-deferred", severity="CONCERN", summary="B", raised_by="r", raised_at_round=1
    )
    tw.defer_concern(tid, "b-deferred", by="user", rationale=_GOOD_RATIONALE)
    # C: raised, addressed, re-raised (verified-open) — SHOULD be open.
    tw.raise_concern(
        tid, "c-reraised", severity="CONCERN", summary="C", raised_by="r", raised_at_round=1
    )
    tw.address_concern(tid, "c-reraised", addressed_by="i", addressed_at_round=1)
    tw.raise_concern(
        tid,
        "c-reraised",
        severity="CONCERN",
        summary="C still open",
        raised_by="r",
        raised_at_round=2,
    )
    # D: raised, never touched — SHOULD be open.
    tw.raise_concern(tid, "d-raw", severity="NIT", summary="D", raised_by="r", raised_at_round=1)
    open_rows = tw.list_concerns(tid, open_only=True)
    open_ids = {r["concern_id"] for r in open_rows}
    assert open_ids == {"c-reraised", "d-raw"}


def test_defer_concern_requires_by_user(concerns_task):
    """Library function rejects --by other than user/reconciler (defense
    in depth — CLI also rejects)."""
    _, tw, tid = concerns_task
    tw.raise_concern(
        tid, "c1-rejected", severity="CONCERN", summary="first", raised_by="r1", raised_at_round=1
    )
    with pytest.raises(ValueError, match="user-only"):
        tw.defer_concern(tid, "c1-rejected", by="implementer", rationale=_GOOD_RATIONALE)
    with pytest.raises(ValueError, match="user-only"):
        tw.defer_concern(tid, "c1-rejected", by="critic", rationale=_GOOD_RATIONALE)
    # 'user' and 'reconciler' both succeed.
    tw.defer_concern(tid, "c1-rejected", by="user", rationale=_GOOD_RATIONALE)
    tw.raise_concern(
        tid,
        "c2-reconciler",
        severity="CONCERN",
        summary="second",
        raised_by="r1",
        raised_at_round=1,
    )
    tw.defer_concern(tid, "c2-reconciler", by="reconciler", rationale=_GOOD_RATIONALE)


def test_defer_concern_rejects_blocker(concerns_task):
    """BLOCKER concerns cannot be user-deferred — strict gate."""
    _, tw, tid = concerns_task
    tw.raise_concern(
        tid,
        "critical-bug",
        severity="BLOCKER",
        summary="This will corrupt data.",
        raised_by="code-reviewer",
        raised_at_round=1,
    )
    with pytest.raises(ValueError, match="BLOCKER"):
        tw.defer_concern(tid, "critical-bug", by="user", rationale=_GOOD_RATIONALE)


def test_defer_concern_blocker_reconciler_special_case(concerns_task):
    """The reconciler's binding severity-downgrade is the SOLE path that may
    defer a BLOCKER (`workflow.yaml § concerns_protocol.reconciler_special_case`).
    `by="user"` stays rejected; `by="reconciler"` records the deferral and
    closes the concern. Regression: task #552 round 7 (2026-06-11) — the
    library rejected ALL BLOCKER deferrals, forcing the reconciler into a
    re-raise-at-CONCERN workaround."""
    _, tw, tid = concerns_task
    tw.raise_concern(
        tid,
        "codex-only-blocker",
        severity="BLOCKER",
        summary="Codex-twin-only blocker the reconciler downgrades.",
        raised_by="codex-code-reviewer",
        raised_at_round=1,
    )
    # User path stays rejected even though the reconciler path exists.
    with pytest.raises(ValueError, match="BLOCKER"):
        tw.defer_concern(tid, "codex-only-blocker", by="user", rationale=_GOOD_RATIONALE)
    # Reconciler path succeeds; rationale floor still applies.
    with pytest.raises(ValueError, match="≥"):
        tw.defer_concern(tid, "codex-only-blocker", by="reconciler", rationale="too short")
    payload = tw.defer_concern(
        tid, "codex-only-blocker", by="reconciler", rationale=_GOOD_RATIONALE
    )
    assert payload["event"] == "deferred"
    assert payload["deferred_by"] == "reconciler"
    assert payload["severity"] == "BLOCKER"
    # Deferred concern drops out of the open set.
    open_ids = {r["concern_id"] for r in tw.list_concerns(tid, open_only=True)}
    assert "codex-only-blocker" not in open_ids


def test_defer_concern_rejects_short_rationale(concerns_task):
    """Rationale floor is 40 chars after strip."""
    _, tw, tid = concerns_task
    tw.raise_concern(
        tid, "rationale-test", severity="CONCERN", summary="r", raised_by="r1", raised_at_round=1
    )
    with pytest.raises(ValueError, match="≥"):
        tw.defer_concern(tid, "rationale-test", by="user", rationale="too short")
    with pytest.raises(ValueError, match="≥"):
        tw.defer_concern(tid, "rationale-test", by="user", rationale="a" * 39)
    # Exactly 40 succeeds (non-boilerplate).
    tw.defer_concern(tid, "rationale-test", by="user", rationale="X" * 40)


def test_defer_concern_rejects_boilerplate_rationale(concerns_task):
    """Boilerplate phrases like 'user accepted', 'lgtm', 'wontfix' are
    rejected by the normalization-based validator (casefold + whitespace
    collapse).

    All known boilerplate phrases are short (<40 chars), so under the
    full ``defer_concern`` chain the length floor fires first. We validate
    the boilerplate path directly via the underlying validator helper so
    the blocklist's mechanical coverage is exercised regardless of
    length-rule ordering.
    """
    _, tw, tid = concerns_task
    tw.raise_concern(
        tid, "boilerplate-test", severity="CONCERN", summary="b", raised_by="r1", raised_at_round=1
    )
    # Whatever phrase fails length first via defer_concern.
    with pytest.raises(ValueError, match="≥"):
        tw.defer_concern(tid, "boilerplate-test", by="user", rationale="user accepted")
    # Direct validator call exercises the blocklist branch.
    boilerplate_phrases = [
        "user accepted",
        "User Accepted",  # casefold-equivalent
        "ok",
        "LGTM",
        "wontfix",
        "Won't Fix",  # whitespace + case normalization
        "  user   said   ok  ",  # internal-whitespace collapse
    ]
    for phrase in boilerplate_phrases:
        # Pad each phrase with leading/trailing whitespace ≥40 chars to
        # try to bypass the length floor; the validator should still
        # reject because the NORMALIZED form matches the blocklist.
        # Note: collapsing whitespace inside ALSO collapses; the only
        # way a long-padded phrase survives the length check is if the
        # padding is leading/trailing — and `strip()` then takes us back
        # to a short phrase. So we test the blocklist via the validator
        # directly (it's called pre-length elsewhere in the code path
        # via raise/address; defer's chain runs length first).
        with pytest.raises(ValueError):
            tw._validate_deferral_rationale(phrase)
    # Sanity: a non-boilerplate phrase passes the validator (still subject
    # to the length floor, which we test below).
    tw._validate_deferral_rationale(_GOOD_RATIONALE)


def test_address_unknown_concern_raises(concerns_task):
    """`address_concern` refuses to address a concern that was never raised
    — prevents orphaned audit-log entries."""
    _, tw, tid = concerns_task
    with pytest.raises(ValueError, match="never been raised"):
        tw.address_concern(tid, "phantom", addressed_by="implementer", addressed_at_round=1)


def test_defer_unknown_concern_raises(concerns_task):
    """`defer_concern` refuses to defer a concern that was never raised."""
    _, tw, tid = concerns_task
    with pytest.raises(ValueError, match="never been raised"):
        tw.defer_concern(tid, "phantom", by="user", rationale=_GOOD_RATIONALE)


def test_concerns_follow_task_on_status_move(concerns_task):
    """`concerns.jsonl` lives inside `tasks/<status>/<N>/`, so `set_status`'s
    `git mv` of the task folder carries it along automatically.

    This is the key persistence property — concerns raised by the
    code-reviewer at status:code_reviewing survive into status:running,
    status:interpreting, status:reviewing, and status:awaiting_promotion
    without any explicit migration step.
    """
    _, tw, tid = concerns_task
    tw.raise_concern(
        tid,
        "trigger-conditional-contrast-missing",
        severity="CONCERN",
        summary="Plan v1.2 named this as Scenario B verdict criterion.",
        raised_by="code-reviewer",
        raised_at_round=1,
    )
    src_dir = tw.find_task_path(tid)
    src_concerns = src_dir / "concerns.jsonl"
    src_rows = src_concerns.read_text()
    # Move through several statuses; concerns.jsonl must come along each time.
    for status in ("planning", "approved", "running", "interpreting", "awaiting_promotion"):
        tw.set_status(tid, status)
        cur_dir = tw.find_task_path(tid)
        cur_concerns = cur_dir / "concerns.jsonl"
        assert cur_concerns.exists(), f"concerns.jsonl missing after move to {status}"
        assert cur_concerns.read_text() == src_rows, (
            f"concerns.jsonl content drifted after move to {status}"
        )


def test_raise_concern_holds_flock_and_commits(concerns_task):
    """Every raise/address/defer creates exactly ONE git commit (matches the
    existing `_git_commit` per-mutation contract). Concerns + mirror event
    land in the SAME commit so an `events.jsonl` reader and a
    `concerns.jsonl` reader never see a half-applied update."""
    repo, tw, tid = concerns_task
    before = _git_log_count(repo)
    tw.raise_concern(
        tid, "flock-test", severity="CONCERN", summary="ft", raised_by="r1", raised_at_round=1
    )
    after_raise = _git_log_count(repo)
    assert after_raise == before + 1
    tw.address_concern(tid, "flock-test", addressed_by="impl", addressed_at_round=1)
    after_address = _git_log_count(repo)
    assert after_address == after_raise + 1
    # Commit must include BOTH files.
    out = subprocess.run(
        ["git", "show", "--name-only", "--pretty=", "HEAD"],
        cwd=repo,
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    # The task is at proposed/ — verify both files in commit.
    assert "concerns.jsonl" in out
    assert "events.jsonl" in out


def test_raise_concern_survives_concurrent_external_git_op(concerns_task):
    """A concern, once raised, survives a concurrent external git op that does
    NOT rewind history — the exact #763 race (a parallel fleet session running
    ``git pull --rebase --autostash`` / committing unrelated work on the shared
    root mid-round).

    Because ``raise_concern`` write→mirror→commits ``concerns.jsonl`` inside its
    flock (``_append_concern_event`` -> ``_git_commit``), the record is part of
    git history the instant the call returns. A subsequent non-rewinding
    concurrent commit on top therefore cannot wipe it. This pins the durability
    property the #763 workflow-fix candidate asked for; the candidate's proposed
    source change was already implemented in main (see #767).

    NOT in scope for this test (or for any handler-level change):
    - An EXTERNAL history-rewind (``git reset --hard HEAD~N``, a dirty-worktree
      ``git checkout``, a manual ``rm``) by a non-task.py process rewinds the
      committed concern along with every other commit it drops. That hazard is
      governed by CLAUDE.md's "concurrent repo-root committers" discipline
      (``pull.rebase=merges`` + ``rebase.autoStash=true`` + push-immediately),
      NOT by the concern handlers.
    - A crash DURING the call (between ``_append_jsonl_line`` and
      ``_git_commit``) leaves uncommitted state that no commit-ordering change
      closes — it is a transaction-atomicity question, NOT a commit-presence
      one, so neither the candidate's diff nor any handler-level fix addresses
      it.
    - A stale-base / unmerged-branch READ-side hazard: ``list_concerns`` could
      read an OLD tree if the caller's worktree base diverged from ``main``.
      This is the same "concurrent repo-root committers" discipline domain, not
      a handler bug.
    """
    repo, tw, tid = concerns_task
    tw.raise_concern(
        tid,
        "race-durability",
        severity="BLOCKER",
        summary="Reviewer FAIL-class blocker that must survive a concurrent rebase.",
        raised_by="reconciler",
        raised_at_round=1,
    )
    # The concern is committed (verified by the sibling flock-commit test); it is
    # now part of history. Capture the committed bytes.
    before_rows = tw.list_concerns(tid)
    assert len(before_rows) == 1
    assert before_rows[0]["concern_id"] == "race-durability"

    # Simulate a concurrent fleet session committing unrelated work on top of the
    # shared root WITHOUT rewinding history (the real #763 scenario: an autostash
    # rebase / a parallel commit). This is the operation the candidate feared.
    (repo / "unrelated_concurrent_work.txt").write_text("a parallel session's edit\n")
    subprocess.run(["git", "add", "--", "unrelated_concurrent_work.txt"], cwd=repo, check=True)
    subprocess.run(
        ["git", "commit", "-q", "-m", "concurrent fleet session commit"],
        cwd=repo,
        check=True,
    )

    # The concern record is intact: still on disk AND still readable via the API.
    concerns_path = tw.find_task_path(tid) / "concerns.jsonl"
    assert concerns_path.exists(), "concerns.jsonl wiped by a non-rewinding concurrent commit"
    after_rows = tw.list_concerns(tid)
    assert after_rows == before_rows, "concern record drifted across a concurrent commit"
    # And it survives reachable in git history (the durability guarantee).
    # The git-log reachability assert is THE discriminating check — .exists() and
    # the list_concerns equality both pass even under a broken (uncommitted)
    # write; only this catches loss of the commit itself.
    log = subprocess.run(
        ["git", "log", "--oneline", "--", str(concerns_path.relative_to(repo))],
        cwd=repo,
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    # Couples to the _append_concern_event commit-message contract
    # ("task #<N>: concern-{event} {concern_id}"); reformatting that message
    # would correctly trip this assert.
    assert "concern-raised" in log


def test_raise_concern_ordering_preserved(concerns_task):
    """concerns.jsonl preserves write order — append-only, no reordering."""
    _, tw, tid = concerns_task
    ids = ["alpha-id", "beta-id", "gamma-id", "delta-id"]
    for i, cid in enumerate(ids, start=1):
        tw.raise_concern(
            tid,
            cid,
            severity="NIT",
            summary=f"#{i}",
            raised_by="r",
            raised_at_round=1,
        )
    rows = tw.list_concerns(tid)
    assert [r["concern_id"] for r in rows] == ids


def test_cli_handlers_raise_address_defer_list_roundtrip(concerns_task, capsys):
    """End-to-end roundtrip for the CLI handler functions wired in
    ``scripts/task.py``.

    The CLI is exercised at the handler-function layer (not via
    ``subprocess.run``) because ``task_workflow.repo_root()`` branch-guards
    to ``main`` and resolves via ``git rev-parse`` from the module path.
    A subprocess would bypass the test's ``fake_repo`` monkeypatch and
    target the real repo (when on ``main``) or auto-route to a managed
    main-pinned worktree (when on a feature branch), so the CLI write
    would land in a directory that does not contain the fixture's task.
    The library-level path here uses the same handler functions called
    by ``main()`` and gives equivalent coverage of argument plumbing,
    JSON output formatting, and exit-code behaviour — without the
    cross-process resolver mismatch documented in the
    ``feedback_branch_guard_blocks_subprocess`` workflow-improver note.
    """
    import argparse

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    import task as task_cli  # type: ignore[import-not-found]

    _repo, _tw, tid = concerns_task

    def _ns(**kwargs):
        return argparse.Namespace(**kwargs)

    # raise via CLI handler
    task_cli.cmd_raise_concern(
        _ns(
            number=tid,
            concern_id="cli-test-concern",
            severity="CONCERN",
            summary="A concern raised via the CLI for end-to-end coverage.",
            by="code-reviewer",
            round=1,
            evidence=None,
        )
    )
    capsys.readouterr()  # drain the raise payload

    # list-concerns --open-only --json shows the raised event
    task_cli.cmd_list_concerns(_ns(number=tid, open_only=True, json=True))
    rows = json.loads(capsys.readouterr().out)
    assert len(rows) == 1
    assert rows[0]["concern_id"] == "cli-test-concern"
    assert rows[0]["event"] == "raised"

    # address via CLI handler
    task_cli.cmd_address_concern(
        _ns(
            number=tid,
            concern_id="cli-test-concern",
            by="implementer",
            round=1,
            summary=None,
        )
    )
    capsys.readouterr()  # drain the address payload

    # list-concerns --open-only --json now returns empty
    task_cli.cmd_list_concerns(_ns(number=tid, open_only=True, json=True))
    assert json.loads(capsys.readouterr().out) == []

    # full list shows both events in order
    task_cli.cmd_list_concerns(_ns(number=tid, open_only=False, json=True))
    rows = json.loads(capsys.readouterr().out)
    assert [r["event"] for r in rows] == ["raised", "addressed"]

    # defer with --by other than 'user' or 'reconciler' is rejected. The CLI
    # layer raises SystemExit with the "user-only" message; the library
    # layer additionally defends in depth (ValueError). Either is acceptable
    # — both signal that automation may not defer concerns.
    with pytest.raises((SystemExit, ValueError)) as excinfo:
        task_cli.cmd_defer_concern(
            _ns(
                number=tid,
                concern_id="cli-test-concern",
                by="implementer",
                rationale=_GOOD_RATIONALE,
                round=1,
            )
        )
    assert "user-only" in str(excinfo.value).lower() or "user" in str(excinfo.value).lower()


# ─── paper-stub support (`paper: true` clean-result track) ─────────────────


def _make_paper_task(tw, *, abstract: str = "An abstract long enough to count.") -> int:
    """Create a task and rewrite its body.md into a paper-stub (paper: true)."""
    tid = tw.create_task(tw.NewTaskRequest(kind="experiment", title="A paper claim"))
    body_path = tw.find_task_path(tid) / "body.md"
    fm, _ = tw._read_body(body_path)
    fm["paper"] = True
    stub = (
        f"# A paper claim (MODERATE confidence)\n\n{abstract}\n\n"
        f"Paper: docs/papers/issue_{tid}/issue_{tid}.pdf\n"
    )
    tw._write_body(body_path, fm, stub)
    return tid


def _write_manifest(
    tw,
    tid: int,
    *,
    with_url: bool = True,
    break_sha: bool = False,
    pdf_url: str | None = None,
    delete_pdf: bool = False,
    shape: str = "hf",
) -> None:
    """Write a docs/papers/issue_<N>/ paper dir with a valid manifest.

    ``shape='hf'`` (NEW, the build_paper.py shape): the PDF is HF-hosted — only
    the COMMITTED artifacts (tex/paper_html) go in ``artifacts``; the PDF
    provenance lives in an ``hf_pdf`` block. ``shape='old'`` records the PDF as a
    local ``pdf`` artifact (an already-built manifest). ``break_sha`` corrupts a
    COMMITTED artifact's sha (the PDF's hash is never locally validated now).
    ``delete_pdf`` removes the local PDF file (HF-only post-commit). ``pdf_url``
    overrides the default https URL (e.g. to test a non-https URL).
    """
    import hashlib

    paper_dir = tw.repo_root() / "docs" / "papers" / f"issue_{tid}"
    paper_dir.mkdir(parents=True, exist_ok=True)
    # The local PDF exists at build time; write it so we can hash it for hf_pdf
    # / the old-shape artifact, then optionally delete it (HF-only post-commit).
    pdf_path = paper_dir / "p.pdf"
    pdf_path.write_text("contents of pdf")
    pdf_sha = hashlib.sha256(pdf_path.read_bytes()).hexdigest()
    pdf_bytes = pdf_path.stat().st_size

    artifacts = {}
    for label, fname in (("tex", "p.tex"), ("paper_html", "paper.html")):
        fpath = paper_dir / fname
        fpath.write_text(f"contents of {label}")
        sha = hashlib.sha256(fpath.read_bytes()).hexdigest()
        if break_sha and label == "paper_html":
            sha = "0" * 64
        artifacts[label] = {
            "path": str(fpath.relative_to(tw.repo_root())),
            "sha256": sha,
            "bytes": fpath.stat().st_size,
        }
    url = pdf_url if pdf_url is not None else ("https://hf/x.pdf" if with_url else None)
    manifest = {
        "schema": "paper_manifest/v1",
        "issue": tid,
        "artifacts": artifacts,
        "pdf_hf_url": url,
    }
    if shape == "hf":
        manifest["hf_pdf"] = {"url": url, "sha256": pdf_sha, "bytes": pdf_bytes}
    elif shape == "old":
        artifacts["pdf"] = {
            "path": str(pdf_path.relative_to(tw.repo_root())),
            "sha256": pdf_sha,
            "bytes": pdf_bytes,
        }
    if delete_pdf:
        pdf_path.unlink()
    import json as _json

    (paper_dir / "paper_manifest.json").write_text(_json.dumps(manifest))


def test_is_paper_task_helper(fake_repo):
    _, tw = fake_repo
    assert tw.is_paper_task({"paper": True})
    assert tw.is_paper_task({"paper": "true"})
    assert tw.is_paper_task({"paper": "TRUE"})
    assert not tw.is_paper_task({"paper": False})
    assert not tw.is_paper_task({"paper": "false"})
    assert not tw.is_paper_task({})


def test_extract_stub_abstract_paragraph(fake_repo):
    _, tw = fake_repo
    body = (
        "# A title\n\nThis is the abstract paragraph that should be extracted.\n\n"
        "Paper: docs/papers/issue_5/issue_5.pdf\n"
    )
    assert (
        tw.extract_stub_abstract(body) == "This is the abstract paragraph that should be extracted."
    )


def test_extract_stub_abstract_h2(fake_repo):
    _, tw = fake_repo
    body = (
        "# T\n\n## Abstract\n\nThe explicit abstract block here.\n\n"
        "Paper: docs/papers/issue_5/x.pdf\n"
    )
    assert tw.extract_stub_abstract(body) == "The explicit abstract block here."


def test_registry_denormalizes_paper_abstract(fake_repo):
    repo, tw = fake_repo
    tid = _make_paper_task(tw, abstract="Denormalized abstract for the hover-card.")
    # Re-flip a registry-updating mutator so _registry_set runs with paper fm.
    tw.set_title(tid, "A paper claim (MODERATE confidence)")
    reg = json.loads((repo / "tasks" / "REGISTRY.json").read_text())
    entry = reg["tasks"][str(tid)]
    assert entry["paper"] is True
    assert entry["abstract"] == "Denormalized abstract for the hover-card."


def test_set_clean_result_paper_requires_manifest(fake_repo):
    _, tw = fake_repo
    tid = _make_paper_task(tw)
    # No manifest yet → hard FAIL (SystemExit).
    with pytest.raises(SystemExit) as exc:
        tw.set_clean_result(tid, value=True)
    assert "paper_manifest" in str(exc.value)


def test_set_clean_result_paper_valid_manifest_passes(fake_repo):
    _, tw = fake_repo
    tid = _make_paper_task(tw)
    _write_manifest(tw, tid, with_url=True)
    tw.set_clean_result(tid, value=True)
    assert tw.get_task(tid)["frontmatter"]["has_clean_result"] is True


def test_set_clean_result_paper_null_url_is_warn_tolerated(fake_repo):
    _, tw = fake_repo
    tid = _make_paper_task(tw)
    _write_manifest(tw, tid, with_url=False)
    # Default allow_paper_warn=True tolerates a null pdf_hf_url (local-only build).
    tw.set_clean_result(tid, value=True)
    assert tw.get_task(tid)["frontmatter"]["has_clean_result"] is True


def test_set_clean_result_paper_null_url_blocked_when_required(fake_repo):
    _, tw = fake_repo
    tid = _make_paper_task(tw)
    _write_manifest(tw, tid, with_url=False)
    with pytest.raises(SystemExit) as exc:
        tw.set_clean_result(tid, value=True, allow_paper_warn=False)
    assert "pdf_hf_url" in str(exc.value)


def test_set_clean_result_paper_sha_mismatch_blocks(fake_repo):
    """A COMMITTED artifact (paper_html) sha mismatch still HARD-blocks promotion."""
    _, tw = fake_repo
    tid = _make_paper_task(tw)
    _write_manifest(tw, tid, with_url=True, break_sha=True)
    with pytest.raises(SystemExit) as exc:
        tw.set_clean_result(tid, value=True)
    assert "sha256" in str(exc.value)


def test_set_clean_result_paper_hf_only_pdf_passes(fake_repo):
    """The storage decision: the PDF lives on HF, NOT committed. Promotion must
    PASS when the manifest's PDF is HF-hosted with NO local pdf file on disk
    (incident #657 — set-clean-result aborted on the local-existence check)."""
    _, tw = fake_repo
    tid = _make_paper_task(tw)
    _write_manifest(tw, tid, with_url=True, delete_pdf=True)
    tw.set_clean_result(tid, value=True)
    assert tw.get_task(tid)["frontmatter"]["has_clean_result"] is True


def test_set_clean_result_paper_old_shape_hf_only_pdf_passes(fake_repo):
    """Tolerance for the OLD manifest shape (#657's existing manifest): a `pdf`
    entry in `artifacts` whose local file is gone (HF-only) still PASSes."""
    _, tw = fake_repo
    tid = _make_paper_task(tw)
    _write_manifest(tw, tid, with_url=True, delete_pdf=True, shape="old")
    tw.set_clean_result(tid, value=True)
    assert tw.get_task(tid)["frontmatter"]["has_clean_result"] is True


def test_set_clean_result_paper_non_https_url_blocks(fake_repo):
    """A present-but-non-https pdf_hf_url is a HARD block — the PDF must resolve
    to a real HF URL now that it is the PDF's authoritative location."""
    _, tw = fake_repo
    tid = _make_paper_task(tw)
    _write_manifest(tw, tid, pdf_url="ftp://nope/x.pdf")
    with pytest.raises(SystemExit) as exc:
        tw.set_clean_result(tid, value=True)
    assert "https" in str(exc.value)


def test_set_clean_result_paper_committed_artifact_missing_blocks(fake_repo):
    """A COMMITTED artifact missing on disk still HARD-blocks (only the PDF is
    exempt from the local-existence check)."""
    _, tw = fake_repo
    tid = _make_paper_task(tw)
    _write_manifest(tw, tid, with_url=True)
    (tw.repo_root() / "docs" / "papers" / f"issue_{tid}" / "paper.html").unlink()
    with pytest.raises(SystemExit) as exc:
        tw.set_clean_result(tid, value=True)
    assert "paper_html" in str(exc.value) and "missing on disk" in str(exc.value)


def test_set_clean_result_nonpaper_skips_manifest_gate(fake_repo):
    """Backward-compat: a non-paper task flips has_clean_result with no manifest
    check (identical to the pre-paper behaviour)."""
    _, tw = fake_repo
    tid = tw.create_task(tw.NewTaskRequest(kind="experiment", title="Ordinary"))
    tw.set_clean_result(tid, value=True)
    assert tw.get_task(tid)["frontmatter"]["has_clean_result"] is True


def test_set_clean_result_unset_paper_skips_gate(fake_repo):
    """Clearing has_clean_result on a paper task never runs the manifest gate."""
    _, tw = fake_repo
    tid = _make_paper_task(tw)
    # No manifest, but value=False → no validation.
    tw.set_clean_result(tid, value=False)
    assert tw.get_task(tid)["frontmatter"]["has_clean_result"] is False


def test_set_clean_result_accepts_report_v1_body(fake_repo):
    """A v2 report body (carries REPORT_V1_SENTINEL) is a valid non-paper
    clean-result form: it is not a paper task and has no paper_manifest.json, so
    set_clean_result flips has_clean_result with no extra gate (mirrors the
    markdown-v4 path). Pins that the report track is accepted, not rejected."""
    _, tw = fake_repo
    tid = tw.create_task(tw.NewTaskRequest(kind="experiment", title="Report task"))
    report_body = (
        "# Experiment: does X predict Y?\n"
        f"{tw.REPORT_V1_SENTINEL}\n\n"
        "## TLDR:\nThomas-written takeaway.\n\n"
        "## Motivation:\nWhy we ran it.\n\n"
        "## Methodology:\nWhat we ran.\n\n"
        "## Metrics:\nWhat we measured.\n\n"
        "## Results:\n### rate\nDescription.\n![r](figures/f.png)\n\n"
        "## Next steps:\nWhat next.\n"
    )
    tw.set_body(tid, report_body)
    assert tw.is_report_body(report_body) is True
    tw.set_clean_result(tid, value=True)
    assert tw.get_task(tid)["frontmatter"]["has_clean_result"] is True


# ─── #657: set_body round-trips the paper-stub opt-in (paper / abstract) ─────


def _stub_text(tid: int, *, paper: bool = True, abstract: str | None = None) -> str:
    """A complete paper-stub markdown document (frontmatter + body) like the
    one the analyzer hands to ``set-body --file STUB_body.md --snapshot``."""
    fm_lines = ['title: "A paper claim (MODERATE confidence)"', "kind: experiment"]
    if paper:
        fm_lines.append("paper: true")
    if abstract is not None:
        fm_lines.append(f'abstract: "{abstract}"')
    fm = "\n".join(fm_lines)
    body = (
        f"# A paper claim\n\n"
        f"An abstract paragraph long enough to count for the stub check.\n\n"
        f"Paper: docs/papers/issue_{tid}/issue_{tid}.pdf\n"
    )
    return f"---\n{fm}\n---\n{body}"


def test_set_body_roundtrips_paper_flag(fake_repo):
    """#657 root cause: `set_body` from a stub carrying `paper: true` must leave
    that key on the on-disk body.md — previously the existing (non-paper)
    frontmatter was preserved and the incoming `paper: true` silently dropped,
    so the dashboard's `isPaperTask` read saw a markdown stub."""
    _, tw = fake_repo
    # An ordinary (non-paper) task — its body.md has no `paper` key.
    tid = tw.create_task(tw.NewTaskRequest(kind="experiment", title="Ordinary"))
    body_path = tw.find_task_path(tid) / "body.md"
    fm_before, _ = tw._read_body(body_path)
    assert "paper" not in fm_before

    tw.set_body(tid, _stub_text(tid, paper=True), snapshot_original=True)

    fm_after, _ = tw._read_body(body_path)
    # The key round-tripped …
    assert fm_after.get("paper") is True
    # … and is_paper_task(fm) now reads True off the actual on-disk body.
    assert tw.is_paper_task(fm_after)
    # REGISTRY denormalizes paper=True for the dashboard list view.
    reg = json.loads(tw.registry_path().read_text())
    assert reg["tasks"][str(tid)]["paper"] is True


def test_set_body_roundtrips_paper_abstract(fake_repo):
    """The denormalized `abstract` stub key round-trips into body.md
    frontmatter. (REGISTRY's `abstract` is independently re-derived from the
    body's first paragraph by `_registry_set` for paper tasks — that
    body-derived value is asserted separately below.)"""
    _, tw = fake_repo
    tid = tw.create_task(tw.NewTaskRequest(kind="experiment", title="Ordinary"))
    tw.set_body(tid, _stub_text(tid, paper=True, abstract="Hover abstract."))
    fm_after, _ = tw._read_body(tw.find_task_path(tid) / "body.md")
    assert fm_after.get("abstract") == "Hover abstract."
    # REGISTRY gains a (body-derived) abstract for the hover-card now that the
    # task reads as a paper-task.
    reg = json.loads(tw.registry_path().read_text())
    assert reg["tasks"][str(tid)]["paper"] is True
    assert reg["tasks"][str(tid)].get("abstract")  # non-empty, body-derived


def test_set_body_nonpaper_does_not_invent_paper_key(fake_repo):
    """Backward-compat: a normal set-body (no `paper` in the incoming body)
    leaves the frontmatter exactly as before — the round-trip allowlist only
    fires when the key is actually present in the incoming body."""
    _, tw = fake_repo
    tid = tw.create_task(tw.NewTaskRequest(kind="experiment", title="Ordinary"))
    tw.set_body(tid, "just a new body, no frontmatter at all\n")
    fm_after, body_after = tw._read_body(tw.find_task_path(tid) / "body.md")
    assert "paper" not in fm_after
    assert "just a new body" in body_after


def test_set_body_then_set_clean_result_paper_end_to_end(fake_repo):
    """The full #657 path: set-body from a `paper: true` stub, then
    set-clean-result with a valid manifest — the on-disk body.md is a real
    paper-task, so the manifest gate runs (and passes)."""
    _, tw = fake_repo
    tid = tw.create_task(tw.NewTaskRequest(kind="experiment", title="Ordinary"))
    tw.set_body(tid, _stub_text(tid, paper=True))
    _write_manifest(tw, tid, with_url=True)
    tw.set_clean_result(tid, value=True)
    fm = tw.get_task(tid)["frontmatter"]
    assert fm["has_clean_result"] is True
    assert tw.is_paper_task(fm)


def test_set_clean_result_gate_fails_when_paper_key_dropped(fake_repo):
    """#657 gate gap: a task whose paper artifacts exist on disk but whose
    body.md frontmatter is MISSING `paper: true` (e.g. written by an older
    set-body that dropped the key, or by hand) must FAIL set-clean-result
    loudly rather than silently passing as a markdown task and skipping the
    paper manifest gate."""
    _, tw = fake_repo
    # Ordinary task body — NO `paper` key.
    tid = tw.create_task(tw.NewTaskRequest(kind="experiment", title="Ordinary"))
    body_path = tw.find_task_path(tid) / "body.md"
    fm_before, _ = tw._read_body(body_path)
    assert "paper" not in fm_before
    # But the paper artifacts + manifest DO exist (this IS intended as a paper).
    _write_manifest(tw, tid, with_url=True)

    with pytest.raises(SystemExit) as exc:
        tw.set_clean_result(tid, value=True)
    msg = str(exc.value)
    assert "paper: true" in msg
    assert "MISSING" in msg


def test_set_clean_result_gate_passes_after_set_body_roundtrip(fake_repo):
    """The gate (above) does NOT fire once `set_body` has correctly
    round-tripped `paper: true` — the body.md is a real paper-task and the
    manifest gate runs normally."""
    _, tw = fake_repo
    tid = tw.create_task(tw.NewTaskRequest(kind="experiment", title="Ordinary"))
    _write_manifest(tw, tid, with_url=True)
    # Fix the body via set-body from a paper stub (the #657 fix path).
    tw.set_body(tid, _stub_text(tid, paper=True))
    tw.set_clean_result(tid, value=True)
    assert tw.get_task(tid)["frontmatter"]["has_clean_result"] is True


def test_set_clean_result_nonpaper_no_manifest_still_skips_gate(fake_repo):
    """A genuinely non-paper task (no manifest on disk) keeps the pre-#657
    behaviour: set-clean-result with no `paper` key and no manifest passes."""
    _, tw = fake_repo
    tid = tw.create_task(tw.NewTaskRequest(kind="experiment", title="Ordinary"))
    tw.set_clean_result(tid, value=True)
    assert tw.get_task(tid)["frontmatter"]["has_clean_result"] is True


# ─── crash-safe events.jsonl (issue #699) ──────────────────────────────────


# T-A
def test_readers_tolerate_partial_trailing_line(fake_repo):
    """A writer killed mid-append leaves a partial JSON line; ALL FOUR readers
    must skip it, not crash (historical #653 corruption)."""
    _, tw = fake_repo
    nid = tw.create_task(tw.NewTaskRequest(kind="infra", title="t"))
    tw.post_event(nid, "epm:plan", by="planner", note="ok line 1")
    tw.post_event(nid, "epm:plan", by="planner", note="ok line 2")
    folder = tw.find_task_path(nid)
    ev = folder / "events.jsonl"
    # Simulate a kill mid-append: a truncated trailing JSON line, no newline.
    with ev.open("a") as f:
        f.write('{"ts": "2026-06-28T00:00:00Z", "kind": "epm:pl')  # no close, no \n
    events = tw.list_events(nid)  # must NOT raise
    kinds = [e["kind"] for e in events]
    assert kinds.count("epm:plan") == 2  # the 2 good plan lines survive
    # _next_event_version must also tolerate (post_event calls it pre-append).
    nxt = tw.post_event(nid, "epm:plan", by="planner", note="ok line 3")
    assert nxt["version"] == 3  # max(1, 2) + 1 over the parseable lines
    assert tw.list_comments(nid) == []  # sibling reader path also fine
    # list_concerns must tolerate a corrupted concerns.jsonl (writer is now
    # crash-safe; the reader must be too — the v1 asymmetry).
    concerns = folder / "concerns.jsonl"
    concerns.write_text('{"event": "raised", "concern_id"')  # garbled, no close
    assert tw.list_concerns(nid) == []  # must NOT raise


def test_readers_tolerate_partial_trailing_multibyte_utf8(fake_repo, caplog):
    """A SIGKILL during a `> PIPE_BUF` `ensure_ascii=False` append can leave a
    TRUNCATED multibyte UTF-8 sequence at the file tail (e.g. `b'{"note":"\\xe2'`
    — a lone first byte of a 3-byte char). Strict UTF-8 (the `read_text()`
    default) raises `UnicodeDecodeError` BEFORE the per-line `json.loads` loop
    reaches the `JSONDecodeError` handler, hard-crashing all four readers. The
    tolerant `errors="replace"` decode must instead let the corrupted line fall
    through to the existing skip path."""
    _, tw = fake_repo
    nid = tw.create_task(tw.NewTaskRequest(kind="infra", title="t"))
    ev = tw.find_task_path(nid) / "events.jsonl"
    # Overwrite with exactly: one valid JSON line + a truncated multibyte tail.
    ev.write_bytes(
        b'{"ts":"2026-06-28T00:00:00Z","kind":"epm:test","version":1,"by":"test"}\n'
        b'{"note":"\xe2'  # lone first byte of a 3-byte UTF-8 sequence, no close
    )
    with caplog.at_level(logging.WARNING, logger="explore_persona_space.task_workflow"):
        events = tw.list_events(nid)  # must NOT raise (UnicodeDecodeError or otherwise)
    assert len(events) == 1  # exactly the one good row survives
    assert events[0]["kind"] == "epm:test"
    # the helper logged a WARNING for the malformed (now U+FFFD-substituted) line
    assert any("skipping malformed line" in r.getMessage() for r in caplog.records), (
        "expected a WARNING for the corrupted multibyte line"
    )


def test_note_with_raw_unicode_line_boundaries_round_trips(fake_repo):
    """#950 regression (incident #825): the `ensure_ascii=False` writer leaves
    raw U+2028/U+2029/NEL inside note strings, and the pre-fix
    `splitlines()`-based `_iter_jsonl` treated those as line boundaries —
    shredding the valid record into skip-malformed fragments = the marker was
    SILENTLY LOST on every read. Under `split("\\n")` the note round-trips
    byte-intact through post_event → list_events."""
    _, tw = fake_repo
    nid = tw.create_task(tw.NewTaskRequest(kind="infra", title="t"))
    # U+2028 LINE SEPARATOR, U+2029 PARAGRAPH SEPARATOR, NEL U+0085 - all
    # written RAW by json.dumps(..., ensure_ascii=False).
    note = "para one\u2028para two\u2029para three\u0085end"
    tw.post_event(nid, "epm:plan", by="planner", note=note)
    plan_events = [e for e in tw.list_events(nid) if e["kind"] == "epm:plan"]
    assert len(plan_events) == 1  # the record survives as exactly ONE row
    assert plan_events[0]["note"] == note  # note byte-intact, boundaries raw


def test_crlf_terminated_record_parses(fake_repo):
    """#950: `split("\\n")` leaves a trailing `\\r` on a `\\r\\n`-terminated
    record; `json.loads` tolerates it as JSON whitespace — pinned here as a
    committed test, not a session probe."""
    _, tw = fake_repo
    nid = tw.create_task(tw.NewTaskRequest(kind="infra", title="t"))
    ev = tw.find_task_path(nid) / "events.jsonl"
    ev.write_bytes(b'{"ts":"2026-07-03T00:00:00Z","kind":"epm:test","version":1,"by":"t"}\r\n')
    events = tw.list_events(nid)
    assert len(events) == 1
    assert events[0]["kind"] == "epm:test"


# T-helper
def test_append_jsonl_line_lands_full_line(fake_repo):
    """A normal-sized append lands a complete, parseable line + newline."""
    _, tw = fake_repo
    nid = tw.create_task(tw.NewTaskRequest(kind="infra", title="t"))
    ev = tw.find_task_path(nid) / "events.jsonl"
    raw = ev.read_text()
    # every non-empty line round-trips and the file ends with a newline
    assert raw.endswith("\n")
    for line in raw.splitlines():
        if line.strip():
            json.loads(line)  # no raise


# T-B (gates the FLOCK no-interleaving property, NOT the single-os.write mechanism)
def test_concurrent_appends_never_interleave_partial_line(fake_repo):
    """N concurrent appenders (serialized by the global flock) must never
    produce a torn line; every line round-trips through json.loads.

    NOTE: this gates the flock's no-interleaving property only. It does NOT
    gate the single-os.write atomicity A1 relies on — see T-C, which gates the
    mechanism directly."""
    _, tw = fake_repo
    nid = tw.create_task(tw.NewTaskRequest(kind="infra", title="t"))
    n_writers, per_writer = 8, 12

    def worker(wid):
        for i in range(per_writer):
            tw.post_event(nid, f"epm:w{wid}", by="t", note=f"{wid}-{i}")

    with concurrent.futures.ThreadPoolExecutor(max_workers=n_writers) as ex:
        list(ex.map(worker, range(n_writers)))

    ev = tw.find_task_path(nid) / "events.jsonl"
    lines = [ln for ln in ev.read_text().splitlines() if ln.strip()]
    for ln in lines:
        json.loads(ln)  # no torn lines
    posted = sum(1 for ln in lines if json.loads(ln)["kind"].startswith("epm:w"))
    assert posted == n_writers * per_writer  # no lost/duplicated lines


# T-C (gates the single-os.write MECHANISM the flock test cannot)
def test_small_append_is_exactly_one_os_write(fake_repo, tmp_path, monkeypatch):
    """The <= PIPE_BUF path MUST perform exactly ONE os.write of the full
    serialized line. A buffered or multi-write refactor would reintroduce the
    #653 torn-line bug on SIGKILL yet pass the flock test (T-B); this catches
    that regression at the mechanism level."""
    _, tw = fake_repo

    calls: list[tuple[int, int]] = []
    real_write = os.write
    target = tmp_path / "small.jsonl"
    # The fd is allocated inside _append_jsonl_line; capture every os.write and
    # filter to writes of our buffer length so unrelated fds (git, logging)
    # never pollute the count.
    payload = {"ts": "2026-06-28T00:00:00Z", "kind": "epm:x", "version": 1, "by": "t"}
    line = json.dumps(payload, ensure_ascii=False) + "\n"
    expected = len(line.encode("utf-8"))
    assert expected <= tw._PIPE_BUF  # precondition: small path

    def counting_write(fd, data):
        calls.append((fd, len(data)))
        return real_write(fd, data)

    monkeypatch.setattr(tw.os, "write", counting_write)
    tw._append_jsonl_line(target, payload)

    assert len(calls) == 1  # EXACTLY one os.write
    assert calls[0][1] == expected  # of the full line length
    assert target.read_text() == line  # round-trips on disk


# T-C-bonus (short-write on the <= PIPE_BUF path must fail loud)
def test_small_append_short_write_raises(fake_repo, tmp_path, monkeypatch):
    """If os.write short-writes on the <= PIPE_BUF path, the helper must raise
    OSError (the line-guard), never leave a silent partial line."""
    _, tw = fake_repo
    real_write = os.write

    def short_then_real(fd, data):
        # write only the first byte, simulating a short atomic write
        return real_write(fd, data[:1])

    monkeypatch.setattr(tw.os, "write", short_then_real)
    target = tmp_path / "short.jsonl"
    payload = {"ts": "t", "kind": "epm:x", "version": 1, "by": "t"}
    with pytest.raises(OSError):
        tw._append_jsonl_line(target, payload)


# T-D (drives the > PIPE_BUF oversize branch no other test exercises)
def test_oversize_append_roundtrips_via_completion_loop(fake_repo):
    """An event note big enough to push the serialized line over PIPE_BUF must
    round-trip through list_events, end the file with a newline, and actually
    take the oversize completion-loop branch."""
    _, tw = fake_repo
    nid = tw.create_task(tw.NewTaskRequest(kind="infra", title="t"))
    big_note = "x" * 5000  # < EVENT_NOTE_MAX (50k), > PIPE_BUF
    # precondition: the constructed payload serializes to > PIPE_BUF
    probe = {"ts": "t", "kind": "epm:big", "version": 1, "by": "t", "note": big_note}
    assert len(json.dumps(probe, ensure_ascii=False).encode("utf-8")) > tw._PIPE_BUF

    tw.post_event(nid, "epm:big", by="t", note=big_note)
    events = tw.list_events(nid)  # must round-trip
    big = [e for e in events if e["kind"] == "epm:big"]
    assert len(big) == 1
    assert big[0]["note"] == big_note  # full note survived
    ev = tw.find_task_path(nid) / "events.jsonl"
    assert ev.read_text().endswith("\n")  # complete line + newline


# T-D-bonus (forces a short write inside the oversize loop)
def test_oversize_append_completes_across_short_writes(fake_repo, tmp_path, monkeypatch):
    """The oversize completion loop must finish the full buffer even when
    os.write returns short counts on each call."""
    _, tw = fake_repo
    real_write = os.write

    def chunked_write(fd, data):
        # write at most 1024 bytes per call to force the while-loop to iterate
        return real_write(fd, data[:1024])

    monkeypatch.setattr(tw.os, "write", chunked_write)
    target = tmp_path / "big.jsonl"
    payload = {"ts": "t", "kind": "epm:big", "version": 1, "by": "t", "note": "y" * 5000}
    line = json.dumps(payload, ensure_ascii=False) + "\n"
    assert len(line.encode("utf-8")) > tw._PIPE_BUF
    tw._append_jsonl_line(target, payload)
    assert target.read_text() == line  # full buffer completed


# ─── index.lock retry + crash-safe set_status (#898) ────────────────────────
#
# Incident #825: a concurrent session held .git/index.lock while set_status
# ran; the `git add` crash left the folder moved with REGISTRY pointing at
# the old path — the task was unfindable until a manual `audit --repair`.
# The fix set: (1) _run_git retries ONCE on the git lock-contention stderr
# signature; (2) set_status completes ALL durable state (FS move + verify,
# REGISTRY save, event append) BEFORE any git op; (3) find_task_path scans
# the tasks/ tree when the registry entry is stale; (4) the ghost-deletion
# sweep (_task_status_dir_pathspecs) reconciles a crashed transition's
# leftover old-status dir on the task's NEXT transition; (5) the
# same-transition early return re-syncs a stale registry entry in place.


def _make_index_lock(repo: Path) -> Path:
    """Create a real .git/index.lock so the repo's own git binary emits the
    REAL lock-contention error (pins the retry regex against reality)."""
    lock = repo / ".git" / "index.lock"
    lock.write_text("")
    return lock


def test_run_git_retries_once_on_index_lock_collision(fake_repo, monkeypatch):
    """A held index.lock that clears during the backoff sleep resolves via
    exactly ONE retry, with the jittered delay drawn from the constant range
    (asserted against tw._GIT_LOCK_RETRY_SLEEP_RANGE_S, not literals)."""
    repo, tw = fake_repo
    (repo / "somefile.txt").write_text("x\n")
    lock = _make_index_lock(repo)

    sleeps: list[float] = []

    def fake_sleep(delay: float) -> None:
        sleeps.append(delay)
        lock.unlink()  # the concurrent committer finishes during the backoff

    monkeypatch.setattr(tw.time, "sleep", fake_sleep)
    result = tw._run_git(["add", "--", "somefile.txt"])

    assert result.returncode == 0
    assert len(sleeps) == 1  # exactly one retry sleep
    lo, hi = tw._GIT_LOCK_RETRY_SLEEP_RANGE_S
    assert lo <= sleeps[0] <= hi


def test_run_git_retry_exhaustion_raises_calledprocesserror(fake_repo, monkeypatch, caplog):
    """A lock that does NOT clear fails after exactly one retry (never two)
    with subprocess.CalledProcessError and the stale-lock remedy logged."""
    repo, tw = fake_repo
    (repo / "somefile.txt").write_text("x\n")
    _make_index_lock(repo)  # never removed — retry hits the lock again

    sleeps: list[float] = []
    monkeypatch.setattr(tw.time, "sleep", lambda d: sleeps.append(d))

    with (
        caplog.at_level(logging.WARNING, logger="explore_persona_space.task_workflow"),
        pytest.raises(subprocess.CalledProcessError),
    ):
        tw._run_git(["add", "--", "somefile.txt"])

    assert len(sleeps) == 1  # one retry sleep, never a second
    assert any("index.lock" in r.getMessage() for r in caplog.records)


def test_run_git_does_not_retry_non_lock_errors(fake_repo, monkeypatch):
    """A non-lock git failure (unmatched pathspec) raises immediately with
    ZERO sleeps — the retry keys on the lock signature only."""
    _, tw = fake_repo
    monkeypatch.setattr(tw.time, "sleep", lambda d: pytest.fail("retry sleep on non-lock error"))
    with pytest.raises(subprocess.CalledProcessError):
        tw._run_git(["add", "--", "does-not-exist.txt"], check=True)


def test_run_git_check_false_rc_signal_not_retried(fake_repo, monkeypatch):
    """`diff --cached --quiet` uses rc=1 as a SIGNAL (staged changes exist),
    with empty stderr — it must pass through un-retried and un-raised."""
    repo, tw = fake_repo
    (repo / "somefile.txt").write_text("x\n")
    subprocess.run(["git", "add", "somefile.txt"], cwd=repo, check=True)
    monkeypatch.setattr(tw.time, "sleep", lambda d: pytest.fail("retry sleep on rc-as-signal"))

    result = tw._run_git(["diff", "--cached", "--quiet"], check=False)

    assert result.returncode == 1  # the rc signal survives, no raise


def test_run_git_happy_path_no_sleep(fake_repo, monkeypatch):
    """A SUCCESSFUL _run_git call takes zero sleeps (AC5: zero added
    happy-path latency)."""
    repo, tw = fake_repo
    (repo / "somefile.txt").write_text("x\n")
    monkeypatch.setattr(tw.time, "sleep", lambda d: pytest.fail("sleep on a successful call"))
    result = tw._run_git(["add", "--", "somefile.txt"])
    assert result.returncode == 0


def _lock_stderr() -> str:
    return "fatal: Unable to create '/x/.git/index.lock': File exists.\n"


def _install_git_add_all_crash(tw, mp: pytest.MonkeyPatch) -> None:
    """Monkeypatch tw._run_git to crash ONLY on the set_status step-6
    standalone staging call (`add` + `--all`) — not _git_commit's internal
    `add` (no `--all`), so the pin cannot drift to the wrong crash point."""
    real_run_git = tw._run_git

    def crashing_run_git(args, *, check=True):
        if args and args[0] == "add" and "--all" in args:
            raise subprocess.CalledProcessError(
                128, ["git", *args], output="", stderr=_lock_stderr()
            )
        return real_run_git(args, check=check)

    mp.setattr(tw, "_run_git", crashing_run_git)


def test_set_status_git_add_crash_leaves_registry_consistent_with_disk(fake_repo):
    """THE #825 regression pin (red on the pre-#898 order): a crash at the
    step-6 `git add --all` staging must leave disk, REGISTRY, and
    events.jsonl all consistent with the transition APPLIED, and
    find_task_path resolving."""
    repo, tw = fake_repo
    new_id = tw.create_task(tw.NewTaskRequest(kind="experiment", title="X"))

    # Install the crash injection AFTER task creation (create_task itself
    # drives _git_commit -> _run_git(["add", ...])).
    with pytest.MonkeyPatch.context() as mp:
        _install_git_add_all_crash(tw, mp)
        with pytest.raises(subprocess.CalledProcessError):
            tw.set_status(new_id, "running")

    new = repo / "tasks" / "running" / str(new_id)
    assert new.is_dir()  # disk: moved
    assert not (repo / "tasks" / "proposed" / str(new_id)).exists()
    reg = json.loads((repo / "tasks" / "REGISTRY.json").read_text())
    assert reg["tasks"][str(new_id)]["path"] == f"tasks/running/{new_id}"  # registry consistent
    events = [json.loads(line) for line in (new / "events.jsonl").read_text().splitlines() if line]
    assert any(
        e["kind"] == "epm:status-changed" and e.get("to") == "running" for e in events
    )  # event appended
    assert tw.find_task_path(new_id) == new  # still resolvable


def test_set_status_git_commit_crash_leaves_registry_consistent_with_disk(fake_repo):
    """REORDER-SURVIVAL pin (green on the pre-#898 order too, since the
    registry was already saved before _git_commit under the old order; test
    above is the sole #825 detector): a _git_commit crash must leave the
    same consistent disk + REGISTRY + events state."""
    repo, tw = fake_repo
    new_id = tw.create_task(tw.NewTaskRequest(kind="experiment", title="X"))

    def crashing_commit(paths, message):
        raise subprocess.CalledProcessError(
            128, ["git", "commit"], output="", stderr=_lock_stderr()
        )

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(tw, "_git_commit", crashing_commit)
        with pytest.raises(subprocess.CalledProcessError):
            tw.set_status(new_id, "running")

    new = repo / "tasks" / "running" / str(new_id)
    assert new.is_dir()
    assert not (repo / "tasks" / "proposed" / str(new_id)).exists()
    reg = json.loads((repo / "tasks" / "REGISTRY.json").read_text())
    assert reg["tasks"][str(new_id)]["path"] == f"tasks/running/{new_id}"
    events = [json.loads(line) for line in (new / "events.jsonl").read_text().splitlines() if line]
    assert any(e["kind"] == "epm:status-changed" and e.get("to") == "running" for e in events)
    assert tw.find_task_path(new_id) == new


def test_find_task_path_stale_registry_entry_falls_back_to_scan(fake_repo, caplog):
    """A stale registry entry (dir moved on disk, registry not updated — the
    hard-kill residue shape) resolves via the on-disk scan with a logged
    drift warning naming REGISTRY + the audit remedy."""
    repo, tw = fake_repo
    new_id = tw.create_task(tw.NewTaskRequest(kind="experiment", title="X"))
    old = repo / "tasks" / "proposed" / str(new_id)
    dest = repo / "tasks" / "interpreting" / str(new_id)
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(old), str(dest))  # registry left pointing at proposed

    with caplog.at_level(logging.WARNING, logger="explore_persona_space.task_workflow"):
        path = tw.find_task_path(new_id)

    assert path == dest
    messages = [r.getMessage() for r in caplog.records]
    assert any("REGISTRY" in m and "audit" in m for m in messages)


def test_find_task_path_ambiguous_duplicate_dirs_raises(fake_repo):
    """A stale registry entry with the task dir present under TWO statuses is
    real corruption: raise StaleTaskPathError naming both paths (never guess
    one silently)."""
    repo, tw = fake_repo
    new_id = tw.create_task(tw.NewTaskRequest(kind="experiment", title="X"))
    old = repo / "tasks" / "proposed" / str(new_id)
    d1 = repo / "tasks" / "running" / str(new_id)
    d2 = repo / "tasks" / "verifying" / str(new_id)
    d1.parent.mkdir(parents=True, exist_ok=True)
    d2.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(old, d1)
    shutil.copytree(old, d2)
    shutil.rmtree(old)  # registry still points at proposed (missing)

    with pytest.raises(tw.StaleTaskPathError) as exc_info:
        tw.find_task_path(new_id)
    msg = str(exc_info.value)
    assert f"tasks/running/{new_id}" in msg
    assert f"tasks/verifying/{new_id}" in msg


def test_set_status_ghost_deletion_swept_by_next_transition(fake_repo):
    """The BINDING v2 Must-Fix pin (red without the §4.4 sweep): after a
    git-crash residue on transition A→B, the NEXT transition B→C must sweep
    the ghost old-status dir out of HEAD — no committed duplicate of the
    task, no permanent unstaged deletions under tasks/."""
    repo, tw = fake_repo
    new_id = tw.create_task(tw.NewTaskRequest(kind="experiment", title="X"))

    # Transition A→B crashes at the step-6 staging: nothing staged/committed;
    # HEAD still tracks tasks/proposed/<id> (the ghost) while disk has moved on.
    with pytest.MonkeyPatch.context() as mp:
        _install_git_add_all_crash(tw, mp)
        with pytest.raises(subprocess.CalledProcessError):
            tw.set_status(new_id, "running")

    # Injection removed — transition B→C must reconcile the residue.
    tw.set_status(new_id, "verifying")

    def ls_tree(pathspec: str) -> str:
        return subprocess.run(
            ["git", "ls-tree", "-r", "HEAD", "--", pathspec],
            cwd=repo,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()

    assert ls_tree(f"tasks/proposed/{new_id}") == ""  # ghost swept
    assert ls_tree(f"tasks/running/{new_id}") == ""  # intermediate swept
    assert ls_tree(f"tasks/verifying/{new_id}") != ""  # current status committed
    porcelain = subprocess.run(
        ["git", "status", "--porcelain", "--", "tasks/"],
        cwd=repo,
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    dirty = [line for line in porcelain.splitlines() if f"/{new_id}/" in line]
    assert dirty == [], f"leftover unstaged/uncommitted task paths: {dirty}"


def test_set_status_same_transition_retry_resyncs_stale_registry(fake_repo, caplog):
    """§4.5 pin (red without the early-return re-sync): retrying the SAME
    transition against a stale registry entry (dir already at the
    destination, registry pointing at the old path — the hard-kill shape)
    must re-sync the registry before the idempotent early return."""
    repo, tw = fake_repo
    new_id = tw.create_task(tw.NewTaskRequest(kind="experiment", title="X"))
    old = repo / "tasks" / "proposed" / str(new_id)
    dest = repo / "tasks" / "running" / str(new_id)
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(old), str(dest))  # hard-kill shape: moved, registry stale

    with caplog.at_level(logging.WARNING, logger="explore_persona_space.task_workflow"):
        returned = tw.set_status(new_id, "running")  # the SAME transition

    assert returned == dest  # early return fired with the on-disk path
    reg = json.loads((repo / "tasks" / "REGISTRY.json").read_text())
    assert reg["tasks"][str(new_id)]["path"] == f"tasks/running/{new_id}"  # re-synced
    assert any("re-synced stale REGISTRY entry" in r.getMessage() for r in caplog.records)


# ─── deferred commits + sequencer-state wait (#1030) ─────────────────────────
#
# Seam (a): _commit_after_durable_append — a git-commit failure AFTER a
# durable append is deferred on the PRIMARY checkout (loud ERROR + forensic
# sidecar row at tw.DEFERRED_COMMITS_LOG, patched into tmp_path by the
# fake_repo fixture), never raised into the caller's retry recipe (the
# 2026-07-03 3x-marker incident). Every failure class where durability is NOT
# guaranteed still raises: routed mode, the routed post-commit CAS
# RuntimeError, genuine bugs (TypeError/...), and append failures.
# Seam (b): _git_commit waits out a concurrent merge/cherry-pick
# (MERGE_HEAD / CHERRY_PICK_HEAD, per-worktree via `rev-parse --git-path`)
# with a bounded env-tunable loop before `commit --only` fatals, raising
# SequencerWaitTimeout on timeout, plus a single TOCTOU re-wait keyed on the
# partial-commit stderr signature. Concurrency is SIMULATED (injected
# failures / hand-created state files) — no real concurrent merge.


def _deferred_rows(tw) -> list[dict]:
    """Parse the per-test deferred-commit sidecar (empty list if absent)."""
    log = tw.DEFERRED_COMMITS_LOG
    if not log.exists():
        return []
    return [json.loads(line) for line in log.read_text().splitlines() if line.strip()]


def _commit_crash(paths, message):
    """Injected _git_commit failure with the real lock-collision stderr."""
    raise subprocess.CalledProcessError(128, ["git", "commit"], output="", stderr=_lock_stderr())


def _head_sha(repo: Path) -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=repo, capture_output=True, text=True, check=True
    ).stdout


def _load_task_cli():
    """Load scripts/task.py as an isolated module (importlib pattern)."""
    import importlib.util

    path = Path(__file__).resolve().parents[1] / "scripts" / "task.py"
    spec = importlib.util.spec_from_file_location("task_cli_under_test_1030", path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_post_event_deferred_commit_returns_payload(fake_repo, monkeypatch, caplog):
    """AC1: a commit failure AFTER the events.jsonl append landed returns the
    payload (success), appends exactly ONE events line, logs an ERROR, and
    records exactly one forensic sidecar row (op=post_event)."""
    _, tw = fake_repo
    new_id = tw.create_task(tw.NewTaskRequest(kind="experiment", title="X"))
    monkeypatch.setattr(tw, "_git_commit", _commit_crash)

    with caplog.at_level(logging.ERROR, logger="explore_persona_space.task_workflow"):
        payload = tw.post_event(new_id, "epm:progress", note="deferred-commit probe")

    assert payload["kind"] == "epm:progress"
    events = tw.list_events(new_id)
    assert sum(1 for e in events if e["kind"] == "epm:progress") == 1  # exactly one append
    assert any("Do NOT re-run" in r.getMessage() for r in caplog.records)
    rows = _deferred_rows(tw)
    assert len(rows) == 1
    assert rows[0]["op"] == "post_event"
    assert rows[0]["task_id"] == new_id


def test_post_event_append_failure_raises_no_deferred_row(fake_repo, monkeypatch):
    """AC2: an append failure still raises out of post_event — no sidecar
    row, no commit attempt (the deferral covers post-append failures ONLY)."""
    _, tw = fake_repo
    new_id = tw.create_task(tw.NewTaskRequest(kind="experiment", title="X"))
    commits: list = []
    monkeypatch.setattr(tw, "_git_commit", lambda paths, message: commits.append(list(paths)))

    def broken_append(path, payload):
        raise OSError("disk full")

    monkeypatch.setattr(tw, "_append_jsonl_line", broken_append)
    with pytest.raises(OSError, match="disk full"):
        tw.post_event(new_id, "epm:progress", note="x")
    assert commits == []  # _git_commit never called
    assert _deferred_rows(tw) == []


def test_deferred_commit_swept_by_next_commit(fake_repo):
    """AC3: after a deferred commit, the NEXT successful mutation touching the
    same file commits the pending line too (git commits file STATE)."""
    repo, tw = fake_repo
    new_id = tw.create_task(tw.NewTaskRequest(kind="experiment", title="X"))
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(tw, "_git_commit", _commit_crash)
        tw.post_event(new_id, "epm:progress", note="first-deferred")
    tw.post_event(new_id, "epm:progress", note="second-committed")  # real git

    ev_rel = f"tasks/proposed/{new_id}/events.jsonl"
    porcelain = subprocess.run(
        ["git", "status", "--porcelain", "--", ev_rel],
        cwd=repo,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    assert porcelain == ""  # clean — the deferred line was swept
    committed = subprocess.run(
        ["git", "show", f"HEAD:{ev_rel}"], cwd=repo, capture_output=True, text=True, check=True
    ).stdout
    assert "first-deferred" in committed
    assert "second-committed" in committed


@pytest.mark.parametrize("which", ["append_comment", "raise_concern"])
def test_append_comment_and_concern_deferred_commit(fake_repo, monkeypatch, which):
    """Deferral covers the comment + concern append sites: injected commit
    failure → row appended, return value intact, sidecar op correct."""
    repo, tw = fake_repo
    new_id = tw.create_task(tw.NewTaskRequest(kind="experiment", title="X"))
    monkeypatch.setattr(tw, "_git_commit", _commit_crash)
    task_dir = repo / "tasks" / "proposed" / str(new_id)

    if which == "append_comment":
        rec = tw.append_comment(new_id, author="tester", kind="note", body="hi")
        assert rec["id"] == "c001"
        appended_file = task_dir / "comments.jsonl"
        expected_op = "append_comment"
    else:
        rec = tw.raise_concern(
            new_id,
            "probe-concern",
            severity="CONCERN",
            summary="a probe concern",
            raised_by="tester",
            raised_at_round=1,
        )
        assert rec["concern_id"] == "probe-concern"
        appended_file = task_dir / "concerns.jsonl"
        expected_op = "append_concern_event"

    assert appended_file.exists() and appended_file.read_text().strip()
    assert [r["op"] for r in _deferred_rows(tw)] == [expected_op]


def test_create_and_new_plan_version_deferred_commit(fake_repo, monkeypatch):
    """Deferral covers create + new_plan_version: the id / next_v return
    values survive an injected commit failure and the artifacts exist."""
    repo, tw = fake_repo
    monkeypatch.setattr(tw, "_git_commit", _commit_crash)
    new_id = tw.create_task(tw.NewTaskRequest(kind="experiment", title="X"))
    assert tw.find_task_path(new_id) == repo / "tasks" / "proposed" / str(new_id)
    v = tw.new_plan_version(new_id, "plan body")
    assert v == 1
    assert (repo / "tasks" / "proposed" / str(new_id) / "plans" / "v1.md").exists()
    assert [r["op"] for r in _deferred_rows(tw)] == ["create", "new_plan_version"]


def test_deferred_commit_sidecar_schema(fake_repo, monkeypatch):
    """The sidecar row is exactly {ts, task_id, op, paths, message, error,
    stderr_tail}, one valid JSONL line per deferral."""
    _, tw = fake_repo
    new_id = tw.create_task(tw.NewTaskRequest(kind="experiment", title="X"))
    monkeypatch.setattr(tw, "_git_commit", _commit_crash)
    tw.post_event(new_id, "epm:progress", note="schema probe")

    raw_lines = [ln for ln in tw.DEFERRED_COMMITS_LOG.read_text().splitlines() if ln.strip()]
    assert len(raw_lines) == 1
    row = json.loads(raw_lines[0])  # valid JSONL
    assert set(row) == {"ts", "task_id", "op", "paths", "message", "error", "stderr_tail"}
    assert row["error"] == "CalledProcessError"
    assert "index.lock" in row["stderr_tail"]
    assert isinstance(row["paths"], list) and row["paths"]


def test_deferred_sidecar_write_failure_does_not_mask_success(fake_repo, monkeypatch, caplog):
    """A sidecar-write failure must not resurrect the duplicate-append bug:
    the payload is still returned and the failure is logged."""
    repo, tw = fake_repo
    new_id = tw.create_task(tw.NewTaskRequest(kind="experiment", title="X"))
    blocker = repo / "sidecar-blocker"
    blocker.write_text("")  # sidecar parent is a FILE -> mkdir raises OSError
    monkeypatch.setattr(tw, "DEFERRED_COMMITS_LOG", blocker / "deferred.jsonl")
    monkeypatch.setattr(tw, "_git_commit", _commit_crash)

    with caplog.at_level(logging.ERROR, logger="explore_persona_space.task_workflow"):
        payload = tw.post_event(new_id, "epm:progress", note="x")

    assert payload["kind"] == "epm:progress"  # success not masked
    assert any("could not record deferred-commit row" in r.getMessage() for r in caplog.records)


@pytest.mark.parametrize("exc_type", [TypeError, AttributeError])
def test_wrapper_propagates_non_git_bugs_no_deferred_row(fake_repo, monkeypatch, exc_type):
    """AC2b / MF-3: a genuine bug from _git_commit propagates out of
    post_event — matches neither caught class, no sidecar row."""
    _, tw = fake_repo
    new_id = tw.create_task(tw.NewTaskRequest(kind="experiment", title="X"))

    def buggy_commit(paths, message):
        raise exc_type("genuine bug")

    monkeypatch.setattr(tw, "_git_commit", buggy_commit)
    with pytest.raises(exc_type):
        tw.post_event(new_id, "epm:progress", note="x")
    assert _deferred_rows(tw) == []


def test_bare_runtimeerror_cas_class_never_deferred(fake_repo, monkeypatch):
    """AC2c / MF-4: bare RuntimeError (the routed post-commit CAS class from
    _git_quiet) is never caught by the wrapper — an `except RuntimeError` /
    `except Exception` mutant fails this test."""
    _, tw = fake_repo
    new_id = tw.create_task(tw.NewTaskRequest(kind="experiment", title="X"))

    def cas_crash(paths, message):
        raise RuntimeError("update-ref CAS failed")

    monkeypatch.setattr(tw, "_git_commit", cas_crash)
    with pytest.raises(RuntimeError, match="CAS"):
        tw.post_event(new_id, "epm:progress", note="x")
    assert _deferred_rows(tw) == []


def test_routed_mode_commit_failure_raises_no_deferral(fake_repo, monkeypatch):
    """AC2c / MF-4: in routed mode ANY commit failure raises — an uncommitted
    deferred line would be PHYSICALLY DELETED by the resolver's next
    `reset --hard main` re-sync."""
    _, tw = fake_repo
    new_id = tw.create_task(tw.NewTaskRequest(kind="experiment", title="X"))
    monkeypatch.setattr(tw, "_is_routed_root", lambda root: True)
    monkeypatch.setattr(tw, "_git_commit", _commit_crash)
    with pytest.raises(subprocess.CalledProcessError):
        tw.post_event(new_id, "epm:progress", note="x")
    assert _deferred_rows(tw) == []


def test_routed_cas_failure_after_commit_raises_no_deferred_row(fake_repo, monkeypatch):
    """MF-4 mid-level integration: REAL _git_commit with the routed flag
    forced on — the commit object IS created, then the CAS leg raises
    RuntimeError; post_event raises and no sidecar row lands. Pins the
    exception ROUTING of the real routed branch, not git topology."""
    repo, tw = fake_repo
    new_id = tw.create_task(tw.NewTaskRequest(kind="experiment", title="X"))
    monkeypatch.setattr(tw, "_is_routed_root", lambda root: True)

    def cas_crash(managed, old_sha, new_sha, env):
        raise RuntimeError(f"`git update-ref` failed: CAS mismatch {old_sha}->{new_sha}")

    monkeypatch.setattr(tw, "_advance_main_ref", cas_crash)
    before = _git_log_count(repo)
    with pytest.raises(RuntimeError, match="update-ref"):
        tw.post_event(new_id, "epm:progress", note="routed cas probe")
    assert _git_log_count(repo) == before + 1  # the commit itself landed
    assert _deferred_rows(tw) == []


def test_set_status_git_failure_still_raises_not_deferred_1030_pin(fake_repo, monkeypatch):
    """AC6: set_status keeps the #898 raise semantics — deliberately NOT
    converted to deferral (the existing #898 tests remain the primary pin)."""
    repo, tw = fake_repo
    new_id = tw.create_task(tw.NewTaskRequest(kind="experiment", title="X"))
    monkeypatch.setattr(tw, "_git_commit", _commit_crash)
    with pytest.raises(subprocess.CalledProcessError):
        tw.set_status(new_id, "running")
    assert (repo / "tasks" / "running" / str(new_id)).is_dir()  # durably applied
    assert _deferred_rows(tw) == []  # and never routed through the deferral sidecar


def test_set_status_sequencer_timeout_keeps_898_recovery_envelope(fake_repo, monkeypatch, caplog):
    """AC6 / MF-1: a SequencerWaitTimeout from _git_commit's merge wait gets
    the same "DURABLY APPLIED" recovery narration as a plain git failure,
    then re-raises; the transition is durably applied throughout."""
    repo, tw = fake_repo
    new_id = tw.create_task(tw.NewTaskRequest(kind="experiment", title="X"))
    (repo / ".git" / "MERGE_HEAD").write_text(_head_sha(repo))  # never clears
    monkeypatch.setenv("EPM_TASKPY_MERGE_WAIT_SECONDS", "0.05")
    monkeypatch.setenv("EPM_TASKPY_MERGE_POLL_SECONDS", "0.01")

    with (
        caplog.at_level(logging.ERROR, logger="explore_persona_space.task_workflow"),
        pytest.raises(tw.SequencerWaitTimeout),
    ):
        tw.set_status(new_id, "running")

    new = repo / "tasks" / "running" / str(new_id)
    assert new.is_dir()  # disk: moved
    reg = json.loads((repo / "tasks" / "REGISTRY.json").read_text())
    assert reg["tasks"][str(new_id)]["path"] == f"tasks/running/{new_id}"  # registry updated
    events = [json.loads(line) for line in (new / "events.jsonl").read_text().splitlines() if line]
    assert any(e["kind"] == "epm:status-changed" and e.get("to") == "running" for e in events)
    assert any("DURABLY APPLIED" in r.getMessage() for r in caplog.records)


@pytest.mark.parametrize("state_file", ["MERGE_HEAD", "CHERRY_PICK_HEAD"])
def test_git_commit_waits_out_transient_sequencer_state(fake_repo, monkeypatch, state_file):
    """AC4: _git_commit polls until a transient merge/cherry-pick clears, then
    commits — exactly one sleep of _merge_poll_s()."""
    repo, tw = fake_repo
    target = repo / "somefile.txt"
    target.write_text("x\n")
    state = repo / ".git" / state_file
    state.write_text(_head_sha(repo))
    sleeps: list[float] = []

    def clearing_sleep(delay: float) -> None:
        sleeps.append(delay)
        state.unlink()  # the concurrent merge finishes during the poll sleep

    monkeypatch.setattr(tw.time, "sleep", clearing_sleep)
    before = _git_log_count(repo)
    tw._git_commit([target], "merge-wait probe")
    assert _git_log_count(repo) == before + 1
    assert sleeps == [tw._merge_poll_s()]


def test_git_commit_no_sequencer_state_zero_sleeps(fake_repo, monkeypatch):
    """AC4 happy path: with no sequencer state, _git_commit takes ZERO sleeps
    (one cheap rev-parse probe only) and commits."""
    repo, tw = fake_repo
    target = repo / "somefile.txt"
    target.write_text("x\n")
    monkeypatch.setattr(tw.time, "sleep", lambda d: pytest.fail("sleep with no sequencer state"))
    before = _git_log_count(repo)
    tw._git_commit([target], "no-wait probe")
    assert _git_log_count(repo) == before + 1


def test_git_commit_sequencer_timeout_raises_loud(fake_repo, monkeypatch):
    """AC5: a never-clearing MERGE_HEAD raises SequencerWaitTimeout naming the
    sync_repo_root remedy, and the commit is never attempted."""
    repo, tw = fake_repo
    target = repo / "somefile.txt"
    target.write_text("x\n")
    (repo / ".git" / "MERGE_HEAD").write_text(_head_sha(repo))
    monkeypatch.setenv("EPM_TASKPY_MERGE_WAIT_SECONDS", "0.05")
    monkeypatch.setenv("EPM_TASKPY_MERGE_POLL_SECONDS", "0.01")
    argvs: list[list[str]] = []
    real_run_git = tw._run_git

    def spying_run_git(args, *, check=True):
        argvs.append(list(args))
        return real_run_git(args, check=check)

    monkeypatch.setattr(tw, "_run_git", spying_run_git)
    with pytest.raises(tw.SequencerWaitTimeout, match="sync_repo_root"):
        tw._git_commit([target], "timeout probe")
    assert argvs, "expected at least the rev-parse --git-path probe"
    assert all(a[0] != "commit" for a in argvs)  # never reached the commit


def test_git_commit_merge_wait_knob_zero_restores_git_fatal(fake_repo, monkeypatch):
    """AC5 escape hatch + A1 reality pin: knob=0 disables the wait entirely —
    zero sleeps — and the REAL git partial-commit fatal (rc=128) surfaces
    unchanged (pins the fatal against the real git binary, the same
    pin-against-reality philosophy as _make_index_lock)."""
    repo, tw = fake_repo
    target = repo / "somefile.txt"
    target.write_text("x\n")
    (repo / ".git" / "MERGE_HEAD").write_text(_head_sha(repo))
    monkeypatch.setenv("EPM_TASKPY_MERGE_WAIT_SECONDS", "0")
    monkeypatch.setattr(tw.time, "sleep", lambda d: pytest.fail("wait ran with knob=0"))

    with pytest.raises(subprocess.CalledProcessError) as exc_info:
        tw._git_commit([target], "knob-zero probe")

    assert exc_info.value.returncode == 128
    assert "cannot do a partial commit" in (exc_info.value.stderr or "")


# ─── Post-commit landing check (#1100) ─────────────────────────────────────
#
# _git_commit's tail runs _warn_if_commit_stranded: a warn-only, fail-open
# tripwire that probes `git merge-base --is-ancestor <new-commit>
# refs/heads/main` and, on a miss, logs ONE ERROR + appends a forensic row to
# STRANDED_COMMITS_LOG (monkeypatched to tmp by the fake_repo fixture). The
# strand class under test is the #1083 guard-escape: a checkout switched off
# main underneath a resolver that already cached its (pid, cwd) resolution.


def _stranded_rows(tw) -> list[dict]:
    """Parse the (fixture-retargeted) stranded-commits sidecar; [] if absent."""
    path = tw.STRANDED_COMMITS_LOG
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _landing_records(caplog) -> list[logging.LogRecord]:
    """All captured log records emitted by the landing check (case-insensitive:
    the stranded ERROR says "LANDING CHECK", the skip-path + fail-open warnings
    say "landing check" — the silence tests must see all of them)."""
    return [r for r in caplog.records if "landing check" in r.getMessage().lower()]


def _strand_repo(repo: Path) -> None:
    """Park the fake repo's checkout on a feature branch. The fixture's
    monkeypatched `repo_root` bypasses the branch-guard resolver, so a
    subsequent `_git_commit` lands OFF main — exactly the guard-escape class
    the landing check (#1100) exists to catch."""
    subprocess.run(["git", "checkout", "-q", "-b", "issue-42"], cwd=repo, check=True)


def test_landing_check_silent_when_commit_reaches_main(fake_repo, caplog):
    """AC-2: a commit that lands on `main` (sha == main tip) produces no
    LANDING CHECK record and no sidecar file at all."""
    repo, tw = fake_repo
    target = repo / "somefile.txt"
    target.write_text("x\n")
    before = _git_log_count(repo)
    with caplog.at_level(logging.WARNING, logger="explore_persona_space.task_workflow"):
        tw._git_commit([target], "landing silent probe")
    assert _git_log_count(repo) == before + 1
    assert _landing_records(caplog) == []
    assert not tw.STRANDED_COMMITS_LOG.exists()


def test_landing_check_warns_on_stranded_commit(fake_repo, caplog):
    """AC-1: a commit unreachable from refs/heads/main → the mutation still
    succeeds, exactly ONE ERROR names the sha[:12] + the greppable phrase +
    the HEAD ref + the sidecar path, and exactly one `kind: stranded` row
    with the exact sidecar schema lands in the sidecar."""
    repo, tw = fake_repo
    _strand_repo(repo)
    target = repo / "somefile.txt"
    target.write_text("x\n")
    before = _git_log_count(repo)
    with caplog.at_level(logging.ERROR, logger="explore_persona_space.task_workflow"):
        tw._git_commit([target], "stranded probe")  # returns normally (warn-only)
    # The mutation itself succeeded: the commit exists (on issue-42).
    assert _git_log_count(repo) == before + 1
    sha = _head_sha(repo).strip()
    records = _landing_records(caplog)
    assert len(records) == 1, [r.getMessage() for r in caplog.records]
    assert records[0].levelno == logging.ERROR
    msg = records[0].getMessage()
    assert sha[:12] in msg
    assert "NOT reachable from refs/heads/main" in msg
    assert "issue-42" in msg
    assert str(tw.STRANDED_COMMITS_LOG) in msg
    rows = _stranded_rows(tw)
    assert len(rows) == 1
    row = rows[0]
    assert set(row) == {
        "ts",
        "kind",
        "sha",
        "head_ref",
        "routed",
        "message",
        "probe_rc",
        "probe_stderr_tail",
    }
    assert row["kind"] == "stranded"
    assert row["head_ref"] == "issue-42"
    assert row["sha"] == sha
    assert row["routed"] is False
    assert row["probe_rc"] == 1
    assert row["message"].startswith("stranded probe")
    # (probe.stderr or "")[-300:] — a str by construction, may legitimately be empty.
    assert isinstance(row["probe_stderr_tail"], str)


def test_landing_check_never_fails_the_mutation(fake_repo, monkeypatch, caplog):
    """AC-3 fail-open is total: a sidecar-write OSError degrades to a warning
    and _git_commit returns normally; a _run_git blow-up INSIDE the helper is
    swallowed too (returns None, never raises)."""
    repo, tw = fake_repo
    _strand_repo(repo)
    target = repo / "somefile.txt"
    target.write_text("x\n")

    def _oserror(path, payload):
        raise OSError("disk full")

    monkeypatch.setattr(tw, "_append_jsonl_line", _oserror)
    before = _git_log_count(repo)
    with caplog.at_level(logging.WARNING, logger="explore_persona_space.task_workflow"):
        tw._git_commit([target], "fail-open probe")  # must NOT raise
    assert _git_log_count(repo) == before + 1  # the commit landed
    assert any("fail-open" in r.getMessage() for r in caplog.records)

    # Second leg: the check's own git plumbing exploding is swallowed too.
    def _boom(args, *, check=True):
        raise RuntimeError("git exploded")

    monkeypatch.setattr(tw, "_run_git", _boom)
    assert tw._warn_if_commit_stranded("m", routed=False) is None  # no raise


def test_landing_check_unverifiable_when_main_ref_missing(fake_repo):
    """§1 item 4: a repo with no refs/heads/main at all → a
    `kind: unverifiable` row (probe rc != 1), never a crash."""
    repo, tw = fake_repo
    subprocess.run(["git", "branch", "-m", "main", "trunk"], cwd=repo, check=True)
    target = repo / "somefile.txt"
    target.write_text("x\n")
    before = _git_log_count(repo)
    tw._git_commit([target], "unverifiable probe")  # must not raise
    assert _git_log_count(repo) == before + 1
    rows = _stranded_rows(tw)
    assert len(rows) == 1
    assert rows[0]["kind"] == "unverifiable"
    assert rows[0]["probe_rc"] != 1
    assert rows[0]["head_ref"] == "trunk"


def test_landing_check_silent_on_ancestor_of_moved_main(fake_repo, caplog):
    """AC-2 moved-main subcase: HEAD = A, refs/heads/main = B, A a strict
    ancestor of B → silent. This is the SOLE discriminator between the chosen
    `merge-base --is-ancestor` reachability predicate and a tip-equality
    implementation (`rev-parse main == HEAD` would spuriously warn here)."""
    repo, tw = fake_repo
    (repo / "a.txt").write_text("a\n")
    subprocess.run(["git", "add", "a.txt"], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-q", "-m", "A"], cwd=repo, check=True)
    sha_a = _head_sha(repo).strip()
    (repo / "b.txt").write_text("b\n")
    subprocess.run(["git", "add", "b.txt"], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-q", "-m", "B"], cwd=repo, check=True)
    # Detach at A: HEAD is now a strict ancestor of the moved main (= B).
    subprocess.run(["git", "checkout", "-q", "--detach", sha_a], cwd=repo, check=True)
    with caplog.at_level(logging.WARNING, logger="explore_persona_space.task_workflow"):
        tw._warn_if_commit_stranded("moved-main probe", routed=False)
    assert _landing_records(caplog) == []
    assert not tw.STRANDED_COMMITS_LOG.exists()


def test_landing_check_disabled_via_env(fake_repo, monkeypatch, caplog):
    """AC-5: EPM_TASKPY_LANDING_CHECK=0 disables the check entirely — no
    ERROR + no row on a stranded end-to-end commit, and the direct-call leg
    issues ZERO git subprocesses."""
    repo, tw = fake_repo
    monkeypatch.setenv("EPM_TASKPY_LANDING_CHECK", "0")
    _strand_repo(repo)
    target = repo / "somefile.txt"
    target.write_text("x\n")
    before = _git_log_count(repo)
    with caplog.at_level(logging.WARNING, logger="explore_persona_space.task_workflow"):
        tw._git_commit([target], "disabled probe")
    assert _git_log_count(repo) == before + 1
    assert _landing_records(caplog) == []
    assert not tw.STRANDED_COMMITS_LOG.exists()

    # Direct-call leg (AC-5 "no probe subprocesses"): zero _run_git calls.
    calls: list[list[str]] = []
    monkeypatch.setattr(tw, "_run_git", lambda args, *, check=True: calls.append(list(args)))
    tw._warn_if_commit_stranded("m", routed=False)
    assert calls == []


def test_landing_check_skipped_under_no_commit(fake_repo, monkeypatch):
    """AC-6: TASK_PY_NO_COMMIT=1 early-returns at the top of _git_commit —
    zero git subprocesses, so the landing check can never fire in
    test/no-commit mode."""
    repo, tw = fake_repo
    monkeypatch.setenv("TASK_PY_NO_COMMIT", "1")
    target = repo / "somefile.txt"
    target.write_text("x\n")
    calls: list[list[str]] = []
    monkeypatch.setattr(tw, "_run_git", lambda args, *, check=True: calls.append(list(args)))
    tw._git_commit([target], "no-commit probe")
    assert calls == []
    assert not tw.STRANDED_COMMITS_LOG.exists()


def test_cli_post_marker_deferred_commit_exits_clean(fake_repo, monkeypatch, capsys):
    """AC1 CLI leg: cmd_post_event returns without raising (rc-0 contract)
    and echoes the payload when the bookkeeping commit was deferred."""
    _, tw = fake_repo
    new_id = tw.create_task(tw.NewTaskRequest(kind="experiment", title="X"))
    monkeypatch.setattr(tw, "_git_commit", _commit_crash)
    task_cli = _load_task_cli()
    args = argparse.Namespace(
        number=new_id,
        marker="epm:progress",
        version=None,
        by="tester",
        note="cli deferred probe",
        file=None,
    )

    task_cli.cmd_post_event(args)  # must not raise — the handler's rc stays 0

    out = capsys.readouterr().out
    assert "epm:progress" in out  # payload echoed
    assert [r["op"] for r in _deferred_rows(tw)] == ["post_event"]


@pytest.mark.parametrize("which", ["merge", "cherry-pick"])
def test_git_commit_toctou_sequencer_retry(fake_repo, monkeypatch, which):
    """TOCTOU closure: the FIRST commit fatals with the sequencer signature,
    the handler RE-WAITS (wait invoked exactly TWICE total — deleting the
    re-wait fails this) and the single FINAL retry succeeds — exactly two
    commit attempts total."""
    repo, tw = fake_repo
    target = repo / "somefile.txt"
    target.write_text("x\n")
    commit_calls: list[list[str]] = []
    real_run_git = tw._run_git

    def toctou_run_git(args, *, check=True):
        if args and args[0] == "commit":
            commit_calls.append(list(args))
            if len(commit_calls) == 1:
                raise subprocess.CalledProcessError(
                    128,
                    ["git", *args],
                    output="",
                    stderr=f"fatal: cannot do a partial commit during a {which}.\n",
                )
        return real_run_git(args, check=check)

    monkeypatch.setattr(tw, "_run_git", toctou_run_git)
    wait_calls: list[Path] = []
    real_wait = tw._wait_for_sequencer_clear

    def counting_wait(repo_arg):
        wait_calls.append(repo_arg)
        return real_wait(repo_arg)

    monkeypatch.setattr(tw, "_wait_for_sequencer_clear", counting_wait)
    before = _git_log_count(repo)
    tw._git_commit([target], "toctou probe")
    assert len(wait_calls) == 2  # pre-staging + the TOCTOU re-wait
    assert len(commit_calls) == 2  # first fatal + FINAL success
    assert _git_log_count(repo) == before + 1


def test_sequencer_wait_negative_primary_merge_does_not_stall_worktree(fake_repo, monkeypatch):
    """Sequencer state is per-worktree: a merge on the PRIMARY checkout must
    NOT stall commits in a linked worktree (zero sleeps)."""
    repo, tw = fake_repo
    wt = repo.parent / f"{repo.name}-linked-wt"
    subprocess.run(
        ["git", "worktree", "add", "--detach", str(wt)], cwd=repo, check=True, capture_output=True
    )
    (repo / ".git" / "MERGE_HEAD").write_text(_head_sha(repo))  # merge on the PRIMARY
    monkeypatch.setattr(tw, "repo_root", lambda: wt)
    monkeypatch.setattr(tw.time, "sleep", lambda d: pytest.fail("stalled on the primary's merge"))
    tw._wait_for_sequencer_clear(wt)  # returns immediately


@pytest.mark.parametrize("state_file", ["MERGE_HEAD", "CHERRY_PICK_HEAD"])
def test_sequencer_wait_positive_in_linked_worktree(fake_repo, monkeypatch, state_file):
    """MF-2 discriminating cell: a merge IN the linked worktree itself IS seen
    via `rev-parse --git-path` (the state lives under
    <primary>/.git/worktrees/<name>/). A mutant hardcoding <root>/.git/<state>
    sees nothing there — the worktree's .git is a FILE — takes zero sleeps,
    and FAILs this test."""
    repo, tw = fake_repo
    wt = repo.parent / f"{repo.name}-wt-{state_file.lower()}"
    subprocess.run(
        ["git", "worktree", "add", "--detach", str(wt)], cwd=repo, check=True, capture_output=True
    )
    state_raw = subprocess.run(
        ["git", "-C", str(wt), "rev-parse", "--git-path", state_file],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    state = Path(state_raw) if Path(state_raw).is_absolute() else wt / state_raw
    assert "worktrees" in str(state)  # sanity: the per-worktree location
    state.parent.mkdir(parents=True, exist_ok=True)
    state.write_text(_head_sha(repo))
    monkeypatch.setattr(tw, "repo_root", lambda: wt)
    sleeps: list[float] = []

    def clearing_sleep(delay: float) -> None:
        sleeps.append(delay)
        state.unlink()  # the worktree's own merge finishes during the sleep

    monkeypatch.setattr(tw.time, "sleep", clearing_sleep)
    tw._wait_for_sequencer_clear(wt)
    assert len(sleeps) >= 1  # the worktree's own merge WAS seen


def test_post_event_sequencer_timeout_defers_on_primary(fake_repo, monkeypatch):
    """Seams (a) x (b) integration: a held MERGE_HEAD + tiny bound on the
    PRIMARY -> post_event still returns the payload (append durable, commit
    deferred); the sidecar row records error == SequencerWaitTimeout."""
    repo, tw = fake_repo
    new_id = tw.create_task(tw.NewTaskRequest(kind="experiment", title="X"))
    (repo / ".git" / "MERGE_HEAD").write_text(_head_sha(repo))  # never clears
    monkeypatch.setenv("EPM_TASKPY_MERGE_WAIT_SECONDS", "0.05")
    monkeypatch.setenv("EPM_TASKPY_MERGE_POLL_SECONDS", "0.01")

    payload = tw.post_event(new_id, "epm:progress", note="merge-held probe")

    assert payload["kind"] == "epm:progress"
    rows = _deferred_rows(tw)
    assert len(rows) == 1
    assert rows[0]["error"] == "SequencerWaitTimeout"
    assert rows[0]["op"] == "post_event"


@pytest.mark.parametrize("bad", ["nan", "inf"])
def test_merge_wait_knob_rejects_non_finite(fake_repo, monkeypatch, bad):
    """Knob validation mirrors #996: a non-finite env value raises ValueError
    (nan would defeat the monotonic deadline comparison and wait unbounded)."""
    _, tw = fake_repo
    monkeypatch.setenv("EPM_TASKPY_MERGE_WAIT_SECONDS", bad)
    with pytest.raises(ValueError, match="EPM_TASKPY_MERGE_WAIT_SECONDS"):
        tw._merge_wait_bound_s()
    monkeypatch.setenv("EPM_TASKPY_MERGE_POLL_SECONDS", bad)
    with pytest.raises(ValueError, match="EPM_TASKPY_MERGE_POLL_SECONDS"):
        tw._merge_poll_s()
