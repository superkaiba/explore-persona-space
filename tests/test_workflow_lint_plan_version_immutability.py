"""Tests for ``workflow_lint --check-plan-version-immutability`` (#2123).

A persisted ``tasks/**/plans/v<K>.md`` plan version is IMMUTABLE — an
amendment goes through ``task.py new-plan-version`` (a NEW ``v<K+1>.md``),
never an in-place edit. Two arms:

* **Arm W (working tree + index)** — bundled into the no-flags default
  run. ``M``/``D`` in EITHER porcelain column, or a staged rename, FAILs.
  The index-column case (modify + ``git add``, no commit — porcelain
  ``M `` with a BLANK worktree column) is the plan #2123 critic-blocker-1
  regression case: a worktree-column-only predicate would be blind to
  exactly the pre-commit-window state the lint runs in (Step 9c /
  Step 10d pre-push), letting the mutation land in history.
* **Arm H (committed history)** — explicit-flag only (the plan #2123 §6
  cost fallback; measured ~1.7-2.7 s on the shared VM). ``M`` or
  ``R<100`` FAILs; a pure status-folder move is ``R100`` and CLEAN (the
  critical no-false-positive case — status moves rename EVERY plan file).

Fixtures build a real throwaway git repo under ``tmp_path`` (identity via
``-c`` flags + scrubbed GIT_* env, the ``workflow_lint`` root-guard fixture
convention) and call ``check_plan_version_immutability(repo_root=...)``
directly; the no-flags DISPATCH test monkeypatches ``wl._REPO_ROOT`` (the
mutation-visible ``test_check_jsonl_splitlines_bundled_in_no_flags``
pattern).
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

_HERE = Path(__file__).resolve().parent
_SCRIPTS = _HERE.parent / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

import workflow_lint as wl  # noqa: E402
from workflow_lint import check_plan_version_immutability  # noqa: E402

# Isolate user/system git config (ambient commit.gpgsign / hooks templates
# would break the fixture build — the root-guard fixture convention).
_GIT_ENV = {k: v for k, v in os.environ.items() if not k.startswith("GIT_")}
_GIT_ENV["GIT_CONFIG_GLOBAL"] = "/dev/null"
_GIT_ENV["GIT_CONFIG_NOSYSTEM"] = "1"

PLAN_REL = "tasks/planning/1/plans/v1.md"


def _git(root: Path, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [
            "git",
            "-C",
            str(root),
            "-c",
            "user.name=pvi-test",
            "-c",
            "user.email=pvi-test@localhost",
            "-c",
            "commit.gpgsign=false",
            *args,
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=60,
        env=_GIT_ENV,
    )


def _repo_with_persisted_plan(root: Path) -> Path:
    """git init + commit one plan version file; returns the plan path."""
    subprocess.run(
        ["git", "init", "-q", "-b", "main", str(root)],
        check=True,
        capture_output=True,
        text=True,
        timeout=60,
        env=_GIT_ENV,
    )
    plan = root / PLAN_REL
    plan.parent.mkdir(parents=True, exist_ok=True)
    plan.write_text("# Plan v1\n\nOriginal persisted plan body.\n", encoding="utf-8")
    _git(root, "add", PLAN_REL)
    _git(root, "commit", "-qm", "task #1: plan v1")
    return plan


# ── clean baseline ──────────────────────────────────────────────────────


def test_persisted_plan_clean_passes_both_arms(tmp_path):
    _repo_with_persisted_plan(tmp_path)
    errors = check_plan_version_immutability(repo_root=tmp_path, include_history=True)
    assert errors == [], f"expected PASS on a clean persisted plan, got: {errors}"


def test_untracked_new_version_is_clean(tmp_path):
    """A NEW not-yet-committed v2.md is the sanctioned amendment path —
    untracked files never flag."""
    plan = _repo_with_persisted_plan(tmp_path)
    (plan.parent / "v2.md").write_text("# Plan v2\n\nAmendment.\n", encoding="utf-8")
    errors = check_plan_version_immutability(repo_root=tmp_path, include_history=True)
    assert errors == [], f"expected PASS (untracked new version), got: {errors}"


def test_staged_new_version_is_clean(tmp_path):
    """A staged pure ADD of a new version file is clean (porcelain ``A ``)."""
    plan = _repo_with_persisted_plan(tmp_path)
    (plan.parent / "v2.md").write_text("# Plan v2\n\nAmendment.\n", encoding="utf-8")
    _git(tmp_path, "add", str(plan.parent / "v2.md"))
    errors = check_plan_version_immutability(repo_root=tmp_path)
    assert errors == [], f"expected PASS (staged pure add), got: {errors}"


# ── Arm W: working tree + index ─────────────────────────────────────────


def test_unstaged_modify_fails_arm_w(tmp_path):
    plan = _repo_with_persisted_plan(tmp_path)
    plan.write_text("# Plan v1\n\nEDITED in place.\n", encoding="utf-8")
    errors = check_plan_version_immutability(repo_root=tmp_path)
    assert len(errors) == 1, f"expected one Arm-W error, got: {errors}"
    assert PLAN_REL in errors[0]
    assert "working tree" in errors[0]
    assert "new-plan-version" in errors[0]


def test_staged_modify_no_commit_fails_arm_w(tmp_path):
    """The index-column case (plan #2123 critic blocker 1): modify +
    ``git add`` with NO commit is porcelain ``M `` — index column ``M``,
    worktree column BLANK. A worktree-column-only predicate misses it,
    which is exactly the state present at a Step 9c / Step 10d no-flags
    run in the moments before the violating commit."""
    plan = _repo_with_persisted_plan(tmp_path)
    plan.write_text("# Plan v1\n\nEDITED and staged.\n", encoding="utf-8")
    _git(tmp_path, "add", PLAN_REL)
    errors = check_plan_version_immutability(repo_root=tmp_path)
    assert len(errors) == 1, f"expected one Arm-W error (staged edit), got: {errors}"
    assert PLAN_REL in errors[0]
    assert "'M '" in errors[0] or "M " in errors[0]


def test_unstaged_delete_fails_arm_w(tmp_path):
    plan = _repo_with_persisted_plan(tmp_path)
    plan.unlink()
    errors = check_plan_version_immutability(repo_root=tmp_path)
    assert len(errors) == 1, f"expected one Arm-W error (deletion), got: {errors}"
    assert PLAN_REL in errors[0]


def test_staged_rename_fails_arm_w(tmp_path):
    """A staged-but-uncommitted rename of a plan version file (index
    column ``R``) is a mutation-adjacent violation — renaming a persisted
    version is not the amendment path."""
    plan = _repo_with_persisted_plan(tmp_path)
    _git(tmp_path, "mv", PLAN_REL, str(Path(PLAN_REL).parent / "v9.md"))
    del plan
    errors = check_plan_version_immutability(repo_root=tmp_path)
    assert len(errors) == 1, f"expected one Arm-W error (staged rename), got: {errors}"


def test_staged_new_version_edited_before_commit_is_clean(tmp_path):
    """The ``AM`` case (#2123 round-1 review finding): a NEW version file
    that is ``git add``-ed and then edited again before committing is
    porcelain ``AM``. Nothing is persisted until a commit exists, so this
    is a v<K+1> still being drafted — the sanctioned amendment path, not
    a mutation. A ``y in "MD"``-only predicate false-FAILs it, which would
    block the fleet-gating no-flags default run for anyone hand-authoring
    a new plan version."""
    _repo_with_persisted_plan(tmp_path)
    new_version = tmp_path / Path(PLAN_REL).parent / "v2.md"
    new_version.write_text("# Plan v2\n\nFirst draft.\n", encoding="utf-8")
    _git(tmp_path, "add", str(Path(PLAN_REL).parent / "v2.md"))
    new_version.write_text("# Plan v2\n\nSecond draft, edited after add.\n", encoding="utf-8")
    errors = check_plan_version_immutability(repo_root=tmp_path)
    assert errors == [], f"expected PASS on a staged-then-edited NEW version, got: {errors}"


def test_staged_status_folder_move_is_clean_arm_w(tmp_path):
    """A staged ``tasks/<old-status>/<N>`` -> ``tasks/<new-status>/<N>``
    move is CLEAN (#2123 round-1 review finding). This is exactly what
    ``task.py set-status`` stages between its ``git mv`` and its commit,
    and Arm W probes the REPO ROOT even when invoked from a worktree — so
    without this exemption a Step 9c / Step 10d gate would false-FAIL on a
    DIFFERENT session's in-flight status transition. Arm H already treats
    the committed form (``R100``) as clean; the two arms must agree."""
    _repo_with_persisted_plan(tmp_path)
    (tmp_path / "tasks" / "approved").mkdir(parents=True, exist_ok=True)
    _git(tmp_path, "mv", "tasks/planning/1", "tasks/approved/1")
    errors = check_plan_version_immutability(repo_root=tmp_path)
    assert errors == [], f"expected PASS on a staged status-folder move, got: {errors}"


def test_staged_status_folder_move_with_STAGED_edit_fails_arm_w(tmp_path):
    """The #2123 round-2 review blocker, reproduced by execution: a
    status-move-shaped rename whose content edit is ALSO STAGED is
    porcelain ``R `` — a BLANK worktree column — while the index blob
    differs from the HEAD blob. No porcelain column can see this (the
    worktree column compares worktree-vs-index, never index-vs-HEAD) and
    the rename similarity score is not exposed, so a path-shape-only
    exemption lets a real mutation through; it then lands in history,
    where Arm H reports it as ``R058`` — and Arm H is explicit-flag-only,
    so the fleet gate never catches it post-commit. Only the
    HEAD-blob-vs-index-blob comparison distinguishes the two cases."""
    _repo_with_persisted_plan(tmp_path)
    plan = tmp_path / PLAN_REL
    # A LONG body is load-bearing for this fixture: git pairs a rename only at
    # >= 50% similarity, so a one-line edit to a long file yields the `R `
    # escape window under test, whereas rewriting a short file degrades to
    # `D `+`A ` — which the D arm already catches (fail-closed, and NOT the
    # case this test exists for).
    body = "# Plan v1\n\n" + "\n".join(f"- design point {i}" for i in range(200)) + "\n"
    plan.write_text(body, encoding="utf-8")
    _git(tmp_path, "add", PLAN_REL)
    _git(tmp_path, "commit", "-qm", "task #1: plan v1 (long body)")
    (tmp_path / "tasks" / "approved").mkdir(parents=True, exist_ok=True)
    _git(tmp_path, "mv", "tasks/planning/1", "tasks/approved/1")
    moved_rel = "tasks/approved/1/plans/v1.md"
    (tmp_path / moved_rel).write_text(
        body.replace("- design point 7\n", "- design point 7 MUTATED\n"), encoding="utf-8"
    )
    _git(tmp_path, "add", moved_rel)
    # Precondition: the state really is a bare `R ` with a blank worktree column.
    porcelain = _git(tmp_path, "status", "--porcelain=v1", "--", "tasks").stdout
    assert porcelain.startswith("R "), f"fixture did not produce a bare 'R ': {porcelain!r}"
    errors = check_plan_version_immutability(repo_root=tmp_path)
    assert len(errors) == 1, f"expected one Arm-W error (staged edit inside a move), got: {errors}"


def test_staged_status_folder_move_with_edit_fails_arm_w(tmp_path):
    """The status-move exemption must NOT swallow a content change: a
    status move whose file is then edited in the working tree is porcelain
    ``RM``, and the worktree-column ``M`` still fires. This is the guard
    that keeps the exemption narrow."""
    _repo_with_persisted_plan(tmp_path)
    (tmp_path / "tasks" / "approved").mkdir(parents=True, exist_ok=True)
    _git(tmp_path, "mv", "tasks/planning/1", "tasks/approved/1")
    moved = tmp_path / "tasks" / "approved" / "1" / "plans" / "v1.md"
    moved.write_text("# Plan v1\n\nEDITED after the move.\n", encoding="utf-8")
    errors = check_plan_version_immutability(repo_root=tmp_path)
    assert len(errors) == 1, f"expected one Arm-W error (RM), got: {errors}"


# ── Arm H: committed history ────────────────────────────────────────────


def test_committed_modify_fails_arm_h_only(tmp_path):
    plan = _repo_with_persisted_plan(tmp_path)
    plan.write_text("# Plan v1\n\nEDITED and committed.\n", encoding="utf-8")
    _git(tmp_path, "add", PLAN_REL)
    _git(tmp_path, "commit", "-qm", "task #1: illegal in-place amendment")
    # Arm H (explicit flag) catches the committed mutation ...
    errors = check_plan_version_immutability(repo_root=tmp_path, include_history=True)
    assert len(errors) == 1, f"expected one Arm-H error, got: {errors}"
    assert PLAN_REL in errors[0]
    assert "MODIFIED in commit" in errors[0]
    # ... which the working tree (now clean) cannot see — the documented
    # flag-gating consequence (plan #2123 §6 fallback).
    assert check_plan_version_immutability(repo_root=tmp_path) == []


def test_pure_status_folder_rename_r100_passes_arm_h(tmp_path):
    """The critical no-false-positive case: a status-folder move
    (``git mv tasks/planning/1 tasks/running/1``) renames every plan file
    with IDENTICAL content — rename detection reports R100, which is
    CLEAN. Without ``--find-renames`` semantics (e.g. ``--no-renames``)
    every status move would read as D+A and per-path add-resolution picks
    the wrong commit (the plan #2123 §3.1 rename-following rationale)."""
    _repo_with_persisted_plan(tmp_path)
    (tmp_path / "tasks/running").mkdir(parents=True)
    _git(tmp_path, "mv", "tasks/planning/1", "tasks/running/1")
    _git(tmp_path, "commit", "-qm", "task #1: planning -> running")
    errors = check_plan_version_immutability(repo_root=tmp_path, include_history=True)
    assert errors == [], f"expected PASS (pure R100 status move), got: {errors}"


def test_rename_with_content_change_fails_arm_h(tmp_path):
    """An R<100 — rename WITH a content change — is an in-place mutation
    smuggled through a rename."""
    plan = _repo_with_persisted_plan(tmp_path)
    # Move the status folder AND edit the plan in the same commit: git's
    # rename detection pairs old->new at similarity < 100.
    (tmp_path / "tasks/running").mkdir(parents=True)
    _git(tmp_path, "mv", "tasks/planning/1", "tasks/running/1")
    moved = tmp_path / "tasks/running/1/plans/v1.md"
    moved.write_text(
        "# Plan v1\n\nOriginal persisted plan body.\nEDITED during the move.\n",
        encoding="utf-8",
    )
    del plan
    _git(tmp_path, "add", "-A")
    _git(tmp_path, "commit", "-qm", "task #1: move + sneak edit")
    errors = check_plan_version_immutability(repo_root=tmp_path, include_history=True)
    assert len(errors) == 1, f"expected one Arm-H R<100 error, got: {errors}"
    assert "renamed WITH content change" in errors[0]


def test_history_arm_not_run_by_default(tmp_path):
    """``include_history`` defaults False — the no-flags default run
    carries Arm W only (the plan #2123 §6 cost fallback: Arm H measured
    at the ~3 s threshold under load)."""
    plan = _repo_with_persisted_plan(tmp_path)
    plan.write_text("# Plan v1\n\nEDITED and committed.\n", encoding="utf-8")
    _git(tmp_path, "add", PLAN_REL)
    _git(tmp_path, "commit", "-qm", "task #1: illegal in-place amendment")
    assert check_plan_version_immutability(repo_root=tmp_path) == []


# ── escape hatches ──────────────────────────────────────────────────────


def test_allowlist_suppresses_both_arms(tmp_path, monkeypatch):
    plan = _repo_with_persisted_plan(tmp_path)
    plan.write_text("# Plan v1\n\nEDITED (sanctioned scrub).\n", encoding="utf-8")
    _git(tmp_path, "add", PLAN_REL)
    _git(tmp_path, "commit", "-qm", "task #1: sanctioned secret scrub")
    plan.write_text("# Plan v1\n\nEDITED again, dirty.\n", encoding="utf-8")
    monkeypatch.setattr(wl, "PLAN_IMMUTABILITY_ALLOWLIST", frozenset({PLAN_REL}))
    errors = check_plan_version_immutability(repo_root=tmp_path, include_history=True)
    assert errors == [], f"expected allowlist suppression, got: {errors}"


def test_kill_switch_suppresses(tmp_path, monkeypatch, capsys):
    plan = _repo_with_persisted_plan(tmp_path)
    plan.write_text("# Plan v1\n\nEDITED in place.\n", encoding="utf-8")
    monkeypatch.setenv("EPM_SKIP_PLAN_IMMUTABILITY_CHECK", "1")
    errors = check_plan_version_immutability(repo_root=tmp_path, include_history=True)
    assert errors == []
    assert "DISABLED" in capsys.readouterr().err


def test_non_git_root_fails_open_with_notice(tmp_path, capsys):
    """A git failure (not a repo / git missing) fail-opens with ONE loud
    stderr notice per arm — never a crash, never a silent skip. The real
    Step 9c / Step 10d gates always run inside a git checkout."""
    (tmp_path / "tasks/planning/1/plans").mkdir(parents=True)
    (tmp_path / PLAN_REL).write_text("# Plan v1\n", encoding="utf-8")
    errors = check_plan_version_immutability(repo_root=tmp_path, include_history=True)
    assert errors == []
    err = capsys.readouterr().err
    assert "--check-plan-version-immutability skipped a git arm" in err


# ── no-flags DISPATCH (mutation-visible bundling test) ──────────────────


def test_check_plan_version_immutability_bundled_in_no_flags(tmp_path, capsys, monkeypatch):
    """The no-flags default run actually DISPATCHES Arm W — deleting its
    ``or no_flags`` branch must fail this test (the
    ``test_check_jsonl_splitlines_bundled_in_no_flags`` mutation-visible
    pattern). Other bundled checks contribute unrelated errors on the
    minimal tree, so the assertion keys on this check's own diagnostic
    token + the offending path."""
    plan = _repo_with_persisted_plan(tmp_path)
    plan.write_text("# Plan v1\n\nEDITED in place.\n", encoding="utf-8")
    monkeypatch.setattr(wl, "_REPO_ROOT", tmp_path)
    rc = wl.main([])
    err = capsys.readouterr().err
    assert rc != 0, f"no-flags default run exited 0 on a mutated-plan tree:\n{err}"
    assert "persisted plan version mutated" in err and "v1.md" in err, (
        f"the plan-version-immutability diagnostic (naming v1.md) is missing "
        f"from the no-flags default run's stderr — Arm W is not bundled into "
        f"no_flags:\n{err}"
    )


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
