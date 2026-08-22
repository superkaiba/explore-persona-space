"""Tests for the create_task allocation-site registry-drift self-heal (#2064).

Registry drift = an on-disk ``tasks/<status>/<id>/`` folder that
``tasks/REGISTRY.json`` does not know about (e.g. a registry write destroyed
by the #2015 pre-commit stash race). Pre-#2064 the id allocator re-issued the
same colliding id to EVERY ``task.py new`` caller and each one died with a
bare ``FileExistsError``, taking ``scripts/file_infra_task.py`` down
fleet-wide. These tests pin the plan's behavioral contract (plan v2 §3):

- A: missing_real orphan is registered in-lock and the create succeeds at the
  next free id, in the create's ONE existing commit (single-commit invariant).
- B: a bodyless empty stub is bumped past WITHOUT registration or deletion.
- C: an un-healable second collision raises RuntimeError naming
  ``audit --repair --apply`` — never a bare FileExistsError.
- D: ``_save_registry``'s best-effort stage (#2015 window narrowing) stages
  REGISTRY.json in a git repo and degrades to a WARN in a non-git tree.

Reuses the ``fake_repo`` fixture + ``_git_log_count`` from
``tests/test_task_workflow.py`` (tests/ is a package).
"""

from __future__ import annotations

import json
import logging
import subprocess
from pathlib import Path

import pytest

import tests.test_task_workflow as _ttw

# Genuine reuse of the shared fake-repo fixture + commit counter (module-attr
# re-export rather than a `from ... import`, so pytest still discovers the
# fixture while ruff's F811 does not fire on the test parameters).
fake_repo = _ttw.fake_repo
_git_log_count = _ttw._git_log_count

ORPHAN_BODY = """---
title: Orphaned by the stash race
kind: infra
tags: []
has_clean_result: false
---

Body of the orphaned task (readable frontmatter, so the heal registers it).
"""


def _make_orphan_dir(repo: Path, status: str, task_id: int, *, with_body: bool) -> Path:
    """Create tasks/<status>/<task_id>/ on disk WITHOUT a registry entry."""
    d = repo / "tasks" / status / str(task_id)
    d.mkdir(parents=True, exist_ok=False)
    if with_body:
        (d / "body.md").write_text(ORPHAN_BODY)
    return d


# ─── A: missing_real heal ───────────────────────────────────────────────────


def test_alloc_drift_missing_real_heals_and_creates(fake_repo, caplog):
    repo, tw = fake_repo
    # Establish a real registry state first: task 1 exists + is registered.
    assert tw.create_task(tw.NewTaskRequest(kind="experiment", title="First")) == 1
    # Orphan at the NEXT allocation id (2): on disk with a readable body.md,
    # absent from REGISTRY.json — the #2052 drift shape.
    orphan = _make_orphan_dir(repo, "proposed", 2, with_body=True)
    orphan_bytes = (orphan / "body.md").read_bytes()

    commits_before = _git_log_count(repo)
    with caplog.at_level(logging.ERROR, logger="explore_persona_space.task_workflow"):
        new_id = tw.create_task(tw.NewTaskRequest(kind="experiment", title="After drift"))

    # The create succeeded at the next genuinely-free id.
    assert new_id == 3
    assert (repo / "tasks" / "proposed" / "3" / "body.md").exists()

    # The orphan was registered (missing_real policy) with its own snapshot,
    # and its folder was left untouched.
    reg = json.loads((repo / "tasks" / "REGISTRY.json").read_text())
    assert reg["tasks"]["2"]["path"] == "tasks/proposed/2"
    assert reg["tasks"]["2"]["title"] == "Orphaned by the stash race"
    assert reg["tasks"]["2"]["kind"] == "infra"
    assert (orphan / "body.md").read_bytes() == orphan_bytes
    assert reg["highest_id"] == 3

    # Single-commit invariant (hypothesis 4): the heal rode the create's ONE
    # commit — no separate reconcile commit.
    assert _git_log_count(repo) == commits_before + 1
    last_msg = subprocess.run(
        ["git", "log", "-1", "--format=%s"], cwd=repo, capture_output=True, text=True, check=True
    ).stdout
    assert "(+registry drift heal, #2064)" in last_msg

    # The drift stays VISIBLE: ERROR-level heal log reports the APPLIED counts.
    drift_lines = [r.getMessage() for r in caplog.records if "registry drift" in r.getMessage()]
    heal_lines = [
        r.getMessage() for r in caplog.records if r.getMessage().startswith("drift heal:")
    ]
    assert drift_lines, "expected the ERROR-level drift-detected log line"
    assert heal_lines, "expected the ERROR-level heal-summary log line"
    assert "1 missing registered" in heal_lines[0]
    assert "0 stale re-pointed" in heal_lines[0]


# ─── B: empty_stub bump ─────────────────────────────────────────────────────


def test_alloc_drift_empty_stub_bumped_past_not_registered(fake_repo):
    repo, tw = fake_repo
    # Bodyless stub at the first allocation id (1). Registry is empty.
    stub = _make_orphan_dir(repo, "proposed", 1, with_body=False)

    new_id = tw.create_task(tw.NewTaskRequest(kind="experiment", title="Past the stub"))

    # Allocated PAST the stub; the stub is neither registered nor deleted.
    assert new_id == 2
    reg = json.loads((repo / "tasks" / "REGISTRY.json").read_text())
    assert "1" not in reg["tasks"]
    assert reg["tasks"]["2"]["path"] == "tasks/proposed/2"
    assert reg["highest_id"] == 2
    assert stub.is_dir()
    assert not (stub / "body.md").exists()


# ─── C: fail-loud fallback (two un-healable variants) ───────────────────────


def test_alloc_drift_second_collision_raises_runtime_error(fake_repo, monkeypatch):
    """Variant (i): heal makes no progress (highest_id bump monkeypatched to a
    no-op alongside a bodyless stub) -> RuntimeError naming the repair command,
    chained from the FileExistsError -- never a bare FileExistsError."""
    repo, tw = fake_repo
    _make_orphan_dir(repo, "proposed", 1, with_body=False)
    monkeypatch.setattr(tw, "_reconcile_highest_id", lambda reg, max_disk_id=0: None)

    with pytest.raises(RuntimeError, match=r"audit --repair --apply") as excinfo:
        tw.create_task(tw.NewTaskRequest(kind="experiment", title="Doomed"))
    assert isinstance(excinfo.value.__cause__, FileExistsError)


def test_alloc_drift_stray_file_at_retry_path_raises_runtime_error(fake_repo):
    """Variant (ii), no monkeypatching: a stray regular FILE at the
    retry-candidate path tasks/proposed/<healed_highest+1> is invisible to
    ``_reconcile_scan_disk``'s is_dir() filter, so after a REAL heal (the
    orphan at id 1 registers) the retry mkdir genuinely re-collides."""
    repo, tw = fake_repo
    _make_orphan_dir(repo, "proposed", 1, with_body=True)  # heal registers this -> highest 1
    stray = repo / "tasks" / "proposed" / "2"  # retry-candidate path
    stray.write_text("not a directory\n")

    with pytest.raises(RuntimeError, match=r"audit --repair --apply") as excinfo:
        tw.create_task(tw.NewTaskRequest(kind="experiment", title="Doomed too"))
    assert isinstance(excinfo.value.__cause__, FileExistsError)
    # Single-save/single-commit invariant on the FAILURE path too: the heal's
    # in-memory registry mutations ride the create's own `_save_registry` +
    # commit, which never run on an aborted create -- so no partial registry
    # write lands on disk (the named `audit --repair --apply` is the remedy).
    assert not (repo / "tasks" / "REGISTRY.json").exists()


# ─── D: fail-soft best-effort registry staging (#2015 window narrowing) ─────


def test_save_registry_stages_in_git_repo(fake_repo):
    repo, tw = fake_repo
    tw._save_registry({"highest_id": 0, "tasks": {}})
    staged = subprocess.run(
        ["git", "diff", "--cached", "--name-only"],
        cwd=repo,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.splitlines()
    assert "tasks/REGISTRY.json" in staged


def test_save_registry_fail_soft_outside_git_repo(tmp_path, monkeypatch, caplog):
    import explore_persona_space.task_workflow as tw

    tw.invalidate_cache()
    # Non-git tmp tree: the atomic write must succeed and the stage must
    # degrade to a WARN -- never raise.
    monkeypatch.setattr(tw, "repo_root", lambda: tmp_path)
    monkeypatch.setattr(tw, "tasks_dir", lambda: tmp_path / "tasks")
    monkeypatch.setattr(tw, "registry_path", lambda: tmp_path / "tasks" / "REGISTRY.json")
    with caplog.at_level(logging.WARNING, logger="explore_persona_space.task_workflow"):
        tw._save_registry({"highest_id": 7, "tasks": {}})
    reg = json.loads((tmp_path / "tasks" / "REGISTRY.json").read_text())
    assert reg["highest_id"] == 7
    warn_lines = [
        r.getMessage() for r in caplog.records if "best-effort registry stage" in r.getMessage()
    ]
    assert warn_lines, "expected the fail-soft WARN when git add cannot run"
