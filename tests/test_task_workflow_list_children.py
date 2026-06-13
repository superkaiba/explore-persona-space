"""Tests for task_workflow.list_children + the `task.py list-children` CLI
handler (task #586). Same fake-repo pattern as tests/test_task_workflow.py."""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from argparse import Namespace
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def fake_repo(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """git-init tmp_path and rebind task_workflow's resolvers to it."""
    subprocess.run(["git", "init", "-q", "-b", "main"], cwd=tmp_path, check=True)
    subprocess.run(["git", "config", "user.email", "test@test.test"], cwd=tmp_path, check=True)
    subprocess.run(["git", "config", "user.name", "test"], cwd=tmp_path, check=True)
    subprocess.run(["git", "config", "commit.gpgsign", "false"], cwd=tmp_path, check=True)
    subprocess.run(["git", "commit", "-q", "--allow-empty", "-m", "init"], cwd=tmp_path, check=True)

    sys.path.insert(0, str(REPO_ROOT / "src"))
    import explore_persona_space.task_workflow as tw

    tw.invalidate_cache()
    monkeypatch.setattr(tw, "repo_root", lambda: tmp_path)
    monkeypatch.setattr(tw, "tasks_dir", lambda: tmp_path / "tasks")
    monkeypatch.setattr(tw, "registry_path", lambda: tmp_path / "tasks" / "REGISTRY.json")
    lock_dir = tmp_path / ".task-workflow"
    monkeypatch.setattr(tw, "LOCK_DIR", lock_dir)
    monkeypatch.setattr(tw, "LOCK_PATH", lock_dir / "lock")
    (tmp_path / "tasks").mkdir()
    return tmp_path, tw


def _seed_family(tw) -> tuple[int, int, int, int]:
    """Create a parent, two children, and one unrelated task. Returns
    (parent, child_a, child_b, unrelated)."""
    parent = tw.create_task(tw.NewTaskRequest(kind="campaign", title="parent campaign"))
    child_a = tw.create_task(
        tw.NewTaskRequest(kind="experiment", title="child A", parent_id=parent)
    )
    child_b = tw.create_task(
        tw.NewTaskRequest(kind="experiment", title="child B", parent_id=parent)
    )
    unrelated = tw.create_task(tw.NewTaskRequest(kind="infra", title="unrelated"))
    return parent, child_a, child_b, unrelated


def test_list_children_filters_by_parent_id(fake_repo):
    _, tw = fake_repo
    parent, child_a, child_b, unrelated = _seed_family(tw)

    rows = tw.list_children(parent)
    assert [r["id"] for r in rows] == [child_a, child_b]
    assert all(set(r) == {"id", "status", "title", "kind", "has_clean_result"} for r in rows)
    assert rows[0]["title"] == "child A"
    assert rows[0]["kind"] == "experiment"
    assert rows[0]["status"] == "proposed"
    assert rows[0]["has_clean_result"] is False

    # The unrelated task has no children; a leaf child has none either.
    assert tw.list_children(unrelated) == []
    assert tw.list_children(child_a) == []


def test_list_children_tracks_status_moves(fake_repo):
    _, tw = fake_repo
    parent, child_a, _child_b, _ = _seed_family(tw)
    tw.set_status(child_a, "running")
    rows = {r["id"]: r for r in tw.list_children(parent)}
    assert rows[child_a]["status"] == "running"


def _load_task_cli():
    """Import scripts/task.py as a module (it is a script, not a package)."""
    spec = importlib.util.spec_from_file_location(
        "task_cli_under_test", REPO_ROOT / "scripts" / "task.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_cli_list_children_json_shape(fake_repo, capsys):
    _, tw = fake_repo
    parent, child_a, child_b, _ = _seed_family(tw)
    cli = _load_task_cli()

    cli.cmd_list_children(Namespace(number=parent, json=True))
    rows = json.loads(capsys.readouterr().out)
    assert [r["id"] for r in rows] == [child_a, child_b]
    assert rows[0]["kind"] == "experiment"

    # No children -> literally `[]` (the smoke-test contract).
    cli.cmd_list_children(Namespace(number=child_a, json=True))
    assert capsys.readouterr().out.strip() == "[]"
