"""Tests for the ``workflow:`` frontmatter plumbing + the campaign v1 carve-out.

Covers the EPS workflow-v2 plan (Assumption 2):

- ``task.py new`` pins a ``workflow: <v1|v2>`` frontmatter key on every NEW
  task, resolved as explicit-arg > env ``EPM_DEFAULT_WORKFLOW`` >
  :data:`task_workflow.DEFAULT_WORKFLOW_VERSION` ("v1").
- :func:`task_workflow.workflow_version` fail-OPENS to "v1" for an absent /
  empty / unknown value, so legacy tasks (no ``workflow:`` key) resolve to the
  current pipeline everywhere and garbage never crashes a caller.
- The ``/campaign`` skill pins its children to v1 explicitly on the same
  ``task.py new`` line, so the future v2 default-flip cannot leak into
  campaign children.

The fake-repo fixture mirrors ``tests/test_task_workflow.py::fake_repo`` — it
monkeypatches the ``task_workflow`` resolver FUNCTIONS so every in-module call
site (including ``task_cli``'s imported ``create_task``, which shares the
cached ``task_workflow`` module) resolves the tmp repo, never the real
``tasks/``. ``scripts/task.py`` is loaded as an importable module (the
``importlib`` pattern from ``tests/test_task_cli_set_body_assertions.py``) so
``main()`` can be driven in-process against that fake repo.
"""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[1]
_SRC = _ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

# Load scripts/task.py as an importable module so we can drive main() in-process
# against the fake repo (subprocess would resolve the REAL tasks/ tree).
_SCRIPT = _ROOT / "scripts" / "task.py"
_spec = importlib.util.spec_from_file_location("task_cli", _SCRIPT)
task_cli = importlib.util.module_from_spec(_spec)  # type: ignore[arg-type]
sys.modules["task_cli"] = task_cli
_spec.loader.exec_module(task_cli)  # type: ignore[union-attr]


@pytest.fixture
def fake_repo(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """tmp_path as a git repo with task_workflow's resolver rebound to it.

    Returns (repo_root, tw). Auto-push is disabled by leaving
    ``TASK_PY_AUTO_PUSH`` unset; ``EPM_DEFAULT_WORKFLOW`` is cleared so a
    stray env from the outer shell can't perturb the default-resolution
    tests (each test that needs it sets it explicitly).
    """
    subprocess.run(["git", "init", "-q", "-b", "main"], cwd=tmp_path, check=True)
    subprocess.run(["git", "config", "user.email", "test@test.test"], cwd=tmp_path, check=True)
    subprocess.run(["git", "config", "user.name", "test"], cwd=tmp_path, check=True)
    subprocess.run(["git", "config", "commit.gpgsign", "false"], cwd=tmp_path, check=True)
    subprocess.run(["git", "commit", "-q", "--allow-empty", "-m", "init"], cwd=tmp_path, check=True)

    import explore_persona_space.task_workflow as tw

    tw.invalidate_cache()
    monkeypatch.setattr(tw, "repo_root", lambda: tmp_path)
    monkeypatch.setattr(tw, "tasks_dir", lambda: tmp_path / "tasks")
    monkeypatch.setattr(tw, "registry_path", lambda: tmp_path / "tasks" / "REGISTRY.json")
    lock_dir = tmp_path / ".task-workflow"
    monkeypatch.setattr(tw, "LOCK_DIR", lock_dir)
    monkeypatch.setattr(tw, "LOCK_PATH", lock_dir / "lock")
    monkeypatch.delenv("TASK_PY_AUTO_PUSH", raising=False)
    monkeypatch.delenv("EPM_DEFAULT_WORKFLOW", raising=False)
    return tmp_path, tw


def _cli_new(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture, *args: str) -> int:
    """Drive ``task.py new <args>`` through ``main()`` in-process; return the new id.

    ``cmd_create`` echoes ``#<id>`` on success (via ``_safe_echo`` -> Python
    ``print``, which capsys captures; the git subprocess output goes to the
    real fd and is not captured).
    """
    monkeypatch.setattr(sys, "argv", ["task.py", "new", *args])
    task_cli.main()
    out = capsys.readouterr().out
    ids = [int(tok[1:]) for tok in out.split() if tok.startswith("#") and tok[1:].isdigit()]
    assert ids, f"no created-task id echoed by `task.py new`; stdout={out!r}"
    return ids[-1]


# ─── (a) no flag -> v1 (end-to-end through argparse default=None) ──────────


def test_new_without_flag_defaults_to_v1(fake_repo, monkeypatch, capsys):
    _, tw = fake_repo
    tid = _cli_new(monkeypatch, capsys, "--title", "no-flag", "--kind", "experiment")
    fm = tw.get_task(tid)["frontmatter"]
    assert fm["workflow"] == "v1"
    assert tw.workflow_version(fm) == "v1"


# ─── (b) --workflow v2 -> v2 in frontmatter, visible via `view` ────────────


def test_new_with_workflow_v2_writes_v2(fake_repo, monkeypatch, capsys):
    _, tw = fake_repo
    tid = _cli_new(
        monkeypatch, capsys, "--title", "v2-task", "--kind", "experiment", "--workflow", "v2"
    )
    # `view` surface: cmd_view --json returns get_task(...)["frontmatter"].
    fm = tw.get_task(tid)["frontmatter"]
    assert fm["workflow"] == "v2"
    assert tw.workflow_version(fm) == "v2"
    # And durably on disk in body.md.
    body = (tw.find_task_path(tid) / "body.md").read_text()
    assert "workflow: v2" in body


# ─── (c) EPM_DEFAULT_WORKFLOW env resolution ───────────────────────────────


def test_env_default_workflow_v2(fake_repo, monkeypatch):
    _, tw = fake_repo
    monkeypatch.setenv("EPM_DEFAULT_WORKFLOW", "v2")
    tid = tw.create_task(tw.NewTaskRequest(kind="experiment", title="env-v2"))
    assert tw.get_task(tid)["frontmatter"]["workflow"] == "v2"


def test_explicit_flag_beats_env(fake_repo, monkeypatch):
    """Explicit arg wins over the env default (precedence order)."""
    _, tw = fake_repo
    monkeypatch.setenv("EPM_DEFAULT_WORKFLOW", "v2")
    tid = tw.create_task(tw.NewTaskRequest(kind="experiment", title="explicit-v1", workflow="v1"))
    assert tw.get_task(tid)["frontmatter"]["workflow"] == "v1"


def test_env_garbage_falls_open_to_v1(fake_repo, monkeypatch):
    """An unknown env value falls through to the "v1" default (fail-open)."""
    _, tw = fake_repo
    monkeypatch.setenv("EPM_DEFAULT_WORKFLOW", "nonsense")
    tid = tw.create_task(tw.NewTaskRequest(kind="experiment", title="env-garbage"))
    assert tw.get_task(tid)["frontmatter"]["workflow"] == "v1"


# ─── (d) workflow_version() helper ─────────────────────────────────────────


def test_workflow_version_helper():
    import explore_persona_space.task_workflow as tw

    assert tw.workflow_version({}) == "v1"  # absent key (legacy task)
    assert tw.workflow_version({"workflow": "v1"}) == "v1"
    assert tw.workflow_version({"workflow": "v2"}) == "v2"
    assert tw.workflow_version({"workflow": "garbage"}) == "v1"  # unknown -> v1
    assert tw.workflow_version({"workflow": ""}) == "v1"  # empty -> v1
    assert tw.workflow_version({"workflow": None}) == "v1"  # non-str -> v1
    assert tw.workflow_version({"workflow": "  v2  "}) == "v2"  # whitespace tolerated


# ─── (e) campaign children pinned to v1 (text-pin invariant) ───────────────


def test_campaign_children_pinned_to_v1():
    """Every `scripts/task.py new` child-creation invocation in the campaign
    skill carries `--workflow v1` on the SAME line (plan Assumption 2)."""
    skill = _ROOT / ".claude" / "skills" / "campaign" / "SKILL.md"
    invocations = [ln for ln in skill.read_text().splitlines() if "scripts/task.py new" in ln]
    assert invocations, (
        "expected a `scripts/task.py new` child-creation invocation in campaign SKILL.md"
    )
    for ln in invocations:
        assert "--workflow v1" in ln, (
            "campaign child-creation `task.py new` must pin `--workflow v1` on the same line "
            f"so the future v2 default-flip cannot leak into campaign children; line: {ln!r}"
        )


# ─── CLI surface: `task.py new --help` advertises --workflow {v1,v2} ───────


def test_cli_new_help_advertises_workflow_flag():
    # Use the current interpreter (the test venv) rather than `uv run`, which
    # would try to build a fresh .venv in this worktree.
    proc = subprocess.run(
        [sys.executable, str(_SCRIPT), "new", "--help"],
        capture_output=True,
        text=True,
        cwd=_ROOT,
    )
    assert proc.returncode == 0, proc.stderr
    assert "--workflow" in proc.stdout
    assert "v1" in proc.stdout and "v2" in proc.stdout
