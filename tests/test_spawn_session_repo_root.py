"""Tests for the #844 canonical-root resolution + spawn-cwd assertion in
``scripts/spawn_session.py``.

What this pins:

1. ``spawn_session.PROJECT_ROOT`` is the canonical PRIMARY checkout (the git
   common dir's parent): a real ``.git`` DIRECTORY, a ``tasks/`` dir, never a
   path under ``.claude/worktrees`` — and equals
   ``task_workflow.primary_checkout_root()``.
2. ``_assert_spawn_cwd`` accepts exactly {canonical root, the TARGET issue's
   own worktree} and refuses everything else with a loud ``SystemExit``
   naming ``#844`` (sibling worktree, random dir, a "root" whose ``.git`` is
   a file). An ``issue=None`` rejection never renders "issue-None".
3. Command-level wiring (#844 plan §12 amendment 1): all three spawn
   commands trip the assertion BEFORE any ``/spawn-session`` daemon POST.
4. Subprocess-shaped worktree-copy import (#844 plan §12 amendment 8): a
   WORKTREE COPY of ``spawn_session.py`` resolves ``PROJECT_ROOT`` to the
   primary checkout — the end-to-end bug shape that produced #844.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import spawn_session  # noqa: E402

from explore_persona_space.task_workflow import primary_checkout_root  # noqa: E402

# ── PROJECT_ROOT resolution ────────────────────────────────────────────────


def test_project_root_is_primary_checkout():
    """PROJECT_ROOT is the canonical primary checkout: `.git` is a real
    DIRECTORY (a linked worktree has a `.git` FILE), `tasks/` exists, no
    `.claude/worktrees` path component, and it equals
    `primary_checkout_root()`."""
    root = spawn_session.PROJECT_ROOT
    assert (root / ".git").is_dir(), f"{root}/.git is not a directory"
    assert (root / "tasks").is_dir(), f"{root} has no tasks/"
    assert ".claude" not in root.parts, f"PROJECT_ROOT is inside .claude/: {root}"
    assert root == primary_checkout_root()


# ── _assert_spawn_cwd unit behavior ────────────────────────────────────────


def _tmp_layout(tmp_path: Path) -> tuple[Path, Path]:
    """Build a canonical-looking tmp layout: root with a `.git/` DIRECTORY
    plus a `worktrees/issue-5` dir. Returns (root, worktree_dir)."""
    root = tmp_path / "root"
    (root / ".git").mkdir(parents=True)
    wt_dir = root / ".claude" / "worktrees"
    (wt_dir / "issue-5").mkdir(parents=True)
    return root, wt_dir


def test_assert_spawn_cwd_accepts_root_and_target_worktree(tmp_path, monkeypatch):
    root, wt_dir = _tmp_layout(tmp_path)
    monkeypatch.setattr(spawn_session, "PROJECT_ROOT", root)
    monkeypatch.setattr(spawn_session, "WORKTREE_DIR", wt_dir)
    # All three accepted shapes return without exiting.
    spawn_session._assert_spawn_cwd(root, issue=None)
    spawn_session._assert_spawn_cwd(root, issue=5)
    spawn_session._assert_spawn_cwd(wt_dir / "issue-5", issue=5)


def test_assert_spawn_cwd_rejects_sibling_worktree(tmp_path, monkeypatch):
    root, wt_dir = _tmp_layout(tmp_path)
    (wt_dir / "issue-7").mkdir()
    monkeypatch.setattr(spawn_session, "PROJECT_ROOT", root)
    monkeypatch.setattr(spawn_session, "WORKTREE_DIR", wt_dir)

    # A SIBLING issue's worktree (the #844 incident shape) is refused.
    with pytest.raises(SystemExit) as exc:
        spawn_session._assert_spawn_cwd(wt_dir / "issue-7", issue=5)
    assert "#844" in str(exc.value)
    assert "issue-5" in str(exc.value)  # names the TARGET worktree

    # A random directory is refused.
    other = tmp_path / "elsewhere"
    other.mkdir()
    with pytest.raises(SystemExit) as exc:
        spawn_session._assert_spawn_cwd(other, issue=5)
    assert "#844" in str(exc.value)

    # issue=None rejection: no "issue-None" rendering (§12 amendment 7).
    with pytest.raises(SystemExit) as exc:
        spawn_session._assert_spawn_cwd(other, issue=None)
    assert "#844" in str(exc.value)
    assert "issue-None" not in str(exc.value)
    assert "a target issue worktree" in str(exc.value)


def test_assert_spawn_cwd_rejects_root_whose_git_is_a_file(tmp_path, monkeypatch):
    """A PROJECT_ROOT whose `.git` is a FILE is a linked worktree, not the
    primary checkout — the tripwire refuses even the cwd == PROJECT_ROOT
    branch (guards a future edit reintroducing __file__ resolution)."""
    fake_root = tmp_path / "fake"
    fake_root.mkdir()
    (fake_root / ".git").write_text("gitdir: /main/.git/worktrees/issue-999\n")
    monkeypatch.setattr(spawn_session, "PROJECT_ROOT", fake_root)
    monkeypatch.setattr(spawn_session, "WORKTREE_DIR", fake_root / ".claude" / "worktrees")
    with pytest.raises(SystemExit) as exc:
        spawn_session._assert_spawn_cwd(fake_root, issue=None)
    assert "#844" in str(exc.value)
    assert "not the primary checkout" in str(exc.value)


# ── command-level wiring (§12 amendment 1) ─────────────────────────────────


@pytest.mark.parametrize("command", ["pm", "issue", "campaign"])
def test_spawn_commands_assert_cwd_before_post(tmp_path, monkeypatch, command):
    """Each of the three spawn commands is WIRED to `_assert_spawn_cwd`: on a
    linked-worktree-like PROJECT_ROOT (`.git` is a FILE) they raise a
    SystemExit naming #844 BEFORE any `/spawn-session` daemon POST."""
    root = tmp_path / "root"
    root.mkdir()
    (root / ".git").write_text("gitdir: /main/.git/worktrees/issue-999\n")
    monkeypatch.setattr(spawn_session, "PROJECT_ROOT", root)
    monkeypatch.setattr(spawn_session, "WORKTREE_DIR", root / ".claude" / "worktrees")

    def _fail_post(route, body):
        raise AssertionError(f"post({route!r}) reached before the #844 spawn-cwd assertion")

    monkeypatch.setattr(spawn_session, "post", _fail_post)
    # Defense in depth: the daemon-patch verifier can sys.exit for unrelated
    # host reasons; stub it so only the cwd assertion can exit here.
    monkeypatch.setattr(spawn_session, "_verify_happy_patch_or_die", lambda *a, **k: None)

    args = argparse.Namespace(
        issue=5,
        model=None,
        betas=None,
        effort=None,
        initial_prompt=None,
        auto=False,
        auto_approve_gpu_hours=100.0,
        budget_gpu_hours=None,
        max_concurrent=None,
        per_child_cap=None,
    )
    fn = {
        "pm": spawn_session.cmd_spawn_pm,
        "issue": spawn_session.cmd_spawn_issue,
        "campaign": spawn_session.cmd_spawn_campaign,
    }[command]
    with pytest.raises(SystemExit) as exc:
        fn(args)
    assert "#844" in str(exc.value)


# ── subprocess-shaped worktree-copy import (§12 amendment 8) ───────────────


def _make_scratch_repo_with_spawn_session(tmp_path: Path) -> tuple[Path, Path]:
    """Scratch primary repo (main branch, tasks/) carrying copies of the real
    `src/explore_persona_space/task_workflow.py` + `scripts/spawn_session.py`,
    committed, plus a linked worktree. Returns (main_repo, worktree)."""
    main_repo = tmp_path / "repo"
    subprocess.run(["git", "init", "-q", "-b", "main", str(main_repo)], check=True)
    subprocess.run(["git", "-C", str(main_repo), "config", "user.email", "t@t.t"], check=True)
    subprocess.run(["git", "-C", str(main_repo), "config", "user.name", "t"], check=True)
    subprocess.run(["git", "-C", str(main_repo), "config", "commit.gpgsign", "false"], check=True)
    (main_repo / "tasks").mkdir()
    (main_repo / "tasks" / ".gitkeep").touch()

    src_dir = main_repo / "src" / "explore_persona_space"
    src_dir.mkdir(parents=True)
    (src_dir / "__init__.py").touch()
    real_tw = (
        Path(__file__).resolve().parents[1] / "src" / "explore_persona_space" / "task_workflow.py"
    )
    (src_dir / "task_workflow.py").write_text(real_tw.read_text())

    scripts_dir = main_repo / "scripts"
    scripts_dir.mkdir()
    (scripts_dir / "spawn_session.py").write_text((SCRIPTS / "spawn_session.py").read_text())

    subprocess.run(["git", "-C", str(main_repo), "add", "."], check=True)
    subprocess.run(["git", "-C", str(main_repo), "commit", "-q", "-m", "init"], check=True)

    worktree = tmp_path / "wt-sibling"
    subprocess.run(
        ["git", "-C", str(main_repo), "worktree", "add", "-b", "issue-999", str(worktree)],
        check=True,
        capture_output=True,
    )
    return main_repo, worktree


def test_worktree_copy_of_spawn_session_resolves_primary_root(tmp_path):
    """End-to-end #844 shape: importing the WORKTREE COPY of spawn_session.py
    (as a worktree copy of file_infra_task.py would) yields PROJECT_ROOT ==
    the PRIMARY checkout — never the worktree the old `Path(__file__)`
    resolution produced."""
    main_repo, worktree = _make_scratch_repo_with_spawn_session(tmp_path)

    env = dict(os.environ)
    env["PYTHONPATH"] = str(worktree / "src") + os.pathsep + env.get("PYTHONPATH", "")
    snippet = textwrap.dedent(
        f"""
        import sys
        sys.path.insert(0, {str(worktree / "scripts")!r})
        import spawn_session
        print("PROJECT_ROOT=" + str(spawn_session.PROJECT_ROOT))
        """
    )
    proc = subprocess.run(
        [sys.executable, "-c", snippet],
        cwd=str(worktree),
        env=env,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, (
        f"worktree-copy import failed:\nstdout: {proc.stdout}\nstderr: {proc.stderr}"
    )
    line = next(ln for ln in proc.stdout.splitlines() if ln.startswith("PROJECT_ROOT="))
    resolved = Path(line.partition("=")[2].strip())
    assert resolved.resolve() == main_repo.resolve(), (
        f"worktree copy resolved {resolved}, expected the primary {main_repo}"
    )
    assert resolved.resolve() != worktree.resolve(), "resolved the worktree — the #844 bug shape"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
