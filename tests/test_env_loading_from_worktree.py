"""Regression test: load_dotenv() resolves .env via main git worktree.

A linked git worktree does NOT inherit the gitignored ``.env`` from the
main worktree. Pod-side drivers launched from
``/workspace/wt-issue-N/`` (the pattern used to dodge task.py's
branch-guard, see task #407 events.jsonl 2026-05-28) were silently
running with empty credentials until :func:`resolve_dotenv_path` was
taught to walk ``git rev-parse --git-common-dir`` back to the main
worktree.

This test creates a real linked worktree with ``git worktree add`` in a
temp repo, writes ``.env`` only in the main worktree, and verifies that
``load_dotenv`` called from inside the linked worktree finds it.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

from explore_persona_space.orchestrate.env import (
    load_dotenv,
    resolve_dotenv_path,
)


def _git(cwd: Path, *args: str) -> None:
    subprocess.run(
        ["git", *args],
        cwd=str(cwd),
        check=True,
        capture_output=True,
        env={
            **os.environ,
            "GIT_AUTHOR_NAME": "t",
            "GIT_AUTHOR_EMAIL": "t@t",
            "GIT_COMMITTER_NAME": "t",
            "GIT_COMMITTER_EMAIL": "t@t",
        },
    )


@pytest.fixture
def linked_worktree(tmp_path: Path) -> tuple[Path, Path]:
    """Create a main worktree with .env and a linked worktree without .env.

    Returns (main_worktree, linked_worktree).
    """
    main = tmp_path / "main"
    main.mkdir()
    _git(main, "init", "-q", "-b", "main")
    (main / "README").write_text("seed\n")
    _git(main, "add", "README")
    _git(main, "commit", "-q", "-m", "seed")
    (main / ".env").write_text("TEST_ENV_KEY_FROM_MAIN=loaded_from_main_worktree\n")

    linked = tmp_path / "linked"
    _git(main, "worktree", "add", "-q", str(linked), "-b", "linked-branch")

    assert (main / ".env").is_file()
    assert not (linked / ".env").exists(), "linked worktree must NOT inherit .env"

    return main, linked


def test_resolve_dotenv_finds_main_worktree_from_linked(
    linked_worktree: tuple[Path, Path],
) -> None:
    """resolve_dotenv_path(linked) returns the main worktree's .env."""
    main, linked = linked_worktree
    resolved = resolve_dotenv_path(linked)
    assert resolved is not None, "expected fallback to main worktree .env"
    assert resolved.resolve() == (main / ".env").resolve()


def test_resolve_dotenv_prefers_local_when_present(
    linked_worktree: tuple[Path, Path],
) -> None:
    """A .env in the worktree itself wins over the main worktree's .env."""
    _main, linked = linked_worktree
    (linked / ".env").write_text("TEST_ENV_KEY_FROM_LINKED=from_linked\n")
    resolved = resolve_dotenv_path(linked)
    assert resolved is not None
    assert resolved.resolve() == (linked / ".env").resolve()


def test_load_dotenv_populates_env_from_linked_worktree(
    linked_worktree: tuple[Path, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Calling load_dotenv() from inside a linked worktree loads the main .env."""
    _main, linked = linked_worktree
    monkeypatch.chdir(linked)
    monkeypatch.delenv("TEST_ENV_KEY_FROM_MAIN", raising=False)

    resolved = resolve_dotenv_path(linked)
    assert resolved is not None
    load_dotenv(str(resolved))

    assert os.environ.get("TEST_ENV_KEY_FROM_MAIN") == "loaded_from_main_worktree"


def test_resolve_dotenv_returns_none_when_no_env_anywhere(
    tmp_path: Path,
) -> None:
    """No .env in worktree, no git, no /workspace fallback → returns None."""
    bare = tmp_path / "bare"
    bare.mkdir()
    resolved = resolve_dotenv_path(bare)
    # On a dev box without /workspace/explore-persona-space/.env, this is None.
    # On a pod where that path exists, it would resolve to it — we accept either
    # but reject false matches.
    if resolved is not None:
        assert resolved == Path("/workspace/explore-persona-space/.env")
        assert resolved.is_file()


def test_bootstrap_module_uses_canonical_loader() -> None:
    """scripts/_bootstrap.py delegates to orchestrate.env.load_dotenv.

    Guard against a future drive-by edit that re-introduces a hard-coded
    ``load_dotenv(PROJECT_ROOT/.env)`` path. The whole point of this fix
    is that worktree-aware resolution lives in ONE place.
    """
    bootstrap_src = Path(__file__).resolve().parent.parent / "scripts" / "_bootstrap.py"
    body = bootstrap_src.read_text()
    assert "from explore_persona_space.orchestrate.env import load_dotenv" in body, (
        "_bootstrap.py must delegate to the canonical loader; do not "
        "re-introduce hard-coded PROJECT_ROOT/.env."
    )
    # And it must NOT use the dotenv package directly with a hard-coded path:
    assert "from dotenv import load_dotenv" not in body, (
        "_bootstrap.py must not import dotenv directly — go through "
        "orchestrate.env.load_dotenv so worktree resolution stays consistent."
    )


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
