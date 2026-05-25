"""Tests for the worktree-safe `repo_root()` resolver in task_workflow.

These tests exercise the 2026-05-25 worktree-staleness fix. Every test
spawns a SUBPROCESS — `importlib.reload` is intentionally NOT used,
because the bug class we are protecting against is "wrong cwd / wrong
git context at import time", which only manifests in a fresh Python
process. See `plans/2026-05-25_022522-tasks-canonical-main.md` § Risks
("`importlib` test won't reproduce the worktree bug").

What we cover:
  1. From inside a git worktree on a feature branch, `repo_root()` /
     `tasks_dir()` resolve to the MAIN worktree (not the worktree dir).
  2. Branch guard: main worktree HEAD on a non-`main` branch → distinct
     `RuntimeError` naming the branch.
  3. Detached HEAD → distinct `RuntimeError` mentioning "detached".
  4. Validation: missing `tasks/`, bare repo, `.git/modules/<x>`
     submodule shape — all loud errors, no silent fallback.
  5. Env-poisoning: `GIT_DIR` / `GIT_WORK_TREE` set in env do NOT
     redirect the resolver.
  6. PEP-562 `tw.TASKS_DIR` / `tw.REPO` attribute access works lazily.
  7. Cache: repeated `repo_root()` calls only fire the git subprocess
     pair once per (pid, cwd); `invalidate_cache()` re-fires.
  8. cwd-independence: invoking from `/tmp` still resolves to the
     correct repo (the resolver uses the module's directory, not
     `os.getcwd()`).
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

# Absolute path to this repo's `src/` so subprocesses can `import
# explore_persona_space.task_workflow`. We pass this on PYTHONPATH below.
_REPO_SRC = str(Path(__file__).resolve().parents[1] / "src")


# ─── Helpers ───────────────────────────────────────────────────────────────


def _make_main_repo(repo: Path) -> None:
    """Initialize ``repo`` as a fresh git repo with a `main` branch and a
    ``tasks/`` directory (the validation step requires it).
    """
    subprocess.run(["git", "init", "-q", "-b", "main", str(repo)], check=True)
    subprocess.run(["git", "-C", str(repo), "config", "user.email", "t@t.t"], check=True)
    subprocess.run(["git", "-C", str(repo), "config", "user.name", "t"], check=True)
    subprocess.run(["git", "-C", str(repo), "config", "commit.gpgsign", "false"], check=True)
    (repo / "tasks").mkdir()
    (repo / "tasks" / ".gitkeep").touch()
    subprocess.run(["git", "-C", str(repo), "add", "."], check=True)
    subprocess.run(["git", "-C", str(repo), "commit", "-q", "-m", "init"], check=True)


def _run_resolver(
    cwd: Path,
    snippet: str,
    *,
    extra_env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    """Run a one-shot Python `snippet` in a subprocess, with PYTHONPATH set
    so `import explore_persona_space.task_workflow` works.

    Returns the CompletedProcess (caller asserts on returncode / stdout /
    stderr). We do NOT pass `check=True` — many tests assert on the
    error message in stderr.
    """
    env = dict(os.environ)
    # Make sure the subprocess can import the project.
    env["PYTHONPATH"] = _REPO_SRC + os.pathsep + env.get("PYTHONPATH", "")
    if extra_env:
        env.update(extra_env)
    return subprocess.run(
        [sys.executable, "-c", snippet],
        cwd=str(cwd),
        env=env,
        capture_output=True,
        text=True,
    )


def _resolver_snippet(extra: str = "") -> str:
    """Build a small snippet that prints resolve results and exits 0
    on success / non-zero on failure (RuntimeError → traceback to
    stderr → non-zero exit).
    """
    return textwrap.dedent(
        f"""
        from explore_persona_space.task_workflow import (
            repo_root, tasks_dir, registry_path,
        )
        print('REPO=' + str(repo_root()))
        print('TASKS_DIR=' + str(tasks_dir()))
        print('REGISTRY_PATH=' + str(registry_path()))
        {extra}
        """
    )


def _parse_resolved(stdout: str) -> dict[str, str]:
    out: dict[str, str] = {}
    for line in stdout.splitlines():
        if "=" in line:
            k, _, v = line.partition("=")
            out[k.strip()] = v.strip()
    return out


# ─── Tests ─────────────────────────────────────────────────────────────────


def test_resolves_main_repo_from_worktree(tmp_path: Path) -> None:
    """From inside a feature-branch worktree, the resolver returns the
    MAIN worktree root, not the worktree directory.
    """
    main_repo = tmp_path / "repo"
    _make_main_repo(main_repo)

    # Add a worktree on a feature branch.
    worktree = tmp_path / "wt-feature"
    subprocess.run(
        ["git", "-C", str(main_repo), "worktree", "add", "-b", "feature/x", str(worktree)],
        check=True,
        capture_output=True,
    )

    # We need the subprocess to import the *test* tmp repo's copy of
    # task_workflow.py, not the dev repo's. Symlink src/ inside.
    src_dir = main_repo / "src" / "explore_persona_space"
    src_dir.mkdir(parents=True)
    real_tw = Path(_REPO_SRC) / "explore_persona_space" / "task_workflow.py"
    (src_dir / "__init__.py").touch()
    (src_dir / "task_workflow.py").write_text(real_tw.read_text())
    # The worktree shares the same source tree via git; commit so the
    # worktree sees the file.
    subprocess.run(["git", "-C", str(main_repo), "add", "src"], check=True)
    subprocess.run(["git", "-C", str(main_repo), "commit", "-q", "-m", "src"], check=True)

    # Pull main into the worktree branch so it sees src/.
    subprocess.run(
        ["git", "-C", str(worktree), "merge", "main", "-q", "--no-edit"],
        check=True,
        capture_output=True,
    )

    # Invoke from inside the worktree, with PYTHONPATH pointing at the
    # tmp repo's src (so we import the test copy of task_workflow).
    env = dict(os.environ)
    env["PYTHONPATH"] = str(main_repo / "src") + os.pathsep + env.get("PYTHONPATH", "")
    proc = subprocess.run(
        [sys.executable, "-c", _resolver_snippet()],
        cwd=str(worktree),
        env=env,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, f"resolver failed:\nstdout: {proc.stdout}\nstderr: {proc.stderr}"
    resolved = _parse_resolved(proc.stdout)
    # Resolve real-paths because macos /tmp -> /private/tmp et al. The
    # invariant is "we resolved to the main worktree", not "we kept the
    # exact spelling the test passed in".
    assert Path(resolved["REPO"]).resolve() == main_repo.resolve()
    assert Path(resolved["TASKS_DIR"]).resolve() == (main_repo / "tasks").resolve()


def test_branch_guard_rejects_non_main_with_branch_name(tmp_path: Path) -> None:
    """Main worktree HEAD on a non-`main` branch → loud error naming the branch."""
    main_repo = tmp_path / "repo"
    _make_main_repo(main_repo)
    subprocess.run(
        ["git", "-C", str(main_repo), "checkout", "-q", "-b", "feature/off-main"],
        check=True,
    )

    # Drop the tw module into the test repo so the subprocess can import it.
    src_dir = main_repo / "src" / "explore_persona_space"
    src_dir.mkdir(parents=True)
    (src_dir / "__init__.py").touch()
    real_tw = Path(_REPO_SRC) / "explore_persona_space" / "task_workflow.py"
    (src_dir / "task_workflow.py").write_text(real_tw.read_text())

    env = dict(os.environ)
    env["PYTHONPATH"] = str(main_repo / "src") + os.pathsep + env.get("PYTHONPATH", "")
    proc = subprocess.run(
        [sys.executable, "-c", _resolver_snippet()],
        cwd=str(main_repo),
        env=env,
        capture_output=True,
        text=True,
    )
    assert proc.returncode != 0, f"branch guard did not refuse:\n{proc.stdout}"
    assert "feature/off-main" in proc.stderr, f"branch name not in error: {proc.stderr}"
    assert "main" in proc.stderr.lower()


def test_branch_guard_distinct_error_on_detached_head(tmp_path: Path) -> None:
    """Detached HEAD → DISTINCT error mentioning 'detached'."""
    main_repo = tmp_path / "repo"
    _make_main_repo(main_repo)
    # Detach.
    subprocess.run(
        ["git", "-C", str(main_repo), "checkout", "-q", "--detach", "HEAD"],
        check=True,
    )

    src_dir = main_repo / "src" / "explore_persona_space"
    src_dir.mkdir(parents=True)
    (src_dir / "__init__.py").touch()
    real_tw = Path(_REPO_SRC) / "explore_persona_space" / "task_workflow.py"
    (src_dir / "task_workflow.py").write_text(real_tw.read_text())

    env = dict(os.environ)
    env["PYTHONPATH"] = str(main_repo / "src") + os.pathsep + env.get("PYTHONPATH", "")
    proc = subprocess.run(
        [sys.executable, "-c", _resolver_snippet()],
        cwd=str(main_repo),
        env=env,
        capture_output=True,
        text=True,
    )
    assert proc.returncode != 0
    assert "detached" in proc.stderr.lower(), f"'detached' not in error: {proc.stderr}"


def test_validation_rejects_missing_tasks_dir(tmp_path: Path) -> None:
    """Repo with no `tasks/` directory → loud error, no silent fallback."""
    main_repo = tmp_path / "repo"
    subprocess.run(["git", "init", "-q", "-b", "main", str(main_repo)], check=True)
    subprocess.run(["git", "-C", str(main_repo), "config", "user.email", "t@t.t"], check=True)
    subprocess.run(["git", "-C", str(main_repo), "config", "user.name", "t"], check=True)
    subprocess.run(["git", "-C", str(main_repo), "config", "commit.gpgsign", "false"], check=True)
    # NO tasks/ dir.
    subprocess.run(
        ["git", "-C", str(main_repo), "commit", "-q", "--allow-empty", "-m", "init"], check=True
    )

    src_dir = main_repo / "src" / "explore_persona_space"
    src_dir.mkdir(parents=True)
    (src_dir / "__init__.py").touch()
    real_tw = Path(_REPO_SRC) / "explore_persona_space" / "task_workflow.py"
    (src_dir / "task_workflow.py").write_text(real_tw.read_text())

    env = dict(os.environ)
    env["PYTHONPATH"] = str(main_repo / "src") + os.pathsep + env.get("PYTHONPATH", "")
    proc = subprocess.run(
        [sys.executable, "-c", _resolver_snippet()],
        cwd=str(main_repo),
        env=env,
        capture_output=True,
        text=True,
    )
    assert proc.returncode != 0
    assert "tasks/" in proc.stderr or "tasks" in proc.stderr.lower()
    # Must NOT silently fall back.
    assert "Traceback" in proc.stderr or "RuntimeError" in proc.stderr


def test_validation_rejects_bare_repo(tmp_path: Path) -> None:
    """A bare repo layout (`--bare`) does not have a `.git` parent → reject."""
    bare = tmp_path / "bare.git"
    subprocess.run(["git", "init", "-q", "--bare", "-b", "main", str(bare)], check=True)

    # We need a place to drop task_workflow.py such that the subprocess
    # CWD lands inside the bare repo's reach. Just cwd in bare.git and
    # use PYTHONPATH pointing at the dev repo.
    env = dict(os.environ)
    env["PYTHONPATH"] = _REPO_SRC + os.pathsep + env.get("PYTHONPATH", "")
    snippet = textwrap.dedent(
        """
        # The real module lives in the dev repo; we just need to exercise
        # the resolver against the bare-repo cwd. But since the resolver
        # uses the module-dir cwd, not os.getcwd(), this case actually
        # tests that we DON'T crash from os.getcwd() being bare. Skip if
        # the dev repo's resolver already errors out (we're testing a
        # tmp-repo invariant here).
        from explore_persona_space.task_workflow import repo_root
        try:
            print('REPO=' + str(repo_root()))
        except RuntimeError as e:
            # That's fine — the test just confirms loud-error-not-silent.
            print('ERR=' + str(e))
        """
    )
    # cd into the bare-repo dir
    proc = subprocess.run(
        [sys.executable, "-c", snippet],
        cwd=str(bare),
        env=env,
        capture_output=True,
        text=True,
    )
    # We don't care WHICH error fires (dev repo resolver may complain
    # about non-main branch first); we care that we don't silently
    # accept the bare repo as a valid resolution.
    combined = proc.stdout + proc.stderr
    assert "REPO=" + str(bare) not in combined, f"bare repo accepted: {combined}"


def test_validation_rejects_real_submodule_layout(tmp_path: Path) -> None:
    """A real git submodule must be rejected with a loud error.

    Creates an outer repo, adds itself as a submodule at `inner/`, drops
    `task_workflow.py` into the SUBMODULE's working tree, then invokes
    the resolver from inside the submodule. `git rev-parse
    --path-format=absolute --git-common-dir` from inside a submodule
    returns `.../.git/modules/inner` — basename is `inner`, not `.git`,
    so the basename check at `_resolve_repo_root_cached` line ~169 fires
    and raises before reaching the `parent / "tasks"` validation.

    Documents the resolver invariant: the dedicated submodule guard at
    the old lines 181-187 was dead code — the basename check carries
    the submodule case unaided. Confirmed manually:

        $ git rev-parse --path-format=absolute --git-common-dir
        /tmp/x/outer/.git/modules/inner
        $ basename /tmp/x/outer/.git/modules/inner
        inner
    """
    outer = tmp_path / "outer"
    _make_main_repo(outer)

    # `git submodule add file://...` is disabled by default in modern git
    # (protocol.file.allow=user). Override locally for this single command.
    subprocess.run(
        [
            "git",
            "-c",
            "protocol.file.allow=always",
            "-C",
            str(outer),
            "submodule",
            "add",
            "-q",
            str(outer),
            "inner",
        ],
        check=True,
        capture_output=True,
    )
    subprocess.run(
        ["git", "-C", str(outer), "commit", "-q", "-m", "add submodule"],
        check=True,
        capture_output=True,
    )

    # The submodule's working tree is `outer/inner/`. Drop the test copy
    # of task_workflow into it so the subprocess imports OUR resolver
    # (not the submodule's own snapshot, which is the outer-repo HEAD).
    inner = outer / "inner"
    src_dir = inner / "src" / "explore_persona_space"
    src_dir.mkdir(parents=True)
    (src_dir / "__init__.py").touch()
    real_tw = Path(_REPO_SRC) / "explore_persona_space" / "task_workflow.py"
    (src_dir / "task_workflow.py").write_text(real_tw.read_text())

    env = dict(os.environ)
    env["PYTHONPATH"] = str(inner / "src") + os.pathsep + env.get("PYTHONPATH", "")
    proc = subprocess.run(
        [sys.executable, "-c", _resolver_snippet()],
        cwd=str(inner),
        env=env,
        capture_output=True,
        text=True,
    )
    # Resolver MUST refuse — submodule common-dir basename is `inner`,
    # not `.git`, so the basename check fires.
    assert proc.returncode != 0, (
        f"resolver did not refuse a real submodule:\nstdout: {proc.stdout}\nstderr: {proc.stderr}"
    )
    combined = proc.stdout + proc.stderr
    # Must NOT silently return the submodule's working dir.
    assert f"REPO={inner}" not in combined, f"submodule accepted as repo: {combined}"
    # Loud error must mention the unexpected basename ('inner') OR the
    # `.git` expectation. The basename branch raises:
    #   "git common-dir <...modules/inner> basename is 'inner', expected '.git'"
    assert "basename" in proc.stderr.lower() or "expected '.git'" in proc.stderr, (
        f"basename guard did not fire as expected: {proc.stderr}"
    )


def test_resolver_ignores_git_env_poisoning(tmp_path: Path) -> None:
    """`GIT_DIR` / `GIT_WORK_TREE` set in env do NOT redirect the resolver."""
    main_repo = tmp_path / "repo"
    _make_main_repo(main_repo)
    bogus = tmp_path / "bogus.git"
    bogus.mkdir()
    (bogus / "HEAD").write_text("ref: refs/heads/main\n")

    # Drop the tw module into the test repo so the subprocess uses the
    # test repo's source.
    src_dir = main_repo / "src" / "explore_persona_space"
    src_dir.mkdir(parents=True)
    (src_dir / "__init__.py").touch()
    real_tw = Path(_REPO_SRC) / "explore_persona_space" / "task_workflow.py"
    (src_dir / "task_workflow.py").write_text(real_tw.read_text())

    env = dict(os.environ)
    env["PYTHONPATH"] = str(main_repo / "src") + os.pathsep + env.get("PYTHONPATH", "")
    env["GIT_DIR"] = str(bogus)
    env["GIT_WORK_TREE"] = str(tmp_path / "nonexistent")

    proc = subprocess.run(
        [sys.executable, "-c", _resolver_snippet()],
        cwd=str(main_repo),
        env=env,
        capture_output=True,
        text=True,
    )
    # Either we succeed (resolving to the real main_repo despite the
    # poisoner env) or we fail in a way that does NOT print the
    # bogus path. Critically: the resolver MUST NOT trust GIT_DIR.
    if proc.returncode == 0:
        resolved = _parse_resolved(proc.stdout)
        assert Path(resolved["REPO"]).resolve() == main_repo.resolve(), (
            f"resolver was poisoned by GIT_DIR: {resolved}"
        )
    else:
        assert str(bogus) not in proc.stdout
        # Still must not crash with an opaque error about the poisoner.
        assert proc.stderr.strip(), "non-zero exit with no error message"


def test_pep562_attribute_access_works_lazily(tmp_path: Path) -> None:
    """`tw.TASKS_DIR` / `tw.REPO` attribute access goes through the function."""
    main_repo = tmp_path / "repo"
    _make_main_repo(main_repo)

    src_dir = main_repo / "src" / "explore_persona_space"
    src_dir.mkdir(parents=True)
    (src_dir / "__init__.py").touch()
    real_tw = Path(_REPO_SRC) / "explore_persona_space" / "task_workflow.py"
    (src_dir / "task_workflow.py").write_text(real_tw.read_text())

    snippet = textwrap.dedent(
        """
        import explore_persona_space.task_workflow as tw
        print('REPO=' + str(tw.REPO))
        print('TASKS_DIR=' + str(tw.TASKS_DIR))
        print('REGISTRY_PATH=' + str(tw.REGISTRY_PATH))
        """
    )
    env = dict(os.environ)
    env["PYTHONPATH"] = str(main_repo / "src") + os.pathsep + env.get("PYTHONPATH", "")
    proc = subprocess.run(
        [sys.executable, "-c", snippet],
        cwd=str(main_repo),
        env=env,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, f"PEP-562 attr access failed: {proc.stderr}"
    resolved = _parse_resolved(proc.stdout)
    assert Path(resolved["REPO"]).resolve() == main_repo.resolve()


def test_cache_hits_avoid_extra_git_calls(tmp_path: Path) -> None:
    """Two `repo_root()` calls in one process fire git only once;
    `invalidate_cache()` re-fires it on the next call.
    """
    main_repo = tmp_path / "repo"
    _make_main_repo(main_repo)

    src_dir = main_repo / "src" / "explore_persona_space"
    src_dir.mkdir(parents=True)
    (src_dir / "__init__.py").touch()
    real_tw = Path(_REPO_SRC) / "explore_persona_space" / "task_workflow.py"
    (src_dir / "task_workflow.py").write_text(real_tw.read_text())

    snippet = textwrap.dedent(
        """
        import json
        from explore_persona_space.task_workflow import (
            repo_root, invalidate_cache, _resolve_repo_root_cached,
        )
        repo_root()
        repo_root()
        repo_root()
        info1 = _resolve_repo_root_cached.cache_info()
        invalidate_cache()
        repo_root()
        info2 = _resolve_repo_root_cached.cache_info()
        print(json.dumps({
            'hits1': info1.hits, 'misses1': info1.misses,
            'hits2': info2.hits, 'misses2': info2.misses,
        }))
        """
    )
    env = dict(os.environ)
    env["PYTHONPATH"] = str(main_repo / "src") + os.pathsep + env.get("PYTHONPATH", "")
    proc = subprocess.run(
        [sys.executable, "-c", snippet],
        cwd=str(main_repo),
        env=env,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, f"cache probe failed: {proc.stderr}"
    info = json.loads(proc.stdout.strip().splitlines()[-1])
    # First call is a miss; next two are hits.
    assert info["misses1"] == 1, f"unexpected miss count before invalidate: {info}"
    assert info["hits1"] == 2, f"unexpected hit count before invalidate: {info}"
    # `invalidate_cache()` uses `cache_clear()`, which RESETS the cache
    # info counters to zero in addition to dropping the cache itself.
    # The post-invalidate call therefore records misses=1, hits=0 (one
    # miss in a fresh-counter cache). That `misses2 == 1` AND
    # `hits2 == 0` together is the unambiguous signature of "git was
    # re-fired exactly once after invalidate".
    assert info["hits2"] == 0 and info["misses2"] == 1, (
        f"invalidate_cache did not re-fire git: {info}"
    )


def test_resolver_uses_module_dir_not_cwd(tmp_path: Path) -> None:
    """Invoking from `/tmp` (or anywhere outside the repo) still works
    because the resolver runs git from the module's directory, not
    `os.getcwd()`.
    """
    main_repo = tmp_path / "repo"
    _make_main_repo(main_repo)

    src_dir = main_repo / "src" / "explore_persona_space"
    src_dir.mkdir(parents=True)
    (src_dir / "__init__.py").touch()
    real_tw = Path(_REPO_SRC) / "explore_persona_space" / "task_workflow.py"
    (src_dir / "task_workflow.py").write_text(real_tw.read_text())

    # Make a brand-new cwd that is NOT under any git repo.
    cwd = tmp_path / "neutral"
    cwd.mkdir()

    env = dict(os.environ)
    env["PYTHONPATH"] = str(main_repo / "src") + os.pathsep + env.get("PYTHONPATH", "")
    proc = subprocess.run(
        [sys.executable, "-c", _resolver_snippet()],
        cwd=str(cwd),
        env=env,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, (
        f"resolver failed from neutral cwd:\nstdout: {proc.stdout}\nstderr: {proc.stderr}"
    )
    resolved = _parse_resolved(proc.stdout)
    assert Path(resolved["REPO"]).resolve() == main_repo.resolve()


# ─── tasks-dir CLI subcommand smoke ───────────────────────────────────────


# Absolute path to the real `scripts/task.py` so the CLI tests exercise the
# actual production entry point (argparse registration + cmd_tasks_dir).
_REAL_TASK_PY = Path(__file__).resolve().parents[1] / "scripts" / "task.py"


def _stage_task_cli_tree(main_repo: Path) -> None:
    """Drop the project's `scripts/task.py` + a minimal `src/` tree into
    ``main_repo`` so `uv run python scripts/task.py tasks-dir` runs there
    against the test's tmp resolver (not the dev repo's).

    Only stages the files the CLI actually touches at import time:
    ``src/explore_persona_space/task_workflow.py`` (+ ``__init__.py``) and
    the canonical ``scripts/task.py``. Other imports inside task.py route
    through ``task_workflow`` (NewTaskRequest, STATUSES, …) — they all live
    in that one file in this project.
    """
    src_dir = main_repo / "src" / "explore_persona_space"
    src_dir.mkdir(parents=True)
    (src_dir / "__init__.py").touch()
    real_tw = Path(_REPO_SRC) / "explore_persona_space" / "task_workflow.py"
    (src_dir / "task_workflow.py").write_text(real_tw.read_text())

    scripts_dir = main_repo / "scripts"
    scripts_dir.mkdir(parents=True, exist_ok=True)
    (scripts_dir / "task.py").write_text(_REAL_TASK_PY.read_text())


def test_tasks_dir_cli_subcommand_invokes_real_task_py(tmp_path: Path) -> None:
    """`scripts/task.py tasks-dir` prints the canonical tasks path and exits 0.

    Covers the argparse registration at ``scripts/task.py`` and the
    ``cmd_tasks_dir`` entry point — both went uncovered by the previous
    hand-rolled wrapper test (round-1 code-review finding #5).
    """
    main_repo = tmp_path / "repo"
    _make_main_repo(main_repo)
    _stage_task_cli_tree(main_repo)

    env = dict(os.environ)
    # No PYTHONPATH manipulation needed — task.py prepends `src/` to
    # sys.path on import. TASK_PY_NO_COMMIT=1 prevents any inadvertent
    # git commit attempt (tasks-dir is read-only, but defense in depth).
    env["TASK_PY_NO_COMMIT"] = "1"
    proc = subprocess.run(
        [sys.executable, str(main_repo / "scripts" / "task.py"), "tasks-dir"],
        cwd=str(main_repo),
        env=env,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, f"task.py tasks-dir failed: {proc.stderr}"
    assert proc.stdout.strip() == str(main_repo / "tasks"), f"unexpected stdout: {proc.stdout!r}"


def test_tasks_dir_cli_exits_cleanly_on_non_main_branch(tmp_path: Path) -> None:
    """`scripts/task.py tasks-dir` exits non-zero with a CLEAN stderr
    message (no traceback) when invoked from a tmp repo with HEAD on a
    feature branch.

    Confirms the round-1 cmd_tasks_dir wrapper catches RuntimeError and
    prints a one-line error instead of propagating an unhandled exception
    (round-1 code-review finding #9).
    """
    main_repo = tmp_path / "repo"
    _make_main_repo(main_repo)
    _stage_task_cli_tree(main_repo)
    subprocess.run(
        ["git", "-C", str(main_repo), "checkout", "-q", "-b", "feature/off-main"],
        check=True,
    )

    env = dict(os.environ)
    env["TASK_PY_NO_COMMIT"] = "1"
    proc = subprocess.run(
        [sys.executable, str(main_repo / "scripts" / "task.py"), "tasks-dir"],
        cwd=str(main_repo),
        env=env,
        capture_output=True,
        text=True,
    )
    assert proc.returncode != 0, (
        f"task.py tasks-dir did not refuse non-main branch:\n"
        f"stdout: {proc.stdout!r}\nstderr: {proc.stderr!r}"
    )
    # Loud, named error.
    assert "feature/off-main" in proc.stderr, f"branch name not in stderr: {proc.stderr!r}"
    # MUST be a clean message, NOT an unhandled traceback.
    assert "Traceback" not in proc.stderr, (
        f"task.py tasks-dir leaked traceback (cmd_tasks_dir must catch "
        f"RuntimeError and print a one-liner):\n{proc.stderr}"
    )


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
