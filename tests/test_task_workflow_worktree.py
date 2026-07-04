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
import re
import subprocess
import sys
import textwrap
import time
from pathlib import Path

import pytest

# Absolute path to this repo's `src/` so subprocesses can `import
# explore_persona_space.task_workflow`. We pass this on PYTHONPATH below.
_REPO_SRC = str(Path(__file__).resolve().parents[1] / "src")

# Bounded live-rebase wait knobs (#996). Every test that exercises the
# resolver's DETACHED path explicitly sets or POPS both in the child env —
# a knob leaked into the CI env (e.g. a non-float value, which the resolver
# parses fail-loud) would otherwise perturb knob-default tests.
_REBASE_WAIT_ENV = "EPM_TASKPY_REBASE_WAIT_SECONDS"
_REBASE_POLL_ENV = "EPM_TASKPY_REBASE_POLL_SECONDS"


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


def test_branch_guard_routes_non_main_to_managed_worktree(tmp_path: Path) -> None:
    """Primary HEAD on a real feature branch → AUTO-ROUTE to the managed
    main-pinned worktree instead of refusing.

    This test's assertion FLIPPED on 2026-05-28 (issue #15): the resolver used
    to raise a loud RuntimeError naming the branch; it now returns the managed
    worktree path under ``.claude/worktrees/_task-main-pin`` so markers/commits
    succeed while the primary is parked off-main. The managed worktree's
    detached HEAD is pinned to `main` so commits land on `main` (covered by
    ``test_routed_commit_lands_on_main_not_feature_branch``). The loud-refusal
    behavior is preserved ONLY for the non-routable states (detached HEAD,
    missing tasks/, no local `main`) — see the other guard tests.
    """
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
    assert proc.returncode == 0, (
        f"resolver refused instead of auto-routing:\nstdout: {proc.stdout}\nstderr: {proc.stderr}"
    )
    resolved = _parse_resolved(proc.stdout)
    managed = (main_repo / ".claude" / "worktrees" / "_task-main-pin").resolve()
    assert Path(resolved["REPO"]).resolve() == managed, (
        f"routed REPO is not the managed worktree: {resolved}"
    )
    assert Path(resolved["TASKS_DIR"]).resolve() == (managed / "tasks").resolve()
    # The managed worktree must actually exist on disk with a tasks/ dir.
    assert (managed / "tasks").is_dir(), "managed worktree missing tasks/ after routing"
    # And it must be DETACHED at main's tip — never holding the `main` branch
    # (which would block the primary from `git checkout main`).
    head = subprocess.run(
        ["git", "-C", str(managed), "symbolic-ref", "--quiet", "HEAD"],
        capture_output=True,
        text=True,
    )
    assert head.returncode != 0, (
        f"managed worktree holds a branch (HEAD={head.stdout!r}); expected DETACHED HEAD"
    )


def test_branch_guard_distinct_error_on_detached_head(tmp_path: Path) -> None:
    """Detached HEAD (no rebase marker) → DISTINCT error mentioning
    'detached'. Under the #996 bounded-wait this path pays at most one 0.5s
    marker-less grace re-probe before the refusal — semantics unchanged."""
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
    # Knob hygiene (#996): pop both rebase-wait knobs so a leaked CI value
    # (e.g. a non-float, which the resolver parses fail-loud) cannot perturb
    # this knob-DEFAULT detached refusal.
    env.pop(_REBASE_WAIT_ENV, None)
    env.pop(_REBASE_POLL_ENV, None)
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


# ─── primary_checkout_root (#844) ──────────────────────────────────────────


def test_primary_checkout_root_resolves_main_from_worktree(tmp_path: Path) -> None:
    """From a WORKTREE COPY of the module (the #844 shape: the executed file
    lives inside a linked worktree), `primary_checkout_root()` returns the
    PRIMARY checkout root, not the worktree directory.

    Subprocess-shaped like ``test_resolves_main_repo_from_worktree`` (the
    bug class manifests only at fresh-process import), but with PYTHONPATH
    pointing at the WORKTREE's src copy so ``_MODULE_DIR`` is inside the
    worktree — exactly what a worktree copy of a `scripts/` consumer sees.
    """
    main_repo = tmp_path / "repo"
    _make_main_repo(main_repo)

    # Drop + COMMIT the module on main so the worktree (created next)
    # carries its own copy of src/.
    src_dir = main_repo / "src" / "explore_persona_space"
    src_dir.mkdir(parents=True)
    (src_dir / "__init__.py").touch()
    real_tw = Path(_REPO_SRC) / "explore_persona_space" / "task_workflow.py"
    (src_dir / "task_workflow.py").write_text(real_tw.read_text())
    subprocess.run(["git", "-C", str(main_repo), "add", "src"], check=True)
    subprocess.run(["git", "-C", str(main_repo), "commit", "-q", "-m", "src"], check=True)

    worktree = tmp_path / "wt-feature"
    subprocess.run(
        ["git", "-C", str(main_repo), "worktree", "add", "-b", "feature/x", str(worktree)],
        check=True,
        capture_output=True,
    )

    # PYTHONPATH -> the WORKTREE's src copy (not the primary's).
    env = dict(os.environ)
    env["PYTHONPATH"] = str(worktree / "src") + os.pathsep + env.get("PYTHONPATH", "")
    snippet = textwrap.dedent(
        """
        from explore_persona_space.task_workflow import primary_checkout_root
        print('PRIMARY=' + str(primary_checkout_root()))
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
        f"primary_checkout_root failed:\nstdout: {proc.stdout}\nstderr: {proc.stderr}"
    )
    resolved = _parse_resolved(proc.stdout)
    assert Path(resolved["PRIMARY"]).resolve() == main_repo.resolve(), (
        f"worktree module copy did not resolve the primary checkout: {resolved}"
    )


def test_primary_checkout_root_ignores_branch_state(tmp_path: Path) -> None:
    """Primary checkout parked on a feature branch: `primary_checkout_root()`
    STILL returns the primary (no `_task-main-pin` routing, no raise), while
    `repo_root()` on the same fixture routes to the managed worktree — pins
    the semantic difference between the two resolvers (#844 §4a).
    """
    main_repo = tmp_path / "repo"
    _make_main_repo(main_repo)
    subprocess.run(
        ["git", "-C", str(main_repo), "checkout", "-q", "-b", "feature/off-main"],
        check=True,
    )

    src_dir = main_repo / "src" / "explore_persona_space"
    src_dir.mkdir(parents=True)
    (src_dir / "__init__.py").touch()
    real_tw = Path(_REPO_SRC) / "explore_persona_space" / "task_workflow.py"
    (src_dir / "task_workflow.py").write_text(real_tw.read_text())

    env = dict(os.environ)
    env["PYTHONPATH"] = str(main_repo / "src") + os.pathsep + env.get("PYTHONPATH", "")
    snippet = textwrap.dedent(
        """
        from explore_persona_space.task_workflow import primary_checkout_root, repo_root
        print('PRIMARY=' + str(primary_checkout_root()))
        print('REPO=' + str(repo_root()))
        """
    )
    proc = subprocess.run(
        [sys.executable, "-c", snippet],
        cwd=str(main_repo),
        env=env,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, (
        f"resolvers failed off-main:\nstdout: {proc.stdout}\nstderr: {proc.stderr}"
    )
    resolved = _parse_resolved(proc.stdout)
    # primary_checkout_root: the PRIMARY, regardless of branch.
    assert Path(resolved["PRIMARY"]).resolve() == main_repo.resolve(), (
        f"primary_checkout_root routed/deviated off-main: {resolved}"
    )
    # repo_root: routes to the managed main-pin worktree, as before.
    managed = (main_repo / ".claude" / "worktrees" / "_task-main-pin").resolve()
    assert Path(resolved["REPO"]).resolve() == managed, (
        f"repo_root no longer routes off-main (behavior change!): {resolved}"
    )


def test_invalidate_cache_clears_primary_checkout_cache(tmp_path: Path) -> None:
    """`invalidate_cache()` clears the `_primary_checkout_cached` LRU too —
    the next `primary_checkout_root()` call re-fires git (#844 §4a)."""
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
            primary_checkout_root, invalidate_cache, _primary_checkout_cached,
        )
        primary_checkout_root()
        primary_checkout_root()
        primary_checkout_root()
        info1 = _primary_checkout_cached.cache_info()
        invalidate_cache()
        primary_checkout_root()
        info2 = _primary_checkout_cached.cache_info()
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
    assert info["misses1"] == 1 and info["hits1"] == 2, f"unexpected pre-invalidate: {info}"
    # cache_clear() resets counters; one fresh miss = git re-fired once.
    assert info["hits2"] == 0 and info["misses2"] == 1, (
        f"invalidate_cache did not clear _primary_checkout_cached: {info}"
    )


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


def test_tasks_dir_cli_routes_on_non_main_branch(tmp_path: Path) -> None:
    """`scripts/task.py tasks-dir` on a feature-branch primary now AUTO-ROUTES
    to the managed main-pinned worktree and exits 0 (assertion flipped
    2026-05-28, issue #15 — it used to refuse).

    Confirms the off-main ergonomics fix is reachable through the real CLI
    entry point, not only the in-process resolver.
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
    assert proc.returncode == 0, (
        f"task.py tasks-dir refused instead of routing:\n"
        f"stdout: {proc.stdout!r}\nstderr: {proc.stderr!r}"
    )
    expected = str(main_repo / ".claude" / "worktrees" / "_task-main-pin" / "tasks")
    assert proc.stdout.strip() == expected, f"unexpected routed tasks-dir: {proc.stdout!r}"


def test_cli_exits_cleanly_on_detached_head(tmp_path: Path) -> None:
    """`scripts/task.py tasks-dir` exits non-zero with a CLEAN stderr message
    (no traceback) when the primary checkout HEAD is DETACHED — a state that
    is NOT auto-routable and must still fail loud.

    Exercises the top-level RuntimeError catch in ``main()`` (every subcommand,
    not only ``cmd_tasks_dir``), added 2026-05-28 with the off-main routing so
    the still-refused non-routable states surface a one-liner rather than a raw
    traceback.
    """
    main_repo = tmp_path / "repo"
    _make_main_repo(main_repo)
    _stage_task_cli_tree(main_repo)
    subprocess.run(
        ["git", "-C", str(main_repo), "checkout", "-q", "--detach", "HEAD"],
        check=True,
    )

    env = dict(os.environ)
    env["TASK_PY_NO_COMMIT"] = "1"
    # Knob hygiene (#996): pop both rebase-wait knobs so a leaked CI value
    # cannot perturb this knob-DEFAULT detached refusal.
    env.pop(_REBASE_WAIT_ENV, None)
    env.pop(_REBASE_POLL_ENV, None)
    proc = subprocess.run(
        [sys.executable, str(main_repo / "scripts" / "task.py"), "tasks-dir"],
        cwd=str(main_repo),
        env=env,
        capture_output=True,
        text=True,
    )
    assert proc.returncode != 0, (
        f"task.py tasks-dir did not refuse detached HEAD:\n"
        f"stdout: {proc.stdout!r}\nstderr: {proc.stderr!r}"
    )
    # Loud, named error.
    assert "detached" in proc.stderr.lower(), f"'detached' not in stderr: {proc.stderr!r}"
    # MUST be a clean message, NOT an unhandled traceback.
    assert "Traceback" not in proc.stderr, (
        f"task.py leaked a traceback on the non-routable detached-HEAD state:\n{proc.stderr}"
    )


# ─── End-to-end off-main routing (issue #15) ──────────────────────────────


def _commit_cli_tree(main_repo: Path) -> None:
    """Stage + commit the task.py CLI tree onto `main` so subsequent feature
    branches inherit it and `main` is a real commit to route through."""
    subprocess.run(["git", "-C", str(main_repo), "add", "."], check=True)
    subprocess.run(
        ["git", "-C", str(main_repo), "commit", "-q", "-m", "add task.py cli tree"],
        check=True,
    )


def _run_task_cli(main_repo: Path, *cli_args: str) -> subprocess.CompletedProcess[str]:
    """Invoke the real `scripts/task.py` in ``main_repo`` (commits enabled)."""
    env = dict(os.environ)
    return subprocess.run(
        [sys.executable, str(main_repo / "scripts" / "task.py"), *cli_args],
        cwd=str(main_repo),
        env=env,
        capture_output=True,
        text=True,
    )


def test_routed_commit_lands_on_main_not_feature_branch(tmp_path: Path) -> None:
    """The HARD INVARIANT for issue #15: a real task.py mutation run while the
    primary checkout is parked on a feature branch lands its commit on `main`
    (via the managed main-pinned worktree) and NEVER strands it on the feature
    branch — and the primary can still `git checkout main` afterwards (no
    leaked managed worktree holding the branch).
    """
    main_repo = tmp_path / "repo"
    _make_main_repo(main_repo)
    _stage_task_cli_tree(main_repo)
    _commit_cli_tree(main_repo)
    main_tip_before = subprocess.run(
        ["git", "-C", str(main_repo), "rev-parse", "main"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()

    # Park the primary on a feature branch.
    subprocess.run(
        ["git", "-C", str(main_repo), "checkout", "-q", "-b", "feature/dash"],
        check=True,
    )

    # Run a real mutation (create a task). This commits via the routed path.
    created = _run_task_cli(main_repo, "new", "--kind", "infra", "--title", "routed task")
    assert created.returncode == 0, f"task.py new failed: {created.stderr}"

    # (a) The commit advanced `main`.
    main_tip_after = subprocess.run(
        ["git", "-C", str(main_repo), "rev-parse", "main"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    assert main_tip_after != main_tip_before, "task.py mutation did not advance `main`"

    # (b) The body.md exists on `main`'s tree (the write landed where it
    #     committed — no divergence between write root and commit root).
    tree = subprocess.run(
        ["git", "-C", str(main_repo), "ls-tree", "-r", "--name-only", "main"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    assert any("body.md" in line for line in tree.splitlines()), (
        f"routed task body.md not present on `main` tree:\n{tree}"
    )

    # (c) NOT stranded on the feature branch: feature/dash must have NO commits
    #     that `main` lacks under tasks/.
    stranded = subprocess.run(
        ["git", "-C", str(main_repo), "log", "--oneline", "main..feature/dash", "--", "tasks/"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    assert stranded == "", f"task commit stranded on feature branch:\n{stranded}"

    # (d) The primary can STILL return to `main` (managed worktree is detached,
    #     not holding the `main` branch).
    back = subprocess.run(
        ["git", "-C", str(main_repo), "checkout", "main"],
        capture_output=True,
        text=True,
    )
    assert back.returncode == 0, (
        f"primary could not checkout main after routing (leaked branch lock?):\n{back.stderr}"
    )


def test_routed_post_marker_succeeds_off_main(tmp_path: Path) -> None:
    """The motivating use case: posting a progress marker while the primary is
    parked off-main succeeds (it used to be silently skipped + surfaced inline)
    and the marker's commit lands on `main`.
    """
    main_repo = tmp_path / "repo"
    _make_main_repo(main_repo)
    _stage_task_cli_tree(main_repo)
    _commit_cli_tree(main_repo)

    # Create a task on `main` first so there is something to post a marker on.
    created = _run_task_cli(main_repo, "new", "--kind", "infra", "--title", "marker target")
    assert created.returncode == 0, f"setup task.py new failed: {created.stderr}"
    task_id = created.stdout.strip().lstrip("#")

    # Park primary off-main, then post a marker through the routed path.
    subprocess.run(
        ["git", "-C", str(main_repo), "checkout", "-q", "-b", "feat/markers"],
        check=True,
    )
    posted = _run_task_cli(
        main_repo, "post-marker", task_id, "epm:smoke", "--note", "routed marker"
    )
    assert posted.returncode == 0, f"routed post-marker failed: {posted.stderr}"

    # The marker landed in events.jsonl on `main`.
    show = subprocess.run(
        ["git", "-C", str(main_repo), "show", f"main:tasks/proposed/{task_id}/events.jsonl"],
        capture_output=True,
        text=True,
    )
    # The task may live under whatever status `new` defaults to; locate it.
    if show.returncode != 0:
        tree = subprocess.run(
            ["git", "-C", str(main_repo), "ls-tree", "-r", "--name-only", "main"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout
        events_path = next(
            (ln for ln in tree.splitlines() if ln.endswith(f"/{task_id}/events.jsonl")),
            None,
        )
        assert events_path is not None, f"events.jsonl not on `main` tree:\n{tree}"
        show = subprocess.run(
            ["git", "-C", str(main_repo), "show", f"main:{events_path}"],
            capture_output=True,
            text=True,
            check=True,
        )
    assert "epm:smoke" in show.stdout, f"routed marker not on `main`: {show.stdout!r}"

    # Primary can still return to main.
    back = subprocess.run(
        ["git", "-C", str(main_repo), "checkout", "main"],
        capture_output=True,
        text=True,
    )
    assert back.returncode == 0, (
        f"primary could not checkout main after routed marker: {back.stderr}"
    )


def test_managed_worktree_dodges_stale_worktree_audit() -> None:
    """The managed worktree name `_task-main-pin` must NOT match the
    stale-worktree audit's target regex (`issue-<N>` / `agent-<hex>` /
    `wf_<id>`), so the audit never reaps the routing worktree out from under a
    live off-main session.

    Reads the regex source directly from ``scripts/worktree_audit.py`` rather
    than importing the module: ``worktree_audit`` resolves ``repo_root()`` at
    import time, which would fire the dev repo's branch guard and couple this
    test to the dev checkout's branch state.
    """
    audit_src = (Path(__file__).resolve().parents[1] / "scripts" / "worktree_audit.py").read_text()
    m = re.search(r"_TARGET_NAME_RE\s*=\s*re\.compile\(\s*r\"([^\"]+)\"", audit_src)
    assert m is not None, "could not locate _TARGET_NAME_RE in scripts/worktree_audit.py"
    target_re = re.compile(m.group(1))
    assert target_re.match("_task-main-pin") is None, (
        "managed main-pin worktree name matches the audit target regex — it "
        "could be reaped while routing is active"
    )


# ─── Bounded live-rebase wait before the detached-HEAD refusal (#996) ──────
#
# A concurrent `git pull --rebase` on the shared repo root detaches the
# primary HEAD for the rebase duration; the resolver now bounded-waits
# (default 120s, `EPM_TASKPY_REBASE_WAIT_SECONDS`; 0 disables) when a live
# rebase state dir (`.git/rebase-merge` / `rebase-apply`) is present, then
# re-runs the FULL branch guard. These tests drive the resolver as a CHILD
# process (the copied-module subprocess fixture) so the pytest parent can
# mutate the repo mid-wait.


def _stage_tw_module(main_repo: Path) -> None:
    """Drop the real task_workflow.py into ``main_repo`` (uncommitted) so a
    subprocess can import the test repo's copy via PYTHONPATH."""
    src_dir = main_repo / "src" / "explore_persona_space"
    src_dir.mkdir(parents=True)
    (src_dir / "__init__.py").touch()
    real_tw = Path(_REPO_SRC) / "explore_persona_space" / "task_workflow.py"
    (src_dir / "task_workflow.py").write_text(real_tw.read_text())


def _child_env(main_repo: Path, *, wait: str | None, poll: str | None) -> dict[str, str]:
    """Child env with PYTHONPATH at the test repo's src and BOTH rebase
    knobs explicitly set (str) or POPPED (None) — test-env knob hygiene."""
    env = dict(os.environ)
    env["PYTHONPATH"] = str(main_repo / "src") + os.pathsep + env.get("PYTHONPATH", "")
    for key, value in ((_REBASE_WAIT_ENV, wait), (_REBASE_POLL_ENV, poll)):
        if value is None:
            env.pop(key, None)
        else:
            env[key] = value
    return env


def _detach_with_rebase_marker(main_repo: Path) -> Path:
    """Detach the primary HEAD and fake a live primary-checkout rebase by
    creating `.git/rebase-merge`. Returns the marker dir path."""
    subprocess.run(
        ["git", "-C", str(main_repo), "checkout", "-q", "--detach", "HEAD"],
        check=True,
    )
    rebase_dir = main_repo / ".git" / "rebase-merge"
    rebase_dir.mkdir()
    return rebase_dir


def _spawn_resolver_and_await_wait_announcement(
    main_repo: Path,
    stderr_path: Path,
    env: dict[str, str],
) -> subprocess.Popen[str]:
    """Start the resolver child (stderr → file) and block until its wait
    announcement appears in stderr — proof it OBSERVED detached+rebasing —
    or the child exits early / ~15s passes. Waiting on the announcement
    (not a fixed sleep) kills the race where a slow child start would let
    the parent finish the fake rebase before the child's first probe, so
    the child would never wait at all.
    """
    stderr_f = open(stderr_path, "w")  # noqa: SIM115 — handle owned by the child process
    child = subprocess.Popen(
        [sys.executable, "-c", _resolver_snippet()],
        cwd=str(main_repo),
        env=env,
        stdout=subprocess.PIPE,
        stderr=stderr_f,
        text=True,
    )
    stderr_f.close()  # the child holds its own dup of the fd
    deadline = time.monotonic() + 15.0
    while time.monotonic() < deadline:
        if "waiting up to" in stderr_path.read_text():
            break
        if child.poll() is not None:
            break  # child exited early — the caller's asserts surface it
        time.sleep(0.05)
    return child


def _finish_child(child: subprocess.Popen[str]) -> tuple[int, str]:
    """Wait for the resolver child (bounded); kill on overrun. Returns
    (returncode, stdout)."""
    try:
        stdout, _ = child.communicate(timeout=30)
    except subprocess.TimeoutExpired:
        child.kill()
        stdout, _ = child.communicate()
    return child.returncode, stdout


def test_detached_live_rebase_waits_then_succeeds(tmp_path: Path) -> None:
    """Detached HEAD + LIVE rebase marker → the resolver WAITS (loudly) and
    succeeds once the rebase 'finishes' (#996 criteria 1 + 6).

    The parent finishes the fake rebase in the race-safe order: re-attach
    HEAD FIRST via plumbing (`git symbolic-ref` — porcelain `checkout` may
    balk at the fake in-progress rebase state), THEN remove the marker, so
    the child only ever observes "attached" (success) or "detached+marker"
    (keep waiting). The stderr assertion also empirically pins the
    assumption that the `_log.warning` announcement reaches a child
    process's stderr via logging's lastResort handler (criterion 6).
    """
    main_repo = tmp_path / "repo"
    _make_main_repo(main_repo)
    _stage_tw_module(main_repo)
    rebase_dir = _detach_with_rebase_marker(main_repo)

    stderr_path = tmp_path / "child-stderr.txt"
    child = _spawn_resolver_and_await_wait_announcement(
        main_repo, stderr_path, _child_env(main_repo, wait="15", poll="0.1")
    )
    # Finish the fake rebase: HEAD re-attach FIRST, marker removal SECOND.
    subprocess.run(
        ["git", "-C", str(main_repo), "symbolic-ref", "HEAD", "refs/heads/main"],
        check=True,
    )
    rebase_dir.rmdir()
    returncode, stdout = _finish_child(child)
    stderr = stderr_path.read_text()

    assert returncode == 0, f"resolver failed:\nstdout: {stdout}\nstderr: {stderr}"
    resolved = _parse_resolved(stdout)
    assert Path(resolved["REPO"]).resolve() == main_repo.resolve(), (
        f"post-wait resolution is not the primary root: {resolved}"
    )
    # Criterion 6: the loud wait announcement reached the child's stderr.
    assert "waiting up to" in stderr, f"wait announcement missing from child stderr: {stderr!r}"


def test_knob_zero_marker_present_refuses_immediately(tmp_path: Path) -> None:
    """The DISCRIMINATIVE knob=0 cell (#996 criterion 5): detached + rebase
    marker PRESENT + `EPM_TASKPY_REBASE_WAIT_SECONDS=0` → IMMEDIATE refusal
    with the byte-identical OLD message — no marker probe, no grace, no
    wait/timeout wording, wall time far under the 120s default bound.

    (The existing ``test_branch_guard_distinct_error_on_detached_head``
    keeps pinning the detached + NO-marker default-knob refusal.)
    """
    main_repo = tmp_path / "repo"
    _make_main_repo(main_repo)
    _stage_tw_module(main_repo)
    _detach_with_rebase_marker(main_repo)

    start = time.monotonic()
    proc = subprocess.run(
        [sys.executable, "-c", _resolver_snippet()],
        cwd=str(main_repo),
        env=_child_env(main_repo, wait="0", poll="0.1"),
        capture_output=True,
        text=True,
    )
    wall = time.monotonic() - start
    assert proc.returncode != 0
    # Byte-identical OLD refusal message.
    assert "is detached; re-attach to 'main' before running task.py." in proc.stderr, (
        f"old refusal message missing: {proc.stderr!r}"
    )
    # No wait/timeout wording — the knob=0 branch fires BEFORE any probe.
    assert "waiting up to" not in proc.stderr, proc.stderr
    assert "rebase state dir" not in proc.stderr, proc.stderr
    assert wall < 5.0, f"knob=0 refusal took {wall:.1f}s (expected immediate)"


def test_detached_stale_rebase_marker_refuses_after_bound(tmp_path: Path) -> None:
    """Detached + a rebase marker that never clears → refusal after the
    bound, with EVERY criterion-3 diagnostic in the message: `detached`, the
    state-dir name, the formatted wait duration, the CRASHED/stale-rebase
    hedge, and the `git rebase --abort` remedy (#996 criterion 3).
    """
    main_repo = tmp_path / "repo"
    _make_main_repo(main_repo)
    _stage_tw_module(main_repo)
    _detach_with_rebase_marker(main_repo)  # never cleared

    start = time.monotonic()
    proc = subprocess.run(
        [sys.executable, "-c", _resolver_snippet()],
        cwd=str(main_repo),
        env=_child_env(main_repo, wait="1", poll="0.1"),
        capture_output=True,
        text=True,
    )
    wall = time.monotonic() - start
    assert proc.returncode != 0
    for needle in (
        "detached",
        "rebase-merge",
        "waiting 1s",
        "CRASHED rebase",
        "rebase --abort",
    ):
        assert needle in proc.stderr, f"{needle!r} missing from stderr: {proc.stderr!r}"
    # Subprocess startup + imports on a loaded VM ride on top of the 1s
    # bound; 10s still discriminates decisively vs the 120s default.
    assert wall <= 10.0, f"stale-marker refusal took {wall:.1f}s (bound was 1s)"


def test_post_wait_nonmain_reattach_routes_managed_worktree(tmp_path: Path) -> None:
    """Post-wait re-attach to a NON-main branch → the re-entered branch
    guard routes through the existing `_ensure_managed_main_worktree` path
    (#844), not a refusal (#996 criterion 4). No monkeypatch — the
    subprocess fixture exercises the real routing.
    """
    main_repo = tmp_path / "repo"
    _make_main_repo(main_repo)
    _stage_tw_module(main_repo)
    rebase_dir = _detach_with_rebase_marker(main_repo)

    stderr_path = tmp_path / "child-stderr.txt"
    child = _spawn_resolver_and_await_wait_announcement(
        main_repo, stderr_path, _child_env(main_repo, wait="15", poll="0.1")
    )
    # Finish the fake rebase onto a FEATURE branch: create it, re-attach
    # HEAD to it FIRST (plumbing), THEN remove the marker.
    subprocess.run(["git", "-C", str(main_repo), "branch", "feature-x"], check=True)
    subprocess.run(
        ["git", "-C", str(main_repo), "symbolic-ref", "HEAD", "refs/heads/feature-x"],
        check=True,
    )
    rebase_dir.rmdir()
    returncode, stdout = _finish_child(child)
    stderr = stderr_path.read_text()

    assert returncode == 0, (
        f"resolver refused instead of routing:\nstdout: {stdout}\nstderr: {stderr}"
    )
    resolved = _parse_resolved(stdout)
    managed = (main_repo / ".claude" / "worktrees" / "_task-main-pin").resolve()
    assert Path(resolved["REPO"]).resolve() == managed, (
        f"post-wait non-main re-attach did not route to the managed worktree: {resolved}"
    )
    assert Path(resolved["TASKS_DIR"]).resolve() == (managed / "tasks").resolve()


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
