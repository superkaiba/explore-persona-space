"""Pin + dispatch tests for the merge-scoped gitleaks pre-commit hook (#1584).

Static pins (yaml.safe_load of ``.pre-commit-config.yaml`` + source greps of
``scripts/hooks/gitleaks_scoped.sh``) guard the stanza and the wrapper's
ordinary-path upstream-token equivalence, worktree-safe merge predicate, and
fail-fast discipline. Hermetic stub-binary functional tests (real git scratch
repos in ``tmp_path``, a stub ``gitleaks`` script first on ``PATH`` — no
network, no real scanner binary, no secret-shaped content) pin the dispatch
logic: merge commits scan ONLY the staged files that differ from BOTH
parents; ordinary commits run the exact upstream staged-git-mode scan;
scanner failures propagate on both branches; the repo-root ``.gitleaksignore``
is carried into the extract dir; an empty scan set exits 0.

Mirrors the local-hook pin pattern of
``test_workflow_lint_upload_or_true.py::test_precommit_hook_covers_new_offender_paths``.
"""

from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path

import yaml

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parent
_CONFIG = _REPO / ".pre-commit-config.yaml"
_WRAPPER = _REPO / "scripts" / "hooks" / "gitleaks_scoped.sh"


# --------------------------------------------------------------------------
# Static pins — stanza + wrapper source
# --------------------------------------------------------------------------


def _gitleaks_local_hook() -> dict:
    cfg = yaml.safe_load(_CONFIG.read_text(encoding="utf-8"))
    hooks = [
        h
        for repo in cfg["repos"]
        if repo["repo"] == "local"
        for h in repo["hooks"]
        if h.get("id") == "gitleaks"
    ]
    assert hooks, "no repo: local hook with id: gitleaks in .pre-commit-config.yaml"
    assert len(hooks) == 1, f"expected exactly one local gitleaks hook, got {len(hooks)}"
    return hooks[0]


def test_stanza_pins_merge_scoped_local_gitleaks() -> None:
    """The gitleaks stanza is the #1584 merge-scoped local wrapper: entry runs
    scripts/hooks/gitleaks_scoped.sh, language golang with the v8.30.1 module
    pin in additional_dependencies, pass_filenames false, always_run true."""
    hook = _gitleaks_local_hook()
    assert "scripts/hooks/gitleaks_scoped.sh" in hook.get("entry", ""), hook
    assert hook.get("language") == "golang", hook
    deps = hook.get("additional_dependencies", [])
    assert any("gitleaks" in d and "@v8.30.1" in d for d in deps), (
        f"v8.30.1 gitleaks module pin missing from additional_dependencies: {deps}"
    )
    assert hook.get("pass_filenames") is False, hook
    assert hook.get("always_run") is True, hook


def test_no_remote_gitleaks_stanza_remains() -> None:
    """No repos: entry points at github.com/gitleaks — prevents double
    scanning / silent reintroduction of the unscoped remote hook."""
    cfg = yaml.safe_load(_CONFIG.read_text(encoding="utf-8"))
    remotes = [r["repo"] for r in cfg["repos"] if "github.com/gitleaks" in r["repo"]]
    assert not remotes, f"remote gitleaks stanza reintroduced: {remotes}"


def test_wrapper_ordinary_path_upstream_equivalent() -> None:
    """The non-merge branch carries the upstream entry's tokens verbatim plus
    this repo's --config arg (ordinary-commit coverage unweakened)."""
    src = _WRAPPER.read_text(encoding="utf-8")
    assert "git --pre-commit --redact --staged --verbose" in src
    assert "--config .gitleaks.toml" in src


def test_wrapper_merge_predicate_worktree_safe() -> None:
    """Merge detection uses rev-parse (resolves per-worktree), never a
    hardcoded .git/MERGE_HEAD path (wrong in linked worktrees)."""
    src = _WRAPPER.read_text(encoding="utf-8")
    assert "rev-parse -q --verify MERGE_HEAD" in src
    assert ".git/MERGE_HEAD" not in src


def test_wrapper_fail_fast() -> None:
    """set -euo pipefail present; no scanner-invoking line swallows failure
    with `|| true` (a finding must block the commit)."""
    src = _WRAPPER.read_text(encoding="utf-8")
    assert "set -euo pipefail" in src
    scanner_lines = [ln for ln in src.splitlines() if re.search(r"gitleaks (git|dir)\b", ln)]
    assert scanner_lines, "no scanner-invoking lines found in wrapper"
    offenders = [ln for ln in scanner_lines if "|| true" in ln]
    assert not offenders, f"scanner line swallows failure: {offenders}"


# --------------------------------------------------------------------------
# Hermetic stub-binary functional tests
# --------------------------------------------------------------------------


def _git_env(extra: dict[str, str] | None = None) -> dict[str, str]:
    """Hermetic git env: no global/system config, fixed identity."""
    env = os.environ.copy()
    env["GIT_CONFIG_GLOBAL"] = os.devnull
    env["GIT_CONFIG_SYSTEM"] = os.devnull
    env["GIT_AUTHOR_NAME"] = env["GIT_COMMITTER_NAME"] = "gl1584-test"
    env["GIT_AUTHOR_EMAIL"] = env["GIT_COMMITTER_EMAIL"] = "gl1584@example.invalid"
    if extra:
        env.update(extra)
    return env


def _git(repo: Path, *args: str, env: dict[str, str]) -> None:
    subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True, env=env)


def _make_stub(tmp_path: Path) -> tuple[Path, Path]:
    """Stub `gitleaks` binary: records argv + cwd + a file listing of cwd to
    $GITLEAKS_STUB_RECORD, exits $GITLEAKS_STUB_EXIT (default 0)."""
    stub_dir = tmp_path / "stub-bin"
    stub_dir.mkdir()
    record = tmp_path / "stub-record.txt"
    stub = stub_dir / "gitleaks"
    stub.write_text(
        "#!/usr/bin/env bash\n"
        "{\n"
        '  echo "argv:$*"\n'
        '  echo "cwd:$(pwd)"\n'
        "  find . -type f | sort\n"
        '} > "$GITLEAKS_STUB_RECORD"\n'
        'exit "${GITLEAKS_STUB_EXIT:-0}"\n',
        encoding="utf-8",
    )
    stub.chmod(0o755)
    return stub_dir, record


def _run_wrapper(
    repo: Path,
    env: dict[str, str],
    stub_dir: Path,
    record: Path,
    stub_exit: str = "0",
) -> subprocess.CompletedProcess:
    wenv = dict(env)
    wenv["PATH"] = f"{stub_dir}:{wenv['PATH']}"
    wenv["GITLEAKS_STUB_RECORD"] = str(record)
    wenv["GITLEAKS_STUB_EXIT"] = stub_exit
    return subprocess.run(
        ["bash", str(_WRAPPER)], cwd=str(repo), env=wenv, capture_output=True, text=True
    )


def _init_repo(tmp_path: Path) -> tuple[Path, dict[str, str]]:
    repo = tmp_path / "repo"
    repo.mkdir()
    env = _git_env()
    _git(repo, "init", "-q", "-b", "main", env=env)
    (repo / ".gitleaks.toml").write_text("[extend]\nuseDefault = true\n", encoding="utf-8")
    (repo / "conflict.txt").write_text("base\n", encoding="utf-8")
    _git(repo, "add", "-A", env=env)
    _git(repo, "commit", "-qm", "base", env=env)
    return repo, env


def _make_conflicted_merge(tmp_path: Path, n_folded: int = 5) -> tuple[Path, dict[str, str]]:
    """Scratch repo mid conflicted merge: side folds `n_folded` files + a
    conflicting edit of conflict.txt; the resolution (a hand edit differing
    from both parents) is staged, MERGE_HEAD present, commit not concluded."""
    repo, env = _init_repo(tmp_path)
    _git(repo, "checkout", "-qb", "side", env=env)
    for i in range(n_folded):
        (repo / f"folded{i}.txt").write_text(f"folded {i}\n", encoding="utf-8")
    (repo / "conflict.txt").write_text("side\n", encoding="utf-8")
    _git(repo, "add", "-A", env=env)
    _git(repo, "commit", "-qm", "side", env=env)
    _git(repo, "checkout", "-q", "main", env=env)
    (repo / "conflict.txt").write_text("main\n", encoding="utf-8")
    _git(repo, "add", "conflict.txt", env=env)
    _git(repo, "commit", "-qm", "main", env=env)
    merge = subprocess.run(["git", "-C", str(repo), "merge", "side"], capture_output=True, env=env)
    assert merge.returncode != 0, "expected a conflicted merge"
    (repo / "conflict.txt").write_text("resolved\n", encoding="utf-8")
    _git(repo, "add", "conflict.txt", env=env)
    assert (repo / ".git" / "MERGE_HEAD").exists()
    return repo, env


def test_merge_path_scans_only_both_parent_diff(tmp_path: Path) -> None:
    """On a merge commit the wrapper invokes `dir` mode over an extract dir
    containing ONLY the staged files that differ from BOTH parents (the
    conflict resolution) — never the folded-advance files."""
    repo, env = _make_conflicted_merge(tmp_path)
    stub_dir, record = _make_stub(tmp_path)
    res = _run_wrapper(repo, env, stub_dir, record)
    assert res.returncode == 0, res.stderr
    assert "scanning 1 staged file(s) that differ from both parents" in res.stdout
    rec = record.read_text(encoding="utf-8").splitlines()
    assert rec[0].startswith("argv:dir ."), rec[0]
    listed = [ln for ln in rec if ln.startswith("./")]
    assert "./conflict.txt" in listed, listed
    assert not any("folded" in ln for ln in listed), (
        f"folded-advance files leaked into the scan set: {listed}"
    )


def test_ordinary_path_uses_staged_git_mode(tmp_path: Path) -> None:
    """With no merge in progress the wrapper execs the exact upstream
    staged-git-mode invocation plus this repo's --config."""
    repo, env = _init_repo(tmp_path)
    (repo / "conflict.txt").write_text("edited\n", encoding="utf-8")
    _git(repo, "add", "conflict.txt", env=env)
    stub_dir, record = _make_stub(tmp_path)
    res = _run_wrapper(repo, env, stub_dir, record)
    assert res.returncode == 0, res.stderr
    rec = record.read_text(encoding="utf-8").splitlines()
    assert rec[0] == (
        "argv:git --pre-commit --redact --staged --verbose --config .gitleaks.toml"
    ), rec[0]


def test_wrapper_propagates_scanner_failure(tmp_path: Path) -> None:
    """A scanner exit 1 (finding) makes the wrapper exit non-zero on BOTH
    branches — incl. the merge path's `(cd ... && gitleaks dir ...)` subshell."""
    # Merge branch.
    repo, env = _make_conflicted_merge(tmp_path)
    stub_dir, record = _make_stub(tmp_path)
    res = _run_wrapper(repo, env, stub_dir, record, stub_exit="1")
    assert res.returncode != 0, "merge-path scanner failure did not propagate"
    # Ordinary branch (fresh scratch repo under a sub-tmpdir).
    sub = tmp_path / "ordinary"
    sub.mkdir()
    repo2, env2 = _init_repo(sub)
    (repo2 / "conflict.txt").write_text("edited\n", encoding="utf-8")
    _git(repo2, "add", "conflict.txt", env=env2)
    stub_dir2, record2 = _make_stub(sub)
    res2 = _run_wrapper(repo2, env2, stub_dir2, record2, stub_exit="1")
    assert res2.returncode != 0, "ordinary-path scanner failure did not propagate"


def test_merge_path_carries_gitleaksignore(tmp_path: Path) -> None:
    """A repo-root .gitleaksignore is copied into the extract dir so
    path:rule:line fingerprints stay applicable in dir mode."""
    repo, env = _make_conflicted_merge(tmp_path)
    (repo / ".gitleaksignore").write_text(
        "# fingerprint carry-in test\nsome/path:rule-id:1\n", encoding="utf-8"
    )
    stub_dir, record = _make_stub(tmp_path)
    res = _run_wrapper(repo, env, stub_dir, record)
    assert res.returncode == 0, res.stderr
    listed = [ln for ln in record.read_text(encoding="utf-8").splitlines() if ln.startswith("./")]
    assert "./.gitleaksignore" in listed, listed


def test_merge_path_empty_scan_set_exits_zero(tmp_path: Path) -> None:
    """A merge whose staged tree matches a parent everywhere (empty
    differs-from-both-parents intersection) logs the skip line, exits 0, and
    never invokes the scanner."""
    repo, env = _init_repo(tmp_path)
    _git(repo, "checkout", "-qb", "side", env=env)
    (repo / "side_only.txt").write_text("side\n", encoding="utf-8")
    _git(repo, "add", "-A", env=env)
    _git(repo, "commit", "-qm", "side", env=env)
    _git(repo, "checkout", "-q", "main", env=env)
    # --no-commit --no-ff leaves MERGE_HEAD present on a clean merge; every
    # staged file matches MERGE_HEAD, so the intersection is empty.
    _git(repo, "merge", "--no-commit", "--no-ff", "side", env=env)
    assert (repo / ".git" / "MERGE_HEAD").exists()
    stub_dir, record = _make_stub(tmp_path)
    res = _run_wrapper(repo, env, stub_dir, record)
    assert res.returncode == 0, res.stderr
    assert "no staged file differs from both parents; skipping scan" in res.stdout
    assert not record.exists(), "scanner was invoked on an empty scan set"
