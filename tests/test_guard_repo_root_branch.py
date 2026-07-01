"""End-to-end tests for the ``scripts/guard_repo_root_branch.sh`` PreToolUse hook.

The guard blocks any git command that would move the SHARED repo-root working
tree off ``main`` OR detach its HEAD, because the repo root is the canonical
commit target for ``scripts/task.py`` and every concurrent VM Claude session
(all assume ``HEAD==main``). A detached repo-root HEAD additionally crashes
``task_workflow``'s main-worktree resolver.

These tests drive the script exactly as the harness does: stdin PreToolUse JSON
``{"tool_input": {"command": <cmd>}}`` -> exit 2 (blocked) or exit 0 (allowed).
This mirrors the subprocess-drives-script convention of the sibling
``tests/test_*guard*`` files.

The guard's on-main gate exits 0 when the repo-root HEAD is already off ``main``
(it never traps a user recovering from an already-detached/off-main state), so
the BLOCK-path tests are guarded by ``@on_main``. The ALLOW-path and fail-soft
tests run regardless — the guard must never trap those shapes in either state.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = _REPO_ROOT / "scripts" / "guard_repo_root_branch.sh"

# The script hardcodes REPO=/home/thomasjiralerspong/explore-persona-space and
# runs ``git -C "$REPO" ...`` for its branch/commit-ish resolution + on-main
# gate — so ref/branch/tag existence and the on-main check are read against that
# canonical checkout, NOT this worktree.
REPO = Path("/home/thomasjiralerspong/explore-persona-space")


def _run(cmd: str) -> int:
    """Feed ``cmd`` to the guard via PreToolUse JSON; return its exit code."""
    payload = json.dumps({"tool_input": {"command": cmd}})
    return subprocess.run([str(SCRIPT)], input=payload, text=True, capture_output=True).returncode


def _run_raw(payload: str) -> int:
    """Feed a raw (possibly malformed) stdin payload; return the exit code."""
    return subprocess.run([str(SCRIPT)], input=payload, text=True, capture_output=True).returncode


def _git(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(["git", "-C", str(REPO), *args], capture_output=True, text=True)


def _on_main() -> bool:
    r = _git("symbolic-ref", "--short", "HEAD")
    return r.returncode == 0 and r.stdout.strip() == "main"


def _repo_head_sha() -> str:
    """Real short SHA of the canonical repo-root HEAD (resolves via ``$REPO``)."""
    return _git("rev-parse", "--short", "HEAD").stdout.strip()


on_main = pytest.mark.skipif(
    not _on_main(),
    reason="guard's on-main gate exits 0 when the repo-root HEAD is off main",
)


@pytest.fixture
def throwaway_branch():
    """A real local branch (in ``$REPO``) so ``refs/heads/<name>`` resolves.

    Created pointing at HEAD (no checkout, no tree change) and deleted on
    teardown, so the branch-switch-regression assertion is hermetic rather than
    depending on a repo-specific branch name existing.
    """
    name = "eps-test-throwaway-guard-796"
    _git("branch", "-f", name, "HEAD")
    try:
        yield name
    finally:
        _git("branch", "-D", name)


@pytest.fixture
def throwaway_tag():
    """A real local tag (in ``$REPO``) so ``<tag>^{commit}`` resolves.

    Detaching to a tag is a real detach shape; a fabricated non-existent tag
    would fail-soft (exit 0) and never exercise the block path — so the tag must
    genuinely exist. Created pointing at HEAD and deleted on teardown.
    """
    name = "eps-test-throwaway-tag-796"
    _git("tag", "-f", name, "HEAD")
    try:
        yield name
    finally:
        _git("tag", "-d", name)


# ---------------------------------------------------------------------------
# MUST BLOCK — detach-inducing shapes (the whole point of #796)
# ---------------------------------------------------------------------------
@on_main
@pytest.mark.parametrize(
    "template",
    [
        "git checkout {sha}",  # bare sha detach
        "git checkout --detach",  # explicit --detach, no ref
        "git checkout --detach {sha}",  # explicit --detach + ref
        "git checkout -f {sha}",  # flag-prefixed detach (-f)
        "git checkout -q {sha}",  # flag-prefixed detach (-q)
        "git checkout -p {sha}",  # flag-prefixed detach (-p)
        "git checkout -m {sha}",  # flag-prefixed detach (-m)
        "git switch --detach {sha}",  # switch --detach
        "git switch -d {sha}",  # switch -d
        "git switch -d main",  # detach AT main (branch-only detector allows `main`)
        "git checkout HEAD~1",  # relative rev
        "git checkout HEAD@{{0}}",  # reflog rev ({{0}} -> {0} after .format)
        "git checkout origin/main",  # remote-tracking ref
    ],
)
def test_detach_shapes_block(template):
    assert _run(template.format(sha=_repo_head_sha())) == 2


@on_main
def test_detach_to_real_tag_blocks(throwaway_tag):
    # A real local tag resolves to a commit-ish -> detach -> block.
    assert _run(f"git checkout {throwaway_tag}") == 2


# ---------------------------------------------------------------------------
# MUST BLOCK — existing branch-switch regression fence
# ---------------------------------------------------------------------------
@on_main
@pytest.mark.parametrize("cmd", ["git checkout -b feature/x", "git switch fix/foo"])
def test_branch_creation_and_switch_still_block(cmd):
    # -b/-B branch creation and `switch <non-main>` are blocked without needing
    # a resolvable ref (the detectors fire on the flag / arg shape).
    assert _run(cmd) == 2


@on_main
def test_existing_local_branch_checkout_still_blocks(throwaway_branch):
    # `git checkout <real-local-branch>` (not main) is the original blocked case.
    assert _run(f"git checkout {throwaway_branch}") == 2


# ---------------------------------------------------------------------------
# MUST ALLOW — file-restore, return-to-main, scoped, and non-git shapes.
# These must never be trapped regardless of the repo-root HEAD state, so they
# are NOT guarded by @on_main.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "cmd",
    [
        "git checkout main",  # return to main
        "git checkout -- scripts/eval.py",  # file restore (explicit --)
        "git checkout HEAD -- foo.py",  # file restore from a ref
        "git checkout .",  # restore-all
        "git checkout -f -- scripts/eval.py",  # flag + file restore
        "git -C .claude/worktrees/issue-123 checkout abc1234",  # scoped to a worktree
        "cd /tmp/foo && git checkout abc1234",  # scoped by cd /tmp
        "cd .claude/worktrees/x && git checkout abc1234",  # scoped by cd worktree
        "git status",  # not a checkout/switch
        "git commit -m x",  # not a checkout/switch
    ],
)
def test_allowed_shapes_exit0(cmd):
    assert _run(cmd) == 0


def test_quoted_git_verb_in_argument_is_not_a_false_positive():
    # A quoted git-verb literal inside another command's argument must NOT be
    # treated as a git invocation (the concern that blocked posting a marker
    # whose note discussed the guard). `--note "... git switch ..."` -> allow.
    assert (
        _run(
            'uv run python scripts/task.py post-marker 796 epm:foo --note "test git switch string"'
        )
        == 0
    )


@on_main
def test_nonexistent_ref_fails_soft():
    # An arg that resolves to nothing (typo / non-existent ref) is NOT blocked:
    # rev-parse fails -> guard exits 0 -> git itself errors on the real call.
    assert _run("git checkout nonexistent-typo-ref-xyz-796") == 0


# ---------------------------------------------------------------------------
# Fail-soft — malformed / empty / non-JSON stdin exits 0 (never traps).
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "payload",
    ["", "{}", '{"tool_input": {}}', "not json", '{"tool_input": {"command": ""}}'],
)
def test_fail_soft_exit0(payload):
    assert _run_raw(payload) == 0
