"""Tests for ``scripts/new_worktree.sh`` — sparse-checkout issue worktrees (task #596).

Runs the real helper (subprocess) against a tiny throwaway repo in
``tmp_path`` (never the real repo), modeled on
``tests/test_env_loading_from_worktree.py``'s fixture pattern. The
``worktree_audit`` import follows ``tests/test_worktree_audit.py``'s
``importlib.util.spec_from_file_location`` pattern.

Covers the 13 plan assertions (plan §4.5, task #596): cone engagement
(the git-2.34 ``set --cone``-as-pattern regression), exclusion +
parent-rule materialization, in-cone commit + reapply persistence,
out-of-cone add refusal + ``sparse-checkout add`` fix, audit-guard
porcelain parity, tree-diff parity for out-of-cone committed paths,
.env symlink, reuse, ``--full``, interrupted-creation repair,
creation/registration uniqueness, branch-exists fallback, and
registered-but-directory-deleted prune recovery.

Items 14-17 pin the round-1 code-review hardening (task #596 Minors):
bare no-``--issue`` creation (the CLAUDE.md infra recipe), non-numeric
``--issue`` refusal, repair preserving previously-added cones, and
main-checkout anchoring when invoked from inside another worktree.
"""

from __future__ import annotations

import importlib.util
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

HELPER = Path(__file__).resolve().parent.parent / "scripts" / "new_worktree.sh"

if "worktree_audit" in sys.modules:
    worktree_audit = sys.modules["worktree_audit"]
else:
    _SPEC = importlib.util.spec_from_file_location(
        "worktree_audit",
        Path(__file__).resolve().parent.parent / "scripts" / "worktree_audit.py",
    )
    worktree_audit = importlib.util.module_from_spec(_SPEC)
    # Register in sys.modules BEFORE exec so @dataclass + `from __future__
    # import annotations` can resolve the module during class creation.
    sys.modules["worktree_audit"] = worktree_audit
    _SPEC.loader.exec_module(worktree_audit)

_has_tracked_changes = worktree_audit._has_tracked_changes

_GIT_ENV = {
    **os.environ,
    "GIT_AUTHOR_NAME": "t",
    "GIT_AUTHOR_EMAIL": "t@t",
    "GIT_COMMITTER_NAME": "t",
    "GIT_COMMITTER_EMAIL": "t@t",
}


def _git(cwd: Path, *args: str, check: bool = True) -> subprocess.CompletedProcess:
    """Run git in ``cwd`` with a pinned identity; capture output."""
    return subprocess.run(
        ["git", *args],
        cwd=str(cwd),
        check=check,
        capture_output=True,
        text=True,
        env=_GIT_ENV,
    )


def _run_helper(
    repo: Path, wt: Path, branch: str, *extra: str, check: bool = True
) -> subprocess.CompletedProcess:
    """Invoke scripts/new_worktree.sh from inside the fixture repo."""
    return subprocess.run(
        ["bash", str(HELPER), str(wt), branch, *extra],
        cwd=str(repo),
        check=check,
        capture_output=True,
        text=True,
        env=_GIT_ENV,
    )


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    """Throwaway repo mirroring the real top-level layout (plan §4.5).

    ``CLAUDE.md`` at root is the helper's ``_is_populated`` sentinel (root
    files are always in-cone); ``.gitignore`` covers the untracked ``.env``
    symlink the helper creates, matching the real repo.
    """
    main = tmp_path / "main"
    main.mkdir()
    _git(main, "init", "-q", "-b", "main")
    files = {
        ".gitignore": ".env\n",
        "CLAUDE.md": "project rules\n",
        "src/x.py": "X = 1\n",
        "figures/f.png": "not-really-a-png\n",
        "eval_results/INDEX.md": "| idx |\n",
        "eval_results/old_exp/big.json": "{}\n",
        "external/ref.txt": "ref\n",
        "ood_eval_results/old/o.json": "{}\n",
    }
    for rel, content in files.items():
        p = main / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content)
    _git(main, "add", *files.keys())
    _git(main, "commit", "-q", "-m", "seed")
    (main / ".env").write_text("KEY=1\n")
    return main


@pytest.fixture
def sparse_wt(repo: Path, tmp_path: Path) -> tuple[Path, Path]:
    """(repo, worktree) after a default sparse helper run for issue 2."""
    wt = tmp_path / "wt"
    _run_helper(repo, wt, "issue-2", "--issue", "2")
    return repo, wt


def _commit_issue_artifacts(wt: Path) -> tuple[str, str]:
    """Create + commit this issue's canonical artifacts (plan item 3 core)."""
    rel_eval = "eval_results/issue_2/r.json"
    rel_fig = "figures/issue_2/h.png"
    for rel in (rel_eval, rel_fig):
        p = wt / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("{}\n")
    _git(wt, "add", rel_eval, rel_fig)
    _git(wt, "commit", "-q", "-m", "issue artifacts")
    return rel_eval, rel_fig


# --- item 1: cone mode engaged -------------------------------------------


def test_cone_mode_engaged(sparse_wt: tuple[Path, Path]) -> None:
    """Pins the git-2.34 `set --cone`-as-literal-pattern regression."""
    _repo, wt = sparse_wt
    out = _git(wt, "config", "--worktree", "core.sparseCheckoutCone")
    assert out.stdout.strip() == "true"


# --- item 2: exclusions + parent rule -------------------------------------


def test_exclusions_hold_and_parent_rule_materializes(sparse_wt: tuple[Path, Path]) -> None:
    _repo, wt = sparse_wt
    assert not (wt / "eval_results/old_exp").exists(), "excluded bulk dir leaked in"
    assert not (wt / "external").exists(), "excluded dir leaked in"
    assert not (wt / "ood_eval_results/old").exists(), "excluded bulk dir leaked in"
    assert (wt / "src/x.py").is_file()
    assert (wt / "figures/f.png").is_file()
    # Cone parent rule: immediate files of a cone's parent dir materialize.
    assert (wt / "eval_results/INDEX.md").is_file()


# --- item 3: in-cone new-file commit (criterion 3 core) --------------------


def test_in_cone_new_file_commit_persists(sparse_wt: tuple[Path, Path]) -> None:
    _repo, wt = sparse_wt
    rel_eval, rel_fig = _commit_issue_artifacts(wt)
    assert (wt / rel_eval).is_file()
    assert (wt / rel_fig).is_file()
    _git(wt, "sparse-checkout", "reapply")
    assert (wt / rel_eval).is_file(), "in-cone file vanished on reapply"
    assert (wt / rel_fig).is_file(), "in-cone file vanished on reapply"
    porcelain = _git(wt, "status", "--porcelain").stdout
    assert porcelain == "", f"expected clean porcelain, got: {porcelain!r}"


# --- item 4: out-of-cone add fails loudly; sparse-checkout add fixes -------


def test_out_of_cone_add_fails_then_cone_add_fixes(sparse_wt: tuple[Path, Path]) -> None:
    _repo, wt = sparse_wt
    p = wt / "eval_results/other/x.json"
    p.parent.mkdir(parents=True)
    p.write_text("{}\n")
    refused = _git(wt, "add", "eval_results/other/x.json", check=False)
    assert refused.returncode != 0, "out-of-cone git add should be refused"
    _git(wt, "sparse-checkout", "add", "eval_results/other")
    ok = _git(wt, "add", "eval_results/other/x.json", check=False)
    assert ok.returncode == 0, f"add after sparse-checkout add failed: {ok.stderr}"


# --- item 5: audit guard parity (criterion 4) ------------------------------


def test_audit_has_tracked_changes_parity(sparse_wt: tuple[Path, Path]) -> None:
    _repo, wt = sparse_wt
    _commit_issue_artifacts(wt)
    assert _has_tracked_changes(str(wt)) is False, "clean sparse worktree must read clean"
    (wt / "src/x.py").write_text("X = 2\n")
    assert _has_tracked_changes(str(wt)) is True, "modified tracked file must be detected"


# --- item 6: tree-diff parity incl. out-of-cone committed path -------------


def test_tree_diff_lists_in_and_out_of_cone_committed_files(
    sparse_wt: tuple[Path, Path],
) -> None:
    """`git diff` is a tree-level op: out-of-cone committed paths must list.

    Constructs a GENUINELY out-of-cone committed file via `git add --sparse`
    (the one sanctioned use of --sparse, inside this test only — plan §4.5
    item 6 as amended by the round-1 critique).
    """
    _repo, wt = sparse_wt
    rel_eval, rel_fig = _commit_issue_artifacts(wt)
    rel_oc = "eval_results/out_of_cone/oc.json"
    p = wt / rel_oc
    p.parent.mkdir(parents=True)
    p.write_text("{}\n")
    _git(wt, "add", "--sparse", rel_oc)
    _git(wt, "commit", "-q", "-m", "out-of-cone committed file")
    added = _git(wt, "diff", "--name-only", "--diff-filter=A", "main", "HEAD").stdout.splitlines()
    assert rel_eval in added
    assert rel_fig in added
    assert rel_oc in added, "tree-diff must list out-of-cone committed paths"


# --- item 7: .env symlink ---------------------------------------------------


def test_env_symlink_resolves_to_repo_env(sparse_wt: tuple[Path, Path]) -> None:
    repo, wt = sparse_wt
    link = wt / ".env"
    assert link.is_symlink()
    assert link.resolve() == (repo / ".env").resolve()


# --- item 8: reuse path -----------------------------------------------------


def test_reuse_existing_worktree_exits_zero(sparse_wt: tuple[Path, Path]) -> None:
    repo, wt = sparse_wt
    res = _run_helper(repo, wt, "issue-2", "--issue", "2")
    assert res.returncode == 0
    assert "reusing as-is" in res.stdout


# --- item 9: --full escape hatch --------------------------------------------


def test_full_flag_creates_full_checkout(repo: Path, tmp_path: Path) -> None:
    wt = tmp_path / "wt-full"
    _run_helper(repo, wt, "issue-4", "--full")
    assert (wt / "eval_results/old_exp/big.json").is_file(), "--full must materialize bulk"
    assert (wt / "external/ref.txt").is_file()
    assert (wt / ".env").is_symlink()


# --- item 10: interrupted-creation repair -----------------------------------


def test_interrupted_creation_is_repaired_not_reused(repo: Path, tmp_path: Path) -> None:
    wt2 = tmp_path / "wt2"
    # Simulate a crash between `worktree add --no-checkout` and `checkout`
    # by running the raw add directly (no helper).
    _git(repo, "worktree", "add", "--no-checkout", str(wt2), "-b", "issue-3")
    assert not (wt2 / "CLAUDE.md").exists(), "limbo tree must be unpopulated"
    # Documents why the audit would have kept the corpse: porcelain shows
    # every tracked file as deleted, so _has_tracked_changes is True and the
    # sweep never reaps it — the repair path below is what un-wedges it.
    assert _has_tracked_changes(str(wt2)) is True
    res = _run_helper(repo, wt2, "issue-3", "--issue", "3")
    assert "reusing as-is" not in res.stdout
    assert "repairing" in res.stdout
    assert (wt2 / "src/x.py").is_file(), "repair must populate the tree"
    cone = _git(wt2, "config", "--worktree", "core.sparseCheckoutCone").stdout.strip()
    assert cone == "true"
    assert _git(wt2, "status", "--porcelain").stdout == ""


# --- item 11: creation/registration uniqueness after repair -----------------


def test_registration_unique_after_repair_and_fresh_create_still_works(
    repo: Path, tmp_path: Path
) -> None:
    wt2 = tmp_path / "wt2"
    _git(repo, "worktree", "add", "--no-checkout", str(wt2), "-b", "issue-3")
    _run_helper(repo, wt2, "issue-3", "--issue", "3")
    porcelain = _git(repo, "worktree", "list", "--porcelain").stdout
    registrations = [line for line in porcelain.splitlines() if line == f"worktree {wt2.resolve()}"]
    assert len(registrations) == 1, f"expected exactly one registration: {porcelain}"
    wt3 = tmp_path / "wt3"
    res = _run_helper(repo, wt3, "issue-5", "--issue", "5")
    assert res.returncode == 0
    assert (wt3 / "src/x.py").is_file()


# --- item 12: branch-exists fallback ----------------------------------------


def test_branch_exists_fallback_attaches_existing_branch(
    sparse_wt: tuple[Path, Path],
) -> None:
    repo, wt = sparse_wt
    rel_eval, _rel_fig = _commit_issue_artifacts(wt)
    _git(repo, "worktree", "remove", "--force", str(wt))
    res = _run_helper(repo, wt, "issue-2", "--issue", "2")
    assert res.returncode == 0, f"fallback attach failed: {res.stderr}"
    assert (wt / "CLAUDE.md").is_file(), "tree must be populated"
    head = _git(wt, "rev-parse", "--abbrev-ref", "HEAD").stdout.strip()
    assert head == "issue-2", "must attach the EXISTING branch (no -b failure leaking)"
    assert (wt / rel_eval).is_file(), "existing branch tip (with its commits) checked out"


# --- item 13: registered-but-directory-deleted ------------------------------


def test_directory_deleted_out_of_band_is_pruned_and_recreated(
    sparse_wt: tuple[Path, Path],
) -> None:
    repo, wt = sparse_wt
    shutil.rmtree(wt)
    res = _run_helper(repo, wt, "issue-2", "--issue", "2")
    assert res.returncode == 0, f"prune+recreate failed: {res.stderr}"
    assert (wt / "src/x.py").is_file()
    cone = _git(wt, "config", "--worktree", "core.sparseCheckoutCone").stdout.strip()
    assert cone == "true"


# --- item 14: bare no---issue creation (the CLAUDE.md infra recipe) ----------


def test_bare_no_issue_sparse_creation(repo: Path, tmp_path: Path) -> None:
    """The infra recipe `new_worktree.sh <path> <branch>` (no --issue)."""
    wt = tmp_path / "wt-bare"
    res = _run_helper(repo, wt, "infra-misc")
    assert res.returncode == 0
    assert (wt / "src/x.py").is_file()
    assert (wt / "CLAUDE.md").is_file()
    assert not (wt / "external").exists(), "excluded dir leaked in"
    assert not (wt / "eval_results/old_exp").exists(), "excluded bulk dir leaked in"
    cone = _git(wt, "config", "--worktree", "core.sparseCheckoutCone").stdout.strip()
    assert cone == "true"
    assert (wt / ".env").is_symlink()


# --- item 15: non-numeric --issue refused ------------------------------------


def test_non_numeric_issue_is_refused(repo: Path, tmp_path: Path) -> None:
    """A non-numeric --issue would create a junk cone — must exit 2, no residue."""
    wt = tmp_path / "wt-bad"
    res = _run_helper(repo, wt, "issue-x", "--issue", "12abc", check=False)
    assert res.returncode == 2
    assert "must be numeric" in res.stderr
    porcelain = _git(repo, "worktree", "list", "--porcelain").stdout
    assert f"worktree {wt.resolve()}" not in porcelain.splitlines(), "junk worktree registered"
    assert not wt.exists(), "junk worktree directory left behind"


# --- item 16: repair preserves previously-present cones ----------------------


def test_repair_without_issue_preserves_existing_cones(repo: Path, tmp_path: Path) -> None:
    """Repair must union the prior cone set, not recompute it from scratch.

    Simulates an interrupted creation where `sparse-checkout set` succeeded
    (per-issue cones present) but the final `checkout` did not — then repairs
    WITHOUT --issue. The pre-fix helper silently dropped the issue cones.
    """
    wt = tmp_path / "wt-repair"
    _git(repo, "worktree", "add", "--no-checkout", str(wt), "-b", "issue-7")
    _git(wt, "sparse-checkout", "init", "--cone")
    _git(wt, "sparse-checkout", "set", "src", "figures", "eval_results/issue_7")
    assert not (wt / "CLAUDE.md").exists(), "limbo tree must be unpopulated"
    res = _run_helper(repo, wt, "issue-7")  # NO --issue
    assert "repairing" in res.stdout
    assert (wt / "src/x.py").is_file(), "repair must populate the tree"
    cones = _git(wt, "sparse-checkout", "list").stdout.split()
    assert "eval_results/issue_7" in cones, "repair dropped the prior per-issue cone"


# --- item 17: anchoring when invoked from inside another worktree ------------


def test_invoked_from_inside_another_worktree_anchors_to_main(repo: Path, tmp_path: Path) -> None:
    """REPO_ROOT must resolve to the MAIN checkout, not the invoking worktree.

    The pre-fix `--show-toplevel` anchor computed the include list (and cut
    the new branch) from the invoking worktree's branch HEAD. A top-level dir
    committed to main AFTER the first worktree's branch was cut discriminates
    the two anchors.
    """
    wt1 = tmp_path / "wt1"
    _run_helper(repo, wt1, "issue-8", "--issue", "8")
    p = repo / "newdir/n.txt"
    p.parent.mkdir()
    p.write_text("n\n")
    _git(repo, "add", "newdir/n.txt")
    _git(repo, "commit", "-q", "-m", "new top-level dir on main")
    wt2 = tmp_path / "wt2"
    res = subprocess.run(
        ["bash", str(HELPER), str(wt2), "issue-9", "--issue", "9"],
        cwd=str(wt1),  # invoked from INSIDE another worktree
        check=True,
        capture_output=True,
        text=True,
        env=_GIT_ENV,
    )
    assert res.returncode == 0
    assert (wt2 / "newdir/n.txt").is_file(), "include list/branch base came from wt1, not main"
    porcelain = _git(repo, "worktree", "list", "--porcelain").stdout
    assert f"worktree {wt2.resolve()}" in porcelain.splitlines()


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
