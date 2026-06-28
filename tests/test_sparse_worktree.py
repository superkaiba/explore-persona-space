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

Items 18-19 pin the ``tests/sparse_cones.txt`` registry (#671): the full
pytest suite (the /issue Step 9c test-verdict gate) reads OTHER issues'
committed ``eval_results/`` artifacts as fixtures, so ``new_worktree.sh``
pre-adds every cone in that registry — a sparse worktree must materialize
them without a manual ``sparse-checkout add``, and a whitespace-bearing
registry line must be refused loudly.
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


# --- item 18: tests/sparse_cones.txt registry is pre-added (#671) ------------


def _seed_cone_registry(repo: Path, *cones: str) -> None:
    """Write tests/sparse_cones.txt + commit the referenced cone dirs.

    The helper reads ``$REPO_ROOT/tests/sparse_cones.txt``; in this fixture
    that resolves to the throwaway repo, so the registry + its cone dirs must
    be seeded here (the real registry lives in the production repo).
    """
    (repo / "tests").mkdir(parents=True, exist_ok=True)
    body = "# test registry\n" + "".join(f"{c}\n" for c in cones)
    (repo / "tests/sparse_cones.txt").write_text(body)
    paths = ["tests/sparse_cones.txt"]
    for cone in cones:
        f = repo / cone / "ref.json"
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text("{}\n")
        paths.append(f"{cone}/ref.json")
    _git(repo, "add", *paths)
    _git(repo, "commit", "-q", "-m", "seed sparse_cones registry")


def test_registry_cones_are_preadded(repo: Path, tmp_path: Path) -> None:
    """A cone listed in tests/sparse_cones.txt materializes in a fresh sparse
    worktree even though it lives under an EXCLUDES bulk dir — so the Step 9c
    full-suite gate passes with no manual `sparse-checkout add`."""
    # eval_results/issue_777 is under the eval_results EXCLUDE, so it would be
    # absent in a default sparse worktree — the registry is what pulls it in.
    _seed_cone_registry(repo, "eval_results/issue_777")
    wt = tmp_path / "wt-reg"
    res = _run_helper(repo, wt, "issue-2", "--issue", "2")
    assert res.returncode == 0, f"helper failed: {res.stderr}"
    assert (wt / "eval_results/issue_777/ref.json").is_file(), (
        "registry cone not materialized — the test-suite gate would FileNotFoundError"
    )
    cones = _git(wt, "sparse-checkout", "list").stdout.split()
    assert "eval_results/issue_777" in cones
    # The per-issue cone for the worktree's OWN issue still works alongside it.
    assert "eval_results/issue_2" in cones


# --- item 19: whitespace-bearing registry line refused loudly (#671) ---------


def test_registry_line_with_whitespace_is_refused(repo: Path, tmp_path: Path) -> None:
    """A registry cone with embedded whitespace would mis-split the unquoted
    cone expansion — the helper must FATAL, not silently create a junk cone."""
    (repo / "tests").mkdir(parents=True, exist_ok=True)
    (repo / "tests/sparse_cones.txt").write_text("# reg\neval_results/bad dir\n")
    _git(repo, "add", "tests/sparse_cones.txt")
    _git(repo, "commit", "-q", "-m", "bad registry")
    wt = tmp_path / "wt-badreg"
    res = _run_helper(repo, wt, "issue-2", "--issue", "2", check=False)
    assert res.returncode != 0, "whitespace cone line must fail loudly"
    assert "whitespace/quoting" in res.stderr


# --- item 20: all-comment / empty registry does not crash (#671) -------------


def test_all_comment_registry_does_not_crash(repo: Path, tmp_path: Path) -> None:
    """A registry with only comments (no cone lines) must NOT abort creation:
    the grep that extracts cones exits 1 on no-match, which under the script's
    `set -o pipefail` would crash the assignment without the `|| true` guard."""
    (repo / "tests").mkdir(parents=True, exist_ok=True)
    (repo / "tests/sparse_cones.txt").write_text("# only comments\n# no cones here\n")
    _git(repo, "add", "tests/sparse_cones.txt")
    _git(repo, "commit", "-q", "-m", "all-comment registry")
    wt = tmp_path / "wt-emptyreg"
    res = _run_helper(repo, wt, "issue-2", "--issue", "2", check=False)
    assert res.returncode == 0, f"all-comment registry crashed creation: {res.stderr}"
    assert (wt / "src/x.py").is_file()
    cone = _git(wt, "config", "--worktree", "core.sparseCheckoutCone").stdout.strip()
    assert cone == "true"


# --- #681 item: data-disk bind assertion + migration LOCK refusal -----------


def test_new_worktree_asserts_data_disk_mounted(repo: Path, tmp_path: Path) -> None:
    """With EPS_WORKTREE_REQUIRE_BIND=1, the helper FAILs loud (exit 4) when the
    bind probe reports the data disk absent, and succeeds when it reports present.
    (A real bind mount needs privilege; the probe is exercised via the
    EPS_WORKTREE_BIND_PROBE seam — `false` force-fails, `true` force-passes.)"""
    # Bind absent → refuse, no worktree created.
    wt_fail = tmp_path / "wt-nobind"
    res = subprocess.run(
        ["bash", str(HELPER), str(wt_fail), "issue-2", "--issue", "2"],
        cwd=str(repo),
        check=False,
        capture_output=True,
        text=True,
        env={**_GIT_ENV, "EPS_WORKTREE_REQUIRE_BIND": "1", "EPS_WORKTREE_BIND_PROBE": "false"},
    )
    assert res.returncode == 4, (
        f"expected exit 4 on missing bind, got {res.returncode}: {res.stderr}"
    )
    assert "bind" in res.stderr.lower()
    assert not wt_fail.exists(), "no worktree must be created when the bind is missing"

    # Bind present → succeeds.
    wt_ok = tmp_path / "wt-bind"
    res2 = subprocess.run(
        ["bash", str(HELPER), str(wt_ok), "issue-3", "--issue", "3"],
        cwd=str(repo),
        check=False,
        capture_output=True,
        text=True,
        env={**_GIT_ENV, "EPS_WORKTREE_REQUIRE_BIND": "1", "EPS_WORKTREE_BIND_PROBE": "true"},
    )
    assert res2.returncode == 0, f"bind-present run failed: {res2.stderr}"
    assert (wt_ok / "src/x.py").is_file()


def test_new_worktree_refuses_under_migration_lock(repo: Path, tmp_path: Path) -> None:
    """The cutover migration LOCK (.claude/cache/worktree-migration.LOCK) makes
    new_worktree.sh refuse (exit 3) before any worktree creation."""
    lock = repo / ".claude" / "cache" / "worktree-migration.LOCK"
    lock.parent.mkdir(parents=True, exist_ok=True)
    lock.write_text("cutover in progress\n")
    wt = tmp_path / "wt-locked"
    res = _run_helper(repo, wt, "issue-2", "--issue", "2", check=False)
    assert res.returncode == 3, f"expected exit 3 under migration LOCK, got {res.returncode}"
    assert "migration" in res.stderr.lower()
    assert not wt.exists(), "no worktree must be created while the LOCK is held"


def test_managed_pin_worktree_refuses_under_migration_lock(repo: Path) -> None:
    """task_workflow._ensure_managed_main_worktree (the _task-main-pin path — a
    SECOND concurrent worktree-creation writer) MUST also refuse while the
    migration LOCK is held, or a task.py write mid-swap could strand task state
    on the .premigrate tree (Codex freeze-audit concern, plan §4 Phase 4)."""
    import importlib.util as _ilu

    tw_path = (
        Path(__file__).resolve().parent.parent
        / "src"
        / "explore_persona_space"
        / "task_workflow.py"
    )
    spec = _ilu.spec_from_file_location("eps_task_workflow_681", tw_path)
    tw = _ilu.module_from_spec(spec)
    sys.modules["eps_task_workflow_681"] = tw
    spec.loader.exec_module(tw)

    lock = repo / ".claude" / "cache" / "worktree-migration.LOCK"
    lock.parent.mkdir(parents=True, exist_ok=True)
    lock.write_text("cutover in progress\n")
    with pytest.raises(RuntimeError, match="migration in progress"):
        tw._ensure_managed_main_worktree(repo, "issue-feature", {**_GIT_ENV})
    # Removing the LOCK lifts the refusal (it then proceeds to the `main` check,
    # which is a DIFFERENT, expected failure here — the fixture repo IS on main,
    # so it would actually try to create the pin; we only assert the LOCK gate no
    # longer fires by checking the error is NOT the migration one).
    lock.unlink()
    try:
        tw._ensure_managed_main_worktree(repo, "issue-feature", {**_GIT_ENV})
    except RuntimeError as exc:
        assert "migration in progress" not in str(exc)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
