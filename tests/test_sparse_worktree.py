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

Items 21-29 pin the #1054 branch-name inference + reuse-path cone repair
(the #906 incident: a flagless creation on branch ``issue-906`` left
``eval_results/issue_906`` out-of-cone, and the exit-0 reuse path kept it
broken for life): inference from ``issue-<N>`` / ``issue-<N>-<suffix>``
branch names (21-22), explicit ``--issue`` precedence (23), conservative
rejection of nonconforming branches (24), reuse-path repair of BOTH
own-issue cones — flagless and flagged (25), full-worktree no-op with no
WARN (26), sparse-non-issue no-op (27), repair-failure WARN-not-fail (28),
and additive-``add`` preservation of non-inferable prior cones (29).

Items 30-35 pin the #1214 origin/main branch-base ladder: a fresh branch
bases on FRESHLY-FETCHED ``origin/main`` — the bare origin is advanced
out-of-band so a stale-tracking-ref-only implementation fails, and
``--no-track`` is pinned via the absent upstream (30); no ``origin``
remote → local HEAD + WARN (31); fetch failure with a previously-fetched
``origin/main`` → the STALE tracking ref, pushed history only (32); fetch
failure with NO ``origin/main`` ref → FATAL exit 5 with no worktree /
branch / registration residue (33); ``--base-local`` skips the fetch
entirely and bases on LOCAL main even with a broken origin URL (34); and
the pre-existing-branch resume stays network-independent (35).
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
    # #1054 env hygiene: branch-name inference makes _assign_project_quota
    # reachable on FLAGLESS issue-<N> creations too (e.g. item 9's `--full`
    # on branch issue-4), so a dev shell exporting EPS_WORKTREE_ASSIGN_QUOTA=1
    # would send fixture runs into `sudo chattr`. Pin the opt-in OFF for every
    # helper invocation in this file (env-only; no test contract changes).
    "EPS_WORKTREE_ASSIGN_QUOTA": "0",
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


def test_new_worktree_production_probe_rejects_plain_dir(repo: Path, tmp_path: Path) -> None:
    """PRODUCTION-PROBE regression (#681 round-2 Critical): with the seam UNSET,
    the real ``findmnt --mountpoint`` runs against ``<repo>/.claude/worktrees``,
    which is a plain (non-mount) directory on the root fs in the fixture — so the
    helper MUST exit non-zero and create NO worktree.

    This is the gap the seam-only test missed: ``EPS_WORKTREE_BIND_PROBE`` force-
    pass/fail short-circuits the production predicate, so the old
    ``findmnt --target`` bug (which returns rc=0 for ANY dir on a mounted fs and
    would silently land the worktree on `/`) was never exercised. Driving the
    real probe against a plain dir proves ``--mountpoint`` correctly rejects."""
    env = {**_GIT_ENV, "EPS_WORKTREE_REQUIRE_BIND": "1"}
    env.pop("EPS_WORKTREE_BIND_PROBE", None)  # force the PRODUCTION findmnt path
    wt = tmp_path / "wt-prod-nobind"
    res = subprocess.run(
        ["bash", str(HELPER), str(wt), "issue-2", "--issue", "2"],
        cwd=str(repo),
        check=False,
        capture_output=True,
        text=True,
        env=env,
    )
    assert res.returncode == 4, (
        "production findmnt --mountpoint MUST reject a plain (non-mount) "
        f".claude/worktrees dir (exit 4); got {res.returncode}: {res.stderr}"
    )
    assert "bind" in res.stderr.lower()
    assert not wt.exists(), "no worktree may be created when the bind is not a live mount"


def test_new_worktree_seam_still_force_passes(repo: Path, tmp_path: Path) -> None:
    """The fix must NOT break the CI seam contract: with the production probe
    short-circuited via ``EPS_WORKTREE_BIND_PROBE=true``, creation still succeeds
    even though ``.claude/worktrees`` is a plain dir (the seam is the intended CI
    mechanism; the production-probe test above is what closes the coverage gap)."""
    wt = tmp_path / "wt-seam-ok"
    res = subprocess.run(
        ["bash", str(HELPER), str(wt), "issue-3", "--issue", "3"],
        cwd=str(repo),
        check=False,
        capture_output=True,
        text=True,
        env={**_GIT_ENV, "EPS_WORKTREE_REQUIRE_BIND": "1", "EPS_WORKTREE_BIND_PROBE": "true"},
    )
    assert res.returncode == 0, f"seam force-pass must still create the worktree: {res.stderr}"
    assert (wt / "src/x.py").is_file()


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


# --- #771: attached-but-inert WARN (data disk mounted, bind dead) ------------
#
# These exercise the once-per-session WARN new_worktree.sh emits exactly when
# the #681 cutover has NOT been applied yet: the data disk IS a live mount, the
# .claude/worktrees bind is NOT live, and the strict EPS_WORKTREE_REQUIRE_BIND
# assertion is OFF. The WARN is the STRICTLY WEAKER complement of that assertion
# (mutually exclusive on EPS_WORKTREE_REQUIRE_BIND) and NEVER blocks creation.
# The new EPS_WORKTREE_DATADISK_PROBE seam mirrors EPS_WORKTREE_BIND_PROBE.

_WARN_SUBSTRINGS = ("EPS_WORKTREE_REQUIRE_BIND=1 EPS_WORKTREE_ASSIGN_QUOTA=1", "#681", "bind")


def test_inert_warn_fires_when_disk_mounted_but_bind_dead(repo: Path, tmp_path: Path) -> None:
    """(a) Headline case: data disk a live mount, bind NOT live, assertion OFF →
    WARN to stderr (all 3 substrings) AND exit 0 (worktree created) AND the
    once-per-session sentinel appears under the fixture repo's .claude/cache."""
    # tmp_path repos do NOT auto-create .claude/cache (the migration-LOCK test
    # above establishes this precedent) — seed it so the sentinel can land.
    (repo / ".claude" / "cache").mkdir(parents=True, exist_ok=True)
    wt = tmp_path / "wt-inert"
    env = {**_GIT_ENV, "EPS_WORKTREE_DATADISK_PROBE": "true", "EPS_WORKTREE_BIND_PROBE": "false"}
    env.pop("EPS_WORKTREE_REQUIRE_BIND", None)  # assertion OFF → WARN owns this case
    res = subprocess.run(
        ["bash", str(HELPER), str(wt), "issue-2", "--issue", "2"],
        cwd=str(repo),
        check=False,
        capture_output=True,
        text=True,
        env=env,
    )
    assert res.returncode == 0, f"WARN must NOT block creation: {res.stderr}"
    assert (wt / "src/x.py").is_file(), "worktree must be created despite the WARN"
    for sub in _WARN_SUBSTRINGS:
        assert sub in res.stderr, f"WARN missing substring {sub!r}: {res.stderr}"
    sentinel = repo / ".claude" / "cache" / "worktree-inert-warned"
    assert sentinel.exists(), "once-per-session sentinel must be touched on first WARN"


def test_inert_warn_suppressed_when_assertion_governs(repo: Path, tmp_path: Path) -> None:
    """(b) Assertion governs: EPS_WORKTREE_REQUIRE_BIND=1 with the bind LIVE →
    silent-OK. The WARN is OFF (its predicate requires the assertion OFF), and
    the strict assertion passes because the bind is live → exit 0, no WARN."""
    (repo / ".claude" / "cache").mkdir(parents=True, exist_ok=True)
    wt = tmp_path / "wt-assert-on"
    res = subprocess.run(
        ["bash", str(HELPER), str(wt), "issue-3", "--issue", "3"],
        cwd=str(repo),
        check=False,
        capture_output=True,
        text=True,
        env={**_GIT_ENV, "EPS_WORKTREE_REQUIRE_BIND": "1", "EPS_WORKTREE_BIND_PROBE": "true"},
    )
    assert res.returncode == 0, f"assertion-on + bind-live must succeed: {res.stderr}"
    for sub in _WARN_SUBSTRINGS:
        assert sub not in res.stderr, f"WARN must not fire when the assertion governs: {sub!r}"


def test_inert_warn_silent_when_disk_absent(repo: Path, tmp_path: Path) -> None:
    """(c) Disk genuinely absent: no data disk (probe false), no bind, assertion
    OFF → silent-OK. Rules out false positives on pre-cutover / non-GCP envs."""
    (repo / ".claude" / "cache").mkdir(parents=True, exist_ok=True)
    wt = tmp_path / "wt-nodisk"
    env = {**_GIT_ENV, "EPS_WORKTREE_DATADISK_PROBE": "false", "EPS_WORKTREE_BIND_PROBE": "false"}
    env.pop("EPS_WORKTREE_REQUIRE_BIND", None)
    res = subprocess.run(
        ["bash", str(HELPER), str(wt), "issue-2", "--issue", "2"],
        cwd=str(repo),
        check=False,
        capture_output=True,
        text=True,
        env=env,
    )
    assert res.returncode == 0, f"disk-absent must succeed: {res.stderr}"
    assert (wt / "src/x.py").is_file()
    for sub in _WARN_SUBSTRINGS:
        assert sub not in res.stderr, f"WARN must not fire when the data disk is absent: {sub!r}"


def test_inert_warn_silent_when_bind_already_live(repo: Path, tmp_path: Path) -> None:
    """(d) Post-cutover: data disk mounted AND bind live, assertion OFF → no WARN
    (the bind being live makes ! _bind_is_live false). Rules out warning when
    correctly wired but the opt-in flag has not yet been flipped."""
    (repo / ".claude" / "cache").mkdir(parents=True, exist_ok=True)
    wt = tmp_path / "wt-bound"
    env = {**_GIT_ENV, "EPS_WORKTREE_DATADISK_PROBE": "true", "EPS_WORKTREE_BIND_PROBE": "true"}
    env.pop("EPS_WORKTREE_REQUIRE_BIND", None)
    res = subprocess.run(
        ["bash", str(HELPER), str(wt), "issue-2", "--issue", "2"],
        cwd=str(repo),
        check=False,
        capture_output=True,
        text=True,
        env=env,
    )
    assert res.returncode == 0, f"bind-live must succeed: {res.stderr}"
    for sub in _WARN_SUBSTRINGS:
        assert sub not in res.stderr, f"WARN must not fire when the bind is already live: {sub!r}"


def test_inert_warn_fires_once_per_session(repo: Path, tmp_path: Path) -> None:
    """(e) Once-per-session dedup: run the headline (a) case TWICE against the
    SAME fixture repo. The sentinel persists in the repo's .claude/cache between
    runs, so the WARN appears on the first run and is suppressed on the second."""
    (repo / ".claude" / "cache").mkdir(parents=True, exist_ok=True)
    env = {**_GIT_ENV, "EPS_WORKTREE_DATADISK_PROBE": "true", "EPS_WORKTREE_BIND_PROBE": "false"}
    env.pop("EPS_WORKTREE_REQUIRE_BIND", None)

    def _run(wt: Path, branch: str) -> subprocess.CompletedProcess:
        return subprocess.run(
            ["bash", str(HELPER), str(wt), branch, "--issue", branch.split("-")[1]],
            cwd=str(repo),
            check=False,
            capture_output=True,
            text=True,
            env=env,
        )

    first = _run(tmp_path / "wt-once-1", "issue-2")
    assert first.returncode == 0, f"first run must succeed: {first.stderr}"
    for sub in _WARN_SUBSTRINGS:
        assert sub in first.stderr, f"first run must WARN: missing {sub!r}"

    second = _run(tmp_path / "wt-once-2", "issue-3")
    assert second.returncode == 0, f"second run must succeed: {second.stderr}"
    for sub in _WARN_SUBSTRINGS:
        assert sub not in second.stderr, f"second run must be silent (sentinel): {sub!r} re-emitted"


def test_inert_warn_production_probe_rejects_plain_dir(repo: Path, tmp_path: Path) -> None:
    """(f) PRODUCTION-PATH probe regression (Statistics Must-Fix): with the
    EPS_WORKTREE_DATADISK_PROBE seam UNSET, the real `findmnt --mountpoint` runs
    against a non-mount EPS_VM_DATA_DISK_PATH (a plain tmp dir) → it reports the
    data disk NOT live, so the WARN does NOT fire (silent-OK). This catches a
    future `--mountpoint` → `--target` slip, which would falsely report ANY
    directory as a live mount and fire the WARN on clean non-GCP / CI machines.
    Mirrors test_new_worktree_production_probe_rejects_plain_dir for the bind."""
    (repo / ".claude" / "cache").mkdir(parents=True, exist_ok=True)
    fake_disk = tmp_path / "fake-data-disk"
    fake_disk.mkdir()  # a plain dir, NOT a mountpoint
    env = {
        **_GIT_ENV,
        "EPS_VM_DATA_DISK_PATH": str(fake_disk),
        "EPS_WORKTREE_BIND_PROBE": "false",
    }
    env.pop("EPS_WORKTREE_DATADISK_PROBE", None)  # force the PRODUCTION findmnt path
    env.pop("EPS_WORKTREE_REQUIRE_BIND", None)
    wt = tmp_path / "wt-prod-datadisk"
    res = subprocess.run(
        ["bash", str(HELPER), str(wt), "issue-2", "--issue", "2"],
        cwd=str(repo),
        check=False,
        capture_output=True,
        text=True,
        env=env,
    )
    assert res.returncode == 0, f"production probe on a plain dir must succeed: {res.stderr}"
    for sub in _WARN_SUBSTRINGS:
        assert sub not in res.stderr, (
            "production findmnt --mountpoint must report a plain (non-mount) data-disk "
            f"path as NOT live → no WARN; got substring {sub!r}: {res.stderr}"
        )


# --- items 21-29: #1054 branch-name inference + reuse-path cone repair -------
#
# The #906 incident: `new_worktree.sh <path> issue-906` (no --issue) created a
# sparse worktree WITHOUT eval_results/issue_906 cones, so committing the
# task's own artifacts failed with "outside of your sparse-checkout
# definition" — and the exit-0 reuse path kept the worktree broken for life.
# Items 21-24 pin the inference at creation; 25-29 pin the reuse-path repair.

_REPAIR_WARN = "WARN — could not ensure"


def test_issue_inferred_from_branch_name(repo: Path, tmp_path: Path) -> None:
    """Item 21: flagless creation on `issue-6` infers --issue 6 — the inverted
    #906 repro: the own-issue cones are present and `git add` of the issue's
    own artifact succeeds with no ceremony."""
    wt = tmp_path / "wt-infer"
    res = _run_helper(repo, wt, "issue-6")  # NO --issue
    assert res.returncode == 0
    assert "inferred --issue 6" in res.stderr, f"inference notice missing: {res.stderr}"
    cones = _git(wt, "sparse-checkout", "list").stdout.split()
    assert "eval_results/issue_6" in cones
    assert "ood_eval_results/issue_6" in cones
    p = wt / "eval_results/issue_6/r.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("{}\n")
    ok = _git(wt, "add", "eval_results/issue_6/r.json", check=False)
    assert ok.returncode == 0, f"the #906 repro must be fixed: {ok.stderr}"


def test_issue_inferred_from_suffix_branch(repo: Path, tmp_path: Path) -> None:
    """Item 22: a same-issue follow-up branch `issue-<N>-<suffix>` infers N."""
    wt = tmp_path / "wt-infer-suffix"
    res = _run_helper(repo, wt, "issue-6-followup-a")  # NO --issue
    assert res.returncode == 0
    assert "inferred --issue 6" in res.stderr
    cones = _git(wt, "sparse-checkout", "list").stdout.split()
    assert "eval_results/issue_6" in cones
    assert "ood_eval_results/issue_6" in cones


def test_explicit_issue_wins_over_inference(repo: Path, tmp_path: Path) -> None:
    """Item 23: an explicit --issue 9 on branch `issue-6` yields issue_9 cones,
    NOT issue_6 — the flag is the deliberate override channel."""
    wt = tmp_path / "wt-explicit"
    res = _run_helper(repo, wt, "issue-6", "--issue", "9")
    assert res.returncode == 0
    assert "inferred" not in res.stderr, "no inference notice when the flag was given"
    cones = _git(wt, "sparse-checkout", "list").stdout.split()
    assert "eval_results/issue_9" in cones
    assert "eval_results/issue_6" not in cones


@pytest.mark.parametrize("branch", ["issue-12abc", "infra-misc"])
def test_no_inference_on_nonconforming_branch(repo: Path, tmp_path: Path, branch: str) -> None:
    """Item 24: branches not matching `issue-<digits>(-…)` get NO issue cones
    (today's behavior — conservative inference; extends existing item 14)."""
    wt = tmp_path / f"wt-noninfer-{branch}"
    res = _run_helper(repo, wt, branch)  # NO --issue
    assert res.returncode == 0
    assert "inferred" not in res.stderr
    cones = _git(wt, "sparse-checkout", "list").stdout
    assert "eval_results/issue_" not in cones, f"unexpected issue cone for {branch!r}: {cones}"


def test_reuse_path_repairs_missing_own_issue_cone(repo: Path, tmp_path: Path) -> None:
    """Item 25: re-invoking the helper on an existing sparse worktree that is
    MISSING its own-issue cones restores BOTH eval_results/issue_<N> AND
    ood_eval_results/issue_<N> (an eval-only repair mutant must not pass),
    exits 0 with "reusing as-is" — flagless (inference) AND flagged variants."""
    wt = tmp_path / "wt-reuse-repair"
    _run_helper(repo, wt, "issue-6")  # flagless creation (inference)
    # Simulate the pre-fix broken state: drop the own-issue cones.
    _git(wt, "sparse-checkout", "set", "src", "figures")
    cones = _git(wt, "sparse-checkout", "list").stdout.split()
    assert "eval_results/issue_6" not in cones, "broken-state precondition failed"
    # Flagless re-invoke: the reuse path repairs BOTH cones via inference.
    res = _run_helper(repo, wt, "issue-6")
    assert res.returncode == 0
    assert "reusing as-is" in res.stdout
    cones = _git(wt, "sparse-checkout", "list").stdout.split()
    assert "eval_results/issue_6" in cones, "reuse repair must restore the eval cone"
    assert "ood_eval_results/issue_6" in cones, "reuse repair must restore the ood cone"
    # The #906 `git add` now succeeds.
    p = wt / "eval_results/issue_6/r.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("{}\n")
    ok = _git(wt, "add", "eval_results/issue_6/r.json", check=False)
    assert ok.returncode == 0, f"the #906 repro must be fixed on reuse: {ok.stderr}"
    _git(wt, "commit", "-q", "-m", "issue artifact")
    # Flagged variant: drop again, repair with an explicit --issue 6.
    _git(wt, "sparse-checkout", "set", "src", "figures")
    res2 = _run_helper(repo, wt, "issue-6", "--issue", "6")
    assert res2.returncode == 0
    assert "reusing as-is" in res2.stdout
    cones = _git(wt, "sparse-checkout", "list").stdout.split()
    assert "eval_results/issue_6" in cones
    assert "ood_eval_results/issue_6" in cones


def test_reuse_path_full_worktree_untouched(repo: Path, tmp_path: Path) -> None:
    """Item 26 (+ T6 addendum): reuse on a --full worktree changes nothing —
    still not sparse, bulk still materialized, exit 0, and NO repair WARN (a
    mutant that attempts + fails the add on a full worktree must not survive)."""
    wt = tmp_path / "wt-full-reuse"
    _run_helper(repo, wt, "issue-4", "--full")
    res = _run_helper(repo, wt, "issue-4", "--issue", "4")
    assert res.returncode == 0
    assert "reusing as-is" in res.stdout
    # Non-sparse probe: the script's own skip predicate (cone-mode worktree
    # config unset). `sparse-checkout list` rc is version-dependent here —
    # git 2.34 warns "this worktree is not sparse" but exits 0.
    cone = _git(wt, "config", "--worktree", "core.sparseCheckoutCone", check=False)
    assert cone.stdout.strip() != "true", "a --full worktree must stay non-sparse after reuse"
    assert _git(wt, "sparse-checkout", "list", check=False).stdout.strip() == "", (
        "a --full worktree must have no cone entries after reuse"
    )
    assert (wt / "external/ref.txt").is_file()
    assert _REPAIR_WARN not in res.stderr, (
        f"full-worktree reuse must not attempt the cone add: {res.stderr}"
    )


def test_reuse_path_sparse_non_issue_noop(repo: Path, tmp_path: Path) -> None:
    """Item 27: reuse on a sparse worktree with NO issue association (branch
    `infra-misc`, no flag) adds no issue cones and emits no WARN."""
    wt = tmp_path / "wt-nonissue-reuse"
    _run_helper(repo, wt, "infra-misc")
    res = _run_helper(repo, wt, "infra-misc")
    assert res.returncode == 0
    assert "reusing as-is" in res.stdout
    cones = _git(wt, "sparse-checkout", "list").stdout
    assert "eval_results/issue_" not in cones
    assert _REPAIR_WARN not in res.stderr


def test_reuse_repair_failure_warns_not_fails(repo: Path, tmp_path: Path) -> None:
    """Item 28: a failed reuse-path repair WARNs to stderr and still exits 0
    (the exit-0 reuse contract is load-bearing — it fires on every /issue
    resume). Failure injection: a pre-created index.lock in the worktree's
    private gitdir makes `sparse-checkout add` fail (rc=128 "File exists";
    fact-checker-probed on git 2.34.1; pre-probed fallback per plan §8:
    chmod 0555 on the worktree gitdir's info/)."""
    wt = tmp_path / "wt-warn-reuse"
    _run_helper(repo, wt, "issue-6")
    _git(wt, "sparse-checkout", "set", "src", "figures")  # drop the own cones
    gitdir = Path(_git(wt, "rev-parse", "--absolute-git-dir").stdout.strip())
    lock = gitdir / "index.lock"
    lock.write_text("")
    try:
        res = _run_helper(repo, wt, "issue-6")
        assert res.returncode == 0, f"reuse must stay exit-0 on a failed repair: {res.stderr}"
        assert "reusing as-is" in res.stdout
        assert _REPAIR_WARN in res.stderr, f"failed repair must WARN: {res.stderr}"
    finally:
        lock.unlink()


def test_reuse_repair_preserves_non_inferable_prior_cones(repo: Path, tmp_path: Path) -> None:
    """Item 29: the reuse repair is ADDITIVE (`sparse-checkout add`) — a prior
    manually-added cone the helper cannot infer (another issue's fixtures)
    SURVIVES alongside the restored own-issue cones. A regression to `set`
    semantics (or dropping the $EXISTING capture) fails this test."""
    wt = tmp_path / "wt-prior-cones"
    _run_helper(repo, wt, "issue-7")  # flagless; inference → issue_7 cones
    # A non-inferable prior cone + the broken own-cone state, together.
    _git(wt, "sparse-checkout", "set", "src", "figures", "eval_results/issue_99")
    res = _run_helper(repo, wt, "issue-7")  # flagless reuse
    assert res.returncode == 0
    assert "reusing as-is" in res.stdout
    cones = _git(wt, "sparse-checkout", "list").stdout.split()
    assert "eval_results/issue_99" in cones, "repair must be additive, never a cone reset"
    assert "eval_results/issue_7" in cones
    assert "ood_eval_results/issue_7" in cones


# --- items 30-35: #1214 branch base = fetched origin/main -----------------


def _add_diverged_origin(repo: Path, tmp_path: Path) -> tuple[str, str]:
    """Bare ``origin`` + push main, then advance LOCAL main by one commit.

    Returns ``(pushed_tip, local_tip)``, pushed != local — the unpushed
    task-state-churn shape (#1214). The push also creates the local
    ``refs/remotes/origin/main`` tracking ref (asserted — plan §12.5).
    """
    bare = tmp_path / "origin.git"
    _git(tmp_path, "init", "--bare", "-q", "-b", "main", str(bare))
    _git(repo, "remote", "add", "origin", str(bare))
    _git(repo, "push", "-q", "origin", "main")
    assert (
        _git(repo, "rev-parse", "--verify", "refs/remotes/origin/main", check=False).returncode == 0
    ), "git push did not create the remote-tracking ref; add an explicit fetch here"
    pushed = _git(repo, "rev-parse", "main").stdout.strip()
    (repo / "src" / "churn.py").write_text("CHURN = 1\n")
    _git(repo, "add", "src/churn.py")
    _git(repo, "commit", "-q", "-m", "unpushed local churn")
    local = _git(repo, "rev-parse", "main").stdout.strip()
    assert pushed != local
    return pushed, local


def test_new_branch_based_on_origin_main_not_local_head(repo: Path, tmp_path: Path) -> None:
    """Item 30 (#1214 durability pin): a fresh branch bases on FETCHED origin/main.

    The bare origin is advanced OUT-OF-BAND (second clone → commit → push)
    after the last push from ``repo``, so the local tracking ref is stale
    until the helper's fetch runs — a stale-ref-only implementation would
    base on the older pushed tip and fail the tip-equality assert. Also
    pins ``--no-track``: the new branch gains no upstream.
    """
    _add_diverged_origin(repo, tmp_path)
    local_tip = _git(repo, "rev-parse", "main").stdout.strip()
    clone2 = tmp_path / "clone2"
    _git(tmp_path, "clone", "-q", str(tmp_path / "origin.git"), str(clone2))
    (clone2 / "remote_advance.txt").write_text("advanced\n")
    _git(clone2, "add", "remote_advance.txt")
    _git(clone2, "commit", "-q", "-m", "remote-side advance")
    _git(clone2, "push", "-q", "origin", "main")
    new_origin_tip = _git(clone2, "rev-parse", "main").stdout.strip()
    stale_tracking = _git(repo, "rev-parse", "refs/remotes/origin/main").stdout.strip()
    assert stale_tracking != new_origin_tip, "tracking ref must be stale pre-helper"

    wt = tmp_path / "wt-1214-fetch"
    _run_helper(repo, wt, "issue-30")
    tip = _git(repo, "rev-parse", "issue-30").stdout.strip()
    assert tip == new_origin_tip, "branch must base on the FRESHLY-FETCHED origin tip"
    assert tip != local_tip
    assert (wt / "CLAUDE.md").exists()
    # --no-track: no upstream configured for the new branch.
    assert _git(repo, "config", "branch.issue-30.merge", check=False).returncode != 0


def test_no_origin_remote_falls_back_to_local_head_with_warn(repo: Path, tmp_path: Path) -> None:
    """Item 31: no ``origin`` remote → local HEAD base + WARN (fixture repos)."""
    wt = tmp_path / "wt-1214-noorigin"
    res = _run_helper(repo, wt, "issue-31")
    assert "no 'origin' remote" in res.stderr
    tip = _git(repo, "rev-parse", "issue-31").stdout.strip()
    assert tip == _git(repo, "rev-parse", "main").stdout.strip()
    assert (wt / "CLAUDE.md").exists()


def test_fetch_failure_falls_back_to_stale_origin_main(repo: Path, tmp_path: Path) -> None:
    """Item 32: fetch fails but origin/main exists → STALE tracking-ref base.

    Pushed-history-only, so the #1214 churn bug is not reintroduced: the
    branch tips at the last-PUSHED commit, never the local churn commit.
    """
    pushed, local = _add_diverged_origin(repo, tmp_path)
    _git(repo, "remote", "set-url", "origin", str(tmp_path / "nonexistent"))
    wt = tmp_path / "wt-1214-stale"
    res = _run_helper(repo, wt, "issue-32")
    assert "STALE" in res.stderr
    tip = _git(repo, "rev-parse", "issue-32").stdout.strip()
    assert tip == pushed
    assert tip != local


def test_fetch_failure_without_origin_main_fails_loud_no_residue(
    repo: Path, tmp_path: Path
) -> None:
    """Item 33: fetch fails AND no origin/main ref → FATAL exit 5, nothing created."""
    _git(repo, "remote", "add", "origin", str(tmp_path / "nonexistent"))
    wt = tmp_path / "wt-1214-fatal"
    res = _run_helper(repo, wt, "issue-33", check=False)
    assert res.returncode == 5, f"expected exit 5, got {res.returncode}: {res.stderr}"
    assert "FATAL" in res.stderr
    assert _git(repo, "rev-parse", "--verify", "issue-33", check=False).returncode != 0
    assert not wt.exists()
    porcelain = _git(repo, "worktree", "list", "--porcelain").stdout
    assert f"worktree {wt}" not in porcelain


def test_base_local_flag_skips_fetch_and_bases_on_local_head(repo: Path, tmp_path: Path) -> None:
    """Item 34: ``--base-local`` bases on LOCAL main and attempts no fetch.

    The origin URL is BROKEN, so mere success proves the fetch was skipped;
    the load-bearing discriminator vs the stale tier is the tip assert:
    branch tip == LOCAL main tip (the stale tier would tip at ``pushed``).
    """
    pushed, local = _add_diverged_origin(repo, tmp_path)
    _git(repo, "remote", "set-url", "origin", str(tmp_path / "nonexistent"))
    wt = tmp_path / "wt-1214-baselocal"
    res = _run_helper(repo, wt, "issue-34", "--base-local")
    assert "--base-local" in res.stderr
    tip = _git(repo, "rev-parse", "issue-34").stdout.strip()
    assert tip == local, "--base-local must base on the LOCAL main tip"
    assert tip != pushed


def test_existing_branch_resume_needs_no_network(repo: Path, tmp_path: Path) -> None:
    """Item 35: a pre-existing branch attaches with NO fetch (offline resume).

    The broken-URL origin has NO origin/main tracking ref, so a fetch
    attempt would take the FATAL tier — success proves ``_resolve_base``
    is skipped when the branch pre-exists (``CREATED_BRANCH=0``).
    """
    _git(repo, "remote", "add", "origin", str(tmp_path / "nonexistent"))
    _git(repo, "branch", "issue-35")
    pre_tip = _git(repo, "rev-parse", "issue-35").stdout.strip()
    wt = tmp_path / "wt-1214-resume"
    res = _run_helper(repo, wt, "issue-35")
    assert res.returncode == 0
    assert _git(wt, "rev-parse", "HEAD").stdout.strip() == pre_tip


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
