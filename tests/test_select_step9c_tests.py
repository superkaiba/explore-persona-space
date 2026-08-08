"""Unit tests for ``scripts/select_step9c_tests.py`` (#754, #851).

The helper maps the files this branch touched to their covering pytest files
plus a pinned workflow-invariant literal set, for the ``/issue`` Step 9c
test-verdict gate. Most cases inject a fake ``git diff`` runner and a
``tmp_path`` ``tests/`` tree so no real git / real branch state is needed.

Two exceptions: the pinned-list tests (cases 6/6b) assert the literal
``WORKFLOW_INVARIANT`` matches the LIVE repo ``tests/`` tree and the sorted
manifest (``tests/step9c_workflow_invariant_manifest.txt``) so an added/removed
invariant test forces a deliberate edit of the literal; and the #851 regression
tests (cases 13-14) build a real throwaway git repo + worktree to pin the
work-root resolution end to end (branch-new test selected, deleted-on-branch
test dropped, empty-diff NOTE at the base checkout).
"""

from __future__ import annotations

import importlib.util
import json
import math
import os
import re
import subprocess
from pathlib import Path

import pytest

# Import the helper by path (it lives under scripts/, not an importable package).
_HELPER_PATH = Path(__file__).resolve().parents[1] / "scripts" / "select_step9c_tests.py"
_spec = importlib.util.spec_from_file_location("select_step9c_tests", _HELPER_PATH)
assert _spec and _spec.loader
sel = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(sel)


def _make_tree(tmp_path: Path, test_files: list[str]) -> Path:
    """Create a fake repo root with the given tests/ files (+ all invariants)."""
    repo = tmp_path / "repo"
    (repo / "tests").mkdir(parents=True)
    # Materialize every pinned invariant so the on-disk filter keeps them all,
    # letting each case assert the per-file mapping against a known invariant set.
    for inv in sel.WORKFLOW_INVARIANT:
        (repo / inv).parent.mkdir(parents=True, exist_ok=True)
        (repo / inv).write_text("# stub\n")
    # Materialize the glob-scan map's test files too (#895) so map-hit cases
    # can assert selection; harmless elsewhere (selection requires a scan-glob
    # hit, so cases asserting set(tests) == set(WORKFLOW_INVARIANT) hold).
    for scan_test in sel.GLOB_SCAN_TESTS:
        (repo / scan_test).parent.mkdir(parents=True, exist_ok=True)
        (repo / scan_test).write_text("# stub\n")
    for tf in test_files:
        (repo / "tests" / tf).write_text("# stub\n")
    return repo


def _runner_for(touched: list[str]):
    """Return a fake git-diff runner that yields the given touched paths."""

    def _runner(argv: list[str]) -> str:
        assert argv[:3] == ["git", "diff", "--name-only"]
        return "\n".join(touched) + ("\n" if touched else "")

    return _runner


# --- Case 1: scripts/X.py -> test_X.py + test_*X*.py glob arm ----------------
def test_code_file_maps_to_exact_and_glob_tests(tmp_path: Path):
    repo = _make_tree(tmp_path, ["test_widget.py", "test_widget_cli.py", "test_other.py"])
    touched = sel.compute_touched("main", repo, _runner=_runner_for(["scripts/widget.py"]))
    tests, untested = sel.select_tests(touched, repo)
    assert "tests/test_widget.py" in tests  # exact match
    assert "tests/test_widget_cli.py" in tests  # glob *widget* arm
    assert "tests/test_other.py" not in tests
    assert untested == []


# --- Case 2: a touched tests/test_<X>.py includes itself ---------------------
def test_touched_test_file_includes_itself(tmp_path: Path):
    repo = _make_tree(tmp_path, ["test_thing.py"])
    touched = sel.compute_touched("main", repo, _runner=_runner_for(["tests/test_thing.py"]))
    tests, untested = sel.select_tests(touched, repo)
    assert "tests/test_thing.py" in tests
    assert untested == []


# --- Case 3: workflow-surface file -> only the invariant set; no WARN --------
@pytest.mark.parametrize(
    "surface_file",
    [
        ".claude/skills/issue/SKILL.md",
        "CLAUDE.md",
        ".claude/workflow.yaml",
        ".claude/agents/planner.md",
        ".claude/rules/code-style.md",
        "tasks/running/999/body.md",
    ],
)
def test_workflow_surface_file_runs_only_invariants(tmp_path: Path, surface_file: str):
    repo = _make_tree(tmp_path, ["test_unrelated.py"])
    touched = sel.compute_touched("main", repo, _runner=_runner_for([surface_file]))
    tests, untested = sel.select_tests(touched, repo)
    # Selection is exactly the present invariant set (no per-file test, no WARN).
    assert set(tests) == set(sel.WORKFLOW_INVARIANT)
    assert untested == []  # a workflow-surface file is a correct SKIP, not "untested"


# --- Case 4: data/config/doc file -> skip, no WARN ---------------------------
@pytest.mark.parametrize("doc_file", ["scripts/data.json", "notes.md", "configs/x.yaml"])
def test_data_doc_file_skipped_no_warn(tmp_path: Path, doc_file: str):
    repo = _make_tree(tmp_path, [])
    touched = sel.compute_touched("main", repo, _runner=_runner_for([doc_file]))
    tests, untested = sel.select_tests(touched, repo)
    assert set(tests) == set(sel.WORKFLOW_INVARIANT)
    assert untested == []


# --- Case 5: touched code file with NO matching test -> untested_touched -----
def test_untested_code_file_warns(tmp_path: Path):
    repo = _make_tree(tmp_path, [])  # no test_orphan*.py anywhere
    touched = sel.compute_touched("main", repo, _runner=_runner_for(["scripts/orphan.py"]))
    tests, untested = sel.select_tests(touched, repo)
    assert untested == ["scripts/orphan.py"]
    # Still runs the invariant set; the #1187-widened scripts/**/*.py glob-scan
    # row now ALSO pulls the thread-caps test for ANY scripts/ file (additive —
    # the untested_touched WARN above is unaffected).
    assert set(tests) == set(sel.WORKFLOW_INVARIANT) | {"tests/test_shared_vm_thread_caps.py"}


# --- Case 6: pinned literal matches the LIVE tests/ tree ---------------------
def test_pinned_invariant_list_matches_live_tree():
    """Every WORKFLOW_INVARIANT entry must exist in the real repo tests/ tree.

    Fails LOUDLY if any pinned invariant test was renamed/removed without
    updating the literal — forcing a deliberate edit. Reads the live repo
    rather than a tmp_path fixture (as does case 6b below).
    """
    repo_root = Path(sel.__file__).resolve().parents[1]
    missing = sel.missing_invariants(repo_root)
    assert missing == [], (
        f"WORKFLOW_INVARIANT entries missing from the live tests/ tree: {missing}. "
        "Update the literal in scripts/select_step9c_tests.py deliberately."
    )
    # De-duplicated (no accidental double-listing). Cardinality is deliberately
    # NOT pinned as an integer (#1593): the retired `== N` count pin made every
    # pair of same-window registering PRs conflict on one shared line (#1584);
    # accidental-removal coverage lives in the manifest set-equality pin below
    # (case 6b). Per-addition rationale belongs as an inline comment on the
    # tuple entry itself, NOT in an accreting block here.
    assert len(sel.WORKFLOW_INVARIANT) == len(set(sel.WORKFLOW_INVARIANT))


# --- Case 6b (#1593): tuple <-> sorted-manifest set-equality pin -------------
# NOTE: this literal relpath is also the GLOB_SCAN_TESTS drift-pin anchor
# (test_glob_scan_map_matches_live_tree asserts it appears verbatim here).
_MANIFEST_RELPATH = "tests/step9c_workflow_invariant_manifest.txt"


def _manifest_lines(path: Path) -> list[str]:
    """Data lines of a WORKFLOW_INVARIANT manifest (blank + '#' lines skipped)."""
    return [
        ln.strip()
        for ln in path.read_text().splitlines()
        if ln.strip() and not ln.strip().startswith("#")
    ]


def _assert_manifest_matches(entries: tuple[str, ...], manifest: Path) -> None:
    """Core two-place pin: manifest well-formed (unique, sorted) + set-equal."""
    assert manifest.exists(), f"manifest missing: {manifest}"
    lines = _manifest_lines(manifest)
    assert len(lines) == len(set(lines)), f"duplicate lines in {manifest}"
    assert lines == sorted(lines), (
        f"{manifest} must stay sorted (python sorted()); add new entries at their sorted position."
    )
    tuple_set, manifest_set = set(entries), set(lines)
    only_tuple = sorted(tuple_set - manifest_set)
    only_manifest = sorted(manifest_set - tuple_set)
    assert tuple_set == manifest_set, (
        "WORKFLOW_INVARIANT (scripts/select_step9c_tests.py) and the manifest "
        f"disagree. In tuple only (add its manifest line, or a manifest line "
        f"was removed): {only_tuple}. In manifest only (add its tuple entry, "
        f"or a tuple entry was removed): {only_manifest}."
    )


def test_workflow_invariant_matches_manifest():
    """The tuple and the sorted manifest agree as SETS (#1593 count-pin
    replacement): an accidental tuple-entry removal (file still on disk, so
    missing_invariants() stays silent) fails HERE unless the manifest line
    was removed too — a deliberate two-place change. Registration = one tuple
    entry + one sorted manifest line; no shared line is ever edited, so
    concurrent registrations 3-way-merge cleanly server-side (#1584)."""
    repo_root = Path(sel.__file__).resolve().parents[1]
    _assert_manifest_matches(sel.WORKFLOW_INVARIANT, repo_root / _MANIFEST_RELPATH)


def test_manifest_pin_negative_paths(tmp_path: Path):
    """The pin's predicates fire on removal / unsorted / duplicate / empty manifests."""
    m = tmp_path / "manifest.txt"
    m.write_text("# header\n\ntests/test_a.py\ntests/test_b.py\n")
    _assert_manifest_matches(("tests/test_b.py", "tests/test_a.py"), m)  # order-free
    with pytest.raises(AssertionError, match="disagree"):
        _assert_manifest_matches(("tests/test_a.py",), m)  # dropped tuple entry
    with pytest.raises(AssertionError, match="disagree"):  # dropped manifest line
        _assert_manifest_matches(("tests/test_a.py", "tests/test_b.py", "tests/test_c.py"), m)
    m.write_text("tests/test_b.py\ntests/test_a.py\n")
    with pytest.raises(AssertionError, match="sorted"):
        _assert_manifest_matches(("tests/test_a.py", "tests/test_b.py"), m)
    m.write_text("tests/test_a.py\ntests/test_a.py\n")
    with pytest.raises(AssertionError, match="duplicate"):
        _assert_manifest_matches(("tests/test_a.py",), m)
    # Empty / comments-only manifest: zero data lines against a non-empty tuple
    # is a loud set mismatch, never a silent pass (critic concern 1).
    m.write_text("# header only\n\n")
    with pytest.raises(AssertionError, match="disagree"):
        _assert_manifest_matches(("tests/test_a.py",), m)


# --- Case 7: determinism — identical sorted output across two invocations ----
def test_selection_is_deterministic_and_sorted(tmp_path: Path):
    repo = _make_tree(tmp_path, ["test_widget.py", "test_widget_cli.py"])
    runner = _runner_for(["scripts/widget.py", "tests/test_widget.py"])
    t1, _ = sel.select_tests(sel.compute_touched("main", repo, _runner=runner), repo)
    t2, _ = sel.select_tests(sel.compute_touched("main", repo, _runner=runner), repo)
    assert t1 == t2
    assert t1 == sorted(t1)  # sorted order is the determinism guarantee


# --- Case 8: empty diff -> degenerate fallback (invariant set, never zero) ----
def test_empty_diff_falls_back_to_invariants(tmp_path: Path):
    repo = _make_tree(tmp_path, [])
    touched = sel.compute_touched("main", repo, _runner=_runner_for([]))
    assert touched == []
    tests, untested = sel.select_tests(touched, repo)
    assert set(tests) == set(sel.WORKFLOW_INVARIANT)
    assert len(tests) > 0  # never zero tests
    assert untested == []


# --- Case 9: #736 motivating case — gcp.py + dispatch_issue.py both covered ---
def test_issue_736_motivating_case(tmp_path: Path):
    """The kill criterion: #736's own change set must map to its tests."""
    repo = _make_tree(tmp_path, ["test_gcp_backend.py", "test_dispatch_issue_cli.py"])
    touched = sel.compute_touched(
        "main",
        repo,
        _runner=_runner_for(
            ["src/explore_persona_space/backends/gcp.py", "scripts/dispatch_issue.py"]
        ),
    )
    tests, untested = sel.select_tests(touched, repo)
    assert "tests/test_gcp_backend.py" in tests  # gcp stem -> test_*gcp*.py glob
    assert "tests/test_dispatch_issue_cli.py" in tests  # dispatch_issue -> *dispatch_issue* glob
    assert untested == []  # both had a mapped test


# --- Case 10: --json output shape -------------------------------------------
def test_json_output_shape(tmp_path: Path, monkeypatch, capsys):
    repo = _make_tree(tmp_path, ["test_widget.py"])
    monkeypatch.setattr(sel, "_resolve_work_root", lambda _arg: repo)
    monkeypatch.setattr(sel, "compute_touched", lambda *_a, **_k: ["scripts/widget.py"])
    # --no-fetch: hermetic (no git fetch); the fixture tree is not a git
    # checkout, so origin/main cannot resolve and the documented #1289
    # fallback resolves the default base to local "main".
    rc = sel.main(["--json", "--no-fetch", "--repo-root", str(repo)])
    assert rc == 0
    out = json.loads(capsys.readouterr().out)
    assert set(out.keys()) == {
        "tests",
        "untested_touched",
        "base",
        "missing_invariants",
        "selection_reasons",
        "n_tests",
        "recommended_timeout_s",
        "slow_tests_selected",
    }
    assert "tests/test_widget.py" in out["tests"]
    assert out["base"] == "main"  # the RESOLVED base (via the #1289 fallback), not the default
    assert out["missing_invariants"] == []  # all invariants present in the fixture tree
    assert out["selection_reasons"]["tests/test_widget.py"] == ["stem-map:scripts/widget.py"]
    # #1046 sizing fields are derived from the SAME selection the command runs:
    assert out["n_tests"] == len(out["tests"])
    assert out["recommended_timeout_s"] == sel.recommended_timeout_s(out["tests"])
    assert out["slow_tests_selected"] == [t for t in out["tests"] if t in sel.SLOW_TESTS]


# --- Case 10b (#1022): select_tests_with_reasons — reasons content ------------
def test_selection_reasons_content(tmp_path: Path):
    """Pins the reason vocabulary (invariant / touched-test / stem-map:<f> /
    glob-scan:<f>) and that select_tests still returns the identical 2-tuple."""
    repo = _make_tree(tmp_path, ["test_widget.py", "test_thing.py"])
    touched = ["scripts/widget.py", "tests/test_thing.py", "scripts/issue999_fake.py"]
    tests, untested, reasons = sel.select_tests_with_reasons(touched, repo)
    # Stem-mapped from a touched code file:
    assert reasons["tests/test_widget.py"] == ["stem-map:scripts/widget.py"]
    # A touched test file includes itself:
    assert reasons["tests/test_thing.py"] == ["touched-test"]
    # Glob-scan arm (#895; #1187 widened the row to scripts/**/*.py, so
    # scripts/widget.py now hits it too) records each covered touched file:
    assert reasons["tests/test_shared_vm_thread_caps.py"] == [
        "glob-scan:scripts/issue999_fake.py",
        "glob-scan:scripts/widget.py",
    ]
    # A pure invariant carries exactly the invariant reason:
    assert reasons["tests/test_task_workflow.py"] == ["invariant"]
    # Reason keys exactly cover the selection, and every reason list is sorted:
    assert set(reasons) == set(tests)
    assert all(rs == sorted(rs) for rs in reasons.values())
    # The unchanged-signature wrapper returns the IDENTICAL selection:
    assert sel.select_tests(touched, repo) == (tests, untested)
    assert untested == ["scripts/issue999_fake.py"]  # scan hit never marks "tested"


# --- Case 11: _resolve_work_root uses --show-toplevel (#851) ------------------
def test_resolve_work_root_uses_show_toplevel(tmp_path: Path, monkeypatch):
    """No-arg path calls `git rev-parse --path-format=absolute --show-toplevel`
    and returns the output resolved (NO dirname step) — the INVOKING checkout's
    root: the issue-worktree root from a worktree (where the branch diff and
    its branch-new tests/ files live), the main repo root when run there.
    Incident #851: the prior --git-common-dir+dirname recipe pinned the MAIN
    root, making `git diff main...HEAD` empty by construction. The
    `--repo-root` override still bypasses git entirely.
    """
    seen: dict[str, list[str]] = {}
    toplevel = tmp_path / "wt"

    class _Result:
        stdout = str(toplevel) + "\n"

    def _fake_run(argv, **_kw):
        seen["argv"] = argv
        return _Result()

    monkeypatch.setattr(sel.subprocess, "run", _fake_run)
    got = sel._resolve_work_root(None)
    # (i) locks the recipe to the invoking checkout's toplevel:
    assert seen["argv"] == ["git", "rev-parse", "--path-format=absolute", "--show-toplevel"]
    # (ii) return is the toplevel itself, resolved (no dirname step):
    assert got == toplevel.resolve()
    # (iii) the override branch never touches git — clear the recorded argv and
    #       confirm the arg path is returned resolved, without a git call:
    seen.clear()
    assert sel._resolve_work_root(str(tmp_path)) == tmp_path.resolve()
    assert "argv" not in seen  # git was not invoked on the override path


# --- Case 12: empty selection fails LOUD (defense-in-depth, #785) ------------
def test_empty_selection_fails_loud(tmp_path: Path, monkeypatch, capsys):
    """A structurally-impossible empty test list (bad repo_root resolution or all
    invariants missing) makes main() return 1 + a stderr line — never a silent
    exit-0 zero-test gate (the same silent-pass class the Step 9c shell guard
    closes at the shell level).
    """
    repo = _make_tree(tmp_path, [])
    monkeypatch.setattr(sel, "_resolve_work_root", lambda _arg: repo)
    monkeypatch.setattr(sel, "compute_touched", lambda *_a, **_k: [])
    # Force the degenerate empty selection the invariant set normally prevents.
    # (main() routes through select_tests_with_reasons as of #1022 — same
    # selection, plus the reasons map the Step 9c compare consumes.)
    monkeypatch.setattr(sel, "select_tests_with_reasons", lambda *_a, **_k: ([], [], {}))
    rc = sel.main(["--repo-root", str(repo)])
    assert rc == 1
    err = capsys.readouterr().err
    assert "EMPTY test selection" in err


# --- Real-git fixture for the #851 regression cases (13-14) -------------------
def _git(cwd: Path, *args: str) -> None:
    """Run git hermetically: no inherited global/system config (so no gpg
    signing, hooksPath, or identity leaks from the host), per-repo identity set
    explicitly by the caller."""
    env = {
        **os.environ,
        "GIT_CONFIG_GLOBAL": "/dev/null",
        "GIT_CONFIG_SYSTEM": "/dev/null",
    }
    subprocess.run(
        ["git", *args], cwd=str(cwd), env=env, check=True, capture_output=True, text=True
    )


def _make_git_repo_with_worktree(tmp_path: Path) -> tuple[Path, Path]:
    """Real-git #851 fixture. Returns ``(repo, wt)``:

    * ``repo`` — a git repo on branch ``main`` holding every invariant stub plus
      a committed ``tests/test_gone_on_branch.py``.
    * ``wt`` — a worktree on branch ``issue-x`` whose commit ADDS
      ``tests/test_foo.py`` + ``scripts/foo.py`` and DELETES
      ``tests/test_gone_on_branch.py``.
    """
    repo = _make_tree(tmp_path, [])
    (repo / "tests" / "test_gone_on_branch.py").write_text("# committed on main\n")
    _git(repo, "init", "-b", "main")
    _git(repo, "config", "user.email", "test@example.com")
    _git(repo, "config", "user.name", "Test")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-m", "main baseline")
    wt = tmp_path / "wt"
    _git(repo, "worktree", "add", "-b", "issue-x", str(wt))
    (wt / "tests" / "test_foo.py").write_text("def test_ok():\n    assert True\n")
    (wt / "scripts").mkdir()
    (wt / "scripts" / "foo.py").write_text("X = 1\n")
    _git(wt, "rm", "-q", "tests/test_gone_on_branch.py")
    _git(wt, "add", "tests/test_foo.py", "scripts/foo.py")
    _git(wt, "commit", "-m", "branch adds foo + deletes gone_on_branch")
    return repo, wt


# --- Case 13: the #851 regression — real-git worktree, end to end -------------
def test_branch_new_test_selected_from_worktree_real_git(tmp_path: Path, monkeypatch, capsys):
    """From an issue worktree, `main([])` must (a) include the branch-new
    touched test in the printed command (it exists ONLY in the worktree —
    the exact file #851 silently dropped), (b) map the branch-new code file to
    that test via the work-root glob (no untested-WARN), (c) drop the
    deleted-on-branch test (no pytest collection error), and (d) print the
    work-root + branch provenance breadcrumb.
    """
    repo, wt = _make_git_repo_with_worktree(tmp_path)
    # Precondition: the branch-new test exists ONLY in the worktree.
    assert not (repo / "tests" / "test_foo.py").exists()
    assert (wt / "tests" / "test_foo.py").exists()
    monkeypatch.chdir(wt)
    rc = sel.main([])
    assert rc == 0
    captured = capsys.readouterr()
    # (a) branch-new touched test selected (the #851 silent miss):
    assert "tests/test_foo.py" in captured.out
    # (c) deleted-on-branch test NOT in the printed command (the existence gate
    #     at the WORK root drops it — it exists at neither worktree HEAD nor
    #     the pytest cwd):
    assert "test_gone_on_branch.py" not in captured.out
    # (b) foo.py mapped to its branch-new test — no WARN:
    assert "untested touched file" not in captured.err
    # A real, non-empty diff: the empty-diff NOTE must NOT fire:
    assert "NOTE — empty diff" not in captured.err
    # (d) provenance breadcrumb: resolved work root + branch, on stderr:
    assert f"work root {wt.resolve()}" in captured.err
    assert "(branch: issue-x)" in captured.err


# --- Case 14: empty diff at the base checkout -> loud NOTE, invariant-only ----
def test_empty_diff_note_at_main_checkout_real_git(tmp_path: Path, monkeypatch, capsys):
    """At the main checkout (HEAD==main -> `main...HEAD` empty by construction)
    the fallback stays exit-0 invariant-only (case 8's documented degenerate
    contract) but is no longer SILENT: a `NOTE — empty diff` stderr line names
    the work root, and the breadcrumb records branch `main` — so a wrong-cwd
    run of a worktree-based task is visible in the Step 9c marker.
    """
    repo, _wt = _make_git_repo_with_worktree(tmp_path)
    monkeypatch.chdir(repo)
    rc = sel.main([])
    assert rc == 0
    captured = capsys.readouterr()
    # The #851 shape is no longer silent:
    assert "NOTE — empty diff" in captured.err
    # #1717 defect (d): the NOTE names uncommitted edits as a likely cause
    # (case-insensitive substring pin — the widened NOTE writes
    # "empty diff — commit first;" mid-sentence).
    assert "commit first" in captured.err.lower()
    # Provenance breadcrumb names the main checkout + branch:
    assert f"work root {repo.resolve()}" in captured.err
    assert "(branch: main)" in captured.err
    # Printed command is exactly the invariant-only set (sorted), behind the
    # #1046 sized-timeout prefix (tuple-derived — never hardcode 1980, so the
    # pin survives a WORKFLOW_INVARIANT count change):
    line = captured.out.strip().splitlines()[-1]
    prefix = (
        "timeout --kill-after=60s "
        f"{sel.recommended_timeout_s(sorted(sel.WORKFLOW_INVARIANT))}s uv run pytest "
    )
    # #1746: the printed command carries --continue-on-collection-errors so a
    # collection-broken selected file reports per-file instead of aborting rc=2.
    suffix = " --continue-on-collection-errors -v --tb=short"
    assert line.startswith(prefix) and line.endswith(suffix)
    files = line.removeprefix(prefix).removesuffix(suffix).split()
    assert files == sorted(sel.WORKFLOW_INVARIANT)


# --- Case 15: untested-WARN reaches stderr through main() ---------------------
def test_untested_warn_through_main(tmp_path: Path, monkeypatch, capsys):
    """Case 5 pins the `select_tests` return value; this pins the stderr
    emission path main() drives (the WARN line the Step 9c marker records)."""
    repo = _make_tree(tmp_path, [])  # no test_orphan*.py anywhere
    monkeypatch.setattr(sel, "_resolve_work_root", lambda _arg: repo)
    monkeypatch.setattr(sel, "compute_touched", lambda *_a, **_k: ["scripts/orphan.py"])
    rc = sel.main([])
    assert rc == 0
    captured = capsys.readouterr()
    assert "untested touched file: scripts/orphan.py" in captured.err
    # The breadcrumb fires on every run; the fixture tree is not a git
    # checkout, so the fail-soft branch read surfaces "unknown" (never a crash):
    assert "(branch: unknown)" in captured.err


# --- Case 16: glob-scan map — a scripts/issue*_*.py file (now via the #1187
# --- scripts/**/*.py row) selects the thread-caps test
def test_glob_scan_map_selects_thread_caps_for_issue_script(tmp_path: Path):
    repo = _make_tree(tmp_path, [])
    touched = sel.compute_touched("main", repo, _runner=_runner_for(["scripts/issue999_fake.py"]))
    tests, untested = sel.select_tests(touched, repo)
    assert "tests/test_shared_vm_thread_caps.py" in tests
    # Additive only: the file's OWN logic still has no mapped test -> WARN stays.
    assert untested == ["scripts/issue999_fake.py"]


# --- Case 17: per-row negatives under the #1187-widened thread-caps globs -----
def test_glob_scan_map_not_selected_for_non_matching_file(tmp_path: Path):
    repo = _make_tree(tmp_path, [])
    # scripts/pod.py: the widened scripts/**/*.py row DOES pull thread-caps
    # (documents the #1187 widening), still NOT the subprocess-env row.
    touched = sel.compute_touched("main", repo, _runner=_runner_for(["scripts/pod.py"]))
    tests, _ = sel.select_tests(touched, repo)
    assert "tests/test_shared_vm_thread_caps.py" in tests
    assert "tests/test_subprocess_env_explicit.py" not in tests
    # A genuinely non-matching src path (outside experiments/) pulls NEITHER row.
    touched = sel.compute_touched(
        "main", repo, _runner=_runner_for(["src/explore_persona_space/llm/api_dispatch.py"])
    )
    tests, _ = sel.select_tests(touched, repo)
    assert "tests/test_shared_vm_thread_caps.py" not in tests
    assert "tests/test_subprocess_env_explicit.py" not in tests


# --- Case 18: experiments run_*.py hits both rows; ** matches zero segments ---
def test_glob_scan_map_experiments_run_file_and_zero_segment(tmp_path: Path):
    repo = _make_tree(tmp_path, [])
    nested = "src/explore_persona_space/experiments/foo/run_bar.py"
    touched = sel.compute_touched("main", repo, _runner=_runner_for([nested]))
    tests, _ = sel.select_tests(touched, repo)
    assert "tests/test_shared_vm_thread_caps.py" in tests  # experiments/**/*.py row (#1187)
    assert "tests/test_subprocess_env_explicit.py" in tests  # */run_*.py row
    zero_seg = "src/explore_persona_space/experiments/run_top.py"
    touched = sel.compute_touched("main", repo, _runner=_runner_for([zero_seg]))
    tests, _ = sel.select_tests(touched, repo)
    assert "tests/test_shared_vm_thread_caps.py" in tests  # zero-segment **/
    assert "tests/test_subprocess_env_explicit.py" not in tests  # */ needs one dir


# --- Case 19: dispatcher glob row --------------------------------------------
def test_glob_scan_map_dispatcher_script(tmp_path: Path):
    repo = _make_tree(tmp_path, [])
    touched = sel.compute_touched(
        "main", repo, _runner=_runner_for(["scripts/dispatch_new_thing.py"])
    )
    tests, _ = sel.select_tests(touched, repo)
    assert "tests/test_subprocess_env_explicit.py" in tests
    # #1187: dispatcher scripts are scripts — the widened scripts/**/*.py
    # thread-caps row now matches them too.
    assert "tests/test_shared_vm_thread_caps.py" in tests


# --- Case 20: map row NOT selected when its test file is absent on disk -------
def test_glob_scan_map_key_missing_on_disk_not_selected(tmp_path: Path):
    """The (work_root / scan_test).exists() gate: a deleted-on-branch (or
    fixture-absent) scanning test is never emitted into the pytest command
    (same existence contract as the invariant set — no collection error)."""
    repo = _make_tree(tmp_path, [])
    (repo / "tests" / "test_shared_vm_thread_caps.py").unlink()
    touched = sel.compute_touched("main", repo, _runner=_runner_for(["scripts/issue999_fake.py"]))
    tests, _ = sel.select_tests(touched, repo)
    assert "tests/test_shared_vm_thread_caps.py" not in tests


# --- Live-tree pin for the map (mirrors case 6's curation discipline) ---------
def test_glob_scan_map_matches_live_tree():
    """Every GLOB_SCAN_TESTS key exists, its glob tuple matches real files on
    the live tree (aggregated per row — individual globs MAY legitimately match
    nothing today, e.g. run_factor_screen_*.py / experiments run_*.py are
    forward-looking guards with 0 hits at freeze time), each glob appears
    VERBATIM in the scanning test's own source (drift pin: a scanner
    renaming/narrowing its scan roots forces a deliberate map edit), and the
    map is disjoint from WORKFLOW_INVARIANT (no double-listing)."""
    repo_root = Path(sel.__file__).resolve().parents[1]
    assert sel.GLOB_SCAN_TESTS
    assert not set(sel.GLOB_SCAN_TESTS) & set(sel.WORKFLOW_INVARIANT)
    for test_file, globs in sel.GLOB_SCAN_TESTS.items():
        assert (repo_root / test_file).exists(), f"map key missing: {test_file}"
        assert globs, f"empty scan-glob tuple for {test_file}"
        hits = [p for g in globs for p in repo_root.glob(g)]
        assert hits, f"scan globs for {test_file} match nothing on the live tree"
        src = (repo_root / test_file).read_text()
        for g in globs:
            assert g in src, (
                f"{test_file} no longer scans {g!r} — GLOB_SCAN_TESTS drifted "
                "from the scanner's source; update the map verbatim."
            )


# --- Case 21 (#1046): recommended gate timeout — formula ----------------------
def test_recommended_timeout_formula():
    """T = BASE + PER_FILE*n + slow surcharges, floored at TIMEOUT_FLOOR_S."""
    no_slow = [f"tests/test_x{i}.py" for i in range(40)]
    assert sel.recommended_timeout_s(no_slow) == sel.TIMEOUT_BASE_S + 40 * sel.TIMEOUT_PER_FILE_S
    with_wl = [*no_slow, "tests/test_workflow_lint.py"]
    assert sel.recommended_timeout_s(with_wl) == (
        sel.TIMEOUT_BASE_S
        + 41 * sel.TIMEOUT_PER_FILE_S
        + sel.SLOW_TESTS["tests/test_workflow_lint.py"]
    )
    assert sel.recommended_timeout_s([]) == sel.TIMEOUT_FLOOR_S  # floor binds


# --- Case 22 (#1046): SLOW_TESTS live-tree drift pin ---------------------------
def test_slow_tests_pinned_to_live_tree():
    """Every SLOW_TESTS key exists in the real repo tests/ tree (drift pin,
    same curation rule as WORKFLOW_INVARIANT / GLOB_SCAN_TESTS)."""
    root = Path(sel.__file__).resolve().parents[1]
    missing = [t for t in sel.SLOW_TESTS if not (root / t).exists()]
    assert missing == [], (
        f"SLOW_TESTS entries missing from the live tests/ tree: {missing}. "
        "Update the literal in scripts/select_step9c_tests.py deliberately."
    )


# --- Case 23 (#1046): stdout carries the sized timeout prefix + stderr line ---
def test_stdout_command_carries_sized_timeout(tmp_path: Path, monkeypatch, capsys):
    """The printed command starts with the sized `timeout --kill-after=60s <T>s`
    prefix, and stderr carries the machine-greppable recommended-timeout-s
    sizing line (both derived from the invariant-only selection)."""
    repo = _make_tree(tmp_path, [])
    monkeypatch.setattr(sel, "_resolve_work_root", lambda _arg: repo)
    monkeypatch.setattr(sel, "compute_touched", lambda *_a, **_k: [])
    rc = sel.main([])
    assert rc == 0
    captured = capsys.readouterr()
    t = sel.recommended_timeout_s(sorted(sel.WORKFLOW_INVARIANT))
    line = captured.out.strip().splitlines()[-1]
    assert line.startswith(f"timeout --kill-after=60s {t}s uv run pytest ")
    assert f"recommended-timeout-s={t}" in captured.err
    # #1289: the sizing line also names the RESOLVED diff base (substring pin).
    assert "diff-base=" in captured.err


# --- Cases 24-29 (#1147): map_scan_tests() + the --map-files CLI mapping mode -
def test_map_scan_tests_thread_caps_glob(tmp_path: Path):
    """A scripts/ path maps to the thread-caps scan-test pair (scripts/**/*.py, #1187)."""
    repo = _make_tree(tmp_path, [])
    pairs = sel.map_scan_tests(["scripts/issue123_foo.py"], repo)
    assert pairs == [("tests/test_shared_vm_thread_caps.py", "scripts/issue123_foo.py")]


def test_map_scan_tests_dispatcher_glob(tmp_path: Path):
    """A scripts/dispatch_*.py path maps to BOTH scan-test pairs (#1187 widening)."""
    repo = _make_tree(tmp_path, [])
    pairs = sel.map_scan_tests(["scripts/dispatch_x.py"], repo)
    assert pairs == [
        ("tests/test_shared_vm_thread_caps.py", "scripts/dispatch_x.py"),
        ("tests/test_subprocess_env_explicit.py", "scripts/dispatch_x.py"),
    ]


def test_map_scan_tests_non_matching_empty(tmp_path: Path):
    """Paths outside every GLOB_SCAN_TESTS glob map to no pairs at all."""
    repo = _make_tree(tmp_path, [])
    files = ["tasks/running/1/body.md", ".claude/skills/issue/SKILL.md", "docs/foo.md"]
    assert sel.map_scan_tests(files, repo) == []


def test_map_scan_tests_missing_test_dropped(tmp_path: Path):
    """A glob hit whose scan test is absent from the work root is dropped."""
    bare = tmp_path / "bare"  # no tests/ tree at all
    bare.mkdir()
    assert sel.map_scan_tests(["scripts/issue123_foo.py"], bare) == []


def test_cli_map_files_tab_output_exit0(tmp_path: Path, capsys):
    """--map-files prints sorted `test<TAB>path` lines to stdout, rc 0; a
    non-matching list yields EMPTY stdout, still rc 0 (the gate's skip signal)."""
    repo = _make_tree(tmp_path, [])
    listing = tmp_path / "payload.txt"
    listing.write_text("scripts/issue123_foo.py\nscripts/dispatch_x.py\n")
    rc = sel.main(["--map-files", str(listing), "--repo-root", str(repo)])
    assert rc == 0
    out = capsys.readouterr().out
    assert out.splitlines() == [
        "tests/test_shared_vm_thread_caps.py\tscripts/dispatch_x.py",
        "tests/test_shared_vm_thread_caps.py\tscripts/issue123_foo.py",
        "tests/test_subprocess_env_explicit.py\tscripts/dispatch_x.py",
    ]
    # Non-matching payload: empty stdout, rc 0 — NOT an error.
    listing.write_text("docs/foo.md\ntasks/running/1/body.md\n")
    rc = sel.main(["--map-files", str(listing), "--repo-root", str(repo)])
    assert rc == 0
    assert capsys.readouterr().out == ""


def test_cli_map_files_unreadable_exit1(tmp_path: Path, capsys):
    """An unreadable --map-files input is exit 1 + one stderr line (fail CLOSED)."""
    repo = _make_tree(tmp_path, [])
    rc = sel.main(["--map-files", str(tmp_path / "nope.txt"), "--repo-root", str(repo)])
    assert rc == 1
    err = capsys.readouterr().err
    assert "cannot read --map-files input" in err


def test_cli_map_files_missing_test_warns(tmp_path: Path, capsys):
    """A glob hit whose scan test is absent from the work root drops the pair
    WITH a stderr WARN naming it (never a silent shrink)."""
    bare = tmp_path / "bare"
    bare.mkdir()
    listing = tmp_path / "payload.txt"
    listing.write_text("scripts/issue123_foo.py\n")
    rc = sel.main(["--map-files", str(listing), "--repo-root", str(bare)])
    assert rc == 0
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "WARN — scan test tests/test_shared_vm_thread_caps.py" in captured.err
    assert "pair dropped" in captured.err


# --- #1613: zero-resolution guard (source-file argument / deletion-only list) ---
@pytest.mark.parametrize("suffix", [".py", ".sh"])
def test_cli_map_files_source_file_argument_exit2(tmp_path: Path, capsys, suffix: str):
    """A .py/.sh FILE argument whose content lines resolve to zero repo paths
    with zero pairs is a usage error: exit 2, empty stdout, one tab-free
    stderr ERROR line (#1613 — the #1610 malformed-verify shape). An existing
    ABSOLUTE-path content line must not silence the guard (for absolute f,
    ``work_root / f`` yields f itself; the existence scan skips such lines)."""
    repo = _make_tree(tmp_path, [])
    abs_line = tmp_path / "abs_exists.txt"
    abs_line.write_text("present\n")
    if suffix == ".py":
        content = f"import os\n\nX = 1\n{abs_line}\n\ndef f():\n    return 1\n"
    else:
        content = f"#!/usr/bin/env bash\nset -euo pipefail\necho hi\n{abs_line}\n"
    src = tmp_path / f"some_module{suffix}"
    src.write_text(content)
    rc = sel.main(["--map-files", str(src), "--repo-root", str(repo)])
    assert rc == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "looks like a source file" in captured.err
    assert "\t" not in captured.err
    assert "recommended-timeout-s=" not in captured.err


def test_cli_map_files_valid_list_zero_pairs_no_guard_exit0(tmp_path: Path, capsys):
    """A valid path-list whose paths EXIST but map to no pairs stays silent-clean
    (rc 0, empty stdout, no #1613 guard line): zero-PAIR alone never fires it."""
    repo = _make_tree(tmp_path, [])
    (repo / "docs").mkdir()
    (repo / "docs" / "foo.md").write_text("# doc\n")
    listing = tmp_path / "payload.txt"
    listing.write_text("docs/foo.md\n")
    rc = sel.main(["--map-files", str(listing), "--repo-root", str(repo)])
    assert rc == 0
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "ZERO existing repo paths" not in captured.err
    assert "ERROR" not in captured.err


def test_cli_map_files_zero_resolution_list_warns_exit0(tmp_path: Path, capsys):
    """A path-list resolving to ZERO existing repo paths with zero pairs (the
    deletion-only ``git diff --name-only`` shape — status-D paths absent from
    the worktree) draws one hedged tab-free stderr WARN, rc 0 — never exit 2.
    An empty / whitespace-only FILE never fires the guard at all (rc 0)."""
    repo = _make_tree(tmp_path, [])
    listing = tmp_path / "payload.txt"
    listing.write_text("docs/nope.md\nexternal/gone.md\n")
    rc = sel.main(["--map-files", str(listing), "--repo-root", str(repo)])
    assert rc == 0
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "WARN — --map-files input" in captured.err
    assert "deletion-only" in captured.err
    assert "\t" not in captured.err
    assert "recommended-timeout-s=" not in captured.err
    # Empty / whitespace-only FILE: `files` is empty, so the guard must NOT fire.
    listing.write_text("\n   \n")
    rc = sel.main(["--map-files", str(listing), "--repo-root", str(repo)])
    assert rc == 0
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "ZERO existing repo paths" not in captured.err
    assert "looks like a source file" not in captured.err


def test_cli_map_files_pairs_suppress_zero_resolution_guard(tmp_path: Path, capsys):
    """A nonexistent (deleted) eligible .py payload that still glob-scan-maps to
    pairs suppresses the guard entirely (the ``not all_pairs`` conjunct):
    mapped stdout, no #1613 line from either arm."""
    repo = _make_tree(tmp_path, [])
    listing = tmp_path / "payload.txt"
    listing.write_text("scripts/issue123_foo.py\n")
    rc = sel.main(["--map-files", str(listing), "--repo-root", str(repo)])
    assert rc == 0
    captured = capsys.readouterr()
    assert captured.out != ""
    assert "ZERO existing repo paths" not in captured.err
    assert "looks like a source file" not in captured.err


# --- #1791: hostile --map-files content must reach the #1613 diagnostic -------
# (not crash at a content-derived filesystem probe). The three line shapes:
# a bare >NAME_MAX prose line (OSError Errno 36 at the guard's exists() scan),
# an arm-eligible scripts/<300 chars>.py line (OSError at the stem probe,
# which runs BEFORE the guard), and a scripts/*.py glob-metachar line
# (ValueError "Invalid pattern" at the content-derived stem glob). The CLI
# cases use a BARE work root (the test_cli_map_files_missing_test_dropped
# precedent): the scripts/ lines match the broad scripts/**/*.py scan glob,
# and against a _make_tree root the resulting pairs would legitimately
# SUPPRESS the guard (the `all_pairs` conjunct) — the bare root drops those
# pairs so the guard's verdict is what these cases pin.
_HOSTILE_LINES = (
    "lorem ipsum dolor sit amet consectetur adipiscing " * 8,
    "scripts/" + "x" * 300 + ".py",
    "scripts/*.py",
)


def test_cli_map_files_hostile_lines_py_argument_exit2(tmp_path: Path, capsys):
    """A .py-named --map-files arg holding all three hostile line shapes
    reaches the #1613 source-file ERROR: rc 2, empty stdout, no traceback
    (an uncaught OSError/ValueError would propagate and fail this test)."""
    bare = tmp_path / "bare"
    bare.mkdir()
    src = tmp_path / "some_module.py"
    src.write_text("\n".join(_HOSTILE_LINES) + "\n")
    rc = sel.main(["--map-files", str(src), "--repo-root", str(bare)])
    assert rc == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "looks like a source file" in captured.err


def test_cli_map_files_hostile_lines_list_argument_warns_exit0(tmp_path: Path, capsys):
    """The same three hostile lines in a .md-named arg take the #1613 hedged
    WARN branch: rc 0, empty stdout, no traceback."""
    bare = tmp_path / "bare"
    bare.mkdir()
    listing = tmp_path / "notes.md"
    listing.write_text("\n".join(_HOSTILE_LINES) + "\n")
    rc = sel.main(["--map-files", str(listing), "--repo-root", str(bare)])
    assert rc == 0
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "ZERO existing repo paths" in captured.err
    assert "looks like a source file" not in captured.err


def test_cli_map_files_binary_argument_exit1(tmp_path: Path, capsys):
    """A mis-passed BINARY (undecodable) --map-files arg takes the existing
    rc-1 "cannot read" path — UnicodeDecodeError is a ValueError (#1791) —
    never an uncaught traceback."""
    repo = _make_tree(tmp_path, [])
    blob = tmp_path / "payload.bin"
    blob.write_bytes(b"\x80\x81\xfe\x00")
    rc = sel.main(["--map-files", str(blob), "--repo-root", str(repo)])
    assert rc == 1
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "cannot read --map-files input" in captured.err


def test_safe_exists_unstatable_paths_false(tmp_path: Path):
    """_safe_exists: a >NAME_MAX path reads as absent (no OSError escape);
    stat-able paths are byte-identical to Path.exists()."""
    assert sel._safe_exists(tmp_path / ("x" * 300)) is False
    assert sel._safe_exists(tmp_path / "nul\x00name") is False
    assert sel._safe_exists(tmp_path) is True
    assert sel._safe_exists(tmp_path / "absent.txt") is False


def test_safe_glob_invalid_pattern_empty(tmp_path: Path):
    """_safe_glob: a content-derived invalid pattern yields [] (no ValueError
    escape); a valid pattern returns exactly sorted(root.glob(...))."""
    (tmp_path / "test_alpha.py").write_text("# stub\n")
    (tmp_path / "test_beta.py").write_text("# stub\n")
    assert sel._safe_glob(tmp_path, "test_***.py") == []
    assert sel._safe_glob(tmp_path, "test_*alpha*.py") == sorted(tmp_path.glob("test_*alpha*.py"))
    assert sel._safe_glob(tmp_path, "test_*alpha*.py") == [tmp_path / "test_alpha.py"]


# --- #1289: diff-base resolution (fetched origin/main default) -----------------
def _git_out(cwd: Path, *args: str) -> str:
    """Hermetic git runner that RETURNS stdout (rev-parse probes for cases 30-32)."""
    env = {
        **os.environ,
        "GIT_CONFIG_GLOBAL": "/dev/null",
        "GIT_CONFIG_SYSTEM": "/dev/null",
    }
    proc = subprocess.run(
        ["git", *args], cwd=str(cwd), env=env, check=True, capture_output=True, text=True
    )
    return proc.stdout.strip()


def _make_lagging_clone_with_worktree(tmp_path: Path) -> tuple[Path, Path, Path]:
    """#1289 fixture: (upstream, clone, wt) — file:// path remotes, zero network.

    upstream main: U1 (invariant stubs) -> U2 (adds scripts/foreign.py +
    tests/test_foreign.py). clone: cloned at U1, so LOCAL main == U1 (behind);
    then ``git fetch origin`` so origin/main == U2; worktree branch issue-y cut
    from origin/main (the #1214 cut recipe), adding scripts/bar.py +
    tests/test_bar.py in one commit. Local main stays at U1 — the lagging-root
    shape of 2026-07-12.
    """
    upstream = _make_tree(tmp_path, [])
    _git(upstream, "init", "-b", "main")
    _git(upstream, "config", "user.email", "test@example.com")
    _git(upstream, "config", "user.name", "Test")
    _git(upstream, "add", "-A")
    _git(upstream, "commit", "-m", "U1 baseline")
    clone = tmp_path / "clone"
    _git(tmp_path, "clone", "--quiet", str(upstream), str(clone))  # origin set automatically
    _git(clone, "config", "user.email", "test@example.com")
    _git(clone, "config", "user.name", "Test")
    # Advance upstream to U2 AFTER the clone: local main in the clone stays at U1.
    (upstream / "scripts").mkdir(exist_ok=True)
    (upstream / "scripts" / "foreign.py").write_text("F = 1\n")
    (upstream / "tests" / "test_foreign.py").write_text("def test_f():\n    assert True\n")
    _git(upstream, "add", "scripts/foreign.py", "tests/test_foreign.py")
    _git(upstream, "commit", "-m", "U2 foreign commit")
    _git(clone, "fetch", "origin")  # origin/main == U2; local main == U1 (lagging)
    wt = tmp_path / "wt-1289"
    _git(clone, "worktree", "add", "-b", "issue-y", str(wt), "origin/main")
    (wt / "scripts" / "bar.py").write_text("B = 1\n")
    (wt / "tests" / "test_bar.py").write_text("def test_b():\n    assert True\n")
    _git(wt, "add", "scripts/bar.py", "tests/test_bar.py")
    _git(wt, "commit", "-m", "branch adds bar")
    return upstream, clone, wt


def _run_json_main(monkeypatch, capsys, wt: Path, argv: list[str]) -> tuple[int, dict, str]:
    """Run sel.main(argv) from *wt*; return (rc, parsed --json stdout, stderr)."""
    monkeypatch.chdir(wt)
    rc = sel.main(argv)
    captured = capsys.readouterr()
    return rc, json.loads(captured.out), captured.err


# --- Case 30: THE acceptance test — lagging local main, foreign files excluded -
def test_default_base_excludes_foreign_files_on_lagging_root(tmp_path: Path, monkeypatch, capsys):
    """With local main behind origin/main (the 2026-07-12 shape), the DEFAULT
    base selects only the branch's own mapped tests — zero foreign-commit
    tests. Contrast leg: the old ``--base main`` default DID select the
    foreign test (the #1281 41-file gate inflation). Twin-run equality leg:
    after fast-forwarding local main to origin/main (a synced root), the
    ``--base main`` selection equals the lagging-root default selection —
    the fix changes which ref NAMES the base, not the selected set.
    """
    _upstream, clone, wt = _make_lagging_clone_with_worktree(tmp_path)
    rc, out, _err = _run_json_main(monkeypatch, capsys, wt, ["--json"])
    assert rc == 0
    assert out["base"] == "origin/main"
    assert "tests/test_bar.py" in out["tests"]  # the branch's own test
    assert "tests/test_foreign.py" not in out["tests"]  # foreign commit excluded
    assert not any(
        r == "stem-map:scripts/foreign.py" for rs in out["selection_reasons"].values() for r in rs
    )
    # Contrast leg (documents the old bug): explicit --base main on the
    # lagging root pulls the foreign origin-side commit into the selection.
    rc2, out_main, _err2 = _run_json_main(monkeypatch, capsys, wt, ["--json", "--base", "main"])
    assert rc2 == 0
    assert "tests/test_foreign.py" in out_main["tests"]
    # Twin-run equality leg: sync local main (ff to origin/main), then the
    # local-main selection == the lagging-root default-base selection.
    _git(clone, "merge", "--ff-only", "origin/main")
    rc3, out_synced, _err3 = _run_json_main(monkeypatch, capsys, wt, ["--json", "--base", "main"])
    assert rc3 == 0
    assert out_synced["tests"] == out["tests"]


# --- Case 31: the default base FETCHES origin/main; merge-base is stable -------
def test_default_base_fetches_origin_main(tmp_path: Path, monkeypatch, capsys):
    """Advance upstream to U3 after the clone's last fetch: the default run
    must fetch (clone-side origin/main advances to the upstream tip) while
    the SELECTION stays the branch's own tests (merge-base == cut point,
    invariant under origin/main advancing)."""
    upstream, clone, wt = _make_lagging_clone_with_worktree(tmp_path)
    (upstream / "scripts" / "foreign2.py").write_text("F2 = 1\n")
    (upstream / "tests" / "test_foreign2.py").write_text("def test_f2():\n    assert True\n")
    _git(upstream, "add", "scripts/foreign2.py", "tests/test_foreign2.py")
    _git(upstream, "commit", "-m", "U3 foreign commit")
    before = _git_out(clone, "rev-parse", "origin/main")
    upstream_tip = _git_out(upstream, "rev-parse", "main")
    assert before != upstream_tip  # precondition: clone is one fetch behind
    rc, out, _err = _run_json_main(monkeypatch, capsys, wt, ["--json"])
    assert rc == 0
    assert _git_out(clone, "rev-parse", "origin/main") == upstream_tip  # the fetch ran
    assert out["base"] == "origin/main"
    # Selection unchanged from case 30's shape (merge-base == the cut point):
    assert "tests/test_bar.py" in out["tests"]
    assert "tests/test_foreign.py" not in out["tests"]
    assert "tests/test_foreign2.py" not in out["tests"]


# --- Case 32: --no-fetch skips the fetch, keeps the origin/main base -----------
def test_no_fetch_skips_fetch(tmp_path: Path, monkeypatch, capsys):
    upstream, clone, wt = _make_lagging_clone_with_worktree(tmp_path)
    (upstream / "scripts" / "foreign2.py").write_text("F2 = 1\n")
    _git(upstream, "add", "scripts/foreign2.py")
    _git(upstream, "commit", "-m", "U3 foreign commit")
    before = _git_out(clone, "rev-parse", "origin/main")
    rc, out, _err = _run_json_main(monkeypatch, capsys, wt, ["--json", "--no-fetch"])
    assert rc == 0
    assert _git_out(clone, "rev-parse", "origin/main") == before  # no ref mutation
    assert out["base"] == "origin/main"  # last-fetched ref still resolves + is used


# --- Case 33: no origin remote -> loud fallback to local main ------------------
def test_fallback_to_local_main_when_origin_main_unresolvable(tmp_path: Path, monkeypatch, capsys):
    """The existing no-remote fixture: origin/main cannot resolve, so the
    default base falls back LOUDLY to local 'main' (pre-#1289 behavior)."""
    _repo, wt = _make_git_repo_with_worktree(tmp_path)
    rc, out, err = _run_json_main(monkeypatch, capsys, wt, ["--json"])
    assert rc == 0
    assert out["base"] == "main"
    assert "falling back to local 'main'" in err


# --- Case 34: a non-origin/ --base is used verbatim with ZERO git calls --------
def test_resolve_base_verbatim_local_ref_makes_no_git_calls(tmp_path: Path, monkeypatch):
    def _boom(*_a, **_k):
        raise AssertionError("git must not be called for a non-origin/ base")

    monkeypatch.setattr(sel.subprocess, "run", _boom)
    assert sel.resolve_base("main", tmp_path) == "main"
    assert sel.resolve_base("feature-x", tmp_path) == "feature-x"


# --- Case 35: fetch failure degrades to the last-fetched origin/main -----------
def test_resolve_base_fetch_failure_degrades_to_stale_origin_main(
    tmp_path: Path, monkeypatch, capsys
):
    """A failing bounded fetch (lock contention / offline / auth) NOTEs and
    degrades to the still-resolving last-fetched origin/main — never blocks,
    never falls back to local main while the remote-tracking ref exists."""
    seen: dict[str, dict] = {}

    class _Ok:
        returncode = 0

    def _fake_run(argv, **kwargs):
        if argv[:3] == ["git", "fetch", "origin"]:
            seen["fetch_kwargs"] = kwargs
            raise subprocess.CalledProcessError(1, argv)
        assert argv[:2] == ["git", "rev-parse"]
        return _Ok()

    monkeypatch.setattr(sel.subprocess, "run", _fake_run)
    got = sel.resolve_base("origin/main", tmp_path)
    assert got == "origin/main"
    err = capsys.readouterr().err
    assert "git fetch origin main failed" in err
    assert seen["fetch_kwargs"]["timeout"] == sel.FETCH_TIMEOUT_S  # the bounded fetch


# --- Cases 36+ (#1299): the import-map arm -------------------------------------
# Fixtures write REAL import statements into tmp-tree test files (the shared
# _make_tree stubs contain no imports and no touched-stem substrings, so the
# new arm no-ops on every pre-existing fixture by construction).


def _make_import_tree(tmp_path: Path, files: dict[str, str]) -> Path:
    """_make_tree plus tests/-relative files with real (import-bearing) content."""
    repo = _make_tree(tmp_path, [])
    for rel, content in files.items():
        p = repo / "tests" / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content)
    return repo


# --- Case 36: THE durability pin — the #1286 fixture shape ---------------------
def test_import_map_selects_unrelated_filename(tmp_path: Path):
    """An importing test whose NAME shares no touched stem is selected (#1299).

    Durability pin. Founding incident #1286:
    tests/test_issue810_uh_pack_validation.py imports issue810_common but was
    reachable by no selection arm ("uh_pack_validation" contains no touched
    stem), so #1286 hand-appended it to the gate command.
    """
    repo = _make_import_tree(
        tmp_path, {"test_uh_pack_validation.py": "from issue810_common import validate\n"}
    )
    tests, untested, reasons = sel.select_tests_with_reasons(["scripts/issue810_common.py"], repo)
    assert "tests/test_uh_pack_validation.py" in tests
    assert reasons["tests/test_uh_pack_validation.py"] == ["import-map:scripts/issue810_common.py"]
    assert untested == []  # the import hit marks the touched file tested


# --- Case 37: function-level import (ast.walk, not module-top-only) ------------
def test_import_map_function_level_import_selected(tmp_path: Path):
    """Mirrors the real file's L138/L178: the ONLY import is inside a test body."""
    repo = _make_import_tree(
        tmp_path,
        {
            "test_uh_pack.py": (
                "def test_x():\n    import issue810_fit_readout as fr\n    assert fr\n"
            )
        },
    )
    tests, untested, reasons = sel.select_tests_with_reasons(
        ["scripts/issue810_fit_readout.py"], repo
    )
    assert "tests/test_uh_pack.py" in tests
    assert reasons["tests/test_uh_pack.py"] == ["import-map:scripts/issue810_fit_readout.py"]
    assert untested == []


# --- Case 38: dotted scripts.X import (scripts/ is a package) ------------------
def test_import_map_dotted_scripts_import(tmp_path: Path):
    repo = _make_import_tree(tmp_path, {"test_consumer.py": "from scripts.widgetlib import x\n"})
    tests, untested, reasons = sel.select_tests_with_reasons(["scripts/widgetlib.py"], repo)
    assert "tests/test_consumer.py" in tests
    # #1688: the dotted-ref arm also fires on the literal `scripts.widgetlib` text.
    assert reasons["tests/test_consumer.py"] == [
        "dotted-ref:scripts/widgetlib.py",
        "import-map:scripts/widgetlib.py",
    ]
    assert untested == []


# --- Case 39: src/explore_persona_space dotted package modules -----------------
def test_import_map_src_package_module(tmp_path: Path):
    repo = _make_import_tree(
        tmp_path,
        {
            # the M.a candidate-join form: touched ...foo/bar.py, import from ...foo
            "test_pkg_consumer.py": "from explore_persona_space.foo import bar\n",
            "test_flat_consumer.py": "from explore_persona_space.task_widget import x\n",
        },
    )
    tests, untested, reasons = sel.select_tests_with_reasons(
        [
            "src/explore_persona_space/foo/bar.py",
            "src/explore_persona_space/task_widget.py",
        ],
        repo,
    )
    assert "tests/test_pkg_consumer.py" in tests
    assert reasons["tests/test_pkg_consumer.py"] == [
        "import-map:src/explore_persona_space/foo/bar.py"
    ]
    assert "tests/test_flat_consumer.py" in tests
    # #1688: the contiguous dotted text `explore_persona_space.task_widget` also
    # fires the dotted-ref arm (the pkg_consumer form above stays import-only —
    # `explore_persona_space.foo.bar` is not contiguous in its text).
    assert reasons["tests/test_flat_consumer.py"] == [
        "dotted-ref:src/explore_persona_space/task_widget.py",
        "import-map:src/explore_persona_space/task_widget.py",
    ]
    assert untested == []


# --- Case 40: an import hit suppresses the untested_touched WARN ----------------
def test_import_map_marks_touched_file_tested(tmp_path: Path):
    """Touched module with an importing test and NO stem-named test -> no WARN."""
    repo = _make_import_tree(tmp_path, {"test_something_else.py": "import orphanlib\n"})
    tests, untested, _ = sel.select_tests_with_reasons(["scripts/orphanlib.py"], repo)
    assert "tests/test_something_else.py" in tests
    assert untested == []


# --- Case 41: monotonicity — the arm only ever GROWS the selection --------------
def test_import_map_only_grows_selection(tmp_path: Path):
    """Same touched set, tree WITH vs WITHOUT the importing test file:
    WITH-selection is a superset and every WITHOUT reason list is preserved
    verbatim (the plan's acceptance criterion (b))."""
    touched = ["scripts/widgetlib.py", "scripts/orphan.py"]
    repo_without = _make_tree(tmp_path / "without", ["test_widgetlib.py"])
    t_without, u_without, r_without = sel.select_tests_with_reasons(touched, repo_without)
    repo_with = _make_tree(tmp_path / "with", ["test_widgetlib.py"])
    (repo_with / "tests" / "test_importer.py").write_text("import widgetlib\n")
    t_with, u_with, r_with = sel.select_tests_with_reasons(touched, repo_with)
    assert set(t_with) >= set(t_without)
    for test, rs in r_without.items():
        assert r_with[test] == rs  # pre-existing reason lists preserved verbatim
    assert "tests/test_importer.py" in t_with
    # orphan.py has no test in either tree; widgetlib.py is stem-mapped in both.
    assert u_without == ["scripts/orphan.py"]
    assert u_with == ["scripts/orphan.py"]


# --- Case 42: a broken test file WARNs + is skipped; never crashes ---------------
def test_import_map_broken_test_file_warns_not_crash(tmp_path: Path, capsys):
    repo = _make_import_tree(
        tmp_path,
        {
            "test_good_importer.py": "import widgetlib\n",
            # contains the pre-filter token, so it IS parsed — and fails.
            "test_broken.py": "import widgetlib\ndef broken(:\n",
        },
    )
    tests, untested, _ = sel.select_tests_with_reasons(["scripts/widgetlib.py"], repo)
    assert "tests/test_good_importer.py" in tests  # the valid hit still selected
    assert "tests/test_broken.py" not in tests  # broken file not import-selected
    assert untested == []
    err = capsys.readouterr().err
    assert err.count("import-map cannot parse") == 1
    assert "test_broken.py" in err
    # 1 failure over the ~41-file fixture tree is < 5%: no aggregate WARN.
    assert "systemic tests/ breakage" not in err


# --- Case 43: undecodable file WARNs via the same fail-soft path ----------------
def test_import_map_undecodable_file_warns_when_scanning(tmp_path: Path, capsys):
    """A read/decode failure (UnicodeDecodeError, a ValueError) takes the same
    WARN-and-skip path as a SyntaxError — the positive control for case 44's
    zero-read proof (the raw read happens BEFORE the substring pre-filter)."""
    repo = _make_import_tree(tmp_path, {"test_ok.py": "import widgetlib\n"})
    (repo / "tests" / "test_undecodable.py").write_bytes(b"import widgetlib\n\xff\xfe bad")
    tests, _, _ = sel.select_tests_with_reasons(["scripts/widgetlib.py"], repo)
    assert "tests/test_ok.py" in tests
    err = capsys.readouterr().err
    assert "import-map cannot parse" in err
    assert "test_undecodable.py" in err


# --- Case 44: workflow-surface-only diff -> ZERO file reads (early return) ------
def test_import_map_no_eligible_touched_skips_scan(tmp_path: Path, capsys):
    """No import-map-eligible touched file -> the scan never reads tests/.

    Proof: an undecodable test file is planted; any scan pass reads raw text
    BEFORE the pre-filter (case 43 shows that read WARNs), so the absence of a
    WARN here proves the zero-read early return.
    """
    repo = _make_import_tree(tmp_path, {})
    (repo / "tests" / "test_undecodable.py").write_bytes(b"import widgetlib\n\xff\xfe bad")
    hits, tested = sel.import_map_hits([".claude/skills/issue/SKILL.md", "notes.md"], repo)
    assert hits == {} and tested == set()
    tests, untested, _ = sel.select_tests_with_reasons([".claude/skills/issue/SKILL.md"], repo)
    err = capsys.readouterr().err
    assert "import-map cannot parse" not in err  # zero reads: never touched the bad file
    assert set(tests) == set(sel.WORKFLOW_INVARIANT)
    assert untested == []


# --- Case 45: subdir tests (tests/experiments/) are in the rglob scope ----------
def test_import_map_subdir_test_selected(tmp_path: Path):
    repo = _make_import_tree(tmp_path, {"experiments/test_sub.py": "import widgetlib\n"})
    tests, _, reasons = sel.select_tests_with_reasons(["scripts/widgetlib.py"], repo)
    assert "tests/experiments/test_sub.py" in tests
    assert reasons["tests/experiments/test_sub.py"] == ["import-map:scripts/widgetlib.py"]


# --- Case 46: precision — an UNRELATED import is not selected -------------------
def test_import_map_unrelated_import_not_selected(tmp_path: Path):
    repo = _make_import_tree(
        tmp_path, {"test_bystander.py": "import numpy\nfrom otherlib import thing\n"}
    )
    tests, _, _ = sel.select_tests_with_reasons(["scripts/widgetlib.py"], repo)
    assert "tests/test_bystander.py" not in tests


# --- Case 47: precision — relative imports (node.level > 0) are skipped ---------
def test_import_map_relative_import_skipped(tmp_path: Path):
    # The text CONTAINS the pre-filter token, so the file IS parsed; the level
    # check (not the pre-filter) is what excludes it.
    repo = _make_import_tree(
        tmp_path, {"test_relative.py": "from . import widgetlib\nfrom .widgetlib import x\n"}
    )
    tests, _, _ = sel.select_tests_with_reasons(["scripts/widgetlib.py"], repo)
    assert "tests/test_relative.py" not in tests


# --- Case 48: aggregate parse-failure WARN (systemic-breakage signal) -----------
def test_import_map_aggregate_parse_failure_warn(tmp_path: Path, capsys):
    """>5% of scanned test files failing to parse emits ONE extra summary WARN.

    The broken-stub count is sized PROPORTIONALLY to the seeded fixture tree:
    ``_make_tree`` seeds one stub per live ``WORKFLOW_INVARIANT`` /
    ``GLOB_SCAN_TESTS`` member, so a hardcoded count is a registry-growth
    knife-edge — at 60 members a fixed 3 broken files read 3/60 = 5.00%,
    which the strict ``> 5%`` gate does NOT fire (#1632: both aggregate-WARN
    tests went red on trunk when the registry grew).
    """
    repo = _make_import_tree(tmp_path, {"test_ok.py": "import widgetlib\n"})
    # Count exactly what the scanner enumerates (recursive test_*.py), then add
    # enough broken files to clear the strict >5% threshold at ANY registry size.
    n_pre = len(list((repo / "tests").rglob("test_*.py")))
    n_broken = math.ceil(0.05 * n_pre) + 1
    assert n_broken / (n_pre + n_broken) > 0.05  # self-check: aggregate WARN must fire
    for i in range(n_broken):
        (repo / "tests" / f"test_broken_{i}.py").write_text("import widgetlib\ndef broken(:\n")
    tests, _, _ = sel.select_tests_with_reasons(["scripts/widgetlib.py"], repo)
    assert "tests/test_ok.py" in tests
    err = capsys.readouterr().err
    assert err.count("import-map cannot parse") == n_broken  # one per-file WARN each
    assert err.count("systemic tests/ breakage") == 1  # exactly one aggregate WARN


# --- Case 49: touched_module_names resolution rules ------------------------------
def test_touched_module_names_resolution():
    m = sel.touched_module_names(
        [
            "scripts/issue810_common.py",
            "scripts/pkg/__init__.py",
            "src/explore_persona_space/a/b.py",
            "src/explore_persona_space/a/__init__.py",
            "tests/test_foo.py",  # never eligible — the touched-test arm owns tests/
            "scripts/__init__.py",  # the scripts package marker itself: no name
            "other/module.py",  # outside the two eligible roots
            "scripts/notes.md",  # not .py
        ]
    )
    assert m["issue810_common"] == {"scripts/issue810_common.py"}
    assert m["scripts.issue810_common"] == {"scripts/issue810_common.py"}
    assert m["pkg"] == {"scripts/pkg/__init__.py"}
    assert m["scripts.pkg"] == {"scripts/pkg/__init__.py"}
    assert m["explore_persona_space.a.b"] == {"src/explore_persona_space/a/b.py"}
    assert m["explore_persona_space.a"] == {"src/explore_persona_space/a/__init__.py"}
    ineligible = {"test_foo", "module", "notes", "scripts"}
    assert not any(name.rsplit(".", 1)[-1] in ineligible for name in m)


# --- Case 50: end-to-end through main() --json — schema unchanged, reason carried
def test_cli_json_carries_import_map_reason(tmp_path: Path, monkeypatch, capsys):
    repo = _make_import_tree(tmp_path, {"test_uh_pack.py": "import widgetlib\n"})
    monkeypatch.setattr(sel, "_resolve_work_root", lambda _arg: repo)
    monkeypatch.setattr(sel, "compute_touched", lambda *_a, **_k: ["scripts/widgetlib.py"])
    rc = sel.main(["--json", "--no-fetch", "--repo-root", str(repo)])
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert set(payload) == {  # --json key set byte-compatible (acceptance (c))
        "tests",
        "untested_touched",
        "base",
        "missing_invariants",
        "selection_reasons",
        "n_tests",
        "recommended_timeout_s",
        "slow_tests_selected",
    }
    assert "tests/test_uh_pack.py" in payload["tests"]
    assert payload["selection_reasons"]["tests/test_uh_pack.py"] == [
        "import-map:scripts/widgetlib.py"
    ]
    assert payload["untested_touched"] == []


# --- Case 51: LIVE-tree pin of the founding incident (#1286) ---------------------
def test_import_map_live_tree_issue1286_shape():
    """No fixtures: the real tree's test_issue810_uh_pack_validation.py is
    selected for a diff touching issue810_common + issue810_fit_readout, with
    BOTH import-map reasons (fit_readout is a function-level import — pins the
    ast.walk extension on the real tree), and neither file is WARNed untested.

    If a future cleanup removes scripts/issue810_*.py or the target test, this
    pin fails loud — repoint it at another committed import-relationship pair.
    """
    repo_root = Path(sel.__file__).resolve().parents[1]
    touched = ["scripts/issue810_common.py", "scripts/issue810_fit_readout.py"]
    tests, untested, reasons = sel.select_tests_with_reasons(touched, repo_root)
    target = "tests/test_issue810_uh_pack_validation.py"
    assert target in tests
    assert "import-map:scripts/issue810_common.py" in reasons[target]
    assert "import-map:scripts/issue810_fit_readout.py" in reasons[target]
    assert "scripts/issue810_common.py" not in untested
    assert "scripts/issue810_fit_readout.py" not in untested


# --- Rules-pin discovery arm (#1496) ---------------------------------------------
# Cases 52-62: a touched .claude/rules/<name>.md selects every tests/**/test_*.py
# whose raw text contains the basename <name>.md (reason rules-pin:<rule path>),
# ADDITIVE to the WORKFLOW_SURFACE skip (which is unchanged); --map-files unions
# the same pairs MINUS WORKFLOW_INVARIANT members.


# --- Case 52: full-path-literal reference form -----------------------------------
def test_rules_pin_full_path_literal_selected(tmp_path: Path):
    repo = _make_tree(tmp_path, [])
    (repo / "tests" / "test_some_rule_pin.py").write_text('RULE = ".claude/rules/some-rule.md"\n')
    touched = [".claude/rules/some-rule.md"]
    tests, untested, reasons = sel.select_tests_with_reasons(touched, repo)
    assert "tests/test_some_rule_pin.py" in tests
    assert reasons["tests/test_some_rule_pin.py"] == ["rules-pin:.claude/rules/some-rule.md"]
    assert untested == []  # rules files stay a correct SKIP, never "untested"


# --- Case 53: path-join reference form (no full-path literal) ---------------------
def test_rules_pin_path_join_basename_selected(tmp_path: Path):
    """The test_battery_basis_prose_pins.py shape: basename via path-join only."""
    repo = _make_tree(tmp_path, [])
    (repo / "tests" / "test_join_pin.py").write_text(
        'text = (ROOT / ".claude" / "rules" / "some-rule.md").read_text()\n'
    )
    tests, _, _ = sel.select_tests_with_reasons([".claude/rules/some-rule.md"], repo)
    assert "tests/test_join_pin.py" in tests


# --- Case 54: comment/docstring mentions count (documented over-select posture) ---
def test_rules_pin_comment_mention_counts(tmp_path: Path):
    repo = _make_tree(tmp_path, [])
    (repo / "tests" / "test_citer.py").write_text(
        "# see .claude/rules/some-rule.md for the recipe\ndef test_x():\n    assert True\n"
    )
    tests, _, reasons = sel.select_tests_with_reasons([".claude/rules/some-rule.md"], repo)
    assert "tests/test_citer.py" in tests
    assert reasons["tests/test_citer.py"] == ["rules-pin:.claude/rules/some-rule.md"]


# --- Case 55: SUPERSTRING basenames over-select by design --------------------------
def test_rules_pin_superstring_basename_over_selects(tmp_path: Path):
    """Substring matching means touching x.md also selects a test mentioning
    only prefix-x.md ("x.md" is a substring of "prefix-x.md"); the live-tree
    instance is critic-lens-reference.md vs
    clean-result-critic-lens-reference.md. Accepted over-select (safe
    direction) — pinned so extra hits are never misread as a scan bug."""
    repo = _make_tree(tmp_path, [])
    (repo / "tests" / "test_superstring_citer.py").write_text('DOC = ".claude/rules/prefix-x.md"\n')
    hits = sel.rules_pin_hits([".claude/rules/x.md"], repo)
    assert ".claude/rules/x.md" in hits.get("tests/test_superstring_citer.py", set())


# --- Case 56: no rules file touched -> zero hits, ZERO file reads ------------------
def test_rules_pin_no_rules_touched_no_hits(tmp_path: Path, capsys):
    """Proof of the zero-read early return: an undecodable test file is
    planted; any scan pass reads raw text (case 58 shows that read WARNs), so
    the absence of a WARN here proves no file was read."""
    repo = _make_tree(tmp_path, [])
    (repo / "tests" / "test_undecodable.py").write_bytes(b"\xff\xfe bad")
    assert sel.rules_pin_hits(["scripts/widget.py", "docs/x.md"], repo) == {}
    assert "rules-pin scan cannot read" not in capsys.readouterr().err


# --- Case 57: monotonicity — the arm only ever GROWS the selection -----------------
def test_rules_pin_selection_only_grows(tmp_path: Path):
    """Same touched set, tree WITH vs WITHOUT the pin test (mirror of case 41):
    WITH-selection is a superset and every WITHOUT reason list is preserved
    verbatim (acceptance R3)."""
    touched = ["scripts/widgetlib.py", ".claude/rules/some-rule.md"]
    repo_without = _make_tree(tmp_path / "without", ["test_widgetlib.py"])
    t_without, u_without, r_without = sel.select_tests_with_reasons(touched, repo_without)
    repo_with = _make_tree(tmp_path / "with", ["test_widgetlib.py"])
    (repo_with / "tests" / "test_rule_pin.py").write_text('R = ".claude/rules/some-rule.md"\n')
    t_with, u_with, r_with = sel.select_tests_with_reasons(touched, repo_with)
    assert set(t_with) >= set(t_without)
    for test, rs in r_without.items():
        assert r_with[test] == rs  # pre-existing reason lists preserved verbatim
    assert "tests/test_rule_pin.py" in t_with
    assert u_without == u_with == []


# --- Case 58: unreadable test file WARNs + is skipped; never crashes ---------------
def test_rules_pin_unreadable_test_file_warns_not_crash(tmp_path: Path, capsys):
    repo = _make_tree(tmp_path, [])
    (repo / "tests" / "test_good_pin.py").write_text('R = ".claude/rules/some-rule.md"\n')
    (repo / "tests" / "test_bad.py").write_bytes(b'R = ".claude/rules/some-rule.md"\n\xff\xfe')
    tests, untested, _ = sel.select_tests_with_reasons([".claude/rules/some-rule.md"], repo)
    assert "tests/test_good_pin.py" in tests  # the valid hit still selected
    assert "tests/test_bad.py" not in tests
    assert untested == []
    err = capsys.readouterr().err
    assert err.count("rules-pin scan cannot read") == 1
    assert "test_bad.py" in err
    # 1 failure over the ~42-file fixture tree is < 5%: no aggregate WARN.
    assert "systemic tests/ breakage" not in err


# --- Case 59: aggregate read-failure WARN (systemic-breakage signal) ---------------
def test_rules_pin_aggregate_read_failure_warn(tmp_path: Path, capsys):
    """>5% of scanned test files unreadable emits ONE extra summary WARN
    (mirrors the import-map arm's #1299 aggregate signal, case 48).

    Broken-stub count sized proportionally to the seeded tree — the same
    #1632 registry-growth knife-edge as case 48 (a fixed 3 broken files read
    3/60 = 5.00% at 60 seeded members, failing the strict ``> 5%`` gate).
    """
    repo = _make_tree(tmp_path, [])
    (repo / "tests" / "test_ok_pin.py").write_text('R = ".claude/rules/some-rule.md"\n')
    n_pre = len(list((repo / "tests").rglob("test_*.py")))
    n_broken = math.ceil(0.05 * n_pre) + 1
    assert n_broken / (n_pre + n_broken) > 0.05  # self-check: aggregate WARN must fire
    for i in range(n_broken):
        (repo / "tests" / f"test_bad_{i}.py").write_bytes(b"\xff\xfe bad")
    tests, _, _ = sel.select_tests_with_reasons([".claude/rules/some-rule.md"], repo)
    assert "tests/test_ok_pin.py" in tests
    err = capsys.readouterr().err
    assert err.count("rules-pin scan cannot read") == n_broken  # one per-file WARN each
    assert err.count("systemic tests/ breakage") == 1  # exactly one aggregate WARN


# --- Case 60: LIVE-tree drift/regression pin (the #1496 durability pin) ------------
def test_rules_pin_live_tree_known_pairs():
    """DRIFT/REGRESSION PIN: the discovery scan, run against the LIVE repo
    tree, finds these verified (rule -> pin test) pairs — the full-path-literal
    form (pod-side-reporting), the path-join form (critic-lens-reference; pins
    the basename-substring semantics on the real tree), and the llm-judging
    pair. SUPERSET assert: new pin tests joining later must not break this; a
    rename of a pinned test legitimately forces a deliberate 1-line update
    here (that loudness is the point)."""
    root = Path(sel.__file__).resolve().parents[1]
    hits = sel.rules_pin_hits(
        [
            ".claude/rules/pod-side-reporting.md",
            ".claude/rules/critic-lens-reference.md",  # path-join form on the live tree
            ".claude/rules/llm-judging.md",
        ],
        root,
    )
    assert ".claude/rules/pod-side-reporting.md" in hits.get(
        "tests/test_pod_side_reporting_push_contract.py", set()
    )
    assert ".claude/rules/critic-lens-reference.md" in hits.get(
        "tests/test_battery_basis_prose_pins.py", set()
    )
    assert ".claude/rules/llm-judging.md" in hits.get("tests/test_judge_dispatch.py", set())


# --- Case 61: --map-files unions rules-pin pairs; no-rules payload unchanged -------
def test_cli_map_files_rules_pin_pair(tmp_path: Path, capsys):
    repo = _make_tree(tmp_path, [])
    (repo / "tests" / "test_some_rule_pin.py").write_text('R = ".claude/rules/some-rule.md"\n')
    listing = tmp_path / "payload.txt"
    listing.write_text(".claude/rules/some-rule.md\n")
    rc = sel.main(["--map-files", str(listing), "--repo-root", str(repo)])
    assert rc == 0
    out = capsys.readouterr().out
    assert out.splitlines() == ["tests/test_some_rule_pin.py\t.claude/rules/some-rule.md"]
    # A payload with NO rules file is byte-identical to today's scan-map output
    # (the pin test's mention is irrelevant: pairs key on the PAYLOAD paths).
    listing.write_text("scripts/dispatch_x.py\n")
    rc = sel.main(["--map-files", str(listing), "--repo-root", str(repo)])
    assert rc == 0
    assert capsys.readouterr().out.splitlines() == [
        "tests/test_shared_vm_thread_caps.py\tscripts/dispatch_x.py",
        "tests/test_subprocess_env_explicit.py\tscripts/dispatch_x.py",
    ]


# --- Case 62: --map-files EXCLUDES invariant members; the 9c arm keeps them --------
def test_cli_map_files_rules_pin_excludes_invariant(tmp_path: Path, capsys):
    """The deliberate asymmetry (#1496 D3): tests/test_workflow_lint.py (a
    WORKFLOW_INVARIANT member and the only SLOW_TESTS entry) is filtered from
    the --map-files pairs, while select_tests_with_reasons still carries the
    rules-pin reason on it (the union dedupes; the extra reason is
    informative)."""
    repo = _make_tree(tmp_path, [])
    (repo / "tests" / "test_workflow_lint.py").write_text('R = ".claude/rules/some-rule.md"\n')
    listing = tmp_path / "payload.txt"
    listing.write_text(".claude/rules/some-rule.md\n")
    rc = sel.main(["--map-files", str(listing), "--repo-root", str(repo)])
    assert rc == 0
    assert capsys.readouterr().out == ""  # invariant member filtered -> the skip signal
    _, _, reasons = sel.select_tests_with_reasons([".claude/rules/some-rule.md"], repo)
    assert set(reasons["tests/test_workflow_lint.py"]) == {
        "invariant",
        "rules-pin:.claude/rules/some-rule.md",
    }


# --- Cases 52+ (#1498): the literal-path pinning arm -----------------------------
# Fixtures write test files whose RAW TEXT hardcodes a touched file's
# repo-relative path (the tests/test_ruff_policy.py LIVE_WORKFLOW_HELPERS
# shape). The shared _make_tree stubs contain no repo paths, so the arm
# no-ops on every pre-existing fixture by construction.


# --- Case 52: a pinning test is selected with the literal-path reason ------------
def test_literal_path_pin_selected(tmp_path: Path):
    repo = _make_import_tree(tmp_path, {"test_pin_x.py": 'LIVE = ["scripts/foo_helper.py"]\n'})
    tests, _, reasons = sel.select_tests_with_reasons(["scripts/foo_helper.py"], repo)
    assert "tests/test_pin_x.py" in tests
    # #1688: the basename inside the full-path literal is the documented
    # harmless duplicate (basename-ref rides along; union dedupes pairs).
    assert reasons["tests/test_pin_x.py"] == [
        "basename-ref:scripts/foo_helper.py",
        "literal-path:scripts/foo_helper.py",
    ]


# --- Case 53: THE durability pin — the #1498 founding case on the LIVE tree ------
def test_literal_path_founding_case_live_tree():
    """No fixtures: the real tree's tests/test_ruff_policy.py hardcodes
    ``scripts/daily_drive_filings.py`` in LIVE_WORKFLOW_HELPERS and lints it at
    test time, yet shares no stem, matches no scan glob, and imports nothing —
    reachable ONLY through the literal-path arm (#1498 founding incident).

    If a future cleanup removes scripts/daily_drive_filings.py or drops it from
    LIVE_WORKFLOW_HELPERS, this pin fails loud — repoint it at another
    committed pin relationship (test_ruff_policy.py has dozens).
    """
    repo_root = Path(sel.__file__).resolve().parents[1]
    tests, _, reasons = sel.select_tests_with_reasons(["scripts/daily_drive_filings.py"], repo_root)
    assert "tests/test_ruff_policy.py" in tests
    assert "literal-path:scripts/daily_drive_filings.py" in reasons["tests/test_ruff_policy.py"]


# --- Case 54: precision — a pin of a DIFFERENT path is not selected --------------
def test_literal_path_negative_not_selected(tmp_path: Path):
    repo = _make_import_tree(
        tmp_path, {"test_bystander_pin.py": 'LIVE = ["scripts/other_helper.py"]\n'}
    )
    tests, _, _ = sel.select_tests_with_reasons(["scripts/foo_helper.py"], repo)
    assert "tests/test_bystander_pin.py" not in tests


# --- Case 55: a literal hit does NOT suppress the untested_touched WARN ----------
def test_literal_path_does_not_suppress_untested_warn(tmp_path: Path):
    """A pinning test asserts an invariant ABOUT the file (e.g. ruff
    cleanliness), not the file's own logic — unlike an import hit, it never
    sets ``matched`` (plan R3; glob-scan precedent)."""
    repo = _make_import_tree(tmp_path, {"test_pin_x.py": 'LIVE = ["scripts/foo_helper.py"]\n'})
    tests, untested, _ = sel.select_tests_with_reasons(["scripts/foo_helper.py"], repo)
    assert "tests/test_pin_x.py" in tests
    assert untested == ["scripts/foo_helper.py"]


# --- Case 56: monotonicity — the arm only ever GROWS the selection ---------------
def test_literal_path_only_grows_selection(tmp_path: Path):
    """Same touched set, tree WITH vs WITHOUT the pin-bearing test file:
    WITH-selection is a superset and every WITHOUT reason list is preserved
    verbatim (mirror of case 41 for the literal arm)."""
    touched = ["scripts/widgetlib.py", "scripts/orphan.py"]
    repo_without = _make_tree(tmp_path / "without", ["test_widgetlib.py"])
    t_without, u_without, r_without = sel.select_tests_with_reasons(touched, repo_without)
    repo_with = _make_tree(tmp_path / "with", ["test_widgetlib.py"])
    (repo_with / "tests" / "test_pins.py").write_text('LIVE = ["scripts/widgetlib.py"]\n')
    t_with, u_with, r_with = sel.select_tests_with_reasons(touched, repo_with)
    assert set(t_with) >= set(t_without)
    for test, rs in r_without.items():
        assert r_with[test] == rs  # pre-existing reason lists preserved verbatim
    assert r_with["tests/test_pins.py"] == [
        "basename-ref:scripts/widgetlib.py",  # #1688 in-path duplicate
        "literal-path:scripts/widgetlib.py",
    ]
    # orphan.py has no test in either tree; widgetlib.py is stem-mapped in both.
    assert u_without == ["scripts/orphan.py"]
    assert u_with == ["scripts/orphan.py"]


# --- Case 57: unreadable file WARNs + is skipped for BOTH arms; never crashes ----
def test_literal_path_unreadable_file_warns_not_crash(tmp_path: Path, capsys):
    repo = _make_import_tree(tmp_path, {"test_good_pin.py": 'LIVE = ["scripts/foo_helper.py"]\n'})
    (repo / "tests" / "test_undecodable.py").write_bytes(
        b'LIVE = ["scripts/foo_helper.py"]\n\xff\xfe bad'
    )
    tests, _, reasons = sel.select_tests_with_reasons(["scripts/foo_helper.py"], repo)
    assert "tests/test_good_pin.py" in tests  # the valid hit still selected
    assert reasons["tests/test_good_pin.py"] == [
        "basename-ref:scripts/foo_helper.py",  # #1688 in-path duplicate
        "literal-path:scripts/foo_helper.py",
    ]
    assert "tests/test_undecodable.py" not in tests  # read failed -> both arms skip
    err = capsys.readouterr().err
    assert "import-map cannot parse" in err
    assert "test_undecodable.py" in err


# --- Case 58: a syntax-error file's literal hit still lands (read/parse split) ---
def test_literal_path_syntax_error_file_still_matched(tmp_path: Path, capsys):
    """The raw read succeeded, so the literal hit is recorded even though the
    ast parse (triggered by the 'foo_helper' pre-filter token) fails — the
    import-arm WARN fires and the literal selection lands."""
    repo = _make_import_tree(
        tmp_path, {"test_broken_pin.py": 'LIVE = ["scripts/foo_helper.py"]\ndef broken(:\n'}
    )
    tests, untested, reasons = sel.select_tests_with_reasons(["scripts/foo_helper.py"], repo)
    assert "tests/test_broken_pin.py" in tests
    assert reasons["tests/test_broken_pin.py"] == [
        "basename-ref:scripts/foo_helper.py",  # #1688: raw-text arm, parse not needed
        "literal-path:scripts/foo_helper.py",
    ]
    assert untested == ["scripts/foo_helper.py"]  # literal hit never sets matched
    err = capsys.readouterr().err
    assert err.count("import-map cannot parse") == 1
    assert "test_broken_pin.py" in err


# --- Case 59: workflow-surface-only diff -> still ZERO file reads ----------------
def test_literal_path_md_only_diff_zero_scan(tmp_path: Path, capsys):
    """Mirror of case 44 under the shared pass: no touched file is eligible for
    EITHER arm, so the scan never reads tests/ (the planted undecodable file
    would WARN on any read — case 57 is the positive control)."""
    repo = _make_import_tree(tmp_path, {})
    (repo / "tests" / "test_undecodable.py").write_bytes(b"bad \xff\xfe")
    assert sel.literal_path_targets([".claude/rules/foo.md", "notes.md"]) == set()
    tests, untested, _ = sel.select_tests_with_reasons([".claude/rules/foo.md", "notes.md"], repo)
    err = capsys.readouterr().err
    assert "import-map cannot parse" not in err  # zero reads: never touched the bad file
    assert set(tests) == set(sel.WORKFLOW_INVARIANT)
    assert untested == []


# --- Case 60: scripts/__init__.py is a NEW scan trigger, with zero parses --------
def test_literal_path_init_py_triggers_scan_zero_parses(tmp_path: Path, capsys):
    """``scripts/__init__.py`` maps to no module name (empty module map) but IS
    a literal target -> the shared pass runs. The empty token pre-filter means
    ZERO ast parses: the planted syntax-error file would WARN if parsed."""
    repo = _make_import_tree(
        tmp_path,
        {
            "test_pkg_pin.py": 'PINNED = "scripts/__init__.py"\n',
            "test_broken.py": "def broken(:\n",
        },
    )
    tests, _, reasons = sel.select_tests_with_reasons(["scripts/__init__.py"], repo)
    assert "tests/test_pkg_pin.py" in tests
    assert reasons["tests/test_pkg_pin.py"] == [
        "basename-ref:scripts/__init__.py",  # #1688 in-path duplicate
        "literal-path:scripts/__init__.py",
    ]
    assert "import-map cannot parse" not in capsys.readouterr().err  # zero parses


# --- Case 61: literal_path_targets eligibility rules ------------------------------
def test_literal_path_touched_test_not_a_target():
    assert sel.literal_path_targets(["tests/test_a.py"]) == set()
    assert sel.literal_path_targets(
        [
            "tests/test_a.py",  # tests/ — the touched-test arm owns it
            "other/module.py",  # outside the two eligible roots
            "scripts/notes.md",  # not .py
            ".claude/skills/issue/SKILL.md",  # workflow surface — #1496's mapping
            "scripts/x.py",
            "src/explore_persona_space/y.py",
            "scripts/__init__.py",  # eligible: a literal target even with no module name
        ]
    ) == {"scripts/x.py", "src/explore_persona_space/y.py", "scripts/__init__.py"}


# --- Case 62: end-to-end through main() --json — the reason token is carried -----
def test_cli_json_carries_literal_path_reason(tmp_path: Path, monkeypatch, capsys):
    repo = _make_import_tree(tmp_path, {"test_pin.py": 'LIVE = ["scripts/widgetlib.py"]\n'})
    monkeypatch.setattr(sel, "_resolve_work_root", lambda _arg: repo)
    monkeypatch.setattr(sel, "compute_touched", lambda *_a, **_k: ["scripts/widgetlib.py"])
    rc = sel.main(["--json", "--no-fetch", "--repo-root", str(repo)])
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert "tests/test_pin.py" in payload["tests"]
    assert payload["selection_reasons"]["tests/test_pin.py"] == [
        "basename-ref:scripts/widgetlib.py",  # #1688 in-path duplicate
        "literal-path:scripts/widgetlib.py",
    ]
    # The WARN still reaches the JSON consumer: literal hits never suppress it.
    assert payload["untested_touched"] == ["scripts/widgetlib.py"]


# --- Case 63 (#1573, REPLACES test_map_files_ignores_literal_path_hits): the
# --- mapping mode now INCLUDES literal-path pins — the deliberate contract flip
def test_map_files_includes_literal_path_hits(tmp_path: Path, capsys):
    """The Step 10d mapping mode carries the src/scripts dependency arms
    (#1573): a payload naming a literal-PINNED but not-scan-globbed file now
    maps to its pinning test (A2 — flips the #1498-era case-63 scoping pin;
    the old GLOB_SCAN-only contract is the founding bug of #1573)."""
    repo = _make_import_tree(
        tmp_path, {"test_pin.py": 'LIVE = ["src/explore_persona_space/widgetlib.py"]\n'}
    )
    listing = tmp_path / "payload.txt"
    listing.write_text("src/explore_persona_space/widgetlib.py\n")
    rc = sel.main(["--map-files", str(listing), "--repo-root", str(repo)])
    assert rc == 0
    out = capsys.readouterr().out
    assert out.splitlines() == ["tests/test_pin.py\tsrc/explore_persona_space/widgetlib.py"]


# --- Cases 64+ (#1573): the --map-files src/scripts dependency arms ---------------


# --- Case 64: THE durability pin — the founding sft.py edge on the LIVE tree ------
def test_cli_map_files_import_edge_sft_regression(tmp_path: Path, capsys):
    """No fixtures (the test_rules_pin_live_tree_known_pairs convention): the
    real tree's tests/test_artifacts_recipe.py statically imports
    ``train_lora`` from src/explore_persona_space/train/sft.py (and
    ``inspect.getsource``s it), yet ``--map-files`` on that payload emitted
    ZERO pairs pre-#1573 — the founding incident (the ``use_rslora`` change,
    commit d7908a3837, broke test_rslora_engine_pin with no gate firing).
    Also pins the WORKFLOW_INVARIANT exclusion on the live tree: sft.py
    literal-hits the 2400 s tests/test_workflow_lint.py (LIVE_WORKFLOW_HELPERS)
    which must NOT reach the 600 s-class TG legs (A1 + A4 live half)."""
    repo_root = Path(sel.__file__).resolve().parents[1]
    listing = tmp_path / "payload.txt"
    listing.write_text("src/explore_persona_space/train/sft.py\n")
    rc = sel.main(["--map-files", str(listing), "--repo-root", str(repo_root)])
    assert rc == 0
    captured = capsys.readouterr()
    lines = captured.out.splitlines()
    assert "tests/test_artifacts_recipe.py\tsrc/explore_persona_space/train/sft.py" in lines
    assert not any(line.startswith("tests/test_workflow_lint.py\t") for line in lines)
    assert "recommended-timeout-s=" in captured.err


# --- Case 65: the stem arm maps a DYNAMIC-import (importorskip) test (A3) ---------
def test_map_files_stem_pair_maps_dynamic_import_test(tmp_path: Path, capsys):
    """A stem-named test that loads the touched module DYNAMICALLY
    (``pytest.importorskip`` — invisible to the import arm's ast scan) is
    mapped via the stem arm; this is the mechanism that closes the
    dynamic-import getsource subclass (plan D4)."""
    repo = _make_import_tree(
        tmp_path,
        {"test_widgetlib_exactness.py": 'import pytest\nw = pytest.importorskip("widgetlib")\n'},
    )
    listing = tmp_path / "payload.txt"
    listing.write_text("scripts/widgetlib.py\n")
    rc = sel.main(["--map-files", str(listing), "--repo-root", str(repo)])
    assert rc == 0
    lines = capsys.readouterr().out.splitlines()
    assert "tests/test_widgetlib_exactness.py\tscripts/widgetlib.py" in lines


# --- Case 66: WORKFLOW_INVARIANT members excluded from the dependency arms (A4) ---
def test_map_files_excludes_invariant_hits(tmp_path: Path, capsys):
    """An invariant-named test importing the touched module is EXCLUDED from
    the --map-files pairs (mirror of the rules-pin asymmetry, case 62) while
    a non-invariant importer of the SAME module is included."""
    repo = _make_import_tree(
        tmp_path,
        {"test_consumer_x.py": "import widgetlib\n"},
    )
    # Overwrite an invariant stub with a REAL import of the touched module.
    (repo / "tests" / "test_verify_plan.py").write_text("import widgetlib\n")
    listing = tmp_path / "payload.txt"
    listing.write_text("scripts/widgetlib.py\n")
    rc = sel.main(["--map-files", str(listing), "--repo-root", str(repo)])
    assert rc == 0
    lines = capsys.readouterr().out.splitlines()
    assert "tests/test_consumer_x.py\tscripts/widgetlib.py" in lines
    assert not any(line.startswith("tests/test_verify_plan.py\t") for line in lines)


# --- Case 67: the zero-mapped fail-loud floor (A5) --------------------------------
def test_map_files_zero_mapped_code_file_warns_rc0(tmp_path: Path, capsys):
    """An eligible src/scripts .py payload with ZERO pairs across all arms
    draws exactly one tab-free stderr WARN; rc stays 0 and stdout stays empty
    (consumers treat helper rc!=0 as crash-class fail-closed)."""
    bare = tmp_path / "bare"  # no tests/ tree at all
    bare.mkdir()
    listing = tmp_path / "payload.txt"
    listing.write_text("src/explore_persona_space/widgetlib.py\n")
    rc = sel.main(["--map-files", str(listing), "--repo-root", str(bare)])
    assert rc == 0
    captured = capsys.readouterr()
    assert captured.out == ""
    assert (
        captured.err.count("no mapped tests for code file src/explore_persona_space/widgetlib.py")
        == 1
    )
    assert not any("\t" in line for line in captured.err.splitlines())


# --- Case 68: the machine-greppable sizing line (A6) -------------------------------
def test_map_files_sizing_line(tmp_path: Path, capsys):
    """A non-empty map prints ONE tab-free `recommended-timeout-s=<T>` stderr
    line: floor 300 for small maps, the #1046 formula above it."""
    repo = _make_import_tree(
        tmp_path,
        {
            "test_consumer_a.py": "from explore_persona_space.widgetlib import f\n",
            "test_consumer_b.py": "from explore_persona_space.widgetlib import f\n",
        },
    )
    listing = tmp_path / "payload.txt"
    listing.write_text("src/explore_persona_space/widgetlib.py\n")
    rc = sel.main(["--map-files", str(listing), "--repo-root", str(repo)])
    assert rc == 0
    captured = capsys.readouterr()
    # 2 tests: 120 + 2*30 = 180 -> floored at MAP_TIMEOUT_FLOOR_S = 600.
    assert "recommended-timeout-s=600" in captured.err
    assert not any("\t" in line for line in captured.err.splitlines())
    assert len(captured.out.splitlines()) == 2
    # 17 tests clear the floor: (120 + 17*30) * 2.0 dispersion = 1260 (#1697).
    repo17 = _make_import_tree(
        tmp_path / "seventeen",
        {
            f"test_consumer_{i}.py": "from explore_persona_space.widgetlib import f\n"
            for i in range(17)
        },
    )
    rc = sel.main(["--map-files", str(listing), "--repo-root", str(repo17)])
    assert rc == 0
    captured = capsys.readouterr()
    assert "recommended-timeout-s=1260" in captured.err
    assert not any("\t" in line for line in captured.err.splitlines())


# --- Case 69: recommended_timeout_s floor kwarg (D2) -------------------------------
def test_recommended_timeout_s_floor_kwarg():
    """Default floor unchanged (TIMEOUT_FLOOR_S = 900, diff-path callers
    byte-identical); an explicit floor= is honored; the formula wins above it."""
    assert sel.MAP_TIMEOUT_FLOOR_S == 600
    two = ["tests/test_a.py", "tests/test_b.py"]
    assert sel.recommended_timeout_s(two) == sel.TIMEOUT_FLOOR_S  # default floor binds
    assert sel.recommended_timeout_s(two, floor=sel.MAP_TIMEOUT_FLOOR_S) == 600
    forty = [f"tests/test_x{i}.py" for i in range(40)]
    expected = sel.TIMEOUT_BASE_S + 40 * sel.TIMEOUT_PER_FILE_S  # 1320 > both floors
    assert sel.recommended_timeout_s(forty, floor=sel.MAP_TIMEOUT_FLOOR_S) == expected
    assert sel.recommended_timeout_s(forty) == expected


# --- Case 69-bis (#1697): --map-files dispersion factor ---------------------------
def test_recommended_timeout_s_map_dispersion():
    """The --map-files path applies MAP_TIMEOUT_DISPERSION=2.0 to base+per_file,
    but NOT to the SLOW_TESTS surcharge; diff-path (default dispersion=1.0)
    stays byte-identical. Ties to #1697 (the #1675/#1682 undersized-bound trap
    at 780 s vs 728-752 s measured walls, ~1.04-1.07x)."""
    assert sel.MAP_TIMEOUT_DISPERSION == 2.0

    # The exact #1682/#1675 shape: 26 tests, no SLOW_TESTS members in the map.
    # NOTE: tests/test_workflow_lint_x{i}.py filenames are NOT in SLOW_TESTS —
    # only the exact `tests/test_workflow_lint.py` key matches (single-file
    # per-key surcharge, not a glob-family), so these 26 synthetic tests
    # contribute 0 surcharge and exercise the base+per_file*dispersion path
    # cleanly.
    twenty_six = [f"tests/test_workflow_lint_x{i}.py" for i in range(26)]
    # Base+per_file only: 120 + 30*26 = 900. Times 2.0 dispersion = 1800.
    assert (
        sel.recommended_timeout_s(
            twenty_six, floor=sel.MAP_TIMEOUT_FLOOR_S, dispersion=sel.MAP_TIMEOUT_DISPERSION
        )
        == 1800
    )
    # Diff-path default: dispersion=1.0, byte-identical to today's arithmetic.
    assert sel.recommended_timeout_s(twenty_six) == 900

    # SLOW_TESTS surcharge is NOT re-scaled by dispersion (already headroom'd).
    with_wl = [*twenty_six, "tests/test_workflow_lint.py"]
    # Dispersed base: (120 + 30*27) * 2 = 1860. Plus 2400 surcharge = 4260.
    expected_map = (
        round(sel.MAP_TIMEOUT_DISPERSION * (sel.TIMEOUT_BASE_S + 27 * sel.TIMEOUT_PER_FILE_S))
        + sel.SLOW_TESTS["tests/test_workflow_lint.py"]
    )
    assert (
        sel.recommended_timeout_s(
            with_wl, floor=sel.MAP_TIMEOUT_FLOOR_S, dispersion=sel.MAP_TIMEOUT_DISPERSION
        )
        == expected_map
    )
    assert expected_map == 4260  # explicit sanity — 1860 + 2400

    # Diff-path with SLOW_TESTS unchanged: 120 + 30*27 + 2400 = 4130.
    assert sel.recommended_timeout_s(with_wl) == (
        sel.TIMEOUT_BASE_S + 27 * sel.TIMEOUT_PER_FILE_S + 2400
    )


# --- Case 70: dependency_map_pairs unit — import + literal + stem union, sorted ----
def test_dependency_map_pairs_import_and_literal_union(tmp_path: Path):
    """One function-level read: import hits + literal hits + stem hits union
    into sorted unique (test, matched_path) pairs; a test hit by BOTH the
    import and literal arms appears once."""
    repo = _make_import_tree(
        tmp_path,
        {
            "test_widgetlib.py": "# stub\n",  # stem arm (exact) — no import needed
            "test_imports_w.py": "import widgetlib\n",  # import arm
            # BOTH arms hit this one -> dedupes to one pair:
            "test_both_w.py": 'import widgetlib\nLIVE = ["scripts/widgetlib.py"]\n',
        },
    )
    pairs = sel.dependency_map_pairs(["scripts/widgetlib.py"], repo)
    assert pairs == sorted(pairs)
    assert pairs == [
        ("tests/test_both_w.py", "scripts/widgetlib.py"),
        ("tests/test_imports_w.py", "scripts/widgetlib.py"),
        ("tests/test_widgetlib.py", "scripts/widgetlib.py"),
    ]


# --- Cases 71+ (#1579): the .sh stem/literal arms ---------------------------------
# A .sh-only diff previously selected ZERO stem-matched tests, returned zero
# --map-files pairs, and skipped the untested_touched WARN entirely (all three
# code-file arms were .py-gated). These cases pin the .sh extension of the stem
# arm + the literal_path_targets eligibility; the .py branch stays byte-identical
# (cases 1-70 unmodified are the pin).


# --- Case 71: scripts/X.sh -> test_X.py + test_*X*.py (mirror of case 1) ----------
def test_sh_stem_map_exact_and_glob(tmp_path: Path):
    """A touched .sh maps via the exact stem test AND the broad *stem* glob."""
    repo = _make_tree(tmp_path, ["test_widget.py", "test_widget_cli.py", "test_other.py"])
    tests, untested, reasons = sel.select_tests_with_reasons(["scripts/widget.sh"], repo)
    assert "tests/test_widget.py" in tests  # exact
    assert "tests/test_widget_cli.py" in tests  # broad *widget* glob
    assert "tests/test_other.py" not in tests
    assert untested == []
    assert "stem-map:scripts/widget.sh" in reasons["tests/test_widget.py"]


# --- Case 72: founding live-tree case — guard_repo_root_branch.sh (#1579) ---------
def test_sh_stem_map_live_tree_founding_case():
    """#1579 founding incident: a .sh-only diff selected zero stem-matched tests."""
    repo_root = _HELPER_PATH.parents[1]
    tests, untested, reasons = sel.select_tests_with_reasons(
        ["scripts/guard_repo_root_branch.sh"], repo_root
    )
    assert "tests/test_guard_repo_root_branch.py" in tests
    assert (
        "stem-map:scripts/guard_repo_root_branch.sh"
        in reasons["tests/test_guard_repo_root_branch.py"]
    )
    assert untested == []


# --- Case 73: .claude/hooks/*.sh reaches the stem arm (live tree) -----------------
def test_sh_hooks_stem_map_live_tree():
    """Hook scripts match no WORKFLOW_SURFACE glob, so the stem arm maps them."""
    repo_root = _HELPER_PATH.parents[1]
    tests, untested, _reasons = sel.select_tests_with_reasons(
        [".claude/hooks/guard_lessons_edit.sh"], repo_root
    )
    assert "tests/test_guard_lessons_edit.py" in tests
    assert untested == []


# --- Case 74: an unmatched .sh lands in untested_touched (no longer silent) -------
def test_sh_untested_code_file_warns(tmp_path: Path):
    """A .sh with no stem-matched test WARNs, symmetric with .py (#1579 gap 3)."""
    repo = _make_tree(tmp_path, ["test_other.py"])
    _tests, untested, _ = sel.select_tests_with_reasons(["scripts/lonely_helper.sh"], repo)
    assert untested == ["scripts/lonely_helper.sh"]


# --- Case 75: literal_path_targets .sh eligibility (unit) -------------------------
def test_sh_literal_path_targets_eligibility():
    """The .sh branch admits scripts/ + hooks/ + src/; the .py branch is unchanged."""
    got = sel.literal_path_targets(
        [
            "scripts/a.sh",
            ".claude/hooks/b.sh",
            "src/explore_persona_space/c.sh",
            "tests/d.sh",
            "external/e.sh",
            "docs/g.sh",
            ".claude/agents/h.sh",
            "scripts/f.py",  # the .py branch: byte-identical eligibility
        ]
    )
    assert got == {
        "scripts/a.sh",
        ".claude/hooks/b.sh",
        "src/explore_persona_space/c.sh",
        "scripts/f.py",
    }


# --- Case 76: a pinning test hardcoding a .sh path selects via literal-path -------
def test_sh_literal_path_pin_selected(tmp_path: Path):
    """A raw-substring .sh path pin is discovered; the WARN is NOT suppressed."""
    repo = _make_import_tree(tmp_path, {"test_pin_sh.py": 'P = "scripts/guardfoo.sh"\n'})
    _tests, untested, reasons = sel.select_tests_with_reasons(["scripts/guardfoo.sh"], repo)
    assert "literal-path:scripts/guardfoo.sh" in reasons["tests/test_pin_sh.py"]
    assert untested == ["scripts/guardfoo.sh"]  # a literal hit never suppresses the WARN


# --- Case 77: dependency_map_pairs carries the .sh stem pair (live tree) ----------
def test_dependency_map_pairs_sh_live_tree():
    """--map-files' dependency arms emit .sh stem pairs (exact AND broad glob)."""
    repo_root = _HELPER_PATH.parents[1]
    pairs = sel.dependency_map_pairs(["scripts/guard_repo_root_branch.sh"], repo_root)
    assert (
        "tests/test_guard_repo_root_branch.py",
        "scripts/guard_repo_root_branch.sh",
    ) in pairs
    # Broad-glob map pair for .sh (the bootstrap_pod class — no exact
    # tests/test_bootstrap_pod.py exists; reachable by NO other arm):
    bp = sel.dependency_map_pairs(["scripts/bootstrap_pod.sh"], repo_root)
    assert ("tests/test_bootstrap_pod_path.py", "scripts/bootstrap_pod.sh") in bp


# --- Case 78: CLI --map-files on a .sh payload prints the stem pair ---------------
def test_cli_map_files_sh_stem_pair(tmp_path: Path, capsys):
    """A .sh payload with a stem-matched test prints a real pytest pair (was empty)."""
    repo = _make_tree(tmp_path, ["test_guardfoo.py"])
    payload = tmp_path / "payload.txt"
    payload.write_text("scripts/guardfoo.sh\n")
    rc = sel.main(["--map-files", str(payload), "--repo-root", str(repo)])
    out = capsys.readouterr().out
    assert rc == 0
    assert "tests/test_guardfoo.py\tscripts/guardfoo.sh" in out


# --- Case 79: .py behavior byte-identical — the .sh arm only grows ----------------
def test_sh_arm_only_grows_selection(tmp_path: Path):
    """Adding a .sh to a diff never drops a .py-derived test or reason."""
    repo = _make_tree(tmp_path, ["test_widget.py", "test_widget_cli.py"])
    t_before, u_before, r_before = sel.select_tests_with_reasons(["scripts/widget.py"], repo)
    t_after, u_after, r_after = sel.select_tests_with_reasons(
        ["scripts/widget.py", "scripts/unrelated.sh"], repo
    )
    assert set(t_before) <= set(t_after)
    assert u_before == []
    for t, rs in r_before.items():
        assert set(rs) <= set(r_after[t])  # every .py-derived reason preserved verbatim
    assert u_after == ["scripts/unrelated.sh"]


# --- Case 80: unmapped eligible .sh payload draws the #1573 WARN floor (CLI) ------
def test_cli_map_files_sh_unmapped_warns(tmp_path: Path, capsys):
    """An eligible .sh payload with zero pairs draws the fail-loud stderr WARN."""
    repo = _make_tree(tmp_path, ["test_other.py"])
    payload = tmp_path / "payload.txt"
    payload.write_text("scripts/lonely_helper.sh\n")
    rc = sel.main(["--map-files", str(payload), "--repo-root", str(repo)])
    captured = capsys.readouterr()
    assert rc == 0
    assert captured.out.strip() == ""  # no pairs
    assert "no mapped tests for code file" in captured.err  # the fail-loud floor
    assert "scripts/lonely_helper.sh" in captured.err


# --- Cases 81+ (#1589): the transitive-consumer pin map ---------------------------
_SELECTOR_KEY = "scripts/select_step9c_tests.py"


# --- Case 81: THE #1589 durability pin — both registered consumers on the LIVE tree
def test_transitive_consumer_map_live_tree():
    """THE #1589 durability pin: a selector-module payload maps to BOTH
    registered transitive consumers on the live tree (neither is reachable by
    any text-scan arm — dynamic loads by constructed path / path-join
    literals)."""
    repo_root = _HELPER_PATH.parents[1]
    pairs = sel.transitive_consumer_pairs([_SELECTOR_KEY], repo_root)
    assert ("tests/test_step9c_baseline.py", _SELECTOR_KEY) in pairs
    assert ("tests/test_inline_lint_gate.py", _SELECTOR_KEY) in pairs


# --- Case 82: drift pin — every map entry exists on disk and is NOT invariant -----
def test_transitive_consumer_entries_exist_on_live_tree():
    """Drift pin: every key + registered test exists; no entry is invariant
    (an invariant member would be silently excluded from the map legs)."""
    repo_root = _HELPER_PATH.parents[1]
    for key, consumer_tests in sel.TRANSITIVE_CONSUMER_TESTS.items():
        assert (repo_root / key).exists(), key
        for t in consumer_tests:
            assert (repo_root / t).exists(), t
            assert t not in sel.WORKFLOW_INVARIANT, (
                f"{t}: invariant members are excluded from --map-files by design; "
                "registering one here is dead weight — remove one of the two."
            )


# --- Case 83: Step 9c diff path — the transitive-consumer reason; additive only ---
def test_transitive_consumer_diff_path_reason(tmp_path: Path):
    """Step 9c diff path: touching a map key selects the consumer with the
    transitive-consumer reason; the arm is additive (never sets ``matched``,
    so the untested WARN for the key still fires in this stem-less fixture)."""
    repo = _make_tree(tmp_path, ["test_step9c_baseline.py"])
    # _make_tree materializes every GLOB_SCAN_TESTS key, including the #1593
    # manifest-scan key tests/test_select_step9c_tests.py — unlink it so the
    # fixture is genuinely stem-less for the selector payload and the
    # additive-only pin below is observable via untested_touched.
    (repo / "tests" / "test_select_step9c_tests.py").unlink()
    tests_, untested, reasons = sel.select_tests_with_reasons([_SELECTOR_KEY], repo)
    assert "tests/test_step9c_baseline.py" in tests_
    assert f"transitive-consumer:{_SELECTOR_KEY}" in reasons["tests/test_step9c_baseline.py"]
    assert untested == [_SELECTOR_KEY]  # additive-only pinned: matched is never set


# --- Case 84: monotonicity — the arm only ever GROWS the selection ----------------
def test_transitive_consumer_only_grows_selection(tmp_path: Path):
    """Parity with the other arms: adding a map-key file to a diff never drops
    a previously selected test or reason."""
    repo = _make_tree(tmp_path, ["test_widget.py", "test_step9c_baseline.py"])
    t_before, u_before, r_before = sel.select_tests_with_reasons(["scripts/widget.py"], repo)
    t_after, _u_after, r_after = sel.select_tests_with_reasons(
        ["scripts/widget.py", _SELECTOR_KEY], repo
    )
    assert set(t_before) <= set(t_after)
    assert u_before == []
    for t, rs in r_before.items():
        assert set(rs) <= set(r_after[t])  # every prior reason preserved verbatim
    assert "tests/test_step9c_baseline.py" in t_after
    assert f"transitive-consumer:{_SELECTOR_KEY}" in r_after["tests/test_step9c_baseline.py"]


# --- Case 85: a registered consumer absent from the work root is dropped ----------
def test_transitive_consumer_missing_on_disk_dropped(tmp_path: Path):
    """A registered consumer absent from the work root is dropped from pairs
    (fixture tree without tests/test_step9c_baseline.py -> no pair for it),
    while the present sibling registration survives."""
    repo = _make_tree(tmp_path, ["test_inline_lint_gate.py"])
    pairs = sel.transitive_consumer_pairs([_SELECTOR_KEY], repo)
    assert pairs == [("tests/test_inline_lint_gate.py", _SELECTOR_KEY)]


# --- Case 86: CLI --map-files end-to-end on the LIVE tree — the 7 pairs verbatim --
def test_cli_map_files_transitive_pairs_live_tree(tmp_path: Path, capsys):
    """CLI end-to-end on the LIVE tree: the selector payload prints all 5
    dependency-arm pairs PLUS the 2 transitive pairs (7 pairs, 7 tests) and
    the sizing line clears the 600 s MAP_TIMEOUT_FLOOR_S at 660
    ((120 + 7*30) * 2.0). Exact-set assert — a new arm/pin
    joining later legitimately forces a deliberate 1-line update here (that
    loudness is the point; cf. the case-60 drift-pin posture)."""
    repo_root = _HELPER_PATH.parents[1]
    payload = tmp_path / "payload.txt"
    payload.write_text(f"{_SELECTOR_KEY}\n")
    rc = sel.main(["--map-files", str(payload), "--repo-root", str(repo_root)])
    captured = capsys.readouterr()
    assert rc == 0
    assert captured.out.splitlines() == [
        f"tests/test_inline_lint_gate.py\t{_SELECTOR_KEY}",
        f"tests/test_inline_payload_lint_gate_contract.py\t{_SELECTOR_KEY}",
        f"tests/test_issue_skill_lint_family_sync.py\t{_SELECTOR_KEY}",
        f"tests/test_ruff_policy.py\t{_SELECTOR_KEY}",
        f"tests/test_select_step9c_tests.py\t{_SELECTOR_KEY}",
        f"tests/test_shared_vm_thread_caps.py\t{_SELECTOR_KEY}",
        f"tests/test_step9c_baseline.py\t{_SELECTOR_KEY}",
    ]
    assert "map-files — 7 pairs, 7 tests; recommended-timeout-s=660" in captured.err


# --- Case 87: map-leg asymmetry — an invariant registration is excluded -----------
def test_transitive_consumer_excludes_invariant(tmp_path: Path, monkeypatch):
    """Map-leg asymmetry: a (monkeypatched) entry naming an invariant member
    is excluded from transitive_consumer_pairs while the non-invariant sibling
    survives — mirror of test_map_files_excludes_invariant_hits. The Step 9c
    diff arm KEEPS the invariant member (harmless extra reason; the union
    dedupes — the rules-pin asymmetry)."""
    repo = _make_tree(tmp_path, ["test_free_consumer.py"])
    inv_member = sel.WORKFLOW_INVARIANT[0]
    monkeypatch.setattr(
        sel,
        "TRANSITIVE_CONSUMER_TESTS",
        {_SELECTOR_KEY: (inv_member, "tests/test_free_consumer.py")},
    )
    pairs = sel.transitive_consumer_pairs([_SELECTOR_KEY], repo)
    assert pairs == [("tests/test_free_consumer.py", _SELECTOR_KEY)]
    # The 9c diff arm keeps the invariant member's transitive-consumer reason.
    _, _, reasons = sel.select_tests_with_reasons([_SELECTOR_KEY], repo)
    assert f"transitive-consumer:{_SELECTOR_KEY}" in reasons[inv_member]
    assert "invariant" in reasons[inv_member]


# --- Cases 88+ (#1688): dotted-ref / basename-ref / transitive-import arms --------
# Fixture module names are non-generic (widgetmod / midmod / farmod / guardthing)
# so no stem-glob or GLOB_SCAN_TESTS interference muddies the per-arm asserts.


# --- Case 88: dotted-module string reference selected (the #1683 escape #1) -------
def test_dotted_ref_string_selected(tmp_path: Path):
    """A monkeypatch-string-target shape — the test's ONLY link to the touched
    module is the dotted string ("scripts.widgetmod", ...) — is selected with
    the dotted-ref reason; the arm never sets ``matched`` (string-shape
    evidence, the literal-path precedent), so the WARN still fires."""
    repo = _make_import_tree(
        tmp_path, {"test_hooky.py": 'TARGET = ("scripts.widgetmod", "attr")\n'}
    )
    tests, untested, reasons = sel.select_tests_with_reasons(["scripts/widgetmod.py"], repo)
    assert "tests/test_hooky.py" in tests
    assert reasons["tests/test_hooky.py"] == ["dotted-ref:scripts/widgetmod.py"]
    assert untested == ["scripts/widgetmod.py"]


# --- Case 89: dotted boundary negatives — superstring + attribute-prefix ----------
def test_dotted_ref_superstring_not_matched(tmp_path: Path):
    """Right boundary: ``scripts.widgetmod`` must NOT fire on
    ``scripts.widgetmod_extra``; left boundary: nor on ``a.scripts.widgetmod``."""
    repo = _make_import_tree(
        tmp_path,
        {
            "test_right_super.py": 'X = "scripts.widgetmod_extra"\n',
            "test_left_prefixed.py": 'Y = "a.scripts.widgetmod"\n',
        },
    )
    tests, _, reasons = sel.select_tests_with_reasons(["scripts/widgetmod.py"], repo)
    assert "tests/test_right_super.py" not in tests
    assert "tests/test_left_prefixed.py" not in tests
    assert not any("dotted-ref:" in r for rs in reasons.values() for r in rs)


# --- Case 90: bare-basename reference selected (the #1683 escape #2) ---------------
def test_basename_ref_selected(tmp_path: Path):
    """The dispatcher-log-assert shape — bare ``widgetmod.py`` with no full
    path, no import — is selected with the basename-ref reason; never sets
    ``matched``."""
    repo = _make_import_tree(tmp_path, {"test_dispatchy.py": 'assert "widgetmod.py" in log_text\n'})
    tests, untested, reasons = sel.select_tests_with_reasons(["scripts/widgetmod.py"], repo)
    assert "tests/test_dispatchy.py" in tests
    assert reasons["tests/test_dispatchy.py"] == ["basename-ref:scripts/widgetmod.py"]
    assert untested == ["scripts/widgetmod.py"]


# --- Case 91: basename boundary negatives — identifier prefix + suffix -------------
def test_basename_ref_identifier_prefix_not_matched(tmp_path: Path):
    """Left boundary: ``widgetmod.py`` must NOT fire on ``codex_widgetmod.py``
    (63/1596 eligible basenames are substrings of another, measured); right
    boundary: nor on ``widgetmod.pyx`` / ``widgetmod.python``."""
    repo = _make_import_tree(
        tmp_path,
        {
            "test_left_ident.py": 'LOG = "codex_widgetmod.py"\n',
            "test_right_ident.py": 'EXT = "widgetmod.pyx and widgetmod.python"\n',
        },
    )
    tests, _, reasons = sel.select_tests_with_reasons(["scripts/widgetmod.py"], repo)
    assert "tests/test_left_ident.py" not in tests
    assert "tests/test_right_ident.py" not in tests
    assert not any("basename-ref:" in r for rs in reasons.values() for r in rs)


# --- Case 92: .sh basename eligibility (the #1579 symmetry) -------------------------
def test_basename_ref_sh_selected(tmp_path: Path):
    """A ``.sh`` payload under the eligibility prefixes basename-selects a test
    mentioning the bare script name (zero AST parses — .sh maps to no module)."""
    repo = _make_import_tree(tmp_path, {"test_guardish.py": 'line = "guardthing.sh ran"\n'})
    tests, untested, reasons = sel.select_tests_with_reasons(["scripts/guardthing.sh"], repo)
    assert "tests/test_guardish.py" in tests
    assert reasons["tests/test_guardish.py"] == ["basename-ref:scripts/guardthing.sh"]
    assert untested == ["scripts/guardthing.sh"]


# --- Case 93: one-hop transitive import selected (the #1683 escape #3) --------------
def test_transitive_import_one_hop_selected(tmp_path: Path):
    """scripts/midmod.py imports the touched module; a test importing midmod
    EXECUTES the touched module at import time -> selected with the
    transitive-import reason; never sets ``matched``."""
    repo = _make_import_tree(tmp_path, {"test_uses_mid.py": "import midmod\n"})
    (repo / "scripts").mkdir(exist_ok=True)
    (repo / "scripts" / "midmod.py").write_text("import widgetmod\n")
    tests, untested, reasons = sel.select_tests_with_reasons(["scripts/widgetmod.py"], repo)
    assert "tests/test_uses_mid.py" in tests
    assert reasons["tests/test_uses_mid.py"] == ["transitive-import:scripts/widgetmod.py"]
    assert untested == ["scripts/widgetmod.py"]


# --- Case 94: the one-hop bound — chains of >= 2 hops are NOT followed --------------
def test_transitive_import_two_hops_not_followed(tmp_path: Path):
    """farmod imports midmod imports the touched module; a test importing only
    farmod is NOT selected (one hop by construction, never recursive)."""
    repo = _make_import_tree(tmp_path, {"test_uses_far.py": "import farmod\n"})
    (repo / "scripts").mkdir(exist_ok=True)
    (repo / "scripts" / "midmod.py").write_text("import widgetmod\n")
    (repo / "scripts" / "farmod.py").write_text("import midmod\n")
    tests, _, reasons = sel.select_tests_with_reasons(["scripts/widgetmod.py"], repo)
    assert "tests/test_uses_far.py" not in tests
    assert not any(
        r == "transitive-import:scripts/widgetmod.py"
        for t, rs in reasons.items()
        if t == "tests/test_uses_far.py"
        for r in rs
    )


# --- Case 95: scripts-scoped on the TOUCHED end — src payloads never expand ---------
def test_transitive_import_src_touched_not_expanded(tmp_path: Path):
    """A touched src/ module with a scripts/ importer gets NO transitive
    expansion (scripts-scoped on BOTH ends; the src-rooted variant measured
    +70 test files and was rejected at plan time)."""
    repo = _make_import_tree(tmp_path, {"test_uses_srcconsumer.py": "import srcconsumer\n"})
    (repo / "scripts").mkdir(exist_ok=True)
    (repo / "scripts" / "srcconsumer.py").write_text(
        "from explore_persona_space.widgetsrc import f\n"
    )
    touched = ["src/explore_persona_space/widgetsrc.py"]
    assert sel.transitive_import_map(touched, repo) == {}
    tests, _, _ = sel.select_tests_with_reasons(touched, repo)
    assert "tests/test_uses_srcconsumer.py" not in tests


# --- Case 96: monotonicity — the #1688 arms only ever GROW the selection ------------
def test_new_arms_only_grow_selection(tmp_path: Path):
    """Same touched set, tree WITHOUT vs WITH all three arm triggers: the WITH
    selection is a superset and every WITHOUT reason list is preserved
    verbatim (the case-41 shape, per arm)."""
    touched = ["scripts/widgetmod.py"]
    repo_without = _make_tree(tmp_path / "without", ["test_widgetmod.py"])
    t_without, _, r_without = sel.select_tests_with_reasons(touched, repo_without)
    repo_with = _make_tree(tmp_path / "with", ["test_widgetmod.py"])
    (repo_with / "tests" / "test_dotted_w.py").write_text('T = ("scripts.widgetmod", "x")\n')
    (repo_with / "tests" / "test_base_w.py").write_text('assert "widgetmod.py" in out\n')
    (repo_with / "tests" / "test_trans_w.py").write_text("import midmod\n")
    (repo_with / "scripts").mkdir(exist_ok=True)
    (repo_with / "scripts" / "midmod.py").write_text("import widgetmod\n")
    t_with, _, r_with = sel.select_tests_with_reasons(touched, repo_with)
    assert set(t_with) >= set(t_without)
    for test, rs in r_without.items():
        assert r_with[test] == rs  # pre-existing reason lists preserved verbatim
    assert "tests/test_dotted_w.py" in t_with
    assert "tests/test_base_w.py" in t_with
    assert "tests/test_trans_w.py" in t_with


# --- Case 97: none of the #1688 arms ever marks the touched file tested -------------
def test_new_arms_never_mark_tested(tmp_path: Path):
    """All three arms firing at once still leave the touched file in the
    untested_touched WARN list (only the import arm sets ``matched`` —
    over-WARN is the safe direction)."""
    repo = _make_import_tree(
        tmp_path,
        {
            "test_dotted_n.py": 'T = ("scripts.widgetmod", "x")\n',
            "test_base_n.py": 'assert "widgetmod.py" in out\n',
            "test_trans_n.py": "import midmod\n",
        },
    )
    (repo / "scripts").mkdir(exist_ok=True)
    (repo / "scripts" / "midmod.py").write_text("import widgetmod\n")
    tests, untested, _ = sel.select_tests_with_reasons(["scripts/widgetmod.py"], repo)
    assert {"tests/test_dotted_n.py", "tests/test_base_n.py", "tests/test_trans_n.py"} <= set(tests)
    assert untested == ["scripts/widgetmod.py"]


# --- Case 98: CLI --map-files emits the new-arm pairs -------------------------------
def test_cli_map_files_new_arm_pairs(tmp_path: Path, capsys):
    """--map-files carries one pair per new-arm hit (union-deduped with the
    other dependency arms; output shape unchanged — pairs + sizing line)."""
    repo = _make_import_tree(
        tmp_path,
        {
            "test_dotted_c.py": 'T = ("scripts.widgetmod", "x")\n',
            "test_base_c.py": 'assert "widgetmod.py" in out\n',
            "test_trans_c.py": "import midmod\n",
        },
    )
    (repo / "scripts").mkdir(exist_ok=True)
    (repo / "scripts" / "midmod.py").write_text("import widgetmod\n")
    listing = tmp_path / "payload.txt"
    listing.write_text("scripts/widgetmod.py\n")
    rc = sel.main(["--map-files", str(listing), "--repo-root", str(repo)])
    assert rc == 0
    lines = capsys.readouterr().out.splitlines()
    assert "tests/test_dotted_c.py\tscripts/widgetmod.py" in lines
    assert "tests/test_base_c.py\tscripts/widgetmod.py" in lines
    assert "tests/test_trans_c.py\tscripts/widgetmod.py" in lines


# --- Case 99: map-leg asymmetry — invariant members excluded from new-arm pairs -----
def test_new_arm_map_pairs_exclude_invariant(tmp_path: Path):
    """An invariant-named test carrying a dotted-ref hit is EXCLUDED from
    dependency_map_pairs while the non-invariant sibling is included; the
    Step 9c selection arm KEEPS the invariant member with the new reason
    (the standing rules-pin asymmetry)."""
    repo = _make_import_tree(tmp_path, {"test_free_d.py": 'T = ("scripts.widgetmod", "x")\n'})
    inv_member = "tests/test_verify_plan.py"
    assert inv_member in sel.WORKFLOW_INVARIANT  # fixture premise
    (repo / inv_member).write_text('T = ("scripts.widgetmod", "x")\n')
    pairs = sel.dependency_map_pairs(["scripts/widgetmod.py"], repo)
    assert ("tests/test_free_d.py", "scripts/widgetmod.py") in pairs
    assert not any(t == inv_member for t, _f in pairs)
    _, _, reasons = sel.select_tests_with_reasons(["scripts/widgetmod.py"], repo)
    assert "dotted-ref:scripts/widgetmod.py" in reasons[inv_member]
    assert "invariant" in reasons[inv_member]


# --- Case 100: THE #1688 durability pin — all 3 escapees on the LIVE tree -----------
def test_issue1688_live_tree_escapee_shape():
    """THE #1688 durability pin: the three #1683-review escapees are selected
    for a scripts/issue667_extract.py payload on the real tree, each via its
    closing arm, and the pre-existing selections are retained.

    If a future cleanup removes scripts/issue667_extract.py or an escapee
    test's referencing form, this pin fails loud — repoint it at another
    committed reference of the same shape (one dotted string target, one
    bare-basename log assert, one scripts->scripts one-hop import chain).
    """
    repo_root = _HELPER_PATH.parents[1]
    probe = "scripts/issue667_extract.py"
    tests, untested, reasons = sel.select_tests_with_reasons([probe], repo_root)
    # Escapee #1 — dotted-module string refs (test file monkeypatches by name).
    assert f"dotted-ref:{probe}" in reasons["tests/test_issue671_extraction_hooks.py"]
    # Escapee #2 — bare-basename literals in dispatcher asserts.
    assert f"basename-ref:{probe}" in reasons["tests/test_issue811_dispatch.py"]
    # Escapee #3 — imports issue833_extract_onpolicy, which imports the probe.
    assert f"transitive-import:{probe}" in reasons["tests/test_issue833_nonemit_filters.py"]
    # Pre-existing selections retained (the count-robust hard gate).
    assert "tests/test_issue667_dispatcher.py" in tests
    assert "tests/test_issue811_maxp.py" in tests
    assert probe not in untested  # stem/import arms already matched it pre-#1688


# --- #1717 defect (a): --map-files + --json fails loud (argparse exit 2) ------
@pytest.mark.parametrize(
    "extra",
    [
        [],  # ordinary two-flag combo
        # A third co-passed flag that IS in the parser (--no-fetch) does not
        # change the parser.error verdict — the (--map-files, --json)
        # combination itself is what fires, regardless of surrounding argv.
        ["--no-fetch"],
    ],
)
def test_cli_map_files_json_flag_rejected(tmp_path: Path, capsys, extra: list[str]):
    """The (a) fix: `--map-files` combined with `--json` is a CLI usage error.
    argparse's `parser.error()` exits 2 with a stderr `: error: ...` line and
    empty stdout; no work-root resolution, no map-files load. Both flag
    orderings are checked (argparse is order-agnostic post-parse). A third
    valid flag (`--no-fetch`) co-passed does NOT change the verdict — the
    combination itself is what fires, regardless of surrounding argv.
    """
    repo = _make_tree(tmp_path, [])
    listing = tmp_path / "payload.txt"
    listing.write_text("scripts/issue123_foo.py\n")
    # Ordering A: --map-files first, --json after.
    with pytest.raises(SystemExit) as excinfo:
        sel.main(["--map-files", str(listing), "--json", "--repo-root", str(repo), *extra])
    assert excinfo.value.code == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "--json is not supported with --map-files" in captured.err
    # Ordering B: --json first, --map-files after.
    with pytest.raises(SystemExit) as excinfo:
        sel.main(["--json", "--map-files", str(listing), "--repo-root", str(repo), *extra])
    assert excinfo.value.code == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "--json is not supported with --map-files" in captured.err


# --- #1717 defect (c): --map-files comma-detected hint (opt-in on comma) ------
def test_cli_map_files_comma_hint(tmp_path: Path, capsys):
    """The (c) fix: a comma in the --map-files argument (the session
    `c0a2df1b` shape — `--map-files a.md,b.md`) APPENDS an opt-in hint
    after the standard Errno-2 line, telling the caller to pass a
    newline-separated file path. A non-comma missing path preserves the
    base OSError text WITHOUT the hint (hint is opt-in on comma
    detection — a real comma-in-path failure still surfaces Errno + path
    verbatim).
    """
    repo = _make_tree(tmp_path, [])
    # Positive branch: a comma in the argument triggers the hint.
    rc = sel.main(["--map-files", "a.md,b.md", "--repo-root", str(repo)])
    assert rc == 1
    err = capsys.readouterr().err
    assert "cannot read --map-files input" in err
    # Base Errno-2 text stays first, then the appended hint.
    assert "Errno 2" in err or "No such file" in err
    assert (
        "--map-files takes a PATH to a newline-separated file list, not a comma-separated list"
    ) in err
    # Negative branch: a non-comma missing path — NO hint appended.
    rc = sel.main(
        ["--map-files", str(tmp_path / "definitely-not-there.txt"), "--repo-root", str(repo)]
    )
    assert rc == 1
    err = capsys.readouterr().err
    assert "cannot read --map-files input" in err
    assert "comma-separated list" not in err


# --- #1717 defect (d): empty-diff NOTE names uncommitted edits ---------------
def test_empty_diff_note_names_uncommitted_edits(tmp_path: Path, monkeypatch, capsys):
    """The (d) fix: the empty-diff NOTE at the main checkout carries the
    widened phrasing calling out uncommitted edits as the first likely
    cause. Reuses the `_make_git_repo_with_worktree` fixture; asserts the
    case-insensitive `commit first` substring survives in stderr.
    Distinct from `test_empty_diff_note_at_main_checkout_real_git` — that
    test now ALSO asserts `commit first` alongside its existing prefix +
    breadcrumb pins, and this test preserves the isolated pin so a future
    NOTE rewording immediately surfaces which assertion caught it.
    """
    repo, _wt = _make_git_repo_with_worktree(tmp_path)
    monkeypatch.chdir(repo)
    rc = sel.main([])
    assert rc == 0
    captured = capsys.readouterr()
    # The widened NOTE still starts with the byte-identical prefix (pinned
    # by test_empty_diff_note_at_main_checkout_real_git):
    assert "NOTE — empty diff" in captured.err
    # The new cause line: uncommitted edits are named first.
    assert "commit first" in captured.err.lower()


# --- #1717 defect (b): --json help text warns against `2>&1` stderr redirect --
def test_json_help_warns_against_stderr_redirect(capsys):
    """The (b) fix: the `--json` flag's help string carries a safety
    warning against redirecting stderr into stdout (`2>&1`), naming the
    safe recipe (`2>/dev/null`). Broader-invariant assertion: the help
    text mentions BOTH the concept (`stderr`) AND the recipe recommendation
    (`2>/dev/null`) — a benign softening of the exact phrasing that keeps
    both survives the pin; dropping either fails.
    """
    with pytest.raises(SystemExit) as excinfo:
        sel.main(["--help"])
    # argparse's --help handler exits 0.
    assert excinfo.value.code == 0
    help_text = capsys.readouterr().out
    assert "stderr" in help_text.lower()
    assert "2>/dev/null" in help_text


# --- Skills-pin discovery arm (#1851) ----------------------------------------------
# A touched .claude/skills/**/*.md selects every tests/**/test_*.py whose raw
# text references that file's skill-dir-QUALIFIED path (reason
# skills-pin:<skill path>) — the skills sibling of the rules-pin arm (#1496),
# ADDITIVE to the WORKFLOW_SURFACE skip (which is unchanged); --map-files
# unions the same pairs MINUS WORKFLOW_INVARIANT members. Unlike rules-pin
# the token is NOT the bare basename (every skill shares SKILL.md): a hit is
# the contiguous .claude/-relative path substring OR the path-join form.


# --- Skills-pin: contiguous full-path-literal reference form -----------------------
def test_skills_pin_contiguous_form_selected(tmp_path: Path):
    repo = _make_tree(tmp_path, [])
    (repo / "tests" / "test_issue_skill_pin.py").write_text(
        'SKILL = ".claude/skills/issue/SKILL.md"\n'
    )
    touched = [".claude/skills/issue/SKILL.md"]
    tests, untested, reasons = sel.select_tests_with_reasons(touched, repo)
    assert "tests/test_issue_skill_pin.py" in tests
    assert reasons["tests/test_issue_skill_pin.py"] == ["skills-pin:.claude/skills/issue/SKILL.md"]
    assert untested == []  # skills files stay a correct SKIP, never "untested"


# --- Skills-pin: path-join reference form, both quote styles -----------------------
def test_skills_pin_join_form_selected(tmp_path: Path):
    """The founding-test shape (all three founding tests use the join form):
    double-quoted Path-join with ``/`` separators AND single-quoted
    comma-separated components both match."""
    repo = _make_tree(tmp_path, [])
    (repo / "tests" / "test_join_dq.py").write_text(
        'text = (ROOT / ".claude" / "skills" / "issue" / "SKILL.md").read_text()\n'
    )
    (repo / "tests" / "test_join_sq.py").write_text(
        "p = Path('.claude', 'skills', 'issue', 'SKILL.md')\n"
    )
    tests, _, reasons = sel.select_tests_with_reasons([".claude/skills/issue/SKILL.md"], repo)
    assert "tests/test_join_dq.py" in tests
    assert "tests/test_join_sq.py" in tests
    assert reasons["tests/test_join_dq.py"] == ["skills-pin:.claude/skills/issue/SKILL.md"]
    assert reasons["tests/test_join_sq.py"] == ["skills-pin:.claude/skills/issue/SKILL.md"]


# --- Skills-pin: cross-skill qualification (plan criterion 6) ----------------------
def test_skills_pin_cross_skill_not_selected(tmp_path: Path):
    """Touching issue/SKILL.md must NOT select a test referencing only
    daily/SKILL.md — bare-basename matching is degenerate for SKILL.md (every
    skill shares the basename), so tokens are skill-dir-qualified."""
    repo = _make_tree(tmp_path, [])
    (repo / "tests" / "test_daily_only.py").write_text(
        'DAILY = ".claude/skills/daily/SKILL.md"\n'
        'JOIN = (ROOT / ".claude" / "skills" / "daily" / "SKILL.md")\n'
    )
    tests, _, _ = sel.select_tests_with_reasons([".claude/skills/issue/SKILL.md"], repo)
    assert "tests/test_daily_only.py" not in tests
    hits = sel.skills_pin_hits([".claude/skills/issue/SKILL.md"], repo)
    assert "tests/test_daily_only.py" not in hits


# --- Skills-pin: non-SKILL.md skill support files are covered too ------------------
def test_skills_pin_nested_support_file_selected(tmp_path: Path):
    """The glob is .claude/skills/**/*.md, not SKILL.md-only: a skill support
    file (markers.md) maps to its referencing test via the same qualified
    tokens (the _matches_any /**/ zero-segment collapse covers both depths)."""
    repo = _make_tree(tmp_path, [])
    (repo / "tests" / "test_markers_pin.py").write_text('M = ".claude/skills/issue/markers.md"\n')
    tests, _, reasons = sel.select_tests_with_reasons([".claude/skills/issue/markers.md"], repo)
    assert "tests/test_markers_pin.py" in tests
    assert reasons["tests/test_markers_pin.py"] == ["skills-pin:.claude/skills/issue/markers.md"]


# --- Skills-pin: no skills file touched -> zero hits, ZERO file reads --------------
def test_skills_pin_no_skills_touched_no_scan(tmp_path: Path, capsys):
    """Proof of the zero-read early return (mirror of the rules-pin case 56):
    an undecodable test file is planted; any scan pass reads raw text (the
    unreadable-file WARN proves a read), so the absence of a skills-pin WARN
    proves no file was read. A touched non-skills .md (docs/x.md) does not
    trigger the scan."""
    repo = _make_tree(tmp_path, [])
    (repo / "tests" / "test_undecodable.py").write_bytes(b"\xff\xfe bad")
    assert sel.skills_pin_hits(["scripts/widget.py", "docs/x.md"], repo) == {}
    assert "skills-pin scan cannot read" not in capsys.readouterr().err


# --- Skills-pin: monotonicity — the arm only ever GROWS the selection --------------
def test_skills_pin_selection_only_grows(tmp_path: Path):
    """Same touched set, tree WITH vs WITHOUT the pin test (mirror of the
    rules-pin case 57): WITH-selection is a superset and every WITHOUT reason
    list is preserved verbatim (plan acceptance criterion 4)."""
    touched = ["scripts/widgetlib.py", ".claude/skills/issue/SKILL.md"]
    repo_without = _make_tree(tmp_path / "without", ["test_widgetlib.py"])
    t_without, u_without, r_without = sel.select_tests_with_reasons(touched, repo_without)
    repo_with = _make_tree(tmp_path / "with", ["test_widgetlib.py"])
    (repo_with / "tests" / "test_skill_pin.py").write_text('S = ".claude/skills/issue/SKILL.md"\n')
    t_with, u_with, r_with = sel.select_tests_with_reasons(touched, repo_with)
    assert set(t_with) >= set(t_without)
    for test, rs in r_without.items():
        assert r_with[test] == rs  # pre-existing reason lists preserved verbatim
    assert "tests/test_skill_pin.py" in t_with
    assert u_without == u_with == []


# --- Skills-pin: unreadable test file WARNs + is skipped; never crashes ------------
def test_skills_pin_unreadable_test_file_warns_not_crash(tmp_path: Path, capsys):
    repo = _make_tree(tmp_path, [])
    (repo / "tests" / "test_good_skill_pin.py").write_text('S = ".claude/skills/issue/SKILL.md"\n')
    (repo / "tests" / "test_bad.py").write_bytes(b'S = ".claude/skills/issue/SKILL.md"\n\xff\xfe')
    tests, untested, _ = sel.select_tests_with_reasons([".claude/skills/issue/SKILL.md"], repo)
    assert "tests/test_good_skill_pin.py" in tests  # the valid hit still selected
    assert "tests/test_bad.py" not in tests
    assert untested == []
    err = capsys.readouterr().err
    assert err.count("skills-pin scan cannot read") == 1
    assert "test_bad.py" in err
    # 1 failure over the ~42-file fixture tree is < 5%: no aggregate WARN.
    assert "systemic tests/ breakage" not in err


# --- Skills-pin: LIVE-tree drift/regression pin (the #1851 founding pairs) ---------
def test_skills_pin_live_tree_known_pairs():
    """DRIFT/REGRESSION PIN: on the LIVE repo tree, a .claude/skills/issue/
    SKILL.md diff selects the three founding tests of the #1851 gap (all
    path-join-form references, none in WORKFLOW_INVARIANT) with a skills-pin
    reason. SUPERSET assert: new pin tests joining later must not break this;
    a rename of a pinned test legitimately forces a deliberate 1-line update
    here (that loudness is the point)."""
    root = Path(sel.__file__).resolve().parents[1]
    touched = [".claude/skills/issue/SKILL.md"]
    tests, untested, reasons = sel.select_tests_with_reasons(touched, root)
    for founding in (
        "tests/test_issue_skill_file_only_verdict_post.py",
        "tests/test_ensemble_review_cap.py",
        "tests/test_issue_skill_workload_cmd_script_pin.py",
    ):
        assert founding in tests
        assert "skills-pin:.claude/skills/issue/SKILL.md" in reasons[founding]
    assert untested == []  # the WORKFLOW_SURFACE skip is unchanged


# --- Skills-pin: --map-files EXCLUDES invariant members; the 9c arm keeps them -----
def test_cli_map_files_skills_pin_excludes_invariant(tmp_path: Path, capsys):
    """The rules_pin_pairs asymmetry, skills edition (plan criterion 3):
    tests/test_workflow_lint.py (a WORKFLOW_INVARIANT member and the only
    SLOW_TESTS entry) is filtered from the --map-files pairs while a
    non-invariant referencing test appears; select_tests_with_reasons still
    carries the skills-pin reason on the invariant member (the union dedupes;
    the extra reason is informative)."""
    repo = _make_tree(tmp_path, [])
    (repo / "tests" / "test_workflow_lint.py").write_text('S = ".claude/skills/issue/SKILL.md"\n')
    (repo / "tests" / "test_issue_skill_pin.py").write_text('S = ".claude/skills/issue/SKILL.md"\n')
    listing = tmp_path / "payload.txt"
    listing.write_text(".claude/skills/issue/SKILL.md\n")
    rc = sel.main(["--map-files", str(listing), "--repo-root", str(repo)])
    assert rc == 0
    out = capsys.readouterr().out
    assert out.splitlines() == [
        "tests/test_issue_skill_pin.py\t.claude/skills/issue/SKILL.md"
    ]  # invariant member filtered; non-invariant pair printed
    _, _, reasons = sel.select_tests_with_reasons([".claude/skills/issue/SKILL.md"], repo)
    assert set(reasons["tests/test_workflow_lint.py"]) == {
        "invariant",
        "skills-pin:.claude/skills/issue/SKILL.md",
    }


# --- Skills-pin: LIVE-tree generative reachability pin (plan criterion 5) ----------
def test_skills_pin_reachability_live_tree():
    """GENERATIVE PIN: with an INDEPENDENT scan (own regex — deliberately NOT
    the arm's functions, so a bug in _skills_pin_tokens cannot vacuously pass
    this), for ALL .claude/skills/*/SKILL.md on the live tree, every
    tests/**/test_*.py referencing that skill's SKILL.md (contiguous OR
    path-join textual form) is in select_tests(['<that path>'], root)[0] —
    the union of the invariant set and the skills-pin arm. A future pin test
    added outside WORKFLOW_INVARIANT can no longer silently fall out of
    selector coverage (the #1851 founding gap)."""
    root = Path(sel.__file__).resolve().parents[1]
    skill_mds = sorted((root / ".claude" / "skills").glob("*/SKILL.md"))
    assert skill_mds, "live-tree precondition: no .claude/skills/*/SKILL.md found"
    # One read pass over the live tests/ tree (cached; the per-skill loop
    # below scans strings, not files).
    texts: dict[str, str] = {}
    for tp in sorted((root / "tests").rglob("test_*.py")):
        try:
            texts[tp.relative_to(root).as_posix()] = tp.read_text(encoding="utf-8")
        except (OSError, ValueError):
            continue
    sep = r"[\"']\s*[,/]+\s*[\"']"
    checked_any = False
    for skill_md in skill_mds:
        rel = skill_md.relative_to(root).as_posix()  # .claude/skills/<skill>/SKILL.md
        skill = skill_md.parent.name
        contiguous = f"skills/{skill}/SKILL.md"
        join_re = re.compile("[\"']skills" + sep + re.escape(skill) + sep + r"SKILL\.md[\"']")
        referencing = [t for t, text in texts.items() if contiguous in text or join_re.search(text)]
        if not referencing:
            continue  # nothing pins this skill; nothing to be reachable
        checked_any = True
        selected, _ = sel.select_tests([rel], root)
        missing = [t for t in referencing if t not in selected]
        assert not missing, f"{rel}: referencing tests not selector-reachable: {missing}"
    # Live-tree sanity: at least one skill (issue) has referencing tests today,
    # so this pin is never vacuously green.
    assert checked_any
