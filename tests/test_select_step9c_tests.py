"""Unit tests for ``scripts/select_step9c_tests.py`` (#754, #851).

The helper maps the files this branch touched to their covering pytest files
plus a pinned workflow-invariant literal set, for the ``/issue`` Step 9c
test-verdict gate. Most cases inject a fake ``git diff`` runner and a
``tmp_path`` ``tests/`` tree so no real git / real branch state is needed.

Two exceptions: the pinned-list test (case 6) asserts the literal
``WORKFLOW_INVARIANT`` matches the LIVE repo ``tests/`` tree so an added/removed
invariant test forces a deliberate edit of the literal; and the #851 regression
tests (cases 13-14) build a real throwaway git repo + worktree to pin the
work-root resolution end to end (branch-new test selected, deleted-on-branch
test dropped, empty-diff NOTE at the base checkout).
"""

from __future__ import annotations

import importlib.util
import json
import os
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
    updating the literal — forcing a deliberate edit (plan §4a-note). This is
    the only test that reads the live repo rather than a tmp_path fixture.
    """
    repo_root = Path(sel.__file__).resolve().parents[1]
    missing = sel.missing_invariants(repo_root)
    assert missing == [], (
        f"WORKFLOW_INVARIANT entries missing from the live tests/ tree: {missing}. "
        "Update the literal in scripts/select_step9c_tests.py deliberately."
    )
    # And it must be a non-trivial, de-duplicated set (no accidental shrink/dup).
    # 37 = plan §5's verbatim enumerated list (31 files) + test_autonomous_session_watch.py
    # (the #754 brief's one curated addition) + the 3 SKILL.md-content-pin suites added by
    # #1242 (test_step10d_guard3 / test_step_completed_resume / test_issue_skill_exit_breadcrumb
    # — SKILL.md diffs gate ONLY via this tuple, so their pins must live in it) + the #1268
    # Step-10d repin/guard hardening pin suite (test_issue_skill_merge_resnapshot_pin) + the
    # #1289 diff-base origin/main pin suite (test_diff_base_origin_main_pin — same tuple
    # rationale: a SKILL.md-only recipe revert must still run the pin).
    assert len(sel.WORKFLOW_INVARIANT) == len(set(sel.WORKFLOW_INVARIANT))
    assert len(sel.WORKFLOW_INVARIANT) == 37


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
    assert line.startswith(prefix) and line.endswith(" -v --tb=short")
    files = line.removeprefix(prefix).removesuffix(" -v --tb=short").split()
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
    assert reasons["tests/test_consumer.py"] == ["import-map:scripts/widgetlib.py"]
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
    assert reasons["tests/test_flat_consumer.py"] == [
        "import-map:src/explore_persona_space/task_widget.py"
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
    """>5% of scanned test files failing to parse emits ONE extra summary WARN."""
    files = {f"test_broken_{i}.py": "import widgetlib\ndef broken(:\n" for i in range(3)}
    files["test_ok.py"] = "import widgetlib\n"
    repo = _make_import_tree(tmp_path, files)  # 3 broken / ~43 scanned ≈ 7% > 5%
    tests, _, _ = sel.select_tests_with_reasons(["scripts/widgetlib.py"], repo)
    assert "tests/test_ok.py" in tests
    err = capsys.readouterr().err
    assert err.count("import-map cannot parse") == 3  # one per-file WARN each
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
