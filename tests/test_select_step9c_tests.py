"""Unit tests for ``scripts/select_step9c_tests.py`` (#754).

The helper maps the files this branch touched to their covering pytest files
plus a pinned workflow-invariant literal set, for the ``/issue`` Step 9c
test-verdict gate. These tests inject a fake ``git diff`` runner and a
``tmp_path`` ``tests/`` tree so no real git / real branch state is needed.

The one exception is the pinned-list test (case 6), which asserts the literal
``WORKFLOW_INVARIANT`` matches the LIVE repo ``tests/`` tree so an added/removed
invariant test forces a deliberate edit of the literal.
"""

from __future__ import annotations

import importlib.util
import json
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
    assert set(tests) == set(sel.WORKFLOW_INVARIANT)  # still runs the invariant set


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
    # 32 = plan §5's verbatim enumerated list (31 files) + test_autonomous_session_watch.py
    # (the one curated addition from the #754 brief — the watcher's own decision-gate
    # test alongside the pinned test_autonomous_plan_gate.py). The brief's "34" figure was
    # arithmetic carried from the plan's mis-stated "33" header (the §5 table enumerates 31).
    assert len(sel.WORKFLOW_INVARIANT) == len(set(sel.WORKFLOW_INVARIANT))
    assert len(sel.WORKFLOW_INVARIANT) == 32


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
    monkeypatch.setattr(sel, "_resolve_repo_root", lambda _arg: repo)
    monkeypatch.setattr(sel, "compute_touched", lambda *_a, **_k: ["scripts/widget.py"])
    rc = sel.main(["--json", "--repo-root", str(repo)])
    assert rc == 0
    out = json.loads(capsys.readouterr().out)
    assert set(out.keys()) == {"tests", "untested_touched", "base", "missing_invariants"}
    assert "tests/test_widget.py" in out["tests"]
    assert out["base"] == "main"
    assert out["missing_invariants"] == []  # all invariants present in the fixture tree


# --- Case 11: _resolve_repo_root uses --git-common-dir + dirnames (#785) ------
def test_resolve_repo_root_uses_git_common_dir_and_dirnames(tmp_path: Path, monkeypatch):
    """No-arg path calls `git rev-parse --path-format=absolute --git-common-dir`
    and dirnames the output — the #506-safe recipe, NOT `--show-toplevel`
    (which from a worktree cwd doubles the path). The `--repo-root` override
    still bypasses git entirely.
    """
    seen: dict[str, list[str]] = {}
    git_dir = tmp_path / "repo" / ".git"

    class _Result:
        stdout = str(git_dir) + "\n"

    def _fake_run(argv, **_kw):
        seen["argv"] = argv
        return _Result()

    monkeypatch.setattr(sel.subprocess, "run", _fake_run)
    got = sel._resolve_repo_root(None)
    # (i) locks the recipe against a regression back to --show-toplevel:
    assert seen["argv"] == ["git", "rev-parse", "--path-format=absolute", "--git-common-dir"]
    # (ii) return is the dirname of the git output (.../repo/.git -> .../repo):
    assert got == (tmp_path / "repo").resolve()
    # (iii) the override branch never touches git — clear the recorded argv and
    #       confirm the arg path is returned resolved, without a git call:
    seen.clear()
    assert sel._resolve_repo_root(str(tmp_path)) == tmp_path.resolve()
    assert "argv" not in seen  # git was not invoked on the override path


# --- Case 12: empty selection fails LOUD (defense-in-depth, #785) ------------
def test_empty_selection_fails_loud(tmp_path: Path, monkeypatch, capsys):
    """A structurally-impossible empty test list (bad repo_root resolution or all
    invariants missing) makes main() return 1 + a stderr line — never a silent
    exit-0 zero-test gate (the same silent-pass class the Step 9c shell guard
    closes at the shell level).
    """
    repo = _make_tree(tmp_path, [])
    monkeypatch.setattr(sel, "_resolve_repo_root", lambda _arg: repo)
    monkeypatch.setattr(sel, "compute_touched", lambda *_a, **_k: [])
    # Force the degenerate empty selection the invariant set normally prevents.
    monkeypatch.setattr(sel, "select_tests", lambda *_a, **_k: ([], []))
    rc = sel.main(["--repo-root", str(repo)])
    assert rc == 1
    err = capsys.readouterr().err
    assert "EMPTY test selection" in err
