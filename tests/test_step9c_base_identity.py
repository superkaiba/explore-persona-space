"""Regression fixtures for the #2302 base-identity filter (Step 9c selector + compare).

#2296's measured shape: the Step 5a sibling-file sync commits main's OWN
``scripts/issue<M>_*`` / ``tests/test_issue<M>_*`` content into the branch, so
those paths enter the three-dot diff (merge-base...HEAD) while being absent
from the two-dot diff (base tip vs HEAD) and blob-identical to the base tip.
Pre-#2302, ``select_step9c_tests.compute_touched`` returned them as
branch-touched — inflating the gate set (61 invariant files -> 217, wall
1:46:36) — and the compare's #2024 precondition 1 read them as branch changes,
hardening a pre-existing order-dependent failure into blocking NEW.

Two-arm synthetic tree (plan #2302 §4 Change 3), REAL tmp git repos:

* arm 1 — the synced-from-main path is excluded by ``compute_touched``, named
  by ``compute_base_identical``, and a failing node in it classifies
  NON-blocking (``ordering_suspect``, not ``new``) through the REAL compare
  classifier path;
* arm 2 (positive control, criterion 2) — the SAME path branch-EDITED stays
  in ``compute_touched``, is absent from ``compute_base_identical``, and its
  failing node stays blocking (``new``).

Plus the fail-closed negative controls: a branch-new file stays touched, an
unresolvable blob OID excludes nothing, and a git error in the auxiliary
probes degrades to ZERO exclusions (selector: stderr NOTE; compare: a
``warns`` row, never ``_Indeterminate``).

The git-invoking helpers execute their REAL bodies against real tmp git repos
(code-style.md one-production-body-test-per-seam-stubbed-function); the
compare arms reuse ``test_step9c_baseline``'s signature-conformant fake
harness ONLY at the external subprocess boundaries (pristine/paired pytest
runners, ruff counts, scratch worktree), while ``_base_identical_files`` runs
real git against the real worktree repo.

Scratch dirs use ``tempfile.mkdtemp``, not ``tmp_path``: concurrent pytest
sessions prune stale ``/tmp/pytest-of-*`` numbered roots, which can delete a
live ``tmp_path`` mid-test under subprocess-heavy fixtures.
"""

from __future__ import annotations

import importlib.util
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))  # sibling harness import below

import test_step9c_baseline as tb

# import precedent: test_issue1092_offvm_refit.py); tb.sb is the SAME step9c_baseline
# module instance the harness's monkeypatches target.

sb = tb.sb

# Import the selector by path (it lives under scripts/, not an importable package).
_SEL_PATH = Path(__file__).resolve().parents[1] / "scripts" / "select_step9c_tests.py"
_sel_spec = importlib.util.spec_from_file_location("select_step9c_tests", _SEL_PATH)
assert _sel_spec and _sel_spec.loader
sel = importlib.util.module_from_spec(_sel_spec)
sys.modules[_sel_spec.name] = sel
_sel_spec.loader.exec_module(sel)


@pytest.fixture(autouse=True)
def _gate_tmp_routing_disabled(monkeypatch):
    """Host-independent determinism (#1408) — mirrors test_step9c_baseline's autouse."""
    monkeypatch.setenv("EPM_STEP9C_TMPDIR", "")


@pytest.fixture()
def scratch_dir():
    """A mkdtemp scratch root (NOT tmp_path — see module docstring), rmtree'd after."""
    d = Path(tempfile.mkdtemp(prefix="eps2302-baseident-"))
    try:
        yield d
    finally:
        shutil.rmtree(d, ignore_errors=True)


# --- Real-git fixture builders --------------------------------------------------

_SIB_SCRIPT = "scripts/issue777_sib.py"
_SIB_TEST = "tests/test_issue777_sib.py"
_SIB_NODE = sb.Node(file=_SIB_TEST, classname="tests.test_issue777_sib", name="test_render_all")
_PRED = "tests/test_aaa_pred.py"  # co-selected predecessor (sorts BEFORE the sibling test)
_PRED_ROW = (_PRED, "tests.test_aaa_pred", "test_ok", "passed")


def _git(repo: Path, *args: str) -> str:
    """Run git in *repo* fail-loud; return stdout."""
    return subprocess.run(
        ["git", "-C", str(repo), *args], capture_output=True, text=True, check=True
    ).stdout


def _write(repo: Path, rel: str, text: str) -> None:
    """Write *text* at *rel* under *repo*, creating parents."""
    p = repo / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text)


def _build_synced_repo(d: Path) -> Path:
    """#2296's exact shape: branch cut at v1; main advances the sibling pair to v2;
    the branch writes main's CURRENT (v2) content in and commits it — byte-identical
    to the base tip, so in the three-dot diff, absent from the two-dot diff,
    blob-equal to base."""
    repo = d / "wtrepo"
    repo.mkdir(parents=True)
    subprocess.run(
        ["git", "init", "-q", "-b", "main", str(repo)],
        capture_output=True,
        text=True,
        check=True,
    )
    _git(repo, "config", "user.email", "step9c-test@example.com")
    _git(repo, "config", "user.name", "step9c-test")
    _git(repo, "config", "commit.gpgsign", "false")
    _write(repo, _SIB_SCRIPT, "V = 1\n")
    _write(repo, _SIB_TEST, "def test_render_all():\n    assert True\n")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "base v1")
    _git(repo, "checkout", "-q", "-b", "issue-x")
    _git(repo, "checkout", "-q", "main")
    _write(repo, _SIB_SCRIPT, "V = 2\n")
    _write(repo, _SIB_TEST, "def test_render_all():\n    assert 2 == 2\n")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "main advances the sibling pair")
    _git(repo, "checkout", "-q", "issue-x")
    # The Step 5a sibling-sync arm's own idiom: checkout main's copy, commit.
    _git(repo, "checkout", "main", "--", _SIB_SCRIPT, _SIB_TEST)
    _git(repo, "commit", "-q", "-m", "sync workflow-surface specs from origin/main")
    return repo


def _branch_edit_sib_test(repo: Path) -> None:
    """Arm 2: the branch genuinely edits the sibling TEST file after the sync."""
    _write(repo, _SIB_TEST, "def test_render_all():\n    assert 3 == 3\n")
    _git(repo, "add", _SIB_TEST)
    _git(repo, "commit", "-q", "-m", "branch edits the sibling test")


# --- Selector arms (real compute_touched / compute_base_identical bodies) -------


def test_synced_sibling_excluded_and_named(scratch_dir: Path):
    """Arm 1, selector half: the synced pair leaves compute_touched and is NAMED
    by compute_base_identical (never a silent exclusion)."""
    repo = _build_synced_repo(scratch_dir)
    touched = sel.compute_touched("main", repo)
    assert _SIB_SCRIPT not in touched
    assert _SIB_TEST not in touched
    assert sel.compute_base_identical("main", repo) == sorted([_SIB_SCRIPT, _SIB_TEST])


def test_branch_new_file_stays_touched(scratch_dir: Path):
    """Fail-closed: a branch-NEW file (absent on base) is in BOTH diffs, so it is
    never an exclusion candidate — it stays touched and is never base-identical."""
    repo = _build_synced_repo(scratch_dir)
    _write(repo, "scripts/issue777_new.py", "N = 1\n")
    _git(repo, "add", "scripts/issue777_new.py")
    _git(repo, "commit", "-q", "-m", "branch-new file")
    touched = sel.compute_touched("main", repo)
    assert "scripts/issue777_new.py" in touched
    assert "scripts/issue777_new.py" not in sel.compute_base_identical("main", repo)


def test_base_identical_excludes_nothing_when_blob_check_unresolvable(scratch_dir: Path):
    """Plan-named fail-closed pin: a candidate whose blob OID fails to resolve stays
    in compute_touched — the REAL ``git cat-file --batch-check`` body runs against
    the real repo and returns ``missing`` for the ghost path, so an unresolvable
    path is never silently swallowed into the excluded set."""
    repo = _build_synced_repo(scratch_dir)
    ghost = "scripts/issue777_ghost.py"  # absent from every commit of the repo

    def runner(argv: list[str]) -> str:
        # Injected DIFFS ONLY (three-dot proposes the ghost; two-dot omits it, so
        # it becomes an exclusion candidate); the blob check runs its real body.
        if any(a.endswith("...HEAD") for a in argv):
            return ghost + "\n"
        return ""

    assert sel.compute_touched("main", repo, _runner=runner) == [ghost]
    assert sel.compute_base_identical("main", repo, _runner=runner) == []


def test_two_dot_failure_degrades_to_no_exclusion(scratch_dir: Path, capsys):
    """Fail-closed: a git error in the AUXILIARY two-dot probe degrades to ZERO
    exclusions with a stderr NOTE — every three-dot path stays branch-touched,
    and the selector never crashes on the auxiliary call."""
    repo = _build_synced_repo(scratch_dir)

    def runner(argv: list[str]) -> str:
        if any(a.endswith("...HEAD") for a in argv):
            return _SIB_SCRIPT + "\n"
        raise subprocess.CalledProcessError(128, argv)

    assert sel.compute_touched("main", repo, _runner=runner) == [_SIB_SCRIPT]
    assert sel.compute_base_identical("main", repo, _runner=runner) == []
    assert "base-identity filter failed" in capsys.readouterr().err


def test_selector_json_emits_base_identical_excluded(scratch_dir: Path, capsys):
    """CLI end to end on the real repo: the excluded set rides the ``--json``
    payload as ``base_identical_excluded`` and a stderr NOTE names each path,
    while a genuinely branch-edited test file stays selected."""
    repo = _build_synced_repo(scratch_dir)
    _branch_edit_sib_test(repo)  # test file stays selected; script stays excluded
    rc = sel.main(["--json", "--repo-root", str(repo), "--base", "main"])
    captured = capsys.readouterr()
    assert rc == 0
    data = json.loads(captured.out)
    assert data["base_identical_excluded"] == [_SIB_SCRIPT]
    assert _SIB_TEST in data["tests"]
    assert _SIB_SCRIPT not in data["untested_touched"]
    assert "base-identical paths excluded" in captured.err


def test_sync_only_branch_exclusion_is_loud(scratch_dir: Path, capsys):
    """M1 (#2302 round 2): a branch whose ENTIRE three-dot diff is base-identical
    (a sync-commit-only branch) filters ``touched`` to ``[]`` — the exclusion
    must STILL be loud: ``base_identical_excluded`` names both synced paths and
    the exclusion NOTE fires alongside the empty-diff fallback NOTE. Pre-fix,
    the ``if not touched`` guard short-circuited the audit and the exclusion
    was the one silent case."""
    repo = _build_synced_repo(scratch_dir)  # NO branch edit: sync commit only
    # Seed one invariant member so the invariant-only fallback selection is
    # non-empty (an empty selection is main()'s own fail-loud exit 1, which
    # would swallow the JSON under test). Untracked -> invisible to the diff.
    inv0 = Path(repo) / sel.WORKFLOW_INVARIANT[0]
    inv0.parent.mkdir(parents=True, exist_ok=True)
    inv0.write_text("# invariant stub\n")
    rc = sel.main(["--json", "--repo-root", str(repo), "--base", "main"])
    captured = capsys.readouterr()
    assert rc == 0
    data = json.loads(captured.out)
    assert data["base_identical_excluded"] == sorted([_SIB_SCRIPT, _SIB_TEST])
    assert "base-identical paths excluded" in captured.err  # never silent
    assert "empty diff" in captured.err  # the invariant-only fallback co-fires
    assert _SIB_TEST not in data["tests"]  # synced test is main's copy, not selected
    assert sel.WORKFLOW_INVARIANT[0] in data["tests"]


# --- Compare arms (real classifier path; real git wt; fakes only at the external
# --- pytest/ruff/scratch boundaries via test_step9c_baseline's harness) ---------


def _compare_argv(
    scratch_dir: Path,
    monkeypatch,
    *,
    wt: Path,
    touched: list[str],
    junit_cases: list[tuple[str, str, str, str]],
    paired_failing=(),
    pytest_rc: int = 1,
) -> tuple[list[str], dict[str, list]]:
    """A compare env whose WORKTREE is caller-supplied (a REAL git repo for the
    #2302 arms — test_step9c_baseline's own materializer writes stub content
    into wt, which would clobber the real repo). Root fixture tree + ledger +
    junit + fakes come verbatim from the tb harness."""
    root = scratch_dir / "root"
    (root / "tests").mkdir(parents=True, exist_ok=True)
    for f in sorted({c[0] for c in junit_cases}):
        p = root / f
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("# stub\n")
    tb._write_ledger(root, failing=())
    junit = scratch_dir / "run-junit.xml"
    junit.write_text(tb._junit_xml(list(junit_cases)))
    # The selection carries a co-selected PREDECESSOR (sorts before the sibling
    # test) — a candidate with no predecessors is a legitimate #2024
    # ``no-predecessors`` skip, which is not the shape under test here.
    fake_sel = tb._FakeSel(touched, {_PRED: ["invariant"], _SIB_TEST: ["touched-test"]}, None)
    calls = tb._install_compare_fakes(
        monkeypatch,
        root=root,
        fake_sel=fake_sel,
        changed_tests=(),
        live_dirty=(),
        pristine_failing=(),
        pristine_exc=None,
        base_ruff=(100, 0),
        wt_ruff=(100, 0),
        touched_ruff=(0, 0),
        sha_known=True,
        code_commits=0,
        paired_failing=paired_failing,
    )
    argv = [
        "compare",
        "--junitxml",
        str(junit),
        "--pytest-rc",
        str(pytest_rc),
        "--worktree",
        str(wt),
        "--repo-root",
        str(root),
        "--json",
        "--run-pristine",
    ]
    return argv, calls


def test_synced_sibling_classifies_ordering_suspect_not_new(scratch_dir: Path, monkeypatch, capsys):
    """Arm 1 (acceptance criteria 1 + 3): a synced-from-main sibling test failing
    order-dependently — single-file pristine PASS, paired prefix REPRODUCES —
    classifies NON-blocking ``ordering_suspect``, because the compare's OWN
    base-identity derivation (real git) subtracts the synced pair from #2024
    precondition 1. Pre-#2302 this exact shape hardened into blocking NEW via
    ``paired_skipped: file-in-branch-diff`` (#2296)."""
    wt = _build_synced_repo(scratch_dir)
    argv, calls = _compare_argv(
        scratch_dir,
        monkeypatch,
        wt=wt,
        touched=[_SIB_SCRIPT, _SIB_TEST],
        junit_cases=[
            _PRED_ROW,
            (_SIB_NODE.file, _SIB_NODE.classname, _SIB_NODE.name, "failed"),
        ],
        paired_failing=(_SIB_NODE,),
    )
    rc, out, _err = tb._run_json(argv, capsys)
    assert rc == 0
    assert out["new"] == []
    assert {(o["file"], o["name"]) for o in out["ordering_suspect"]} == {
        (_SIB_NODE.file, _SIB_NODE.name)
    }
    assert out["base_identical_files"] == sorted([_SIB_SCRIPT, _SIB_TEST])
    assert out["paired_skipped"] == []  # precondition 1 no longer fires on the synced pair
    # The paired discriminator actually RAN, under the co-selected predecessor.
    assert calls["paired"] == [[_PRED, _SIB_TEST]]
    assert out["paired_files_run"] == [_PRED, _SIB_TEST]


def test_branch_edited_sibling_still_classifies_new(scratch_dir: Path, monkeypatch, capsys):
    """Plan-named positive control (acceptance criterion 2): a content-differing
    path is absent from compute_base_identical and its failing node stays
    blocking NEW on the real classifier path — the carve-out never widens into
    a hole for genuinely branch-authored changes."""
    wt = _build_synced_repo(scratch_dir)
    _branch_edit_sib_test(wt)
    # Selector half: the branch-edited file is NOT base-identical (the still-synced
    # script is — per-path granularity).
    assert sel.compute_base_identical("main", wt) == [_SIB_SCRIPT]
    argv, calls = _compare_argv(
        scratch_dir,
        monkeypatch,
        wt=wt,
        touched=[_SIB_SCRIPT, _SIB_TEST],
        junit_cases=[
            _PRED_ROW,
            (_SIB_NODE.file, _SIB_NODE.classname, _SIB_NODE.name, "failed"),
        ],
        paired_failing=(_SIB_NODE,),  # even a would-reproduce failure must stay NEW
    )
    rc, out, _err = tb._run_json(argv, capsys)
    assert rc == 1
    assert out["new"] == [_SIB_NODE._asdict()]
    assert out["ordering_suspect"] == []
    assert {r["reason"] for r in out["paired_skipped"]} == {"file-in-branch-diff"}
    assert _SIB_TEST not in out["base_identical_files"]
    assert out["base_identical_files"] == [_SIB_SCRIPT]
    assert calls["paired"] == []  # the paired stage never spent on a branch-authored file


def test_base_identity_derivation_failure_warns_not_indeterminate(
    scratch_dir: Path, monkeypatch, capsys
):
    """Fail-closed disposition (plan §4 Change 2): a git error in the compare's
    base-identity derivation (here: a non-git worktree, the existing tmp-repo
    fixture shape) degrades to ``base_identical = ∅`` plus a warns row — the
    compare still classifies (exit 0/1), NEVER ``_Indeterminate``/exit 2."""
    wt = scratch_dir / "wt"
    (wt / "tests").mkdir(parents=True)
    argv, _calls = _compare_argv(
        scratch_dir,
        monkeypatch,
        wt=wt,
        touched=[_SIB_TEST],  # non-empty -> the derivation's git call actually fires
        junit_cases=[(_SIB_NODE.file, _SIB_NODE.classname, _SIB_NODE.name, "passed")],
        pytest_rc=0,
    )
    rc, out, _err = tb._run_json(argv, capsys)
    assert rc == 0
    assert not out.get("indeterminate")
    assert out["base_identical_files"] == []
    assert any(w.startswith("BASE-IDENTITY WARN:") for w in out["warns"])
    # M3 (#2302 round 2): the compare JSON surfaces the RESOLVED base it
    # derived against (the fake selector has no resolve_base -> "main").
    assert out["base"] == "main"
